"""
Streaming Parallel Decoding for WeDLM Inference

Implements the streaming parallel decoding algorithm from WeDLM paper:
1. Fixed-size sliding window W with [filled | mask] slots
2. Distance-adjusted entropy selection for parallel predictions
3. Immediate prefix commitment for KV cache validity
4. Dynamic refill to maintain constant workload

Key metrics:
- p_cache: Prefix cacheability - how much of completed sequence was contiguous prefix
- Speedup: Number of tokens generated per forward pass

Reference: WeDLM paper Section 4.3, Algorithm 1
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass

from .config import WeDLMConfig
from .topological_reorder import topological_reorder, inverse_reorder


@dataclass
class StreamingDecoderState:
    """State for streaming parallel decoder."""

    # Window state
    window_tokens: torch.Tensor  # [B, W] current window tokens
    window_positions: torch.Tensor  # [B, W] logical positions
    window_filled: torch.Tensor  # [B, W] boolean mask of filled positions

    # Generation state
    generated_tokens: List[List[int]]  # List of generated token IDs per batch
    next_position: int  # Next logical position to fill

    # KV cache (if supported by model)
    kv_cache: Optional[Tuple] = None


class StreamingParallelDecoder:
    """
    Streaming Parallel Decoder for WeDLM inference.

    Generates tokens in parallel within a fixed-size window,
    committing filled prefixes immediately for KV cache efficiency.
    """

    def __init__(
        self,
        model: nn.Module,
        config: WeDLMConfig,
        mask_token_id: int,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
    ):
        """
        Args:
            model: The language model to use for generation
            config: WeDLM configuration
            mask_token_id: Token ID for [MASK]
            pad_token_id: Token ID for padding (optional)
            eos_token_id: Token ID for end-of-sequence (optional)
        """
        self.model = model
        self.config = config
        self.mask_token_id = mask_token_id
        self.pad_token_id = pad_token_id or mask_token_id
        self.eos_token_id = eos_token_id

        # Decoding parameters
        self.window_size = config.window_size
        self.entropy_threshold = config.entropy_threshold
        self.distance_penalty = config.distance_penalty

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        min_p: Optional[float] = None,
        verbose: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Generate tokens using streaming parallel decoding.

        Args:
            prompt_ids: [B, P] prompt token IDs
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling (optional)
            top_p: Nucleus sampling threshold (optional)
            min_p: Minimum probability threshold (optional)
            verbose: Print progress

        Returns:
            generated_ids: [B, P + N] full sequence including prompt
            metrics: Dictionary with generation metrics (p_cache, speedup, etc.)
        """
        B, P = prompt_ids.shape
        device = prompt_ids.device
        W = self.window_size

        self.model.eval()

        # Initialize state
        state = self._init_state(prompt_ids)

        # Metrics tracking
        total_forward_passes = 0
        tokens_committed = 0
        prefix_committed = 0  # Tokens committed as contiguous prefix

        while tokens_committed < max_new_tokens:
            # Check for EOS in all batches
            if self._check_eos(state):
                break

            # 1. Reorder window: [filled | masks]
            reordered_tokens, reordered_positions, num_filled = self._reorder_window(state)

            # 2. Forward pass on window
            logits = self._forward_window(
                prompt_ids,
                reordered_tokens,
                reordered_positions,
                state,
            )
            total_forward_passes += 1

            # 3. Compute probabilities and entropy
            probs = self._compute_probs(logits, temperature, top_k, top_p, min_p)
            entropies = self._compute_entropy(probs)

            # 4. Restore original order for processing
            # The logits are in reordered space, need to map back
            logits_original = self._restore_order(logits, reordered_positions, state)
            probs_original = self._restore_order(probs, reordered_positions, state)
            entropies_original = self._restore_order(entropies.unsqueeze(-1), reordered_positions, state).squeeze(-1)

            # 5. Select masks to fill based on distance-adjusted entropy
            to_fill = self._select_positions_to_fill(
                state,
                entropies_original,
            )

            # 6. Sample tokens for selected positions
            new_tokens = self._sample_tokens(probs_original, to_fill)

            # 7. Update window with new tokens
            self._update_window(state, new_tokens, to_fill)

            # 8. Commit leftmost contiguous filled prefix
            n_committed = self._commit_prefix(state)
            tokens_committed += n_committed

            # Track prefix commits vs non-prefix commits
            if n_committed > 0:
                prefix_committed += n_committed

            # 9. Refill window with new masks
            self._refill_window(state, max_new_tokens - tokens_committed)

            if verbose and total_forward_passes % 10 == 0:
                print(f"  Forward passes: {total_forward_passes}, Tokens: {tokens_committed}/{max_new_tokens}")

        # Collect results
        generated_ids = self._collect_results(prompt_ids, state)

        # Compute metrics
        p_cache = prefix_committed / max(tokens_committed, 1)
        speedup = tokens_committed / max(total_forward_passes, 1)

        metrics = {
            'p_cache': p_cache,
            'speedup': speedup,
            'total_forward_passes': total_forward_passes,
            'tokens_generated': tokens_committed,
        }

        return generated_ids, metrics

    def _init_state(self, prompt_ids: torch.Tensor) -> StreamingDecoderState:
        """Initialize decoder state."""
        B, P = prompt_ids.shape
        device = prompt_ids.device
        W = self.window_size

        # Initialize window with masks
        window_tokens = torch.full((B, W), self.mask_token_id, device=device, dtype=torch.long)
        window_positions = torch.arange(P, P + W, device=device).unsqueeze(0).expand(B, -1).clone()
        window_filled = torch.zeros(B, W, dtype=torch.bool, device=device)

        return StreamingDecoderState(
            window_tokens=window_tokens,
            window_positions=window_positions,
            window_filled=window_filled,
            generated_tokens=[[] for _ in range(B)],
            next_position=P + W,
        )

    def _reorder_window(
        self,
        state: StreamingDecoderState,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Reorder window to [filled | masks] for efficient attention."""
        B, W = state.window_tokens.shape
        device = state.window_tokens.device

        reordered_tokens = torch.zeros_like(state.window_tokens)
        reordered_positions = torch.zeros_like(state.window_positions)
        num_filled = torch.zeros(B, dtype=torch.long, device=device)

        for b in range(B):
            filled_idx = state.window_filled[b].nonzero(as_tuple=True)[0]
            mask_idx = (~state.window_filled[b]).nonzero(as_tuple=True)[0]

            n_filled = len(filled_idx)
            num_filled[b] = n_filled

            perm = torch.cat([filled_idx, mask_idx])
            reordered_tokens[b] = state.window_tokens[b, perm]
            reordered_positions[b] = state.window_positions[b, perm]

        return reordered_tokens, reordered_positions, num_filled

    def _forward_window(
        self,
        prompt_ids: torch.Tensor,
        window_tokens: torch.Tensor,
        window_positions: torch.Tensor,
        state: StreamingDecoderState,
    ) -> torch.Tensor:
        """Forward pass on window with prompt context."""
        B = prompt_ids.shape[0]
        P = prompt_ids.shape[1]
        W = window_tokens.shape[1]
        device = prompt_ids.device

        # Concatenate prompt + window
        full_input = torch.cat([prompt_ids, window_tokens], dim=1)

        # Position IDs: prompt uses sequential, window uses reordered positions
        prompt_positions = torch.arange(P, device=device).unsqueeze(0).expand(B, -1)
        full_positions = torch.cat([prompt_positions, window_positions], dim=1)

        # Forward pass
        logits, _ = self.model(
            full_input,
            position_ids=full_positions,
            return_all_logits=True,
        )

        # Extract window logits
        window_logits = logits[:, P:, :]  # [B, W, V]

        return window_logits

    def _compute_probs(
        self,
        logits: torch.Tensor,
        temperature: float,
        top_k: Optional[int],
        top_p: Optional[float],
        min_p: Optional[float],
    ) -> torch.Tensor:
        """Compute probabilities with optional filtering."""
        logits = logits / max(temperature, 1e-8)

        # Apply top-k filtering
        if top_k is not None and top_k > 0:
            top_k = min(top_k, logits.size(-1))
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits = logits.masked_fill(indices_to_remove, float('-inf'))

        # Apply top-p (nucleus) filtering
        if top_p is not None and top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices_to_remove.scatter(-1, sorted_indices, sorted_indices_to_remove)
            logits = logits.masked_fill(indices_to_remove, float('-inf'))

        # Apply min-p filtering
        if min_p is not None and min_p > 0:
            probs = F.softmax(logits, dim=-1)
            max_probs = probs.max(dim=-1, keepdim=True)[0]
            min_threshold = min_p * max_probs
            indices_to_remove = probs < min_threshold
            logits = logits.masked_fill(indices_to_remove, float('-inf'))

        return F.softmax(logits, dim=-1)

    def _compute_entropy(self, probs: torch.Tensor) -> torch.Tensor:
        """Compute entropy of probability distribution."""
        # Avoid log(0)
        log_probs = torch.log(probs.clamp(min=1e-10))
        entropy = -(probs * log_probs).sum(dim=-1)  # [B, W]
        return entropy

    def _restore_order(
        self,
        tensor: torch.Tensor,
        reordered_positions: torch.Tensor,
        state: StreamingDecoderState,
    ) -> torch.Tensor:
        """Restore tensor to original window order."""
        B, W = state.window_tokens.shape
        device = tensor.device

        # Create inverse permutation based on positions
        result = torch.zeros_like(tensor)

        for b in range(B):
            # Find the mapping from reordered positions back to original indices
            original_positions = state.window_positions[b]
            reordered = reordered_positions[b]

            for i, pos in enumerate(reordered):
                # Find where this position should go in original order
                original_idx = (original_positions == pos).nonzero(as_tuple=True)[0]
                if len(original_idx) > 0:
                    result[b, original_idx[0]] = tensor[b, i]

        return result

    def _select_positions_to_fill(
        self,
        state: StreamingDecoderState,
        entropies: torch.Tensor,
    ) -> torch.Tensor:
        """Select mask positions to fill based on distance-adjusted entropy."""
        B, W = state.window_tokens.shape
        device = entropies.device

        # Only consider unfilled positions
        is_mask = ~state.window_filled

        # Compute distance penalty (favor positions closer to the left)
        distances = torch.arange(W, device=device).float().unsqueeze(0).expand(B, -1)
        adjusted_entropy = entropies + self.distance_penalty * distances

        # Select positions with adjusted entropy below threshold
        to_fill = (adjusted_entropy < self.entropy_threshold) & is_mask

        # If no positions selected, select the one with lowest adjusted entropy
        if not to_fill.any():
            # For each batch, select lowest entropy mask position
            adjusted_entropy_masked = adjusted_entropy.clone()
            adjusted_entropy_masked[~is_mask] = float('inf')
            min_idx = adjusted_entropy_masked.argmin(dim=1)

            for b in range(B):
                if is_mask[b].any():
                    to_fill[b, min_idx[b]] = True

        return to_fill

    def _sample_tokens(
        self,
        probs: torch.Tensor,
        to_fill: torch.Tensor,
    ) -> torch.Tensor:
        """Sample tokens for positions to fill."""
        B, W, V = probs.shape
        device = probs.device

        new_tokens = torch.full((B, W), self.mask_token_id, device=device, dtype=torch.long)

        for b in range(B):
            fill_idx = to_fill[b].nonzero(as_tuple=True)[0]
            for idx in fill_idx:
                # Sample from distribution
                token = torch.multinomial(probs[b, idx], 1).item()
                new_tokens[b, idx] = token

        return new_tokens

    def _update_window(
        self,
        state: StreamingDecoderState,
        new_tokens: torch.Tensor,
        to_fill: torch.Tensor,
    ):
        """Update window with newly sampled tokens."""
        # Update tokens and filled status
        state.window_tokens = torch.where(
            to_fill,
            new_tokens,
            state.window_tokens,
        )
        state.window_filled = state.window_filled | to_fill

    def _commit_prefix(self, state: StreamingDecoderState) -> int:
        """Commit leftmost contiguous filled prefix."""
        B, W = state.window_tokens.shape

        # Find how many contiguous filled tokens from the left
        n_commit = W
        for b in range(B):
            # Find first unfilled position
            unfilled = (~state.window_filled[b]).nonzero(as_tuple=True)[0]
            if len(unfilled) > 0:
                n_commit = min(n_commit, unfilled[0].item())

        if n_commit == 0:
            return 0

        # Commit tokens
        for b in range(B):
            committed = state.window_tokens[b, :n_commit].tolist()
            state.generated_tokens[b].extend(committed)

        # Shift window
        if n_commit < W:
            state.window_tokens[:, :-n_commit] = state.window_tokens[:, n_commit:].clone()
            state.window_positions[:, :-n_commit] = state.window_positions[:, n_commit:].clone()
            state.window_filled[:, :-n_commit] = state.window_filled[:, n_commit:].clone()

        # Clear committed slots (will be refilled)
        state.window_tokens[:, -n_commit:] = self.mask_token_id
        state.window_positions[:, -n_commit:] = 0
        state.window_filled[:, -n_commit:] = False

        return n_commit

    def _refill_window(self, state: StreamingDecoderState, remaining_tokens: int):
        """Refill empty window slots with new mask tokens."""
        B, W = state.window_tokens.shape
        device = state.window_tokens.device

        # Find empty slots
        n_empty = (~state.window_filled).sum(dim=1).min().item()

        if n_empty == 0 or remaining_tokens <= 0:
            return

        # Refill with new positions
        n_refill = min(n_empty, remaining_tokens)

        for b in range(B):
            empty_idx = (~state.window_filled[b]).nonzero(as_tuple=True)[0]
            for i, idx in enumerate(empty_idx[:n_refill]):
                state.window_tokens[b, idx] = self.mask_token_id
                state.window_positions[b, idx] = state.next_position + i
                state.window_filled[b, idx] = False

        state.next_position += n_refill

    def _check_eos(self, state: StreamingDecoderState) -> bool:
        """Check if EOS token has been generated in all batches."""
        if self.eos_token_id is None:
            return False

        for b, tokens in enumerate(state.generated_tokens):
            if self.eos_token_id not in tokens:
                return False
        return True

    def _collect_results(
        self,
        prompt_ids: torch.Tensor,
        state: StreamingDecoderState,
    ) -> torch.Tensor:
        """Collect generated tokens into a tensor."""
        B, P = prompt_ids.shape
        device = prompt_ids.device

        # Find max generated length
        max_gen = max(len(tokens) for tokens in state.generated_tokens)

        # Create output tensor
        output = torch.full(
            (B, P + max_gen),
            self.pad_token_id,
            device=device,
            dtype=torch.long,
        )

        # Fill with prompt
        output[:, :P] = prompt_ids

        # Fill with generated tokens
        for b, tokens in enumerate(state.generated_tokens):
            if tokens:
                output[b, P:P + len(tokens)] = torch.tensor(tokens, device=device)

        return output


def create_streaming_decoder(
    model: nn.Module,
    config: WeDLMConfig,
    tokenizer,
) -> StreamingParallelDecoder:
    """
    Factory function to create a StreamingParallelDecoder.

    Args:
        model: Language model
        config: WeDLM configuration
        tokenizer: Tokenizer (for mask/pad/eos token IDs)

    Returns:
        StreamingParallelDecoder instance
    """
    # Get token IDs from tokenizer
    mask_token_id = config.mask_token_id
    if mask_token_id < 0 and hasattr(tokenizer, 'mask_token_id'):
        mask_token_id = tokenizer.mask_token_id

    pad_token_id = getattr(tokenizer, 'pad_token_id', None)
    if pad_token_id is None:
        pad_token_id = getattr(tokenizer, 'eos_token_id', mask_token_id)

    eos_token_id = getattr(tokenizer, 'eos_token_id', None)

    return StreamingParallelDecoder(
        model=model,
        config=config,
        mask_token_id=mask_token_id,
        pad_token_id=pad_token_id,
        eos_token_id=eos_token_id,
    )
