"""
Topological Reordering for WeDLM

Key innovation: Physically reorder tokens to [observed | masked] while preserving
logical positions via RoPE position IDs. This allows masked tokens to access the
full observed context under standard causal attention.

Reference: WeDLM paper Section 4.1
"""

import torch
from typing import Tuple, Optional


def topological_reorder(
    tokens: torch.Tensor,
    mask_positions: torch.Tensor,
    position_ids: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Reorder sequence: [observed tokens | masked tokens]
    Preserve logical positions via position_ids for RoPE.

    This is the core operation that enables mask recovery under causal attention:
    - Observed tokens are placed at the physical prefix
    - Masked tokens are placed at the physical suffix
    - Logical positions are preserved for RoPE computation

    Args:
        tokens: [B, L] token IDs
        mask_positions: [B, L] boolean mask (True = masked position)
        position_ids: [B, L] original logical positions, defaults to [0, 1, ..., L-1]

    Returns:
        reordered_tokens: [B, L] with observed first, masked last
        reordered_position_ids: [B, L] preserving original logical positions
        inverse_permutation: [B, L] to recover original order
    """
    B, L = tokens.shape
    device = tokens.device

    if position_ids is None:
        position_ids = torch.arange(L, device=device).unsqueeze(0).expand(B, -1)

    # Vectorized implementation for efficiency
    reordered_tokens = torch.zeros_like(tokens)
    reordered_position_ids = torch.zeros_like(position_ids)
    inverse_permutation = torch.zeros(B, L, dtype=torch.long, device=device)

    for b in range(B):
        # Get indices of observed (not masked) and masked positions
        obs_mask = ~mask_positions[b]
        obs_indices = obs_mask.nonzero(as_tuple=True)[0]
        mask_indices = mask_positions[b].nonzero(as_tuple=True)[0]

        # Concatenate: observed first, masked last
        permutation = torch.cat([obs_indices, mask_indices])

        # Apply permutation
        reordered_tokens[b] = tokens[b, permutation]
        reordered_position_ids[b] = position_ids[b, permutation]

        # Compute inverse permutation for recovery
        inverse_permutation[b, permutation] = torch.arange(L, device=device)

    return reordered_tokens, reordered_position_ids, inverse_permutation


def topological_reorder_batched(
    tokens: torch.Tensor,
    mask_positions: torch.Tensor,
    position_ids: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Batched version of topological reorder with additional metadata.

    Returns:
        reordered_tokens: [B, L]
        reordered_position_ids: [B, L]
        inverse_permutation: [B, L]
        num_observed: [B] number of observed tokens per batch
    """
    B, L = tokens.shape
    device = tokens.device

    if position_ids is None:
        position_ids = torch.arange(L, device=device).unsqueeze(0).expand(B, -1)

    reordered_tokens = torch.zeros_like(tokens)
    reordered_position_ids = torch.zeros_like(position_ids)
    inverse_permutation = torch.zeros(B, L, dtype=torch.long, device=device)
    num_observed = torch.zeros(B, dtype=torch.long, device=device)

    for b in range(B):
        obs_mask = ~mask_positions[b]
        obs_indices = obs_mask.nonzero(as_tuple=True)[0]
        mask_indices = mask_positions[b].nonzero(as_tuple=True)[0]

        num_observed[b] = len(obs_indices)
        permutation = torch.cat([obs_indices, mask_indices])

        reordered_tokens[b] = tokens[b, permutation]
        reordered_position_ids[b] = position_ids[b, permutation]
        inverse_permutation[b, permutation] = torch.arange(L, device=device)

    return reordered_tokens, reordered_position_ids, inverse_permutation, num_observed


def inverse_reorder(
    tensor: torch.Tensor,
    inverse_permutation: torch.Tensor,
) -> torch.Tensor:
    """
    Restore original sequence order from reordered tensor.

    Args:
        tensor: [B, L, ...] reordered tensor (can have extra dimensions)
        inverse_permutation: [B, L] inverse permutation from topological_reorder

    Returns:
        original_order_tensor: [B, L, ...] tensor in original order
    """
    B, L = tensor.shape[:2]
    device = tensor.device

    # Handle tensors with extra dimensions (e.g., logits [B, L, V])
    result = torch.zeros_like(tensor)

    for b in range(B):
        result[b] = tensor[b, inverse_permutation[b]]

    return result


def intra_block_reorder(
    tokens: torch.Tensor,
    mask_positions: torch.Tensor,
    block_size: int,
    position_ids: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply topological reordering within each block.
    Used for dual-stream masking where each prediction block is reordered independently.

    Args:
        tokens: [B, L] token IDs
        mask_positions: [B, L] boolean mask
        block_size: size of each block
        position_ids: [B, L] original positions

    Returns:
        reordered_tokens: [B, L] with intra-block reordering
        reordered_position_ids: [B, L]
    """
    B, L = tokens.shape
    device = tokens.device

    if position_ids is None:
        position_ids = torch.arange(L, device=device).unsqueeze(0).expand(B, -1).clone()
    else:
        position_ids = position_ids.clone()

    tokens = tokens.clone()
    num_blocks = (L + block_size - 1) // block_size

    for k in range(num_blocks):
        start = k * block_size
        end = min((k + 1) * block_size, L)

        for b in range(B):
            block_mask = mask_positions[b, start:end]
            obs_idx = (~block_mask).nonzero(as_tuple=True)[0]
            mask_idx = block_mask.nonzero(as_tuple=True)[0]

            if len(mask_idx) > 0:  # Only reorder if there are masked tokens
                block_perm = torch.cat([obs_idx, mask_idx])

                # Store original values
                block_tokens = tokens[b, start:end].clone()
                block_pos = position_ids[b, start:end].clone()

                # Apply permutation
                tokens[b, start:end] = block_tokens[block_perm]
                position_ids[b, start:end] = block_pos[block_perm]

    return tokens, position_ids


def create_reordered_causal_mask(
    seq_len: int,
    num_observed: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Create causal attention mask for reordered sequence.

    Under topological reordering:
    - Observed tokens (positions 0 to num_observed-1) use standard causal
    - Masked tokens (positions num_observed to seq_len-1) can attend to all observed

    Args:
        seq_len: total sequence length
        num_observed: number of observed tokens at prefix
        device: torch device

    Returns:
        mask: [seq_len, seq_len] attention mask (True = can attend)
    """
    # Standard causal mask
    mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))

    # Masked tokens can attend to all observed tokens
    # This is already satisfied by causal mask since observed are at prefix

    return mask
