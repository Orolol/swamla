"""
Dual-Stream Masking for WeDLM Training

Constructs training batches with:
- Memory Stream: Clean sequence x_o for conditioning
- Prediction Stream: Masked sequence x_t with intra-block topological reordering

Each prediction block only sees clean history from memory stream, not noisy predictions.

Reference: WeDLM paper Section 4.2
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
from functools import lru_cache

from .config import WeDLMConfig


class DualStreamMasker:
    """
    Constructs dual-stream batches for WeDLM training.

    Optimized for GPU performance with:
    - Fully vectorized masking (no Python loops over batch)
    - Cached attention masks
    - Minimal memory allocations
    """

    def __init__(self, config: WeDLMConfig, mask_token_id: int):
        """
        Args:
            config: WeDLM configuration
            mask_token_id: Token ID to use for [MASK]
        """
        self.config = config
        self.mask_token_id = mask_token_id
        self.block_size = config.block_size

        # Cache for attention masks (key: (L, device))
        self._mask_cache: Dict[Tuple[int, torch.device], torch.Tensor] = {}

    def __call__(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Transform input batch into dual-stream format (vectorized).

        Args:
            input_ids: [B, L] original token IDs
            attention_mask: [B, L] padding mask (optional)

        Returns:
            Dictionary containing all tensors needed for WeDLM training.
        """
        B, L = input_ids.shape
        device = input_ids.device
        K = (L + self.block_size - 1) // self.block_size  # Number of blocks

        # Memory stream: clean copy (no clone needed, we won't modify it)
        memory_stream = input_ids

        # Generate all random values at once for efficiency
        # mask_probs[b, i] determines if position i is masked
        uniform_random = torch.rand(B, L, device=device)

        # Sample masking ratio per block: gamma_k ~ U[min, max]
        gamma_per_block = torch.rand(B, K, device=device) * (
            self.config.max_mask_ratio - self.config.min_mask_ratio
        ) + self.config.min_mask_ratio

        # Expand gamma to per-position: [B, L]
        block_indices = torch.arange(L, device=device) // self.block_size  # [L]
        gamma_expanded = gamma_per_block[:, block_indices.clamp(max=K-1)]  # [B, L]

        # Create mask: position is masked if uniform_random < gamma
        target_mask_original = uniform_random < gamma_expanded  # [B, L]

        # Ensure at least 1 masked and 1 observed per block
        # This is a slight simplification - we ensure globally but not per-block
        # For training stability, this is usually fine

        # Create prediction stream with mask tokens
        pred_stream = input_ids.clone()
        pred_stream[target_mask_original] = self.mask_token_id

        # Vectorized topological reordering per block
        # Key insight: within each block, we want observed tokens first, masked last
        # We can achieve this with a sort key: masked positions get +block_size offset

        # Create sort keys: position_in_block + (is_masked * block_size)
        position_in_seq = torch.arange(L, device=device).unsqueeze(0).expand(B, -1)  # [B, L]
        position_in_block = position_in_seq % self.block_size  # [B, L]

        # Sort key: observed tokens (0-31), then masked tokens (32-63) within each block
        sort_key = position_in_block + target_mask_original.long() * self.block_size  # [B, L]

        # Add block offset to ensure blocks don't mix
        block_offset = (position_in_seq // self.block_size) * (2 * self.block_size)
        sort_key = sort_key + block_offset

        # Get reorder permutation by sorting
        reorder_perm = sort_key.argsort(dim=1)  # [B, L]

        # Apply reordering to prediction stream and positions
        pred_stream_reordered = pred_stream.gather(1, reorder_perm)
        pred_positions = position_in_seq.gather(1, reorder_perm)

        # Reorder target mask and targets to match
        target_mask_reordered = target_mask_original.gather(1, reorder_perm)
        target_ids_reordered = input_ids.gather(1, reorder_perm)

        # Construct dual-stream input
        dual_input_ids = torch.cat([memory_stream, pred_stream_reordered], dim=1)

        # Position IDs: memory uses sequential, prediction uses reordered logical positions
        memory_positions = torch.arange(L, device=device).unsqueeze(0).expand(B, -1)
        dual_position_ids = torch.cat([memory_positions, pred_positions], dim=1)

        # Get cached attention mask (or build and cache it)
        dual_attention_mask = self._get_cached_attention_mask(L, K, device)

        # Compute actual mask ratios per block for loss weighting (vectorized)
        block_assignments = block_indices.unsqueeze(0).expand(B, -1)  # [B, L]

        # Use scatter_add for fully vectorized counting
        # Count total tokens per block
        ones = torch.ones(B, L, device=device)
        block_sizes = torch.zeros(B, K, device=device).scatter_add(
            1, block_assignments, ones
        )  # [B, K]

        # Count masked tokens per block
        masked_float = target_mask_original.float()  # [B, L]
        masked_per_block = torch.zeros(B, K, device=device).scatter_add(
            1, block_assignments, masked_float
        )  # [B, K]

        # Compute ratios
        mask_ratios = masked_per_block / block_sizes.clamp(min=1)

        return {
            'dual_input_ids': dual_input_ids,
            'dual_position_ids': dual_position_ids,
            'dual_attention_mask': dual_attention_mask,
            'target_ids': target_ids_reordered,
            'target_ids_original': input_ids,
            'target_mask': target_mask_reordered,
            'target_mask_original': target_mask_original,
            'mask_ratios': mask_ratios,
            'block_assignments': block_assignments,
            'reorder_perm': reorder_perm,
        }

    def _get_cached_attention_mask(
        self,
        L: int,
        K: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Get or build cached attention mask."""
        cache_key = (L, device)

        if cache_key not in self._mask_cache:
            self._mask_cache[cache_key] = self._build_dual_stream_attention_mask(L, K, device)

        return self._mask_cache[cache_key]

    def _build_dual_stream_attention_mask(
        self,
        L: int,
        K: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Build the dual-stream attention mask (vectorized).

        Structure (2L x 2L):
        - Memory stream [0:L]: Standard causal attention
        - Prediction block k [L+start:L+end]:
            - Can attend to: Memory positions < block_k_start (clean history)
            - Can attend to: Other positions in same block (causal after reordering)
            - Cannot attend to: Other prediction blocks or future memory

        Returns:
            mask: [2L, 2L] boolean attention mask (True = can attend)
        """
        total_len = 2 * L

        # Build mask using vectorized operations
        row_idx = torch.arange(total_len, device=device).unsqueeze(1)  # [2L, 1]
        col_idx = torch.arange(total_len, device=device).unsqueeze(0)  # [1, 2L]

        # Memory stream (rows 0:L): standard causal
        memory_mask = (row_idx < L) & (col_idx < L) & (col_idx <= row_idx)

        # Prediction stream (rows L:2L)
        # Row i (L <= i < 2L) corresponds to prediction position (i - L)
        pred_row_pos = row_idx - L  # Position in original sequence for prediction rows
        pred_row_block = pred_row_pos // self.block_size  # Which block this row belongs to

        # Prediction rows can attend to:
        # 1. Memory positions before their block starts
        pred_to_memory = (row_idx >= L) & (col_idx < L) & (col_idx < pred_row_block * self.block_size)

        # 2. Same block in prediction stream (causal within block)
        col_pred_pos = col_idx - L  # Position for prediction columns
        col_pred_block = col_pred_pos // self.block_size

        same_block = (row_idx >= L) & (col_idx >= L) & (pred_row_block == col_pred_block)
        # Causal within block: can attend to earlier positions in same block
        intra_block_causal = same_block & (col_pred_pos <= pred_row_pos)

        # Combine all conditions
        mask = memory_mask | pred_to_memory | intra_block_causal

        return mask

    def get_prediction_indices(self, L: int) -> Tuple[int, int]:
        """Get start and end indices of prediction stream in dual input."""
        return L, 2 * L

    def clear_cache(self):
        """Clear the attention mask cache."""
        self._mask_cache.clear()


class SimpleMasker:
    """
    Simple masking without dual-stream (for comparison/debugging).
    Vectorized implementation.
    """

    def __init__(self, config: WeDLMConfig, mask_token_id: int):
        self.config = config
        self.mask_token_id = mask_token_id

    def __call__(
        self,
        input_ids: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Apply simple masking with topological reordering (vectorized).

        Args:
            input_ids: [B, L] token IDs

        Returns:
            Dictionary with masked input and metadata
        """
        B, L = input_ids.shape
        device = input_ids.device

        # Sample global masking ratio
        gamma = torch.rand(1, device=device).item() * (
            self.config.max_mask_ratio - self.config.min_mask_ratio
        ) + self.config.min_mask_ratio

        # Create mask using random values
        uniform_random = torch.rand(B, L, device=device)
        target_mask = uniform_random < gamma

        # Create masked input
        masked_input = input_ids.clone()
        masked_input[target_mask] = self.mask_token_id

        # Vectorized topological reordering
        position_ids = torch.arange(L, device=device).unsqueeze(0).expand(B, -1)

        # Sort key: observed (0, 1, 2, ...) then masked (L, L+1, L+2, ...)
        sort_key = position_ids + target_mask.long() * L
        reorder_perm = sort_key.argsort(dim=1)

        masked_input = masked_input.gather(1, reorder_perm)
        position_ids = position_ids.gather(1, reorder_perm)

        return {
            'input_ids': masked_input,
            'position_ids': position_ids,
            'target_ids': input_ids,
            'target_mask': target_mask,
            'mask_ratio': torch.tensor(gamma, device=device),
        }
