"""
FlexAttention mask functions for WeDLM dual-stream attention.

FlexAttention (PyTorch 2.5+) allows efficient attention with arbitrary masks
by compiling the mask function into optimized CUDA kernels.

This is significantly faster than using dense attention masks with SDPA.
"""

import torch
from typing import Optional, Callable, Tuple
from functools import lru_cache

# Import FlexAttention
try:
    from torch.nn.attention.flex_attention import (
        flex_attention,
        create_block_mask,
        BlockMask,
    )
    FLEX_ATTENTION_AVAILABLE = True
except ImportError:
    FLEX_ATTENTION_AVAILABLE = False
    flex_attention = None
    create_block_mask = None
    BlockMask = None


def create_wedlm_mask_mod(
    seq_len: int,
    block_size: int,
) -> Callable:
    """
    Create a mask_mod function for WeDLM dual-stream attention.

    The mask follows this structure for a sequence of length 2L:
    - Memory stream [0:L]: Standard causal attention within memory
    - Prediction stream [L:2L]:
        - Block k [L+k*B : L+(k+1)*B] can attend to:
            - Memory positions [0 : k*B) (clean history)
            - Its own block causally (after reordering: observed first, masked last)
        - Cannot attend to other prediction blocks or future memory

    Args:
        seq_len: Length of the original sequence (L), total will be 2L
        block_size: Size of each prediction block

    Returns:
        mask_mod function compatible with flex_attention
    """
    L = seq_len
    B = block_size

    def mask_mod(b, h, q_idx, kv_idx):
        """
        Mask function for WeDLM dual-stream.

        Args:
            b: batch index
            h: head index
            q_idx: query position (0 to 2L-1)
            kv_idx: key/value position (0 to 2L-1)

        Returns:
            True if attention is allowed, False otherwise
        """
        # Memory stream queries (0 <= q_idx < L): standard causal
        is_memory_query = q_idx < L
        memory_causal = is_memory_query & (kv_idx < L) & (kv_idx <= q_idx)

        # Prediction stream queries (L <= q_idx < 2L)
        is_pred_query = q_idx >= L
        pred_pos = q_idx - L  # Position in original sequence
        pred_block = pred_pos // B  # Which block this query belongs to

        # Prediction can attend to memory before its block starts
        pred_to_memory = is_pred_query & (kv_idx < L) & (kv_idx < pred_block * B)

        # Prediction can attend to same block in prediction stream (causal)
        kv_pred_pos = kv_idx - L
        kv_pred_block = kv_pred_pos // B
        same_block = is_pred_query & (kv_idx >= L) & (pred_block == kv_pred_block)
        intra_block_causal = same_block & (kv_pred_pos <= pred_pos)

        return memory_causal | pred_to_memory | intra_block_causal

    return mask_mod


@lru_cache(maxsize=8)
def get_wedlm_block_mask(
    seq_len: int,
    block_size: int,
    num_heads: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> "BlockMask":
    """
    Create and cache a compiled BlockMask for WeDLM attention.

    Args:
        seq_len: Original sequence length (L), total will be 2L
        block_size: WeDLM prediction block size
        num_heads: Number of attention heads
        device: Target device
        dtype: Data type for the mask

    Returns:
        BlockMask object for use with flex_attention
    """
    if not FLEX_ATTENTION_AVAILABLE:
        raise RuntimeError("FlexAttention not available. Requires PyTorch 2.5+")

    mask_mod = create_wedlm_mask_mod(seq_len, block_size)
    total_len = 2 * seq_len

    # Create block mask - this compiles the mask function
    block_mask = create_block_mask(
        mask_mod,
        B=None,  # Batch-independent mask
        H=None,  # Head-independent mask
        Q_LEN=total_len,
        KV_LEN=total_len,
        device=str(device),  # create_block_mask expects string device
    )

    return block_mask


def wedlm_flex_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seq_len: int,
    block_size: int,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """
    Compute attention using FlexAttention with WeDLM mask.

    Args:
        q: Query tensor [B, H, 2L, D]
        k: Key tensor [B, H, 2L, D]
        v: Value tensor [B, H, 2L, D]
        seq_len: Original sequence length (L)
        block_size: WeDLM prediction block size
        scale: Optional attention scale (default: 1/sqrt(D))

    Returns:
        Attention output [B, H, 2L, D]
    """
    if not FLEX_ATTENTION_AVAILABLE:
        raise RuntimeError("FlexAttention not available")

    B, H, total_len, D = q.shape
    assert total_len == 2 * seq_len, f"Expected 2*{seq_len}={2*seq_len}, got {total_len}"

    # Get cached block mask
    block_mask = get_wedlm_block_mask(
        seq_len=seq_len,
        block_size=block_size,
        num_heads=H,
        device=q.device,
        dtype=q.dtype,
    )

    # Run flex_attention
    output = flex_attention(
        q, k, v,
        block_mask=block_mask,
        scale=scale,
    )

    return output


def clear_mask_cache():
    """Clear the cached block masks."""
    get_wedlm_block_mask.cache_clear()
