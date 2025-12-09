"""
FlexAttention utilities for efficient attention with padding-aware computation.

Uses PyTorch 2.5+ FlexAttention with BlockMask to skip padded tokens entirely,
providing actual compute savings (not just loss masking).

References:
- https://pytorch.org/blog/flexattention/
- https://medium.com/@lucasmgomez/casual-attention-with-padded-inputs-via-pytorch-flexattention-25e21b294551
- https://github.com/pytorch-labs/attention-gym
"""

import torch
from typing import Optional, Tuple, Callable
from functools import lru_cache

# Check FlexAttention availability (PyTorch >= 2.5)
FLEX_ATTENTION_AVAILABLE = False
try:
    from torch.nn.attention.flex_attention import (
        flex_attention,
        create_block_mask,
        and_masks,
        BlockMask,
    )
    FLEX_ATTENTION_AVAILABLE = True
    print("FlexAttention available (PyTorch >= 2.5)")
except ImportError:
    print("FlexAttention not available (requires PyTorch >= 2.5)")
    flex_attention = None
    create_block_mask = None
    and_masks = None
    BlockMask = None


def causal_mask(b: int, h: int, q_idx: int, kv_idx: int) -> bool:
    """Causal attention mask: query can only attend to previous positions."""
    return q_idx >= kv_idx


def create_padding_mask_fn(seq_lengths: torch.Tensor, max_len: int) -> Callable:
    """
    Create a padding mask function for FlexAttention.

    Args:
        seq_lengths: Tensor of shape (batch_size,) with actual sequence lengths
        max_len: Maximum sequence length (for validation)

    Returns:
        A mask_mod function compatible with create_block_mask
    """
    def padding_mask(b: int, h: int, q_idx: int, kv_idx: int) -> bool:
        # Both query and key positions must be within the actual sequence length
        seq_len = seq_lengths[b]
        return (q_idx < seq_len) & (kv_idx < seq_len)

    return padding_mask


def create_sliding_window_mask_fn(window_size: int, sink_size: int = 4) -> Callable:
    """
    Create a sliding window mask function with attention sink.

    Args:
        window_size: Size of the sliding attention window
        sink_size: Number of initial tokens to always attend to (attention sink)

    Returns:
        A mask_mod function compatible with create_block_mask
    """
    def sliding_window_mask(b: int, h: int, q_idx: int, kv_idx: int) -> bool:
        # Causal constraint
        causal = q_idx >= kv_idx
        # Window constraint (key must be within window of query)
        in_window = (q_idx - kv_idx) <= window_size
        # Attention sink (always attend to first N tokens)
        is_sink = kv_idx < sink_size
        return causal & (in_window | is_sink)

    return sliding_window_mask


def create_causal_padding_mask(
    seq_lengths: torch.Tensor,
    max_seq_len: int,
    batch_size: int,
    n_heads: Optional[int] = None,
    device: torch.device = None,
) -> "BlockMask":
    """
    Create a combined causal + padding BlockMask.

    This mask allows:
    - Causal attention (query attends to previous positions only)
    - Skipping padded positions entirely (actual compute savings!)

    Args:
        seq_lengths: Tensor of shape (batch_size,) with actual sequence lengths
        max_seq_len: Maximum sequence length in the batch
        batch_size: Batch size
        n_heads: Number of attention heads (None = head-agnostic mask)
        device: Device for the mask

    Returns:
        BlockMask for use with flex_attention
    """
    if not FLEX_ATTENTION_AVAILABLE:
        raise RuntimeError("FlexAttention not available. Requires PyTorch >= 2.5")

    if device is None:
        device = seq_lengths.device

    # Move seq_lengths to the right device
    seq_lengths = seq_lengths.to(device)

    # Create mask functions
    padding_fn = create_padding_mask_fn(seq_lengths, max_seq_len)

    # Combine causal + padding masks
    combined_mask = and_masks(causal_mask, padding_fn)

    # Create BlockMask with compilation for speed
    # H=None means the mask is the same for all heads
    block_mask = create_block_mask(
        combined_mask,
        B=batch_size,
        H=n_heads,
        Q_LEN=max_seq_len,
        KV_LEN=max_seq_len,
        device=str(device),
        _compile=True,  # Much faster mask creation
    )

    return block_mask


def create_causal_padding_sliding_mask(
    seq_lengths: torch.Tensor,
    max_seq_len: int,
    batch_size: int,
    window_size: int,
    sink_size: int = 4,
    n_heads: Optional[int] = None,
    device: torch.device = None,
) -> "BlockMask":
    """
    Create a combined causal + padding + sliding window BlockMask.

    This mask allows:
    - Causal attention
    - Sliding window (local attention)
    - Attention sink (always attend to first N tokens)
    - Skipping padded positions entirely

    Args:
        seq_lengths: Tensor of shape (batch_size,) with actual sequence lengths
        max_seq_len: Maximum sequence length in the batch
        batch_size: Batch size
        window_size: Sliding window size
        sink_size: Attention sink size
        n_heads: Number of attention heads
        device: Device for the mask

    Returns:
        BlockMask for use with flex_attention
    """
    if not FLEX_ATTENTION_AVAILABLE:
        raise RuntimeError("FlexAttention not available. Requires PyTorch >= 2.5")

    if device is None:
        device = seq_lengths.device

    seq_lengths = seq_lengths.to(device)

    # Create mask functions
    padding_fn = create_padding_mask_fn(seq_lengths, max_seq_len)
    sliding_fn = create_sliding_window_mask_fn(window_size, sink_size)

    # Combine all masks
    combined_mask = and_masks(sliding_fn, padding_fn)

    block_mask = create_block_mask(
        combined_mask,
        B=batch_size,
        H=n_heads,
        Q_LEN=max_seq_len,
        KV_LEN=max_seq_len,
        device=str(device),
        _compile=True,
    )

    return block_mask


def flex_attention_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_mask: "BlockMask",
    scale: Optional[float] = None,
) -> torch.Tensor:
    """
    Perform attention using FlexAttention with BlockMask.

    Args:
        q: Query tensor of shape (B, H, T, D)
        k: Key tensor of shape (B, H, T, D)
        v: Value tensor of shape (B, H, T, D)
        block_mask: BlockMask from create_*_mask functions
        scale: Optional attention scale (default: 1/sqrt(head_dim))

    Returns:
        Attention output of shape (B, H, T, D)
    """
    if not FLEX_ATTENTION_AVAILABLE:
        raise RuntimeError("FlexAttention not available. Requires PyTorch >= 2.5")

    if scale is None:
        scale = 1.0 / (q.size(-1) ** 0.5)

    # FlexAttention automatically skips masked blocks!
    output = flex_attention(
        q, k, v,
        block_mask=block_mask,
        scale=scale,
    )

    return output


# Utility to extract sequence lengths from labels
def get_seq_lengths_from_labels(labels: torch.Tensor, ignore_index: int = -100) -> torch.Tensor:
    """
    Extract actual sequence lengths from labels tensor.

    Args:
        labels: Labels tensor of shape (B, T) where padding positions have ignore_index
        ignore_index: The value used for padding (default: -100)

    Returns:
        Tensor of shape (B,) with actual sequence lengths
    """
    # Find the last non-padding position for each sequence
    non_pad_mask = labels != ignore_index  # (B, T)

    # Get indices of non-padding positions
    # We want the maximum index + 1 for each sequence
    batch_size, seq_len = labels.shape

    # Create position indices
    positions = torch.arange(seq_len, device=labels.device).unsqueeze(0).expand(batch_size, -1)

    # Mask out padding positions with -1
    masked_positions = torch.where(non_pad_mask, positions, torch.tensor(-1, device=labels.device))

    # Get max position + 1 = sequence length
    seq_lengths = masked_positions.max(dim=1).values + 1

    # Handle edge case where entire sequence is padding
    seq_lengths = torch.clamp(seq_lengths, min=1)

    return seq_lengths


# Test function
def test_flex_attention():
    """Test FlexAttention with padding."""
    if not FLEX_ATTENTION_AVAILABLE:
        print("Skipping test: FlexAttention not available")
        return

    print("\n" + "=" * 60)
    print("Testing FlexAttention with padding")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    # Test parameters
    batch_size = 4
    max_seq_len = 512
    n_heads = 8
    head_dim = 64

    # Variable sequence lengths (simulating padded batch)
    seq_lengths = torch.tensor([128, 256, 64, 512], device=device)

    print(f"\nBatch size: {batch_size}")
    print(f"Max seq len: {max_seq_len}")
    print(f"Actual seq lengths: {seq_lengths.tolist()}")
    print(f"Total real tokens: {seq_lengths.sum().item()}")
    print(f"Total positions: {batch_size * max_seq_len}")
    print(f"Padding ratio: {1 - seq_lengths.sum().item() / (batch_size * max_seq_len):.2%}")

    # Create random Q, K, V
    q = torch.randn(batch_size, n_heads, max_seq_len, head_dim, device=device, dtype=dtype)
    k = torch.randn(batch_size, n_heads, max_seq_len, head_dim, device=device, dtype=dtype)
    v = torch.randn(batch_size, n_heads, max_seq_len, head_dim, device=device, dtype=dtype)

    # Create BlockMask
    print("\nCreating BlockMask...")
    import time
    start = time.perf_counter()
    block_mask = create_causal_padding_mask(
        seq_lengths=seq_lengths,
        max_seq_len=max_seq_len,
        batch_size=batch_size,
        n_heads=None,  # Same mask for all heads
        device=device,
    )
    mask_time = time.perf_counter() - start
    print(f"BlockMask creation time: {mask_time*1000:.2f}ms")

    # Run FlexAttention
    print("\nRunning FlexAttention...")

    # Warmup
    for _ in range(3):
        _ = flex_attention_forward(q, k, v, block_mask)
    torch.cuda.synchronize() if device.type == "cuda" else None

    # Benchmark
    import time
    n_runs = 10
    start = time.perf_counter()
    for _ in range(n_runs):
        output = flex_attention_forward(q, k, v, block_mask)
    torch.cuda.synchronize() if device.type == "cuda" else None
    flex_time = (time.perf_counter() - start) / n_runs

    print(f"FlexAttention time: {flex_time*1000:.2f}ms")
    print(f"Output shape: {output.shape}")

    # Compare with standard SDPA (no sparsity)
    print("\nComparing with standard SDPA...")
    import torch.nn.functional as F

    # Warmup
    for _ in range(3):
        _ = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    torch.cuda.synchronize() if device.type == "cuda" else None

    # Benchmark
    start = time.perf_counter()
    for _ in range(n_runs):
        output_sdpa = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    torch.cuda.synchronize() if device.type == "cuda" else None
    sdpa_time = (time.perf_counter() - start) / n_runs

    print(f"SDPA time: {sdpa_time*1000:.2f}ms")
    print(f"Speedup: {sdpa_time/flex_time:.2f}x")

    print("\n" + "=" * 60)
    print("Test completed!")
    print("=" * 60)


if __name__ == "__main__":
    test_flex_attention()
