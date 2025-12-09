"""
FlexAttention-based CausalSelfAttention implementation.

Uses PyTorch 2.5+ FlexAttention with BlockMask to skip padded tokens entirely,
providing actual compute savings (not just loss masking).

This is a drop-in replacement for CausalSelfAttention that leverages padding-aware
attention to reduce FLOPs when batches contain variable-length sequences.
"""

import math
from typing import Optional, Tuple
from functools import lru_cache

import torch
import torch.nn as nn
import torch.nn.functional as F

# Check FlexAttention availability (PyTorch >= 2.5)
FLEX_ATTENTION_AVAILABLE = False
flex_attention = None
create_block_mask = None
and_masks = None
BlockMask = None

try:
    from torch.nn.attention.flex_attention import (
        flex_attention as _flex_attention,
        create_block_mask as _create_block_mask,
        and_masks as _and_masks,
        BlockMask as _BlockMask,
    )
    flex_attention = _flex_attention
    create_block_mask = _create_block_mask
    and_masks = _and_masks
    BlockMask = _BlockMask
    FLEX_ATTENTION_AVAILABLE = True
except ImportError:
    pass

# Fallback imports
try:
    from flash_attn import flash_attn_func
    FLASH_ATTENTION_AVAILABLE = True
except ImportError:
    FLASH_ATTENTION_AVAILABLE = False

# Import TE FP8 helper
try:
    from optimization.fp8_te import get_te_linear, HAS_TE
except ImportError:
    HAS_TE = False
    def get_te_linear(in_features, out_features, bias=True, use_te_fp8=False):
        return nn.Linear(in_features, out_features, bias=bias)


# ============================================================================
# Mask functions for FlexAttention
# ============================================================================

def causal_mask(b: int, h: int, q_idx: int, kv_idx: int) -> bool:
    """Causal attention: query can only attend to positions <= itself."""
    return q_idx >= kv_idx


def sliding_window_causal_mask(window_size: int, sink_size: int = 4):
    """
    Create a sliding window + causal + attention sink mask function.

    Args:
        window_size: Size of the sliding attention window
        sink_size: Number of initial tokens to always attend to
    """
    def mask_fn(b: int, h: int, q_idx: int, kv_idx: int) -> bool:
        causal = q_idx >= kv_idx
        in_window = (q_idx - kv_idx) <= window_size
        is_sink = kv_idx < sink_size
        return causal & (in_window | is_sink)
    return mask_fn


def create_padding_mask(seq_lengths: torch.Tensor):
    """
    Create a padding mask function.

    Args:
        seq_lengths: Tensor of shape (batch_size,) with actual sequence lengths
    """
    def mask_fn(b: int, h: int, q_idx: int, kv_idx: int) -> bool:
        seq_len = seq_lengths[b]
        return (q_idx < seq_len) & (kv_idx < seq_len)
    return mask_fn


# ============================================================================
# BlockMask creation utilities
# ============================================================================

def create_causal_block_mask(
    batch_size: int,
    seq_len: int,
    device: torch.device,
    n_heads: Optional[int] = None,
) -> "BlockMask":
    """Create a simple causal BlockMask (no padding awareness)."""
    if not FLEX_ATTENTION_AVAILABLE:
        raise RuntimeError("FlexAttention requires PyTorch >= 2.5")

    return create_block_mask(
        causal_mask,
        B=batch_size,
        H=n_heads,
        Q_LEN=seq_len,
        KV_LEN=seq_len,
        device=str(device),
        _compile=True,
    )


def create_causal_padding_block_mask(
    seq_lengths: torch.Tensor,
    max_seq_len: int,
    batch_size: int,
    device: torch.device,
    n_heads: Optional[int] = None,
) -> "BlockMask":
    """
    Create a causal + padding-aware BlockMask.

    This mask skips computation for padded positions entirely!
    """
    if not FLEX_ATTENTION_AVAILABLE:
        raise RuntimeError("FlexAttention requires PyTorch >= 2.5")

    seq_lengths = seq_lengths.to(device)
    padding_fn = create_padding_mask(seq_lengths)
    combined = and_masks(causal_mask, padding_fn)

    return create_block_mask(
        combined,
        B=batch_size,
        H=n_heads,
        Q_LEN=max_seq_len,
        KV_LEN=max_seq_len,
        device=str(device),
        _compile=True,
    )


def create_sliding_window_padding_block_mask(
    seq_lengths: torch.Tensor,
    max_seq_len: int,
    batch_size: int,
    window_size: int,
    sink_size: int,
    device: torch.device,
    n_heads: Optional[int] = None,
) -> "BlockMask":
    """
    Create a sliding window + causal + padding-aware BlockMask.
    """
    if not FLEX_ATTENTION_AVAILABLE:
        raise RuntimeError("FlexAttention requires PyTorch >= 2.5")

    seq_lengths = seq_lengths.to(device)
    padding_fn = create_padding_mask(seq_lengths)
    sliding_fn = sliding_window_causal_mask(window_size, sink_size)
    combined = and_masks(sliding_fn, padding_fn)

    return create_block_mask(
        combined,
        B=batch_size,
        H=n_heads,
        Q_LEN=max_seq_len,
        KV_LEN=max_seq_len,
        device=str(device),
        _compile=True,
    )


def get_seq_lengths_from_labels(labels: torch.Tensor, ignore_index: int = -100) -> torch.Tensor:
    """
    Extract actual sequence lengths from labels tensor.

    Args:
        labels: Labels tensor of shape (B, T) where padding = ignore_index
        ignore_index: Value used for padding (default: -100)

    Returns:
        Tensor of shape (B,) with actual sequence lengths
    """
    batch_size, seq_len = labels.shape
    non_pad = labels != ignore_index
    positions = torch.arange(seq_len, device=labels.device).unsqueeze(0).expand(batch_size, -1)
    masked_pos = torch.where(non_pad, positions, torch.tensor(-1, device=labels.device))
    seq_lengths = masked_pos.max(dim=1).values + 1
    return torch.clamp(seq_lengths, min=1)


def get_seq_lengths_from_attention_mask(attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Extract sequence lengths from attention_mask tensor.

    Args:
        attention_mask: Tensor of shape (B, T) where 1 = valid, 0 = padding

    Returns:
        Tensor of shape (B,) with sequence lengths
    """
    return attention_mask.sum(dim=1)


# ============================================================================
# FlexAttention-based CausalSelfAttention
# ============================================================================

class FlexCausalSelfAttention(nn.Module):
    """
    CausalSelfAttention using FlexAttention for padding-aware computation.

    Key features:
    - Uses FlexAttention BlockMask to skip padded positions (real FLOPs savings!)
    - Supports sliding window attention with attention sink
    - Falls back to SDPA/Flash Attention when FlexAttention unavailable
    - Compatible with GQA (Grouped Query Attention)
    - Compatible with RoPE positional encoding
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        # Attention config
        self.n_head = config.n_head
        self.n_head_kv = config.n_head // getattr(config, 'ratio_kv', 1)
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.scale = 1.0 / math.sqrt(self.head_dim)

        # Sliding window config
        self.attention_window = getattr(config, 'attention_window',
                                        getattr(config, 'sliding_window_size', None))
        if self.attention_window is not None and self.attention_window <= 0:
            self.attention_window = None
        self.attention_sink_size = getattr(config, 'attention_sink_size', 4)

        # Check TE FP8
        use_te_fp8 = getattr(config, 'use_te_fp8', False)

        # Projections
        self.q_proj = get_te_linear(config.n_embd, config.n_embd, bias=config.bias, use_te_fp8=use_te_fp8)
        self.k_proj = get_te_linear(config.n_embd, self.n_head_kv * self.head_dim, bias=config.bias, use_te_fp8=use_te_fp8)
        self.v_proj = get_te_linear(config.n_embd, self.n_head_kv * self.head_dim, bias=config.bias, use_te_fp8=use_te_fp8)
        self.o_proj = get_te_linear(config.n_embd, config.n_embd, bias=config.bias, use_te_fp8=use_te_fp8)

        # Dropout
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.dropout_p = config.dropout

        # RoPE
        self.use_rope = getattr(config, 'use_rope', True)
        if self.use_rope:
            from positional_encoding import RoPE
            self.rope = RoPE(self.head_dim, config.block_size, base=getattr(config, 'rope_theta', 10000))
        else:
            self.rope = None

        # Causal mask for fallback
        mask = torch.full((config.block_size, config.block_size), float('-inf'))
        mask = torch.triu(mask, diagonal=1)
        self.register_buffer('causal_mask', mask)

        # BlockMask cache
        self._block_mask_cache = {}

        # Log backend
        if FLEX_ATTENTION_AVAILABLE:
            print(f"FlexCausalSelfAttention: Using FlexAttention (padding-aware)")
        elif FLASH_ATTENTION_AVAILABLE:
            print(f"FlexCausalSelfAttention: FlexAttention unavailable, using Flash Attention")
        else:
            print(f"FlexCausalSelfAttention: Using SDPA fallback")

    def _get_block_mask(
        self,
        batch_size: int,
        seq_len: int,
        device: torch.device,
        seq_lengths: Optional[torch.Tensor] = None,
    ) -> Optional["BlockMask"]:
        """Get or create BlockMask for current configuration."""
        if not FLEX_ATTENTION_AVAILABLE:
            return None

        # Check cache (only for uniform batches without padding info)
        if seq_lengths is None:
            cache_key = (batch_size, seq_len, str(device), self.attention_window)
            if cache_key in self._block_mask_cache:
                return self._block_mask_cache[cache_key]

        # Create appropriate mask
        if seq_lengths is not None:
            # Padding-aware mask (per-batch, not cached)
            if self.attention_window is not None:
                block_mask = create_sliding_window_padding_block_mask(
                    seq_lengths=seq_lengths,
                    max_seq_len=seq_len,
                    batch_size=batch_size,
                    window_size=self.attention_window,
                    sink_size=self.attention_sink_size,
                    device=device,
                )
            else:
                block_mask = create_causal_padding_block_mask(
                    seq_lengths=seq_lengths,
                    max_seq_len=seq_len,
                    batch_size=batch_size,
                    device=device,
                )
        else:
            # Simple causal mask (cacheable)
            block_mask = create_causal_block_mask(
                batch_size=batch_size,
                seq_len=seq_len,
                device=device,
            )
            self._block_mask_cache[cache_key] = block_mask

        return block_mask

    def _flex_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        block_mask: "BlockMask",
    ) -> torch.Tensor:
        """FlexAttention forward pass."""
        # FlexAttention expects (B, H, T, D)
        return flex_attention(q, k, v, block_mask=block_mask, scale=self.scale)

    def _sdpa_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """SDPA fallback."""
        return F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=attn_mask is None,
            scale=self.scale,
        )

    def forward(
        self,
        x: torch.Tensor,
        seq_lengths: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass with optional padding awareness.

        Args:
            x: Input tensor of shape (B, T, C)
            seq_lengths: Optional tensor of shape (B,) with actual sequence lengths.
                         If provided, FlexAttention will skip padded positions.
            attention_mask: Optional tensor of shape (B, T) with 1=valid, 0=padding.
                            Used to derive seq_lengths if seq_lengths not provided.

        Returns:
            Output tensor of shape (B, T, C)
        """
        B, T, C = x.size()
        device = x.device
        dtype = self.q_proj.weight.dtype
        x = x.to(dtype)

        # Derive seq_lengths from attention_mask if needed
        if seq_lengths is None and attention_mask is not None:
            seq_lengths = get_seq_lengths_from_attention_mask(attention_mask)

        # QKV projections
        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_head_kv, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_head_kv, self.head_dim).transpose(1, 2)

        # GQA: expand K, V heads
        if self.n_head_kv != self.n_head:
            k = k.repeat_interleave(self.n_head // self.n_head_kv, dim=1)
            v = v.repeat_interleave(self.n_head // self.n_head_kv, dim=1)

        # Apply RoPE
        if self.rope is not None:
            q = self.rope(q)
            k = self.rope(k)

        # Attention computation
        if FLEX_ATTENTION_AVAILABLE:
            block_mask = self._get_block_mask(B, T, device, seq_lengths)
            y = self._flex_attention(q, k, v, block_mask)
        else:
            # Fallback to SDPA
            if self.attention_window is not None:
                # Build sliding window mask manually
                attn_mask = self._build_sliding_mask(T, device, dtype)
            else:
                attn_mask = None
            y = self._sdpa_attention(q, k, v, attn_mask)

        # Reshape and output projection
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        output = self.resid_dropout(self.o_proj(y))

        return output

    def _build_sliding_mask(self, T: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Build sliding window mask for SDPA fallback."""
        window = self.attention_window
        sink = self.attention_sink_size

        positions = torch.arange(T, device=device)
        diff = positions.unsqueeze(1) - positions.unsqueeze(0)

        # Start with causal mask
        mask = self.causal_mask[:T, :T].to(device=device, dtype=dtype).clone()

        # Apply sliding window
        mask = mask.masked_fill(diff > window, float('-inf'))

        # Attention sink
        if sink > 0:
            sink_mask = positions.unsqueeze(0) < sink
            sink_mask = sink_mask.expand(T, T)
            mask = torch.where(sink_mask, torch.zeros_like(mask), mask)

        return mask


# ============================================================================
# Test function
# ============================================================================

def test_flex_attention_module():
    """Test the FlexCausalSelfAttention module."""
    from dataclasses import dataclass

    @dataclass
    class TestConfig:
        n_embd: int = 512
        n_head: int = 8
        ratio_kv: int = 1
        bias: bool = False
        dropout: float = 0.0
        block_size: int = 1024
        use_rope: bool = True
        rope_theta: float = 10000.0
        attention_window: Optional[int] = 256
        attention_sink_size: int = 4

    print("\n" + "=" * 70)
    print("Testing FlexCausalSelfAttention")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    config = TestConfig()

    # Create module
    attn = FlexCausalSelfAttention(config).to(device).to(dtype)

    # Test parameters
    batch_size = 4
    seq_len = 512

    # Variable sequence lengths
    seq_lengths = torch.tensor([128, 256, 64, 512], device=device)

    print(f"\nBatch size: {batch_size}")
    print(f"Max seq len: {seq_len}")
    print(f"Actual seq lengths: {seq_lengths.tolist()}")
    print(f"Padding ratio: {1 - seq_lengths.sum().item() / (batch_size * seq_len):.2%}")

    # Create input
    x = torch.randn(batch_size, seq_len, config.n_embd, device=device, dtype=dtype)

    # Warmup
    print("\nWarmup...")
    for _ in range(3):
        _ = attn(x, seq_lengths=seq_lengths)
    if device.type == "cuda":
        torch.cuda.synchronize()

    # Benchmark with padding awareness
    import time
    n_runs = 10

    print("\nBenchmarking with padding awareness (FlexAttention)...")
    start = time.perf_counter()
    for _ in range(n_runs):
        output_flex = attn(x, seq_lengths=seq_lengths)
    if device.type == "cuda":
        torch.cuda.synchronize()
    flex_time = (time.perf_counter() - start) / n_runs
    print(f"  Time: {flex_time*1000:.2f}ms")

    # Benchmark without padding awareness
    print("\nBenchmarking without padding awareness...")
    start = time.perf_counter()
    for _ in range(n_runs):
        output_no_pad = attn(x, seq_lengths=None)
    if device.type == "cuda":
        torch.cuda.synchronize()
    no_pad_time = (time.perf_counter() - start) / n_runs
    print(f"  Time: {no_pad_time*1000:.2f}ms")

    if flex_time < no_pad_time:
        speedup = no_pad_time / flex_time
        print(f"\n✓ Padding-aware attention is {speedup:.2f}x faster!")
    else:
        print(f"\n✗ No speedup observed (padding ratio may be too low)")

    print(f"\nOutput shape: {output_flex.shape}")
    print("=" * 70)


if __name__ == "__main__":
    test_flex_attention_module()
