"""
FlashAttention-based CausalSelfAttention with variable-length sequence support.

Supports both Flash Attention 3 (Hopper, for H100/H800) and Flash Attention 2.
FA3 uses flash_attn_interface, FA2 uses flash_attn module.

Uses flash_attn_varlen_func to skip padded tokens entirely, providing real
compute savings when batches contain variable-length sequences.
"""

import math
from typing import Optional

import torch
import torch.nn as nn

# Check Flash Attention availability
# Priority: FA3 (Hopper) > FA2
FLASH_ATTENTION_AVAILABLE = False
FLASH_ATTENTION_VERSION = None
flash_attn_func = None
flash_attn_varlen_func = None
unpad_input = None
pad_input = None

# Try Flash Attention 3 (Hopper) first - for H100/H800
try:
    from flash_attn_interface import flash_attn_func as _flash_attn_func_fa3
    from flash_attn_interface import flash_attn_varlen_func as _flash_attn_varlen_func_fa3
    flash_attn_func = _flash_attn_func_fa3
    flash_attn_varlen_func = _flash_attn_varlen_func_fa3
    FLASH_ATTENTION_AVAILABLE = True
    FLASH_ATTENTION_VERSION = 3
    print("Flash Attention 3 (Hopper) available!")
except ImportError:
    pass

# Fall back to Flash Attention 2 if FA3 not available
if not FLASH_ATTENTION_AVAILABLE:
    try:
        from flash_attn import flash_attn_func as _flash_attn_func_fa2
        from flash_attn import flash_attn_varlen_func as _flash_attn_varlen_func_fa2
        from flash_attn.bert_padding import unpad_input as _unpad_input, pad_input as _pad_input
        flash_attn_func = _flash_attn_func_fa2
        flash_attn_varlen_func = _flash_attn_varlen_func_fa2
        unpad_input = _unpad_input
        pad_input = _pad_input
        FLASH_ATTENTION_AVAILABLE = True
        FLASH_ATTENTION_VERSION = 2
        print("Flash Attention 2 available!")
    except ImportError:
        print("Flash Attention 2 not available: No module named 'flash_attn'")


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


def seq_lengths_to_attention_mask(seq_lengths: torch.Tensor, max_len: int) -> torch.Tensor:
    """
    Convert sequence lengths to attention mask.

    Args:
        seq_lengths: Tensor of shape (B,) with sequence lengths
        max_len: Maximum sequence length

    Returns:
        Tensor of shape (B, T) with 1=valid, 0=padding
    """
    batch_size = seq_lengths.shape[0]
    positions = torch.arange(max_len, device=seq_lengths.device).unsqueeze(0).expand(batch_size, -1)
    return (positions < seq_lengths.unsqueeze(1)).to(torch.int32)


class FlashCausalSelfAttention(nn.Module):
    """
    CausalSelfAttention using FlashAttention with variable-length support.

    Key features:
    - Uses flash_attn_varlen_func to skip padded positions (real FLOPs savings!)
    - Supports sliding window attention
    - Compatible with GQA (Grouped Query Attention)
    - Compatible with RoPE positional encoding
    - Falls back to standard SDPA when Flash Attention unavailable
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
        self.window_size = getattr(config, 'attention_window',
                                   getattr(config, 'sliding_window_size', None))
        if self.window_size is not None and self.window_size <= 0:
            self.window_size = None

        # Projections
        self.q_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.k_proj = nn.Linear(config.n_embd, self.n_head_kv * self.head_dim, bias=config.bias)
        self.v_proj = nn.Linear(config.n_embd, self.n_head_kv * self.head_dim, bias=config.bias)
        self.o_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

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

        # Causal mask for SDPA fallback
        mask = torch.full((config.block_size, config.block_size), float('-inf'))
        mask = torch.triu(mask, diagonal=1)
        self.register_buffer('causal_mask', mask)

        # Log backend
        if FLASH_ATTENTION_AVAILABLE:
            window_info = f", window={self.window_size}" if self.window_size else ""
            version_info = f"FA{FLASH_ATTENTION_VERSION}" if FLASH_ATTENTION_VERSION else "FA"
            print(f"FlashCausalSelfAttention: Using {version_info} (padding-aware{window_info})")
        else:
            print(f"FlashCausalSelfAttention: Flash Attention unavailable, using SDPA fallback")

    def _compute_cu_seqlens(self, seq_lengths: torch.Tensor) -> torch.Tensor:
        """
        Compute cumulative sequence lengths for flash_attn_varlen_func.

        Args:
            seq_lengths: Tensor of shape (B,) with sequence lengths

        Returns:
            Tensor of shape (B+1,) with cumulative lengths starting at 0
        """
        cu_seqlens = torch.zeros(seq_lengths.shape[0] + 1, dtype=torch.int32, device=seq_lengths.device)
        cu_seqlens[1:] = torch.cumsum(seq_lengths.to(torch.int32), dim=0)
        return cu_seqlens

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
                         If provided, Flash Attention will skip padded positions.
            attention_mask: Optional tensor of shape (B, T) with 1=valid, 0=padding.
                            Used to derive seq_lengths if seq_lengths not provided.

        Returns:
            Output tensor of shape (B, T, C)
        """
        B, T, C = x.size()
        device = x.device
        dtype = x.dtype

        # Derive attention_mask from seq_lengths if needed
        if attention_mask is None and seq_lengths is not None:
            attention_mask = seq_lengths_to_attention_mask(seq_lengths, T)

        # QKV projections: (B, T, C) -> (B, T, n_head, head_dim)
        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim)
        k = self.k_proj(x).view(B, T, self.n_head_kv, self.head_dim)
        v = self.v_proj(x).view(B, T, self.n_head_kv, self.head_dim)

        # Apply RoPE before unpadding (needs positional info)
        if self.rope is not None:
            # RoPE expects (B, n_head, T, head_dim), so transpose, apply, transpose back
            # IMPORTANT: Make tensors contiguous for torch.compile compatibility
            q = self.rope(q.transpose(1, 2).contiguous()).transpose(1, 2).contiguous()
            k = self.rope(k.transpose(1, 2).contiguous()).transpose(1, 2).contiguous()

        # Flash Attention only supports fp16 and bf16
        # RoPE or other ops might have converted to float32
        if FLASH_ATTENTION_AVAILABLE:
            if q.dtype not in (torch.float16, torch.bfloat16):
                q = q.to(torch.bfloat16)
                k = k.to(torch.bfloat16)
                v = v.to(torch.bfloat16)

        # Use Flash Attention with varlen if available and we have padding info
        if FLASH_ATTENTION_AVAILABLE and attention_mask is not None and FLASH_ATTENTION_VERSION == 2:
            # FA2 with varlen (requires unpad_input/pad_input helpers)
            output = self._flash_attention_varlen(q, k, v, attention_mask, B, T)
        elif FLASH_ATTENTION_AVAILABLE:
            # FA3 or FA2 without varlen - use standard flash attention path
            output = self._flash_attention_standard(q, k, v)
        else:
            # Fallback to SDPA
            output = self._sdpa_attention(q, k, v)

        # Output projection
        # Flash attention returns (B, T, n_head, head_dim), need to reshape to (B, T, C)
        output = output.contiguous().view(B, T, C)
        output = self.resid_dropout(self.o_proj(output))

        return output

    def _flash_attention_varlen(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: torch.Tensor,
        batch_size: int,
        max_seqlen: int,
    ) -> torch.Tensor:
        """
        Flash Attention with variable-length sequences (skips padding).

        Args:
            q, k, v: Tensors of shape (B, T, n_head, head_dim)
            attention_mask: Tensor of shape (B, T) with 1=valid, 0=padding
            batch_size: Batch size
            max_seqlen: Maximum sequence length

        Returns:
            Output tensor of shape (B, T, n_head, head_dim)
        """
        # Debug: print mask stats on first call
        if not hasattr(self, '_debug_printed'):
            valid_tokens = attention_mask.sum().item()
            total_tokens = attention_mask.numel()
            print(f"  [FlashAttn DEBUG] valid_tokens={valid_tokens}/{total_tokens} "
                  f"({100*valid_tokens/total_tokens:.1f}%), "
                  f"seq_lengths={attention_mask.sum(dim=1).tolist()[:4]}...")
            self._debug_printed = True

        # Unpad Q, K, V
        # unpad_input expects (B, T, ...) and attention_mask (B, T)
        q_unpadded, indices_q, cu_seqlens_q, max_seqlen_q, _ = unpad_input(
            q, attention_mask
        )
        k_unpadded, indices_k, cu_seqlens_k, max_seqlen_k, _ = unpad_input(
            k, attention_mask
        )
        v_unpadded, _, _, _, _ = unpad_input(
            v, attention_mask
        )

        # GQA: expand K, V heads if needed
        if self.n_head_kv != self.n_head:
            # k_unpadded: (total_tokens, n_head_kv, head_dim)
            # Need to expand to (total_tokens, n_head, head_dim)
            k_unpadded = k_unpadded.repeat_interleave(self.n_head // self.n_head_kv, dim=1)
            v_unpadded = v_unpadded.repeat_interleave(self.n_head // self.n_head_kv, dim=1)

        # Flash attention varlen
        # window_size is (left, right) for sliding window, (-1, -1) for full attention
        if self.window_size is not None:
            window = (self.window_size, 0)  # (left_window, right_window=0 for causal)
        else:
            window = (-1, -1)  # Full attention

        output_unpadded = flash_attn_varlen_func(
            q_unpadded,
            k_unpadded,
            v_unpadded,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            dropout_p=self.dropout_p if self.training else 0.0,
            causal=True,
            window_size=window,
            softmax_scale=self.scale,
        )

        # Repad output
        output = pad_input(output_unpadded, indices_q, batch_size, max_seqlen)

        return output

    def _flash_attention_standard(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        """
        Standard Flash Attention (no variable length, all tokens computed).
        Works with both FA2 and FA3.

        Args:
            q, k, v: Tensors of shape (B, T, n_head, head_dim)

        Returns:
            Output tensor of shape (B, T, n_head, head_dim)
        """
        # GQA: expand K, V heads if needed
        if self.n_head_kv != self.n_head:
            k = k.repeat_interleave(self.n_head // self.n_head_kv, dim=2)
            v = v.repeat_interleave(self.n_head // self.n_head_kv, dim=2)

        # window_size for sliding window
        if self.window_size is not None:
            window = (self.window_size, 0)
        else:
            window = (-1, -1)

        # Use the globally imported flash_attn_func (FA2 or FA3)
        if FLASH_ATTENTION_VERSION == 3:
            # FA3 API: slightly different parameter names
            output = flash_attn_func(
                q, k, v,
                softmax_scale=self.scale,
                causal=True,
                window_size=window,
            )
        else:
            # FA2 API
            output = flash_attn_func(
                q, k, v,
                dropout_p=self.dropout_p if self.training else 0.0,
                causal=True,
                window_size=window,
                softmax_scale=self.scale,
            )

        return output

    def _sdpa_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        """
        SDPA fallback (no padding awareness).

        Args:
            q, k, v: Tensors of shape (B, T, n_head, head_dim)

        Returns:
            Output tensor of shape (B, T, n_head, head_dim)
        """
        B, T, n_head, head_dim = q.shape

        # Transpose to (B, n_head, T, head_dim) for SDPA
        # IMPORTANT: Make tensors contiguous for torch.compile compatibility
        q = q.transpose(1, 2).contiguous()
        k = k.transpose(1, 2).contiguous()
        v = v.transpose(1, 2).contiguous()

        # GQA: expand K, V heads if needed
        if self.n_head_kv != self.n_head:
            k = k.repeat_interleave(self.n_head // self.n_head_kv, dim=1)
            v = v.repeat_interleave(self.n_head // self.n_head_kv, dim=1)

        # Use causal mask
        output = torch.nn.functional.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=True,
            scale=self.scale,
        )

        # Transpose back to (B, T, n_head, head_dim)
        output = output.transpose(1, 2).contiguous()

        return output


# ============================================================================
# Test function
# ============================================================================

def test_flash_attention_module():
    """Test the FlashCausalSelfAttention module."""
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

    print("\n" + "=" * 70)
    print("Testing FlashCausalSelfAttention")
    print("=" * 70)

    if not FLASH_ATTENTION_AVAILABLE:
        print("Flash Attention not available, skipping test")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("CUDA not available, skipping test")
        return

    dtype = torch.bfloat16

    config = TestConfig()

    # Create module
    attn = FlashCausalSelfAttention(config).to(device).to(dtype)

    # Test parameters
    batch_size = 4
    seq_len = 512

    # Variable sequence lengths
    seq_lengths = torch.tensor([128, 256, 64, 512], device=device)
    attention_mask = seq_lengths_to_attention_mask(seq_lengths, seq_len)

    print(f"\nBatch size: {batch_size}")
    print(f"Max seq len: {seq_len}")
    print(f"Actual seq lengths: {seq_lengths.tolist()}")
    padding_ratio = 1 - seq_lengths.sum().item() / (batch_size * seq_len)
    print(f"Padding ratio: {padding_ratio:.2%}")

    # Create input
    x = torch.randn(batch_size, seq_len, config.n_embd, device=device, dtype=dtype)

    # Warmup
    print("\nWarmup...")
    for _ in range(3):
        _ = attn(x, seq_lengths=seq_lengths)
    torch.cuda.synchronize()

    # Benchmark with padding awareness
    import time
    n_runs = 20

    print("\nBenchmarking with padding awareness (varlen)...")
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(n_runs):
        output_varlen = attn(x, seq_lengths=seq_lengths)
    torch.cuda.synchronize()
    varlen_time = (time.perf_counter() - start) / n_runs
    print(f"  Time: {varlen_time*1000:.2f}ms")

    # Benchmark without padding awareness
    print("\nBenchmarking without padding awareness (standard)...")
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(n_runs):
        output_standard = attn(x, seq_lengths=None)
    torch.cuda.synchronize()
    standard_time = (time.perf_counter() - start) / n_runs
    print(f"  Time: {standard_time*1000:.2f}ms")

    if varlen_time < standard_time:
        speedup = standard_time / varlen_time
        print(f"\n✓ Padding-aware attention is {speedup:.2f}x faster!")
    else:
        slowdown = varlen_time / standard_time
        print(f"\n✗ Padding-aware attention is {slowdown:.2f}x slower (overhead > savings)")

    # Verify output shapes
    print(f"\nOutput shape: {output_varlen.shape}")
    assert output_varlen.shape == x.shape, f"Shape mismatch: {output_varlen.shape} vs {x.shape}"
    print("✓ Output shape correct")

    # Test backward pass
    print("\nTesting backward pass...")
    x_grad = x.clone().requires_grad_(True)
    output = attn(x_grad, seq_lengths=seq_lengths)
    loss = output.sum()
    loss.backward()
    assert x_grad.grad is not None, "Gradients not computed"
    print("✓ Backward pass successful")

    print("=" * 70)


if __name__ == "__main__":
    test_flash_attention_module()
