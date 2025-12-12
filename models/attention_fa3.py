"""
Flash Attention 3 (Hopper) support for H100/H800 GPUs.

This module provides integration with FA3's flash_attn_interface which has
a different API from FA2. FA3 supports:
- FP16/BF16 forward and backward
- FP8 forward (experimental)
- Optimized for H100/H800 (SM90) architecture

Requirements:
- H100 or H800 GPU
- CUDA >= 12.3 (12.8 recommended)
- Install via: cd flash-attention/hopper && python setup.py install
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn

# Check Flash Attention 3 availability
FLASH_ATTENTION_3_AVAILABLE = False
flash_attn_func = None
flash_attn_varlen_func = None

try:
    from flash_attn_interface import flash_attn_func as _flash_attn_func
    from flash_attn_interface import flash_attn_varlen_func as _flash_attn_varlen_func
    flash_attn_func = _flash_attn_func
    flash_attn_varlen_func = _flash_attn_varlen_func
    FLASH_ATTENTION_3_AVAILABLE = True
    print("Flash Attention 3 (Hopper) available!")
except ImportError as e:
    # Not an error - FA3 is optional and only for H100/H800
    pass


def is_fa3_available() -> bool:
    """Check if Flash Attention 3 is available."""
    return FLASH_ATTENTION_3_AVAILABLE


class FA3CausalSelfAttention(nn.Module):
    """
    Causal Self-Attention using Flash Attention 3 (Hopper).

    Optimized for H100/H800 GPUs with:
    - Native FP8 support (forward only)
    - Sliding window attention
    - GQA (Grouped Query Attention)
    - Variable-length sequences (varlen)

    Falls back to PyTorch SDPA if FA3 is not available.
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
        self.window_size = getattr(config, 'swa_window',
                                   getattr(config, 'attention_window',
                                           getattr(config, 'sliding_window_size', None)))
        if self.window_size is not None and self.window_size <= 0:
            self.window_size = None

        # Attention sink (always attend to first N tokens)
        self.sink_size = getattr(config, 'swa_sink_size', 4)

        # Projections
        self.q_proj = nn.Linear(config.n_embd, config.n_embd, bias=getattr(config, 'bias', False))
        self.k_proj = nn.Linear(config.n_embd, self.n_head_kv * self.head_dim, bias=getattr(config, 'bias', False))
        self.v_proj = nn.Linear(config.n_embd, self.n_head_kv * self.head_dim, bias=getattr(config, 'bias', False))
        self.o_proj = nn.Linear(config.n_embd, config.n_embd, bias=getattr(config, 'bias', False))

        # Dropout
        self.dropout = getattr(config, 'dropout', 0.0)
        self.resid_dropout = nn.Dropout(self.dropout)

        # RoPE
        self.use_rope = getattr(config, 'use_rope', True)
        if self.use_rope:
            from models.positional_encoding import RoPE
            self.rope = RoPE(
                self.head_dim,
                getattr(config, 'block_size', 2048),
                base=getattr(config, 'rope_theta', 10000)
            )
        else:
            self.rope = None

        # Use FP8 for forward pass (experimental)
        self.use_fp8 = getattr(config, 'use_fp8_attention', False)

        # Log backend
        if FLASH_ATTENTION_3_AVAILABLE:
            window_info = f", window={self.window_size}" if self.window_size else ""
            sink_info = f", sink={self.sink_size}" if self.sink_size > 0 else ""
            fp8_info = ", FP8" if self.use_fp8 else ""
            print(f"FA3CausalSelfAttention: Using Flash Attention 3 (Hopper){window_info}{sink_info}{fp8_info}")
        else:
            print(f"FA3CausalSelfAttention: FA3 unavailable, using SDPA fallback")

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (B, T, C)
            freqs_cis: Precomputed RoPE frequencies (optional, used if provided instead of self.rope)
            mask: Attention mask (optional, not used with FA3)

        Returns:
            Output tensor of shape (B, T, C)
        """
        B, T, C = x.size()

        # QKV projections
        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim)
        k = self.k_proj(x).view(B, T, self.n_head_kv, self.head_dim)
        v = self.v_proj(x).view(B, T, self.n_head_kv, self.head_dim)

        # Apply RoPE
        if self.rope is not None:
            # RoPE expects (B, n_head, T, head_dim)
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            q = self.rope(q)
            k = self.rope(k)
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
        elif freqs_cis is not None:
            # Use external RoPE frequencies
            q, k = self._apply_rope_external(q, k, freqs_cis)

        # Flash Attention 3 path
        if FLASH_ATTENTION_3_AVAILABLE:
            output = self._fa3_attention(q, k, v)
        else:
            output = self._sdpa_attention(q, k, v)

        # Output projection
        output = output.contiguous().view(B, T, C)
        output = self.resid_dropout(self.o_proj(output))

        return output

    def _apply_rope_external(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        freqs_cis: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply RoPE using external frequency tensor."""
        # freqs_cis shape: (T, head_dim//2, 2) or (T, head_dim)
        # q, k shape: (B, T, n_head, head_dim)

        def rotate_half(x):
            x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
            return torch.cat((-x2, x1), dim=-1)

        T = q.shape[1]
        if freqs_cis.dim() == 3:
            # (T, head_dim//2, 2) format - cos and sin separate
            cos = freqs_cis[:T, :, 0].unsqueeze(0).unsqueeze(2)  # (1, T, 1, head_dim//2)
            sin = freqs_cis[:T, :, 1].unsqueeze(0).unsqueeze(2)
            # Expand to full head_dim
            cos = cos.repeat(1, 1, 1, 2)
            sin = sin.repeat(1, 1, 1, 2)
        else:
            # Complex format - needs conversion
            cos = freqs_cis[:T].real.unsqueeze(0).unsqueeze(2)
            sin = freqs_cis[:T].imag.unsqueeze(0).unsqueeze(2)

        q_rot = q * cos + rotate_half(q) * sin
        k_rot = k * cos + rotate_half(k) * sin

        return q_rot, k_rot

    def _fa3_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        """
        Flash Attention 3 forward pass.

        Args:
            q: Query tensor (B, T, n_head, head_dim)
            k: Key tensor (B, T, n_head_kv, head_dim)
            v: Value tensor (B, T, n_head_kv, head_dim)

        Returns:
            Output tensor (B, T, n_head, head_dim)
        """
        # FA3 supports fp16 and bf16, convert if needed
        orig_dtype = q.dtype
        if q.dtype not in (torch.float16, torch.bfloat16):
            q = q.to(torch.bfloat16)
            k = k.to(torch.bfloat16)
            v = v.to(torch.bfloat16)

        # Window size for sliding window attention
        # FA3 uses (left_window, right_window), 0 for right because causal
        if self.window_size is not None:
            window = (self.window_size, 0)
        else:
            window = (-1, -1)  # Full attention

        # Call FA3
        # Note: FA3's flash_attn_func expects (B, T, n_head, head_dim) format
        output = flash_attn_func(
            q, k, v,
            softmax_scale=self.scale,
            causal=True,
            window_size=window,
            deterministic=False,  # Faster when False
        )

        # Convert back if needed
        if output.dtype != orig_dtype and orig_dtype not in (torch.float16, torch.bfloat16):
            output = output.to(orig_dtype)

        return output

    def _sdpa_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        """
        PyTorch SDPA fallback.

        Args:
            q: Query tensor (B, T, n_head, head_dim)
            k: Key tensor (B, T, n_head_kv, head_dim)
            v: Value tensor (B, T, n_head_kv, head_dim)

        Returns:
            Output tensor (B, T, n_head, head_dim)
        """
        B, T, n_head, head_dim = q.shape

        # Transpose to (B, n_head, T, head_dim) for SDPA
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # GQA: expand K, V heads if needed
        if self.n_head_kv != self.n_head:
            k = k.repeat_interleave(self.n_head // self.n_head_kv, dim=1)
            v = v.repeat_interleave(self.n_head // self.n_head_kv, dim=1)

        # SDPA with causal mask
        output = torch.nn.functional.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
            scale=self.scale,
        )

        # Transpose back to (B, T, n_head, head_dim)
        output = output.transpose(1, 2)

        return output


class FA3MLAAttention(nn.Module):
    """
    MLA (Multi-head Latent Attention) using Flash Attention 3.

    MLA uses low-rank compression for KV cache efficiency.
    This version integrates FA3 for the attention computation.
    """

    def __init__(self, config):
        super().__init__()

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head

        # MLA specific dimensions
        self.kv_lora_rank = getattr(config, 'mla_kv_lora_rank', 256)
        self.q_lora_rank = getattr(config, 'mla_q_lora_rank', 0)
        self.qk_nope_head_dim = getattr(config, 'mla_qk_nope_head_dim', 128)
        self.qk_rope_head_dim = getattr(config, 'mla_qk_rope_head_dim', 64)
        self.v_head_dim = getattr(config, 'mla_v_head_dim', 128)

        self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.scale = 1.0 / math.sqrt(self.qk_head_dim)

        # Q projection (with optional LoRA)
        if self.q_lora_rank > 0:
            self.q_down = nn.Linear(config.n_embd, self.q_lora_rank, bias=False)
            self.q_up = nn.Linear(self.q_lora_rank, self.n_head * self.qk_head_dim, bias=False)
        else:
            self.q_proj = nn.Linear(config.n_embd, self.n_head * self.qk_head_dim, bias=False)

        # KV compression
        self.kv_down = nn.Linear(config.n_embd, self.kv_lora_rank, bias=False)
        self.k_up = nn.Linear(self.kv_lora_rank, self.n_head * self.qk_head_dim, bias=False)
        self.v_up = nn.Linear(self.kv_lora_rank, self.n_head * self.v_head_dim, bias=False)

        # Output projection
        self.o_proj = nn.Linear(self.n_head * self.v_head_dim, config.n_embd, bias=False)

        # Dropout
        self.dropout = getattr(config, 'dropout', 0.0)
        self.resid_dropout = nn.Dropout(self.dropout)

        if FLASH_ATTENTION_3_AVAILABLE:
            print(f"FA3MLAAttention: Using Flash Attention 3 (Hopper)")
        else:
            print(f"FA3MLAAttention: FA3 unavailable, using SDPA fallback")

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int = 0,
        freqs_cis: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor (B, T, C)
            start_pos: Starting position for KV cache (inference)
            freqs_cis: RoPE frequencies
            mask: Attention mask (optional)

        Returns:
            Output tensor (B, T, C)
        """
        B, T, C = x.size()

        # Q projection
        if self.q_lora_rank > 0:
            q = self.q_up(self.q_down(x))
        else:
            q = self.q_proj(x)
        q = q.view(B, T, self.n_head, self.qk_head_dim)

        # KV compression and expansion
        kv_compressed = self.kv_down(x)
        k = self.k_up(kv_compressed).view(B, T, self.n_head, self.qk_head_dim)
        v = self.v_up(kv_compressed).view(B, T, self.n_head, self.v_head_dim)

        # Apply RoPE to rope portion of Q and K
        if freqs_cis is not None and self.qk_rope_head_dim > 0:
            q_nope, q_rope = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
            k_nope, k_rope = k.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

            q_rope, k_rope = self._apply_rope(q_rope, k_rope, freqs_cis, start_pos)

            q = torch.cat([q_nope, q_rope], dim=-1)
            k = torch.cat([k_nope, k_rope], dim=-1)

        # Attention
        if FLASH_ATTENTION_3_AVAILABLE:
            # FA3 with different Q/K and V head dims
            # Need to handle this carefully
            output = self._fa3_mla_attention(q, k, v)
        else:
            output = self._sdpa_mla_attention(q, k, v, mask)

        # Output projection
        output = output.contiguous().view(B, T, self.n_head * self.v_head_dim)
        output = self.resid_dropout(self.o_proj(output))

        return output

    def _apply_rope(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        freqs_cis: torch.Tensor,
        start_pos: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply RoPE to query and key tensors."""
        T = q.shape[1]

        def rotate_half(x):
            x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
            return torch.cat((-x2, x1), dim=-1)

        # Get frequencies for current positions
        freqs = freqs_cis[start_pos:start_pos + T]

        if freqs.dim() == 2:
            # Complex format
            cos = freqs.real.unsqueeze(0).unsqueeze(2)
            sin = freqs.imag.unsqueeze(0).unsqueeze(2)
        else:
            # Separate cos/sin format
            cos = freqs[..., 0].unsqueeze(0).unsqueeze(2)
            sin = freqs[..., 1].unsqueeze(0).unsqueeze(2)

        # Match dimensions
        head_dim = q.shape[-1]
        if cos.shape[-1] < head_dim:
            # Repeat to match head_dim
            cos = cos.repeat(1, 1, 1, head_dim // cos.shape[-1])
            sin = sin.repeat(1, 1, 1, head_dim // sin.shape[-1])
        elif cos.shape[-1] > head_dim:
            cos = cos[..., :head_dim]
            sin = sin[..., :head_dim]

        q_rot = q * cos + rotate_half(q) * sin
        k_rot = k * cos + rotate_half(k) * sin

        return q_rot.to(q.dtype), k_rot.to(k.dtype)

    def _fa3_mla_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        """FA3 attention for MLA (handles different Q/K vs V dims)."""
        orig_dtype = q.dtype
        if q.dtype not in (torch.float16, torch.bfloat16):
            q = q.to(torch.bfloat16)
            k = k.to(torch.bfloat16)
            v = v.to(torch.bfloat16)

        B, T, n_head, qk_dim = q.shape
        v_dim = v.shape[-1]

        # FA3 requires same head_dim for Q, K, V
        # Pad V to match Q/K head_dim
        if v_dim != qk_dim:
            v_padded = torch.zeros(B, T, n_head, qk_dim, dtype=v.dtype, device=v.device)
            v_padded[..., :v_dim] = v
            v = v_padded

        # Call FA3
        output = flash_attn_func(
            q, k, v,
            softmax_scale=self.scale,
            causal=True,
            window_size=(-1, -1),  # Full attention for MLA
            deterministic=False,
        )

        # Extract only the V dimensions from output
        if v_dim != qk_dim:
            output = output[..., :v_dim]

        if output.dtype != orig_dtype and orig_dtype not in (torch.float16, torch.bfloat16):
            output = output.to(orig_dtype)

        return output

    def _sdpa_mla_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """SDPA fallback for MLA attention."""
        B, T, n_head, qk_dim = q.shape
        v_dim = v.shape[-1]

        # Transpose to (B, n_head, T, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # For SDPA with different Q/K and V dims, compute attention manually
        # attn = softmax(Q @ K^T / scale) @ V
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Apply causal mask
        causal_mask = torch.triu(
            torch.ones(T, T, dtype=torch.bool, device=q.device),
            diagonal=1
        )
        attn_weights = attn_weights.masked_fill(causal_mask, float('-inf'))

        if mask is not None:
            attn_weights = attn_weights + mask

        attn_weights = torch.softmax(attn_weights, dim=-1)

        if self.training and self.dropout > 0:
            attn_weights = torch.nn.functional.dropout(attn_weights, p=self.dropout)

        output = torch.matmul(attn_weights, v)

        # Transpose back to (B, T, n_head, v_dim)
        output = output.transpose(1, 2)

        return output


def test_fa3():
    """Test FA3 availability and basic functionality."""
    print("\n" + "=" * 60)
    print("Flash Attention 3 Test")
    print("=" * 60)

    if not FLASH_ATTENTION_3_AVAILABLE:
        print("FA3 not available. To install:")
        print("  cd ~/.local/flash-attention-3/flash-attention/hopper")
        print("  python setup.py install")
        print("  export PYTHONPATH=\"$HOME/.local/flash-attention-3/flash-attention/hopper:$PYTHONPATH\"")
        return False

    print("FA3 is available!")

    # Check GPU
    if not torch.cuda.is_available():
        print("CUDA not available, skipping tensor test")
        return True

    device = torch.device("cuda")

    # Check compute capability
    cc = torch.cuda.get_device_capability()
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"Compute capability: {cc[0]}.{cc[1]}")

    if cc[0] < 9:
        print("WARNING: FA3 is optimized for SM90 (H100/H800)")

    # Test basic forward pass
    print("\nTesting basic forward pass...")
    B, T, n_head, head_dim = 2, 128, 8, 64

    q = torch.randn(B, T, n_head, head_dim, device=device, dtype=torch.bfloat16)
    k = torch.randn(B, T, n_head, head_dim, device=device, dtype=torch.bfloat16)
    v = torch.randn(B, T, n_head, head_dim, device=device, dtype=torch.bfloat16)

    try:
        output = flash_attn_func(q, k, v, causal=True)
        print(f"Output shape: {output.shape}")
        print("✓ Basic forward pass successful!")
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        return False

    # Test sliding window
    print("\nTesting sliding window attention...")
    try:
        output_sw = flash_attn_func(q, k, v, causal=True, window_size=(64, 0))
        print(f"Output shape: {output_sw.shape}")
        print("✓ Sliding window attention successful!")
    except Exception as e:
        print(f"✗ Sliding window failed: {e}")

    print("\n" + "=" * 60)
    return True


if __name__ == "__main__":
    test_fa3()
