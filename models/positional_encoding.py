"""Positional encoding methods for transformer models."""

import math
from typing import Optional

import torch
import torch.nn as nn

class RoPE(nn.Module):
    """
    Rotary Position Embeddings implementation.
    Based on the paper: https://arxiv.org/abs/2104.09864
    """
    def __init__(self, dim: int, max_seq_len: int = 2048, base: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        
        # Cache cos and sin values
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_seq_len).type_as(inv_freq)
        freqs = torch.einsum('i,j->ij', t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        
        # Reshape for broadcasting: [1, 1, seq_len, dim]
        self.register_buffer('cos_cached', emb.cos().view(1, 1, max_seq_len, dim), persistent=False)
        self.register_buffer('sin_cached', emb.sin().view(1, 1, max_seq_len, dim), persistent=False)

    def _rotate_half(self, x: torch.Tensor, seq_len: int) -> torch.Tensor:
        B, H, T, D = x.shape

        # Safety check for sequence length
        seq_len = min(seq_len, self.max_seq_len)

        # IMPORTANT: Ensure input is contiguous for torch.compile compatibility
        x = x.contiguous()

        # Reshape with explicit dimensions
        x_reshaped = x.view(B, H, T, D // 2, 2)
        x1, x2 = x_reshaped[..., 0].contiguous(), x_reshaped[..., 1].contiguous()

        # Make sure we have enough cached values
        if T > self.max_seq_len:
            # Extend the cache if needed
            self._extend_cos_sin_cache(T)

        # Get the cos and sin values for the current sequence length
        cos = self.cos_cached[:, :, :T, :(D//2)]
        sin = self.sin_cached[:, :, :T, :(D//2)]

        # Ensure broadcasting works correctly by explicitly matching dimensions
        cos = cos.expand(B, H, T, -1).contiguous()
        sin = sin.expand(B, H, T, -1).contiguous()

        # Apply rotation
        rotated = torch.stack([
            x1 * cos - x2 * sin,
            x2 * cos + x1 * sin,
        ], dim=-1)

        return rotated.view(B, H, T, D).contiguous()
        
    def _extend_cos_sin_cache(self, new_max_len):
        """Extend the cached cos and sin values if needed."""
        if new_max_len <= self.max_seq_len:
            return
            
        # Update max_seq_len
        old_max_len = self.max_seq_len
        self.max_seq_len = new_max_len
        
        # Recalculate for the new sequence length
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        inv_freq = inv_freq.to(self.cos_cached.device)
        t = torch.arange(self.max_seq_len, device=self.cos_cached.device).type_as(inv_freq)
        freqs = torch.einsum('i,j->ij', t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        
        # Create new cache
        cos_cached = emb.cos().view(1, 1, self.max_seq_len, self.dim)
        sin_cached = emb.sin().view(1, 1, self.max_seq_len, self.dim)
        
        # Update buffers
        self.register_buffer('cos_cached', cos_cached, persistent=False)
        self.register_buffer('sin_cached', sin_cached, persistent=False)

    def forward(self, x: torch.Tensor, seq_len: Optional[int] = None) -> torch.Tensor:
        if seq_len is None:
            seq_len = x.shape[-2]
        return self._rotate_half(x, seq_len)



def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> torch.Tensor:
    """Precompute the frequency tensor for complex exponentials (RoPE)."""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def precompute_freqs_cis_with_linear_scaling(
    dim: int,
    end: int,
    theta: float = 10000.0,
    scaling_factor: float = 1.0,
    original_seq_len: int = 2048
) -> torch.Tensor:
    """Precompute frequency tensor with linear scaling for extended context."""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device) / scaling_factor
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


def apply_rope(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """Apply rotary embeddings to input tensors."""
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    freqs_cis = freqs_cis.to(x_complex.device)
    x_rotated = x_complex * freqs_cis
    x_out = torch.view_as_real(x_rotated).flatten(3)
    return x_out.type_as(x)


def gather_freqs_by_positions(
    freqs_cis: torch.Tensor,
    position_ids: torch.Tensor,
) -> torch.Tensor:
    """
    Gather precomputed RoPE frequencies by arbitrary position indices.

    This is the key function enabling WeDLM's Topological Reordering:
    - Physical positions can differ from logical positions
    - RoPE is applied using logical positions for correct attention scores

    Args:
        freqs_cis: [max_len, dim//2] complex tensor of precomputed frequencies
        position_ids: [B, T] or [T] tensor of logical position indices

    Returns:
        gathered_freqs: [B, T, dim//2] or [T, dim//2] complex tensor
                        frequencies for the specified positions
    """
    if position_ids.dim() == 1:
        # Single sequence: [T] -> [T, dim//2]
        return freqs_cis[position_ids]
    else:
        # Batched: [B, T] -> [B, T, dim//2]
        B, T = position_ids.shape
        # Flatten, gather, reshape
        flat_positions = position_ids.view(-1)  # [B*T]
        gathered = freqs_cis[flat_positions]    # [B*T, dim//2]
        return gathered.view(B, T, -1)          # [B, T, dim//2]


def compute_yarn_inv_freq(
    dim: int,
    base: float = 10000.0,
    scale_factor: float = 1.0,
    beta_fast: float = 32.0,
    beta_slow: float = 1.0,
    original_max_seq_len: int = 2048,
) -> torch.Tensor:
    """
    Compute YaRN-scaled inverse frequencies using NTK-by-parts interpolation.

    YaRN (Yet another RoPE extensioN) enables context length extension at inference
    time by applying different scaling strategies to different frequency bands:
    - High frequencies (short wavelengths): Keep original frequencies (extrapolation)
    - Low frequencies (long wavelengths): Apply full interpolation
    - Middle frequencies: Smooth transition via ramp function

    Based on: "YaRN: Efficient Context Window Extension of Large Language Models"
    https://arxiv.org/abs/2309.00071

    Args:
        dim: Dimension of the RoPE embeddings (typically qk_rope_head_dim).
        base: Base frequency for RoPE computation (default: 10000.0).
        scale_factor: Context extension ratio (new_max_len / original_max_len).
                     Must be >= 1.0. When 1.0, returns standard RoPE frequencies.
        beta_fast: High frequency boundary for NTK-by-parts interpolation.
                  Frequencies with wavelength ratio > beta_fast get full interpolation.
        beta_slow: Low frequency boundary for NTK-by-parts interpolation.
                  Frequencies with wavelength ratio < beta_slow keep original values.
        original_max_seq_len: The context length the model was originally trained on.

    Returns:
        Tensor of shape [dim // 2] containing scaled inverse frequencies.
    """
    # Standard RoPE inverse frequency computation
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))

    # If no scaling needed, return standard frequencies
    if scale_factor <= 1.0:
        return inv_freq

    # Compute wavelengths: λ_d = 2π / inv_freq[d]
    wavelengths = 2 * math.pi / inv_freq

    # Compute wavelength ratios for the ramp function
    # r = λ_d / original_max_seq_len
    ratios = wavelengths / original_max_seq_len

    # Ramp function γ(r) for NTK-by-parts interpolation:
    # γ(r) = 0              if r < beta_slow  (high freq: extrapolation)
    # γ(r) = (r - β_s)/(β_f - β_s)  if beta_slow ≤ r ≤ beta_fast  (transition)
    # γ(r) = 1              if r > beta_fast  (low freq: interpolation)
    gamma = torch.clamp((ratios - beta_slow) / (beta_fast - beta_slow), 0.0, 1.0)

    # NTK-by-parts formula: h(θ) = (1-γ)*θ/s + γ*θ = θ * ((1-γ)/s + γ)
    # For inverse frequencies, we apply the same scaling
    scale_mult = (1 - gamma) / scale_factor + gamma
    scaled_inv_freq = inv_freq * scale_mult

    return scaled_inv_freq


def precompute_freqs_cis_yarn(
    dim: int,
    end: int,
    theta: float = 10000.0,
    scale_factor: float = 1.0,
    beta_fast: float = 32.0,
    beta_slow: float = 1.0,
    original_max_seq_len: int = 2048,
) -> torch.Tensor:
    """
    Precompute the frequency tensor for complex exponentials with YaRN scaling.

    This function combines YaRN's NTK-by-parts frequency interpolation with the
    standard RoPE complex exponential precomputation.

    Args:
        dim: Dimension of the RoPE embeddings.
        end: Maximum sequence length to precompute.
        theta: Base frequency (default: 10000.0).
        scale_factor: Context extension ratio (new_max_len / original_max_len).
        beta_fast: High frequency boundary for NTK-by-parts interpolation.
        beta_slow: Low frequency boundary for NTK-by-parts interpolation.
        original_max_seq_len: Original training context length.

    Returns:
        Complex tensor of shape [end, dim // 2] for RoPE application.
    """
    # Get YaRN-scaled inverse frequencies
    inv_freq = compute_yarn_inv_freq(
        dim,
        base=theta,
        scale_factor=scale_factor,
        beta_fast=beta_fast,
        beta_slow=beta_slow,
        original_max_seq_len=original_max_seq_len,
    )

    # Compute position indices
    t = torch.arange(end, device=inv_freq.device)

    # Compute frequencies for all positions: [end, dim//2]
    freqs = torch.outer(t, inv_freq)

    # Convert to complex exponentials
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)

    return freqs_cis


def apply_rope_with_positions(
    x: torch.Tensor,
    freqs_cis: torch.Tensor,
    position_ids: torch.Tensor,
) -> torch.Tensor:
    """
    Apply rotary embeddings using arbitrary position IDs.

    This supports WeDLM's topological reordering where physical order
    differs from logical positions.

    Args:
        x: [B, H, T, D] or [B, T, H, D] query/key tensor
        freqs_cis: [max_len, dim//2] precomputed frequencies
        position_ids: [B, T] logical position indices

    Returns:
        x_rotated: tensor with RoPE applied using specified positions
    """
    # Gather frequencies for the specified positions
    # freqs_cis: [max_len, dim//2] -> [B, T, dim//2]
    gathered_freqs = gather_freqs_by_positions(freqs_cis, position_ids)

    # Apply RoPE
    # x shape is typically [B, H, T, D] for attention
    # gathered_freqs is [B, T, dim//2], need to broadcast for heads
    if x.dim() == 4:
        B, H, T, D = x.shape
        # Reshape gathered_freqs for broadcasting: [B, 1, T, dim//2]
        gathered_freqs = gathered_freqs.unsqueeze(1)

    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    gathered_freqs = gathered_freqs.to(x_complex.device)
    x_rotated = x_complex * gathered_freqs
    x_out = torch.view_as_real(x_rotated).flatten(-2)

    return x_out.type_as(x)
 