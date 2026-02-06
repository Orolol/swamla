"""Positional encoding methods for transformer models."""

import math
from typing import Optional, Tuple

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
        # Also convert to input dtype to avoid float32 upcast
        cos = cos.expand(B, H, T, -1).to(x.dtype).contiguous()
        sin = sin.expand(B, H, T, -1).to(x.dtype).contiguous()

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


class FoPE(nn.Module):
    """
    Fourier Position Embedding (FoPE) implementation.

    FoPE extends RoPE by modeling each dimension as a Fourier Series (multiple
    frequency components) rather than a single frequency. This improves length
    generalization by:
    1. Using learnable coefficients to combine multiple frequencies
    2. Zeroing out undertrained frequency components (below floor frequency)

    Based on: "Fourier Position Embedding: Enhancing Attention's Periodic
    Extension for Length Generalization" (arXiv:2412.17739)

    NOTE: Unlike RoPE, FoPE has learnable parameters (sin_coef, cos_coef).
    We do NOT cache the Fourier embeddings because:
    1. Gradients must flow through the coefficients on every forward pass
    2. Caching breaks gradient checkpointing (different tensor counts)

    Args:
        dim: Dimension of the position embeddings (typically qk_rope_head_dim)
        max_seq_len: Maximum sequence length for precomputation
        base: Base frequency for RoPE computation (default: 10000)
        n_harmonics: Number of harmonic components per dimension (default: 4)
        floor_ratio: Fraction of frequencies to zero out (default: 0.1)
        coef_init_std: Standard deviation for coefficient initialization (default: 0.3)
    """

    def __init__(
        self,
        dim: int,
        max_seq_len: int = 2048,
        base: float = 10000.0,
        n_harmonics: int = 4,
        floor_ratio: float = 0.1,
        coef_init_std: float = 0.3,
    ):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        self.n_harmonics = n_harmonics
        self.floor_ratio = floor_ratio
        self.half_dim = dim // 2

        # Compute base inverse frequencies (same as RoPE)
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq, persistent=False)

        # Determine which frequencies are above the floor threshold
        n_floor = int(self.half_dim * floor_ratio)
        n_active = self.half_dim - n_floor

        self.n_floor = n_floor
        self.n_active = n_active

        # Learnable Fourier coefficients for harmonics
        # Shape: [n_active, n_harmonics] for each of sin and cos
        if n_active > 0:
            self.sin_coef = nn.Parameter(
                torch.randn(n_active, n_harmonics) * coef_init_std
            )
            self.cos_coef = nn.Parameter(
                torch.randn(n_active, n_harmonics) * coef_init_std
            )

            # Harmonic multipliers: [1, 2, 3, 4, ...] for n_harmonics
            harmonic_mult = torch.arange(1, n_harmonics + 1).float()
            self.register_buffer('harmonic_mult', harmonic_mult, persistent=False)
        else:
            self.register_parameter('sin_coef', None)
            self.register_parameter('cos_coef', None)
            self.register_buffer('harmonic_mult', None, persistent=False)

    def _compute_embeddings(self, seq_len: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute Fourier position embeddings (always recomputed for gradient flow)."""
        t = torch.arange(seq_len, device=device).float()

        # For floor frequencies (low freq, long wavelength): use constant 1
        # This effectively disables rotation for these dimensions
        if self.n_floor > 0:
            floor_cos = torch.ones(seq_len, self.n_floor, device=device)
            floor_sin = torch.zeros(seq_len, self.n_floor, device=device)
        else:
            floor_cos = None
            floor_sin = None

        # For active frequencies: apply Fourier series with learnable coefficients
        if self.n_active > 0:
            # Get active inverse frequencies
            active_inv_freq = self.inv_freq[self.n_floor:]  # [n_active]

            # Compute base angles: [seq_len, n_active]
            base_angles = torch.outer(t, active_inv_freq)

            # Compute harmonic angles: [seq_len, n_active, n_harmonics]
            harmonic_angles = base_angles.unsqueeze(-1) * self.harmonic_mult.view(1, 1, -1)

            # Compute sin and cos for all harmonics
            sin_harmonics = torch.sin(harmonic_angles)
            cos_harmonics = torch.cos(harmonic_angles)

            # Create coefficient tensor with 1 for first harmonic (base frequency)
            # Higher harmonics use learned coefficients
            sin_weights = torch.cat([
                torch.ones(self.n_active, 1, device=device),
                self.sin_coef[:, 1:] if self.n_harmonics > 1 else torch.empty(self.n_active, 0, device=device)
            ], dim=1)
            cos_weights = torch.cat([
                torch.ones(self.n_active, 1, device=device),
                self.cos_coef[:, 1:] if self.n_harmonics > 1 else torch.empty(self.n_active, 0, device=device)
            ], dim=1)

            # Weighted sum of harmonics: [seq_len, n_active]
            active_sin = torch.einsum('snh,nh->sn', sin_harmonics, sin_weights)
            active_cos = torch.einsum('snh,nh->sn', cos_harmonics, cos_weights)

            # Normalize to prevent explosion (use detached norm for stability)
            with torch.no_grad():
                coef_var = self.sin_coef[:, 1:].pow(2).mean().item() if self.n_harmonics > 1 else 0
                norm_factor = math.sqrt(1 + (self.n_harmonics - 1) * coef_var)
            active_sin = active_sin / max(norm_factor, 1.0)
            active_cos = active_cos / max(norm_factor, 1.0)
        else:
            active_sin = None
            active_cos = None

        # Concatenate floor and active parts
        if floor_cos is not None and active_cos is not None:
            cos_emb = torch.cat([floor_cos, active_cos], dim=-1)
            sin_emb = torch.cat([floor_sin, active_sin], dim=-1)
        elif floor_cos is not None:
            cos_emb = floor_cos
            sin_emb = floor_sin
        else:
            cos_emb = active_cos
            sin_emb = active_sin

        # Duplicate for full dim (RoPE uses paired dimensions)
        cos_emb = torch.cat([cos_emb, cos_emb], dim=-1)
        sin_emb = torch.cat([sin_emb, sin_emb], dim=-1)

        # Reshape for broadcasting: [1, 1, seq_len, dim]
        return (
            cos_emb.view(1, 1, seq_len, self.dim),
            sin_emb.view(1, 1, seq_len, self.dim)
        )

    def forward(self, x: torch.Tensor, seq_len: Optional[int] = None) -> torch.Tensor:
        """
        Apply FoPE to input tensor.

        Args:
            x: Input tensor [B, H, T, D]
            seq_len: Optional sequence length (defaults to x.shape[-2])

        Returns:
            Rotated tensor with same shape as input
        """
        if seq_len is None:
            seq_len = x.shape[-2]

        B, H, T, D = x.shape

        # Always recompute embeddings - required for gradient flow and checkpointing compatibility
        cos_emb, sin_emb = self._compute_embeddings(T, x.device)

        # Ensure input is contiguous
        x = x.contiguous()

        # Reshape for rotation
        x_reshaped = x.view(B, H, T, D // 2, 2)
        x1, x2 = x_reshaped[..., 0].contiguous(), x_reshaped[..., 1].contiguous()

        # Get cos/sin for current sequence
        cos = cos_emb[:, :, :T, :(D//2)]
        sin = sin_emb[:, :, :T, :(D//2)]

        # Broadcast to batch and heads, convert to input dtype to avoid upcast
        cos = cos.expand(B, H, T, -1).to(x.dtype)
        sin = sin.expand(B, H, T, -1).to(x.dtype)

        # Apply rotation
        rotated = torch.stack([
            x1 * cos - x2 * sin,
            x2 * cos + x1 * sin,
        ], dim=-1)

        return rotated.view(B, H, T, D).contiguous()


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
 