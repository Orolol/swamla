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
 