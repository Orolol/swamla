"""
YaRN Configuration Contract

This file defines the expected interface for YaRN configuration.
It serves as a contract between the specification and implementation.
"""

from dataclasses import dataclass
from typing import Optional
import math


@dataclass
class YaRNConfig:
    """
    Configuration for YaRN (Yet another RoPE extensioN) context extension.

    Attributes:
        enabled: Whether YaRN is active. When False, standard RoPE is used.
        scale_factor: Context extension ratio (new_max_len / original_max_len).
                     Must be >= 1.0.
        original_max_seq_len: The context length the model was originally trained on.
        beta_fast: High frequency boundary for NTK-by-parts interpolation.
                  Frequencies with wavelength ratio > beta_fast get full interpolation.
        beta_slow: Low frequency boundary for NTK-by-parts interpolation.
                  Frequencies with wavelength ratio < beta_slow keep original values.
        attn_factor: Temperature scaling factor for attention.
                    If None, computed automatically from scale_factor.
    """
    enabled: bool = False
    scale_factor: float = 1.0
    original_max_seq_len: int = 2048
    beta_fast: float = 32.0
    beta_slow: float = 1.0
    attn_factor: Optional[float] = None

    def __post_init__(self):
        """Validate configuration and compute derived values."""
        if self.scale_factor < 1.0:
            raise ValueError(f"scale_factor must be >= 1.0, got {self.scale_factor}")

        if self.beta_fast <= self.beta_slow:
            raise ValueError(
                f"beta_fast ({self.beta_fast}) must be > beta_slow ({self.beta_slow})"
            )

        if self.original_max_seq_len <= 0:
            raise ValueError(
                f"original_max_seq_len must be > 0, got {self.original_max_seq_len}"
            )

        # Auto-compute attention factor if not provided
        if self.attn_factor is None and self.enabled:
            if self.scale_factor > 1.0:
                # YaRN paper formula: sqrt(1/t) = 0.1 * ln(s) + 1
                self.attn_factor = 0.1 * math.log(self.scale_factor) + 1.0
            else:
                self.attn_factor = 1.0

    @property
    def effective_max_seq_len(self) -> int:
        """Compute the effective maximum sequence length after scaling."""
        return int(self.original_max_seq_len * self.scale_factor)


# Expected function signatures for implementation

def compute_yarn_inv_freq(
    dim: int,
    base: float = 10000.0,
    scale_factor: float = 1.0,
    beta_fast: float = 32.0,
    beta_slow: float = 1.0,
    original_max_seq_len: int = 2048,
) -> "torch.Tensor":
    """
    Compute YaRN-scaled inverse frequencies using NTK-by-parts interpolation.

    Args:
        dim: Dimension of the RoPE embeddings (typically qk_rope_head_dim).
        base: Base frequency for RoPE computation.
        scale_factor: Context extension ratio.
        beta_fast: High frequency boundary.
        beta_slow: Low frequency boundary.
        original_max_seq_len: Original training context length.

    Returns:
        Tensor of shape [dim // 2] containing scaled inverse frequencies.

    Contract:
        - When scale_factor <= 1.0, returns standard RoPE inverse frequencies
        - When scale_factor > 1.0, applies NTK-by-parts interpolation
        - High frequencies (small wavelengths) are preserved
        - Low frequencies (large wavelengths) are interpolated
    """
    raise NotImplementedError("To be implemented in positional_encoding.py")


def precompute_freqs_cis_yarn(
    dim: int,
    end: int,
    theta: float = 10000.0,
    scale_factor: float = 1.0,
    beta_fast: float = 32.0,
    beta_slow: float = 1.0,
    original_max_seq_len: int = 2048,
) -> "torch.Tensor":
    """
    Precompute the frequency tensor for complex exponentials with YaRN scaling.

    Args:
        dim: Dimension of the RoPE embeddings.
        end: Maximum sequence length to precompute.
        theta: Base frequency (default 10000.0).
        scale_factor: Context extension ratio.
        beta_fast: High frequency boundary.
        beta_slow: Low frequency boundary.
        original_max_seq_len: Original training context length.

    Returns:
        Complex tensor of shape [end, dim // 2] for RoPE application.

    Contract:
        - When scale_factor <= 1.0, equivalent to precompute_freqs_cis()
        - Supports sequences up to scale_factor * original_max_seq_len
    """
    raise NotImplementedError("To be implemented in positional_encoding.py")
