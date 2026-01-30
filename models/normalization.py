"""Normalization layers for transformer models."""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Try to import Triton RMSNorm
try:
    from triton_kernels import TritonRMSNorm, fused_rms_norm
    TRITON_RMSNORM_AVAILABLE = True
except ImportError:
    TRITON_RMSNORM_AVAILABLE = False
    TritonRMSNorm = None
    fused_rms_norm = None


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization"""
    
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # For bf16/fp16 training: compute norm in-dtype to avoid fp32 round-trip copies
        # torch.rsqrt is numerically stable enough in bf16 for LLM training
        if x.dtype in (torch.bfloat16, torch.float16):
            norm_x = self._norm(x)
            return self.weight * norm_x
        # For fp32 or exotic dtypes (e.g. Float8): compute in fp32 for stability
        input_dtype = x.dtype
        norm_x = self._norm(x.float())
        if input_dtype != torch.float32:
            norm_x = norm_x.to(input_dtype)
        return self.weight * norm_x


class DynamicTanh(nn.Module):
    """Dynamic Tanh (DyT) - A simple replacement for normalization layers.
    
    Based on the paper "Transformers without Normalization" by Meta FAIR.
    DyT(x) = tanh(α * x) * weight + bias
    
    This reduces training time by ~8.2% and inference by ~7.8% compared to RMSNorm/LayerNorm
    while maintaining equivalent performance.
    """
    
    def __init__(self, dim: int, alpha_init: float = 0.5, elementwise_affine: bool = True):
        """Initialize DynamicTanh.
        
        Args:
            dim: Dimension of the input features
            alpha_init: Initial value for the alpha parameter (default: 0.5)
            elementwise_affine: Whether to include learnable weight and bias (default: True)
        """
        super().__init__()
        # Alpha is a learnable scalar parameter that controls the scaling
        self.alpha = nn.Parameter(torch.tensor(alpha_init))
        
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            # Learnable per-feature weight and bias
            self.weight = nn.Parameter(torch.ones(dim))
            self.bias = nn.Parameter(torch.zeros(dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply Dynamic Tanh normalization.
        
        Args:
            x: Input tensor of shape (..., dim)
            
        Returns:
            Normalized tensor of the same shape
        """
        # Apply tanh with learnable alpha scaling
        x = torch.tanh(self.alpha * x)
        
        # Apply elementwise affine transformation if enabled
        if self.elementwise_affine:
            x = x * self.weight + self.bias
            
        return x
    
    def extra_repr(self) -> str:
        return f'alpha={self.alpha.item():.3f}, elementwise_affine={self.elementwise_affine}'


def create_norm_layer(dim: int, norm_type: str = 'rmsnorm', use_triton: bool = True, **kwargs) -> nn.Module:
    """
    Factory function to create normalization layers.

    Args:
        dim: Hidden dimension
        norm_type: Type of normalization ('rmsnorm', 'dyt', 'layernorm')
        use_triton: Whether to use Triton kernel when available
        **kwargs: Additional arguments passed to the norm layer

    Returns:
        Normalization module
    """
    if norm_type == 'dyt':
        return DynamicTanh(dim, **kwargs)
    elif norm_type == 'layernorm':
        return nn.LayerNorm(dim, **kwargs)
    else:  # Default: rmsnorm
        if use_triton and TRITON_RMSNORM_AVAILABLE and TritonRMSNorm is not None:
            return TritonRMSNorm(dim, **kwargs)
        return RMSNorm(dim, **kwargs)