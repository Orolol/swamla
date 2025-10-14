"""Normalization layers for transformer models."""

import torch
import torch.nn as nn
import torch.nn.functional as F

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization"""
    
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Handle Float8 or other exotic types that might cause issues
        try:
            # Always calculate norm in fp32 for stability, then convert back
            norm_x = self._norm(x.float())
            # Safely convert back to original type
            if x.dtype in [torch.float32, torch.float16, torch.bfloat16]:
                norm_x = norm_x.to(x.dtype)
            return self.weight * norm_x
        except Exception as e:
            # Fallback to a safer implementation if there are dtype issues
            print(f"Warning: Using fallback normalization due to: {e}")
            working_type = torch.float32
            x_float = x.to(working_type)
            norm_x = self._norm(x_float)
            result = self.weight.to(working_type) * norm_x
            return result


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