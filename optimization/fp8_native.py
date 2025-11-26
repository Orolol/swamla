"""
Native PyTorch FP8 Implementation for SWA-MLA Model.

This module provides a native FP8 implementation using PyTorch's `float8_e4m3fn` type
and `_scaled_mm` function, targeting H100 GPUs with CUDA >= 12.6.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, Tuple, Union
import math

# Check for native FP8 support
HAS_NATIVE_FP8 = hasattr(torch, "float8_e4m3fn") and hasattr(torch, "_scaled_mm")

class FP8LinearFunction(torch.autograd.Function):
    """
    Autograd function for FP8 Linear layer.
    Handles forward and backward passes using FP8 matrix multiplication.
    """

    @staticmethod
    def forward(ctx, input, weight, bias):
        # Save original shapes and types
        ctx.input_shape = input.shape
        ctx.input_dtype = input.dtype
        ctx.weight_dtype = weight.dtype
        
        # Flatten input: (..., in_features) -> (M, K)
        input_flat = input.view(-1, input.shape[-1])
        
        # 1. Quantize Input
        # Dynamic scaling: scale = max_fp8 / max_val
        # E4M3 max value is 448.0
        input_abs_max = input_flat.abs().max()
        input_scale = torch.tensor(448.0, device=input.device) / (input_abs_max + 1e-12)
        input_fp8 = (input_flat * input_scale).to(torch.float8_e4m3fn)
        
        # 2. Quantize Weight
        weight_abs_max = weight.abs().max()
        weight_scale = torch.tensor(448.0, device=input.device) / (weight_abs_max + 1e-12)
        weight_fp8 = (weight * weight_scale).to(torch.float8_e4m3fn)
        
        # 3. Forward MM: Input @ Weight.T
        # _scaled_mm(A, B) computes A @ B
        # We pass input_fp8 (M, K) and weight_fp8.t() (K, N)
        # weight_fp8 is (N, K) Row-Major.
        # weight_fp8.t() is (K, N) Column-Major.
        # So we have Row-Major @ Column-Major. This is supported by cuBLASLt.
        
        # Scales for dequantization (inverse of quantization scales)
        # Keep as tensors for _scaled_mm (requires Tensor arguments)
        scale_a_inv = 1.0 / input_scale
        scale_b_inv = 1.0 / weight_scale

        output = torch._scaled_mm(
            input_fp8,
            weight_fp8.t(),
            scale_a=scale_a_inv,
            scale_b=scale_b_inv,
            out_dtype=input.dtype,
            bias=bias
        )
        
        # Save for backward
        # We save FP8 tensors to save memory!
        ctx.save_for_backward(input_fp8, weight_fp8, input_scale, weight_scale)
        ctx.bias_requires_grad = bias is not None
        
        return output.view(*ctx.input_shape[:-1], weight.shape[0])

    @staticmethod
    def backward(ctx, grad_output):
        input_fp8, weight_fp8, input_scale, weight_scale = ctx.saved_tensors
        
        grad_output_flat = grad_output.view(-1, grad_output.shape[-1])

        # 1. Quantize Grad Output
        grad_abs_max = grad_output_flat.abs().max()
        grad_scale = torch.tensor(448.0, device=grad_output.device) / (grad_abs_max + 1e-12)
        grad_fp8 = (grad_output_flat * grad_scale).to(torch.float8_e4m3fn)
        
        grad_input = None
        grad_weight = None
        grad_bias = None
        
        # Scales for dequantization
        # Keep as tensors for _scaled_mm (requires Tensor arguments)
        scale_grad_inv = 1.0 / grad_scale
        scale_input_inv = 1.0 / input_scale
        scale_weight_inv = 1.0 / weight_scale
        
        # 2. Compute Grad Input: dY @ W
        # dY: (M, N), W: (N, K) -> (M, K)
        # _scaled_mm(A, B) -> A @ B.T
        # We want dY @ W.
        # We have W_fp8 (N, K).
        # We need B such that B.T = W. So B = W.T.
        # W_fp8.T is (K, N).
        # _scaled_mm(grad_fp8, W_fp8.T) -> grad_fp8 @ (W_fp8.T).T = grad_fp8 @ W_fp8.
        # Correct.
        
        # 2. Compute Grad Input: dY @ W
        # dY: (M, N), W: (N, K) -> (M, K)
        # _scaled_mm(A, B) -> A @ B
        # We want dY @ W.
        # dY is (M, N) Row-Major.
        # W is (N, K) Row-Major.
        # We need B to be Column-Major.
        # W_col = W.t().contiguous().t() -> (N, K) Column-Major.
        
        weight_col = weight_fp8.t().contiguous().t()
        
        grad_input = torch._scaled_mm(
            grad_fp8,
            weight_col,
            scale_a=scale_grad_inv,
            scale_b=scale_weight_inv,
            out_dtype=ctx.input_dtype
        )
        grad_input = grad_input.view(*ctx.input_shape)
        
        # 3. Compute Grad Weight: dY.T @ X
        # dY: (M, N), X: (M, K) -> (N, K)
        # We want dY.T @ X.
        # _scaled_mm(A, B) -> A @ B.T
        # Let A = dY.T (N, M).
        # Let B = X.T (K, M). (Wait, X is (M, K))
        # We want A @ B.T? No.
        # We want (N, M) @ (M, K).
        # _scaled_mm(dY.T, X) -> dY.T @ X.T ? No.
        # _scaled_mm(dY.T, X) -> dY.T @ X.T (if X is interpreted as B)
        # Wait.
        # dW = dY.T @ X
        # dY_fp8.T is (N, M).
        # input_fp8 is (M, K).
        # We want (N, M) @ (M, K).
        # _scaled_mm(A, B) -> A @ B.T
        # We need B such that B.T = input_fp8.
        # So B = input_fp8.T.
        # _scaled_mm(grad_fp8.t(), input_fp8.t()) -> grad_fp8.t() @ (input_fp8.t()).T
        # = grad_fp8.t() @ input_fp8.
        # Correct.
        
        # 3. Compute Grad Weight: dY.T @ X
        # dY: (M, N), X: (M, K) -> (N, K)
        # dW = dY.T @ X
        # _scaled_mm(A, B) -> A @ B
        # We want dY.T @ X.
        
        # dY.T is (N, M) Column-Major.
        # We need A to be Row-Major.
        grad_fp8_t_row = grad_fp8.t().contiguous()
        
        # X is (M, K) Row-Major.
        # We need B to be Column-Major.
        input_fp8_col = input_fp8.t().contiguous().t()
        
        grad_weight = torch._scaled_mm(
            grad_fp8_t_row,
            input_fp8_col,
            scale_a=scale_grad_inv,
            scale_b=scale_input_inv,
            out_dtype=ctx.weight_dtype
        )
        
        if ctx.bias_requires_grad:
            grad_bias = grad_output_flat.sum(0, dtype=ctx.input_dtype)
            
        return grad_input, grad_weight, grad_bias


class FP8Linear(nn.Module):
    """
    Linear layer with native FP8 support using torch._scaled_mm.
    
    Maintains master weights in BF16/FP32 and performs forward pass in FP8.
    Uses dynamic scaling for both inputs and weights.
    """
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True, dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dtype = dtype
        
        # Master weights in high precision (BF16 usually)
        self.weight = nn.Parameter(torch.empty((out_features, in_features), dtype=dtype))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, dtype=dtype))
        else:
            self.register_parameter('bias', None)
            
        # Reset parameters
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass using FP8 matrix multiplication.
        """
        # If native FP8 is not available or disabled, fall back to standard linear
        if not HAS_NATIVE_FP8 or not x.is_cuda:
            return F.linear(x, self.weight, self.bias)

        return FP8LinearFunction.apply(x, self.weight, self.bias)

    @classmethod
    def from_linear(cls, linear: nn.Linear) -> 'FP8Linear':
        """Convert a standard Linear layer to FP8Linear."""
        fp8_layer = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            bias=linear.bias is not None,
            dtype=linear.weight.dtype
        )
        
        fp8_layer.weight.data = linear.weight.data.clone()
        if linear.bias is not None:
            fp8_layer.bias.data = linear.bias.data.clone()
            
        return fp8_layer

def convert_to_native_fp8(model: nn.Module, exclude_modules: list = None) -> nn.Module:
    """
    Convert model Linear layers to FP8Linear.
    
    Args:
        model: PyTorch model to convert
        exclude_modules: List of module names (substrings) to exclude from conversion
        
    Returns:
        Converted model
    """
    if not HAS_NATIVE_FP8:
        print("Warning: Native FP8 not supported on this system. Returning original model.")
        return model
        
    if exclude_modules is None:
        exclude_modules = ['lm_head', 'wte', 'wpe']
        
    print("Converting model to native FP8...")
    
    # Helper to check exclusion
    def should_exclude(name):
        return any(ex in name for ex in exclude_modules)
    
    # Recursive replacement
    for name, module in model.named_children():
        if should_exclude(name):
            print(f"  Skipping {name} (excluded)")
            continue
            
        if isinstance(module, nn.Linear):
            # Check dimensions (must be divisible by 16 for FP8 kernels usually)
            if module.in_features % 16 != 0 or module.out_features % 16 != 0:
                print(f"  Skipping {name} (dimensions {module.in_features}x{module.out_features} not divisible by 16)")
                continue
                
            # Replace
            print(f"  Converting {name} to FP8Linear")
            setattr(model, name, FP8Linear.from_linear(module))
        else:
            # Recurse
            convert_to_native_fp8(module, exclude_modules)
            
    return model
