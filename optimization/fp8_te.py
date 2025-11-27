"""
Transformer Engine FP8 Integration for SWA-MLA Model.

This module provides FP8 training using NVIDIA's Transformer Engine library,
which offers production-ready FP8 with automatic scaling, torch.compile compatibility,
and proven performance at scale.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Type

# Check for Transformer Engine availability
try:
    import transformer_engine.pytorch as te
    from transformer_engine.common.recipe import DelayedScaling, Format
    HAS_TE = True
    # Check for quantized_model_init (newer API) or fp8_model_init (older API)
    if hasattr(te, 'quantized_model_init'):
        te_model_init = te.quantized_model_init
    elif hasattr(te, 'fp8_model_init'):
        te_model_init = te.fp8_model_init
    else:
        te_model_init = None
except ImportError:
    HAS_TE = False
    te = None
    te_model_init = None


def get_fp8_init_context(enabled: bool = True, preserve_high_precision: bool = False):
    """
    Get the appropriate FP8 model initialization context manager.

    Note: fp8_model_init/quantized_model_init does NOT accept a recipe parameter.
    The recipe is only used in fp8_autocast. When using this init context,
    fp8_autocast should NOT specify a recipe (use defaults) to avoid mismatch.

    Args:
        enabled: Whether FP8 initialization is enabled
        preserve_high_precision: Whether to keep high precision copies of weights.
                                 Set to False to minimize memory usage.

    Returns:
        Context manager for FP8 model initialization
    """
    if enabled and HAS_TE and te_model_init is not None:
        return te_model_init(enabled=True, preserve_high_precision_init_val=preserve_high_precision)
    else:
        from contextlib import nullcontext
        return nullcontext()


def get_te_linear(in_features: int, out_features: int, bias: bool = True, use_te_fp8: bool = False) -> nn.Module:
    """
    Get appropriate Linear layer class based on FP8 configuration and dimension compatibility.

    Args:
        in_features: Input dimension
        out_features: Output dimension
        bias: Whether to include bias
        use_te_fp8: Whether to use TE FP8 Linear

    Returns:
        nn.Linear or te.Linear instance
    """
    # Check if TE FP8 should be used and dimensions are compatible
    if use_te_fp8 and HAS_TE:
        # TE requires dimensions divisible by 16
        if in_features % 16 == 0 and out_features % 16 == 0:
            return te.Linear(in_features, out_features, bias=bias)

    # Fallback to standard nn.Linear
    return nn.Linear(in_features, out_features, bias=bias)


def is_te_compatible(in_features: int, out_features: int) -> bool:
    """Check if dimensions are compatible with TE FP8 (divisible by 16)."""
    return in_features % 16 == 0 and out_features % 16 == 0


class PaddedTELinear(nn.Module):
    """
    Wrapper around TE Linear that pads inputs to meet FP8 dimension requirements.
    
    TE requires:
    - Product of all dims except last divisible by 8 (Batch dimension)
    - Last dim divisible by 16 (Feature dimension)
    """

    def __init__(self, te_linear: te.Linear):
        super().__init__()
        self.te_linear = te_linear
        self.in_features = te_linear.in_features
        self.out_features = te_linear.out_features

    @property
    def weight(self):
        """Expose weight from underlying TE layer."""
        return self.te_linear.weight

    @property
    def bias(self):
        """Expose bias from underlying TE layer."""
        return self.te_linear.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward with automatic padding using F.pad."""
        # 1. Capture original shape and flatten to 2D [Batch, InFeatures]
        orig_shape = x.shape
        # Ensure input is contiguous before view
        x = x.contiguous()
        x_flat = x.view(-1, self.in_features)
        
        M, K = x_flat.shape
        
        # 2. Calculate padding requirements
        # TE requires M % 8 == 0 and K % 16 == 0
        pad_M = (8 - (M % 8)) % 8
        pad_K = (16 - (K % 16)) % 16
        
        # 3. Pad if necessary
        if pad_M > 0 or pad_K > 0:
            # We use F.pad: (pad_left, pad_right, pad_top, pad_bottom)
            # x_flat is [M, K]
            # We want to pad dim 1 (K) by pad_K, and dim 0 (M) by pad_M
            x_padded = F.pad(x_flat, (0, pad_K, 0, pad_M))
            
            # 4. Forward pass through TE
            # TE outputs [M_padded, OutFeatures]
            out_padded = self.te_linear(x_padded)
            
            # 5. Unpad
            # Slice back to original M
            # Note: We don't need to slice the feature dim because output feature dim 
            # is determined by the layer weights, not the input padding
            out = out_padded[:M, :]
        else:
            # No padding needed
            out = self.te_linear(x_flat)
            
        # 6. Reshape to original batch dimensions
        # Output shape should be [*orig_batch_dims, out_features]
        return out.view(*orig_shape[:-1], self.out_features)


def convert_to_te_fp8(model: nn.Module, exclude_modules: list = None) -> nn.Module:
    """
    Convert model Linear layers to Transformer Engine FP8Linear.

    Args:
        model: PyTorch model to convert
        exclude_modules: List of module names (substrings) to exclude from conversion

    Returns:
        Converted model with TE FP8 layers
    """
    if not HAS_TE:
        print("Warning: Transformer Engine not installed. Install with:")
        print("  pip install transformer-engine[pytorch]")
        return model

    if exclude_modules is None:
        exclude_modules = ['lm_head', 'wte', 'wpe']

    print("Converting model to Transformer Engine FP8...")

    # Helper to check exclusion
    def should_exclude(name):
        return any(ex in name for ex in exclude_modules)

    # Recursive replacement
    for name, module in model.named_children():
        if should_exclude(name):
            print(f"  Skipping {name} (excluded)")
            continue

        if isinstance(module, nn.Linear):
            # Check if dimensions are compatible with FP8 (must be divisible by 16 for weights)
            # If not, we skip conversion
            if module.in_features % 16 != 0 or module.out_features % 16 != 0:
                print(f"  Skipping {name} (dimensions {module.in_features}x{module.out_features} not divisible by 16)")
                continue

            # Convert to TE Linear directly (no wrapper for memory efficiency)
            print(f"  Converting {name} to TE FP8Linear")

            te_layer = te.Linear(
                in_features=module.in_features,
                out_features=module.out_features,
                bias=module.bias is not None,
                params_dtype=module.weight.dtype,
            )

            # Copy weights
            te_layer.weight.data = module.weight.data.clone()
            if module.bias is not None:
                te_layer.bias.data = module.bias.data.clone()

            # Use TE Linear directly without wrapper
            setattr(model, name, te_layer)
        else:
            # Recurse
            convert_to_te_fp8(module, exclude_modules)

    return model


def create_fp8_recipe(
    fp8_format: str = "HYBRID",
    margin: int = 0,
    interval: int = 1,
    amax_history_len: int = 1024,
    amax_compute_algo: str = "max"
) -> Optional[DelayedScaling]:
    """
    Create FP8 recipe for Transformer Engine.

    Args:
        fp8_format: FP8 format - "HYBRID" (E4M3 forward, E5M2 backward) or "E4M3"
        margin: Margin for scaling factor computation (default: 0 for no margin)
        interval: Interval for updating scaling factors (default: 1 = every iteration)
        amax_history_len: History length for amax calculation
        amax_compute_algo: Algorithm for amax computation ("max" or "most_recent")

    Returns:
        DelayedScaling recipe or None if TE not available
    """
    if not HAS_TE:
        return None

    # Choose format
    if fp8_format == "HYBRID":
        fmt = Format.HYBRID
    elif fp8_format == "E4M3":
        fmt = Format.E4M3
    else:
        raise ValueError(f"Unknown fp8_format: {fp8_format}")

    recipe = DelayedScaling(
        fp8_format=fmt,
        margin=margin,
        interval=interval,
        amax_history_len=amax_history_len,
        amax_compute_algo=amax_compute_algo,
    )

    return recipe


class FP8ContextManager:
    """
    Context manager for FP8 training with Transformer Engine.

    Usage:
        fp8_recipe = create_fp8_recipe()
        fp8_ctx = FP8ContextManager(fp8_recipe, enabled=True)

        with fp8_ctx:
            output = model(input)
    """

    def __init__(self, recipe: Optional[DelayedScaling], enabled: bool = True):
        """
        Initialize FP8 context manager.

        Args:
            recipe: FP8 recipe from create_fp8_recipe()
            enabled: Whether FP8 is enabled
        """
        self.recipe = recipe
        self.enabled = enabled and HAS_TE and recipe is not None

    def __enter__(self):
        if self.enabled:
            return te.fp8_autocast(enabled=True, fp8_recipe=self.recipe)
        else:
            # Return a dummy context that does nothing
            from contextlib import nullcontext
            return nullcontext()

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


def get_fp8_stats(model: nn.Module) -> dict:
    """
    Get FP8 statistics from a model with TE layers.

    Args:
        model: Model with TE FP8 layers

    Returns:
        Dictionary with FP8 statistics (scaling factors, amax, etc.)
    """
    if not HAS_TE:
        return {}

    stats = {
        'te_linear_count': 0,
        'fp8_enabled_layers': [],
    }

    for name, module in model.named_modules():
        if isinstance(module, te.Linear):
            stats['te_linear_count'] += 1
            stats['fp8_enabled_layers'].append(name)

    return stats


def te_checkpoint(function, *args, use_te_fp8: bool = False, **kwargs):
    """
    FP8-aware gradient checkpointing wrapper.

    Uses Transformer Engine's checkpoint when FP8 is enabled (handles FP8 states like
    RNG states, amax history properly), otherwise falls back to torch.utils.checkpoint.

    This is critical for memory efficiency with FP8: standard torch checkpoint doesn't
    handle TE's internal FP8 states, leading to higher memory usage.

    Args:
        function: The function to checkpoint
        *args: Arguments to pass to the function
        use_te_fp8: Whether TE FP8 is being used
        **kwargs: Additional keyword arguments

    Returns:
        Output of the function
    """
    if use_te_fp8 and HAS_TE:
        # Use TE checkpoint which is FP8-aware
        return te.checkpoint(function, *args, **kwargs)
    else:
        # Fall back to standard PyTorch checkpoint
        import torch.utils.checkpoint as torch_checkpoint
        return torch_checkpoint.checkpoint(function, *args, use_reentrant=False, **kwargs)
