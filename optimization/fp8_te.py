"""Transformer Engine FP8 integration for SWA-MLA model.

This module provides wrappers for NVIDIA Transformer Engine (TE) FP8 training.
When TE is available and FP8 is enabled, it uses TE's optimized Linear layers
and FP8-aware gradient checkpointing. Otherwise, it falls back to standard PyTorch.

Requirements:
    - NVIDIA GPU with FP8 support (H100/H200 or later)
    - transformer-engine>=1.0
    - CUDA >= 12.0

Usage:
    from optimization.fp8_te import get_te_linear, get_fp8_recipe, te_checkpoint, HAS_TE

    # In model definition
    self.linear = get_te_linear(in_features, out_features, bias=True, use_te_fp8=config.use_te_fp8)

    # In training loop
    fp8_recipe = get_fp8_recipe()
    with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
        output = model(input)
"""

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint

# Check Transformer Engine availability
HAS_TE = False
te = None
Format = None
DelayedScaling = None

try:
    import transformer_engine.pytorch as te
    from transformer_engine.common.recipe import Format, DelayedScaling
    HAS_TE = True
except ImportError:
    pass


def is_fp8_compatible(in_features: int, out_features: int) -> bool:
    """Check if dimensions are compatible with TE FP8.

    TE Linear requires both dimensions to be divisible by 16 for FP8 operations.

    Args:
        in_features: Input dimension
        out_features: Output dimension

    Returns:
        True if both dimensions are divisible by 16
    """
    return in_features % 16 == 0 and out_features % 16 == 0


def get_te_linear(
    in_features: int,
    out_features: int,
    bias: bool = True,
    use_te_fp8: bool = False
) -> nn.Module:
    """Get a Linear layer - TE Linear if FP8 enabled and compatible, else nn.Linear.

    This function provides a unified interface for creating linear layers that
    automatically uses Transformer Engine's FP8-optimized Linear when possible.

    Args:
        in_features: Size of each input sample
        out_features: Size of each output sample
        bias: If True, adds a learnable bias to the output
        use_te_fp8: If True, attempt to use TE Linear for FP8

    Returns:
        nn.Module: Either te.Linear or nn.Linear depending on availability and compatibility

    Note:
        Falls back to nn.Linear if:
        - TE is not installed
        - use_te_fp8 is False
        - Dimensions are not divisible by 16
    """
    if use_te_fp8 and HAS_TE and is_fp8_compatible(in_features, out_features):
        return te.Linear(in_features, out_features, bias=bias)
    return nn.Linear(in_features, out_features, bias=bias)


def get_fp8_recipe(
    amax_history_len: int = 16,
    amax_compute_algo: str = "max",
    fp8_format: str = "hybrid"
):
    """Get FP8 recipe for training.

    Creates a DelayedScaling recipe for FP8 training. The recipe controls
    how scaling factors are computed and updated during training.

    Args:
        amax_history_len: Number of iterations to track amax history for scaling
        amax_compute_algo: Algorithm for computing amax ("max" or "most_recent")
        fp8_format: FP8 format - "hybrid" (E4M3 forward, E5M2 backward) or "e4m3"

    Returns:
        DelayedScaling recipe if TE available, None otherwise

    Note:
        HYBRID format is recommended as it provides:
        - E4M3 for forward pass: more precision (8 mantissa levels)
        - E5M2 for backward pass: more dynamic range (32 exponent levels)
    """
    if not HAS_TE:
        return None

    format_map = {
        "hybrid": Format.HYBRID,
        "e4m3": Format.E4M3,
    }
    fp8_fmt = format_map.get(fp8_format.lower(), Format.HYBRID)

    return DelayedScaling(
        fp8_format=fp8_fmt,
        amax_history_len=amax_history_len,
        amax_compute_algo=amax_compute_algo,
    )


def te_checkpoint(fn, *args, use_te_fp8: bool = False, **kwargs):
    """FP8-aware gradient checkpointing wrapper.

    When using FP8 training, standard gradient checkpointing may not properly
    handle FP8 states during recomputation. This wrapper uses TE's checkpoint
    function when FP8 is enabled, which properly preserves FP8 scaling factors.

    Args:
        fn: Function to checkpoint
        *args: Arguments to pass to the function
        use_te_fp8: If True, use TE checkpoint for FP8-aware checkpointing
        **kwargs: Keyword arguments to pass to the function

    Returns:
        Output of fn(*args, **kwargs)

    Note:
        The 'use_te_fp8' kwarg is consumed by this wrapper and not passed to fn.
    """
    # Remove use_te_fp8 from kwargs if present (it's for this wrapper, not fn)
    kwargs.pop('use_te_fp8', None)

    if use_te_fp8 and HAS_TE and hasattr(te, 'checkpoint'):
        # TE checkpoint handles FP8 states properly
        return te.checkpoint(fn, *args, **kwargs)

    # Fall back to standard PyTorch checkpointing
    return checkpoint.checkpoint(fn, *args, use_reentrant=False, **kwargs)


def check_fp8_support(verbose: bool = True) -> bool:
    """Check if the current environment supports FP8 training.

    Verifies:
    1. Transformer Engine is installed
    2. CUDA is available
    3. GPU supports FP8 (compute capability >= 9.0 for native FP8)

    Args:
        verbose: If True, print diagnostic information

    Returns:
        True if FP8 training is supported
    """
    if not HAS_TE:
        if verbose:
            print("FP8 not supported: transformer-engine not installed")
            print("  Install with: pip install transformer-engine")
        return False

    if not torch.cuda.is_available():
        if verbose:
            print("FP8 not supported: CUDA not available")
        return False

    # Check compute capability (H100 = 9.0, H200 = 9.0)
    capability = torch.cuda.get_device_capability()
    compute_cap = capability[0] + capability[1] / 10

    if compute_cap < 8.9:
        if verbose:
            gpu_name = torch.cuda.get_device_name(0)
            print(f"FP8 warning: GPU {gpu_name} (compute {compute_cap}) may not fully support FP8")
            print("  FP8 works best on H100/H200 (compute 9.0+)")
        # Still return True as TE may emulate FP8 on older GPUs

    if verbose:
        gpu_name = torch.cuda.get_device_name(0)
        print(f"FP8 support: transformer-engine={HAS_TE}, GPU={gpu_name} (compute {compute_cap})")

    return True
