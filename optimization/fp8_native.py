"""Native PyTorch FP8 training via torchao.float8.

This module provides FP8 training using torchao's Float8Linear, which converts
nn.Linear layers post-construction. This is simpler than Transformer Engine's
approach and doesn't require the heavy TE C++ build dependency.

Requirements:
    - NVIDIA GPU with FP8 support (H100/H200 or later)
    - torchao >= 0.5.0
    - torch.compile recommended for performance

Usage:
    from optimization.fp8_native import convert_model_to_fp8, HAS_NATIVE_FP8

    # After model creation
    if HAS_NATIVE_FP8:
        convert_model_to_fp8(model)
        model = torch.compile(model)  # recommended for perf
"""

import torch
import torch.nn as nn

# Check torchao float8 availability
HAS_NATIVE_FP8 = False
_convert_to_float8_training = None
_Float8LinearConfig = None

try:
    from torchao.float8 import convert_to_float8_training, Float8LinearConfig
    _convert_to_float8_training = convert_to_float8_training
    _Float8LinearConfig = Float8LinearConfig
    HAS_NATIVE_FP8 = True
except ImportError:
    pass


def _is_fp8_compatible(mod: nn.Module) -> bool:
    """Check if a Linear module has dimensions compatible with FP8.

    Both in_features and out_features must be divisible by 16.
    """
    if not isinstance(mod, nn.Linear):
        return False
    return mod.in_features % 16 == 0 and mod.out_features % 16 == 0


def _default_module_filter(mod: nn.Module, fqn: str) -> bool:
    """Default filter: convert Linear layers with compatible dims, skip GatedDeltaNet blocks.

    Args:
        mod: The module to check.
        fqn: Fully qualified name of the module in the model.

    Returns:
        True if this module should be converted to Float8Linear.
    """
    if not isinstance(mod, nn.Linear):
        return False
    # Skip GatedDeltaNet blocks (same behavior as TE)
    if 'gated_deltanet' in fqn or 'deltanet' in fqn.lower():
        return False
    # Skip modules with incompatible dimensions
    if not _is_fp8_compatible(mod):
        return False
    return True


def convert_model_to_fp8(
    model: nn.Module,
    module_filter_fn=None,
    config=None,
) -> nn.Module:
    """Convert model's nn.Linear layers to Float8Linear for FP8 training.

    This is a post-construction conversion — call after model creation and
    before torch.compile.

    Args:
        model: The model to convert.
        module_filter_fn: Optional function (mod, fqn) -> bool to control which
            modules are converted. Defaults to _default_module_filter which
            skips GatedDeltaNet blocks and incompatible dimensions.
        config: Optional Float8LinearConfig for fine-tuning FP8 behavior.

    Returns:
        The model with Float8Linear layers (modified in-place).

    Raises:
        RuntimeError: If torchao.float8 is not available.
    """
    if not HAS_NATIVE_FP8:
        raise RuntimeError(
            "Native FP8 requires torchao. Install with: pip install torchao>=0.5.0"
        )

    if module_filter_fn is None:
        module_filter_fn = _default_module_filter

    kwargs = {}
    if config is not None:
        kwargs['config'] = config
    elif _Float8LinearConfig is not None:
        # Use "tensorwise" recipe for best throughput (matches TE's per-tensor scaling)
        # "rowwise" is more accurate but slower; "tensorwise" closes the gap with TE
        try:
            kwargs['config'] = _Float8LinearConfig.from_recipe_name("tensorwise")
        except (AttributeError, TypeError):
            # Fallback for older torchao versions without from_recipe_name
            kwargs['config'] = _Float8LinearConfig()

    _convert_to_float8_training(
        model,
        module_filter_fn=module_filter_fn,
        **kwargs,
    )

    return model


def check_native_fp8_support(verbose: bool = True) -> bool:
    """Check if the current environment supports native FP8 training.

    Verifies:
    1. torchao.float8 is installed
    2. CUDA is available
    3. GPU supports FP8 (compute capability >= 8.9)

    Args:
        verbose: If True, print diagnostic information.

    Returns:
        True if native FP8 training is supported.
    """
    if not HAS_NATIVE_FP8:
        if verbose:
            print("Native FP8 not supported: torchao not installed")
            print("  Install with: pip install torchao>=0.5.0")
        return False

    if not torch.cuda.is_available():
        if verbose:
            print("Native FP8 not supported: CUDA not available")
        return False

    capability = torch.cuda.get_device_capability()
    compute_cap = capability[0] + capability[1] / 10

    if compute_cap < 8.9:
        if verbose:
            gpu_name = torch.cuda.get_device_name(0)
            print(f"Native FP8 warning: GPU {gpu_name} (compute {compute_cap}) may not fully support FP8")
            print("  FP8 works best on H100/H200 (compute 9.0+)")

    if verbose:
        gpu_name = torch.cuda.get_device_name(0)
        print(f"Native FP8 support: torchao={HAS_NATIVE_FP8}, GPU={gpu_name} (compute {compute_cap})")

    return True
