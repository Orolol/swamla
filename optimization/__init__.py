"""Optimization utilities for SWA-MLA model training."""

from .fp8_te import (
    HAS_TE,
    get_te_linear,
    get_fp8_recipe,
    te_checkpoint,
    is_fp8_compatible,
    check_fp8_support,
)

# Native PyTorch FP8 via torchao
from .fp8_native import (
    HAS_NATIVE_FP8,
    convert_model_to_fp8,
    check_native_fp8_support,
)

# μP (Maximal Update Parametrization)
from .mup import (
    MuPConfig,
    LayerType,
    classify_layer,
    mup_init,
    configure_mup_optimizer,
    get_mup_lr_scale,
    get_mup_weight_decay,
    mup_scale_output,
)

# Progressive Training
from .progressive import ProgressiveScheduler, ProgressivePhase

# EMA (Stochastic Weight Averaging)
from .swa import EMAModel

__all__ = [
    # FP8
    "HAS_TE",
    "get_te_linear",
    "get_fp8_recipe",
    "te_checkpoint",
    "is_fp8_compatible",
    "check_fp8_support",
    # Native FP8
    "HAS_NATIVE_FP8",
    "convert_model_to_fp8",
    "check_native_fp8_support",
    # μP
    "MuPConfig",
    "LayerType",
    "classify_layer",
    "mup_init",
    "configure_mup_optimizer",
    "get_mup_lr_scale",
    "get_mup_weight_decay",
    "mup_scale_output",
    # Progressive
    "ProgressiveScheduler",
    "ProgressivePhase",
    # EMA
    "EMAModel",
]
