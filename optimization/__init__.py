"""Optimization utilities for SWA-MLA model training."""

from .fp8_te import (
    HAS_TE,
    get_te_linear,
    get_fp8_recipe,
    te_checkpoint,
    is_fp8_compatible,
    check_fp8_support,
)

__all__ = [
    "HAS_TE",
    "get_te_linear",
    "get_fp8_recipe",
    "te_checkpoint",
    "is_fp8_compatible",
    "check_fp8_support",
]
