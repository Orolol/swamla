"""Maximal Update Parametrization (μP) for width-independent hyperparameters."""

import math
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
from enum import Enum

import torch
import torch.nn as nn


class LayerType(Enum):
    """Classification of layer types for μP scaling."""
    EMBEDDING = "embedding"      # wte, wpe: 1/√d init, 1x LR, no decay
    ATTENTION = "attention"      # Q,K,V,O: 1/√d_in init, 1/width LR
    MLP_IN = "mlp_in"           # gate/up proj: 1/√d_in init, 1/width LR
    MLP_OUT = "mlp_out"         # down proj: 1/√(width*d_in) init, 1/width LR
    LM_HEAD = "lm_head"         # output: zero init, 1/width LR, no decay
    NORM = "norm"               # RMSNorm, LayerNorm: skip (no scaling)
    ENGRAM_EMBED = "engram_embed"  # Engram tables: 1/√d_mem, 5x LR, no decay
    OTHER = "other"             # Biases, etc: no scaling


@dataclass
class MuPConfig:
    """Configuration for μP scaling."""
    base_width: int = 256       # Reference width for scaling
    output_mult: float = 1.0    # Output logits multiplier (temperature)
    width: int = 1024           # Actual model width (n_embd)

    @property
    def width_mult(self) -> float:
        """Width multiplier: actual_width / base_width."""
        return self.width / self.base_width


def classify_layer(name: str, param: torch.Tensor) -> LayerType:
    """
    Classify a parameter by its role in the model.

    Args:
        name: Parameter name (e.g., 'transformer.h.0.attn.q_proj.weight')
        param: The parameter tensor

    Returns:
        LayerType classification
    """
    name_lower = name.lower()

    # Embeddings
    if any(x in name_lower for x in ['wte', 'wpe', 'token_emb', 'pos_emb']):
        return LayerType.EMBEDDING

    # LM Head (output projection)
    if 'lm_head' in name_lower or name_lower.endswith('head.weight'):
        return LayerType.LM_HEAD

    # Engram embeddings (special: 5x LR)
    if 'engram' in name_lower and 'embed' in name_lower and 'table' in name_lower:
        return LayerType.ENGRAM_EMBED

    # Normalization layers (skip)
    if any(x in name_lower for x in ['norm', 'ln_', 'layernorm', 'rmsnorm']):
        return LayerType.NORM

    # Attention projections
    if any(x in name_lower for x in ['q_proj', 'k_proj', 'v_proj', 'qkv', 'w_q', 'w_k', 'w_v',
                                      'q_a_proj', 'q_b_proj', 'kv_a_proj', 'kv_b_proj',  # MLA
                                      'o_proj', 'out_proj', 'w_o']):
        return LayerType.ATTENTION

    # MLP layers
    if any(x in name_lower for x in ['mlp', 'ffn', 'feed_forward']):
        if any(x in name_lower for x in ['down', 'out', 'fc2', 'w2']):
            return LayerType.MLP_OUT
        elif any(x in name_lower for x in ['up', 'gate', 'in', 'fc1', 'w1', 'w3']):
            return LayerType.MLP_IN
        # Default MLP to MLP_IN (more common)
        return LayerType.MLP_IN

    # MoE expert layers (treat as MLP)
    if 'expert' in name_lower:
        if 'down' in name_lower:
            return LayerType.MLP_OUT
        return LayerType.MLP_IN

    # Default: OTHER (biases, etc.)
    return LayerType.OTHER


def get_mup_lr_scale(layer_type: LayerType, config: MuPConfig, engram_lr_mult: float = 5.0) -> float:
    """
    Get the learning rate scale factor for a layer type.

    Args:
        layer_type: The layer classification
        config: μP configuration
        engram_lr_mult: LR multiplier for Engram embeddings

    Returns:
        LR scale factor (multiply base_lr by this)
    """
    width_mult = config.width_mult

    if layer_type == LayerType.EMBEDDING:
        return 1.0  # Embeddings use base LR
    elif layer_type == LayerType.LM_HEAD:
        return 1.0 / width_mult  # Scale down with width
    elif layer_type == LayerType.ATTENTION:
        return 1.0 / width_mult
    elif layer_type == LayerType.MLP_IN:
        return 1.0 / width_mult
    elif layer_type == LayerType.MLP_OUT:
        return 1.0 / width_mult
    elif layer_type == LayerType.ENGRAM_EMBED:
        return engram_lr_mult  # Keep existing 5x multiplier
    elif layer_type == LayerType.NORM:
        return 1.0  # Norms use base LR
    else:
        return 1.0  # Default: base LR


def get_mup_weight_decay(layer_type: LayerType, base_wd: float) -> float:
    """Get weight decay for a layer type."""
    if layer_type in (LayerType.EMBEDDING, LayerType.LM_HEAD, LayerType.ENGRAM_EMBED, LayerType.NORM):
        return 0.0
    return base_wd


def mup_init_weight(param: torch.Tensor, layer_type: LayerType, config: MuPConfig) -> None:
    """
    Apply μP-compliant initialization to a weight tensor in-place.

    Args:
        param: Weight tensor to initialize
        layer_type: Classification of the layer
        config: μP configuration
    """
    width = config.width
    width_mult = config.width_mult

    if layer_type == LayerType.EMBEDDING:
        # Embeddings: standard 1/√d
        std = 1.0 / math.sqrt(width)
        nn.init.normal_(param, mean=0.0, std=std)

    elif layer_type == LayerType.LM_HEAD:
        # LM Head: zero init (critical for μP!)
        nn.init.zeros_(param)

    elif layer_type == LayerType.ATTENTION:
        # Attention: 1/√d_in
        fan_in = param.shape[1] if param.dim() >= 2 else param.shape[0]
        std = 1.0 / math.sqrt(fan_in)
        nn.init.normal_(param, mean=0.0, std=std)

    elif layer_type == LayerType.MLP_IN:
        # MLP input: 1/√d_in
        fan_in = param.shape[1] if param.dim() >= 2 else param.shape[0]
        std = 1.0 / math.sqrt(fan_in)
        nn.init.normal_(param, mean=0.0, std=std)

    elif layer_type == LayerType.MLP_OUT:
        # MLP output: 1/√(width * d_in) - scaled down by width
        fan_in = param.shape[1] if param.dim() >= 2 else param.shape[0]
        std = 1.0 / math.sqrt(width_mult * fan_in)
        nn.init.normal_(param, mean=0.0, std=std)

    elif layer_type == LayerType.ENGRAM_EMBED:
        # Engram: preserve existing init (1/√d_mem handled in engram.py)
        pass

    # NORM, OTHER: keep existing init


def mup_init(model: nn.Module, config: MuPConfig) -> None:
    """
    Apply μP-compliant initialization to all model parameters.

    This should be called AFTER model creation but BEFORE optimizer creation.

    Args:
        model: The model to initialize
        config: μP configuration
    """
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        layer_type = classify_layer(name, param)

        # Only reinitialize weight tensors, not biases
        if 'bias' not in name.lower() and param.dim() >= 2:
            mup_init_weight(param, layer_type, config)


def mup_scale_output(logits: torch.Tensor, config: MuPConfig) -> torch.Tensor:
    """
    Scale output logits for μP.

    In μP, the output logits are scaled by output_mult / width_mult
    to maintain consistent gradient magnitudes.

    Args:
        logits: Output logits from lm_head
        config: μP configuration

    Returns:
        Scaled logits
    """
    scale = config.output_mult / config.width_mult
    return logits * scale
