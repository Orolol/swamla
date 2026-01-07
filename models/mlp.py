"""MLP components for transformer models."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

from normalization import RMSNorm

# Try to import fused Triton kernels
try:
    from triton_kernels import fused_swiglu
    TRITON_SWIGLU_AVAILABLE = True
except ImportError:
    TRITON_SWIGLU_AVAILABLE = False
    fused_swiglu = None


class MLP(nn.Module):
    """
    Multi-Layer Perceptron with SwiGLU activation.

    Supports fused Triton kernel for ~15-25% speedup when available.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config

        # Default hidden dimension: 4 * n_embd
        hidden_dim = 4 * config.n_embd

        # Combined gate+up projection for efficiency
        self.gate_up_proj = nn.Linear(config.n_embd, 2 * hidden_dim, bias=config.bias)
        self.down_proj = nn.Linear(hidden_dim, config.n_embd, bias=config.bias)

        self.dropout = nn.Dropout(config.dropout)

        # Use fused Triton kernel when available (significant speedup)
        self.use_triton = getattr(config, 'use_triton_kernels', True) and TRITON_SWIGLU_AVAILABLE

        # Activation function setup (fallback when Triton not available)
        self.act_fn = self._get_optimized_activation()

        # Gradient checkpointing control
        self.use_gradient_checkpointing = getattr(config, 'use_gradient_checkpointing', True)

        # Initialize weights with a special scale for better gradient flow
        self._init_weights()

    def _init_weights(self):
        """Initialize weights for better gradient flow."""
        scale = 2 / (self.config.n_embd ** 0.5)

        nn.init.normal_(self.gate_up_proj.weight, mean=0.0, std=scale)
        if self.gate_up_proj.bias is not None:
            nn.init.zeros_(self.gate_up_proj.bias)

        nn.init.normal_(self.down_proj.weight, mean=0.0, std=scale)
        if self.down_proj.bias is not None:
            nn.init.zeros_(self.down_proj.bias)

    def _get_optimized_activation(self):
        """
        Retourne la fonction d'activation optimisée selon le device et les capacités
        """
        try:
            # Vérifier si on peut utiliser une implémentation CUDA optimisée
            if torch.cuda.is_available() and hasattr(torch.nn.functional, 'silu'):
                def optimized_swiglu(x):
                    x1, x2 = x.chunk(2, dim=-1)
                    return F.silu(x2) * x1
                return optimized_swiglu
        except:
            pass
        
        # Fallback sur l'implémentation standard
        def standard_swiglu(x):
            x1, x2 = x.chunk(2, dim=-1)
            return F.silu(x2) * x1
        return standard_swiglu

    def _fuse_operations(self, x: torch.Tensor) -> torch.Tensor:
        """
        Fused operations for better efficiency.

        Uses Triton kernel when available for ~15-25% speedup.
        """
        # Remember input dtype for output conversion
        input_dtype = x.dtype

        # Combined gate+up projection
        combined = self.gate_up_proj(x)

        # Apply activation: use fused Triton kernel when available
        if self.use_triton and fused_swiglu is not None:
            # Fused Triton kernel: chunk + silu + mul in one kernel
            hidden = fused_swiglu(combined)
        else:
            # Fallback to standard implementation
            hidden = self.act_fn(combined)

        # Down projection
        output = self.down_proj(hidden)

        # Handle FP8 conversion: ensure output matches input dtype for residual connections
        if output.dtype in [torch.float8_e4m3fn, torch.float8_e5m2] and input_dtype not in [torch.float8_e4m3fn, torch.float8_e5m2]:
            output = output.to(input_dtype)

        # Apply dropout
        return self.dropout(output)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass avec gestion optimisée de la mémoire
        """
        if torch.jit.is_scripting() or not self.training:
            # Mode inference ou JIT: utiliser l'implémentation fusionnée
            return self._fuse_operations(x)

        # Mode training: utiliser la version avec checkpointing si:
        # - checkpointing est activé dans la config
        # - la séquence est longue (> 1024 tokens)
        if self.use_gradient_checkpointing and x.shape[1] > 1024:
            return checkpoint.checkpoint(self._fuse_operations, x, use_reentrant=False)

        return self._fuse_operations(x)