"""MLP components for transformer models."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

from normalization import RMSNorm


class MLP(nn.Module):
    """
    Multi-Layer Perceptron with SwiGLU activation.
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

        # Activation function setup
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
        Fusionne les opérations quand c'est possible pour une meilleure efficacité
        """
        # Remember input dtype for output conversion
        input_dtype = x.dtype
        
        # Combiner les projections up et gate en une seule opération
        combined = self.gate_up_proj(x)
        
        # Appliquer l'activation
        hidden = self.act_fn(combined)
        
        # Projection finale
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