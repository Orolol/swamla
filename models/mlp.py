"""MLP and Block components for transformer models."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

from normalization import RMSNorm
from attention import CausalSelfAttention

# Import TE FP8 helper
try:
    from optimization.fp8_te import get_te_linear, HAS_TE
except ImportError:
    HAS_TE = False
    def get_te_linear(in_features, out_features, bias=True, use_te_fp8=False):
        return nn.Linear(in_features, out_features, bias=bias)


class MLP(nn.Module):
    """
    Multi-Layer Perceptron avec activation SwiGLU optimisée et gestion efficace de la mémoire.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config

        # Utiliser 4 * n_embd comme dimension cachée par défaut
        hidden_dim = 4 * config.n_embd

        # Check if TE FP8 should be used
        use_te_fp8 = getattr(config, 'use_te_fp8', False)

        # Projections combinées pour réduire le nombre d'opérations
        # Use TE FP8 linear layers if enabled
        self.gate_up_proj = get_te_linear(config.n_embd, 2 * hidden_dim, bias=config.bias, use_te_fp8=use_te_fp8)
        self.down_proj = get_te_linear(hidden_dim, config.n_embd, bias=config.bias, use_te_fp8=use_te_fp8)
        self.using_te_fp8 = use_te_fp8 and HAS_TE

        self.dropout = nn.Dropout(config.dropout)

        # Activation function setup
        self.act_fn = self._get_optimized_activation()

        # Gradient checkpointing control
        self.use_gradient_checkpointing = getattr(config, 'use_gradient_checkpointing', True)

        # Initialize weights with a special scale for better gradient flow
        self._init_weights()

    def _init_weights(self):
        """
        Initialisation spéciale des poids pour une meilleure convergence
        """
        # Scaled initialization for better gradient flow
        scale = 2 / (self.config.n_embd ** 0.5)

        # TE Linear has its own init, so only init if not using TE FP8
        if not self.using_te_fp8:
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

class Block(nn.Module):
    """
    Transformer block with attention and MLP
    """
    def __init__(self, config):
        super().__init__()
        self.ln_1 = RMSNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = RMSNorm(config.n_embd)
        self.mlp = MLP(config)
        # Respect config for gradient checkpointing
        self.use_checkpoint = getattr(config, 'use_gradient_checkpointing', True)

    def _attn_block(self, x, key_value=None):
        ln_out = self.ln_1(x)
        if key_value is not None:
            return self.attn(ln_out, key_value=key_value)
        return self.attn(ln_out)

    def _mlp_block(self, x):
        return self.mlp(self.ln_2(x))

    def forward(self, x, key_value=None):
        # Disable gradient checkpointing when using compiled model
        if hasattr(self, '_compiled_forward'):
            self.use_checkpoint = False
        
        if self.use_checkpoint and self.training:
            # Modified wrapper for gradient checkpointing
            def create_custom_forward(func):
                def custom_forward(*inputs):
                    # Filter out None inputs
                    valid_inputs = [inp for inp in inputs if inp is not None]
                    return func(*valid_inputs)
                return custom_forward

            # Attention with checkpoint
            attn_func = create_custom_forward(self._attn_block)
            attn_out = checkpoint.checkpoint(
                attn_func, 
                x, 
                key_value,
                use_reentrant=False,
                preserve_rng_state=True
            )
            x = x + attn_out

            # MLP with checkpoint
            mlp_func = create_custom_forward(self._mlp_block)
            mlp_out = checkpoint.checkpoint(
                mlp_func, 
                x,
                use_reentrant=False,
                preserve_rng_state=True
            )
            x = x + mlp_out
        else:
            # Standard forward pass without checkpoint
            x = x + self._attn_block(x, key_value)
            x = x + self._mlp_block(x)

        return x 