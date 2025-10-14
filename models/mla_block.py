"""MLA Block components for transformer models."""

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint

from normalization import RMSNorm, DynamicTanh
from mla import MLA
# from mla_fp8 import MLA_FP8
from mlp import MLP
# from tensor_utils import isolate_tensor, prevent_backward_reuse

class MLABlock(nn.Module):
    """
    Transformer block with Multi-Head Latent Attention (MLA) and MLP
    
    This block implements the architecture used in DeepSeek models with MLA for
    efficient memory usage through latent attention mechanisms.
    
    Attributes:
        layer_id (int): Identifier for this layer (used for debugging)
        attn_norm (nn.Module): Layer normalization for attention input
        attn (nn.Module): Multi-Head Latent Attention module
        ffn_norm (nn.Module): Layer normalization for feed-forward input
        ffn (nn.Module): MLP feed-forward network
        use_checkpoint (bool): Whether to use gradient checkpointing
    """
    def __init__(self, config, layer_id=None):
        super().__init__()
        # Layer identifier (for debugging)
        self.layer_id = layer_id
        
        # RMSNorm for attention and feed-forward
        if config.use_dyt:
            self.attn_norm = DynamicTanh(config.n_embd, alpha_init=config.dyt_alpha_init)
            self.ffn_norm = DynamicTanh(config.n_embd, alpha_init=config.dyt_alpha_init)
        else:
            self.attn_norm = RMSNorm(config.n_embd)
            self.ffn_norm = RMSNorm(config.n_embd)

        # MLA attention - FP8 version not included in standalone build
        self.attn = MLA(config)
        
        # Always use standard MLP (no MoE)
        self.ffn = MLP(config)
        
        # Gradient checkpointing
        self.use_checkpoint = config.use_gradient_checkpointing if hasattr(config, 'use_gradient_checkpointing') else True

    def _attn_block(self, x, start_pos=0, freqs_cis=None, mask=None):
        """Attention portion of the block with appropriate normalization"""
        x_norm = self.attn_norm(x)
        return self.attn(x_norm, start_pos, freqs_cis, mask)

    def _ffn_block(self, x):
        """Feed-forward portion of the block with appropriate normalization"""
        # Always use regular MLP (no MoE)
        return self.ffn(self.ffn_norm(x))

    def forward(self, x, start_pos=0, freqs_cis=None, mask=None):
        """
        Forward pass for the MLABlock.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, dim)
            start_pos (int, optional): Starting position for attention caching. Defaults to 0.
            freqs_cis (torch.Tensor, optional): Precomputed position embeddings. Defaults to None.
            mask (torch.Tensor, optional): Attention mask tensor. Defaults to None.
            
        Returns:
            torch.Tensor: Processed tensor with same shape as input
        """
        # Disable gradient checkpointing when using compiled model
        if hasattr(self, '_compiled_forward'):
            self.use_checkpoint = False
            
        # Standard residual transformer block implementation
        if self.use_checkpoint and self.training:
            # Gradient checkpointing for memory efficiency
            attn_out = checkpoint.checkpoint(
                self._attn_block, 
                x, 
                start_pos,
                freqs_cis,
                mask,
                use_reentrant=False
            )
            # First residual connection
            x = x + attn_out
            
            # FFN with checkpoint
            ffn_output = checkpoint.checkpoint(
                self._ffn_block, 
                x,
                use_reentrant=False
            )
            # Second residual connection
            x = x + ffn_output
        else:
            # Standard forward pass without checkpoint
            # First residual connection
            x = x + self._attn_block(x, start_pos, freqs_cis, mask)
            # Second residual connection
            x = x + self._ffn_block(x)
        
        return x