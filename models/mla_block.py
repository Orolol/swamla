"""MLA Block components for transformer models."""

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint

# Import TE checkpoint for FP8-aware gradient checkpointing
try:
    from optimization.fp8_te import te_checkpoint
except ImportError:
    te_checkpoint = None

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
        use_te_fp8 (bool): Whether using TE FP8 (for FP8-aware checkpointing)
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

        # FFN: MoE (standard or latent) or standard MLP based on config
        if getattr(config, 'use_moe', False):
            from moe import create_moe_layer
            use_latent = getattr(config, 'use_latent_moe', False)
            self.ffn = create_moe_layer(config, use_latent=use_latent)
        else:
            self.ffn = MLP(config)

        # Gradient checkpointing
        self.use_checkpoint = config.use_gradient_checkpointing if hasattr(config, 'use_gradient_checkpointing') else True
        self.use_te_fp8 = getattr(config, 'use_te_fp8', False)

        # Engram: Conditional Memory via N-gram Lookup (applied BEFORE attention)
        self.engram = None
        if getattr(config, 'use_engram', False):
            engram_layers = getattr(config, 'engram_layers', [])
            if layer_id is not None and layer_id in engram_layers:
                from engram import create_engram_for_config
                self.engram = create_engram_for_config(config, layer_id)

    def _engram_block(self, x, input_ids):
        """Engram conditional memory lookup (applied BEFORE attention)"""
        if self.engram is not None and input_ids is not None:
            return self.engram(x, input_ids)
        return torch.zeros_like(x)

    def _attn_block(self, x, start_pos=0, freqs_cis=None, mask=None, position_ids=None):
        """Attention portion of the block with appropriate normalization"""
        x_norm = self.attn_norm(x)
        return self.attn(x_norm, start_pos, freqs_cis, mask, position_ids=position_ids)

    def _ffn_block(self, x):
        """Feed-forward portion of the block with appropriate normalization"""
        return self.ffn(self.ffn_norm(x))

    def forward(self, x, start_pos=0, freqs_cis=None, mask=None, position_ids=None, input_ids=None):
        """
        Forward pass for the MLABlock.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, dim)
            start_pos (int, optional): Starting position for attention caching. Defaults to 0.
            freqs_cis (torch.Tensor, optional): Precomputed position embeddings. Defaults to None.
            mask (torch.Tensor, optional): Attention mask tensor. Defaults to None.
            position_ids (torch.Tensor, optional): Explicit position IDs for WeDLM. Defaults to None.
            input_ids (torch.Tensor, optional): Original token IDs for Engram lookup. Defaults to None.

        Returns:
            torch.Tensor: Processed tensor with same shape as input
        """
        # Disable gradient checkpointing when using compiled model
        if hasattr(self, '_compiled_forward'):
            self.use_checkpoint = False

        use_ckpt = self.use_checkpoint and self.training

        # 1. Engram: Conditional memory lookup BEFORE attention
        # Position: H = H + Engram(H, input_ids)
        if self.engram is not None and input_ids is not None:
            if use_ckpt:
                engram_out = checkpoint.checkpoint(
                    self._engram_block,
                    x,
                    input_ids,
                    use_reentrant=False
                )
            else:
                engram_out = self._engram_block(x, input_ids)
            x = x + engram_out

        # 2. Standard residual transformer block implementation
        if use_ckpt:
            # Use TE checkpoint for FP8-aware checkpointing (handles FP8 states properly)
            if self.use_te_fp8 and te_checkpoint is not None:
                attn_out = te_checkpoint(
                    self._attn_block,
                    x,
                    start_pos,
                    freqs_cis,
                    mask,
                    position_ids,
                    use_te_fp8=True
                )
            else:
                attn_out = checkpoint.checkpoint(
                    self._attn_block,
                    x,
                    start_pos,
                    freqs_cis,
                    mask,
                    position_ids,
                    use_reentrant=False
                )
            # First residual connection
            x = x + attn_out

            # FFN with checkpoint
            if self.use_te_fp8 and te_checkpoint is not None:
                ffn_output = te_checkpoint(self._ffn_block, x, use_te_fp8=True)
            else:
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
            x = x + self._attn_block(x, start_pos, freqs_cis, mask, position_ids)
            # Second residual connection
            x = x + self._ffn_block(x)

        return x