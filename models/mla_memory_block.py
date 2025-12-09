"""MLA Block augmented with Neural Long-term Memory.

This module implements an MLA attention block with an integrated neural memory
module and learned gating to interpolate between attention and memory outputs.

Based on the Titans paper (arXiv:2501.00663) combined with DeepSeek MLA.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

# Import TE checkpoint for FP8-aware gradient checkpointing
try:
    from optimization.fp8_te import te_checkpoint
except ImportError:
    te_checkpoint = None

from normalization import RMSNorm, DynamicTanh
from mla import MLA
from mlp import MLP
from neural_memory import NeuralLongTermMemory, MemoryState


class MLAMemoryBlock(nn.Module):
    """MLA block augmented with Neural Long-term Memory and learned gating.

    This block combines Multi-Head Latent Attention (MLA) with a neural memory
    module that learns at test-time. The key insight is that MLA's latent KV
    space (from W_kv_down) is reused as input to the memory module, providing:
    1. Computational efficiency by avoiding redundant projections
    2. Conceptual coherence since the latent space captures "what to remember"

    Data flow:
    1. x_norm = norm(x)
    2. kv_latent = W_kv_down(x_norm)           # Latent KV for both attention and memory
    3. attn_out = MLA(x_norm)                   # Standard MLA attention
    4. memory_out, new_state = Memory(kv_latent) # Neural memory with test-time learning
    5. memory_out = W_up(memory_out)            # Project back to model dimension
    6. gate = sigmoid(W_gate([attn_out || memory_out]))
    7. fused = gate * memory_out + (1-gate) * attn_out
    8. x = x + fused
    9. x = x + FFN(norm2(x))

    Attributes:
        layer_id: Identifier for this layer
        attn_norm: Pre-attention normalization
        ffn_norm: Pre-FFN normalization
        attn: MLA attention module
        memory: Neural long-term memory module
        memory_up: Projection from latent to model dimension
        gate_proj: Gating projection
        ffn: Feed-forward network
        use_checkpoint: Whether to use gradient checkpointing
    """

    def __init__(self, config, layer_id: int = None):
        """Initialize MLAMemoryBlock.

        Args:
            config: Configuration object with model parameters
            layer_id: Layer identifier for debugging
        """
        super().__init__()
        self.layer_id = layer_id
        self.config = config

        # Get dimensions from config
        n_embd = config.n_embd
        kv_lora_rank = getattr(config, 'kv_lora_rank', 512)
        memory_dim = getattr(config, 'memory_dim', 256)
        memory_depth = getattr(config, 'memory_depth', 2)
        use_dyt = getattr(config, 'use_dyt', False)
        dyt_alpha_init = getattr(config, 'dyt_alpha_init', 0.5)

        # Normalization layers
        if use_dyt:
            self.attn_norm = DynamicTanh(n_embd, alpha_init=dyt_alpha_init)
            self.ffn_norm = DynamicTanh(n_embd, alpha_init=dyt_alpha_init)
        else:
            self.attn_norm = RMSNorm(n_embd)
            self.ffn_norm = RMSNorm(n_embd)

        # MLA attention module
        self.attn = MLA(config)

        # Store kv_lora_rank for extracting latent from wkv_a output
        self.kv_lora_rank = kv_lora_rank

        # Neural memory operating on latent KV space
        self.memory = NeuralLongTermMemory(
            d_input=kv_lora_rank,
            memory_dim=memory_dim,
            memory_depth=memory_depth,
            d_output=kv_lora_rank,
            activation="silu"
        )

        # Projection from latent space back to model dimension
        self.memory_up = nn.Linear(kv_lora_rank, n_embd, bias=False)

        # Gating: concatenate attention and memory outputs, project to gate values
        # Gate is per-dimension, allowing fine-grained control
        self.gate_proj = nn.Linear(n_embd * 2, n_embd, bias=False)

        # Initialize gate bias toward attention (conservative start)
        # This ensures the model starts by relying on attention, then learns when to use memory
        nn.init.zeros_(self.gate_proj.weight)

        # Feed-forward network
        self.ffn = MLP(config)

        # Gradient checkpointing
        self.use_checkpoint = getattr(config, 'use_gradient_checkpointing', True)
        self.use_te_fp8 = getattr(config, 'use_te_fp8', False)

    def _compute_kv_latent(self, x_norm: torch.Tensor) -> torch.Tensor:
        """Extract KV latent representation for memory.

        The MLA module computes kv_latent = wkv_a(x)[:, :, :kv_lora_rank].
        We recompute this here to share between attention and memory.

        Args:
            x_norm: Normalized input [B, S, n_embd]

        Returns:
            kv_latent: Latent representation [B, S, kv_lora_rank]
        """
        # wkv_a projects to (kv_lora_rank + qk_rope_head_dim)
        # We only need the first kv_lora_rank dimensions
        kv_full = self.attn.wkv_a(x_norm)
        kv_latent = kv_full[:, :, :self.kv_lora_rank]
        return kv_latent

    def _attn_forward(
        self,
        x_norm: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Compute attention output.

        Args:
            x_norm: Normalized input
            start_pos: Starting position for caching
            freqs_cis: RoPE frequencies
            mask: Attention mask

        Returns:
            Attention output
        """
        return self.attn(x_norm, start_pos, freqs_cis, mask)

    def _memory_forward(
        self,
        kv_latent: torch.Tensor,
        memory_state: Optional[MemoryState]
    ) -> Tuple[torch.Tensor, MemoryState]:
        """Compute memory output.

        Args:
            kv_latent: Latent KV representation
            memory_state: Previous memory state

        Returns:
            memory_out: Memory output projected to model dimension
            new_state: Updated memory state
        """
        memory_latent, new_state = self.memory(kv_latent, memory_state)
        memory_out = self.memory_up(memory_latent)
        return memory_out, new_state

    def _fuse_outputs(
        self,
        attn_out: torch.Tensor,
        memory_out: torch.Tensor
    ) -> torch.Tensor:
        """Fuse attention and memory outputs via learned gating.

        Gate interpretation:
        - g -> 1: Trust memory (distant context needed)
        - g -> 0: Trust attention (recent context sufficient)

        Args:
            attn_out: Attention output [B, S, n_embd]
            memory_out: Memory output [B, S, n_embd]

        Returns:
            Fused output [B, S, n_embd]
        """
        # Concatenate and compute gate
        combined = torch.cat([attn_out, memory_out], dim=-1)
        gate = torch.sigmoid(self.gate_proj(combined))

        # Fuse outputs
        fused = gate * memory_out + (1 - gate) * attn_out
        return fused

    def _combined_forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
        memory_state: Optional[MemoryState]
    ) -> Tuple[torch.Tensor, MemoryState]:
        """Combined forward pass for attention + memory block.

        This is the main computation that can be checkpointed.
        """
        x_norm = self.attn_norm(x)

        # Compute KV latent for memory (shared with attention internally)
        kv_latent = self._compute_kv_latent(x_norm)

        # Attention path
        attn_out = self._attn_forward(x_norm, start_pos, freqs_cis, mask)

        # Memory path
        memory_out, new_state = self._memory_forward(kv_latent, memory_state)

        # Fuse via gating
        fused = self._fuse_outputs(attn_out, memory_out)

        return fused, new_state

    def _ffn_block(self, x: torch.Tensor) -> torch.Tensor:
        """Feed-forward block."""
        return self.ffn(self.ffn_norm(x))

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int = 0,
        freqs_cis: torch.Tensor = None,
        mask: Optional[torch.Tensor] = None,
        memory_state: Optional[MemoryState] = None
    ) -> Tuple[torch.Tensor, Optional[MemoryState]]:
        """Forward pass for MLAMemoryBlock.

        Args:
            x: Input tensor [B, S, n_embd]
            start_pos: Starting position for attention caching
            freqs_cis: Precomputed RoPE frequencies
            mask: Attention mask
            memory_state: Previous memory state (None for fresh start)

        Returns:
            output: Processed tensor [B, S, n_embd]
            new_state: Updated memory state
        """
        # Determine if we should use checkpointing
        use_ckpt = self.use_checkpoint and self.training

        if use_ckpt:
            # Note: Memory state updates happen inside the forward pass
            # but the memory module uses manual gradients, so this is compatible
            # with checkpointing

            if self.use_te_fp8 and te_checkpoint is not None:
                # FP8-aware checkpointing
                fused, new_state = te_checkpoint(
                    self._combined_forward,
                    x, start_pos, freqs_cis, mask, memory_state,
                    use_te_fp8=True
                )
            else:
                # Standard checkpointing
                # We need to handle memory_state carefully as it contains tensors
                fused, new_state = checkpoint.checkpoint(
                    self._combined_forward,
                    x, start_pos, freqs_cis, mask, memory_state,
                    use_reentrant=False
                )
        else:
            fused, new_state = self._combined_forward(
                x, start_pos, freqs_cis, mask, memory_state
            )

        # First residual connection (attention + memory)
        x = x + fused

        # FFN with optional checkpointing
        if use_ckpt:
            if self.use_te_fp8 and te_checkpoint is not None:
                ffn_out = te_checkpoint(self._ffn_block, x, use_te_fp8=True)
            else:
                ffn_out = checkpoint.checkpoint(
                    self._ffn_block, x,
                    use_reentrant=False
                )
        else:
            ffn_out = self._ffn_block(x)

        # Second residual connection (FFN)
        x = x + ffn_out

        return x, new_state
