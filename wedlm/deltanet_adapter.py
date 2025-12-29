"""
DeltaNet Adapter for WeDLM Training

Adapts GatedDeltaNet blocks to support WeDLM's topological reordering.

Problem: DeltaNet uses recurrent state without RoPE, so it cannot naturally
decouple physical from logical positions like attention with RoPE can.

Solution: Add learnable position embeddings before DeltaNet processing,
allowing the model to encode logical position information even when
physical order differs.

Reference: WeDLM paper Section 4.2 - handling non-RoPE architectures
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple


class WeDLMGatedDeltaNet(nn.Module):
    """
    GatedDeltaNet wrapper adapted for WeDLM with position embeddings.

    Adds learnable position embeddings that encode logical positions,
    enabling the model to maintain correct position awareness even
    when tokens are physically reordered.
    """

    def __init__(
        self,
        original_deltanet: nn.Module,
        max_seq_len: int,
        d_model: int,
        position_embedding_type: str = "learned",
    ):
        """
        Args:
            original_deltanet: The original GatedDeltaNet module to wrap
            max_seq_len: Maximum sequence length for position embeddings
            d_model: Model dimension (n_embd)
            position_embedding_type: Type of position embedding ("learned" or "sinusoidal")
        """
        super().__init__()

        self.deltanet = original_deltanet
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.position_embedding_type = position_embedding_type

        # Learnable position embeddings (like BERT/GPT)
        if position_embedding_type == "learned":
            self.position_embeddings = nn.Embedding(max_seq_len, d_model)
            self._init_position_embeddings()
        elif position_embedding_type == "sinusoidal":
            # Fixed sinusoidal embeddings
            pe = self._create_sinusoidal_embeddings(max_seq_len, d_model)
            self.register_buffer("position_embeddings", pe)
        else:
            raise ValueError(f"Unknown position_embedding_type: {position_embedding_type}")

        # Optional gating for position information
        # This allows the model to learn how much position info to use
        self.position_gate = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Sigmoid(),
        )

        # Layer norm to combine position + content
        self.pos_norm = nn.LayerNorm(d_model)

    def _init_position_embeddings(self):
        """Initialize position embeddings with small values."""
        nn.init.normal_(self.position_embeddings.weight, mean=0.0, std=0.02)

    def _create_sinusoidal_embeddings(self, max_len: int, d_model: int) -> torch.Tensor:
        """Create sinusoidal position embeddings."""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def forward(
        self,
        x: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        state: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Forward pass with position-aware input.

        Args:
            x: [B, L, D] hidden states
            position_ids: [B, L] logical positions (may differ from physical order)
                         If None, uses sequential positions 0, 1, 2, ...
            state: Optional recurrent state for inference
            **kwargs: Additional arguments (ignored, for compatibility)

        Returns:
            output: [B, L, D] transformed hidden states
        """
        B, L, D = x.shape
        device = x.device

        # Default to sequential positions if not provided
        if position_ids is None:
            position_ids = torch.arange(L, device=device).unsqueeze(0).expand(B, -1)

        # Clamp position_ids to valid range
        position_ids = position_ids.clamp(0, self.max_seq_len - 1)

        # Get position embeddings for the specified positions
        if self.position_embedding_type == "learned":
            pos_emb = self.position_embeddings(position_ids)  # [B, L, D]
        else:
            # Sinusoidal: gather from buffer
            pos_emb = self.position_embeddings[position_ids]  # [B, L, D]

        # Gate the position information
        # This allows the model to adaptively weight position vs content
        gate = self.position_gate(x)  # [B, L, D]
        gated_pos = gate * pos_emb

        # Combine position and content information
        x_with_pos = self.pos_norm(x + gated_pos)

        # Forward through original DeltaNet
        output = self.deltanet(x_with_pos, state=state)

        return output

    def forward_with_position_ids(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Explicit interface for WeDLM forward pass with position IDs.

        This method is called by the model when WeDLM mode is active
        and position_ids need to be explicitly passed.
        """
        return self.forward(x, position_ids=position_ids, **kwargs)


class WeDLMGatedDeltaNetBlock(nn.Module):
    """
    Wrapper for GatedDeltaNetBlock that supports WeDLM position_ids.

    This wraps the entire block (including norms and MLP) to provide
    a consistent interface with WeDLM's position_ids parameter.
    """

    def __init__(
        self,
        original_block: nn.Module,
        max_seq_len: int,
        d_model: int,
    ):
        """
        Args:
            original_block: The original GatedDeltaNetBlock
            max_seq_len: Maximum sequence length
            d_model: Model dimension
        """
        super().__init__()

        # Keep the original block structure
        self.norm1 = original_block.norm1
        self.norm2 = original_block.norm2
        self.mlp = original_block.mlp
        self.use_checkpoint = original_block.use_checkpoint

        # Wrap the attention with position embedding support
        self.attn = WeDLMGatedDeltaNet(
            original_deltanet=original_block.attn,
            max_seq_len=max_seq_len,
            d_model=d_model,
        )

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int = 0,
        freqs_cis: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        seq_lengths: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass with optional position_ids for WeDLM.

        Args:
            x: [B, L, D] input hidden states
            start_pos: Starting position (for KV cache, ignored here)
            freqs_cis: RoPE frequencies (ignored, DeltaNet doesn't use RoPE)
            mask: Attention mask (ignored, DeltaNet uses recurrent state)
            seq_lengths: Sequence lengths (ignored)
            position_ids: [B, L] logical positions for WeDLM

        Returns:
            output: [B, L, D] transformed hidden states
        """
        # Self-attention with residual (using position_ids if provided)
        x = x + self.attn(self.norm1(x), position_ids=position_ids)

        # MLP with residual
        x = x + self.mlp(self.norm2(x))

        return x

    def forward_with_position_ids(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Explicit interface for WeDLM forward pass.
        """
        return self.forward(x, position_ids=position_ids, **kwargs)


def adapt_deltanet_for_wedlm(
    model: nn.Module,
    max_seq_len: int,
    d_model: Optional[int] = None,
) -> nn.Module:
    """
    Replace GatedDeltaNetBlock modules with WeDLM-adapted versions.

    This function recursively traverses the model and wraps any
    GatedDeltaNetBlock instances with WeDLMGatedDeltaNetBlock.

    Args:
        model: The model containing GatedDeltaNetBlock modules
        max_seq_len: Maximum sequence length for position embeddings
        d_model: Model dimension (inferred from blocks if not provided)

    Returns:
        model: The model with adapted DeltaNet blocks
    """
    # Import here to avoid circular imports
    try:
        from models.gated_deltanet import GatedDeltaNetBlock
    except ImportError:
        from gated_deltanet import GatedDeltaNetBlock

    # Find and replace GatedDeltaNetBlock instances
    replacements = []

    for name, module in model.named_modules():
        if isinstance(module, GatedDeltaNetBlock):
            # Infer d_model from the module if not provided
            module_d_model = d_model or module.attn.n_embd

            # Create adapted block
            adapted_block = WeDLMGatedDeltaNetBlock(
                original_block=module,
                max_seq_len=max_seq_len,
                d_model=module_d_model,
            )

            replacements.append((name, adapted_block))

    # Apply replacements
    for name, new_module in replacements:
        _set_module_by_name(model, name, new_module)

    if replacements:
        print(f"[WeDLM] Adapted {len(replacements)} GatedDeltaNet block(s) for WeDLM training")

    return model


def _get_parent_module(model: nn.Module, name: str) -> Tuple[nn.Module, str]:
    """Get parent module and child name from dotted name."""
    parts = name.split('.')
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def _set_module_by_name(model: nn.Module, name: str, new_module: nn.Module):
    """Set a module by its dotted name path."""
    parent, child_name = _get_parent_module(model, name)
    setattr(parent, child_name, new_module)


def check_deltanet_wedlm_compatible(model: nn.Module) -> bool:
    """
    Check if the model's DeltaNet blocks are WeDLM-compatible.

    Returns True if all DeltaNet blocks have the forward_with_position_ids method.
    """
    for name, module in model.named_modules():
        if hasattr(module, 'deltanet') or 'deltanet' in name.lower():
            if not hasattr(module, 'forward_with_position_ids'):
                return False
    return True
