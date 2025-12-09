"""Hybrid model interleaving Sliding-Window Attention (SWA) and MLA/MLA-Selective blocks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Dict, Tuple, List

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
from attention import CausalSelfAttention
from mlp import MLP
from mla_block import MLABlock
from mla_memory_block import MLAMemoryBlock
from neural_memory import MemoryState
# from mla_selective_fast import MLASelectiveFast as MLASelective
from positional_encoding import (
    precompute_freqs_cis,
    precompute_freqs_cis_with_linear_scaling,
)
# from optimizers import configure_optimizer_for_gpt

# FlexAttention support (PyTorch >= 2.5)
try:
    from attention_flex import FlexCausalSelfAttention, get_seq_lengths_from_labels, FLEX_ATTENTION_AVAILABLE
except ImportError:
    FlexCausalSelfAttention = None
    FLEX_ATTENTION_AVAILABLE = False
    def get_seq_lengths_from_labels(labels, ignore_index=-100):
        # Fallback implementation
        batch_size, seq_len = labels.shape
        non_pad = labels != ignore_index
        positions = torch.arange(seq_len, device=labels.device).unsqueeze(0).expand(batch_size, -1)
        masked_pos = torch.where(non_pad, positions, torch.tensor(-1, device=labels.device))
        seq_lengths = masked_pos.max(dim=1).values + 1
        return torch.clamp(seq_lengths, min=1)


@dataclass
class SWALayerConfig:
    """Configuration forwarded to SWA attention blocks."""

    n_embd: int
    n_head: int
    block_size: int
    dropout: float
    bias: bool
    ratio_kv: int
    attention_backend: Optional[str]
    use_rope: bool
    attention_window: Optional[int]
    attention_sink_size: int  # Number of initial tokens to always attend to
    rope_theta: float
    logit_scale_base: Optional[float]
    logit_scale_window: int
    logit_scale_offset: int
    logit_scale_min: float
    logit_scale_max: Optional[float]
    logit_scale_during_training: bool
    use_gradient_checkpointing: bool
    use_dyt: bool
    dyt_alpha_init: float
    # FP8 support for SWA blocks
    use_fp8: bool = False
    fp8_mla_params: bool = False
    fp8_tile_size: int = 128
    use_te_fp8: bool = False  # Use Transformer Engine native FP8 Linear layers
    # FlexAttention support (padding-aware attention)
    use_flex_attention: bool = False


class SWALocalBlock(nn.Module):
    """Sliding-window attention block with residual MLP."""

    def __init__(self, config: SWALayerConfig):
        super().__init__()
        if config.use_dyt:
            self.norm1 = DynamicTanh(config.n_embd, alpha_init=config.dyt_alpha_init)
            self.norm2 = DynamicTanh(config.n_embd, alpha_init=config.dyt_alpha_init)
        else:
            self.norm1 = RMSNorm(config.n_embd)
            self.norm2 = RMSNorm(config.n_embd)

        # Choose attention implementation based on config
        self.use_flex_attention = getattr(config, 'use_flex_attention', False)
        if self.use_flex_attention and FlexCausalSelfAttention is not None:
            self.attn = FlexCausalSelfAttention(config)
        else:
            self.attn = CausalSelfAttention(config)
            if self.use_flex_attention and FlexCausalSelfAttention is None:
                print("Warning: FlexAttention requested but not available, falling back to standard attention")
                self.use_flex_attention = False

        self.mlp = MLP(config)
        self.use_checkpoint = config.use_gradient_checkpointing
        self.use_te_fp8 = getattr(config, 'use_te_fp8', False)

    def _attn_block(self, x: torch.Tensor, seq_lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.use_flex_attention and seq_lengths is not None:
            return self.attn(self.norm1(x), seq_lengths=seq_lengths)
        else:
            return self.attn(self.norm1(x))

    def _mlp_block(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.norm2(x))

    def forward(self, x: torch.Tensor, seq_lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        use_ckpt = self.use_checkpoint and self.training

        if use_ckpt:
            # Use TE checkpoint for FP8-aware checkpointing (handles FP8 states properly)
            if self.use_te_fp8 and te_checkpoint is not None:
                attn_out = te_checkpoint(self._attn_block, x, seq_lengths, use_te_fp8=True)
            else:
                # Note: seq_lengths passed but checkpointing may not preserve it correctly
                # For now, skip seq_lengths in checkpointed mode
                attn_out = checkpoint.checkpoint(
                    lambda inp: self._attn_block(inp, seq_lengths),
                    x,
                    use_reentrant=False
                )
        else:
            attn_out = self._attn_block(x, seq_lengths)
        x = x + attn_out

        if use_ckpt:
            if self.use_te_fp8 and te_checkpoint is not None:
                mlp_out = te_checkpoint(self._mlp_block, x, use_te_fp8=True)
            else:
                mlp_out = checkpoint.checkpoint(self._mlp_block, x, use_reentrant=False)
        else:
            mlp_out = self._mlp_block(x)
        x = x + mlp_out
        return x


@dataclass
class SWAMLAConfig:
    """Top-level configuration for the SWA+MLA hybrid model."""

    vocab_size: int = 50304
    block_size: int = 4096
    n_layer: int = 24
    n_head: int = 16
    n_embd: int = 1536
    dropout: float = 0.0
    bias: bool = False
    ratio_kv: int = 1
    attention_backend: Optional[str] = None
    use_gradient_checkpointing: bool = True
    use_dyt: bool = False
    dyt_alpha_init: float = 0.5

    swa_layers_per_cycle: int = 2
    mla_layers_per_cycle: int = 1
    swa_window: int = 256
    swa_sink_size: int = 4  # Number of initial tokens to always attend to (attention sink)
    rope_theta: float = 10000.0

    logit_scale_base: Optional[float] = None
    logit_scale_window: int = 128
    logit_scale_offset: int = 0
    logit_scale_min: float = 1.0
    logit_scale_max: Optional[float] = None
    apply_logit_scale_during_training: bool = False

    # MLA specific parameters
    q_lora_rank: int = 0
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128
    attn_impl: str = "absorb"
    world_size: int = 1
    rope_scaling: Optional[Dict[str, float]] = None
    rope_factor: float = 1.0
    mscale: float = 1.0
    use_fp8: bool = False
    fp8_mla_params: bool = False
    fp8_tile_size: int = 128
    use_te_fp8: bool = False  # Use Transformer Engine native FP8 Linear layers

    # FlexAttention (padding-aware attention for real FLOPs savings)
    use_flex_attention: bool = False  # Use FlexAttention with BlockMask for SWA blocks

    # MLA Selective specific parameters
    use_mla_selective: bool = False  # Use MLA Selective instead of standard MLA
    selection_head_idx: int = 0  # Which attention head to use for selection

    # Neural Memory configuration (Titans paper)
    use_neural_memory: bool = False  # Enable neural long-term memory for MLA blocks
    memory_dim: int = 256  # Internal MLP dimension in memory module
    memory_depth: int = 2  # Number of layers in memory MLP
    memory_layers: Optional[List[int]] = None  # Which layers have memory (None = all MLA layers)

    label_smoothing: float = 0.0

    # MoE configuration (only applied to MLA blocks)
    use_moe: bool = False
    n_experts: int = 32
    n_shared_experts: int = 1
    n_activated: int = 3  # routed experts per token (+ shared = 4 total active)
    expert_dim: Optional[int] = None  # defaults to n_embd if None
    router_z_loss_coef: float = 0.001

    def __post_init__(self) -> None:
        # Set expert_dim to n_embd if not specified
        if self.expert_dim is None:
            self.expert_dim = self.n_embd
        if self.swa_layers_per_cycle < 0 or self.mla_layers_per_cycle < 0:
            raise ValueError("Layer counts per cycle must be non-negative")
        if self.swa_layers_per_cycle + self.mla_layers_per_cycle == 0:
            raise ValueError("At least one SWA or MLA layer per cycle is required")


class MLASelectiveBlock(nn.Module):
    """MLA Selective block with residual MLP."""

    def __init__(self, config, layer_id: int):
        super().__init__()
        self.config = config
        self.layer_id = layer_id

        if config.use_dyt:
            self.norm1 = DynamicTanh(config.n_embd, alpha_init=config.dyt_alpha_init)
            self.norm2 = DynamicTanh(config.n_embd, alpha_init=config.dyt_alpha_init)
        else:
            self.norm1 = RMSNorm(config.n_embd)
            self.norm2 = RMSNorm(config.n_embd)

        # MLASelective not included in this standalone version - use standard MLA instead
        from mla import MLA
        self.attn = MLA(config)
        self.mlp = MLP(config)
        self.use_checkpoint = config.use_gradient_checkpointing
        self.use_te_fp8 = getattr(config, 'use_te_fp8', False)

    def _attn_block(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        return self.attn(self.norm1(x), start_pos, freqs_cis, mask)

    def _mlp_block(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.norm2(x))

    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        use_ckpt = self.use_checkpoint and self.training

        if use_ckpt:
            # Use TE checkpoint for FP8-aware checkpointing
            if self.use_te_fp8 and te_checkpoint is not None:
                attn_out = te_checkpoint(self._attn_block, x, start_pos, freqs_cis, mask, use_te_fp8=True)
            else:
                attn_out = checkpoint.checkpoint(self._attn_block, x, start_pos, freqs_cis, mask, use_reentrant=False)
        else:
            attn_out = self._attn_block(x, start_pos, freqs_cis, mask)
        x = x + attn_out

        if use_ckpt:
            if self.use_te_fp8 and te_checkpoint is not None:
                mlp_out = te_checkpoint(self._mlp_block, x, use_te_fp8=True)
            else:
                mlp_out = checkpoint.checkpoint(self._mlp_block, x, use_reentrant=False)
        else:
            mlp_out = self._mlp_block(x)
        x = x + mlp_out
        return x


class SWAMLAModel(nn.Module):
    """Model interleaving SWA layers with MLA Selective blocks."""

    def __init__(self, config: SWAMLAConfig):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                drop=nn.Dropout(config.dropout),
                h=nn.ModuleList(),
                ln_f=DynamicTanh(config.n_embd, alpha_init=config.dyt_alpha_init)
                if config.use_dyt
                else RMSNorm(config.n_embd),
            )
        )

        # Precompute RoPE frequencies for MLA layers
        head_dim = config.qk_rope_head_dim
        if config.rope_scaling is not None:
            scaling_type = config.rope_scaling.get("type")
            scaling_factor = config.rope_scaling.get("factor")
            if scaling_type == "linear":
                freqs = precompute_freqs_cis_with_linear_scaling(
                    head_dim,
                    config.block_size,
                    config.rope_theta,
                    scaling_factor,
                    config.block_size,
                )
            else:
                raise ValueError(f"Unknown RoPE scaling type: {scaling_type}")
        else:
            freqs = precompute_freqs_cis(
                head_dim,
                config.block_size,
                config.rope_theta,
            )
        self.register_buffer("freqs_cis", freqs, persistent=False)

        cycle_len = config.swa_layers_per_cycle + config.mla_layers_per_cycle
        swa_per_cycle = config.swa_layers_per_cycle

        for layer_idx in range(config.n_layer):
            # Calculate scaling factor for this layer: 0.5x at first layer to 2.0x at last layer
            if config.n_layer > 1:
                scale_factor = 0.5 + 1.5 * (layer_idx / (config.n_layer - 1))
            else:
                scale_factor = 1.0
            
            position_in_cycle = layer_idx % cycle_len
            if position_in_cycle < swa_per_cycle:
                # SWA Layer
                scaled_window = int(config.swa_window * scale_factor)
                # Ensure window is at least 1 and even (optional but good practice)
                scaled_window = max(1, scaled_window)
                
                logit_scale_window = config.logit_scale_window if config.logit_scale_window is not None else 128
                logit_scale_offset = config.logit_scale_offset if config.logit_scale_offset is not None else 0
                logit_scale_min = config.logit_scale_min if config.logit_scale_min is not None else 1.0
                layer_config = SWALayerConfig(
                    n_embd=config.n_embd,
                    n_head=config.n_head,
                    block_size=config.block_size,
                    dropout=config.dropout,
                    bias=config.bias,
                    ratio_kv=config.ratio_kv,
                    attention_backend=config.attention_backend,
                    use_rope=True,
                    attention_window=scaled_window,
                    attention_sink_size=config.swa_sink_size,
                    rope_theta=config.rope_theta,
                    logit_scale_base=config.logit_scale_base,
                    logit_scale_window=logit_scale_window,
                    logit_scale_offset=logit_scale_offset,
                    logit_scale_min=logit_scale_min,
                    logit_scale_max=config.logit_scale_max,
                    logit_scale_during_training=config.apply_logit_scale_during_training,
                    use_gradient_checkpointing=config.use_gradient_checkpointing,
                    use_dyt=config.use_dyt,
                    dyt_alpha_init=config.dyt_alpha_init,
                    # Pass FP8 configuration to SWA blocks
                    use_fp8=config.use_fp8,
                    fp8_mla_params=config.fp8_mla_params,
                    fp8_tile_size=config.fp8_tile_size,
                    use_te_fp8=config.use_te_fp8,
                    # FlexAttention for padding-aware attention
                    use_flex_attention=config.use_flex_attention,
                )
                block = SWALocalBlock(layer_config)
            else:
                # MLA Layer
                scaled_rank = int(config.kv_lora_rank * scale_factor)
                # Ensure rank is at least 1 and multiple of 16/32 if needed, but let's just keep it int for now
                scaled_rank = max(1, scaled_rank)

                # Create a copy of config with modified rank
                # We can't easily copy and modify the dataclass if it's frozen, but SWAMLAConfig is not frozen
                # However, to avoid side effects, let's use the helper or modify a copy

                if config.use_mla_selective:
                    # For MLASelective, we need to pass a config object
                    # Let's create a proxy or modified copy
                    import copy
                    layer_specific_config = copy.copy(config)
                    layer_specific_config.kv_lora_rank = scaled_rank
                    block = MLASelectiveBlock(layer_specific_config, layer_id=layer_idx)
                elif config.use_neural_memory:
                    # Check if this layer should have memory
                    has_memory = (
                        config.memory_layers is None or
                        layer_idx in config.memory_layers
                    )
                    if has_memory:
                        # Use MLAMemoryBlock with neural long-term memory
                        mla_config = _create_mla_block_config(config)
                        mla_config.kv_lora_rank = scaled_rank
                        mla_config.memory_dim = config.memory_dim
                        mla_config.memory_depth = config.memory_depth
                        block = MLAMemoryBlock(mla_config, layer_id=layer_idx)
                    else:
                        # Standard MLA without memory
                        mla_config = _create_mla_block_config(config)
                        mla_config.kv_lora_rank = scaled_rank
                        block = MLABlock(mla_config, layer_id=layer_idx)
                else:
                    mla_config = _create_mla_block_config(config)
                    mla_config.kv_lora_rank = scaled_rank
                    block = MLABlock(mla_config, layer_id=layer_idx)
            block.use_checkpoint = config.use_gradient_checkpointing
            self.transformer.h.append(block)

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight = self.transformer.wte.weight

        self.apply(self._init_weights)
        self.param_count = sum(p.numel() for p in self.parameters())

        # Count SWA and MLA blocks
        swa_count = sum(1 for block in self.transformer.h if isinstance(block, SWALocalBlock))
        mla_selective_count = sum(1 for block in self.transformer.h if isinstance(block, MLASelectiveBlock))
        mla_memory_count = sum(1 for block in self.transformer.h if isinstance(block, MLAMemoryBlock))
        mla_standard_count = sum(1 for block in self.transformer.h if isinstance(block, MLABlock) and not isinstance(block, MLAMemoryBlock))

        if config.use_mla_selective:
            model_type = "SWAMLA-Selective"
        elif config.use_neural_memory:
            model_type = "SWAMLA-Memory"
        else:
            model_type = "SWAMLA"

        print(f"{model_type} Model - Number of parameters: {self.param_count / 1e6:.2f}M")
        if config.use_mla_selective:
            print(f"  - {swa_count} SWA blocks, {mla_selective_count} MLA Selective blocks")
        elif config.use_neural_memory:
            print(f"  - {swa_count} SWA blocks, {mla_memory_count} MLA Memory blocks, {mla_standard_count} MLA blocks")
        else:
            print(f"  - {swa_count} SWA blocks, {mla_standard_count} MLA blocks")

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        return_all_logits: bool = False,
        memory_states: Optional[List[MemoryState]] = None,
        return_memory_states: bool = False
    ):
        """Forward pass with optional memory state management.

        Args:
            idx: Input token indices [B, S]
            targets: Target token indices for loss computation [B, S]
            return_all_logits: If True, return logits for all positions
            memory_states: List of MemoryState objects for MLAMemoryBlocks
            return_memory_states: If True, return updated memory states

        Returns:
            If return_memory_states is False:
                (logits, loss)
            If return_memory_states is True:
                (logits, loss, new_memory_states)

        For training with truncated BPTT:
            logits, loss, new_states = model(x, targets, memory_states=states,
                                             return_memory_states=True)
            states = [s.detach() for s in new_states]  # Truncate gradients
        """
        device = idx.device
        b, t = idx.size()
        if t > self.config.block_size:
            raise ValueError(
                f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
            )

        tok_emb = self.transformer.wte(idx)
        x = self.transformer.drop(tok_emb)

        attn_mask = None
        if t > 1:
            attn_mask = torch.full((t, t), float("-inf"), device=device)
            attn_mask = torch.triu(attn_mask, diagonal=1)

        # Move freqs_cis to the correct device on-demand to avoid VRAM duplication in DDP
        freqs_cis = self.freqs_cis[:t].to(device, non_blocking=True).detach()

        # Extract sequence lengths for FlexAttention (to skip padded positions)
        seq_lengths = None
        if self.config.use_flex_attention and targets is not None:
            seq_lengths = get_seq_lengths_from_labels(targets, ignore_index=-100)

        # Track memory states for MLAMemoryBlocks
        new_memory_states = []
        memory_idx = 0

        for block in self.transformer.h:
            if isinstance(block, SWALocalBlock):
                x = block(x, seq_lengths=seq_lengths)
            elif isinstance(block, MLAMemoryBlock):
                # MLAMemoryBlock handles memory state
                state = memory_states[memory_idx] if memory_states else None
                x, new_state = block(x, 0, freqs_cis, attn_mask, memory_state=state)
                new_memory_states.append(new_state)
                memory_idx += 1
            elif isinstance(block, (MLASelectiveBlock, MLABlock)):
                x = block(x, 0, freqs_cis, attn_mask)
            else:
                # Fallback for any other block type
                x = block(x, 0, freqs_cis, attn_mask)

        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            if self.config.label_smoothing > 0.0:
                num_classes = logits.size(-1)
                smoothing = self.config.label_smoothing
                confidence = 1.0 - smoothing
                smoothing_value = smoothing / (num_classes - 1)
                with torch.no_grad():
                    true_dist = torch.zeros_like(logits)
                    true_dist.fill_(smoothing_value)
                    true_dist.scatter_(-1, targets.unsqueeze(-1), confidence)
                log_probs = F.log_softmax(logits.view(-1, num_classes), dim=-1)
                loss = -(true_dist.view(-1, num_classes) * log_probs).sum(-1)
                with torch.no_grad():
                    loss_mask = (targets != -100).float()
                loss = (loss * loss_mask.view(-1)).sum() / (loss_mask.sum() + 1e-6)
            else:
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    targets.view(-1),
                    ignore_index=-100,
                )
        elif return_all_logits:
            # Training mode: return logits for all positions without computing loss
            logits = self.lm_head(x)
            loss = None
        else:
            # Inference mode: return logits only for last position
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        if return_memory_states:
            return logits, loss, new_memory_states

        return logits, loss

    def get_moe_aux_loss(self) -> torch.Tensor:
        """Collect auxiliary losses from all MoE layers."""
        total_aux_loss = torch.tensor(0.0, device=next(self.parameters()).device)
        moe_count = 0

        for block in self.transformer.h:
            # Check if block has MoE FFN
            ffn = getattr(block, 'ffn', getattr(block, 'mlp', None))
            if ffn is not None and hasattr(ffn, 'get_aux_loss'):
                total_aux_loss = total_aux_loss + ffn.get_aux_loss()
                moe_count += 1

        return total_aux_loss

    def get_moe_load_balance_stats(self) -> dict:
        """Collect load balance statistics from all MoE layers."""
        stats = {}
        for i, block in enumerate(self.transformer.h):
            ffn = getattr(block, 'ffn', getattr(block, 'mlp', None))
            if ffn is not None and hasattr(ffn, 'get_load_balance_stats'):
                stats[f'layer_{i}'] = ffn.get_load_balance_stats()
        return stats

    def update_moe_bias(self):
        """Update MoE router biases after backward pass.

        This must be called AFTER loss.backward() to maintain gradient
        checkpointing compatibility. The bias update is deferred from
        forward() to avoid state changes during checkpoint recomputation.
        """
        for block in self.transformer.h:
            ffn = getattr(block, 'ffn', getattr(block, 'mlp', None))
            if ffn is not None and hasattr(ffn, 'router'):
                router = ffn.router
                if hasattr(router, 'update_bias_from_last_forward'):
                    router.update_bias_from_last_forward()

    @torch.no_grad()
    def generate(
        self,
        idx: Optional[torch.Tensor] = None,
        max_new_tokens: Optional[int] = None,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        prompt=None,
        gen_length: Optional[int] = None,
        use_memory: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Generate tokens autoregressively.

        Args:
            idx: Input token indices [B, S]
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            prompt: Alternative name for idx
            gen_length: Alternative name for max_new_tokens
            use_memory: Whether to use persistent memory states during generation

        Returns:
            (generated_tokens, None)
        """
        if idx is None and prompt is not None:
            idx = prompt
        if idx is None:
            raise TypeError("SWAMLAModel.generate requires either 'idx' or 'prompt'.")

        tokens_to_generate = (
            max_new_tokens if max_new_tokens is not None else gen_length if gen_length is not None else 20
        )
        self.eval()

        # Initialize memory states for generation if model uses neural memory
        memory_states = None
        has_memory = self.config.use_neural_memory and use_memory

        for _ in range(tokens_to_generate):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]

            if has_memory:
                logits, _, memory_states = self(
                    idx_cond,
                    memory_states=memory_states,
                    return_memory_states=True
                )
            else:
                logits, _ = self(idx_cond)

            logits = logits[:, -1, :] / max(temperature, 1e-6)
            if top_k is not None:
                values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < values[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx, None

    def configure_optimizers(
        self,
        weight_decay: float,
        learning_rate: float,
        betas,
        device_type: str,
        optimizer_type: Optional[str] = None,
        **kwargs,
    ):
        """
        Configure optimizer for the SWA-MLA model.

        This is handled by the training script - optimizer configuration
        moved to train.py for standalone operation.
        """
        # Optimizer configuration is now in train.py
        raise NotImplementedError("Use the training script's optimizer configuration")


def _create_mla_block_config(config: SWAMLAConfig):
    """Create a lightweight config object for MLABlock based on the hybrid config."""

    @dataclass
    class _Config:
        n_embd: int = config.n_embd
        n_head: int = config.n_head
        dropout: float = config.dropout
        bias: bool = config.bias
        use_gradient_checkpointing: bool = config.use_gradient_checkpointing
        use_dyt: bool = config.use_dyt
        dyt_alpha_init: float = config.dyt_alpha_init
        q_lora_rank: int = config.q_lora_rank
        kv_lora_rank: int = config.kv_lora_rank
        qk_nope_head_dim: int = config.qk_nope_head_dim
        qk_rope_head_dim: int = config.qk_rope_head_dim
        v_head_dim: int = config.v_head_dim
        attn_impl: str = config.attn_impl
        world_size: int = config.world_size
        dropout_p: float = config.dropout
        max_seq_len: int = config.block_size
        original_max_seq_len: int = config.block_size
        rope_scaling: Optional[Dict[str, float]] = config.rope_scaling
        rope_theta: float = config.rope_theta
        rope_factor: float = config.rope_factor
        mscale: float = config.mscale
        use_fp8: bool = config.use_fp8
        fp8_mla_params: bool = config.fp8_mla_params
        fp8_tile_size: int = config.fp8_tile_size
        use_te_fp8: bool = config.use_te_fp8
        # Neural memory parameters (for MLAMemoryBlock)
        memory_dim: int = config.memory_dim
        memory_depth: int = config.memory_depth
        # MoE parameters
        use_moe: bool = config.use_moe
        n_experts: int = config.n_experts
        n_shared_experts: int = config.n_shared_experts
        n_activated: int = config.n_activated
        expert_dim: int = config.expert_dim
        router_z_loss_coef: float = config.router_z_loss_coef

    return _Config()


def create_swa_mla_model(
    size: str = "base",
    vocab_size: int = 50304,
    block_size: int = 4096,
    dropout: float = 0.0,
    **kwargs,
) -> SWAMLAModel:
    size = size.lower()
    presets = {
        "small": dict(n_layer=12, n_embd=1024, n_head=16),    # 768/12=64 head_dim
        "base": dict(n_layer=24, n_embd=1536, n_head=16),    # 1536/16=96 head_dim
        "large": dict(n_layer=28, n_embd=2048, n_head=16),   # 2048/16=128 head_dim
        "xl": dict(n_layer=32, n_embd=4096, n_head=32),      # 4096/32=128 head_dim
        # MoE presets - params are total, active params are much smaller
        "moe-1b": dict(  # ~770M total, ~250M active
            n_layer=12, n_embd=768, n_head=12,
            swa_layers_per_cycle=2, mla_layers_per_cycle=1,
            use_moe=True, n_experts=32, n_shared_experts=1, n_activated=3, expert_dim=768 * 3,
        ),
        "moe-2b": dict(  # ~1.5B total, ~400M active
            n_layer=18, n_embd=1024, n_head=16,
            swa_layers_per_cycle=2, mla_layers_per_cycle=1,
            use_moe=True, n_experts=32, n_shared_experts=1, n_activated=3, expert_dim=1024 * 3,
        ),
    }
    if size not in presets:
        raise ValueError(f"Unknown SWAMLA model size: {size}. Available: {list(presets.keys())}")

    cfg_kwargs = presets[size].copy()
    cfg_kwargs.update(dict(vocab_size=vocab_size, block_size=block_size, dropout=dropout))
    cfg_kwargs.update(kwargs)
    config = SWAMLAConfig(**cfg_kwargs)
    return SWAMLAModel(config)
