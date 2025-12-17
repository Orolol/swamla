"""Hybrid model interleaving GatedDeltaNet (O(n) linear attention) and MLA blocks with LatentMoE."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from normalization import RMSNorm, DynamicTanh
from mla_block import MLABlock
from positional_encoding import (
    precompute_freqs_cis,
    precompute_freqs_cis_with_linear_scaling,
)

# Gated DeltaNet (linear attention O(n))
try:
    from gated_deltanet import GatedDeltaNetBlock, GATED_DELTANET_AVAILABLE
except ImportError:
    GatedDeltaNetBlock = None
    GATED_DELTANET_AVAILABLE = False


@dataclass
class DeltaNetLayerConfig:
    """Configuration for DeltaNet attention blocks."""

    n_embd: int
    n_head: int
    block_size: int
    dropout: float
    bias: bool
    use_gradient_checkpointing: bool
    # DeltaNet latent compression
    deltanet_latent_dim: int = 0  # 0 = disabled, >0 = latent dimension
    deltanet_share_qk: bool = False  # Share Q and K projection


@dataclass
class SWAMLAConfig:
    """Configuration for the DeltaNet+MLA hybrid model with LatentMoE."""

    vocab_size: int = 50304
    block_size: int = 4096
    n_layer: int = 24
    n_head: int = 16
    n_embd: int = 1536
    dropout: float = 0.0
    bias: bool = False
    use_gradient_checkpointing: bool = True
    use_dyt: bool = False
    dyt_alpha_init: float = 0.5

    # Architecture: DeltaNet + MLA interleaving
    swa_layers_per_cycle: int = 2  # DeltaNet blocks per cycle (kept name for compatibility)
    mla_layers_per_cycle: int = 1  # MLA blocks per cycle
    rope_theta: float = 10000.0

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

    # Flash Attention for MLA blocks
    use_flash_attention: bool = True

    # DeltaNet configuration
    use_gated_deltanet: bool = True  # Use GatedDeltaNet for local attention (O(n) linear)
    deltanet_latent_dim: int = 0  # 0 = disabled, >0 = latent dimension for projections
    deltanet_share_qk: bool = False  # If True, Q and K share projection weights

    label_smoothing: float = 0.0

    # MoE configuration (only applied to MLA blocks)
    use_moe: bool = False
    n_experts: int = 32
    n_shared_experts: int = 1
    n_activated: int = 3  # routed experts per token (+ shared = 4 total active)
    expert_dim: Optional[int] = None  # defaults to n_embd if None
    router_z_loss_coef: float = 0.001

    # LatentMoE configuration (NVIDIA Nemotron-3 style)
    use_latent_moe: bool = False
    latent_ratio: int = 4  # d_model / latent_dim ratio
    latent_dim: Optional[int] = None  # Explicit latent dim (overrides latent_ratio)
    latent_n_experts: Optional[int] = None  # Override: total experts for LatentMoE
    latent_n_activated: Optional[int] = None  # Override: activated experts for LatentMoE
    latent_preserve_expert_dim: bool = False  # If True, keep full expert_dim

    def __post_init__(self) -> None:
        if self.expert_dim is None:
            self.expert_dim = self.n_embd
        if self.swa_layers_per_cycle < 0 or self.mla_layers_per_cycle < 0:
            raise ValueError("Layer counts per cycle must be non-negative")
        if self.swa_layers_per_cycle + self.mla_layers_per_cycle == 0:
            raise ValueError("At least one DeltaNet or MLA layer per cycle is required")


class SWAMLAModel(nn.Module):
    """DeltaNet-MLA hybrid model with LatentMoE."""

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
        deltanet_per_cycle = config.swa_layers_per_cycle

        for layer_idx in range(config.n_layer):
            # Calculate scaling factor for this layer: 0.5x at first layer to 2.0x at last layer
            if config.n_layer > 1:
                scale_factor = 0.5 + 1.5 * (layer_idx / (config.n_layer - 1))
            else:
                scale_factor = 1.0

            position_in_cycle = layer_idx % cycle_len
            if position_in_cycle < deltanet_per_cycle:
                # DeltaNet Layer (O(n) linear attention)
                if GatedDeltaNetBlock is None:
                    raise RuntimeError("GatedDeltaNet not available. Install fla: pip install fla")

                layer_config = DeltaNetLayerConfig(
                    n_embd=config.n_embd,
                    n_head=config.n_head,
                    block_size=config.block_size,
                    dropout=config.dropout,
                    bias=config.bias,
                    use_gradient_checkpointing=config.use_gradient_checkpointing,
                    deltanet_latent_dim=config.deltanet_latent_dim,
                    deltanet_share_qk=config.deltanet_share_qk,
                )
                block = GatedDeltaNetBlock(layer_config)
            else:
                # MLA Layer
                scaled_rank = int(config.kv_lora_rank * scale_factor)
                scaled_rank = max(1, scaled_rank)

                mla_config = _create_mla_block_config(config)
                mla_config.kv_lora_rank = scaled_rank
                block = MLABlock(mla_config, layer_id=layer_idx)

            block.use_checkpoint = config.use_gradient_checkpointing
            self.transformer.h.append(block)

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight = self.transformer.wte.weight

        self.apply(self._init_weights)
        self.param_count = sum(p.numel() for p in self.parameters())

        # Count DeltaNet and MLA blocks
        deltanet_count = sum(1 for block in self.transformer.h if GatedDeltaNetBlock is not None and isinstance(block, GatedDeltaNetBlock))
        mla_count = sum(1 for block in self.transformer.h if isinstance(block, MLABlock))

        # Calculate active parameters for MoE models
        active_param_count = None
        if config.use_moe:
            def is_moe_layer(module):
                return module.__class__.__name__ in ('MoELayer', 'LatentMoELayer')

            moe_active_params = 0
            all_moe_params = 0

            for name, module in self.named_modules():
                if is_moe_layer(module):
                    layer_params = sum(p.numel() for p in module.parameters())
                    all_moe_params += layer_params

                    for expert in module.shared_experts:
                        moe_active_params += sum(p.numel() for p in expert.parameters())
                    total_routed_params = sum(p.numel() for p in module.experts.parameters())
                    active_routed_params = total_routed_params * module.n_activated // module.n_experts
                    moe_active_params += active_routed_params
                    moe_active_params += sum(p.numel() for p in module.router.parameters())
                    if hasattr(module, 'down_proj'):
                        moe_active_params += sum(p.numel() for p in module.down_proj.parameters())
                    if hasattr(module, 'up_proj'):
                        moe_active_params += sum(p.numel() for p in module.up_proj.parameters())

            non_moe_params = self.param_count - all_moe_params
            active_param_count = non_moe_params + moe_active_params

        model_type = "DeltaNet-MLA"
        if config.use_moe:
            model_type += "-MoE"

        print(f"{model_type} Model - Number of parameters: {self.param_count / 1e6:.2f}M")
        if active_param_count is not None:
            print(f"  - Active parameters per forward: {active_param_count / 1e6:.2f}M ({100 * active_param_count / self.param_count:.1f}%)")
        print(f"  - {deltanet_count} DeltaNet blocks, {mla_count} MLA blocks")

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
    ):
        """Forward pass.

        Args:
            idx: Input token indices [B, S]
            targets: Target token indices for loss computation [B, S]
            return_all_logits: If True, return logits for all positions

        Returns:
            (logits, loss) tuple
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

        for block in self.transformer.h:
            if GatedDeltaNetBlock is not None and isinstance(block, GatedDeltaNetBlock):
                x = block(x)
            elif isinstance(block, MLABlock):
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
            logits = self.lm_head(x)
            loss = None
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None

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
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Generate tokens autoregressively.

        Args:
            idx: Input token indices [B, S]
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            prompt: Alternative name for idx
            gen_length: Alternative name for max_new_tokens

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

        for _ in range(tokens_to_generate):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)

            logits = logits[:, -1, :] / max(temperature, 1e-6)
            if top_k is not None:
                values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < values[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx, None


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
        # MoE parameters
        use_moe: bool = config.use_moe
        n_experts: int = config.n_experts
        n_shared_experts: int = config.n_shared_experts
        n_activated: int = config.n_activated
        expert_dim: int = config.expert_dim
        router_z_loss_coef: float = config.router_z_loss_coef
        # LatentMoE parameters
        use_latent_moe: bool = config.use_latent_moe
        latent_ratio: int = config.latent_ratio
        latent_dim: Optional[int] = config.latent_dim
        latent_n_experts: Optional[int] = config.latent_n_experts
        latent_n_activated: Optional[int] = config.latent_n_activated
        latent_preserve_expert_dim: bool = config.latent_preserve_expert_dim
        # Flash Attention for MLA
        use_flash_attention: bool = config.use_flash_attention

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
            n_layer=12, n_embd=1024, n_head=12,
            swa_layers_per_cycle=2, mla_layers_per_cycle=1,
            use_moe=True, n_experts=32, n_shared_experts=1, n_activated=2, expert_dim=512 * 2,
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
    # Pop compile_mode if it exists, as it's not part of SWAMLAConfig
    cfg_kwargs.pop("compile_mode", None)
    config = SWAMLAConfig(**cfg_kwargs)
    return SWAMLAModel(config)
