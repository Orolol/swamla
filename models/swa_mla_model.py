"""Hybrid model interleaving Sliding-Window Attention (SWA) and MLA/MLA-Selective blocks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

from normalization import RMSNorm, DynamicTanh
from attention import CausalSelfAttention
from mlp import MLP
from mla_block import MLABlock
# from mla_selective_fast import MLASelectiveFast as MLASelective
from positional_encoding import (
    precompute_freqs_cis,
    precompute_freqs_cis_with_linear_scaling,
)
# from optimizers import configure_optimizer_for_gpt


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
        self.attn = CausalSelfAttention(config)
        self.mlp = MLP(config)
        self.use_checkpoint = config.use_gradient_checkpointing

    def _attn_block(self, x: torch.Tensor) -> torch.Tensor:
        return self.attn(self.norm1(x))

    def _mlp_block(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.norm2(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        use_checkpoint = self.use_checkpoint and self.training
        if use_checkpoint:
            attn_out = checkpoint.checkpoint(self._attn_block, x, use_reentrant=False)
        else:
            attn_out = self._attn_block(x)
        x = x + attn_out

        if use_checkpoint:
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

    # MLA Selective specific parameters
    use_mla_selective: bool = False  # Use MLA Selective instead of standard MLA
    selection_head_idx: int = 0  # Which attention head to use for selection

    label_smoothing: float = 0.0

    def __post_init__(self) -> None:
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

    def _attn_block(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        return self.attn(self.norm1(x), start_pos, freqs_cis, mask)

    def _mlp_block(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.norm2(x))

    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        use_checkpoint = self.use_checkpoint and self.training

        if use_checkpoint:
            attn_out = checkpoint.checkpoint(self._attn_block, x, start_pos, freqs_cis, mask, use_reentrant=False)
        else:
            attn_out = self._attn_block(x, start_pos, freqs_cis, mask)
        x = x + attn_out

        if use_checkpoint:
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
            position_in_cycle = layer_idx % cycle_len
            if position_in_cycle < swa_per_cycle:
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
                    attention_window=config.swa_window,
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
                )
                block = SWALocalBlock(layer_config)
            else:
                # Choose between MLA Selective or standard MLA based on config
                if config.use_mla_selective:
                    block = MLASelectiveBlock(config, layer_id=layer_idx)
                else:
                    mla_config = _create_mla_block_config(config)
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
        mla_standard_count = sum(1 for block in self.transformer.h if isinstance(block, MLABlock))

        model_type = "SWAMLA-Selective" if config.use_mla_selective else "SWAMLA"
        print(f"{model_type} Model - Number of parameters: {self.param_count / 1e6:.2f}M")
        if config.use_mla_selective:
            print(f"  - {swa_count} SWA blocks, {mla_selective_count} MLA Selective blocks")
        else:
            print(f"  - {swa_count} SWA blocks, {mla_standard_count} MLA blocks")

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None, return_all_logits: bool = False):
        device = idx.device
        b, t = idx.size()
        if t > self.config.block_size:
            raise ValueError(
                f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
            )

        tok_emb = self.transformer.wte(idx)
        x = self.transformer.drop(tok_emb)

        mask = None
        if t > 1:
            mask = torch.full((t, t), float("-inf"), device=device)
            mask = torch.triu(mask, diagonal=1)

        # Move freqs_cis to the correct device on-demand to avoid VRAM duplication in DDP
        freqs_cis = self.freqs_cis[:t].to(device, non_blocking=True).detach()

        for block in self.transformer.h:
            if isinstance(block, SWALocalBlock):
                x = block(x)
            elif isinstance(block, (MLASelectiveBlock, MLABlock)):
                x = block(x, 0, freqs_cis, mask)
            else:
                # Fallback for any other block type
                x = block(x, 0, freqs_cis, mask)

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
                    mask = (targets != -100).float()
                loss = (loss * mask.view(-1)).sum() / (mask.sum() + 1e-6)
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

        return logits, loss

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
        if idx is None and prompt is not None:
            idx = prompt
        if idx is None:
            raise TypeError("SWAMLAModel.generate requires either 'idx' or 'prompt'.")

        tokens_to_generate = (
            max_new_tokens if max_new_tokens is not None else gen_length if gen_length is not None else 20
        )
        self.eval()
        for _ in range(tokens_to_generate):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size :]
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
    }
    if size not in presets:
        raise ValueError(f"Unknown SWAMLA model size: {size}")

    cfg_kwargs = presets[size].copy()
    cfg_kwargs.update(dict(vocab_size=vocab_size, block_size=block_size, dropout=dropout))
    cfg_kwargs.update(kwargs)
    config = SWAMLAConfig(**cfg_kwargs)
    return SWAMLAModel(config)
