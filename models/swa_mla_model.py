"""Hybrid model interleaving GatedDeltaNet (O(n) linear attention) and MLA blocks with LatentMoE."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Dict, Tuple, List

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
    local_layers_per_cycle: int = 2  # DeltaNet blocks per cycle (local attention)
    mla_layers_per_cycle: int = 1  # MLA blocks per cycle (global attention)
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

    # Variable-length attention (PyTorch 2.10+)
    # Eliminates padding waste for packed sequences with document boundaries
    use_varlen_attn: bool = False

    # Custom Triton MLA kernel (H100 compatible alternative to FA2)
    # Use this when FA2 causes CUDA graph issues with torch.compile on H100
    use_triton_mla: bool = False

    # Use fused Triton kernels for SwiGLU and RMSNorm (15-25% speedup)
    use_triton_kernels: bool = True

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

    # FP8 training
    use_te_fp8: bool = False  # Legacy: Use TE Linear layers (kept for checkpoint compat, now a no-op)
    fp8_backend: str = "none"  # "none", "native", "te", or "auto"

    # Engram: Conditional Memory via Scalable N-gram Lookup
    # Complements MoE by performing O(1) lookups in N-gram embedding tables
    use_engram: bool = False
    engram_layers: List[int] = field(default_factory=lambda: [2, 6])  # Layers with Engram
    engram_d_mem: int = 512  # Memory embedding dimension
    engram_n_hash_heads: int = 8  # Hash heads per N-gram order (paper: K=8)
    engram_ngram_orders: List[int] = field(default_factory=lambda: [2, 3])  # N-gram orders
    engram_conv_kernel: int = 4  # Causal conv kernel size
    engram_table_sizes: Optional[Dict[Tuple[int, int], int]] = None  # Custom table sizes
    engram_lr_multiplier: float = 5.0  # LR multiplier for Engram embeddings

    # μP (Maximal Update Parametrization)
    use_mup: bool = False
    mup_base_width: int = 256  # Reference width for LR scaling
    mup_output_mult: float = 1.0  # Output logits multiplier

    # Progressive Training
    use_progressive: bool = False
    progressive_schedule: str = "512:500M,1024:2B,2048:inf"  # seq_len:tokens schedule

    # EMA (Exponential Moving Average)
    use_ema: bool = False
    ema_decay: float = 0.9999  # EMA decay factor

    def __post_init__(self) -> None:
        if self.expert_dim is None:
            self.expert_dim = self.n_embd
        if self.local_layers_per_cycle < 0 or self.mla_layers_per_cycle < 0:
            raise ValueError("Layer counts per cycle must be non-negative")
        if self.local_layers_per_cycle + self.mla_layers_per_cycle == 0:
            raise ValueError("At least one DeltaNet or MLA layer per cycle is required")

    @property
    def swa_layers_per_cycle(self) -> int:
        """Deprecated alias for local_layers_per_cycle. Use local_layers_per_cycle instead."""
        import warnings
        warnings.warn(
            "swa_layers_per_cycle is deprecated, use local_layers_per_cycle instead",
            DeprecationWarning,
            stacklevel=2
        )
        return self.local_layers_per_cycle


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

        # Pre-allocate causal mask for the maximum sequence length
        # This avoids repeated tensor allocation in forward pass
        causal_mask = torch.triu(
            torch.full((config.block_size, config.block_size), float("-inf")),
            diagonal=1
        )
        self.register_buffer("causal_mask", causal_mask, persistent=False)

        cycle_len = config.local_layers_per_cycle + config.mla_layers_per_cycle
        deltanet_per_cycle = config.local_layers_per_cycle

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

        # Count Engram modules
        engram_count = sum(
            1 for block in self.transformer.h
            if isinstance(block, MLABlock) and getattr(block, 'engram', None) is not None
        )
        engram_params = sum(
            block.engram.get_num_params()
            for block in self.transformer.h
            if isinstance(block, MLABlock) and getattr(block, 'engram', None) is not None
        )

        model_type = "DeltaNet-MLA"
        if config.use_moe:
            model_type += "-MoE"
        if config.use_engram:
            model_type += "-Engram"

        print(f"{model_type} Model - Number of parameters: {self.param_count / 1e6:.2f}M")
        if active_param_count is not None:
            print(f"  - Active parameters per forward: {active_param_count / 1e6:.2f}M ({100 * active_param_count / self.param_count:.1f}%)")
        print(f"  - {deltanet_count} DeltaNet blocks, {mla_count} MLA blocks")
        if engram_count > 0:
            print(f"  - {engram_count} Engram modules ({engram_params / 1e6:.2f}M params)")

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
        position_ids: Optional[torch.Tensor] = None,
        attention_mask_2d: Optional[torch.Tensor] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
    ):
        """Forward pass.

        Args:
            idx: Input token indices [B, S]
            targets: Target token indices for loss computation [B, S]
            return_all_logits: If True, return logits for all positions
            position_ids: Explicit position IDs for RoPE [B, S].
                If provided, allows physical position to differ from logical position.
                Used for WeDLM topological reordering.
            attention_mask_2d: Custom 2D attention mask [S, S] or [B, S, S].
                If provided, overrides the default causal mask.
                Used for WeDLM dual-stream masking.
            cu_seqlens: Cumulative sequence lengths for varlen_attn [num_docs + 1].
                Used for variable-length attention without padding waste.
            max_seqlen: Maximum sequence length in the batch for varlen_attn.

        Returns:
            (logits, loss) tuple
        """
        device = idx.device
        b, t = idx.size()

        # For WeDLM dual-stream, sequence length can be 2x block_size
        # but position_ids (logical positions) should stay within block_size
        # Note: We use t (physical seq length) for freqs_cis slicing to avoid
        # GPU->CPU sync (.item()) that would break CUDAGraphs.
        # Position validation is done at data preparation time, not here.
        if position_ids is None and t > self.config.block_size:
            raise ValueError(
                f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
            )
        max_pos = t

        tok_emb = self.transformer.wte(idx)
        x = self.transformer.drop(tok_emb)

        # Build attention mask
        if attention_mask_2d is not None:
            # Use provided 2D mask (for WeDLM dual-stream)
            # Convert boolean mask to float mask with -inf for False
            if attention_mask_2d.dtype == torch.bool:
                attn_mask = torch.where(
                    attention_mask_2d,
                    torch.tensor(0.0, device=device),
                    torch.tensor(float("-inf"), device=device),
                )
            else:
                attn_mask = attention_mask_2d
            # Ensure correct shape [S, S] for MLA
            if attn_mask.dim() == 3:
                # [B, S, S] -> use first batch as template (assuming same for all)
                attn_mask = attn_mask[0]
        elif t > 1:
            # Use cached causal mask (slice to current sequence length)
            # Move to device on-demand to avoid VRAM duplication in DDP
            attn_mask = self.causal_mask[:t, :t].to(device, non_blocking=True)
        else:
            attn_mask = None

        # Move freqs_cis to the correct device on-demand to avoid VRAM duplication in DDP
        freqs_cis = self.freqs_cis[:max_pos].to(device, non_blocking=True).detach()

        for block in self.transformer.h:
            if GatedDeltaNetBlock is not None and isinstance(block, GatedDeltaNetBlock):
                # GatedDeltaNet: pass position_ids if using WeDLM adapter
                if position_ids is not None and hasattr(block, 'forward_with_positions'):
                    x = block.forward_with_positions(x, position_ids)
                else:
                    x = block(x)
            elif isinstance(block, MLABlock):
                # MLA: pass position_ids for WeDLM, input_ids for Engram, and varlen metadata
                x = block(x, 0, freqs_cis, attn_mask, position_ids=position_ids, input_ids=idx, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)
            else:
                # Fallback for any other block type
                x = block(x, 0, freqs_cis, attn_mask)

        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            # Use built-in label_smoothing parameter (PyTorch 1.10+)
            # This is more efficient than manual implementation
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-100,
                label_smoothing=self.config.label_smoothing,
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

    def set_engram_tokenizer_compression(self, tokenizer):
        """Set tokenizer compression for all Engram modules.

        This must be called after model creation if using Engram.
        The tokenizer compression maps original token IDs to canonical
        IDs for consistent N-gram hashing.

        Args:
            tokenizer: Tokenizer with get_vocab() method
        """
        from engram import TokenizerCompression

        compression = TokenizerCompression.from_tokenizer(tokenizer)
        engram_count = 0

        for block in self.transformer.h:
            if isinstance(block, MLABlock) and getattr(block, 'has_engram', False):
                block.engram.set_tokenizer_compression(compression)
                engram_count += 1

        if engram_count > 0:
            print(f"Engram: Set tokenizer compression for {engram_count} layers")
            print(f"  - Original vocab: {compression.original_vocab_size:,}")
            print(f"  - Compressed vocab: {compression.compressed_vocab_size:,} "
                  f"({100 * compression.compressed_vocab_size / compression.original_vocab_size:.1f}%)")

    def get_engram_modules(self):
        """Get all Engram modules in the model.

        Returns:
            List of (layer_id, engram_module) tuples
        """
        engram_modules = []
        for block in self.transformer.h:
            if isinstance(block, MLABlock) and getattr(block, 'has_engram', False):
                engram_modules.append((block.layer_id, block.engram))
        return engram_modules

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
        # Engram parameters
        use_engram: bool = config.use_engram
        engram_layers: List[int] = field(default_factory=lambda: config.engram_layers.copy())
        engram_d_mem: int = config.engram_d_mem
        engram_n_hash_heads: int = config.engram_n_hash_heads
        engram_ngram_orders: List[int] = field(default_factory=lambda: config.engram_ngram_orders.copy())
        engram_conv_kernel: int = config.engram_conv_kernel
        engram_table_sizes: Optional[Dict[Tuple[int, int], int]] = config.engram_table_sizes

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
            local_layers_per_cycle=2, mla_layers_per_cycle=1,
            use_moe=True, n_experts=32, n_shared_experts=1, n_activated=2, expert_dim=512 * 2,
        ),
        "moe-2b": dict(  # ~1.5B total, ~400M active
            n_layer=18, n_embd=1024, n_head=16,
            local_layers_per_cycle=2, mla_layers_per_cycle=1,
            use_moe=True, n_experts=32, n_shared_experts=1, n_activated=3, expert_dim=1024 * 3,
        ),
        # Engram+MoE preset - MoE with N-gram conditional memory
        "engram-moe-1b": dict(  # MoE + Engram for N-gram memory lookup
            n_layer=12, n_embd=1024, n_head=12,
            local_layers_per_cycle=2, mla_layers_per_cycle=1,
            use_moe=True, n_experts=64, n_shared_experts=1, n_activated=2, expert_dim=512 * 2,
            use_engram=True,
            engram_layers=[2, 6],  # Apply Engram at layers 2 and 6
            engram_d_mem=512,
            engram_n_hash_heads=8,  # Paper: K=8 hash heads for collision attenuation
            engram_ngram_orders=[2, 3],
        ),
        # μP (Maximal Update Parametrization) preset with LatentMoE and Engram
        "mup-1b": dict(
            n_layer=12,
            n_embd=1024,
            n_head=16,
            use_moe=True,
            n_experts=64,
            n_shared_experts=1,
            n_activated=2,
            use_latent_moe=True,
            latent_ratio=4,
            use_engram=True,
            engram_layers=[2, 6],
            engram_d_mem=512,
            # μP settings
            use_mup=True,
            mup_base_width=256,
        ),
    }
    if size not in presets:
        raise ValueError(f"Unknown SWAMLA model size: {size}. Available: {list(presets.keys())}")

    cfg_kwargs = presets[size].copy()
    cfg_kwargs.update(dict(vocab_size=vocab_size, block_size=block_size, dropout=dropout))
    cfg_kwargs.update(kwargs)
    # Pop compile_mode if it exists, as it's not part of SWAMLAConfig
    cfg_kwargs.pop("compile_mode", None)
    cfg_kwargs.pop("use_tensorboard", None)
    cfg_kwargs.pop("use_fp8", None)
    # WeDLM parameters are handled in train.py, not in model config
    cfg_kwargs.pop("use_wedlm", None)
    cfg_kwargs.pop("wedlm_block_size", None)
    cfg_kwargs.pop("wedlm_min_mask_ratio", None)
    cfg_kwargs.pop("wedlm_max_mask_ratio", None)
    cfg_kwargs.pop("wedlm_ar_loss_weight", None)
    cfg_kwargs.pop("wedlm_mask_token_id", None)
    # resume_from is handled in train.py, not in model config
    cfg_kwargs.pop("resume_from", None)
    cfg_kwargs.pop("resume_from_hf", None)
    # Backward compatibility: map swa_layers_per_cycle to local_layers_per_cycle
    if "swa_layers_per_cycle" in cfg_kwargs:
        import warnings
        warnings.warn(
            "swa_layers_per_cycle is deprecated, use local_layers_per_cycle instead",
            DeprecationWarning,
            stacklevel=2
        )
        if "local_layers_per_cycle" not in cfg_kwargs:
            cfg_kwargs["local_layers_per_cycle"] = cfg_kwargs.pop("swa_layers_per_cycle")
        else:
            cfg_kwargs.pop("swa_layers_per_cycle")
    config = SWAMLAConfig(**cfg_kwargs)
    return SWAMLAModel(config)
