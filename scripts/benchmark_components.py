"""
Micro-benchmark profiling script for SWAMLA model components.

Profiles each model component individually (forward + backward) using the
production engram-moe-1b configuration with all features enabled:
  - 12 layers, 1024 embed dim, 12 heads
  - LatentMoE (64 experts, 2 activated, ratio=4)
  - Engram (layers 2,6, d_mem=512, K=8 hash heads, orders=[2,3])
  - FoPE (4 harmonics)
  - Value Embeddings
  - MLA: nope=128, rope=64, v=128, kv_lora_rank=256

Reports wall clock time (ms), peak VRAM (MB), and percentage of total.

Usage:
    python scripts/benchmark_components.py
    python scripts/benchmark_components.py --batch_size 4 --seq_len 2048
    python scripts/benchmark_components.py --warmup 20 --iterations 100
    python scripts/benchmark_components.py --preset tiny   # small config for quick test
"""

import argparse
import gc
import sys
import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Add models/ to path for local imports
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models"))


@dataclass
class BenchmarkResult:
    name: str
    time_ms: float
    peak_vram_mb: float
    pct_of_total: float = 0.0


def reset_peak_memory():
    """Reset CUDA peak memory stats."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()


def get_peak_memory_mb() -> float:
    """Get peak VRAM usage in MB since last reset."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        return torch.cuda.max_memory_allocated() / (1024 * 1024)
    return 0.0


def get_current_memory_mb() -> float:
    """Get current VRAM usage in MB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024 * 1024)
    return 0.0


def benchmark_fn(
    fn,
    warmup: int = 10,
    iterations: int = 50,
    device: torch.device = torch.device("cuda"),
) -> Tuple[float, float]:
    """
    Benchmark a callable using CUDA events for precise timing.

    Args:
        fn: Callable to benchmark (should include both forward and backward).
        warmup: Number of warmup iterations.
        iterations: Number of timed iterations.
        device: CUDA device.

    Returns:
        (avg_time_ms, peak_vram_mb) tuple.
    """
    # Warmup
    for _ in range(warmup):
        fn()

    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()

    # Measure peak VRAM during timed runs
    reset_peak_memory()
    baseline_mem = get_current_memory_mb()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    total_time = 0.0
    for _ in range(iterations):
        torch.cuda.synchronize()
        start_event.record()
        fn()
        end_event.record()
        torch.cuda.synchronize()
        total_time += start_event.elapsed_time(end_event)

    peak_vram = get_peak_memory_mb() - baseline_mem
    avg_time = total_time / iterations

    return avg_time, max(peak_vram, 0.0)


def create_production_config():
    """Create config matching production engram-moe-1b with all features."""
    from swa_mla_model import SWAMLAConfig

    return SWAMLAConfig(
        vocab_size=50304,
        block_size=2048,
        n_layer=12,
        n_embd=1024,
        n_head=12,
        dropout=0.0,
        bias=False,
        use_gradient_checkpointing=False,  # Disabled for accurate timing
        # MLA config (production dims)
        q_lora_rank=0,
        kv_lora_rank=256,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        # DeltaNet config
        local_layers_per_cycle=2,
        mla_layers_per_cycle=1,
        use_gated_deltanet=True,
        deltanet_latent_dim=0,
        deltanet_share_qk=False,
        # LatentMoE (production)
        use_moe=True,
        n_experts=64,
        n_shared_experts=1,
        n_activated=2,
        expert_dim=512 * 2,
        use_latent_moe=True,
        latent_ratio=4,
        # Engram (production)
        use_engram=True,
        engram_layers=[2, 6],
        engram_d_mem=512,
        engram_n_hash_heads=8,
        engram_ngram_orders=[2, 3],
        # FoPE (production)
        fope_enabled=True,
        fope_n_harmonics=4,
        fope_floor_ratio=0.1,
        fope_coef_init_std=0.3,
        # Value Embeddings
        use_value_embeds=True,
        ve_gate_dim=32,
        # Residual scalars
        use_residual_scalars=True,
        # Disable features that need special hardware / slow down benchmark
        use_flash_attention=False,
        use_triton_kernels=False,
        use_cudnn_sdpa=False,
        use_triton_mla=False,
        yarn_enabled=False,
    )


def create_tiny_config():
    """Create a tiny config for quick testing (~4GB VRAM)."""
    from swa_mla_model import SWAMLAConfig

    return SWAMLAConfig(
        vocab_size=50304,
        block_size=1024,
        n_layer=4,
        n_embd=512,
        n_head=8,
        dropout=0.0,
        bias=False,
        use_gradient_checkpointing=False,
        q_lora_rank=0,
        kv_lora_rank=128,
        qk_nope_head_dim=48,
        qk_rope_head_dim=16,
        v_head_dim=48,
        use_gated_deltanet=True,
        use_moe=False,
        use_engram=False,
        use_flash_attention=False,
        use_triton_kernels=False,
        use_cudnn_sdpa=False,
        use_value_embeds=False,
        fope_enabled=False,
        yarn_enabled=False,
        use_residual_scalars=False,
    )


# ---------------------------------------------------------------------------
# Individual component benchmarks
# ---------------------------------------------------------------------------

def benchmark_full_model(
    config, batch_size: int, seq_len: int, device: torch.device, warmup: int, iterations: int
) -> BenchmarkResult:
    """Benchmark full model forward + backward."""
    from swa_mla_model import SWAMLAModel

    model = SWAMLAModel(config).to(device).to(torch.bfloat16)
    model.train()

    idx = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)
    targets = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)

    def fn():
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits, loss = model(idx, targets=targets)
            if config.use_moe:
                aux_loss = model.get_moe_aux_loss()
                loss = loss + aux_loss
        loss.backward()
        model.zero_grad(set_to_none=True)

    avg_time, peak_vram = benchmark_fn(fn, warmup=warmup, iterations=iterations, device=device)

    del model, idx, targets
    gc.collect()
    torch.cuda.empty_cache()

    return BenchmarkResult("Full Model (fwd+bwd)", avg_time, peak_vram)


def benchmark_mla_attention(
    config, batch_size: int, seq_len: int, device: torch.device, warmup: int, iterations: int
) -> BenchmarkResult:
    """Benchmark MLA attention forward + backward in isolation."""
    from mla import MLA

    @dataclass
    class _MLAConfig:
        n_embd: int = config.n_embd
        n_head: int = config.n_head
        q_lora_rank: int = config.q_lora_rank
        kv_lora_rank: int = config.kv_lora_rank
        qk_nope_head_dim: int = config.qk_nope_head_dim
        qk_rope_head_dim: int = config.qk_rope_head_dim
        v_head_dim: int = config.v_head_dim
        bias: bool = config.bias
        dropout: float = config.dropout
        world_size: int = 1
        max_seq_len: int = config.block_size
        attn_impl: str = "absorb"
        use_flash_attention: bool = False
        use_triton_mla: bool = False
        use_flex_attention: bool = False
        use_cudnn_sdpa: bool = False
        rope_theta: float = 10000.0
        rope_factor: float = 1.0
        mscale: float = 1.0
        fope_enabled: bool = config.fope_enabled
        fope_n_harmonics: int = config.fope_n_harmonics
        fope_floor_ratio: float = config.fope_floor_ratio
        fope_coef_init_std: float = config.fope_coef_init_std
        use_value_embeds: bool = config.use_value_embeds
        ve_gate_dim: int = config.ve_gate_dim
        vocab_size: int = config.vocab_size
        yarn_enabled: bool = False

    mla_cfg = _MLAConfig()
    mla = MLA(mla_cfg, layer_id=0).to(device).to(torch.bfloat16)
    mla.train()

    x = torch.randn(batch_size, seq_len, config.n_embd, device=device, dtype=torch.bfloat16, requires_grad=True)
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)

    def fn():
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = mla(x, start_pos=0, input_ids=input_ids)
        out.sum().backward()
        mla.zero_grad(set_to_none=True)
        if x.grad is not None:
            x.grad = None

    avg_time, peak_vram = benchmark_fn(fn, warmup=warmup, iterations=iterations, device=device)

    del mla, x, input_ids
    gc.collect()
    torch.cuda.empty_cache()

    return BenchmarkResult("MLA Attention (fwd+bwd)", avg_time, peak_vram)


def benchmark_gated_deltanet(
    config, batch_size: int, seq_len: int, device: torch.device, warmup: int, iterations: int
) -> BenchmarkResult:
    """Benchmark GatedDeltaNet attention forward + backward in isolation."""
    from gated_deltanet import GatedDeltaNet
    from swa_mla_model import DeltaNetLayerConfig

    dn_config = DeltaNetLayerConfig(
        n_embd=config.n_embd,
        n_head=config.n_head,
        block_size=config.block_size,
        dropout=config.dropout,
        bias=config.bias,
        use_gradient_checkpointing=False,
        deltanet_latent_dim=config.deltanet_latent_dim,
        deltanet_share_qk=config.deltanet_share_qk,
        use_value_embeds=config.use_value_embeds,
        ve_gate_dim=config.ve_gate_dim,
        vocab_size=config.vocab_size,
        use_triton_kernels=False,
        layer_id=0,
    )

    dn = GatedDeltaNet(dn_config).to(device).to(torch.bfloat16)
    dn.train()

    x = torch.randn(batch_size, seq_len, config.n_embd, device=device, dtype=torch.bfloat16, requires_grad=True)
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)

    def fn():
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = dn(x, input_ids=input_ids)
        out.sum().backward()
        dn.zero_grad(set_to_none=True)
        if x.grad is not None:
            x.grad = None

    avg_time, peak_vram = benchmark_fn(fn, warmup=warmup, iterations=iterations, device=device)

    del dn, x, input_ids
    gc.collect()
    torch.cuda.empty_cache()

    return BenchmarkResult("GatedDeltaNet (fwd+bwd)", avg_time, peak_vram)


def benchmark_mlp(
    config, batch_size: int, seq_len: int, device: torch.device, warmup: int, iterations: int
) -> BenchmarkResult:
    """Benchmark dense SwiGLU MLP forward + backward (non-MoE baseline)."""
    from mlp import MLP

    @dataclass
    class _MLPConfig:
        n_embd: int = config.n_embd
        bias: bool = config.bias
        dropout: float = config.dropout
        use_triton_kernels: bool = False
        use_gradient_checkpointing: bool = False

    mlp = MLP(_MLPConfig()).to(device).to(torch.bfloat16)
    mlp.train()

    x = torch.randn(batch_size, seq_len, config.n_embd, device=device, dtype=torch.bfloat16, requires_grad=True)

    def fn():
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = mlp(x)
        out.sum().backward()
        mlp.zero_grad(set_to_none=True)
        if x.grad is not None:
            x.grad = None

    avg_time, peak_vram = benchmark_fn(fn, warmup=warmup, iterations=iterations, device=device)

    del mlp, x
    gc.collect()
    torch.cuda.empty_cache()

    return BenchmarkResult("MLP / SwiGLU dense (fwd+bwd)", avg_time, peak_vram)


def benchmark_latent_moe(
    config, batch_size: int, seq_len: int, device: torch.device, warmup: int, iterations: int
) -> BenchmarkResult:
    """Benchmark LatentMoE layer forward + backward."""
    from moe import create_moe_layer

    @dataclass
    class _MoEConfig:
        n_embd: int = config.n_embd
        bias: bool = config.bias
        dropout: float = config.dropout
        use_moe: bool = True
        use_latent_moe: bool = True
        n_experts: int = config.n_experts
        n_shared_experts: int = config.n_shared_experts
        n_activated: int = config.n_activated
        expert_dim: int = config.expert_dim
        latent_ratio: int = config.latent_ratio
        latent_dim: Optional[int] = config.latent_dim
        latent_n_experts: Optional[int] = config.latent_n_experts
        latent_n_activated: Optional[int] = config.latent_n_activated
        latent_preserve_expert_dim: bool = config.latent_preserve_expert_dim
        router_z_loss_coef: float = config.router_z_loss_coef
        use_triton_kernels: bool = False
        use_gradient_checkpointing: bool = False

    moe = create_moe_layer(_MoEConfig(), use_latent=True).to(device).to(torch.bfloat16)
    moe.train()

    x = torch.randn(batch_size, seq_len, config.n_embd, device=device, dtype=torch.bfloat16, requires_grad=True)

    def fn():
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = moe(x)
        out.sum().backward()
        moe.zero_grad(set_to_none=True)
        if x.grad is not None:
            x.grad = None

    avg_time, peak_vram = benchmark_fn(fn, warmup=warmup, iterations=iterations, device=device)

    del moe, x
    gc.collect()
    torch.cuda.empty_cache()

    return BenchmarkResult("LatentMoE (fwd+bwd)", avg_time, peak_vram)


def benchmark_engram(
    config, batch_size: int, seq_len: int, device: torch.device, warmup: int, iterations: int
) -> BenchmarkResult:
    """Benchmark Engram module forward + backward."""
    from engram import create_engram_for_config

    # Build an Engram-enabled config-like object
    @dataclass
    class _EngramConfig:
        use_engram: bool = True
        engram_layers: List[int] = field(default_factory=lambda: list(config.engram_layers))
        engram_d_mem: int = config.engram_d_mem
        engram_n_hash_heads: int = config.engram_n_hash_heads
        engram_ngram_orders: List[int] = field(default_factory=lambda: list(config.engram_ngram_orders))
        engram_conv_kernel: int = config.engram_conv_kernel
        engram_table_sizes: Optional[dict] = None
        n_embd: int = config.n_embd
        vocab_size: int = config.vocab_size

    ecfg = _EngramConfig()
    engram = create_engram_for_config(ecfg, layer_id=config.engram_layers[0])
    if engram is None:
        return BenchmarkResult("Engram (fwd+bwd)", 0.0, 0.0)
    engram = engram.to(device).to(torch.bfloat16)
    engram.train()

    x = torch.randn(batch_size, seq_len, config.n_embd, device=device, dtype=torch.bfloat16, requires_grad=True)
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)

    def fn():
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = engram(x, input_ids)
        out.sum().backward()
        engram.zero_grad(set_to_none=True)
        if x.grad is not None:
            x.grad = None

    avg_time, peak_vram = benchmark_fn(fn, warmup=warmup, iterations=iterations, device=device)

    del engram, x, input_ids
    gc.collect()
    torch.cuda.empty_cache()

    return BenchmarkResult("Engram (fwd+bwd)", avg_time, peak_vram)


def benchmark_rmsnorm(
    config, batch_size: int, seq_len: int, device: torch.device, warmup: int, iterations: int
) -> BenchmarkResult:
    """Benchmark RMSNorm forward + backward in isolation."""
    from normalization import RMSNorm

    norm = RMSNorm(config.n_embd).to(device).to(torch.bfloat16)
    norm.train()

    x = torch.randn(batch_size, seq_len, config.n_embd, device=device, dtype=torch.bfloat16, requires_grad=True)

    def fn():
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = norm(x)
        out.sum().backward()
        norm.zero_grad(set_to_none=True)
        if x.grad is not None:
            x.grad = None

    avg_time, peak_vram = benchmark_fn(fn, warmup=warmup, iterations=iterations, device=device)

    del norm, x
    gc.collect()
    torch.cuda.empty_cache()

    return BenchmarkResult("RMSNorm (fwd+bwd)", avg_time, peak_vram)


def benchmark_embedding(
    config, batch_size: int, seq_len: int, device: torch.device, warmup: int, iterations: int
) -> BenchmarkResult:
    """Benchmark Embedding + lm_head (output projection) forward + backward."""
    embed = nn.Embedding(config.vocab_size, config.n_embd).to(device).to(torch.bfloat16)
    lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False).to(device).to(torch.bfloat16)
    lm_head.weight = embed.weight

    idx = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)
    targets = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)

    def fn():
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            h = embed(idx)
            logits = lm_head(h)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        loss.backward()
        embed.zero_grad(set_to_none=True)
        lm_head.zero_grad(set_to_none=True)

    avg_time, peak_vram = benchmark_fn(fn, warmup=warmup, iterations=iterations, device=device)

    del embed, lm_head, idx, targets
    gc.collect()
    torch.cuda.empty_cache()

    return BenchmarkResult("Embed + lm_head (fwd+bwd)", avg_time, peak_vram)


def benchmark_fope(
    config, batch_size: int, seq_len: int, device: torch.device, warmup: int, iterations: int
) -> BenchmarkResult:
    """Benchmark FoPE (Fourier Position Embedding) forward + backward."""
    from positional_encoding import FoPE

    rope_dim = config.qk_rope_head_dim
    fope = FoPE(
        dim=rope_dim,
        max_seq_len=config.block_size,
        n_harmonics=config.fope_n_harmonics,
        floor_ratio=config.fope_floor_ratio,
        coef_init_std=config.fope_coef_init_std,
    ).to(device)

    x = torch.randn(
        batch_size, config.n_head, seq_len, rope_dim,
        device=device, dtype=torch.bfloat16, requires_grad=True,
    )

    def fn():
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = fope(x)
        out.sum().backward()
        fope.zero_grad(set_to_none=True)
        if x.grad is not None:
            x.grad = None

    avg_time, peak_vram = benchmark_fn(fn, warmup=warmup, iterations=iterations, device=device)

    del fope, x
    gc.collect()
    torch.cuda.empty_cache()

    return BenchmarkResult("FoPE (fwd+bwd)", avg_time, peak_vram)


def benchmark_rope(
    config, batch_size: int, seq_len: int, device: torch.device, warmup: int, iterations: int
) -> BenchmarkResult:
    """Benchmark RoPE application forward + backward."""
    from positional_encoding import RoPE

    rope_dim = config.qk_rope_head_dim
    rope = RoPE(rope_dim, max_seq_len=config.block_size).to(device)

    x = torch.randn(
        batch_size, config.n_head, seq_len, rope_dim,
        device=device, dtype=torch.bfloat16, requires_grad=True,
    )

    def fn():
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = rope(x)
        out.sum().backward()
        if x.grad is not None:
            x.grad = None

    avg_time, peak_vram = benchmark_fn(fn, warmup=warmup, iterations=iterations, device=device)

    del rope, x
    gc.collect()
    torch.cuda.empty_cache()

    return BenchmarkResult("RoPE (fwd+bwd)", avg_time, peak_vram)


def benchmark_deltanet_block(
    config, batch_size: int, seq_len: int, device: torch.device, warmup: int, iterations: int
) -> BenchmarkResult:
    """Benchmark a full GatedDeltaNetBlock (attn + MLP + norms) forward + backward."""
    from gated_deltanet import GatedDeltaNetBlock
    from swa_mla_model import DeltaNetLayerConfig

    dn_config = DeltaNetLayerConfig(
        n_embd=config.n_embd,
        n_head=config.n_head,
        block_size=config.block_size,
        dropout=config.dropout,
        bias=config.bias,
        use_gradient_checkpointing=False,
        deltanet_latent_dim=config.deltanet_latent_dim,
        deltanet_share_qk=config.deltanet_share_qk,
        use_value_embeds=config.use_value_embeds,
        ve_gate_dim=config.ve_gate_dim,
        vocab_size=config.vocab_size,
        use_triton_kernels=False,
        layer_id=0,
    )

    block = GatedDeltaNetBlock(dn_config).to(device).to(torch.bfloat16)
    block.train()

    x = torch.randn(batch_size, seq_len, config.n_embd, device=device, dtype=torch.bfloat16, requires_grad=True)
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)

    def fn():
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = block(x, input_ids=input_ids)
        out.sum().backward()
        block.zero_grad(set_to_none=True)
        if x.grad is not None:
            x.grad = None

    avg_time, peak_vram = benchmark_fn(fn, warmup=warmup, iterations=iterations, device=device)

    del block, x, input_ids
    gc.collect()
    torch.cuda.empty_cache()

    return BenchmarkResult("DeltaNet Block (full, fwd+bwd)", avg_time, peak_vram)


def benchmark_mla_block(
    config, batch_size: int, seq_len: int, device: torch.device, warmup: int, iterations: int
) -> BenchmarkResult:
    """Benchmark a full MLABlock (MLA attn + LatentMoE FFN + norms) forward + backward."""
    from mla_block import MLABlock
    from swa_mla_model import _create_mla_block_config
    from positional_encoding import precompute_freqs_cis

    mla_cfg = _create_mla_block_config(config)
    mla_cfg.use_gradient_checkpointing = False
    # Use a non-Engram layer for clean MLA+MoE measurement
    block = MLABlock(mla_cfg, layer_id=3).to(device).to(torch.bfloat16)
    block.use_checkpoint = False
    block.train()

    freqs_cis = precompute_freqs_cis(
        config.qk_rope_head_dim, config.block_size, config.rope_theta
    ).to(device)[:seq_len]

    x = torch.randn(batch_size, seq_len, config.n_embd, device=device, dtype=torch.bfloat16, requires_grad=True)
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)

    def fn():
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = block(x, 0, freqs_cis, None, input_ids=input_ids)
        out.sum().backward()
        block.zero_grad(set_to_none=True)
        if x.grad is not None:
            x.grad = None

    avg_time, peak_vram = benchmark_fn(fn, warmup=warmup, iterations=iterations, device=device)

    del block, freqs_cis, x, input_ids
    gc.collect()
    torch.cuda.empty_cache()

    return BenchmarkResult("MLA Block+MoE (full, fwd+bwd)", avg_time, peak_vram)


def benchmark_mla_block_with_engram(
    config, batch_size: int, seq_len: int, device: torch.device, warmup: int, iterations: int
) -> BenchmarkResult:
    """Benchmark MLABlock with Engram (MLA + MoE + Engram) forward + backward."""
    if not config.use_engram:
        return BenchmarkResult("MLA Block+MoE+Engram (fwd+bwd)", 0.0, 0.0)

    from mla_block import MLABlock
    from swa_mla_model import _create_mla_block_config
    from positional_encoding import precompute_freqs_cis

    mla_cfg = _create_mla_block_config(config)
    mla_cfg.use_gradient_checkpointing = False
    # Use an Engram layer
    engram_layer_id = config.engram_layers[0]
    block = MLABlock(mla_cfg, layer_id=engram_layer_id).to(device).to(torch.bfloat16)
    block.use_checkpoint = False
    block.train()

    freqs_cis = precompute_freqs_cis(
        config.qk_rope_head_dim, config.block_size, config.rope_theta
    ).to(device)[:seq_len]

    x = torch.randn(batch_size, seq_len, config.n_embd, device=device, dtype=torch.bfloat16, requires_grad=True)
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)

    def fn():
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            # Engram is called from SWAMLAModel.forward() before the block,
            # so simulate that here
            if getattr(block, 'has_engram', False):
                x_with_engram = x + block.engram(x, input_ids)
            else:
                x_with_engram = x
            out = block(x_with_engram, 0, freqs_cis, None, input_ids=input_ids)
        out.sum().backward()
        block.zero_grad(set_to_none=True)
        if x.grad is not None:
            x.grad = None

    avg_time, peak_vram = benchmark_fn(fn, warmup=warmup, iterations=iterations, device=device)

    del block, freqs_cis, x, input_ids
    gc.collect()
    torch.cuda.empty_cache()

    return BenchmarkResult("MLA Block+MoE+Engram (fwd+bwd)", avg_time, peak_vram)


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def print_results(results: List[BenchmarkResult], batch_size: int, seq_len: int, config):
    """Print a formatted table of benchmark results."""
    # Compute percentage of total model time
    total_result = next((r for r in results if r.name.startswith("Full Model")), None)
    total_time = total_result.time_ms if total_result else 1.0

    for r in results:
        if not r.name.startswith("Full Model"):
            r.pct_of_total = (r.time_ms / total_time) * 100.0

    # Print header
    features = []
    if config.use_moe:
        features.append("LatentMoE" if config.use_latent_moe else "MoE")
    if config.use_engram:
        features.append("Engram")
    if config.fope_enabled:
        features.append("FoPE")
    if config.use_value_embeds:
        features.append("VE")
    if config.use_residual_scalars:
        features.append("ResScalar")
    feature_str = " + ".join(features) if features else "vanilla"

    print()
    print("=" * 95)
    print(f"  SWAMLA Component Benchmark Results")
    print(f"  Config: {config.n_layer}L / {config.n_embd}d / {config.n_head}H | {feature_str}")
    print(f"  Batch: {batch_size} | Seq len: {seq_len} | dtype: bf16")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        total_vram = torch.cuda.get_device_properties(0).total_mem / (1024 ** 3)
        print(f"  GPU: {gpu_name} ({total_vram:.1f} GB)")
    if config.use_moe:
        print(f"  MoE: {config.n_experts} experts, {config.n_activated} activated, "
              f"ratio={config.latent_ratio}")
    print("=" * 95)
    print()
    print(f"  {'Component':<40} {'Time (ms)':>10} {'VRAM (MB)':>10} {'% of total':>12}")
    print(f"  {'-' * 40} {'-' * 10} {'-' * 10} {'-' * 12}")

    for r in results:
        if r.time_ms == 0.0 and r.peak_vram_mb == 0.0 and not r.name.startswith("Full"):
            continue  # Skip skipped benchmarks
        pct_str = f"{r.pct_of_total:>10.1f}%" if r.pct_of_total > 0 else f"{'baseline':>11}"
        print(f"  {r.name:<40} {r.time_ms:>10.2f} {r.peak_vram_mb:>10.1f} {pct_str}")

    print()

    # Summary
    component_results = [r for r in results if not r.name.startswith("Full Model") and r.time_ms > 0]
    if component_results:
        # Sort by time descending
        sorted_by_time = sorted(component_results, key=lambda r: r.time_ms, reverse=True)
        most_vram = max(component_results, key=lambda r: r.peak_vram_mb)

        print(f"  --- Bottleneck ranking (by time) ---")
        for i, r in enumerate(sorted_by_time[:5], 1):
            print(f"  {i}. {r.name}: {r.time_ms:.2f} ms ({r.pct_of_total:.1f}%)")

        print(f"\n  Largest VRAM: {most_vram.name} ({most_vram.peak_vram_mb:.1f} MB)")

        # Isolated component sum (exclude composite blocks)
        isolated = [r for r in component_results if "Block" not in r.name]
        sum_isolated = sum(r.time_ms for r in isolated)
        print(f"\n  Sum of isolated components: {sum_isolated:.2f} ms")
        if total_result:
            overhead = total_time - sum_isolated
            print(f"  Full model time:            {total_time:.2f} ms")
            print(f"  Overhead (scheduling etc.):  {overhead:.2f} ms ({overhead / total_time * 100:.1f}%)")

    print()
    print("=" * 95)


def main():
    parser = argparse.ArgumentParser(description="Benchmark SWAMLA model components")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size (default: 2)")
    parser.add_argument("--seq_len", type=int, default=512, help="Sequence length (default: 512)")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations (default: 10)")
    parser.add_argument("--iterations", type=int, default=50, help="Timed iterations (default: 50)")
    parser.add_argument("--preset", type=str, default="production",
                        choices=["production", "tiny"],
                        help="Config preset: 'production' (engram-moe-1b) or 'tiny' (quick test)")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available. This benchmark requires a GPU.")
        sys.exit(1)

    device = torch.device("cuda")

    if args.preset == "tiny":
        config = create_tiny_config()
    else:
        config = create_production_config()

    print(f"\nStarting benchmark: preset={args.preset}, batch_size={args.batch_size}, seq_len={args.seq_len}")
    print(f"Warmup={args.warmup}, Iterations={args.iterations}")
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"VRAM total: {torch.cuda.get_device_properties(0).total_mem / (1024 ** 3):.1f} GB")
    print()

    results: List[BenchmarkResult] = []
    step = 1
    total_steps = 12 if config.use_moe and config.use_engram else 9

    # --- Isolated primitive components ---

    print(f"[{step}/{total_steps}] Benchmarking RMSNorm...")
    results.append(benchmark_rmsnorm(config, args.batch_size, args.seq_len, device, args.warmup, args.iterations))
    step += 1

    if config.fope_enabled:
        print(f"[{step}/{total_steps}] Benchmarking FoPE...")
        results.append(benchmark_fope(config, args.batch_size, args.seq_len, device, args.warmup, args.iterations))
        step += 1
    else:
        print(f"[{step}/{total_steps}] Benchmarking RoPE...")
        results.append(benchmark_rope(config, args.batch_size, args.seq_len, device, args.warmup, args.iterations))
        step += 1

    print(f"[{step}/{total_steps}] Benchmarking Embedding + lm_head...")
    results.append(benchmark_embedding(config, args.batch_size, args.seq_len, device, args.warmup, args.iterations))
    step += 1

    print(f"[{step}/{total_steps}] Benchmarking MLP (SwiGLU dense)...")
    results.append(benchmark_mlp(config, args.batch_size, args.seq_len, device, args.warmup, args.iterations))
    step += 1

    if config.use_moe:
        print(f"[{step}/{total_steps}] Benchmarking LatentMoE...")
        results.append(benchmark_latent_moe(config, args.batch_size, args.seq_len, device, args.warmup, args.iterations))
        step += 1

    if config.use_engram:
        print(f"[{step}/{total_steps}] Benchmarking Engram...")
        results.append(benchmark_engram(config, args.batch_size, args.seq_len, device, args.warmup, args.iterations))
        step += 1

    # --- Isolated attention cores ---

    print(f"[{step}/{total_steps}] Benchmarking GatedDeltaNet attention...")
    results.append(benchmark_gated_deltanet(config, args.batch_size, args.seq_len, device, args.warmup, args.iterations))
    step += 1

    print(f"[{step}/{total_steps}] Benchmarking MLA attention...")
    results.append(benchmark_mla_attention(config, args.batch_size, args.seq_len, device, args.warmup, args.iterations))
    step += 1

    # --- Composite blocks ---

    print(f"[{step}/{total_steps}] Benchmarking DeltaNet Block (attn + MLP + norms)...")
    results.append(benchmark_deltanet_block(config, args.batch_size, args.seq_len, device, args.warmup, args.iterations))
    step += 1

    print(f"[{step}/{total_steps}] Benchmarking MLA Block + MoE (attn + MoE + norms)...")
    results.append(benchmark_mla_block(config, args.batch_size, args.seq_len, device, args.warmup, args.iterations))
    step += 1

    if config.use_engram:
        print(f"[{step}/{total_steps}] Benchmarking MLA Block + MoE + Engram...")
        results.append(benchmark_mla_block_with_engram(config, args.batch_size, args.seq_len, device, args.warmup, args.iterations))
        step += 1

    # --- Full model (baseline) ---

    print(f"[{step}/{total_steps}] Benchmarking Full Model...")
    full_result = benchmark_full_model(config, args.batch_size, args.seq_len, device, args.warmup, args.iterations)
    results.insert(0, full_result)

    print_results(results, args.batch_size, args.seq_len, config)


if __name__ == "__main__":
    main()
