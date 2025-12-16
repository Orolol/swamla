# DeltaNet-MLA: Hybrid Linear Attention + Multi-head Latent Attention with LatentMoE

A high-performance hybrid architecture combining O(n) linear attention with MLA and efficient mixture-of-experts.

## Features

- **GatedDeltaNet**: O(n) linear attention for local context (replaces quadratic SWA)
- **MLA (Multi-head Latent Attention)**: Global attention with low-rank KV compression
- **LatentMoE**: NVIDIA Nemotron-3 style mixture-of-experts in latent space
- **Flash Attention 2/3**: Optimized attention kernels for MLA blocks
- **Muon + AdamW**: Native PyTorch Muon optimizer with AdamW fallback
- **DeltaNet Latent Compression**: Bottleneck projections for Q/K/V/O
- **Packed Sequence Loading**: Minimizes padding waste
- **Multi-GPU Ready**: Full DDP support with auto-detection
- **torch.compile**: Max-autotune mode for optimal throughput
- **HuggingFace Integration**: Auto-push checkpoints and resume from HF

## Architecture Overview

The model interleaves two types of attention blocks in a cyclic pattern (default: 2 DeltaNet + 1 MLA):

### GatedDeltaNet Blocks (Local Context)
- **O(n) linear attention** using the delta rule
- Gated mechanism for improved expressiveness
- Optional latent compression for Q/K/V/O projections
- Optional shared Q/K projection (K is normalized Q)
- Memory-efficient for long sequences

### MLA Blocks (Global Context)
- **Flash Attention** for efficient computation
- Low-rank KV compression via LoRA
- Separate NoPE and RoPE head dimensions
- **LatentMoE FFN**: Projects to latent space before expert routing
  - 4x compression ratio (e.g., 1024 → 256 latent dim)
  - 4x more experts with same compute budget
  - Better quality per FLOP

## Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install Flash Linear Attention (for GatedDeltaNet)
pip install fla
```

## Quick Start

### Basic Training

```bash
# Train with defaults (auto-detects GPUs, uses DDP if multiple)
./scripts/train_swa_mla_latent_deltanet.sh 8 2048

# Custom output directory
./scripts/train_swa_mla_latent_deltanet.sh 8 2048 outputs/my_model

# With HuggingFace auto-push
export HF_TOKEN="your_token"
./scripts/train_swa_mla_latent_deltanet.sh 8 2048 outputs/my_model false muon "YourUser/your-repo"
```

### Advanced Training

```bash
python train.py \
    --size moe-1b \
    --batch_size 8 \
    --block_size 2048 \
    --optimizer_type muon \
    --learning_rate 1e-4 \
    --compile \
    --compile_mode max-autotune \
    --gradient_checkpointing \
    --deltanet_latent_dim 256 \
    --deltanet_share_qk \
    --mla_q_lora_rank 256
```

### Multi-GPU Training

```bash
# Auto-detected by the script
./scripts/train_swa_mla_latent_deltanet.sh 8 2048

# Or manual with torchrun
torchrun --nproc_per_node=4 train.py \
    --size moe-1b \
    --batch_size 4 \
    --compile
```

## Model Sizes

| Size    | Layers | Embed Dim | Heads | Total Params | Active Params |
|---------|--------|-----------|-------|--------------|---------------|
| small   | 12     | 1024      | 16    | ~150M        | ~150M         |
| base    | 24     | 1536      | 16    | ~400M        | ~400M         |
| large   | 28     | 2048      | 16    | ~800M        | ~800M         |
| moe-1b  | 12     | 1024      | 12    | ~800M        | ~280M (35%)   |
| moe-2b  | 18     | 1024      | 16    | ~1.5B        | ~400M (27%)   |

## Configuration

### DeltaNet Parameters

- `--swa_layers_per_cycle`: DeltaNet blocks per cycle (default: 2)
- `--deltanet_latent_dim`: Latent dimension for projections (0=disabled, default: 256)
- `--deltanet_share_qk`: Share Q/K projection (K is normalized Q)

### MLA Parameters

- `--mla_layers_per_cycle`: MLA blocks per cycle (default: 1)
- `--mla_q_lora_rank`: Q projection LoRA rank (default: 0)
- `--mla_kv_lora_rank`: KV projection LoRA rank (default: 256)
- `--mla_qk_nope_head_dim`: Non-positional head dim (default: 128)
- `--mla_qk_rope_head_dim`: RoPE head dim (default: 64)
- `--mla_v_head_dim`: Value head dim (default: 128)

### LatentMoE Parameters

- `--use_moe`: Enable MoE for MLA blocks (default: true for moe-* sizes)
- `--use_latent_moe`: Use LatentMoE instead of standard MoE (default: true)
- `--latent_ratio`: Compression ratio (default: 4, i.e., latent_dim = n_embd/4)
- `--n_experts`: Base number of routed experts (default: 32)
- `--n_activated`: Base activated experts per token (default: 3)
- `--n_shared_experts`: Always-active shared experts (default: 1)
- `--latent_preserve_expert_dim`: Keep full expert_dim in latent space

### Training Parameters

- `--learning_rate`: Peak learning rate (default: 1e-4)
- `--warmup_iters`: Warmup iterations (default: 400)
- `--grad_clip`: Gradient clipping norm (default: 1.0)
- `--weight_decay`: Weight decay (default: 0.1)
- `--gradient_checkpointing`: Enable gradient checkpointing

### Optimizer Options

- `muon`: Native PyTorch Muon optimizer (requires PyTorch 2.6+)
  - Uses ~200x higher LR than AdamW internally
  - Only for exactly 2D parameters (matrices)
  - Non-2D params use AdamW automatically
- `adamw`: Standard fused AdamW

## Performance

### Throughput (moe-1b on 2x RTX 5090)

| Configuration | Tokens/sec |
|---------------|------------|
| BF16 + compile (max-autotune) | ~35K |
| + Gradient checkpointing | ~32K |

### Memory Efficiency

The LatentMoE architecture provides:
- **4x more experts** with same compute budget
- **~35% active parameters** per forward pass
- Better scaling compared to dense models

## Environment Variables

```bash
# LatentMoE configuration
LATENT_RATIO=4           # Compression ratio
N_EXPERTS=32             # Base experts (scaled by LATENT_RATIO → 128)
N_ACTIVATED=3            # Base activated (scaled → 12)

# DeltaNet Latent Compression
DELTANET_LATENT_DIM=256  # Latent dimension (0=disabled)
DELTANET_SHARE_QK=true   # Share Q/K projection

# MLA Q LoRA
MLA_Q_LORA_RANK=256      # Q LoRA rank (0=disabled)
```

## Project Structure

```
swamla/
├── models/
│   ├── swa_mla_model.py      # Main hybrid model
│   ├── gated_deltanet.py     # GatedDeltaNet blocks
│   ├── mla.py                # MLA attention
│   ├── mla_block.py          # MLA transformer block
│   ├── moe.py                # MoE and LatentMoE layers
│   ├── mlp.py                # SwiGLU MLP
│   ├── normalization.py      # RMSNorm, DynamicTanh
│   └── positional_encoding.py # RoPE
├── data/
│   └── data_loader_packed.py # Packed sequence loader
├── scripts/
│   └── train_swa_mla_latent_deltanet.sh
├── train.py                  # Main training script
└── README.md
```

## Troubleshooting

### CUDA Out of Memory

1. Reduce `--batch_size`
2. Enable `--gradient_checkpointing`
3. Reduce `--block_size`
4. Use smaller `--deltanet_latent_dim`

### NaN Loss

1. Reduce `--learning_rate`
2. Increase `--warmup_iters`
3. Ensure `--grad_clip 1.0` is set

### Slow Training

1. Use `--compile --compile_mode max-autotune`
2. Increase `--batch_size` if memory allows
3. Use `--num_workers 8` or higher

### Muon Not Available

PyTorch Muon requires PyTorch 2.6+. The script automatically falls back to AdamW.

## References

- [DeltaNet: Linear Transformers](https://arxiv.org/abs/2310.00701)
- [Multi-head Latent Attention (DeepSeek)](https://arxiv.org/abs/2401.06066)
- [LatentMoE (NVIDIA Nemotron-3)](https://arxiv.org/abs/2407.08936)
- [Flash Attention](https://arxiv.org/abs/2205.14135)
- [Muon Optimizer](https://arxiv.org/abs/2502.16982)
- [RoPE](https://arxiv.org/abs/2104.09864)

## Citation

```bibtex
@software{deltanet_mla,
  title = {DeltaNet-MLA: Hybrid Linear Attention with LatentMoE},
  year = {2025},
  note = {O(n) DeltaNet + MLA + LatentMoE Architecture}
}
```
