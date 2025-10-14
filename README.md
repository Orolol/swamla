# SWA-MLA: Standalone Sliding Window Attention + Multi-head Latent Attention

A completely self-contained implementation of the SWA-MLA hybrid architecture for efficient large language model training.

## Features

- **Hybrid Architecture**: Combines Sliding Window Attention (SWA) for local context with Multi-head Latent Attention (MLA) for global understanding
- **Memory Efficient**: MLA uses low-rank compression for KV cache, reducing memory footprint
- **FP8 Support**: Automatic FP8 training on H100/H200 GPUs for 25-30% VRAM reduction
- **Optimized Data Loading**: Packed sequence data loader minimizes padding waste
- **Multiple Optimizers**: Supports AdamW, Lion, and FP8 variants
- **Wandb Integration**: Built-in experiment tracking
- **Multi-GPU Ready**: Full DDP support with optimized communication
- **torch.compile Compatible**: Enhanced performance with PyTorch 2.0+ compilation

## Architecture Overview

The SWA-MLA model interleaves two types of attention blocks:

### SWA (Sliding Window Attention) Blocks
- Local attention with configurable window size (default: 256 tokens)
- Attention sink: Always attends to first N tokens for stability
- Uses RoPE (Rotary Position Embeddings)
- Efficient for capturing local patterns

### MLA (Multi-head Latent Attention) Blocks
- Global attention with low-rank KV compression
- Configurable LoRA ranks for Q and KV projections
- Separate NoPE and RoPE head dimensions
- Memory-efficient for long contexts

## Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### Basic Training

```bash
# Train small model (12 layers, 1024 dim)
./scripts/train_swa_mla.sh small 8 2048

# Train medium model (24 layers, 1536 dim)
./scripts/train_swa_mla.sh medium 4 2048

# Train with Lion optimizer
./scripts/train_swa_mla.sh small 16 2048 outputs/my_model false lion
```

### Advanced Training

```bash
python train.py \
    --size medium \
    --batch_size 8 \
    --block_size 2048 \
    --output_dir outputs/swa_mla \
    --optimizer_type lion \
    --learning_rate 1e-4 \
    --max_iters 100000 \
    --wandb_project my-llm-project \
    --compile \
    --gradient_checkpointing \
    --use_fp8  # Only on H100/H200
```

### Multi-GPU Training

```bash
# Use torchrun for distributed training
torchrun --nproc_per_node=4 train.py \
    --size medium \
    --batch_size 4 \
    --block_size 2048 \
    --gradient_accumulation_steps 4 \
    --compile
```

## Model Sizes

| Size   | Layers | Embed Dim | Heads | Parameters |
|--------|--------|-----------|-------|------------|
| small  | 12     | 1024      | 12    | ~150M      |
| base   | 24     | 1536      | 16    | ~400M      |
| large  | 28     | 2048      | 24    | ~800M      |
| xl     | 32     | 4096      | 32    | ~2B        |

## Configuration

### SWA Parameters

- `--swa_layers_per_cycle`: Number of SWA blocks per cycle (default: 2)
- `--swa_window`: Sliding window size in tokens (default: 256)
- `--swa_sink_size`: Number of initial tokens for attention sink (default: 4)

### MLA Parameters

- `--mla_layers_per_cycle`: Number of MLA blocks per cycle (default: 1)
- `--mla_q_lora_rank`: Q projection LoRA rank (default: 0 = no compression)
- `--mla_kv_lora_rank`: KV projection LoRA rank (default: 256)
- `--mla_qk_nope_head_dim`: Non-positional head dimension (default: 128)
- `--mla_qk_rope_head_dim`: RoPE head dimension (default: 64)
- `--mla_v_head_dim`: Value head dimension (default: 128)

### Training Parameters

- `--learning_rate`: Peak learning rate (default: 1e-4)
- `--warmup_iters`: Warmup iterations (default: 400)
- `--grad_clip`: Gradient clipping norm (default: 1.0)
- `--weight_decay`: AdamW weight decay (default: 0.1)
- `--gradient_accumulation_steps`: Gradient accumulation (default: 1)

### Optimizer Options

- `adamw`: Standard AdamW (fused on CUDA)
- `lion`: Lion optimizer (50% less memory than AdamW)

With `--use_fp8` on H100/H200:
- `FP8AdamW`: BF16 moments + FP8 compute
- `FP8Lion`: BF16 moment + FP8 compute

## Data Loading

The packed data loader automatically:
- Downloads FineWeb-Edu dataset (CC-MAIN-2024-10)
- Packs multiple documents per sequence to minimize padding
- Handles tokenization with configurable tokenizer
- Supports DDP with automatic sharding

### Custom Dataset

Modify `data/data_loader_packed.py` to use your own dataset:

```python
self.dataset = load_dataset(
    "your-org/your-dataset",
    split=split,
    streaming=True,
)
```

## FP8 Training

### Requirements

- NVIDIA H100 or H200 GPU
- PyTorch 2.0+
- transformer-engine (`pip install transformer-engine[pytorch]`)

### Benefits

- ~25-30% VRAM reduction vs BF16
- ~38% speedup with torch.compile
- BF16 optimizer moments for numerical stability
- FP32 master weights and gradients

### Usage

```bash
# FP8 is automatically enabled on H100/H200
./scripts/train_swa_mla.sh medium 8 2048
```

## Logging

### Console Logging

```
Step    100 | Loss: 3.4567 | LR: 1.00e-04 | Tokens/sec: 45,123
Step    200 | Loss: 3.2134 | LR: 1.00e-04 | Tokens/sec: 46,892
```

### Wandb Logging

```bash
export WANDB_API_KEY=your_key_here

python train.py \
    --wandb_project my-project \
    --wandb_run_name swa_mla_experiment
```

Tracks:
- Training loss
- Validation loss and perplexity
- Learning rate schedule
- Tokens per second
- GPU memory usage

## Performance Tips

1. **Use torch.compile**: Add `--compile` for 10-20% speedup (requires PyTorch 2.0+)
2. **Gradient Checkpointing**: Use `--gradient_checkpointing` to reduce memory at cost of 20% slower training
3. **Batch Size**: Increase until GPU memory is ~90% used for best throughput
4. **Gradient Accumulation**: Use `--gradient_accumulation_steps` to simulate larger batch sizes
5. **Lion Optimizer**: Consider Lion for 50% less optimizer memory
6. **FP8**: Use FP8 on H100/H200 for best memory efficiency
7. **Packed Sequences**: The packed data loader is already optimized, but you can increase `buffer_docs`

## Checkpointing

Checkpoints are saved every `--save_interval` steps to `--output_dir`:

```
outputs/swa_mla/
├── checkpoint_5000.pt
├── checkpoint_10000.pt
└── checkpoint_15000.pt
```

Each checkpoint contains:
- Model state dict
- Optimizer state dict
- Training step
- Full configuration

### Resume Training

To resume training from a checkpoint:

```python
checkpoint = torch.load('outputs/swa_mla/checkpoint_10000.pt')
model.load_state_dict(checkpoint['model'])
optimizer.load_state_dict(checkpoint['optimizer'])
start_step = checkpoint['step']
```

## Project Structure

```
swa_mla/
├── models/
│   ├── swa_mla_model.py      # Main hybrid model
│   ├── mla.py                 # MLA attention implementation
│   ├── mla_block.py           # MLA transformer block
│   ├── attention.py           # SWA attention implementation
│   ├── mlp.py                 # Feed-forward network
│   ├── normalization.py       # RMSNorm, DynamicTanh
│   └── positional_encoding.py # RoPE implementation
├── data/
│   └── data_loader_packed.py  # Packed sequence data loader
├── optimization/
│   └── fp8_trainer.py         # FP8 optimizers and utilities
├── scripts/
│   └── train_swa_mla.sh       # Training launch script
├── train.py                   # Main training script
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## Technical Details

### Memory Usage (Medium Model, Block Size 2048)

| Configuration | VRAM per GPU |
|---------------|--------------|
| BF16 + AdamW | ~24 GB |
| BF16 + Lion | ~18 GB |
| FP8 + FP8AdamW | ~16 GB |
| FP8 + FP8Lion | ~12 GB |

### Training Speed (Medium Model, 4x H100)

| Configuration | Tokens/sec |
|---------------|------------|
| BF16 | ~180K |
| BF16 + compile | ~220K |
| FP8 | ~200K |
| FP8 + compile | ~270K |

## Troubleshooting

### CUDA Out of Memory

1. Reduce `--batch_size`
2. Enable `--gradient_checkpointing`
3. Reduce `--block_size`
4. Use `--use_fp8` on H100/H200
5. Switch to `--optimizer_type lion`

### NaN Loss

1. Reduce `--learning_rate`
2. Increase `--warmup_iters`
3. Ensure `--grad_clip 1.0` is set
4. Check data quality

### Slow Training

1. Use `--compile` (PyTorch 2.0+)
2. Increase `--batch_size` if memory allows
3. Disable `--gradient_checkpointing` if memory allows
4. Use FP8 on H100/H200
5. Increase `--num_workers` for data loading

## License

This is a standalone implementation extracted from the gptoughts project for easy deployment and experimentation.

## Citation

If you use this code, please cite:

```bibtex
@software{swa_mla_standalone,
  title = {SWA-MLA: Standalone Implementation},
  year = {2025},
  note = {Hybrid Sliding Window + Multi-head Latent Attention}
}
```

## References

- [Multi-head Latent Attention (DeepSeek)](https://arxiv.org/abs/2401.06066)
- [Sliding Window Attention](https://arxiv.org/abs/2004.05150)
- [RoPE: Rotary Position Embedding](https://arxiv.org/abs/2104.09864)
- [Lion Optimizer](https://arxiv.org/abs/2302.06675)
- [FP8 Training (NVIDIA)](https://arxiv.org/abs/2209.05433)
