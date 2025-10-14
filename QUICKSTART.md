# SWA-MLA Quick Start Guide

## Installation (5 minutes)

```bash
cd swa_mla

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install core dependencies
pip install torch transformers datasets wandb lion-pytorch

# Optional: Install performance enhancers
pip install flash-attn xformers  # If on compatible GPU
```

## Test Your Setup (30 seconds)

```bash
python test_setup.py
```

You should see all tests pass!

## Train Your First Model (2 minutes setup)

### Option 1: Use the Shell Script (Easiest)

```bash
chmod +x scripts/train_swa_mla.sh
./scripts/train_swa_mla.sh small 8 2048
```

That's it! Training will start immediately.

### Option 2: Use Python Directly (More Control)

```bash
python train.py \
    --size small \
    --batch_size 8 \
    --block_size 2048 \
    --max_iters 10000 \
    --wandb_project my-llm-project
```

## Configuration Cheat Sheet

### Model Sizes

| Size   | Parameters | VRAM (BF16) | Training Speed |
|--------|------------|-------------|----------------|
| small  | ~125M      | ~8 GB       | ~60K tok/s     |
| base   | ~400M      | ~16 GB      | ~40K tok/s     |
| large  | ~700M      | ~24 GB      | ~25K tok/s     |
| xl     | ~2B        | ~40 GB      | ~12K tok/s     |

*Single H100 GPU estimates*

### Quick Performance Tweaks

```bash
# Reduce memory usage
--gradient_checkpointing \
--optimizer_type lion \
--batch_size 4

# Maximize speed
--compile \
--batch_size 16 \
--num_workers 16

# Multi-GPU (4x GPUs)
torchrun --nproc_per_node=4 train.py --size medium --batch_size 4
```

### Essential Hyperparameters

```bash
--learning_rate 1e-4       # Good starting point
--warmup_iters 400         # Warmup for stability
--grad_clip 1.0            # Essential for SWA-MLA
--weight_decay 0.1         # Standard for LLMs
```

## Monitoring Training

### Console Output
```
Step    100 | Loss: 3.4567 | LR: 1.00e-04 | Tokens/sec: 45,123
```

### Wandb Dashboard
```bash
export WANDB_API_KEY=your_key

python train.py --wandb_project my-project
```

Then open: https://wandb.ai/your-username/my-project

## Common Issues & Solutions

### Out of Memory
1. Reduce `--batch_size`
2. Add `--gradient_checkpointing`
3. Try `--optimizer_type lion`
4. Reduce `--block_size`

### Slow Training
1. Add `--compile`
2. Increase `--batch_size`
3. Remove `--gradient_checkpointing`
4. Increase `--num_workers`

### NaN Loss
1. Reduce `--learning_rate`
2. Increase `--warmup_iters`
3. Check `--grad_clip 1.0` is set

## Next Steps

1. **Monitor your training** - Watch loss decrease and perplexity improve
2. **Experiment with hyperparameters** - Try different learning rates
3. **Scale up** - Move to larger model sizes as you optimize
4. **Fine-tune** - Use your trained model for downstream tasks

## File Overview

```
swa_mla/
├── train.py                 # ← Main training script
├── models/                  # Model architecture
│   └── swa_mla_model.py    # ← Core model implementation
├── scripts/
│   └── train_swa_mla.sh    # ← Easy launch script
├── requirements.txt         # Dependencies
├── README.md               # Full documentation
└── QUICKSTART.md           # ← You are here
```

## Getting Help

- **Full docs**: See README.md
- **Test setup**: Run `python test_setup.py`
- **Training issues**: Check console output for errors
- **Questions**: The code is fully standalone and documented

## Tips for Success

1. **Start small** - Use `--size small` for initial experiments
2. **Use wandb** - Track experiments from day one
3. **Save checkpoints** - Models save every `--save_interval` steps
4. **Monitor GPU** - Run `nvidia-smi` to check utilization
5. **Be patient** - Good models take time!

---

**Ready to train?**

```bash
./scripts/train_swa_mla.sh small 8 2048
```

Happy training! 🚀
