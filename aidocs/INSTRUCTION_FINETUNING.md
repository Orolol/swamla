# Instruction Fine-Tuning Guide

This guide explains how to fine-tune your SWA-MLA model on instruction-following tasks using the SlimOrca dataset with ChatML format.

## Overview

The instruction fine-tuning pipeline includes:
- **Dataset**: [Open-Orca/SlimOrca](https://huggingface.co/datasets/Open-Orca/SlimOrca) - High-quality instruction-response pairs
- **Format**: ChatML with special tokens (`<|im_start|>`, `<|im_end|>`)
- **Loss Masking**: Loss calculated **only on assistant responses** (not on system/user prompts)
- **Packing**: Multiple conversations packed into each sequence for efficiency
- **Auto-Resume**: Automatically loads the latest pre-trained checkpoint from HuggingFace

## Quick Start

### 1. Prerequisites

Make sure you have:
- A pre-trained SWA-MLA checkpoint (from `train.py`)
- HuggingFace token set in environment: `export HF_TOKEN=your_token_here`
- Your model pushed to HuggingFace (e.g., `Orosius/swamla`)

### 2. Launch Instruction Fine-Tuning

The easiest way is to use the provided shell script:

```bash
# Basic usage (auto-resumes from latest HF checkpoint)
./scripts/train_instruct.sh small 16 2048 outputs/instruct true adamw "Orosius/swamla"

# Parameters:
#   $1: Model size (small, base, large, xl)
#   $2: Batch size (default: 16)
#   $3: Block size (default: 2048)
#   $4: Output directory (default: outputs/swa_mla_instruct)
#   $5: Resume from HF (default: true)
#   $6: Optimizer (default: adamw)
#   $7: HuggingFace repo ID
```

### 3. Monitor Training

The script will:
- Load the latest pre-trained checkpoint from your HF repo
- Add ChatML special tokens and resize embeddings (+2 tokens)
- Fine-tune on SlimOrca with loss masking
- Save checkpoints to `instruct/` subdirectory on HuggingFace every 100 steps
- Validate more frequently (every 100 steps) compared to pre-training

## Manual Usage

If you prefer manual control:

```bash
python train_instruct.py \
    --size small \
    --batch_size 16 \
    --block_size 2048 \
    --learning_rate 5e-5 \
    --min_lr 5e-6 \
    --warmup_iters 100 \
    --max_iters 10000 \
    --hf_repo_id "Orosius/swamla" \
    --resume_from_hf \
    --wandb_project "swamla-instruct" \
    --use_fp8 \
    --gradient_checkpointing
```

## Key Differences from Pre-Training

| Parameter | Pre-Training | Instruction Fine-Tuning | Reason |
|-----------|-------------|------------------------|--------|
| Learning Rate | `1e-4` | `5e-5` | Lower LR to avoid catastrophic forgetting |
| Min LR | `1e-5` | `5e-6` | Lower minimum for fine-tuning |
| Warmup Steps | `400` | `100` | Less warmup needed (model already trained) |
| Max Iterations | `100,000` | `10,000` | Fewer iterations for fine-tuning |
| Eval Interval | `1000` | `100` | More frequent validation |
| Save Interval | `5000` | `500` | More frequent checkpointing |
| Dataset | FineWeb-Edu | SlimOrca | Instruction-following dataset |
| Loss | All tokens | Masked (assistant only) | Only learn to generate responses |

## ChatML Format

The instruction fine-tuning uses ChatML format:

```
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
What is 2+2?<|im_end|>
<|im_start|>assistant
2+2 equals 4.<|im_end|>
```

### Loss Masking

The loss is calculated **only on assistant responses**:

```
Tokens:    [<|im_start|>, user, ..., <|im_end|>, <|im_start|>, assistant, Hello, world, <|im_end|>]
Loss Mask: [     0,        0,   ...,      0,          0,         1,      1,      1,        0     ]
                                                              ^^^^^^^^^^^^^^^^^^^^^^^^^
                                                              Loss calculated ONLY here
```

This ensures the model learns to **generate** good responses, not just to copy prompts.

## Using the Fine-Tuned Model

### Inference with ChatML

Once fine-tuned, use the `--chatml` flag in `inference.py`:

```bash
# Load from HuggingFace (latest instruct checkpoint)
python inference.py \
    --hf_repo_id "Orosius/swamla" \
    --hf_checkpoint "instruct/checkpoint_tokens_500k_loss_1.2345" \
    --mode chat \
    --chatml

# The model will automatically format prompts in ChatML and stop at <|im_end|>
```

### Example Chat Session

```
You: What is the capital of France?
Assistant: The capital of France is Paris. It has been the capital since the 12th century and is known for its art, culture, and landmarks like the Eiffel Tower.

You: /temp 0.5
[Temperature set to 0.5]

You: Tell me more about Paris
Assistant: Paris is France's largest city with over 2 million residents. It's divided into 20 districts called arrondissements...
```

## Checkpoint Organization

The fine-tuning script saves checkpoints to a **separate subdirectory** on HuggingFace:

```
your-repo/
├── checkpoint_tokens_10M_loss_2.5000/          # Pre-training checkpoints
│   ├── pytorch_model.bin
│   ├── config.json
│   └── tokenizer files...
├── instruct/                                    # Instruction fine-tuning checkpoints
│   ├── checkpoint_tokens_500k_loss_1.2345/
│   │   ├── pytorch_model.bin
│   │   ├── config.json
│   │   └── tokenizer files (with ChatML tokens)
│   └── checkpoint_tokens_1M_loss_1.1000/
│       └── ...
```

This organization:
- ✅ Keeps pre-training and fine-tuning checkpoints separate
- ✅ Allows easy comparison between models
- ✅ Prevents accidental overwriting

## Advanced Configuration

### Custom Hyperparameters

```bash
python train_instruct.py \
    --size base \
    --batch_size 8 \
    --block_size 2048 \
    --learning_rate 3e-5 \           # Even lower LR
    --warmup_iters 200 \              # More warmup
    --max_iters 20000 \               # More training
    --gradient_accumulation_steps 2 \ # Larger effective batch
    --eval_interval 50 \              # More frequent validation
    --hf_repo_id "username/model" \
    --resume_from_hf
```

### Multi-GPU Training

The script auto-detects GPUs and launches DDP:

```bash
# Will automatically use all available GPUs
./scripts/train_instruct.sh small 16 2048

# Manual control with CUDA_VISIBLE_DEVICES
CUDA_VISIBLE_DEVICES=0,1,2,3 ./scripts/train_instruct.sh medium 8 2048
```

### FP8 Training (H100/H200)

FP8 is enabled by default for ~1.45x speedup:

```bash
# FP8 is automatic on H100/H200
./scripts/train_instruct.sh small 16 2048

# Disable FP8 (not recommended)
python train_instruct.py --size small --batch_size 16 # (no --use_fp8 flag)
```

## Monitoring and Logging

### WandB Integration

The fine-tuning automatically logs to WandB:

```bash
# Set project name
./scripts/train_instruct.sh small 16 2048 outputs/instruct true adamw "username/model"

# Metrics logged:
# - train/loss
# - train/lr (learning rate schedule)
# - train/tokens_per_sec
# - val/loss
# - val/perplexity
```

### Console Output

```
Step   1000 | Loss: 1.2345 | LR: 4.50e-05 | Tokens/sec: 250,000 | Total: 10.5M

Validation | Loss: 1.1234 | Perplexity: 3.08

Pushing model to Hugging Face...
✓ Successfully uploaded to https://huggingface.co/username/model/tree/main/instruct/checkpoint_tokens_10.5M_loss_1.1234
```

## Troubleshooting

### Issue: "No checkpoints found in HuggingFace repo"

**Solution**: Make sure you have pre-trained checkpoints in your HF repo first. Run `train.py` before `train_instruct.py`.

### Issue: "Token indices sequence length is longer than..."

**Solution**: This is a benign warning from the tokenizer. The model handles longer sequences correctly. The warning comes from the GPT-2 tokenizer's default max length (1024), but our model supports 2048.

### Issue: Loss not decreasing

**Possible causes**:
1. Learning rate too high → Try `--learning_rate 3e-5` or `--learning_rate 1e-5`
2. Not enough warmup → Try `--warmup_iters 200`
3. Need more training → Increase `--max_iters`

### Issue: Out of memory

**Solutions**:
1. Reduce batch size: `--batch_size 8` or `--batch_size 4`
2. Reduce block size: `--block_size 1024`
3. Enable gradient checkpointing: `--gradient_checkpointing` (already enabled by default)
4. Use gradient accumulation: `--gradient_accumulation_steps 2`

## Data Loader Statistics

The instruction data loader provides useful statistics:

```
[InstructDataset] Stats:
  Batches built: 1000
  Conversations packed: 4523
  Avg padding ratio: 35.20%           # ~35% of tokens are padding
  Avg mask ratio (loss tokens): 25.50%  # ~25% of tokens contribute to loss
```

**Interpretation**:
- **Padding ratio**: Lower is better (more efficient)
- **Mask ratio**: This is normal! Only assistant responses contribute to loss
- Typical mask ratios: 15-35% (depends on conversation length)

## Performance Expectations

Based on testing with a small model (250M params):

| Configuration | Tokens/sec | Training Time (10k steps) |
|---------------|------------|--------------------------|
| 1x H100 + FP8 | ~250k | ~8 hours |
| 4x H100 + FP8 + DDP | ~900k | ~2 hours |
| 1x A100 + BF16 | ~180k | ~11 hours |

Note: Performance scales well with model size and GPU count.

## Best Practices

1. **Always pre-train first**: Instruction fine-tuning works best on pre-trained models
2. **Use lower learning rate**: Fine-tuning requires gentler updates (5e-5 vs 1e-4)
3. **Validate frequently**: Check every 100 steps to catch overfitting early
4. **Monitor mask ratio**: Should be 15-35% (if too low, check data loader)
5. **Save often**: Fine-tuning can be unstable, save every 500 steps
6. **Use FP8 on H100**: Significant speedup with minimal accuracy loss

## Files Overview

```
swamla/
├── train_instruct.py              # Main instruction fine-tuning script
├── data/
│   └── data_loader_instruct.py   # SlimOrca data loader with ChatML + masking
├── scripts/
│   └── train_instruct.sh         # Launch script with auto-GPU detection
├── inference.py                   # Supports --chatml flag for inference
└── INSTRUCTION_FINETUNING.md     # This file
```

## Next Steps

After fine-tuning:

1. **Test the model**: Use `inference.py --chatml --mode chat`
2. **Compare checkpoints**: Test multiple instruct checkpoints to find the best one
3. **Deploy**: Use the fine-tuned model in your application
4. **Iterate**: If quality isn't sufficient, try:
   - More training iterations
   - Different learning rate
   - Larger model size

## References

- [SlimOrca Dataset](https://huggingface.co/datasets/Open-Orca/SlimOrca)
- [ChatML Format Specification](https://github.com/openai/openai-python/blob/main/chatml.md)
- [Instruction Tuning Papers](https://arxiv.org/abs/2109.01652)

---

**Questions or issues?** Check the main [README.md](README.md) or open an issue on GitHub.