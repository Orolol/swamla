# Quickstart: FoPE + YaRN Context Extension

**Feature**: 001-fope-yarn
**Date**: 2026-02-04

## Overview

YaRN (Yet another RoPE extensioN) allows models trained on short contexts to perform inference on much longer sequences without retraining.

## Quick Usage

### Enable 4x Context Extension

```bash
# Training with 8K context using a model architecture designed for 2K
./scripts/train.sh --preset base 8 8192 \
    --yarn_enabled \
    --yarn_scale_factor 4.0 \
    --yarn_original_max_seq_len 2048
```

### Python Configuration

```python
from models.swa_mla_model import SWAMLAConfig, create_swa_mla_model

# Create model with YaRN enabled
config = SWAMLAConfig(
    block_size=8192,           # Extended context length
    yarn_enabled=True,         # Enable YaRN
    yarn_scale_factor=4.0,     # 8192 / 2048 = 4x extension
    yarn_original_max_seq_len=2048,
)

model = create_swa_mla_model(
    size="base",
    vocab_size=50304,
    block_size=8192,
    yarn_enabled=True,
    yarn_scale_factor=4.0,
)
```

## Common Scenarios

### Inference on Long Documents

Use a model trained on 2K tokens to process 16K tokens:

```python
# Load trained model
model = create_swa_mla_model(
    size="base",
    block_size=16384,  # Extended for inference
    yarn_enabled=True,
    yarn_scale_factor=8.0,  # 16384 / 2048
    yarn_original_max_seq_len=2048,
)
model.load_state_dict(torch.load("checkpoint.pt"))

# Run inference on long sequence
with torch.inference_mode():
    output = model(long_input_ids)  # Can be up to 16K tokens
```

### Training with Extended Context

Train from scratch with YaRN for better long-context capabilities:

```bash
./scripts/train.sh --preset engram-moe 4 4096 \
    --yarn_enabled \
    --yarn_scale_factor 2.0 \
    --yarn_original_max_seq_len 2048
```

## Parameters Reference

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--yarn_enabled` | False | Enable YaRN context extension |
| `--yarn_scale_factor` | 1.0 | Extension ratio (target / original) |
| `--yarn_original_max_seq_len` | 2048 | Original training context length |
| `--yarn_beta_fast` | 32.0 | High frequency boundary |
| `--yarn_beta_slow` | 1.0 | Low frequency boundary |

## Expected Quality

| Extension | Typical Perplexity Increase |
|-----------|----------------------------|
| 2x | <5% |
| 4x | <10% |
| 8x | <15% |
| 16x | 15-25% (may need fine-tuning) |

## Troubleshooting

### Model outputs degrade at long sequences

- Ensure `yarn_scale_factor` matches your actual extension ratio
- Try increasing `yarn_beta_fast` (e.g., 64 instead of 32)
- Consider fine-tuning on longer sequences

### Memory issues with very long contexts

- YaRN doesn't reduce memory - attention is still O(n²)
- Use gradient checkpointing: `--gradient_checkpointing`
- Reduce batch size proportionally to context increase

### Backward compatibility

When `yarn_enabled=False` (default), the model behaves exactly like standard RoPE. Existing checkpoints load without any changes.
