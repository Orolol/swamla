# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SWA-MLA is a **standalone, self-contained** implementation of a hybrid language model architecture that interleaves Sliding Window Attention (SWA) blocks with Multi-head Latent Attention (MLA) blocks. This is a production-ready extraction focused on efficiency and ease of deployment.

## Key Commands

### Setup
```bash
# Create virtual environment and install dependencies
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Verify setup
python test_setup.py
```

### Training
```bash
# Using the shell script (recommended)
./scripts/train_swa_mla.sh small 8 2048

# Using Python directly
python train.py --size small --batch_size 8 --block_size 2048

# Multi-GPU training with torchrun
torchrun --nproc_per_node=4 train.py --size medium --batch_size 4 --block_size 2048
```

### Testing
```bash
# Run setup verification
python test_setup.py
```

## Architecture Overview

### Hybrid Block Structure
The model uses a **cyclic pattern** of attention blocks:
- **SWA blocks**: Local sliding window attention with RoPE, optimized for capturing short-range dependencies
- **MLA blocks**: Global attention with low-rank KV compression, efficient for long-range context

The cycle pattern is controlled by:
- `swa_layers_per_cycle` (default: 2) - Number of SWA blocks per cycle
- `mla_layers_per_cycle` (default: 1) - Number of MLA blocks per cycle

Example: With defaults (2 SWA + 1 MLA), a 12-layer model has the pattern:
`[SWA, SWA, MLA, SWA, SWA, MLA, SWA, SWA, MLA, SWA, SWA, MLA]`

### Key Components

**SWA (Sliding Window Attention) Blocks** ([models/attention.py](models/attention.py)):
- Causal self-attention with configurable window size (default: 256 tokens)
- **Attention sink**: Always attends to first N tokens (default: 4) for stability
- Uses RoPE positional embeddings
- Efficient for local patterns without memory overhead

**MLA (Multi-head Latent Attention) Blocks** ([models/mla.py](models/mla.py), [models/mla_block.py](models/mla_block.py)):
- Global attention with low-rank compression via LoRA
- Configurable ranks: `q_lora_rank` (default: 0), `kv_lora_rank` (default: 256)
- Separate head dimensions: NoPE (`qk_nope_head_dim`=128) and RoPE (`qk_rope_head_dim`=64)
- Memory-efficient KV cache for long contexts

**Model Architecture** ([models/swa_mla_model.py](models/swa_mla_model.py)):
- Main class: `SWAMLAModel`
- Factory function: `create_swa_mla_model(size, vocab_size, block_size, ...)`
- Configuration: `SWAMLAConfig` dataclass
- Four preset sizes: small (~150M), base (~400M), large (~800M), xl (~2B)

### Data Loading
**Packed Sequence Loader** ([data/data_loader_packed.py](data/data_loader_packed.py)):
- Uses FineWeb-Edu dataset by default (CC-MAIN-2024-10)
- Packs multiple documents into each sequence to minimize padding waste
- Handles tokenization with GPT-2 tokenizer by default
- DDP-compatible with automatic sharding

### Optimization
**FP8 Training** ([optimization/fp8_trainer.py](optimization/fp8_trainer.py)):
- Automatic FP8 on H100/H200 GPUs for ~25-30% VRAM reduction
- Custom optimizers: `FP8AdamW`, `FP8Lion`
- BF16 optimizer moments, FP32 master weights
- Compatible with torch.compile for additional speedup

**Supported Optimizers**:
- `adamw`: Standard AdamW (fused on CUDA)
- `lion`: Lion optimizer (50% less memory than AdamW)
- FP8 variants automatically selected with `--use_fp8` flag

## File Structure

```
swamla/
├── train.py                           # Main training script
├── test_setup.py                      # Setup verification
├── models/
│   ├── swa_mla_model.py              # Main hybrid model + config
│   ├── attention.py                   # SWA attention implementation
│   ├── mla.py                         # MLA attention core
│   ├── mla_block.py                   # MLA transformer block
│   ├── mlp.py                         # Feed-forward networks
│   ├── normalization.py               # RMSNorm, DynamicTanh
│   └── positional_encoding.py         # RoPE implementation
├── data/
│   └── data_loader_packed.py          # Packed sequence data loader
├── optimization/
│   └── fp8_trainer.py                 # FP8 optimizers
└── scripts/
    └── train_swa_mla.sh               # Training launch script
```

## Model Configuration

### Preset Sizes
Defined in `create_swa_mla_model()` ([models/swa_mla_model.py:448-469](models/swa_mla_model.py#L448-L469)):

| Size   | Layers | Embed Dim | Heads | Head Dim | Parameters |
|--------|--------|-----------|-------|----------|------------|
| small  | 12     | 1024      | 16    | 64       | ~150M      |
| base   | 24     | 1536      | 16    | 96       | ~400M      |
| large  | 28     | 2048      | 16    | 128      | ~800M      |
| xl     | 32     | 4096      | 32    | 128      | ~2B        |

### Critical Parameters

**SWA Configuration**:
- `swa_window`: Window size for sliding attention (default: 256)
- `swa_sink_size`: Number of initial tokens always attended to (default: 4)
- Important: The attention sink prevents loss instability by maintaining access to early tokens

**MLA Configuration**:
- `mla_kv_lora_rank`: KV compression rank (default: 256) - higher = more capacity, more memory
- `mla_q_lora_rank`: Q compression rank (default: 0 = disabled)
- Head dimensions must satisfy: `n_embd` is compatible with the head configuration

**Training Stability**:
- `grad_clip`: Always use 1.0 for SWA-MLA (essential for stability)
- `warmup_iters`: Minimum 400 iterations recommended
- `learning_rate`: Start with 1e-4, adjust based on model size

## Training Loop Details

Main training script: [train.py](train.py)

### Key Features:
1. **Mixed Precision**: BF16 autocast by default, FP8 optional on H100/H200
2. **Gradient Accumulation**: Supports multi-step accumulation via `--gradient_accumulation_steps`
3. **DDP Support**: Automatic distributed training with torchrun
4. **torch.compile**: Optional compilation with `--compile` flag (~10-20% speedup)
5. **Gradient Checkpointing**: Reduces memory at ~20% speed cost via `--gradient_checkpointing`
6. **Learning Rate Schedule**: Linear warmup + cosine decay

### Training Flow:
1. Setup distributed training (if using DDP)
2. Create model with preset or custom config
3. Initialize packed data loader (automatically handles FineWeb-Edu)
4. Configure optimizer (AdamW/Lion with proper param grouping)
5. Training loop with gradient accumulation
6. Periodic validation (every `eval_interval` steps)
7. Checkpointing (every `save_interval` steps)

### Optimizer Configuration
Implemented in `configure_optimizer()` ([train.py:80-119](train.py#L80-L119)):
- Separate parameter groups: decay (weights) vs no_decay (biases, norms, embeddings)
- Automatic FP8 optimizer selection on H100/H200
- Fused AdamW on CUDA for better performance

## Important Implementation Details

### Attention Sink Pattern
SWA blocks use an "attention sink" ([models/attention.py](models/attention.py)):
```python
# First swa_sink_size tokens are always attended to
# Prevents catastrophic forgetting of early context
```
This is critical for training stability and must not be removed.

### RoPE Frequency Computation
Precomputed once in model init ([models/swa_mla_model.py:213-233](models/swa_mla_model.py#L213-L233)):
- Supports optional linear scaling via `rope_scaling` config
- Cached as non-persistent buffer
- Moved to device on-demand to avoid DDP VRAM duplication

### Gradient Checkpointing
Applied at block level with `use_reentrant=False`:
- Only active during training
- Separate checkpointing for attention and MLP sub-blocks
- Controlled by `use_gradient_checkpointing` config flag

### FP8 Integration
Automatic detection in training script:
- Checks GPU type (H100/H200) via nvidia-smi
- Uses FP8 optimizers from `optimization/fp8_trainer.py`
- BF16 moments + FP8 compute for numerical stability

## Debugging and Troubleshooting

### Common Issues

**CUDA Out of Memory**:
1. Reduce `--batch_size`
2. Enable `--gradient_checkpointing`
3. Reduce `--block_size`
4. Try `--optimizer_type lion` (50% less memory)
5. Use `--use_fp8` on H100/H200

**NaN Loss**:
1. Verify `--grad_clip 1.0` is set (critical for SWA-MLA)
2. Reduce `--learning_rate`
3. Increase `--warmup_iters`
4. Check data quality (run test_setup.py)

**Slow Training**:
1. Add `--compile` flag (PyTorch 2.0+ required)
2. Increase `--batch_size` if memory allows
3. Remove `--gradient_checkpointing` if memory allows
4. Increase `--num_workers` for data loading

### Performance Expectations
From [README.md](README.md), medium model (400M params) on 4x H100:
- BF16: ~180K tokens/sec
- BF16 + compile: ~220K tokens/sec
- FP8 + compile: ~270K tokens/sec

## Standalone Nature

This codebase is **completely self-contained**:
- No dependencies on parent project
- No `../` relative imports
- All components included locally
- Can be copied anywhere and run independently

**Important**: When modifying this code, maintain the standalone property. Do not introduce external dependencies beyond those in [requirements.txt](requirements.txt).

## Development Guidelines

### When Adding Features:
1. Maintain self-contained nature - no external project dependencies
2. Preserve the hybrid SWA+MLA architecture pattern
3. Keep FP8 compatibility in mind
4. Test with `test_setup.py` after changes
5. Update model parameter counts if architecture changes

### When Debugging:
1. Run `test_setup.py` first to verify environment
2. Check `nvidia-smi` for GPU memory usage
3. Enable wandb logging for detailed metrics
4. Use `--log_interval 1` for fine-grained loss tracking
5. Verify gradient norms aren't exploding (check grad_clip is working)

### When Optimizing:
1. Profile with torch.profiler before making changes
2. Test with and without `--compile` to measure impact
3. Use smaller model size (small) for rapid iteration
4. Compare tokens/sec before and after changes
5. Verify memory usage doesn't increase unexpectedly

## Critical Code Patterns

### Block Type Detection in Forward Pass
In `SWAMLAModel.forward()` ([models/swa_mla_model.py:326-333](models/swa_mla_model.py#L326-L333)):
```python
for block in self.transformer.h:
    if isinstance(block, SWALocalBlock):
        x = block(x)  # SWA blocks don't need freqs_cis
    elif isinstance(block, (MLASelectiveBlock, MLABlock)):
        x = block(x, 0, freqs_cis, mask)  # MLA blocks need RoPE freqs
```
This pattern must be preserved when adding new block types.

### Parameter Grouping for Optimization
Weight decay is selectively applied:
- Applied: Linear weights in attention and MLP
- Excluded: Biases, normalization layers, embeddings
Pattern: `if any(nd in name for nd in ['.bias', 'norm', 'ln_', 'wte', 'wpe'])`

### DDP-Safe Buffer Management
Frequency buffers registered as non-persistent:
```python
self.register_buffer("freqs_cis", freqs, persistent=False)
```
Then moved to device on-demand in forward pass to avoid duplication.

## User Instructions from Global CLAUDE.md

The following instructions from the user's global configuration apply to this project:

1. **Never simplify or skip functionality**: Always implement features completely as specified. Never take shortcuts by disabling functions or parameters. Focus on fixing errors, not avoiding them.

2. **Go deep on problems**: If a model is intended to work in a certain way, follow the rules and specifications exactly. Don't change to something simpler just because it's difficult to implement.

These principles are especially important for this project because:
- The hybrid SWA+MLA architecture has precise mathematical requirements
- The attention sink pattern is critical for stability
- FP8 integration requires exact dtype management
- The cyclic block pattern must be preserved exactly
