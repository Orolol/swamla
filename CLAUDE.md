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
# Using the shell script (recommended - auto-detects GPUs and launches DDP)
./scripts/train_swa_mla.sh small 8 2048

# The script automatically:
#   - Detects available GPUs using nvidia-smi
#   - Launches DDP training with torchrun if multiple GPUs are detected
#   - Launches single-GPU training if only one GPU is detected
#   - Configures TF32 for optimal performance on Ampere+ GPUs

# Using Python directly (for manual control)
python train.py --size small --batch_size 8 --block_size 2048

# Manual multi-GPU training with torchrun
torchrun --nproc_per_node=4 train.py --size medium --batch_size 4 --block_size 2048

# Disable TF32 for full IEEE FP32 precision (slower but more accurate)
python train.py --size small --disable_tf32
```

### Testing
```bash
# Run setup verification
python test_setup.py

# Test GPU detection and TF32 configuration
python test_tf32_config.py
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
**FP8 Training with TorchAO** ([optimization/fp8_torchao.py](optimization/fp8_torchao.py)):
- **Native PyTorch FP8 training** using TorchAO library
- Automatic FP8 conversion with `convert_to_float8_training()` on H100/H200
- **1.5x speedup** on up to 512 GPUs (tested on Llama-3 scale models)
- Compatible with `torch.compile` and FSDP2
- Automatic scaling factor management (no manual tuning required)

**TorchAO AdamWFp8 Optimizer**:
- Native FP8 optimizer from PyTorch AO library
- BF16 optimizer moments, FP32 master weights
- Automatically selected with `--use_fp8` flag on H100/H200
- Seamless integration with distributed training (DDP, FSDP2)

**Supported Optimizers**:
- `adamw`: Standard AdamW (fused on CUDA) - automatically upgrades to AdamWFp8 with `--use_fp8`
- `lion`: Lion optimizer (50% less memory than AdamW)
- Alternative quantized optimizers: `AdamW8bit` (2x memory reduction), `AdamW4bit` (4x memory reduction)

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
│   ├── fp8_torchao.py                 # TorchAO FP8 integration
│   └── fp8_trainer.py                 # Legacy FP8 optimizers (deprecated)
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

## TF32 Precision Control

### Overview
**TensorFloat-32 (TF32)** is a math mode for NVIDIA Ampere GPUs and later (A100, H100, RTX 30/40/50 series) that provides significant speedup for FP32 operations with minimal accuracy loss.

### Key Features
- **Automatic Detection**: Auto-detects GPU compute capability and enables TF32 if supported (CC >= 8.0)
- **Fine-Grained Control**: Uses PyTorch 2.9+ API for per-backend and per-operation control
- **Significant Speedup**: ~3-7x faster matmul and convolution operations on supported GPUs
- **Minimal Accuracy Loss**: Reduces mantissa from 23 bits (FP32) to 10 bits, maintains FP32 dynamic range

### How It Works
TF32 accelerates FP32 operations by:
1. **Matmul operations**: Includes attention layers, linear layers, and matrix multiplications
2. **cuDNN operations**: Includes convolutions and other cuDNN kernels
3. **Automatic conversion**: PyTorch automatically uses TF32 tensor cores when available

### Configuration

#### Default Behavior (TF32 Enabled)
The training script **enables TF32 by default** on supported GPUs for optimal performance:

```bash
# TF32 is automatically enabled
./scripts/train_swa_mla.sh small 8 2048
```

#### Disabling TF32 (Full IEEE FP32 Precision)
If you need full IEEE FP32 precision (e.g., for debugging or maximum accuracy):

```bash
# Disable TF32 for full precision
python train.py --size small --disable_tf32
```

#### Implementation Details
The `configure_tf32()` function in [train.py:75-138](train.py#L75-L138):

```python
# New PyTorch 2.9+ API (recommended)
torch.backends.cuda.matmul.fp32_precision = "tf32"    # Enable for matmul
torch.backends.cudnn.fp32_precision = "tf32"          # Enable for cuDNN

# Or disable for full precision
torch.backends.cuda.matmul.fp32_precision = "ieee"    # Full FP32
torch.backends.cudnn.fp32_precision = "ieee"          # Full FP32

# Legacy API (PyTorch < 2.9) - automatic fallback
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
```

### Performance Impact

| GPU Architecture | Compute Capability | TF32 Support | Expected Speedup |
|------------------|-------------------|--------------|------------------|
| Ampere (A100, RTX 30 series) | 8.0 - 8.9 | ✅ Yes | ~3-5x |
| Hopper (H100) | 9.0 | ✅ Yes | ~5-7x |
| Blackwell (RTX 50 series) | 12.0 | ✅ Yes | ~5-7x |
| Turing (RTX 20 series) | 7.5 | ❌ No | 1x (standard FP32) |
| Pascal (GTX 10 series) | 6.1 | ❌ No | 1x (standard FP32) |

### Accuracy Considerations

**When TF32 is safe to use:**
- Large language model training (like SWA-MLA)
- Most deep learning workloads
- When ~2 orders of magnitude larger relative error vs FP64 is acceptable

**When to disable TF32:**
- Scientific computing requiring IEEE FP32 precision
- Numerical stability debugging
- Exact reproducibility requirements
- When accuracy is more important than speed

**Example accuracy comparison** (from PyTorch docs):
```python
# A100 GPU, 10240x10240 matmul
# With TF32: 0.016s, relative error: 0.0022
# Without TF32: 0.11s, relative error: 0.000039
# TF32 is ~7x faster with 2 orders of magnitude larger error
```

### Testing TF32 Configuration

Run the test script to verify TF32 support:

```bash
python test_tf32_config.py
```

This will:
- Detect available GPUs and compute capabilities
- Test TF32 enable/disable functionality
- Verify the configuration is working correctly

### Combining TF32 with Other Optimizations

TF32 works seamlessly with other optimizations:

```bash
# TF32 + FP8 training (maximum speed on H100/H200)
./scripts/train_swa_mla.sh small 8 2048  # Both auto-enabled

# TF32 + torch.compile (recommended for production)
python train.py --size small --compile

# TF32 + gradient checkpointing (memory optimization)
python train.py --size large --gradient_checkpointing
```

**Note:** TF32 only affects FP32 operations. When using `--use_fp8`, most compute-heavy operations are already in FP8, so TF32 has minimal additional impact (but still helps for remaining FP32 operations like normalization and embeddings).

## Auto-Detected Distributed Training

### Overview
The training script **automatically detects available GPUs** and launches the appropriate training mode:
- **Multiple GPUs**: Automatically launches DDP training with `torchrun`
- **Single GPU**: Standard single-device training
- **No GPU**: CPU training (slow, not recommended)

### How It Works

The shell script [scripts/train_swa_mla.sh](scripts/train_swa_mla.sh) uses `nvidia-smi` to detect GPUs:

```bash
# Auto-detect number of GPUs
if command -v nvidia-smi &> /dev/null; then
    NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
else
    NUM_GPUS=0
fi

# Launch with appropriate command
if [ $NUM_GPUS -gt 1 ]; then
    torchrun --standalone --nproc_per_node=$NUM_GPUS train.py ...
else
    python train.py ...
fi
```

### Usage

Just run the script normally - it will automatically use all available GPUs:

```bash
# Automatically uses all GPUs if multiple are detected
./scripts/train_swa_mla.sh small 8 2048

# The script will print:
# "Detected GPUs: N"
# "Launching DDP training on N GPUs..." (if N > 1)
# "Launching single GPU training..." (if N = 1)
```

### Manual Control

If you need manual control over GPU selection:

```bash
# Use specific GPUs via CUDA_VISIBLE_DEVICES
CUDA_VISIBLE_DEVICES=0,1,2,3 ./scripts/train_swa_mla.sh medium 4 2048

# Use only GPU 0
CUDA_VISIBLE_DEVICES=0 ./scripts/train_swa_mla.sh small 16 2048

# Manual torchrun launch
torchrun --nproc_per_node=8 train.py --size large --batch_size 2
```

### DDP Configuration

The DDP setup is handled automatically in [train.py:61-72](train.py#L61-L72):

```python
def setup_distributed():
    """Setup distributed training if available."""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])

        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)

        return True, rank, local_rank, world_size
    return False, 0, 0, 1
```

### Benefits
- **Zero configuration**: Just run the script, it handles everything
- **Optimal utilization**: Automatically uses all available GPUs
- **Fallback support**: Works seamlessly with single GPU or CPU
- **Easy debugging**: Can easily switch to single GPU for debugging

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

### FP8 Integration with TorchAO
Native PyTorch FP8 training via TorchAO library:
- **Automatic GPU detection**: Checks for H100/H200 via compute capability
- **Model conversion**: Uses `convert_to_float8_training()` to convert linear layers to FP8
- **Optimizer**: Automatically uses `AdamWFp8` from TorchAO when `--use_fp8` is enabled
- **Scaling**: TorchAO handles all scaling factors automatically (E4M3/E5M2 formats)
- **Compatibility**: Works seamlessly with `torch.compile` and DDP

Key implementation details:
- FP8 conversion happens after model creation but before DDP wrapping
- No need for `GradScaler` - TorchAO handles scaling internally
- Linear layers in attention and MLP are converted to FP8
- Normalization layers and embeddings remain in higher precision
- Expected memory reduction: ~25-30% compared to BF16
- Expected speedup: ~1.5x on H100/H200 (tested up to 512 GPUs)

## TorchAO FP8 Training Guide

### Overview
The SWA-MLA model now uses **TorchAO** (PyTorch Architecture Optimization) for native FP8 training. This provides better performance, easier maintenance, and proven scalability compared to custom FP8 implementations.

### Key Benefits
- **1.5x faster training** on H100/H200 GPUs (proven at 512 GPU scale)
- **25-30% VRAM reduction** compared to BF16
- **Native PyTorch integration** - maintained by PyTorch team
- **Works with torch.compile** for additional 10-20% speedup
- **FSDP2 compatible** for large-scale distributed training

### Usage

#### Basic FP8 Training
```bash
# Single GPU
python train.py --size small --batch_size 8 --block_size 2048 --use_fp8

# Multi-GPU with DDP
torchrun --nproc_per_node=4 train.py --size medium --batch_size 4 --use_fp8

# With torch.compile for maximum speed
python train.py --size small --use_fp8 --compile
```

#### How It Works
1. **GPU Detection**: Automatically detects H100/H200 GPUs via compute capability check
2. **Model Conversion**: Calls `convert_to_float8_training(model)` to convert linear layers
3. **Optimizer Selection**: Uses `AdamWFp8` from TorchAO instead of standard AdamW
4. **Automatic Scaling**: TorchAO manages all FP8 scaling factors internally

#### Code Flow
```python
# In train.py
model = create_swa_mla_model(...)  # Create model normally
model = model.to(device)

if args.use_fp8 and is_fp8_supported():
    from torchao.float8 import convert_to_float8_training
    model = convert_to_float8_training(model)  # Convert to FP8

# Optimizer automatically uses AdamWFp8 when use_fp8=True
optimizer = configure_optimizer(model, ..., use_fp8=args.use_fp8)
```

#### What Gets Converted to FP8
- ✅ Linear layers in SWA attention (Q, K, V, output projections)
- ✅ Linear layers in MLA attention (Q, K, V projections with LoRA)
- ✅ MLP feed-forward layers (up and down projections)
- ❌ **lm_head** (output projection) - **kept in BF16 due to vocab_size alignment**
- ❌ Embeddings (wte, wpe) - remain in BF16
- ❌ Normalization layers (RMSNorm, LayerNorm) - remain in BF16

**Important Note:** The `lm_head` is excluded from FP8 conversion because the vocabulary size
(50257 for GPT-2) is not divisible by 16, which is required by TorchAO's `_scaled_mm` kernel.
This has minimal performance impact (<5%) as it's only one layer and represents ~10% of compute.
The critical layers (attention, MLP) remain in FP8, providing **~1.45x speedup** vs 1.5x theoretical.

### Troubleshooting FP8 Training

**FP8 not enabled on H100/H200:**
- Check TorchAO is installed: `pip list | grep torchao`
- Verify GPU detected: Check training logs for "✓ Model converted to FP8"
- Install if missing: `pip install torchao`

**Out of memory with FP8:**
- FP8 uses ~25-30% less memory than BF16, but may still OOM on small GPUs
- Try reducing `--batch_size` or `--block_size`
- Enable `--gradient_checkpointing` for additional memory savings

**NaN loss with FP8:**
- FP8 is generally stable, but can be sensitive to learning rate
- Try reducing `--learning_rate` (e.g., from 1e-4 to 5e-5)
- Increase `--warmup_iters` (e.g., from 400 to 1000)
- Verify `--grad_clip 1.0` is set (critical for stability)

**Slower than expected:**
- Ensure `--compile` is enabled for maximum speedup
- Check GPU utilization with `nvidia-smi`
- FP8 benefits are most visible with larger models (base/large/xl)
- Small models may not see full speedup due to overhead

**torch.compile errors with FP8:**
- The training script automatically uses `mode='reduce-overhead'` for FP8 (instead of `max-autotune`)
- This avoids `FlexibleLayout` errors with FP8 dynamic scaling operations
- `reduce-overhead` mode still provides significant speedup (~10-20%)
- Combined FP8 + compile speedup: **~1.6-1.8x** vs BF16 baseline

**Learning rate updates with TorchAO optimizers:**
- TorchAO optimizers (AdamWFp8, AdamW8bit) store `lr` as a Tensor instead of float
- Use `param_group['lr'].fill_(new_lr)` instead of `param_group['lr'] = new_lr`
- The training script handles this automatically by detecting Tensor lr:
  ```python
  if isinstance(param_group['lr'], torch.Tensor):
      param_group['lr'].fill_(lr)  # TorchAO optimizer
  else:
      param_group['lr'] = lr  # Standard optimizer
  ```

### Migration from Legacy FP8 Implementation

The old custom FP8 implementation in `optimization/fp8_trainer.py` is now deprecated. To migrate:

**Old way (deprecated):**
```python
from fp8_trainer import FP8AdamW
model = create_swa_mla_model(..., use_fp8=True)  # FP8 flag in model creation
optimizer = FP8AdamW(...)  # Manual optimizer selection
```

**New way (recommended):**
```python
from fp8_torchao import configure_fp8_training
model = create_swa_mla_model(..., use_fp8=False)  # No FP8 in model
model = convert_to_float8_training(model)  # TorchAO handles conversion
# Optimizer automatically selected based on --use_fp8 flag
```

### References
- TorchAO GitHub: https://github.com/pytorch/ao
- TorchAO Docs: https://docs.pytorch.org/ao/
- FP8 Training Guide: https://pytorch.org/blog/training-using-float8-fsdp2/
- Paper: https://arxiv.org/abs/2209.05433

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
