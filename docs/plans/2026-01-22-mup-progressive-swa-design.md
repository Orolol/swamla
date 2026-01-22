# μP + Progressive Training + SWA Design

## Overview

Training improvements for faster convergence and higher throughput on 1B-scale models. Addresses two pain points:
- **Early training inefficiency** - Model takes too long to "get going"
- **Late training plateau** - Loss stops improving before expected

**Approach:** μP-First Foundation - Build μP into the architecture, then layer progressive training and SWA on top.

---

## Section 1: μP Foundation

### What μP Solves

Standard initialization and learning rates don't transfer across model widths. A learning rate that works for a 125M model often fails at 1B. μP (Maximal Update Parametrization) defines scaling rules so that training dynamics remain consistent regardless of width.

### Scaling Rules

| Component | Init Scale | LR Scale | Weight Decay |
|-----------|-----------|----------|--------------|
| Embeddings (wte) | 1/√d | 1x (base LR) | 0 |
| Attention Q/K/V | 1/√d_in | 1/width | standard |
| Attention Output | 1/√d_in | 1/width | standard |
| MLP (up/gate) | 1/√d_in | 1/width | standard |
| MLP (down) | 1/√(width × d_in) | 1/width | standard |
| LM Head | 0 (zero init) | 1/width | 0 |
| MLA LoRA projections | 1/√d_in | 1/width | standard |
| Engram embeddings | 1/√d_mem | 1x (keep 5x mult) | 0 |

### Width Multiplier

Define a "base width" (e.g., 256). For a 1024-dim model, multiplier = 4. LR for most layers = base_lr / 4.

**Practical impact:** Tune hyperparameters on a small (256-dim) proxy model, then scale to 1024 or 2048 with the same settings.

---

## Section 2: μP Implementation

### New Configuration Fields

In `SWAMLAConfig`:

```python
# μP settings
use_mup: bool = False
mup_base_width: int = 256  # Reference width for scaling
mup_output_mult: float = 1.0  # Output logits temperature
```

### New File: `optimization/mup.py`

Three main components:

1. **`MuPScaler`** - Computes per-parameter LR multipliers based on layer type and width ratio. Returns a dict mapping param names to their scale factors.

2. **`mup_init(model, base_width)`** - Applies μP-compliant initialization:
   - Rescales existing weights by appropriate factors
   - Zero-inits the LM head (critical for μP)
   - Preserves existing zero-init for Engram conv

3. **`configure_mup_optimizer(model, base_lr, base_width, ...)`** - Creates parameter groups with scaled LRs. Integrates with existing `configure_optimizer` logic (Engram 5x multiplier, no decay for norms/embeds).

### Integration in `train.py`

```python
if config.use_mup:
    from optimization.mup import mup_init, configure_mup_optimizer
    mup_init(model, config.mup_base_width)
    optimizer = configure_mup_optimizer(model, args.learning_rate, ...)
else:
    optimizer = configure_optimizer(model, ...)  # existing path
```

### Special Handling

- MoE experts: Each expert follows MLP rules
- GatedDeltaNet: Q/K/V/O projections follow attention rules
- MLA LoRA: Treat like attention projections

---

## Section 3: Progressive Training

### Core Idea

Start with easier tasks, progressively increase difficulty. Main levers: sequence length and data complexity.

### Sequence Length Progression

| Phase | Tokens | Seq Length | Batch Size | Effective Batch |
|-------|--------|------------|------------|-----------------|
| 1 | 0 - 500M | 512 | 16 | 8K tokens |
| 2 | 500M - 2B | 1024 | 8 | 8K tokens |
| 3 | 2B+ | 2048 | 4 | 8K tokens |

**Key principle:** Keep effective batch size (tokens) constant across phases.

### Why This Helps Early Training

- Shorter sequences = more gradient updates per token
- Attention is O(n²) - 512-length is 16x cheaper than 2048
- Model learns local patterns first (which transfer to longer contexts)

### Phase Transition Strategy

**Hard switch** (recommended): Simple, abrupt transition. Works fine in practice because μP ensures stable dynamics regardless of sequence length.

### Implementation

A `ProgressiveScheduler` wraps the data loader and adjusts `block_size` based on current step. Also updates batch size to maintain constant token throughput.

---

## Section 4: Stochastic Weight Averaging (SWA)

### Core Idea

Maintain an exponential moving average (EMA) of model weights throughout training. The averaged model generalizes better and smooths out late-training oscillations.

### Continuous EMA (Chosen Approach)

- Update averaged weights every step: `θ_avg = β × θ_avg + (1-β) × θ`
- Typical β = 0.9999 (higher = slower averaging)
- No extra memory cost beyond storing one copy of weights
- Use averaged weights for validation, original for training

### When SWA Helps Most

- Late training when loss oscillates around minimum
- Breaking out of sharp minima (averaged weights find flatter regions)
- Final model for deployment (better generalization)

### Implementation: `optimization/swa.py`

```python
class EMAModel:
    def __init__(self, model, decay=0.9999):
        self.ema_params = {n: p.clone() for n, p in model.named_parameters()}

    def update(self, model):
        for n, p in model.named_parameters():
            self.ema_params[n].lerp_(p.data, 1 - self.decay)

    def apply(self, model):  # For validation
        # Temporarily swap weights
```

**Integration:** Call `ema.update(model)` after each optimizer step. Use `ema.apply(model)` before validation.

---

## Section 5: Unified Training Pipeline

### Component Interaction

```
┌─────────────────────────────────────────────────────────────────┐
│                        Training Loop                             │
├─────────────────────────────────────────────────────────────────┤
│  Step 0                                                          │
│  ├── μP: Initialize with scaled weights, zero LM head           │
│  ├── Progressive: Start at seq_len=512                          │
│  └── EMA: Clone initial weights                                  │
│                                                                  │
│  Each Step                                                       │
│  ├── Progressive: Check if phase transition needed              │
│  ├── Forward/Backward: Standard with μP-scaled LRs              │
│  ├── Optimizer: Step with per-param LR multipliers              │
│  └── EMA: Update averaged weights (θ_avg.lerp_(θ, 0.0001))      │
│                                                                  │
│  Validation (every N steps)                                      │
│  ├── Swap to EMA weights                                        │
│  ├── Evaluate at current seq_len AND max seq_len                │
│  └── Swap back to training weights                              │
│                                                                  │
│  Checkpoint                                                      │
│  ├── Save training weights + EMA weights + optimizer state      │
│  └── Save progressive scheduler state (current phase)           │
└─────────────────────────────────────────────────────────────────┘
```

### Training Script Flags

```bash
./scripts/train_mup_progressive.sh \
    --use_mup --mup_base_width 256 \
    --use_progressive --prog_schedule "512:500M,1024:2B,2048:inf" \
    --use_ema --ema_decay 0.9999
```

### Validation Note

Validate at both current phase length AND full length. This catches regressions in long-range modeling before reaching that phase.

---

## Section 6: File Structure

### New Files

```
optimization/
├── mup.py              # μP initialization + optimizer config
├── swa.py              # EMAModel class
└── progressive.py      # ProgressiveScheduler + DataLoader wrapper

scripts/
└── train_mup_progressive.sh  # Training script with all features
```

### Files to Modify

| File | Changes |
|------|---------|
| `models/swa_mla_model.py` | Add μP config fields, preset `mup-1b` |
| `train.py` | Integrate μP init, progressive scheduler, EMA updates |
| `data/data_loader_packed.py` | Support dynamic `block_size` changes |

### New Config Preset

```python
"mup-1b": dict(
    n_layer=12,
    n_embd=1024,
    n_head=16,
    use_moe=True,
    n_experts=64,
    use_engram=True,
    engram_layers=[2, 6],
    # μP settings
    use_mup=True,
    mup_base_width=256,
)
```

### Estimated Scope

- `mup.py`: ~200 lines
- `swa.py`: ~80 lines
- `progressive.py`: ~150 lines
- `train.py` changes: ~100 lines
- Total: ~530 lines new/modified

---

## Section 7: Testing and Validation

### μP Verification (Critical)

1. **Coord check:** Train tiny models (width 64, 128, 256) for 100 steps. Plot activation magnitudes per layer. They should be approximately constant across widths.

2. **LR transfer test:** Find optimal LR for width=256 model. Apply same base_lr to width=1024. Loss curves should be similar.

### Progressive Training Verification

1. **Phase transition smoothness:** Log loss before/after each transition. Small spike is okay, large spike means adjustment needed.

2. **Long-context retention:** After training at 512, validate at 2048 occasionally. Performance should not collapse.

### EMA Verification

1. **EMA vs training weights:** Compare validation loss between EMA and training weights. EMA should be equal or better (especially late in training).

2. **Decay sensitivity:** Try β=0.999, 0.9995, 0.9999. Typically 0.9999 works best for longer runs.

### Smoke Test

```bash
python train.py --size mup-tiny --use_mup --use_progressive --use_ema \
    --max_iters 1000 --batch_size 4 --block_size 512
```

---

## References

- [μP Paper: Tensor Programs V](https://arxiv.org/abs/2203.03466)
- [Progressive Training / Curriculum Learning](https://arxiv.org/abs/2108.06084)
- [SWA: Averaging Weights Leads to Wider Optima](https://arxiv.org/abs/1803.05407)
