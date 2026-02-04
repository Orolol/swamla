# Implementation Plan: FoPE + YaRN Position Embeddings

**Branch**: `001-fope-yarn` | **Date**: 2026-02-04 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/001-fope-yarn/spec.md`

## Summary

Replace the current RoPE (Rotary Position Embeddings) implementation with YaRN (Yet another RoPE extensioN) to enable context length extension at inference time. YaRN uses NTK-by-parts frequency interpolation combined with attention temperature scaling to allow models trained on short contexts (e.g., 2048 tokens) to perform inference on much longer sequences (8K-32K+) with minimal quality degradation.

**Key Insight**: The codebase already has partial YaRN support (temperature scaling in MLA). This plan completes the implementation by adding NTK-by-parts frequency interpolation and full configuration exposure.

## Technical Context

**Language/Version**: Python 3.12+, PyTorch 2.10+
**Primary Dependencies**: torch, fla (for GatedDeltaNet)
**Storage**: N/A (in-memory computation)
**Testing**: pytest, manual perplexity benchmarks
**Target Platform**: Linux (CUDA), RTX 5090 Blackwell GPU
**Project Type**: Single ML training codebase
**Performance Goals**: <5% throughput impact vs baseline RoPE
**Constraints**: Backward compatibility with existing checkpoints, O(n) memory for position embeddings
**Scale/Scope**: 4 files modified, ~200 lines of new code

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

The project constitution template is not fully configured. Following general best practices:

| Principle | Status | Notes |
|-----------|--------|-------|
| Backward Compatibility | ✅ Pass | Default config preserves RoPE behavior |
| Test Coverage | ✅ Pass | Unit tests for new functions, integration tests for perplexity |
| No Breaking Changes | ✅ Pass | Existing API preserved, new features opt-in |
| Documentation | ✅ Pass | CLAUDE.md will be updated with YaRN usage |

## Project Structure

### Documentation (this feature)

```text
specs/001-fope-yarn/
├── spec.md              # Feature specification
├── plan.md              # This file
├── research.md          # YaRN algorithm research
├── data-model.md        # Configuration schema
├── quickstart.md        # Usage guide
└── checklists/
    └── requirements.md  # Validation checklist
```

### Source Code (repository root)

```text
models/
├── positional_encoding.py   # MODIFY: Add YaRN frequency computation
├── mla.py                   # MODIFY: Integrate YaRN temperature scaling
├── mla_block.py             # NO CHANGE: Uses MLA internally
├── swa_mla_model.py         # MODIFY: Add YaRN config, update freq precomputation
└── gated_deltanet.py        # NO CHANGE: Uses linear attention, no RoPE

train.py                     # MODIFY: Add CLI arguments for YaRN config

tests/
└── test_yarn.py             # NEW: Unit tests for YaRN implementation
```

**Structure Decision**: This is a modification to existing ML model code. No new modules or packages needed - changes are localized to position encoding and configuration.

## Implementation Phases

### Phase 1: Core YaRN Implementation

**File**: `models/positional_encoding.py`

Add new functions:
1. `compute_yarn_inv_freq()` - NTK-by-parts frequency interpolation
2. `precompute_freqs_cis_yarn()` - YaRN-aware frequency precomputation
3. Update `RoPE` class to optionally use YaRN frequencies

**Key Algorithm** (NTK-by-parts):
```python
def compute_yarn_inv_freq(
    dim: int,
    base: float = 10000.0,
    scale_factor: float = 1.0,
    beta_fast: float = 32.0,
    beta_slow: float = 1.0,
    original_max_seq_len: int = 2048,
) -> torch.Tensor:
    """Compute YaRN-scaled inverse frequencies using NTK-by-parts interpolation."""
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))

    if scale_factor <= 1.0:
        return inv_freq  # No scaling needed

    # Compute wavelengths: λ_d = 2π / inv_freq[d]
    wavelengths = 2 * math.pi / inv_freq

    # Compute ratio r = λ_d / original_max_seq_len
    ratios = wavelengths / original_max_seq_len

    # Ramp function: γ(r) for NTK-by-parts
    gamma = torch.clamp((ratios - beta_slow) / (beta_fast - beta_slow), 0.0, 1.0)

    # NTK-by-parts: h(θ) = (1-γ)*θ/s + γ*θ = θ * ((1-γ)/s + γ)
    # Equivalently for inv_freq: scale by ((1-γ)/s + γ)
    scale_mult = (1 - gamma) / scale_factor + gamma
    scaled_inv_freq = inv_freq * scale_mult

    return scaled_inv_freq
```

### Phase 2: Configuration Updates

**File**: `models/swa_mla_model.py`

Add to `SWAMLAConfig`:
```python
# YaRN (Yet another RoPE extensioN) configuration
yarn_enabled: bool = False
yarn_scale_factor: float = 1.0  # new_max_len / original_max_len
yarn_original_max_seq_len: int = 2048
yarn_beta_fast: float = 32.0  # High frequency boundary
yarn_beta_slow: float = 1.0   # Low frequency boundary
yarn_attn_factor: Optional[float] = None  # Auto-computed if None
```

Update `SWAMLAModel.__init__()` to use YaRN frequencies when enabled.

### Phase 3: MLA Integration

**File**: `models/mla.py`

The softmax scale adjustment already exists (lines 118-126). Ensure it integrates with the new YaRN config:
- When `yarn_enabled=True`, compute `attn_factor` from `yarn_scale_factor`
- Apply temperature scaling: `softmax_scale *= attn_factor ** 2`

### Phase 4: Training Script Updates

**File**: `train.py`

Add CLI arguments:
```python
parser.add_argument('--yarn_enabled', action='store_true')
parser.add_argument('--yarn_scale_factor', type=float, default=1.0)
parser.add_argument('--yarn_original_max_seq_len', type=int, default=2048)
```

### Phase 5: Testing & Validation

**File**: `tests/test_yarn.py`

1. Unit tests for `compute_yarn_inv_freq()`
2. Backward compatibility test: `yarn_enabled=False` matches original RoPE
3. Integration test: Model runs with YaRN enabled

## API Contracts

### Configuration Interface

```python
# Enable YaRN for 4x context extension
config = SWAMLAConfig(
    block_size=8192,  # Extended context
    yarn_enabled=True,
    yarn_scale_factor=4.0,  # 8192 / 2048
    yarn_original_max_seq_len=2048,
)
```

### CLI Interface

```bash
# Train with YaRN enabled for 4x context extension
./scripts/train.sh --preset base 8 8192 \
    --yarn_enabled \
    --yarn_scale_factor 4.0 \
    --yarn_original_max_seq_len 2048
```

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Breaking existing checkpoints | Default `yarn_enabled=False` preserves exact RoPE behavior |
| Numerical instability at high scales | Clamp gamma values, validate beta_fast > beta_slow |
| Performance regression | Profile frequency computation, ensure caching works |
| Incorrect temperature scaling | Unit test against reference vLLM implementation |

## Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Backward compatibility | 100% | Identical outputs with `yarn_enabled=False` |
| Perplexity at 4x context | <15% increase | Benchmark on validation set |
| Throughput impact | <5% | Tokens/sec comparison |
| Memory overhead | O(n) | Profile position embedding memory |

## Dependencies

- **Blocked by**: None (self-contained feature)
- **Blocks**: Future context extension features

## Complexity Tracking

> No constitution violations. Implementation follows existing patterns.

| Decision | Rationale |
|----------|-----------|
| Extend existing RoPE class | Maintains compatibility, minimal code changes |
| Add config fields to SWAMLAConfig | Consistent with existing configuration pattern |
| Opt-in via `yarn_enabled` flag | Zero risk to existing workflows |
