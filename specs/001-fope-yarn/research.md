# Research: FoPE + YaRN Implementation

**Feature**: Replace RoPE with FoPE + YaRN
**Date**: 2026-02-04

## 1. YaRN Algorithm Details

### Source
- Paper: "YaRN: Efficient Context Window Extension of Large Language Models" (arXiv:2309.00071, ICLR 2024)
- Authors: Bowen Peng, Jeffrey Quesnelle, Honglu Fan, Enrico Shippole

### Core Formula: NTK-by-Parts Interpolation

The frequency adjustment applies a piecewise linear ramp function:

```
h(θ_d) = (1 - γ) * (θ_d / s) + γ * θ_d
```

Where:
- `θ_d` = original RoPE frequency for dimension d
- `s` = scaling factor (new_length / original_length)
- `γ(r)` = piecewise linear ramp function

### Ramp Function

```
γ(r) = {
  0,                if r < α (low freq boundary)
  (r - α)/(β - α),  if α ≤ r ≤ β (transition)
  1,                if r > β (high freq boundary)
}
```

Where `r` = wavelength ratio `(λ_d / L_original)`:
- **High frequencies** (short wavelengths, r < α): Keep original frequencies unchanged
- **Low frequencies** (long wavelengths, r > β): Apply full interpolation
- **Middle frequencies**: Smooth transition via the ramp

### Wavelength Calculation

```
λ_d = 2π * base^(2d/D)
```

Or using inverse frequencies: `λ_d = 2π / inv_freq[d]`

### Attention Temperature Scaling

Critical YaRN innovation - modify softmax temperature:

```
softmax(q_m^T * k_n / (t * √|D|))
```

Where `t` is computed as:
```
√(1/t) = a * ln(s) + b
```

**Default parameters**:
- `a = 0.1`, `b = 1.0` (LLaMA family)
- DeepSeek-V2 uses `a = 0.0707` (empirically tuned)

---

## 2. FoPE (Fourier Position Embedding)

### Source
- Paper: "Fourier Position Embedding: Enhancing Attention's Periodic Extension for Length Generalization" (arXiv:2412.17739, December 2024)

### Key Concept

FoPE extends RoPE by modeling each dimension as a **Fourier Series** (multiple frequency components) rather than a single frequency:

```
RoPE: Each dimension = single sinusoid
FoPE: Each dimension = dominant frequency + harmonic components
```

### Benefits Over Standard RoPE
- More stable perplexity across varying context windows
- More consistent needle-in-haystack performance
- Better length generalization by zeroing destructive frequency components

### Implementation Note
For this project, "FoPE" primarily refers to **fractional/continuous position support** - the ability to use non-integer position values. RoPE naturally supports this since `angle = position * inv_freq` works for any float position value.

---

## 3. Existing Implementation Patterns

### vLLM YaRN Implementation

```python
class YarnRotaryEmbedding:
    def __init__(
        self,
        dim: int,
        max_position_embeddings: int,
        base: float = 10000,
        scaling_factor: float = 1.0,
        original_max_position_embeddings: int = 2048,
        extrapolation_factor: float = 1.0,
        attn_factor: float = 1.0,
        beta_fast: int = 32,
        beta_slow: int = 1,
    ):
        ...
```

### HuggingFace Transformers Configuration

```python
rope_scaling = {
    "type": "yarn",
    "factor": 4.0,
    "original_max_position_embeddings": 2048,
}
```

### Frequency Scaling Computation Pattern

```python
def compute_yarn_freqs(dim, max_pos, original_max_pos, base=10000.0, beta_fast=32, beta_slow=1):
    scale = max_pos / original_max_pos

    # Compute inverse frequencies
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))

    # Compute wavelength ratios for ramp function
    # Apply NTK-by-parts: h(θ) = (1-γ)*θ/s + γ*θ

    return scaled_inv_freq
```

---

## 4. Current Codebase Analysis

### Files Affected

| File | Current Role | Changes Needed |
|------|--------------|----------------|
| `models/positional_encoding.py` | RoPE class + utility functions | Add YaRN frequency computation, FoPE support |
| `models/mla.py` | MLA attention with RoPE | Update softmax_scale for YaRN temperature |
| `models/swa_mla_model.py` | Model config + freq precomputation | Add YaRN config fields, update freq computation |
| `train.py` | Training script | Add CLI args for YaRN config |

### Current RoPE Implementation

The existing `RoPE` class in `positional_encoding.py`:
- Uses standard frequency computation: `inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))`
- Caches cos/sin values up to `max_seq_len`
- Supports dynamic cache extension via `_extend_cos_sin_cache()`

### MLA Softmax Scale

Current implementation in `mla.py:118-126`:
```python
self.softmax_scale = self.qk_head_dim ** -0.5

# For extended sequences (partial YaRN support already exists!)
rope_factor = getattr(config, 'rope_factor', 1.0)
if rope_factor > 1.0 and hasattr(config, 'original_seq_len') and hasattr(config, 'max_seq_len'):
    if config.max_seq_len > config.original_seq_len:
        mscale = getattr(config, 'mscale', 1.0)
        mscale = 0.1 * mscale * math.log(rope_factor) + 1.0
        self.softmax_scale = self.softmax_scale * mscale * mscale
```

**Finding**: The codebase already has partial YaRN temperature scaling support! We need to:
1. Add the NTK-by-parts frequency interpolation
2. Expose full YaRN configuration through SWAMLAConfig

---

## 5. Design Decisions

### Decision 1: YaRN Parameters

**Choice**: Use LLaMA-family defaults
- `alpha` (beta_slow) = 1
- `beta` (beta_fast) = 32
- Temperature coefficient `a` = 0.1, intercept `b` = 1.0

**Rationale**: These are proven parameters from the YaRN paper, and the model architecture is similar to LLaMA.

**Alternative Rejected**: DeepSeek-specific parameters - would require empirical tuning for this architecture.

### Decision 2: Implementation Location

**Choice**: Extend `positional_encoding.py` with new functions, keep RoPE class interface compatible

**Rationale**:
- Maintains backward compatibility
- Clear separation of concerns
- Easy to test independently

**Alternative Rejected**: Creating entirely new classes - would break existing checkpoints.

### Decision 3: Configuration Approach

**Choice**: Add YaRN fields to `SWAMLAConfig` with defaults that preserve RoPE behavior

```python
# New fields in SWAMLAConfig
yarn_enabled: bool = False
yarn_scale_factor: float = 1.0
yarn_original_max_seq_len: int = 2048
yarn_beta_fast: float = 32.0
yarn_beta_slow: float = 1.0
yarn_attn_factor: float = 1.0  # Temperature scaling
```

**Rationale**: Opt-in behavior ensures backward compatibility. When `yarn_enabled=False`, behavior is identical to current RoPE.

### Decision 4: Fractional Position Support

**Choice**: RoPE already supports fractional positions naturally - no changes needed

**Rationale**: The rotation formula `angle = position * inv_freq` works for any float value of position.

---

## 6. Integration with MLA

### MLA-Specific Considerations

MLA has separate head dimensions:
- `qk_nope_head_dim` = 128 (no positional encoding)
- `qk_rope_head_dim` = 64 (receives RoPE/YaRN)

YaRN only affects the RoPE dimensions, so:
- Frequency scaling applies to `qk_rope_head_dim` = 64 dimensions
- Temperature scaling affects the full attention computation

### Implementation Pattern

```python
# In positional_encoding.py
def compute_yarn_inv_freq(dim, base, scale_factor, beta_fast, beta_slow):
    """Compute YaRN-scaled inverse frequencies."""
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))

    if scale_factor <= 1.0:
        return inv_freq  # No scaling needed

    # Compute wavelengths and ramp values
    wavelengths = 2 * math.pi / inv_freq
    # ... apply NTK-by-parts interpolation

    return scaled_inv_freq
```

---

## 7. Testing Strategy

### Unit Tests
1. Verify YaRN frequency computation matches reference implementations
2. Test backward compatibility: `yarn_enabled=False` produces identical outputs
3. Test temperature scaling formula

### Integration Tests
1. Run inference on sequences 4x training length
2. Measure perplexity degradation
3. Verify no numerical instability

### Perplexity Benchmarks
- Baseline: Model trained on 2K tokens, evaluated on 2K tokens
- Extended: Same model, evaluated on 8K tokens (4x)
- Target: <15% perplexity increase

---

## 8. References

- [YaRN Paper](https://arxiv.org/abs/2309.00071)
- [EleutherAI YaRN Blog](https://blog.eleuther.ai/yarn/)
- [vLLM YaRN Implementation](https://docs.vllm.ai/en/latest/api/vllm/model_executor/layers/rotary_embedding/yarn_scaling_rope/)
- [HuggingFace RoPE Utils](https://huggingface.co/docs/transformers/main/en/internal/rope_utils)
- [FoPE Paper](https://arxiv.org/abs/2412.17739)
