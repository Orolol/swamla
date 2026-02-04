# Data Model: FoPE + YaRN Configuration

**Feature**: 001-fope-yarn
**Date**: 2026-02-04

## Configuration Schema

### YaRN Configuration Fields

These fields are added to `SWAMLAConfig` in `models/swa_mla_model.py`:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `yarn_enabled` | `bool` | `False` | Enable YaRN context extension |
| `yarn_scale_factor` | `float` | `1.0` | Context extension ratio (new_max / original_max) |
| `yarn_original_max_seq_len` | `int` | `2048` | Original training context length |
| `yarn_beta_fast` | `float` | `32.0` | High frequency boundary (extrapolation) |
| `yarn_beta_slow` | `float` | `1.0` | Low frequency boundary (interpolation) |
| `yarn_attn_factor` | `Optional[float]` | `None` | Temperature scaling factor (auto-computed if None) |

### Validation Rules

1. `yarn_scale_factor >= 1.0` - Cannot shrink context
2. `yarn_beta_fast > yarn_beta_slow` - Valid ramp function boundaries
3. `yarn_original_max_seq_len > 0` - Positive training length
4. When `yarn_enabled=False`, all other YaRN fields are ignored

### State Transitions

```
┌─────────────────┐
│  yarn_enabled   │
│     = False     │
│  (Default RoPE) │
└────────┬────────┘
         │ User sets yarn_enabled=True
         ▼
┌─────────────────┐
│  yarn_enabled   │
│     = True      │
│  (YaRN Active)  │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│ Frequency Computation:                  │
│ 1. Compute base inv_freq                │
│ 2. Calculate wavelengths                │
│ 3. Apply NTK-by-parts interpolation     │
│ 4. Cache scaled cos/sin values          │
└────────┬────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│ Temperature Scaling:                    │
│ 1. Compute attn_factor from scale       │
│ 2. Adjust softmax_scale in MLA          │
└─────────────────────────────────────────┘
```

## Computed Values

### Attention Factor Formula

When `yarn_attn_factor` is `None`, it is computed automatically:

```python
if yarn_attn_factor is None:
    # YaRN paper formula: sqrt(1/t) = 0.1 * ln(s) + 1
    if yarn_scale_factor > 1.0:
        yarn_attn_factor = 0.1 * math.log(yarn_scale_factor) + 1.0
    else:
        yarn_attn_factor = 1.0
```

### Frequency Interpolation

For each frequency dimension `d`:

```python
# Base inverse frequency
inv_freq[d] = 1.0 / (base ** (2*d / dim))

# Wavelength
wavelength[d] = 2 * pi / inv_freq[d]

# Ratio for ramp function
ratio[d] = wavelength[d] / original_max_seq_len

# Ramp function (gamma)
if ratio[d] < beta_slow:
    gamma[d] = 0.0  # Full extrapolation (keep original)
elif ratio[d] > beta_fast:
    gamma[d] = 1.0  # Full interpolation (scale down)
else:
    gamma[d] = (ratio[d] - beta_slow) / (beta_fast - beta_slow)

# Scaled frequency
scaled_inv_freq[d] = inv_freq[d] * ((1 - gamma[d]) / scale_factor + gamma[d])
```

## Relationships

```
┌─────────────────┐      ┌─────────────────┐
│  SWAMLAConfig   │──────│  YaRN Fields    │
└────────┬────────┘      └─────────────────┘
         │
         │ creates
         ▼
┌─────────────────┐
│  SWAMLAModel    │
└────────┬────────┘
         │
         │ precomputes
         ▼
┌─────────────────┐      ┌─────────────────┐
│   freqs_cis     │◄─────│ YaRN inv_freq   │
│   (buffer)      │      │ computation     │
└────────┬────────┘      └─────────────────┘
         │
         │ used by
         ▼
┌─────────────────┐      ┌─────────────────┐
│      MLA        │──────│ softmax_scale   │
│    (layers)     │      │ (temperature)   │
└─────────────────┘      └─────────────────┘
```

## Example Configurations

### 4x Context Extension (2K → 8K)

```python
SWAMLAConfig(
    block_size=8192,
    yarn_enabled=True,
    yarn_scale_factor=4.0,
    yarn_original_max_seq_len=2048,
    # beta_fast=32.0, beta_slow=1.0 (defaults)
)
```

### 8x Context Extension (2K → 16K)

```python
SWAMLAConfig(
    block_size=16384,
    yarn_enabled=True,
    yarn_scale_factor=8.0,
    yarn_original_max_seq_len=2048,
)
```

### Custom Parameters (DeepSeek-style)

```python
SWAMLAConfig(
    block_size=32768,
    yarn_enabled=True,
    yarn_scale_factor=16.0,
    yarn_original_max_seq_len=2048,
    yarn_attn_factor=1.0 + 0.0707 * math.log(16.0),  # DeepSeek coefficient
)
```
