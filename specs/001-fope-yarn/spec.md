# Feature Specification: Replace RoPE with FoPE + YaRN

**Feature Branch**: `001-fope-yarn`
**Created**: 2026-02-04
**Status**: Draft
**Input**: User description: "tu peux remplacer Rope pat Fope + Yarn ?"

## Overview

Replace the current Rotary Position Embeddings (RoPE) implementation with Fractional Position Embeddings (FoPE) combined with Yet another RoPE extensioN (YaRN) to enable extended context length support without significant quality degradation.

**Business Value**: Models trained with standard context windows (e.g., 2048 or 4096 tokens) will be able to perform inference on much longer sequences (e.g., 32K, 64K, or beyond) with minimal perplexity degradation, enabling new use cases like processing entire documents, long conversations, or codebases.

## User Scenarios & Testing

### User Story 1 - Extended Inference Context (Priority: P1)

A researcher wants to use a model trained on 2048 tokens to process documents that are 8x longer (16K tokens) without retraining.

**Why this priority**: This is the core value proposition - extending context length at inference time without fine-tuning is the primary use case for YaRN-style approaches.

**Independent Test**: Can be tested by running inference on a 16K token document with a model trained on 2048 tokens and verifying coherent outputs with acceptable perplexity.

**Acceptance Scenarios**:

1. **Given** a model trained on 2048-token sequences, **When** running inference on a 16K-token document, **Then** the model produces coherent outputs without attention pattern collapse
2. **Given** a 4x context extension (2K → 8K), **When** measuring perplexity on validation data, **Then** perplexity increase is less than 15% compared to in-distribution sequences
3. **Given** extended context inference, **When** monitoring memory usage, **Then** memory overhead is linear with sequence length (no exponential blowup from position encoding)

---

### User Story 2 - Configurable Scaling Factors (Priority: P2)

A developer wants to configure the context extension factor and YaRN parameters based on their specific use case and quality requirements.

**Why this priority**: Different applications have different quality vs. length tradeoffs. Configurability allows users to optimize for their needs.

**Independent Test**: Can be tested by setting different scaling factors via configuration and verifying the model respects these settings.

**Acceptance Scenarios**:

1. **Given** scaling factor set to 4x, **When** creating the model, **Then** the model can process sequences up to 4x the training length
2. **Given** YaRN alpha/beta parameters specified, **When** computing position embeddings, **Then** the embeddings use the specified interpolation parameters
3. **Given** an invalid scaling configuration (e.g., negative factor), **When** initializing the model, **Then** a clear error message is displayed

---

### User Story 3 - Backward Compatibility (Priority: P2)

A user with existing trained models wants to continue using them without modification when not needing extended context.

**Why this priority**: Existing workflows must not break. The new implementation should be a drop-in replacement with opt-in extended features.

**Independent Test**: Can be tested by loading an existing checkpoint and running inference on standard-length sequences.

**Acceptance Scenarios**:

1. **Given** an existing model checkpoint trained with RoPE, **When** loading with the new FoPE+YaRN code, **Then** inference produces identical results for sequences within training length
2. **Given** no scaling factor configured (default), **When** running inference, **Then** behavior is identical to original RoPE
3. **Given** a training run with default settings, **When** compared to RoPE training, **Then** convergence speed and final loss are within 1% of baseline

---

### User Story 4 - Training with Extended Context (Priority: P3)

A researcher wants to train a model from scratch with FoPE+YaRN enabled to potentially improve long-context capabilities.

**Why this priority**: While inference-time extension is the primary use case, some users may want to train with these embeddings from the start.

**Independent Test**: Can be tested by running a training job with FoPE+YaRN enabled and verifying loss convergence.

**Acceptance Scenarios**:

1. **Given** FoPE+YaRN enabled during training, **When** training on 2048-token sequences, **Then** training converges normally with no stability issues
2. **Given** a model trained with FoPE+YaRN, **When** evaluating on validation data, **Then** perplexity is comparable to RoPE-trained model

---

### Edge Cases

- What happens when context length exceeds maximum configurable extension? System should truncate or return an error with clear guidance.
- How does the system handle position IDs that are non-contiguous (e.g., from packed sequences)? FoPE should support fractional/arbitrary positions.
- What happens if YaRN scaling is applied to a model with very small head dimension (e.g., 32)? System should validate and warn if head dimension may cause numerical issues.

## Requirements

### Functional Requirements

- **FR-001**: System MUST provide FoPE (Fractional Position Embeddings) as an alternative to integer position RoPE
- **FR-002**: System MUST implement YaRN-style attention scaling with configurable alpha and beta parameters
- **FR-003**: System MUST support NTK-aware interpolation for frequency adjustment across different context scales
- **FR-004**: System MUST allow configuration of target context length extension factor (e.g., 4x, 8x, 16x)
- **FR-005**: System MUST maintain backward compatibility with existing RoPE-trained checkpoints when no scaling is configured
- **FR-006**: System MUST integrate with existing MLA (Multi-head Latent Attention) blocks without requiring architectural changes
- **FR-007**: System MUST support both precomputed position embeddings and on-the-fly computation for variable-length sequences
- **FR-008**: System MUST provide configuration options through the existing model configuration system (SWAMLAConfig)

### Key Entities

- **PositionEmbedding**: Abstract concept representing how position information is encoded (FoPE replaces RoPE as the concrete implementation)
- **ScalingConfiguration**: Configuration for context extension including scale factor, alpha, beta, and interpolation method
- **FrequencyCache**: Precomputed trigonometric values for efficient position embedding application

## Success Criteria

### Measurable Outcomes

- **SC-001**: Models can process sequences at least 4x longer than training length with perplexity degradation under 15%
- **SC-002**: 100% backward compatibility - existing checkpoints produce identical outputs for in-distribution sequences
- **SC-003**: Training throughput impact under 5% when using FoPE+YaRN compared to standard RoPE
- **SC-004**: Memory overhead for position embeddings scales linearly with sequence length (O(n) not O(n²))
- **SC-005**: Context extension up to 16x original training length supported without numerical instability

## Assumptions

1. **YaRN paper parameters**: Default alpha=1, beta=32 based on YaRN paper recommendations for LLaMA-style models
2. **NTK-aware scaling**: The implementation follows NTK-aware interpolation as described in the YaRN paper rather than simple linear interpolation
3. **MLA compatibility**: FoPE+YaRN needs to work specifically with the MLA attention mechanism which has separate RoPE and non-RoPE head dimensions
4. **GatedDeltaNet unchanged**: DeltaNet blocks use linear attention without RoPE, so they are unaffected by this change
5. **Default behavior**: When no scaling is specified, the system behaves identically to current RoPE implementation
6. **FoPE definition**: FoPE refers to the ability to use fractional/continuous position values rather than integer positions, enabling smoother interpolation for context extension

## Out of Scope

- Dynamic context length adaptation during inference (context length is fixed at model initialization)
- Other position embedding methods (ALiBi, Sinusoidal, etc.)
- Changes to DeltaNet/linear attention blocks
- Fine-tuning recipes for adapting models to extended context
- Quantization-specific optimizations for position embeddings
