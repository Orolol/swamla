# NVIDIA B200, PyTorch 2.10, and CUDA 13 Optimization Guide for LLM Pre-Training

**Document Version**: 1.0
**Last Updated**: January 2026
**Target Architecture**: NVIDIA Blackwell B200 GPUs
**Software Stack**: PyTorch 2.10 + CUDA 13.1

---

## Executive Summary

This document provides comprehensive specifications and optimization opportunities for large language model (LLM) pre-training using NVIDIA's Blackwell B200 GPUs, PyTorch 2.10, and CUDA 13.1. Key highlights:

- **B200 GPUs**: 2.2-4x faster training than H100, compute capability 10.0, 192GB HBM3e memory, 8 TB/s bandwidth
- **PyTorch 2.10**: New `varlen_attn()` for ragged sequences, combo-kernels horizontal fusion, FP8 on Intel GPUs
- **CUDA 13.1**: Revolutionary CUDA Tile programming model, Green Contexts, Memory Locality Optimization (MLOPart)

---

## 1. NVIDIA B200 (Blackwell) GPU Specifications

### 1.1 Architecture Overview

**Compute Capability**: 10.0 (Blackwell architecture)

**Core Architecture**:
- 148 Streaming Multiprocessors (SMs) total (74 SMs per die in dual-die design)
- Each SM supports up to 64 concurrent warps
- Dual-die design connected via ultra-high-speed inter-chip link
- 8x B200 GPUs fully interconnected in hybrid cube-mesh topology via NVSwitch in HGX B200

### 1.2 Memory Specifications

| Specification | Value | Comparison to H100 |
|---------------|-------|-------------------|
| **Memory Capacity** | 192GB HBM3e (some configs: 180GB) | 1.9x (H100: 80GB) |
| **Memory Bandwidth** | 8 TB/s | 2.4x (H100: 3.35 TB/s) |
| **NVLink Bandwidth** | 1.8 TB/s (NVLink 5) | 2x (H100: 900 GB/s) |
| **NVLink Configuration** | 18 links @ 100 GB/s each | 9 more links than H100 |

**Multi-GPU Topology**:
- **8-GPU HGX B200**: 14.4 TB/s total GPU-to-GPU bandwidth (bidirectional)
- **NVSwitch Fabric**: All-to-all connection, any GPU can communicate at full speed
- **NVLink 5 Benefits**: ~40% reduction in all-reduce time for 8-GPU training replicas

### 1.3 Tensor Core Capabilities

**6th Generation Tensor Cores** with new precision formats:

#### Performance by Precision (per GPU)

| Precision | Dense Performance | Sparse Performance | Use Case |
|-----------|------------------|-------------------|----------|
| **FP4** | 9 PFLOPS | 18 PFLOPS | Maximum throughput inference, trillion-param models |
| **FP6** | 4.5 PFLOPS | 9 PFLOPS | Balance of precision and speed (limited software support) |
| **FP8** | 4.5 PFLOPS | 9 PFLOPS | Training with 2nd-gen Transformer Engine (3x faster than H100) |
| **FP16/BF16** | 2.25 PFLOPS | 4.5 PFLOPS | High-precision training, stable convergence |
| **TF32** | 1.2 PFLOPS | 2.25 PFLOPS | Default FP32 training on Ampere+ GPUs |
| **FP64** | 40 TFLOPS | N/A | Scientific computing, high-precision required |

#### 2nd Generation Transformer Engine

**Key Features**:
- Automatic FP8/FP4 precision management for optimal performance
- 3x faster LLM training vs H100 (e.g., GPT-MoE-1.8T)
- 4x faster training at scale with GB200 NVL72 (72-GPU configuration)
- 30x faster real-time inference for trillion-parameter models with FP4

**FP4/FP6 Software Status (2026)**:
- FP4: Hardware ready, software tooling available via Transformer Engine
- FP6: Hardware support exists, limited software tooling
- Requires per-layer precision selection (some layers tolerate FP4, others need FP8)
- FP4 average: 8.2% perplexity degradation (varies by layer)

### 1.4 Training Performance Benchmarks

**Llama 3.1 405B Pre-training** (512 GPUs):
- B200: 2.2x faster than H100 at same scale
- GB200 NVL72: 4x faster training for LLMs at scale

**Memory Efficiency**:
- 192GB enables 25-50% larger batch sizes vs H100 (80GB)
- Reduced gradient accumulation steps = faster convergence
- Supports longer sequence lengths (8K-16K tokens) with less memory pressure

### 1.5 Optimal Batch Sizes and Memory Patterns

#### Recommended Batch Sizes (per GPU)

| Model Size | Sequence Length | BF16 Batch Size | FP8 Batch Size | Memory Usage (BF16) |
|-----------|-----------------|-----------------|----------------|---------------------|
| 1B params | 2048 tokens | 32-64 | 48-96 | ~40-80GB |
| 7B params | 2048 tokens | 8-16 | 12-24 | ~80-120GB |
| 13B params | 2048 tokens | 4-8 | 6-12 | ~100-150GB |
| 70B params | 2048 tokens | 1-2 | 2-3 | ~160-192GB |
| 70B params | 4096 tokens | 1 | 1-2 | ~180-192GB |

**Memory Pattern Optimization**:
1. **Gradient Checkpointing**: Trade 20% speed for 40-50% memory reduction
2. **FP8 Training**: 25-30% memory reduction vs BF16 (see Section 3.5)
3. **Activation Recomputation**: Selective recomputation of expensive ops
4. **Sequence Packing**: Use PyTorch 2.10's `varlen_attn()` to eliminate padding waste

#### Multi-GPU Scaling Efficiency

| GPU Count | NVLink Topology | Scaling Efficiency | Optimal Per-GPU Batch Size |
|-----------|-----------------|-------------------|---------------------------|
| 1 GPU | N/A | 100% | Max possible |
| 2-4 GPUs | Direct NVLink | 95-98% | 75% of single-GPU max |
| 8 GPUs | HGX cube-mesh | 90-95% | 60-70% of single-GPU max |
| 16+ GPUs | Multi-node | 85-92% | 50-60% of single-GPU max |

**Best Practices**:
- Start with largest batch size that fits in memory
- Use gradient accumulation for effective batch sizes >1M tokens
- Monitor GPU utilization with `nvidia-smi dmon` (target: >90%)
- Profile with `torch.profiler` to identify memory bottlenecks

### 1.6 New Features vs H100/H200

| Feature | H100 | H200 | B200 | Improvement |
|---------|------|------|------|-------------|
| Compute Capability | 9.0 | 9.0 | 10.0 | New architecture |
| Tensor Core Generation | 5th | 5th | 6th | FP4/FP6 support |
| Memory Capacity | 80GB | 141GB | 192GB | 2.4x vs H100 |
| Memory Bandwidth | 3.35 TB/s | 4.8 TB/s | 8 TB/s | 2.4x vs H100 |
| NVLink Bandwidth | 900 GB/s | 900 GB/s | 1.8 TB/s | 2x vs H100 |
| FP8 Training | 1st-gen TE | 1st-gen TE | 2nd-gen TE | 3x faster |
| Power Consumption | 700W | 700W | 1000W | 43% increase |

**Cost-Performance Analysis (2026)**:
- B200 premium: 25%+ over H200 initially
- Performance/$ at scale: B200 is 40-60% better for LLM training
- Availability: Broad availability expected Q1-Q2 2026
- Recommendation: B200 for new deployments, H200 for budget-constrained projects

---

## 2. PyTorch 2.10 Features

**Release Date**: January 21, 2026
**Release Cadence**: 1 release per 2 months (increased from quarterly)
**Python Support**: Python 3.14 (including 3.14t freethreaded build, experimental)

### 2.1 Variable-Length Attention (varlen_attn)

**Purpose**: Efficient attention for ragged/packed sequences without padding waste

#### API Overview

```python
import torch.nn.attention as nn_attn

# New varlen_attn() for packed sequences
output = nn_attn.varlen_attn(
    query,           # [total_tokens, num_heads, head_dim]
    key,             # [total_tokens, num_heads, head_dim]
    value,           # [total_tokens, num_heads, head_dim]
    cu_seqlens_q,    # [batch_size + 1] cumulative sequence lengths
    cu_seqlens_k,    # [batch_size + 1] cumulative sequence lengths
    max_seqlen_q,    # Maximum sequence length in batch
    max_seqlen_k,    # Maximum sequence length in batch
)
```

**Key Features**:
- Forward + backward pass support
- **torch.compile compatible** (critical for performance)
- Backend support: FlashAttention 2 (FA2) currently, cuDNN and FA4 planned
- Hardware: NVIDIA CUDA with A100+ GPU
- Data types: BF16, FP16 (FP8 support planned)

**Performance Benefits**:
- Eliminates padding waste (10-50% throughput gain for variable-length datasets)
- Reduced memory usage (no padding tokens stored)
- Better cache efficiency (contiguous memory layout)

**Integration with Data Loaders**:
```python
# Data loader returns packed format
batch = {
    'input_ids': packed_tokens,        # [total_tokens] flat tensor
    'cu_seqlens': cu_seqlens,         # [batch_size + 1] cumulative lengths
    'max_seqlen': max_seqlen,         # scalar
}

# Use in model forward pass
x = self.embed(batch['input_ids'])
attn_output = nn_attn.varlen_attn(
    q, k, v,
    cu_seqlens_q=batch['cu_seqlens'],
    cu_seqlens_k=batch['cu_seqlens'],
    max_seqlen_q=batch['max_seqlen'],
    max_seqlen_k=batch['max_seqlen'],
)
```

**Current Limitations**:
- A100+ GPU required (not available on older architectures)
- Limited backend support (only FA2 as of 2.10.0, cuDNN/FA4 in future releases)
- No FP8 support yet (BF16/FP16 only)

### 2.2 Compilation Improvements

#### Combo-Kernels Horizontal Fusion

**What It Does**: Combines multiple independent operations (no data dependencies) into a single GPU kernel to reduce kernel launch overhead

**Performance Gains**:
- Up to 7% geomean speedup on Dynamo benchmark suites
- Up to 20% improvement in next-token latency for LLM inference
- Most effective for models with many small, independent operations

**Example**: Fusing multiple element-wise operations
```python
# Before: 3 separate kernel launches
x = x + bias1
y = y * scale
z = torch.relu(z)

# After horizontal fusion: 1 kernel launch
# (automatically done by TorchInductor with torch.compile)
```

**How to Enable**:
```python
import torch

model = MyModel().cuda()
model = torch.compile(model, mode='default')  # Horizontal fusion enabled by default

# For maximum fusion (may increase compile time):
model = torch.compile(model, mode='max-autotune')
```

#### Other Compilation Enhancements

**Mix Order Reduction**:
- More aggressive fusion across different data types (FP32, BF16, FP8)
- Improved heuristics for fusion decision-making
- Better fusion of normalization layers with attention/MLP

**Compilation Modes** (relevant for LLM training):
```python
# Default: balanced compile time vs runtime performance
model = torch.compile(model, mode='default')

# Reduce overhead: faster for repeated small ops (good for FP8)
model = torch.compile(model, mode='reduce-overhead')

# Max autotune: best runtime, longer compile (good for inference)
model = torch.compile(model, mode='max-autotune')
```

**Expected Speedups for LLM Training**:
- Small models (1B params): 5-10% with torch.compile
- Medium models (7-13B params): 10-15% with torch.compile
- Large models (70B+ params): 15-25% with torch.compile (more fusion opportunities)

### 2.3 Attention Backends

**Scaled Dot-Product Attention (SDPA)** remains the primary API:

```python
import torch.nn.functional as F

# Standard SDPA (auto-selects backend)
attn_output = F.scaled_dot_product_attention(
    query, key, value,
    attn_mask=mask,
    is_causal=True,  # Causal masking for autoregressive LMs
)
```

**Backend Selection** (PyTorch 2.10):
1. **FlashAttention 2** (default on A100+, BF16/FP16)
2. **cuDNN Flash Attention** (alternative backend, requires CUDA 12.8+)
3. **Memory-efficient attention** (fallback for complex masks)
4. **Math attention** (PyTorch native, slowest but most flexible)

**Backend Control**:
```python
from torch.nn.attention import sdpa_kernel, SDPBackend

# Force FlashAttention 2
with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
    output = F.scaled_dot_product_attention(q, k, v)

# Auto-select best backend
with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
    output = F.scaled_dot_product_attention(q, k, v)
```

**Performance Recommendations**:
- Use `is_causal=True` instead of custom causal masks (enables FlashAttention optimizations)
- Avoid complex attention masks that prevent FlashAttention usage
- For variable-length sequences, use `varlen_attn()` instead of SDPA with padding

### 2.4 FP8 Training Improvements

**Intel GPU FP8 Support** (new in PyTorch 2.10):
- Basic FP8 operators (add, mul, matmul, etc.) on Intel Arc/Data Center GPUs
- Scaled matrix multiplication (`_scaled_mm`)
- Type promotion and shape operators for FP8 tensors

**NVIDIA GPU FP8** (existing, no major changes in 2.10):
- Continue using native PyTorch FP8 (`torch.float8_e4m3fn`) or Transformer Engine
- GradScaler required for proper gradient scaling (see project CLAUDE.md)
- Expected 25-30% memory reduction vs BF16

**Recommendations for DeltaNet-MLA**:
1. Use native PyTorch FP8 implementation (see `optimization/fp8_native.py`)
2. Enable GradScaler: `scaler = torch.amp.GradScaler('cuda', enabled=True)`
3. Use `torch.compile(mode='reduce-overhead')` with FP8 (avoids layout issues)
4. Monitor gradient norms to ensure convergence (log every 100 steps)

### 2.5 Distributed Training Enhancements

#### FSDP2 (Fully Sharded Data Parallel v2)

**New Architecture**:
- **DTensor-based sharding**: Per-parameter dim-0 sharding (simpler than FSDP1's flat-parameter approach)
- **Improved memory management**: Avoids `torch.Tensor.record_stream`, deterministic memory usage
- **Manual control APIs**: Expose prefetching and collective scheduling for power users
- **Faster checkpointing**: `SHARDED_STATE_DICT` with no extra communication, async checkpointing support

**Key Improvements vs FSDP1**:

| Feature | FSDP1 | FSDP2 |
|---------|-------|-------|
| Sharding Approach | Flat-parameter (concat+chunk) | Per-parameter (DTensor dim-0) |
| Memory Management | `record_stream` + blocking | Deterministic, non-blocking |
| Checkpointing | Cross-rank communication | Rank-local shards only |
| Mixed Precision | Complex setup | Native FP8 + BF16 mixing |
| Partial Freezing | Difficult | Simple (DTensor-based) |
| Async Checkpointing | No | Yes |

**Usage Example**:
```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

# FSDP2 with DTensor sharding
model = FSDP(
    model,
    auto_wrap_policy=transformer_auto_wrap_policy,
    sharding_strategy="FULL_SHARD",  # FSDP2 uses DTensor internally
    mixed_precision=mixed_precision_policy,
    device_id=torch.cuda.current_device(),
)

# Checkpointing (rank-local, no communication)
from torch.distributed.checkpoint import save_state_dict, load_state_dict

save_state_dict(
    state_dict=model.state_dict(),
    storage_writer=...,
    checkpoint_id="checkpoint_1000",
)
```

**When to Use FSDP2 vs DDP**:
- **DDP**: Models fit in single GPU memory (up to ~20GB model size on B200)
- **FSDP2**: Models exceed single GPU memory (70B+ params), or need memory efficiency

**Expected Performance**:
- FSDP2 overhead: 5-10% slower than DDP for small models
- Memory savings: 8x reduction (8-GPU setup) for optimizer states + gradients
- Scaling efficiency: 85-95% up to 128 GPUs with NVLink

#### DDP Improvements

No major changes in PyTorch 2.10 for DDP, but benefits from:
- Faster NCCL operations (see Section 4.3)
- Better integration with `torch.compile`
- Combo-kernels horizontal fusion reduces per-rank compute time

### 2.6 Memory Management Improvements

**Caching Allocator Enhancements**:
- Better fragmentation handling for variable-length sequences
- Reduced overhead for small allocations (<1MB)
- Improved CUDA graph compatibility

**Memory Profiling** (improved in 2.10):
```python
from torch.profiler import profile, ProfilerActivity

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True,
    with_stack=True,
) as prof:
    model(input_ids)

print(prof.key_averages().table(sort_by="cuda_memory_usage", row_limit=10))
```

**Best Practices for LLM Training**:
- Use `torch.cuda.empty_cache()` between train/validation loops
- Monitor fragmentation with `torch.cuda.memory_stats()`
- Consider CUDA graphs for inference (not supported with gradient checkpointing)

---

## 3. CUDA 13 Features

**CUDA 13.0 Release**: August 2025
**CUDA 13.1 Release**: December 2025
**Current Version**: CUDA 13.1 Update 1 (January 2026)

### 3.1 Architecture Support

**Supported Architectures**:
- **Blackwell** (CC 10.0, 12.0): B200, GB200, GB300, RTX 50 series, RTX PRO Blackwell
- **Hopper** (CC 9.0): H100, H200
- **Ada Lovelace** (CC 8.9): RTX 40 series, L40S
- **Ampere** (CC 8.0-8.7): A100, A10, RTX 30 series
- **Turing** (CC 7.5): RTX 20 series, T4

**Important**: CUDA 13.0 removed support for offline compilation of GPUs with CC < 7.5 (Pascal and older). Runtime support remains for Pascal, but architectures prior to Turing (CC 7.5) are considered feature-complete.

### 3.2 CUDA Tile Programming Model

**Overview**: CUDA 13.1 introduces the largest update to CUDA in 20 years with CUDA Tile, a revolutionary tile-based programming model that abstracts tensor core usage.

#### Key Components

**1. CUDA Tile IR (Virtual ISA)**:
- MLIR-based intermediate representation for tile-based kernels
- Compiler infrastructure for automatic optimization
- Forward-compatible across future GPU architectures
- Targets tensor core units directly

**2. cuTile Python DSL**:
- Domain-specific language for authoring high-performance GPU kernels in Python
- Abstracts away low-level tensor core details
- C++ support planned for future releases

#### Programming Model

**Traditional CUDA** (element-by-element):
```cuda
__global__ void gemm_naive(float *A, float *B, float *C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}
```

**CUDA Tile** (tile-based):
```python
import cutile as ct

@ct.tile_kernel
def gemm_tile(A: ct.Tile[M, K], B: ct.Tile[K, N], C: ct.Tile[M, N]):
    # Specify computation on tiles, compiler handles tensor cores
    C[:, :] = A @ B  # Compiler auto-generates tensor core code
```

#### Benefits for LLM Training

**1. Tensor Core Abstraction**:
- No manual wmma or mma.sync instructions
- Compiler automatically selects optimal tensor core layout
- Forward-compatible with future architectures (no rewrite needed for Rubin, Feynman)

**2. Performance Portability**:
- Same tile code runs on Blackwell (CC 10.x/12.x) today
- Will run on future architectures without changes
- Compiler optimizations improve automatically with new CUDA releases

**3. Development Velocity**:
- Write kernels in Python (cuTile Python DSL)
- 10-100x faster kernel development vs hand-written CUDA
- Easier to optimize (just tune tile sizes and layouts)

#### Current Limitations (2026)

- **Hardware Support**: Blackwell (CC 10.x, 12.x) only
- **Language Support**: Python only (C++ coming in future releases)
- **Ecosystem Maturity**: Early stage, limited library support
- **Performance**: May not match hand-tuned CUDA kernels yet (improving rapidly)

#### Recommendations for DeltaNet-MLA

**When to Use CUDA Tile**:
- Custom attention kernels (beyond what PyTorch/FlashAttention provide)
- Novel operators (e.g., GatedDeltaNet optimizations)
- Research prototypes needing tensor core performance

**When to Stick with PyTorch**:
- Standard operations (matmul, attention, MLP) - PyTorch + torch.compile is sufficient
- Rapid iteration - PyTorch is faster for development
- Multi-architecture support - PyTorch runs everywhere, CUDA Tile is Blackwell-only

### 3.3 Green Contexts

**Purpose**: Lightweight alternative to traditional CUDA contexts for finer-grained spatial partitioning and resource provisioning on GPUs.

**Key Features**:
- Lower overhead than full CUDA contexts
- Better resource isolation between concurrent workloads
- Improved multi-tenant GPU sharing
- Available in CUDA runtime API (CUDA 13.1+)

**Use Cases**:
- Multi-tenant inference servers (multiple models on one GPU)
- Dynamic resource allocation (scale resources per request)
- Fine-grained QoS control (priority lanes for different workloads)

**Relevance to LLM Training**:
- Limited use for single-model pre-training
- More relevant for multi-model or multi-task training
- Potential for dynamic batch size adjustment based on sequence length

**Example** (conceptual):
```c
// Traditional CUDA context (heavyweight)
CUcontext ctx;
cuCtxCreate(&ctx, 0, device);

// Green context (lightweight, CUDA 13.1+)
CUgreenCtx gctx;
cuGreenCtxCreate(&gctx, CU_GREEN_CTX_DEFAULT_STREAM, device, ...);
```

### 3.4 Memory Locality Optimization (MLOPart)

**Purpose**: Create specialized CUDA devices optimized for improving memory locality on NVIDIA Blackwell GPUs.

**How It Works**:
- Partition a physical GPU into multiple logical "devices"
- Each partition has optimized memory locality for specific memory regions
- Reduces cross-partition memory traffic

**Use Cases**:
- Large models with distinct memory access patterns (encoder vs decoder)
- Pipeline parallelism (each stage on different partition)
- Workloads with clear memory region separation

**Blackwell-Specific**:
- MLOPart is only available on Blackwell GPUs with specific configurations
- Requires CUDA 13.1+ and compatible GPU firmware

**Relevance to DeltaNet-MLA**:
- Could benefit hybrid DeltaNet+MLA architecture (different memory patterns)
- DeltaNet: Sequential recurrent state access (high locality)
- MLA: Global attention access (lower locality)
- Potential: Place DeltaNet blocks on one partition, MLA on another

**Considerations**:
- Experimental feature, limited documentation
- May require custom data loading and model parallelism
- Profile before adopting (complexity may outweigh benefits)

### 3.5 Math Library Improvements

**cuBLAS Enhancements** (CUDA 13.0):

**1. Improved Performance** on Blackwell:
- SYRK, HERK, TRMM, SYMM kernels optimized for FP32/CF32
- 10-30% speedup for non-GEMM Level-3 BLAS operations

**2. Auto-Tuning API**:
```c
// New in CUDA 13.0: automatic algorithm selection
cublasGemmEx(
    handle, CUBLAS_OP_N, CUBLAS_OP_N,
    m, n, k,
    &alpha, A, CUDA_R_16BF, lda,
            B, CUDA_R_16BF, ldb,
    &beta,  C, CUDA_R_16BF, ldc,
    CUBLAS_COMPUTE_32F,
    CUBLAS_GEMM_AUTOTUNE  // Benchmarks internally, selects optimal algo
);
```

**Impact on PyTorch**:
- PyTorch uses cuBLAS internally for matmul operations
- Auto-tuning may provide 5-10% speedup for linear layers
- Most benefit for irregular matrix sizes (non-powers-of-2)

**cuSOLVER Enhancements** (CUDA 13.0):
- `DnXgeev` for efficient general eigenvalue decomposition on NVIDIA GPUs
- Relevant for advanced optimization techniques (e.g., natural gradient descent)

### 3.6 Developer Tools

**Compile Time Advisor (ctadvisor)** - New in CUDA 13.0:

**Purpose**: Analyze CUDA C++ compilation time and provide actionable suggestions to reduce build time.

**Usage**:
```bash
# Analyze compilation of a CUDA source file
nvcc -Xcompiler -ftime-trace -c my_kernel.cu

# Generate report with ctadvisor
ctadvisor my_kernel.json
```

**Benefits**:
- Identifies slow-to-compile templates and headers
- Suggests refactoring opportunities (e.g., move to separate translation units)
- Useful for large projects with many custom CUDA kernels

**Relevance to DeltaNet-MLA**:
- Limited (most compute is in PyTorch, not custom CUDA)
- Useful if developing custom FP8 kernels or DeltaNet optimizations

---

## 4. Multi-GPU Communication (NCCL)

**Current Version**: NCCL 2.29.2 (January 2026)
**Supported CUDA Versions**: CUDA 12.2, 12.4, 12.8 (CUDA 13 support expected soon)

### 4.1 NCCL Enhancements

#### Device-Side APIs (Experimental)

**Feature**: Integrate communication directly into application kernels

**Benefits**:
- Reduced kernel launch overhead (no separate communication kernels)
- Lower latency for fine-grained communication patterns
- Better overlap of compute and communication

**APIs**:
- **LSA (Load/Store Access)**: CUDA P2P communication over NVLink and some PCIe platforms
- **Multimem**: Hardware multicast using NVLink SHARP

**Example Use Case**: Overlapping attention computation with gradient all-reduce
```c
// Conceptual: compute attention and start gradient sync in same kernel
__global__ void attention_with_sync(
    float *Q, float *K, float *V, float *output,
    float *gradients,  // gradients to all-reduce
    ncclComm_t comm
) {
    // Compute attention
    compute_attention(Q, K, V, output);

    // Start gradient all-reduce (device-side API)
    ncclAllReduceDevice(gradients, ..., comm);
}
```

#### NVLink Optimizations

**Automatic Topology Detection**:
- NCCL detects NVLink domains and GPU connectivity
- Selects optimal algorithms based on:
  - Number of GPUs
  - NVLink domain size (e.g., 8 GPUs in HGX B200)
  - NVLink speed (1.8 TB/s for NVLink 5)
  - PCIe topology (fallback for non-NVLink connections)

**NVLink SHARP (NVLS)**:
- Hardware multicast for efficient collectives (all-reduce, broadcast)
- Reduces memory usage vs traditional all-reduce
- Automatically enabled on supported systems (GB200 NVL72)

**Performance Improvements**:
- Accelerated intra-node NVLink detection (faster initialization)
- Up to 40% reduction in all-reduce time with NVLink 5 (8-GPU replicas)

### 4.2 Multi-Node Scaling

**GB200 NVL72 Optimization**:
- NCCL automatically detects 72-GPU NVLink domains
- Optimizes within-domain vs cross-domain communication
- Uses NVSwitch fabric for all-to-all connectivity

**Expected Scaling Efficiency** (LLM training):

| GPU Count | Configuration | NCCL All-Reduce Time | Scaling Efficiency |
|-----------|--------------|---------------------|-------------------|
| 8 GPUs | Single HGX B200 | Baseline | 100% |
| 16 GPUs | 2x HGX B200, InfiniBand | 1.2x baseline | 95-98% |
| 64 GPUs | 8x HGX B200, InfiniBand | 1.5x baseline | 90-93% |
| 72 GPUs | GB200 NVL72 (single domain) | 1.3x baseline | 93-96% |
| 128 GPUs | 16x HGX B200, InfiniBand | 2.0x baseline | 85-90% |

**Recommendations for Multi-Node**:
- Use InfiniBand (400 Gbps+) or RoCE for inter-node connectivity
- Enable NCCL_IB_GID_INDEX for InfiniBand
- Tune NCCL_MIN_NCHANNELS and NCCL_MAX_NCHANNELS for your topology
- Monitor communication overhead with NCCL_DEBUG=INFO

### 4.3 Integration with PyTorch DDP/FSDP

**PyTorch 2.10 + NCCL 2.29**:
- Automatic NCCL backend initialization with DDP/FSDP
- NVLink detection and optimal collective algorithm selection
- Gradient bucketing (default: 25MB buckets) for efficient communication

**DDP Optimizations**:
```python
from torch.nn.parallel import DistributedDataParallel as DDP

model = DDP(
    model,
    device_ids=[local_rank],
    broadcast_buffers=False,  # Skip buffer broadcast if not needed
    gradient_as_bucket_view=True,  # Avoid gradient copy (faster)
    find_unused_parameters=False,  # Set True if some params unused
)
```

**FSDP2 Optimizations**:
```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

model = FSDP(
    model,
    sharding_strategy="FULL_SHARD",  # All-gather + reduce-scatter
    limit_all_gathers=False,  # FSDP2 doesn't need this (better memory mgmt)
    forward_prefetch=True,  # Overlap communication with compute
    backward_prefetch="BACKWARD_PRE",  # Prefetch before backward pass
)
```

---

## 5. Optimization Recommendations for DeltaNet-MLA

### 5.1 Hardware Setup

**Recommended GPU Configuration**:
- **8x NVIDIA B200** in HGX configuration (hybrid cube-mesh NVLink topology)
- **Memory**: 192GB per GPU (total 1.5TB for 8-GPU system)
- **Interconnect**: NVLink 5 (1.8 TB/s per GPU, 14.4 TB/s total)
- **InfiniBand** (optional): For multi-node scaling (400 Gbps+ recommended)

**Power and Cooling**:
- B200 TDP: 1000W per GPU (8000W total for 8-GPU system)
- Requires robust cooling (liquid cooling recommended)
- PCIe Gen5 x16 for maximum throughput

### 5.2 Software Stack

**Recommended Versions**:
- PyTorch 2.10 (or latest 2.x)
- CUDA 13.1 Update 1
- NCCL 2.29.2 (or latest 2.x)
- Python 3.11 or 3.12 (3.14 experimental for torch.compile)

**Environment Setup**:
```bash
# Install PyTorch 2.10 with CUDA 13.1
pip install torch==2.10.0+cu131 --index-url https://download.pytorch.org/whl/cu131

# Install NCCL 2.29 (if not bundled with PyTorch)
# Usually included with PyTorch, but can install separately if needed

# Verify versions
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}')"
```

### 5.3 Training Configuration

#### Precision Strategy

**FP8 Training** (recommended for B200):
```python
# train.py
model = create_swa_mla_model(..., use_fp8=False)  # Create in BF16
model = model.to(device)

if args.use_fp8:
    from optimization.fp8_native import convert_to_native_fp8
    model = convert_to_native_fp8(model)

# Critical: GradScaler must be enabled for FP8
scaler = torch.amp.GradScaler('cuda', enabled=args.use_fp8)

# Training loop
for batch in dataloader:
    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        loss = model(input_ids, labels=labels)

    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)  # Critical for correct grad magnitudes
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()
```

**Expected Memory and Speed**:
- FP8: 25-30% less memory vs BF16, 10-20% faster on B200
- BF16: Stable baseline, use if FP8 convergence issues
- TF32: Automatically enabled on B200 for FP32 ops (3-7x speedup)

#### Batch Size and Sequence Length

**Recommended Settings** (per GPU):

| Model Size | Precision | Batch Size | Seq Length | Gradient Accum Steps | Effective Batch Size |
|-----------|-----------|------------|------------|---------------------|---------------------|
| 1B params | FP8 | 64 | 2048 | 2 | 128 (16M tokens) |
| 1B params | BF16 | 48 | 2048 | 2 | 96 (12M tokens) |
| 7B params | FP8 | 16 | 2048 | 4 | 64 (8M tokens) |
| 7B params | BF16 | 12 | 2048 | 4 | 48 (6M tokens) |
| 13B params | FP8 | 8 | 2048 | 8 | 64 (8M tokens) |
| 13B params | BF16 | 6 | 2048 | 8 | 48 (6M tokens) |

**Adjust Based On**:
- Available memory (monitor with `nvidia-smi`)
- Gradient accumulation (higher = more memory for activations)
- Gradient checkpointing (reduces memory by ~40%, slows by ~20%)

#### Compilation Settings

**torch.compile Configuration**:
```bash
# For FP8 training (avoid layout issues)
python train.py --size large --use_fp8 --compile --compile_mode reduce-overhead

# For BF16 training (maximum speedup)
python train.py --size large --compile --compile_mode max-autotune
```

**Expected Speedup**:
- 1B params: 5-10% with torch.compile
- 7-13B params: 10-15% with torch.compile
- 70B+ params: 15-25% with torch.compile

**First-Iteration Compilation Time**:
- Add ~2-5 minutes for initial compilation (one-time cost)
- Subsequent iterations run at full speed
- Use `TORCH_LOGS=recompiles` to debug recompilation issues

#### Variable-Length Attention

**Enable varlen_attn for MLA Blocks**:
```bash
python train.py --size large --use_varlen_attn --batch_size 8
```

**Requirements**:
- PyTorch 2.10+
- A100+ GPU
- BF16 or FP16 data type
- Packed data loader (returns `cu_seqlens` and `max_seqlen`)

**Expected Benefits**:
- 10-50% throughput gain (depends on sequence length variance)
- Reduced memory usage (no padding tokens)
- Better for datasets with high length variance (e.g., code, web text)

### 5.4 Multi-GPU Training

**DDP Training** (recommended for models ≤ 20GB):
```bash
# Auto-detect GPUs and launch DDP
./scripts/train.sh --preset engram-moe 8 2048

# Manual DDP launch with 8 GPUs
torchrun --nproc_per_node=8 train.py --size large --batch_size 4 --use_fp8
```

**FSDP2 Training** (for models > 20GB per GPU):
```bash
# FSDP2 with full sharding
torchrun --nproc_per_node=8 train.py \
    --size xl \
    --batch_size 2 \
    --use_fsdp \
    --fsdp_sharding_strategy FULL_SHARD
```

**Multi-Node Training** (16+ GPUs):
```bash
# Node 0 (master)
torchrun --nnodes=2 --nproc_per_node=8 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=NODE0_IP:29500 \
    train.py --size xl --batch_size 2

# Node 1 (worker)
torchrun --nnodes=2 --nproc_per_node=8 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=NODE0_IP:29500 \
    train.py --size xl --batch_size 2
```

**NCCL Tuning** (environment variables):
```bash
# Optimal settings for 8x B200 with NVLink 5
export NCCL_IB_DISABLE=1  # Disable InfiniBand for single-node
export NCCL_P2P_LEVEL=NVL  # Use NVLink for P2P
export NCCL_MIN_NCHANNELS=16  # More channels for high bandwidth
export NCCL_MAX_NCHANNELS=32

# For multi-node with InfiniBand
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3  # Adjust based on your IB config
export NCCL_SOCKET_IFNAME=ib0  # InfiniBand interface name
```

### 5.5 Memory Optimization

**Gradient Checkpointing**:
```bash
# Enable for large models (trade 20% speed for 40% memory)
python train.py --size xl --gradient_checkpointing
```

**Optimizer Selection**:
```bash
# Lion optimizer (50% less memory than AdamW)
python train.py --size large --optimizer_type lion

# AdamW8bit (2x memory reduction, slight accuracy loss)
python train.py --size large --optimizer_type adamw8bit
```

**Memory Monitoring**:
```bash
# Monitor GPU memory every 2 seconds
watch -n 2 nvidia-smi

# Detailed memory profiling
python -c "
import torch
from models.swa_mla_model import create_swa_mla_model

model = create_swa_mla_model('large', vocab_size=50304, block_size=2048)
print(torch.cuda.memory_summary())
"
```

### 5.6 Performance Benchmarking

**Baseline Benchmark** (single GPU, BF16):
```bash
python train.py --size medium --batch_size 8 --block_size 2048 \
    --max_iters 100 --log_interval 10
```

**Optimized Benchmark** (single GPU, FP8 + compile):
```bash
python train.py --size medium --batch_size 12 --block_size 2048 \
    --use_fp8 --compile --compile_mode reduce-overhead \
    --max_iters 100 --log_interval 10
```

**8-GPU DDP Benchmark**:
```bash
torchrun --nproc_per_node=8 train.py \
    --size medium --batch_size 8 --block_size 2048 \
    --use_fp8 --compile --compile_mode reduce-overhead \
    --max_iters 100 --log_interval 10
```

**Expected Throughput** (tokens/sec, medium model ~400M params):

| Configuration | BF16 | BF16 + compile | FP8 + compile |
|--------------|------|----------------|---------------|
| 1x B200 | ~35K | ~45K | ~55K |
| 8x B200 (DDP) | ~260K | ~320K | ~400K |
| 64x B200 (multi-node) | ~1.8M | ~2.2M | ~2.8M |

### 5.7 Debugging and Troubleshooting

**Enable Detailed Logging**:
```bash
# NCCL debug info
export NCCL_DEBUG=INFO

# PyTorch distributed debug
export TORCH_DISTRIBUTED_DEBUG=DETAIL

# Compile debug info
export TORCH_LOGS=recompiles,graph_breaks

# Run training
python train.py --size small --log_interval 1
```

**Common Issues**:

**1. CUDA Out of Memory**:
- Reduce `--batch_size`
- Enable `--gradient_checkpointing`
- Use `--optimizer_type lion` (50% less memory)
- Try FP8 (25-30% less memory)

**2. NaN Loss**:
- Verify `--grad_clip 1.0` (critical for DeltaNet-MLA)
- Reduce `--learning_rate`
- Increase `--warmup_iters`
- Check GradScaler is enabled for FP8

**3. Slow Training**:
- Add `--compile` (10-20% speedup)
- Increase `--batch_size` if memory allows
- Check GPU utilization (`nvidia-smi dmon`)
- Profile with `torch.profiler` to find bottlenecks

**4. NCCL Hangs** (multi-GPU):
- Check all ranks are launched (use `ps aux | grep train.py`)
- Verify NCCL environment variables
- Test with `NCCL_DEBUG=INFO` to see communication patterns
- Ensure all GPUs are visible (`nvidia-smi`)

---

## 6. Roadmap and Future Improvements

### 6.1 Short-Term (Q1-Q2 2026)

**PyTorch 2.11/2.12** (expected Q2 2026):
- FlashAttention 4 (FA4) backend for `varlen_attn()`
- cuDNN Flash Attention backend for `varlen_attn()`
- FP8 support for `varlen_attn()`
- Further horizontal fusion improvements

**CUDA 13.2** (expected Q2 2026):
- Broader CUDA Tile architecture support (Hopper, Ada)
- C++ support for CUDA Tile (cuTile C++)
- MLOPart enhancements for more GPU configurations

**NCCL 2.30+** (expected Q1 2026):
- CUDA 13 official support
- Further NVLink 5 optimizations
- Better multi-tenant GPU support with Green Contexts

### 6.2 Medium-Term (H2 2026)

**PyTorch 3.0** (expected late 2026):
- Native FP4/FP6 support (leverage B200 Transformer Engine)
- Improved FSDP2 with better memory efficiency
- Enhanced torch.compile with more aggressive fusion

**CUDA 14** (expected late 2026):
- Next-gen architectures (Rubin, compute capability 11.x)
- CUDA Tile maturity (production-ready)
- Advanced memory management (MLOPart everywhere)

**Hardware**:
- NVIDIA Rubin architecture GPUs (successor to Blackwell)
- Improved NVLink (NVLink 6?)
- Higher memory bandwidth (10+ TB/s)

### 6.3 Recommendations for Forward Compatibility

**Code Practices**:
1. Use PyTorch high-level APIs (`F.scaled_dot_product_attention`, `nn_attn.varlen_attn()`)
2. Avoid custom CUDA unless absolutely necessary
3. Leverage `torch.compile` for automatic optimization
4. Use standard distributed training (DDP, FSDP2) instead of custom collectives

**Architecture Choices**:
1. Design models to work with variable sequence lengths (use `varlen_attn`)
2. Ensure models are FSDP2-compatible (avoid global state)
3. Profile on target hardware (B200) before large-scale training
4. Document precision requirements (FP8, BF16, FP32) per layer

**Deployment**:
1. Containerize training (Docker, Singularity) for reproducibility
2. Pin software versions (PyTorch, CUDA, NCCL) in requirements
3. Use checkpointing frequently (every 1000-5000 steps)
4. Monitor hardware health (GPU errors, NVLink status) continuously

---

## 7. References and Resources

### 7.1 NVIDIA B200 Documentation

- [Blackwell Architecture Wikipedia](https://en.wikipedia.org/wiki/Blackwell_(microarchitecture))
- [NVIDIA B200 Technical Analysis by Chips and Cheese](https://chipsandcheese.com/p/nvidias-b200-keeping-the-cuda-juggernaut)
- [NVIDIA Blackwell B200 Datasheet (PDF)](https://www.primeline-solutions.com/media/categories/server/nach-gpu/nvidia-hgx-h200/nvidia-blackwell-b200-datasheet.pdf)
- [Comparing Blackwell vs Hopper (Exxact Blog)](https://www.exxactcorp.com/blog/hpc/comparing-nvidia-tensor-core-gpus)
- [NVIDIA DGX B200 Official Page](https://www.nvidia.com/en-us/data-center/dgx-b200/)
- [B200 vs H200 Comparison (Northflank Blog)](https://northflank.com/blog/b200-vs-h200)
- [Choosing GPU Infrastructure for LLM Training (WhiteFiber)](https://www.whitefiber.com/blog/choosing-gpu-infrastructure)
- [NVIDIA Blackwell Platform Announcement](https://nvidianews.nvidia.com/news/nvidia-blackwell-platform-arrives-to-power-a-new-era-of-computing)

### 7.2 PyTorch 2.10 Documentation

- [PyTorch 2.10 Release Blog (Official)](https://pytorch.org/blog/pytorch-2-10-release-blog/)
- [PyTorch 2.10 Documentation](https://docs.pytorch.org/docs/stable/)
- [torch.nn.attention.varlen_attn Documentation](https://docs.pytorch.org/docs/stable/nn.attention.html)
- [Scaled Dot-Product Attention Tutorial](https://docs.pytorch.org/tutorials/intermediate/scaled_dot_product_attention_tutorial.html)
- [FSDP2 Tutorial](https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html)
- [torch.compile Documentation](https://docs.pytorch.org/docs/stable/generated/torch.compile.html)
- [PyTorch GitHub Releases](https://github.com/pytorch/pytorch/releases)

### 7.3 CUDA 13 Documentation

- [CUDA 13.0 Release Blog (NVIDIA)](https://developer.nvidia.com/blog/whats-new-and-important-in-cuda-toolkit-13-0/)
- [CUDA 13.1 Release Blog (NVIDIA)](https://developer.nvidia.com/blog/nvidia-cuda-13-1-powers-next-gen-gpu-programming-with-nvidia-cuda-tile-and-performance-gains)
- [CUDA 13.1 Release Notes (Official)](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html)
- [CUDA Tile Official Page](https://developer.nvidia.com/cuda/tile)
- [CUDA Tile GitHub Repository](https://github.com/NVIDIA/cuda-tile)
- [Focus on Your Algorithm—CUDA Tile Handles the Hardware](https://developer.nvidia.com/blog/focus-on-your-algorithm-nvidia-cuda-tile-handles-the-hardware/)
- [NVIDIA CUDA Tile Examined (Tom's Hardware)](https://www.tomshardware.com/pc-components/gpus/nvidias-cuda-tile-examined-ai-giant-releases-programming-style-for-rubin-feynman-and-beyond-tensor-native-execution-model-lays-the-foundation-for-blackwell-and-beyond)

### 7.4 NCCL and Multi-GPU

- [NVIDIA NCCL GitHub](https://github.com/NVIDIA/nccl)
- [NCCL Official Developer Page](https://developer.nvidia.com/nccl)
- [NCCL Release Notes (PDF)](https://docs.nvidia.com/deeplearning/nccl/pdf/NCCL-Release-Notes.pdf)
- [NVIDIA GB200 NVL Multi-Node Tuning Guide](https://docs.nvidia.com/multi-node-nvlink-systems/multi-node-tuning-guide/nccl.html)
- [Demystifying NCCL (arXiv Paper)](https://arxiv.org/html/2507.04786v1)

### 7.5 Additional Resources

- [PyTorch Supercharging Training with float8 and FSDP2](https://pytorch.org/blog/training-using-float8-fsdp2/)
- [Accelerating PyTorch Training with FP8 (Medium)](https://medium.com/data-science/accelerating-pytorch-training-workloads-with-fp8-5a5123aec7d7)
- [PyTorch Native FP8 (Medium)](https://medium.com/data-science/pytorch-native-fp8-fedc06f1c9f7)
- [DDP vs FSDP in PyTorch (Jellyfish Technologies)](https://www.jellyfishtechnologies.com/ddp-vs-fsdp-in-pytorch-unlocking-efficient-multi-gpu-training/)
- [FSDP1 vs FSDP2 (Hugging Face Docs)](https://huggingface.co/docs/accelerate/en/concept_guides/fsdp1_vs_fsdp2)

---

## 8. Appendix: Quick Reference

### 8.1 Command Cheat Sheet

```bash
# Single GPU, BF16 baseline
python train.py --size medium --batch_size 8 --block_size 2048

# Single GPU, FP8 + compile (optimized)
python train.py --size medium --batch_size 12 --use_fp8 --compile

# 8-GPU DDP, FP8 + compile
torchrun --nproc_per_node=8 train.py --size large --batch_size 4 --use_fp8 --compile

# FSDP2 for large models (70B+)
torchrun --nproc_per_node=8 train.py --size xl --batch_size 2 --use_fsdp

# Enable varlen_attn for variable-length sequences
python train.py --size medium --use_varlen_attn --batch_size 8

# Gradient checkpointing for memory efficiency
python train.py --size xl --gradient_checkpointing --batch_size 4

# Full preset training (all features)
./scripts/train.sh --preset engram-moe 8 2048
```

### 8.2 Environment Variables

```bash
# TF32 precision control
export TORCH_CUDNN_FP32_PRECISION=tf32  # PyTorch 2.10+

# NCCL tuning (single-node 8x B200)
export NCCL_IB_DISABLE=1
export NCCL_P2P_LEVEL=NVL
export NCCL_MIN_NCHANNELS=16
export NCCL_MAX_NCHANNELS=32

# NCCL tuning (multi-node with InfiniBand)
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=ib0

# PyTorch debugging
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export TORCH_LOGS=recompiles,graph_breaks
export NCCL_DEBUG=INFO

# Memory optimization
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
```

### 8.3 Model Size Guidelines

| Model Size | Params | BF16 Memory (per GPU) | FP8 Memory (per GPU) | Recommended GPU Count |
|-----------|--------|----------------------|---------------------|----------------------|
| Small | 150M | ~5GB | ~3.5GB | 1 |
| Medium | 400M | ~12GB | ~9GB | 1 |
| Large | 800M | ~24GB | ~18GB | 1-4 |
| XL | 2B | ~60GB | ~45GB | 4-8 |
| 7B | 7B | ~200GB | ~150GB | 8+ (FSDP2) |
| 13B | 13B | ~350GB | ~260GB | 16+ (FSDP2) |
| 70B | 70B | ~1.8TB | ~1.3TB | 64+ (FSDP2) |

### 8.4 Troubleshooting Checklist

**CUDA Out of Memory**:
- [ ] Reduce batch size
- [ ] Enable gradient checkpointing
- [ ] Use FP8 precision
- [ ] Try Lion optimizer (50% less memory)
- [ ] Increase gradient accumulation steps

**NaN Loss**:
- [ ] Verify `--grad_clip 1.0` is set
- [ ] Reduce learning rate
- [ ] Increase warmup iterations
- [ ] Check GradScaler enabled for FP8
- [ ] Validate data quality

**Slow Training**:
- [ ] Enable `--compile`
- [ ] Increase batch size if memory allows
- [ ] Check GPU utilization (`nvidia-smi dmon`)
- [ ] Profile with `torch.profiler`
- [ ] Verify TF32 is enabled

**NCCL Hangs**:
- [ ] Check all ranks launched
- [ ] Verify NCCL environment variables
- [ ] Test with `NCCL_DEBUG=INFO`
- [ ] Ensure all GPUs visible
- [ ] Check network connectivity (ping other nodes)

---

**End of Document**
