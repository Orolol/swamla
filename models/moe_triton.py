
import torch
import triton
import triton.language as tl


# ============================================================================
# Fused Routing Kernel: bincount + argsort in one pass
# ============================================================================
# Avoids CPU synchronization by keeping everything on GPU

@triton.jit
def _compute_permutation_kernel(
    expert_indices_ptr,  # [N] input expert assignments
    offsets_ptr,         # [E] write cursors (starting positions, will be modified)
    permutation_ptr,     # [N] output permutation indices
    N,                   # number of tokens
):
    """
    Single-token kernel: each program handles one token.
    Uses pre-computed offsets as starting write positions.
    """
    token_idx = tl.program_id(0)

    if token_idx < N:
        # Load expert assignment for this token
        expert_id = tl.load(expert_indices_ptr + token_idx)

        # Atomically get write position and increment cursor
        write_pos = tl.atomic_add(offsets_ptr + expert_id, 1)

        # Store permutation: sorted position -> original token index
        tl.store(permutation_ptr + write_pos, token_idx)


def fused_moe_routing(
    expert_indices: torch.Tensor,
    n_experts: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Fused routing computation: histogram + permutation with minimal sync.

    Replaces: bincount + argsort with optimized GPU operations.

    Args:
        expert_indices: [N] or [N, K] tensor of expert assignments
        n_experts: number of experts

    Returns:
        tokens_per_expert: [E] count of tokens per expert
        expert_offsets: [E+1] cumsum for expert boundaries
        permutation: [N] indices to reorder tokens by expert
    """
    # Flatten if needed
    flat_indices = expert_indices.view(-1).contiguous()
    N = flat_indices.shape[0]
    device = flat_indices.device

    # Step 1: Compute histogram -- use scatter_add_ instead of bincount to avoid
    # graph breaks (bincount has data-dependent output shape that dynamo can't trace)
    tokens_per_expert = torch.zeros(n_experts, dtype=torch.int64, device=device)
    tokens_per_expert.scatter_add_(0, flat_indices.long(), torch.ones(N, dtype=torch.int64, device=device))

    # Step 2: Compute expert offsets (cumsum is fast)
    expert_offsets = torch.zeros(n_experts + 1, dtype=torch.int64, device=device)
    expert_offsets[1:] = torch.cumsum(tokens_per_expert, dim=0)

    # Step 3: Compute permutation using Triton kernel with pre-computed offsets
    # Clone offsets as write cursors (will be modified by atomic adds)
    write_cursors = expert_offsets[:-1].clone()
    permutation = torch.empty(N, dtype=torch.int64, device=device)

    grid = (N,)
    _compute_permutation_kernel[grid](
        flat_indices, write_cursors, permutation, N
    )

    return tokens_per_expert, expert_offsets, permutation


# ============================================================================
# MoE GEMM Kernel -- Optimized for high expert counts (256+)
# ============================================================================
# Key optimization: Use a 2D grid (flat_block_id, N_blocks) instead of 3D
# (experts, M_blocks, N_blocks). A precomputed layout table maps each
# flat_block_id to (expert_idx, local_m_block) so we launch exactly the
# number of M-blocks needed per expert -- no wasted early-exit blocks.
#
# With 256 experts and ~3072 tokens/expert (BLOCK_M=64 => 48 M-blocks each),
# this launches 256*48 = 12,288 M-blocks vs the old 256*cdiv(786432,128) =
# 1,572,864 M-blocks -- a 128x reduction in grid dimension 1.

@triton.jit
def _moe_gemm_kernel_v2(
    # Pointers
    a_ptr, b_ptr, c_ptr,
    expert_offsets_ptr,
    layout_expert_idx_ptr,  # [max_m_blocks] expert index per block (sentinel=-1 for padding)
    layout_m_block_ptr,     # [max_m_blocks] local M block index per expert
    # Dimensions
    K, N,
    # Strides
    stride_am, stride_ak,
    stride_be, stride_bk, stride_bn,
    stride_cm, stride_cn,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    ACTIVATION: tl.constexpr,
):
    # Grid: (max_m_blocks, N_blocks) — uses upper bound to avoid GPU sync.
    # Padding blocks have expert_idx=-1 (sentinel) and are skipped.
    flat_m_id = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Look up which expert and local M-block this program handles
    expert_idx = tl.load(layout_expert_idx_ptr + flat_m_id)

    # Skip padding blocks (sentinel value from unfilled layout slots)
    if expert_idx < 0:
        return

    pid_m = tl.load(layout_m_block_ptr + flat_m_id)

    # Get this expert's row range in A/C
    off_start = tl.load(expert_offsets_ptr + expert_idx)
    off_end = tl.load(expert_offsets_ptr + expert_idx + 1)
    m_size = off_end - off_start

    # Offsets for this block
    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    # Base pointers
    a_base = a_ptr + (off_start * stride_am)
    b_base = b_ptr + (expert_idx * stride_be)
    c_base = c_ptr + (off_start * stride_cm)

    # Initialize accumulator
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Masks that are loop-invariant
    a_mask = offs_am < m_size
    n_mask = offs_bn < N

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        k_start = k * BLOCK_SIZE_K
        k_mask = (k_start + offs_k) < K

        # Load A tile [BLOCK_M, BLOCK_K]
        a_ptrs = a_base + (offs_am[:, None] * stride_am + (k_start + offs_k[None, :]) * stride_ak)
        a = tl.load(a_ptrs, mask=a_mask[:, None] & k_mask[None, :], other=0.0)

        # Load B tile [BLOCK_K, BLOCK_N]
        b_ptrs = b_base + ((k_start + offs_k)[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
        b = tl.load(b_ptrs, mask=k_mask[:, None] & n_mask[None, :], other=0.0)

        # Accumulate
        accumulator += tl.dot(a, b)

    # Activation
    if ACTIVATION == "silu":
        accumulator = accumulator * tl.sigmoid(accumulator)
    elif ACTIVATION == "relu":
        accumulator = tl.maximum(accumulator, 0.0)

    # Store C
    c_ptrs = c_base + (offs_am[:, None] * stride_cm + offs_bn[None, :] * stride_cn)
    c_mask = (offs_am[:, None] < m_size) & (offs_bn[None, :] < N)
    tl.store(c_ptrs, accumulator.to(c_ptr.dtype.element_ty), mask=c_mask)


def _build_layout_tables(expert_offsets, num_experts, block_size_m, total_tokens, device):
    """
    Build layout tables that map flat M-block IDs to (expert_idx, local_m_block).

    Uses a fixed-size upper-bound allocation with sentinel values (-1) to avoid
    any GPU-CPU sync (.item()). The kernel checks for sentinel and skips padding.

    Returns:
        layout_expert_idx: [max_m_blocks] int32 tensor (-1 for padding blocks)
        layout_m_block: [max_m_blocks] int32 tensor
        max_m_blocks: int (Python int, no GPU sync needed)
    """
    # Compute tokens per expert from offsets: offsets[i+1] - offsets[i]
    tokens_per_expert = expert_offsets[1:] - expert_offsets[:-1]  # [E]

    # Number of M-blocks per expert: ceil(tokens / block_size_m)
    m_blocks_per_expert = (tokens_per_expert + block_size_m - 1) // block_size_m  # [E]

    # Cumulative block offsets per expert
    block_cumsum = torch.zeros(num_experts + 1, device=device, dtype=torch.int32)
    block_cumsum[1:] = torch.cumsum(m_blocks_per_expert.int(), dim=0)

    # Upper bound on total blocks — Python int, zero GPU sync
    max_m_blocks = (total_tokens + block_size_m - 1) // block_size_m

    if max_m_blocks == 0:
        empty = torch.empty(0, device=device, dtype=torch.int32)
        return empty, empty, 0

    # Initialize with sentinel (-1) so padding blocks are skipped by kernel
    layout_expert_idx = torch.full((max_m_blocks,), -1, device=device, dtype=torch.int32)
    layout_m_block = torch.zeros(max_m_blocks, device=device, dtype=torch.int32)

    # Fill valid positions with a simple Triton kernel: one program per expert
    _fill_layout_kernel[(num_experts,)](
        block_cumsum, layout_expert_idx, layout_m_block,
        num_experts,
    )

    return layout_expert_idx, layout_m_block, max_m_blocks


@triton.jit
def _fill_layout_kernel(
    block_cumsum_ptr,   # [E+1] cumulative block counts
    expert_idx_ptr,     # [max_m_blocks] output: expert index per block
    m_block_ptr,        # [max_m_blocks] output: local M-block index
    NUM_EXPERTS: tl.constexpr,
):
    """Fill layout tables: one program per expert, writes its range of blocks."""
    eid = tl.program_id(0)
    start = tl.load(block_cumsum_ptr + eid)
    end = tl.load(block_cumsum_ptr + eid + 1)
    n_blocks = end - start

    # Each expert writes its block range
    for i in range(n_blocks):
        tl.store(expert_idx_ptr + start + i, eid)
        tl.store(m_block_ptr + start + i, i)


# ============================================================================
# Legacy MoE GEMM Kernel (kept for H100 backward compat / small expert counts)
# ============================================================================

@triton.autotune(
    configs=[
        # H100/Hopper configs
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        # B200/Blackwell configs
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=5, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=8),
    ],
    key=['N', 'K'],
)
@triton.jit
def moe_gemm_kernel(
    # Pointers
    a_ptr, b_ptr, c_ptr,
    expert_offsets_ptr,
    # Dimensions
    K, N,
    # Strides
    stride_am, stride_ak,  # A is [Total_M, K]
    stride_be, stride_bk, stride_bn, # B is [E, K, N] or [E, N, K]
    stride_cm, stride_cn,  # C is [Total_M, N]
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    ACTIVATION: tl.constexpr
):
    expert_idx = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)

    off_start = tl.load(expert_offsets_ptr + expert_idx)
    off_end = tl.load(expert_offsets_ptr + expert_idx + 1)

    m_size = off_end - off_start

    # Check if this block is within bounds
    if pid_m * BLOCK_SIZE_M >= m_size:
        return

    # Offsets for this block
    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    a_base = a_ptr + (off_start * stride_am)
    b_base = b_ptr + (expert_idx * stride_be)
    c_base = c_ptr + (off_start * stride_cm)

    # Initialize accumulator
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    a_mask = offs_am < m_size
    n_mask = offs_bn < N

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        k_start = k * BLOCK_SIZE_K
        k_mask = (k_start + offs_k) < K

        a_ptrs = a_base + (offs_am[:, None] * stride_am + (k_start + offs_k[None, :]) * stride_ak)
        a = tl.load(a_ptrs, mask=a_mask[:, None] & k_mask[None, :], other=0.0)

        b_ptrs = b_base + ((k_start + offs_k)[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
        b = tl.load(b_ptrs, mask=k_mask[:, None] & n_mask[None, :], other=0.0)

        accumulator += tl.dot(a, b)

    # Activation
    if ACTIVATION == "silu":
        accumulator = accumulator * tl.sigmoid(accumulator)
    elif ACTIVATION == "relu":
        accumulator = tl.maximum(accumulator, 0.0)

    # Store C
    c_ptrs = c_base + (offs_am[:, None] * stride_cm + offs_bn[None, :] * stride_cn)
    c_mask = (offs_am[:, None] < m_size) & (offs_bn[None, :] < N)
    tl.store(c_ptrs, accumulator.to(c_ptr.dtype.element_ty), mask=c_mask)


# Threshold: use layout-based kernel when num_experts exceeds this value.
# For small expert counts (e.g., 32), the 3D grid overhead is negligible
# and the layout table construction cost is not worth it.
_LAYOUT_KERNEL_EXPERT_THRESHOLD = 64


def moe_gemm(a, b, expert_offsets, activation="", max_tokens_hint=None):
    """
    Grouped GEMM for MoE: A[Total_Tokens, K] @ B[E, K, N] -> C[Total_Tokens, N].

    Each expert processes a contiguous slice of rows in A/C, with boundaries
    defined by expert_offsets[E+1].

    Args:
        a: [Total_Tokens, K] input activations (sorted by expert)
        b: [Num_Experts, K, N] expert weight matrices
        expert_offsets: [Num_Experts + 1] cumulative token counts
        activation: Optional activation ("silu", "relu", or "")
        max_tokens_hint: Optional upper bound on tokens per expert (avoids sync)
    """
    assert a.ndim == 2
    assert b.ndim == 3
    assert a.shape[1] == b.shape[1]

    total_tokens, K = a.shape
    num_experts, _, N = b.shape

    # Output
    c = torch.empty((total_tokens, N), device=a.device, dtype=a.dtype)

    if total_tokens == 0:
        return c

    # Choose kernel strategy based on expert count.
    # For high expert counts (256+), the 3D grid with max_m=total_tokens is
    # catastrophically wasteful: 256 * cdiv(786432, 128) * cdiv(512, 128) =
    # ~6M blocks, of which 99.6% early-exit. The layout-based 2D kernel
    # launches only the blocks that have actual work.
    if num_experts >= _LAYOUT_KERNEL_EXPERT_THRESHOLD:
        _moe_gemm_layout(a, b, c, expert_offsets, num_experts, K, N, activation)
    else:
        _moe_gemm_legacy(a, b, c, expert_offsets, num_experts, total_tokens, K, N,
                         activation, max_tokens_hint)

    return c


def _moe_gemm_layout(a, b, c, expert_offsets, num_experts, K, N, activation):
    """Layout-based kernel: builds a mapping table so each GPU block does useful work."""
    # We need to pick a BLOCK_SIZE_M for layout construction. Since the kernel
    # is not autotuned (we use fixed configs optimized for the target shapes),
    # we select BLOCK_SIZE_M based on the problem geometry.
    #
    # For LatentMoE with K=256, N=256/512 and ~3072 tokens/expert:
    #   BLOCK_M=64: 48 blocks/expert, good occupancy, fits register file
    #   BLOCK_M=128: 24 blocks/expert, may underutilize for small experts
    #
    # Choose based on expected tokens per expert
    avg_tokens = a.shape[0] // max(num_experts, 1)
    if avg_tokens >= 512:
        block_m = 128
    elif avg_tokens >= 64:
        block_m = 64
    else:
        block_m = 32

    # Build layout tables (no GPU sync — uses sentinel for padding blocks)
    layout_expert_idx, layout_m_block, max_m_blocks = _build_layout_tables(
        expert_offsets, num_experts, block_m, a.shape[0], a.device
    )

    if max_m_blocks == 0:
        return

    # Select block sizes for N and K dimensions based on problem shape
    if K >= 256:
        block_k = 128 if K >= 512 else 64
    else:
        block_k = min(32, K)

    if N >= 256:
        block_n = 128
    else:
        block_n = min(64, N)

    # Ensure block sizes are powers of 2 and at least 16
    block_m = max(16, block_m)
    block_n = max(16, block_n)
    block_k = max(16, block_k)

    # Dynamically compute num_stages to fit hardware shared memory limit.
    # Shared memory per stage ≈ (BLOCK_M*BLOCK_K + BLOCK_K*BLOCK_N) * 2 bytes (bf16).
    # B200 (sm_100) limit: 232,448 bytes. Leave margin for alignment overhead.
    smem_per_stage = (block_m * block_k + block_k * block_n) * 2
    max_smem = 228 * 1024  # 228 KB conservative (hw limit ~232 KB)
    num_stages = min(4, max(1, max_smem // smem_per_stage))

    n_blocks = triton.cdiv(N, block_n)
    grid = (max_m_blocks, n_blocks)

    _moe_gemm_kernel_v2[grid](
        a, b, c,
        expert_offsets,
        layout_expert_idx, layout_m_block,
        K, N,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1), b.stride(2),
        c.stride(0), c.stride(1),
        BLOCK_SIZE_M=block_m,
        BLOCK_SIZE_N=block_n,
        BLOCK_SIZE_K=block_k,
        GROUP_SIZE_M=8,
        ACTIVATION=activation,
        num_warps=8 if (block_m >= 64 and block_n >= 64) else 4,
        num_stages=num_stages,
    )


def _moe_gemm_legacy(a, b, c, expert_offsets, num_experts, total_tokens, K, N,
                      activation, max_tokens_hint):
    """Legacy 3D-grid kernel for small expert counts where overhead is acceptable."""
    if max_tokens_hint is not None:
        max_m = max_tokens_hint
    else:
        max_m = total_tokens

    grid = lambda META: (
        num_experts,
        triton.cdiv(max_m, META['BLOCK_SIZE_M']),
        triton.cdiv(N, META['BLOCK_SIZE_N']),
    )

    moe_gemm_kernel[grid](
        a, b, c,
        expert_offsets,
        K, N,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1), b.stride(2),
        c.stride(0), c.stride(1),
        ACTIVATION=activation,
    )
