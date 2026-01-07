
import torch
import triton
import triton.language as tl


# ============================================================================
# Fused Routing Kernel: bincount + argsort in one pass
# ============================================================================
# Avoids CPU synchronization by keeping everything on GPU

@triton.jit
def _histogram_kernel(
    expert_indices_ptr,  # [N] input expert assignments
    histogram_ptr,       # [E] output histogram (tokens per expert)
    N,                   # number of tokens
    E,                   # number of experts
    BLOCK_SIZE: tl.constexpr,
):
    """Count tokens per expert using atomic adds."""
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N

    expert_ids = tl.load(expert_indices_ptr + offs, mask=mask, other=0)

    # Atomic increment histogram
    for i in range(BLOCK_SIZE):
        if offs[i] < N:
            idx = tl.load(expert_indices_ptr + offs[i])
            tl.atomic_add(histogram_ptr + idx, 1)


@triton.jit
def _compute_permutation_kernel(
    expert_indices_ptr,  # [N] input expert assignments
    offsets_ptr,         # [E] prefix sum of histogram (write positions)
    permutation_ptr,     # [N] output permutation indices
    N,
    BLOCK_SIZE: tl.constexpr,
):
    """Compute permutation indices using atomic fetch-add for write positions."""
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N

    # Process each token
    for i in range(BLOCK_SIZE):
        token_idx = pid * BLOCK_SIZE + i
        if token_idx < N:
            expert_id = tl.load(expert_indices_ptr + token_idx)
            # Atomically get write position and increment
            write_pos = tl.atomic_add(offsets_ptr + expert_id, 1)
            # Store: permutation[write_pos] = token_idx
            tl.store(permutation_ptr + write_pos, token_idx)


def fused_moe_routing(
    expert_indices: torch.Tensor,
    n_experts: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Fused routing computation: histogram + permutation in GPU kernels.

    Replaces: bincount + argsort with fused GPU operations.

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

    # Allocate outputs
    histogram = torch.zeros(n_experts, dtype=torch.int32, device=device)

    # Step 1: Compute histogram (tokens per expert)
    BLOCK_SIZE = 256
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    _histogram_kernel[grid](
        flat_indices, histogram,
        N, n_experts,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    # Step 2: Compute prefix sum (expert offsets) - this is fast on GPU
    expert_offsets = torch.zeros(n_experts + 1, dtype=torch.int64, device=device)
    expert_offsets[1:] = torch.cumsum(histogram.to(torch.int64), dim=0)

    # Step 3: Compute permutation using atomic write positions
    # We need a copy of offsets as write cursors (will be modified)
    write_cursors = expert_offsets[:-1].clone()
    permutation = torch.empty(N, dtype=torch.int64, device=device)

    _compute_permutation_kernel[grid](
        flat_indices, write_cursors, permutation,
        N,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return histogram.to(torch.int64), expert_offsets, permutation


# ============================================================================
# Original MoE GEMM Kernel
# ============================================================================

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
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
    # Grid: (Num_Experts, M_blocks, N_blocks)
    # Note: When using autotune, we cannot pass grid as a callable that depends on META parameters easily if we want to use the same grid logic.
    # But here we are launching the kernel from python with explicit grid.
    # Wait, autotune requires the kernel to be called with .run() or similar if we want it to manage grid?
    # No, autotune works with JIT functions.
    # But we need to make sure the grid calculation in the python wrapper matches the block sizes chosen by autotune.
    # The python wrapper `moe_gemm` calls `moe_gemm_kernel[grid](...)`.
    # `moe_gemm_kernel` is now the Autotuner object.
    # We need to pass the grid to it.
    # But the grid depends on BLOCK_SIZE_M, which is chosen by autotune!
    # So we cannot pass a fixed grid tuple.
    # We must pass a callable grid that accepts META.
    
    expert_idx = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)
    
    # Get start and end of this expert's rows in A/C
    # expert_offsets_ptr is [E+1]
    # We need to load it.
    # Note: pointers in Triton are 64-bit.
    
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
    
    # Pointers
    # A: start at off_start rows
    # a_ptrs = a_ptr + (off_start + offs_am)[:, None] * stride_am + offs_k[None, :] * stride_ak
    # But we iterate K.
    
    a_base = a_ptr + (off_start * stride_am)
    b_base = b_ptr + (expert_idx * stride_be)
    c_base = c_ptr + (off_start * stride_cm)
    
    # Initialize accumulator
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load A
        # Mask for M dimension (variable size)
        a_mask = offs_am < m_size
        
        # A pointers for this K chunk
        # [BLOCK_M, BLOCK_K]
        a_ptrs = a_base + (offs_am[:, None] * stride_am + (k * BLOCK_SIZE_K + offs_k[None, :]) * stride_ak)
        # Load A with boundary checks
        # K dimension check: k * BLOCK_K + offs_k < K
        k_mask = (k * BLOCK_SIZE_K + offs_k) < K
        
        a = tl.load(a_ptrs, mask=a_mask[:, None] & k_mask[None, :], other=0.0)
        
        # Load B
        # [BLOCK_K, BLOCK_N]
        # B is [E, K, N] usually.
        # b_ptrs = b_base + ((k * BLOCK_SIZE_K + offs_k)[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
        b_ptrs = b_base + ((k * BLOCK_SIZE_K + offs_k)[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
        
        # N dimension check: offs_bn < N
        n_mask = offs_bn < N
        
        b = tl.load(b_ptrs, mask=k_mask[:, None] & n_mask[None, :], other=0.0)
        
        # Accumulate
        accumulator += tl.dot(a, b)
        
    # Activation
    if ACTIVATION == "silu":
        accumulator = accumulator * tl.sigmoid(accumulator)
    elif ACTIVATION == "relu":
        accumulator = tl.maximum(accumulator, 0.0)
        
    # Store C
    # c_ptrs = c_base + (offs_am[:, None] * stride_cm + offs_bn[None, :] * stride_cn)
    c_ptrs = c_base + (offs_am[:, None] * stride_cm + offs_bn[None, :] * stride_cn)
    
    c_mask = (offs_am[:, None] < m_size) & (offs_bn[None, :] < N)
    tl.store(c_ptrs, accumulator.to(c_ptr.dtype.element_ty), mask=c_mask)

def moe_gemm(a, b, expert_offsets, activation="", max_tokens_hint=None):
    """
    a: [Total_Tokens, K]
    b: [Num_Experts, K, N]
    expert_offsets: [Num_Experts + 1]
    max_tokens_hint: Optional hint for max tokens per expert (avoids CPU sync)
    """
    # Checks
    assert a.ndim == 2
    assert b.ndim == 3
    assert a.shape[1] == b.shape[1]

    total_tokens, K = a.shape
    num_experts, _, N = b.shape

    # Output
    c = torch.empty((total_tokens, N), device=a.device, dtype=a.dtype)

    # Grid
    # To avoid CPU sync (.item()) which breaks cudagraphs, we use a safe upper bound.
    # Worst case: all tokens go to one expert = total_tokens
    # With good load balancing: ~total_tokens / num_experts * some_factor
    # We use total_tokens as safe upper bound (kernel early-exits for empty blocks anyway)
    if max_tokens_hint is not None:
        max_m = max_tokens_hint
    else:
        # Use total_tokens as upper bound - kernel will early-exit for out-of-bounds blocks
        # This avoids the CPU sync that breaks cudagraphs
        max_m = total_tokens

    grid = lambda META: (num_experts, triton.cdiv(max_m, META['BLOCK_SIZE_M']), triton.cdiv(N, META['BLOCK_SIZE_N']))

    moe_gemm_kernel[grid](
        a, b, c,
        expert_offsets,
        K, N,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1), b.stride(2),
        c.stride(0), c.stride(1),
        ACTIVATION=activation
    )

    return c
