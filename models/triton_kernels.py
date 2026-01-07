"""
Fused Triton kernels for SWA-MLA model optimization.

Contains:
- Fused SwiGLU activation (silu(gate) * up)
- Fused RMSNorm + Linear projection
- Custom MLA attention kernel (for H100 compatibility)
"""

import torch
import triton
import triton.language as tl


# ============================================================================
# Fused SwiGLU Activation Kernel
# ============================================================================
# Fuses: chunk(x, 2) -> silu(gate) * up
# Saves one memory round-trip by not materializing intermediate tensors

@triton.jit
def _swiglu_fwd_kernel(
    # Pointers
    input_ptr,
    output_ptr,
    # Dimensions
    N,  # Number of elements per row (2 * hidden_dim)
    stride_in,
    stride_out,
    # Meta-parameters
    BLOCK_SIZE: tl.constexpr,
):
    """Forward kernel for fused SwiGLU: silu(gate) * up."""
    row_idx = tl.program_id(0)
    half_N = N // 2

    # Compute offsets for this row
    input_row_start = row_idx * stride_in
    output_row_start = row_idx * stride_out

    # Process in blocks
    for block_start in range(0, half_N, BLOCK_SIZE):
        offs = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offs < half_N

        # Load gate (first half) and up (second half)
        gate = tl.load(input_ptr + input_row_start + offs, mask=mask, other=0.0)
        up = tl.load(input_ptr + input_row_start + half_N + offs, mask=mask, other=0.0)

        # SiLU activation: x * sigmoid(x)
        gate_sigmoid = tl.sigmoid(gate)
        gate_silu = gate * gate_sigmoid

        # SwiGLU: silu(gate) * up
        result = gate_silu * up

        # Store result
        tl.store(output_ptr + output_row_start + offs, result, mask=mask)


@triton.jit
def _swiglu_bwd_kernel(
    # Pointers
    grad_output_ptr,
    input_ptr,
    grad_input_ptr,
    # Dimensions
    N,  # Number of elements per row (2 * hidden_dim)
    stride_grad_out,
    stride_in,
    stride_grad_in,
    # Meta-parameters
    BLOCK_SIZE: tl.constexpr,
):
    """Backward kernel for fused SwiGLU."""
    row_idx = tl.program_id(0)
    half_N = N // 2

    # Compute offsets
    grad_out_row = row_idx * stride_grad_out
    input_row = row_idx * stride_in
    grad_in_row = row_idx * stride_grad_in

    for block_start in range(0, half_N, BLOCK_SIZE):
        offs = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offs < half_N

        # Load values
        grad_out = tl.load(grad_output_ptr + grad_out_row + offs, mask=mask, other=0.0)
        gate = tl.load(input_ptr + input_row + offs, mask=mask, other=0.0)
        up = tl.load(input_ptr + input_row + half_N + offs, mask=mask, other=0.0)

        # Forward computation for backward
        gate_sigmoid = tl.sigmoid(gate)
        gate_silu = gate * gate_sigmoid

        # Gradient w.r.t. up: grad_out * silu(gate)
        grad_up = grad_out * gate_silu

        # Gradient w.r.t. gate: grad_out * up * d_silu/d_gate
        # d_silu/d_gate = sigmoid(gate) + gate * sigmoid(gate) * (1 - sigmoid(gate))
        #               = sigmoid(gate) * (1 + gate * (1 - sigmoid(gate)))
        d_silu = gate_sigmoid * (1.0 + gate * (1.0 - gate_sigmoid))
        grad_gate = grad_out * up * d_silu

        # Store gradients
        tl.store(grad_input_ptr + grad_in_row + offs, grad_gate, mask=mask)
        tl.store(grad_input_ptr + grad_in_row + half_N + offs, grad_up, mask=mask)


class FusedSwiGLUFunction(torch.autograd.Function):
    """Autograd function for fused SwiGLU."""

    @staticmethod
    def forward(ctx, x):
        """
        Forward pass for fused SwiGLU.

        Args:
            x: Input tensor of shape [..., 2 * hidden_dim]
               First half is gate, second half is up

        Returns:
            Output tensor of shape [..., hidden_dim]
        """
        # Flatten to 2D for kernel
        original_shape = x.shape
        x_2d = x.view(-1, x.shape[-1])
        num_rows, N = x_2d.shape
        half_N = N // 2

        # Allocate output
        output = torch.empty(num_rows, half_N, device=x.device, dtype=x.dtype)

        # Choose block size
        BLOCK_SIZE = min(1024, triton.next_power_of_2(half_N))

        # Launch kernel
        grid = (num_rows,)
        _swiglu_fwd_kernel[grid](
            x_2d, output,
            N,
            x_2d.stride(0), output.stride(0),
            BLOCK_SIZE=BLOCK_SIZE,
        )

        # Save for backward
        ctx.save_for_backward(x_2d)
        ctx.original_shape = original_shape
        ctx.BLOCK_SIZE = BLOCK_SIZE

        # Reshape output
        output_shape = list(original_shape[:-1]) + [half_N]
        return output.view(output_shape)

    @staticmethod
    def backward(ctx, grad_output):
        """Backward pass for fused SwiGLU."""
        x_2d, = ctx.saved_tensors
        num_rows, N = x_2d.shape

        # Flatten grad_output to 2D
        grad_output_2d = grad_output.view(-1, grad_output.shape[-1])

        # Allocate grad_input
        grad_input = torch.empty_like(x_2d)

        # Launch backward kernel
        grid = (num_rows,)
        _swiglu_bwd_kernel[grid](
            grad_output_2d, x_2d, grad_input,
            N,
            grad_output_2d.stride(0), x_2d.stride(0), grad_input.stride(0),
            BLOCK_SIZE=ctx.BLOCK_SIZE,
        )

        return grad_input.view(ctx.original_shape)


def fused_swiglu(x: torch.Tensor) -> torch.Tensor:
    """
    Fused SwiGLU activation.

    Computes: silu(x[..., :half]) * x[..., half:]
    where silu(x) = x * sigmoid(x)

    Args:
        x: Input tensor of shape [..., 2 * hidden_dim]

    Returns:
        Output tensor of shape [..., hidden_dim]
    """
    return FusedSwiGLUFunction.apply(x)


# ============================================================================
# Fused RMSNorm Kernel
# ============================================================================
# Single kernel that computes RMSNorm without intermediate memory allocation

@triton.jit
def _rms_norm_fwd_kernel(
    # Pointers
    x_ptr,
    weight_ptr,
    output_ptr,
    # Dimensions
    N,  # Hidden dimension
    stride_x,
    stride_out,
    # Parameters
    eps,
    # Meta-parameters
    BLOCK_SIZE: tl.constexpr,
):
    """Forward kernel for RMSNorm."""
    row_idx = tl.program_id(0)

    x_row_start = row_idx * stride_x
    out_row_start = row_idx * stride_out

    # Compute mean of squares
    sum_sq = tl.zeros([1], dtype=tl.float32)
    for block_start in range(0, N, BLOCK_SIZE):
        offs = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offs < N
        x = tl.load(x_ptr + x_row_start + offs, mask=mask, other=0.0).to(tl.float32)
        sum_sq += tl.sum(x * x)

    # RMS normalization
    mean_sq = sum_sq / N
    rrms = tl.rsqrt(mean_sq + eps)

    # Apply normalization and weight
    for block_start in range(0, N, BLOCK_SIZE):
        offs = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offs < N
        x = tl.load(x_ptr + x_row_start + offs, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(weight_ptr + offs, mask=mask, other=0.0).to(tl.float32)

        out = x * rrms * w
        tl.store(output_ptr + out_row_start + offs, out, mask=mask)


@triton.jit
def _rms_norm_bwd_kernel(
    # Pointers
    grad_output_ptr,
    x_ptr,
    weight_ptr,
    grad_input_ptr,
    grad_weight_ptr,  # Atomic accumulation
    # Dimensions
    N,
    num_rows,
    stride_grad_out,
    stride_x,
    stride_grad_in,
    # Parameters
    eps,
    # Meta-parameters
    BLOCK_SIZE: tl.constexpr,
):
    """Backward kernel for RMSNorm."""
    row_idx = tl.program_id(0)

    grad_out_row = row_idx * stride_grad_out
    x_row = row_idx * stride_x
    grad_in_row = row_idx * stride_grad_in

    # Recompute RMS for this row
    sum_sq = tl.zeros([1], dtype=tl.float32)
    for block_start in range(0, N, BLOCK_SIZE):
        offs = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offs < N
        x = tl.load(x_ptr + x_row + offs, mask=mask, other=0.0).to(tl.float32)
        sum_sq += tl.sum(x * x)

    mean_sq = sum_sq / N
    rrms = tl.rsqrt(mean_sq + eps)

    # Compute dot product of grad_out and normalized x for gradient
    dot_prod = tl.zeros([1], dtype=tl.float32)
    for block_start in range(0, N, BLOCK_SIZE):
        offs = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offs < N
        grad_out = tl.load(grad_output_ptr + grad_out_row + offs, mask=mask, other=0.0).to(tl.float32)
        x = tl.load(x_ptr + x_row + offs, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(weight_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        dot_prod += tl.sum(grad_out * w * x * rrms)

    # Compute gradients
    for block_start in range(0, N, BLOCK_SIZE):
        offs = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offs < N

        grad_out = tl.load(grad_output_ptr + grad_out_row + offs, mask=mask, other=0.0).to(tl.float32)
        x = tl.load(x_ptr + x_row + offs, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(weight_ptr + offs, mask=mask, other=0.0).to(tl.float32)

        # grad_input = grad_out * w * rrms - x * rrms^3 * dot_prod / N
        x_norm = x * rrms
        grad_in = grad_out * w * rrms - x_norm * dot_prod * rrms * rrms / N

        tl.store(grad_input_ptr + grad_in_row + offs, grad_in, mask=mask)

        # Atomic add to grad_weight
        grad_w = grad_out * x_norm
        tl.atomic_add(grad_weight_ptr + offs, grad_w, mask=mask)


class FusedRMSNormFunction(torch.autograd.Function):
    """Autograd function for fused RMSNorm."""

    @staticmethod
    def forward(ctx, x, weight, eps=1e-6):
        """
        Forward pass for fused RMSNorm.

        Args:
            x: Input tensor of shape [..., hidden_dim]
            weight: Scale parameter of shape [hidden_dim]
            eps: Epsilon for numerical stability

        Returns:
            Normalized tensor of same shape as x
        """
        original_shape = x.shape
        x_2d = x.view(-1, x.shape[-1])
        num_rows, N = x_2d.shape

        output = torch.empty_like(x_2d)

        BLOCK_SIZE = min(1024, triton.next_power_of_2(N))

        grid = (num_rows,)
        _rms_norm_fwd_kernel[grid](
            x_2d, weight, output,
            N, x_2d.stride(0), output.stride(0),
            eps,
            BLOCK_SIZE=BLOCK_SIZE,
        )

        ctx.save_for_backward(x_2d, weight)
        ctx.eps = eps
        ctx.BLOCK_SIZE = BLOCK_SIZE
        ctx.original_shape = original_shape

        return output.view(original_shape)

    @staticmethod
    def backward(ctx, grad_output):
        """Backward pass for fused RMSNorm."""
        x_2d, weight = ctx.saved_tensors
        num_rows, N = x_2d.shape

        grad_output_2d = grad_output.view(-1, N)

        grad_input = torch.empty_like(x_2d)
        grad_weight = torch.zeros_like(weight)

        grid = (num_rows,)
        _rms_norm_bwd_kernel[grid](
            grad_output_2d, x_2d, weight, grad_input, grad_weight,
            N, num_rows,
            grad_output_2d.stride(0), x_2d.stride(0), grad_input.stride(0),
            ctx.eps,
            BLOCK_SIZE=ctx.BLOCK_SIZE,
        )

        return grad_input.view(ctx.original_shape), grad_weight, None


def fused_rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Fused RMSNorm operation.

    Args:
        x: Input tensor of shape [..., hidden_dim]
        weight: Scale parameter of shape [hidden_dim]
        eps: Epsilon for numerical stability

    Returns:
        Normalized tensor of same shape as x
    """
    return FusedRMSNormFunction.apply(x, weight, eps)


# ============================================================================
# MLA Attention Kernel (H100 compatible)
# ============================================================================
# Custom attention kernel that avoids the FA2 + CUDA graph issues on H100
# by using explicit memory management and avoiding problematic transpose patterns

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=3, num_warps=8),
    ],
    key=['SEQ_LEN', 'HEAD_DIM'],
)
@triton.jit
def _mla_attention_fwd_kernel(
    # Pointers
    q_ptr, k_ptr, v_ptr, out_ptr,
    # Dimensions
    BATCH, N_HEADS, SEQ_LEN, HEAD_DIM: tl.constexpr,
    V_HEAD_DIM: tl.constexpr,
    # Strides (Q, K: [B, T, H, D], V: [B, T, H, Dv])
    stride_qb, stride_qt, stride_qh, stride_qd,
    stride_kb, stride_kt, stride_kh, stride_kd,
    stride_vb, stride_vt, stride_vh, stride_vd,
    stride_ob, stride_ot, stride_oh, stride_od,
    # Parameters
    scale,
    # Meta-parameters
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    """
    Forward kernel for MLA-style attention.

    Handles Q, K with head_dim and V with potentially different v_head_dim.
    Uses causal masking when IS_CAUSAL is True.
    """
    # Get program IDs
    batch_head_idx = tl.program_id(0)
    batch_idx = batch_head_idx // N_HEADS
    head_idx = batch_head_idx % N_HEADS
    m_block_idx = tl.program_id(1)

    # Compute query row range
    m_start = m_block_idx * BLOCK_M
    m_offs = m_start + tl.arange(0, BLOCK_M)
    m_mask = m_offs < SEQ_LEN

    # Initialize output accumulator and softmax normalization
    acc = tl.zeros([BLOCK_M, V_HEAD_DIM], dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)  # Running sum of exp(scores)
    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)  # Running max

    # Base pointers for this batch and head
    q_base = q_ptr + batch_idx * stride_qb + head_idx * stride_qh
    k_base = k_ptr + batch_idx * stride_kb + head_idx * stride_kh
    v_base = v_ptr + batch_idx * stride_vb + head_idx * stride_vh

    # Load Q block: [BLOCK_M, HEAD_DIM]
    q_block = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    for d in range(0, HEAD_DIM, BLOCK_K):
        d_offs = d + tl.arange(0, BLOCK_K)
        d_mask = d_offs < HEAD_DIM
        q_ptrs = q_base + m_offs[:, None] * stride_qt + d_offs[None, :] * stride_qd
        q_loaded = tl.load(q_ptrs, mask=m_mask[:, None] & d_mask[None, :], other=0.0)
        q_block = tl.where(
            (tl.arange(0, HEAD_DIM)[None, :] >= d) & (tl.arange(0, HEAD_DIM)[None, :] < d + BLOCK_K),
            tl.broadcast_to(q_loaded[:, :, None], [BLOCK_M, BLOCK_K, 1])[:, :, 0],
            q_block
        )
    q_block = q_block * scale

    # Determine K/V range based on causality
    if IS_CAUSAL:
        n_end = min(m_start + BLOCK_M, SEQ_LEN)
    else:
        n_end = SEQ_LEN

    # Iterate over K/V blocks
    for n_start in range(0, n_end, BLOCK_N):
        n_offs = n_start + tl.arange(0, BLOCK_N)
        n_mask = n_offs < SEQ_LEN

        # Load K block: [BLOCK_N, HEAD_DIM]
        k_block = tl.zeros([BLOCK_N, HEAD_DIM], dtype=tl.float32)
        for d in range(0, HEAD_DIM, BLOCK_K):
            d_offs = d + tl.arange(0, BLOCK_K)
            d_mask = d_offs < HEAD_DIM
            k_ptrs = k_base + n_offs[:, None] * stride_kt + d_offs[None, :] * stride_kd
            k_loaded = tl.load(k_ptrs, mask=n_mask[:, None] & d_mask[None, :], other=0.0)
            # Manual assignment for k_block
            for i in range(BLOCK_K):
                if d + i < HEAD_DIM:
                    k_block = tl.where(
                        tl.arange(0, HEAD_DIM)[None, :] == d + i,
                        tl.broadcast_to(k_loaded[:, i:i+1], [BLOCK_N, HEAD_DIM]),
                        k_block
                    )

        # Compute attention scores: [BLOCK_M, BLOCK_N]
        scores = tl.dot(q_block, tl.trans(k_block))

        # Apply causal mask
        if IS_CAUSAL:
            causal_mask = m_offs[:, None] >= n_offs[None, :]
            scores = tl.where(causal_mask, scores, float("-inf"))

        # Online softmax update
        m_ij = tl.max(scores, axis=1)
        m_i_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_i_new)
        p_ij = tl.exp(scores - m_i_new[:, None])

        # Load V block: [BLOCK_N, V_HEAD_DIM]
        v_block = tl.zeros([BLOCK_N, V_HEAD_DIM], dtype=tl.float32)
        for d in range(0, V_HEAD_DIM, BLOCK_K):
            d_offs = d + tl.arange(0, BLOCK_K)
            d_mask = d_offs < V_HEAD_DIM
            v_ptrs = v_base + n_offs[:, None] * stride_vt + d_offs[None, :] * stride_vd
            v_loaded = tl.load(v_ptrs, mask=n_mask[:, None] & d_mask[None, :], other=0.0)
            for i in range(BLOCK_K):
                if d + i < V_HEAD_DIM:
                    v_block = tl.where(
                        tl.arange(0, V_HEAD_DIM)[None, :] == d + i,
                        tl.broadcast_to(v_loaded[:, i:i+1], [BLOCK_N, V_HEAD_DIM]),
                        v_block
                    )

        # Update accumulator
        acc = acc * alpha[:, None] + tl.dot(p_ij.to(v_block.dtype), v_block)
        l_i = l_i * alpha + tl.sum(p_ij, axis=1)
        m_i = m_i_new

    # Final normalization
    acc = acc / l_i[:, None]

    # Store output
    out_base = out_ptr + batch_idx * stride_ob + head_idx * stride_oh
    for d in range(0, V_HEAD_DIM, BLOCK_K):
        d_offs = d + tl.arange(0, BLOCK_K)
        d_mask = d_offs < V_HEAD_DIM
        out_ptrs = out_base + m_offs[:, None] * stride_ot + d_offs[None, :] * stride_od
        out_vals = acc[:, d:d+BLOCK_K] if d + BLOCK_K <= V_HEAD_DIM else tl.zeros([BLOCK_M, BLOCK_K], dtype=acc.dtype)
        tl.store(out_ptrs, out_vals, mask=m_mask[:, None] & d_mask[None, :])


def mla_attention_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float = None,
    causal: bool = True,
) -> torch.Tensor:
    """
    MLA-style attention using custom Triton kernel.

    This avoids the FA2 + CUDA graph issues on H100 by using explicit
    memory management without problematic transpose patterns.

    Args:
        q: Query tensor [B, T, H, D]
        k: Key tensor [B, T, H, D]
        v: Value tensor [B, T, H, Dv] (may have different head dim)
        scale: Attention scale (default: 1/sqrt(D))
        causal: Whether to use causal masking

    Returns:
        Output tensor [B, T, H, Dv]
    """
    B, T, H, D = q.shape
    _, _, _, Dv = v.shape

    if scale is None:
        scale = D ** -0.5

    # Ensure contiguous
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()

    # Allocate output
    output = torch.empty(B, T, H, Dv, device=q.device, dtype=q.dtype)

    # Grid: (batch * heads, seq_len blocks)
    BLOCK_M = 64  # Will be autotuned
    grid = (B * H, triton.cdiv(T, BLOCK_M))

    _mla_attention_fwd_kernel[grid](
        q, k, v, output,
        B, H, T, D, Dv,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        output.stride(0), output.stride(1), output.stride(2), output.stride(3),
        scale,
        IS_CAUSAL=causal,
    )

    return output


# ============================================================================
# Convenience wrappers and module classes
# ============================================================================

class FusedSwiGLU(torch.nn.Module):
    """Module wrapper for fused SwiGLU."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return fused_swiglu(x)


class TritonRMSNorm(torch.nn.Module):
    """RMSNorm using fused Triton kernel."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return fused_rms_norm(x, self.weight, self.eps)


# Export public API
__all__ = [
    'fused_swiglu',
    'fused_rms_norm',
    'mla_attention_triton',
    'FusedSwiGLU',
    'TritonRMSNorm',
]
