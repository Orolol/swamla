"""
Gated DeltaNet - Linear attention with delta rule and gating.

Uses the optimized CUDA kernel from flash-linear-attention for O(n) complexity.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Try to import the optimized kernel (chunk version supports backward pass)
try:
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule
    FLA_AVAILABLE = True
except ImportError:
    FLA_AVAILABLE = False
    print("Warning: flash-linear-attention not available, GatedDeltaNet will be slow")

GATED_DELTANET_AVAILABLE = True


class ShortConvolution(nn.Module):
    """Short 1D convolution for local context mixing."""

    def __init__(self, d_model: int, kernel_size: int = 4):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(
            d_model, d_model,
            kernel_size=kernel_size,
            groups=d_model,  # Depthwise
            padding=kernel_size - 1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, D)
        Returns:
            (B, T, D)
        """
        # Conv1d expects (B, D, T)
        x = x.transpose(1, 2)
        x = self.conv(x)[:, :, :x.shape[2]]  # Causal: remove future padding
        return x.transpose(1, 2)


class GatedDeltaNet(nn.Module):
    """
    Gated DeltaNet attention using flash-linear-attention CUDA kernel.

    O(n) complexity linear attention with delta rule updates.
    """

    def __init__(self, config):
        super().__init__()

        self.n_embd = config.n_embd
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head

        # Projections
        self.q_proj = nn.Linear(config.n_embd, self.n_head * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.n_embd, self.n_head * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.n_embd, self.n_head * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_head * self.head_dim, config.n_embd, bias=False)

        # Gate projection (single gate g for decay)
        self.g_proj = nn.Linear(config.n_embd, self.n_head, bias=False)

        # Beta projection (per-position beta for delta rule)
        self.beta_proj = nn.Linear(config.n_embd, self.n_head, bias=False)

        # Short convolution for local context
        self.use_short_conv = getattr(config, 'deltanet_use_conv', True)
        if self.use_short_conv:
            conv_kernel = getattr(config, 'deltanet_conv_kernel', 4)
            self.q_conv = ShortConvolution(self.n_head * self.head_dim, conv_kernel)
            self.k_conv = ShortConvolution(self.n_head * self.head_dim, conv_kernel)
            self.v_conv = ShortConvolution(self.n_head * self.head_dim, conv_kernel)

        # Dropout
        self.dropout = nn.Dropout(config.dropout)

        # Scale factor
        self.scale = self.head_dim ** -0.5

        # Initialize
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.q_proj.weight, gain=0.1)
        nn.init.xavier_uniform_(self.k_proj.weight, gain=0.1)
        nn.init.xavier_uniform_(self.v_proj.weight, gain=0.1)
        nn.init.xavier_uniform_(self.o_proj.weight, gain=0.1)
        nn.init.xavier_uniform_(self.g_proj.weight, gain=0.1)
        nn.init.xavier_uniform_(self.beta_proj.weight, gain=0.1)

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass using flash-linear-attention kernel.

        Args:
            x: Input tensor (B, T, D)
            state: Optional previous state for inference

        Returns:
            output: (B, T, D)
        """
        B, T, D = x.shape

        # Project Q, K, V
        q = self.q_proj(x)  # (B, T, n_head * head_dim)
        k = self.k_proj(x)  # (B, T, n_head * head_dim)
        v = self.v_proj(x)  # (B, T, n_head * head_dim)

        # Apply short convolutions for local context
        if self.use_short_conv:
            q = self.q_conv(q)
            k = self.k_conv(k)
            v = self.v_conv(v)

        # Compute gate (decay) and beta
        g = self.g_proj(x)  # (B, T, n_head)
        g = F.logsigmoid(g)  # Log-space for numerical stability

        beta = self.beta_proj(x)  # (B, T, n_head)
        beta = torch.sigmoid(beta)  # Range [0, 1]

        # Reshape to (B, T, n_head, head_dim)
        q = q.view(B, T, self.n_head, self.head_dim)
        k = k.view(B, T, self.n_head, self.head_dim)
        v = v.view(B, T, self.n_head, self.head_dim)

        # L2 normalize keys for stability (required by delta rule)
        k = F.normalize(k, p=2, dim=-1)

        # FLA kernel requires bf16 - convert all tensors
        if FLA_AVAILABLE:
            q = q.to(torch.bfloat16)
            k = k.to(torch.bfloat16)
            v = v.to(torch.bfloat16)
            g = g.to(torch.bfloat16)
            beta = beta.to(torch.bfloat16)


        else:
            # Ensure all tensors have the same dtype
            dtype = q.dtype
            if g.dtype != dtype:
                g = g.to(dtype)
            if beta.dtype != dtype:
                beta = beta.to(dtype)

        if FLA_AVAILABLE:
            # Use optimized CUDA kernel (chunk version supports backward pass)
            # chunk_gated_delta_rule expects:
            # q: (B, T, H, K)
            # k: (B, T, H, K) - should be L2 normalized
            # v: (B, T, H, V)
            # g: (B, T, H) - decay in log space
            # beta: (B, T, H)
            output, _ = chunk_gated_delta_rule(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                scale=self.scale,
                initial_state=state,
                output_final_state=False,
                use_qk_l2norm_in_kernel=False,  # We already normalized k
            )
        else:
            # Fallback to slow Python implementation
            output = self._slow_recurrence(q, k, v, g, beta, state)

        # Reshape output
        output = output.reshape(B, T, self.n_head * self.head_dim)

        # Convert back to input dtype before output projection (for mixed precision)
        if output.dtype != x.dtype:
            output = output.to(x.dtype)

        output = self.o_proj(output)
        output = self.dropout(output)

        return output

    def _slow_recurrence(
        self,
        q: torch.Tensor,  # (B, T, H, K)
        k: torch.Tensor,  # (B, T, H, K)
        v: torch.Tensor,  # (B, T, H, V)
        g: torch.Tensor,  # (B, T, H) log-space decay
        beta: torch.Tensor,  # (B, T, H)
        state: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Slow Python fallback for when FLA is not available."""
        B, T, H, K = q.shape
        V = v.shape[-1]

        # Initialize state
        if state is None:
            S = torch.zeros(B, H, K, V, device=q.device, dtype=q.dtype)
        else:
            S = state

        outputs = []

        for t in range(T):
            q_t = q[:, t]  # (B, H, K)
            k_t = k[:, t]  # (B, H, K)
            v_t = v[:, t]  # (B, H, V)
            g_t = torch.exp(g[:, t]).unsqueeze(-1).unsqueeze(-1)  # (B, H, 1, 1)
            beta_t = beta[:, t].unsqueeze(-1).unsqueeze(-1)  # (B, H, 1, 1)

            # Delta rule update:
            # S = g * S + beta * (v @ k^T - (S @ k) @ k^T)

            # S @ k: (B, H, K, V) @ (B, H, K) -> (B, H, V)
            Sk = torch.einsum('bhkv,bhk->bhv', S, k_t)

            # (S @ k) @ k^T: (B, H, V) x (B, H, K) -> (B, H, K, V)
            correction = torch.einsum('bhv,bhk->bhkv', Sk, k_t)

            # v @ k^T: (B, H, V) x (B, H, K) -> (B, H, K, V)
            vk = torch.einsum('bhv,bhk->bhkv', v_t, k_t)

            # Update state
            delta = vk - correction
            S = g_t * S + beta_t * delta

            # Output: S @ q
            o_t = torch.einsum('bhkv,bhk->bhv', S, q_t)  # (B, H, V)
            outputs.append(o_t)

        output = torch.stack(outputs, dim=1)  # (B, T, H, V)
        return output


class GatedDeltaNetBlock(nn.Module):
    """Transformer block using Gated DeltaNet instead of standard attention."""

    def __init__(self, config):
        super().__init__()

        # Import here to avoid circular imports
        from normalization import RMSNorm
        from mlp import MLP

        self.norm1 = RMSNorm(config.n_embd)
        self.norm2 = RMSNorm(config.n_embd)
        self.attn = GatedDeltaNet(config)
        self.mlp = MLP(config)
        self.use_checkpoint = getattr(config, 'use_gradient_checkpointing', False)

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int = 0,
        freqs_cis: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        seq_lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass. Extra args (freqs_cis, mask) are ignored (linear attention doesn't need them)."""
        # Self-attention with residual
        x = x + self.attn(self.norm1(x))

        # MLP with residual
        x = x + self.mlp(self.norm2(x))

        return x
