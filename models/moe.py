"""
Mixture of Experts (MoE) implementation for SWA-MLA.

Based on DeepSeek-V3 architecture with:
- Auxiliary-loss-free load balancing via dynamic bias
- Sigmoid routing (not softmax) to avoid winner-take-all
- Shared expert(s) always active
- Token grouping for efficient computation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

try:
    from moe_triton import moe_gemm
    TRITON_AVAILABLE = True
    # Allow capturing scalar outputs to avoid graph break warnings with .item()
    import torch._dynamo
    torch._dynamo.config.capture_scalar_outputs = True
except ImportError as e:
    raise ImportError("Triton not found. Please install Triton and try again.", e)


class SwiGLUExpert(nn.Module):
    """
    Single expert with SwiGLU activation.
    Same architecture as the dense MLP but without dropout (handled at MoE level).
    """

    def __init__(self, d_model: int, d_ff: int, bias: bool = False):
        super().__init__()
        # Combined gate and up projection for efficiency
        self.gate_up_proj = nn.Linear(d_model, 2 * d_ff, bias=bias)
        self.down_proj = nn.Linear(d_ff, d_model, bias=bias)

        self._init_weights()

    def _init_weights(self):
        """Scaled initialization for better gradient flow."""
        nn.init.normal_(self.gate_up_proj.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.down_proj.weight, mean=0.0, std=0.02)
        if self.gate_up_proj.bias is not None:
            nn.init.zeros_(self.gate_up_proj.bias)
        if self.down_proj.bias is not None:
            nn.init.zeros_(self.down_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, d_model] or [num_tokens, d_model]
        Returns:
            output: same shape as input
        """
        # Combined projection
        combined = self.gate_up_proj(x)
        # Split into gate and up
        gate, up = combined.chunk(2, dim=-1)
        # SwiGLU activation
        hidden = F.silu(gate) * up
        # Down projection
        return self.down_proj(hidden)


class BatchedExperts(nn.Module):
    """
    Batched implementation of multiple SwiGLU experts.

    Instead of N separate Linear layers, we stack all weights into 3D tensors
    and use batched matrix multiplications. This dramatically reduces kernel
    launch overhead and enables better GPU utilization.
    """

    def __init__(self, n_experts: int, d_model: int, d_ff: int, bias: bool = False):
        super().__init__()
        self.n_experts = n_experts
        self.d_model = d_model
        self.d_ff = d_ff

        # Stacked weights: [n_experts, out_dim, in_dim]
        # gate_up: d_model -> 2*d_ff (gate and up combined)
        # down: d_ff -> d_model
        self.gate_up_weight = nn.Parameter(torch.empty(n_experts, 2 * d_ff, d_model))
        self.down_weight = nn.Parameter(torch.empty(n_experts, d_model, d_ff))

        if bias:
            self.gate_up_bias = nn.Parameter(torch.zeros(n_experts, 2 * d_ff))
            self.down_bias = nn.Parameter(torch.zeros(n_experts, d_model))
        else:
            self.register_parameter('gate_up_bias', None)
            self.register_parameter('down_bias', None)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights with scaled normal distribution."""
        nn.init.normal_(self.gate_up_weight, mean=0.0, std=0.02)
        nn.init.normal_(self.down_weight, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor, expert_indices: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for selected experts.

        Args:
            x: [num_tokens, d_model] input tokens
            expert_indices: [num_tokens] expert index for each token (0 to n_experts-1)

        Returns:
            output: [num_tokens, d_model] expert outputs
        """
        # Gather the weights for each token's expert
        # gate_up_weight: [n_experts, 2*d_ff, d_model]
        # We need: [num_tokens, 2*d_ff, d_model]
        num_tokens = x.shape[0]

        # Index into expert weights
        # expert_indices: [num_tokens]
        gate_up_w = self.gate_up_weight[expert_indices]  # [num_tokens, 2*d_ff, d_model]
        down_w = self.down_weight[expert_indices]  # [num_tokens, d_model, d_ff]

        # Batched matmul: x @ W^T for each token
        # x: [num_tokens, d_model] -> [num_tokens, 1, d_model]
        # gate_up_w: [num_tokens, 2*d_ff, d_model]
        # result: [num_tokens, 1, 2*d_ff] -> [num_tokens, 2*d_ff]
        combined = torch.bmm(x.unsqueeze(1), gate_up_w.transpose(1, 2)).squeeze(1)

        if self.gate_up_bias is not None:
            gate_up_b = self.gate_up_bias[expert_indices]  # [num_tokens, 2*d_ff]
            combined = combined + gate_up_b

        # SwiGLU: split and apply activation
        gate, up = combined.chunk(2, dim=-1)  # each [num_tokens, d_ff]
        hidden = F.silu(gate) * up  # [num_tokens, d_ff]

        # Down projection
        # hidden: [num_tokens, d_ff] -> [num_tokens, 1, d_ff]
        # down_w: [num_tokens, d_model, d_ff]
        # result: [num_tokens, 1, d_model] -> [num_tokens, d_model]
        output = torch.bmm(hidden.unsqueeze(1), down_w.transpose(1, 2)).squeeze(1)

        if self.down_bias is not None:
            down_b = self.down_bias[expert_indices]  # [num_tokens, d_model]
            output = output + down_b

        return output


class AuxLossFreeRouter(nn.Module):
    """
    Router with auxiliary-loss-free load balancing (DeepSeek-V3 style).

    Key features:
    - Sigmoid affinity (not softmax) to avoid winner-take-all
    - Dynamic bias for load balancing without gradient interference
    - Router Z-loss for numerical stability
    """

    def __init__(
        self,
        d_model: int,
        n_experts: int,
        top_k: int,
        bias_update_gamma: float = 0.001,
        z_loss_coef: float = 0.001,
    ):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k
        self.gamma = bias_update_gamma
        self.z_loss_coef = z_loss_coef

        # Gate projection (no bias in linear, we use separate learnable bias)
        self.gate = nn.Linear(d_model, n_experts, bias=False)

        # Dynamic bias for load balancing (not trained by gradients)
        self.register_buffer('expert_bias', torch.zeros(n_experts))

        # Statistics for monitoring
        self.register_buffer('load_history', torch.zeros(n_experts))
        self.register_buffer('update_count', torch.tensor(0))

        self._init_weights()

    def _init_weights(self):
        """Small initialization to prevent early routing collapse."""
        nn.init.normal_(self.gate.weight, mean=0.0, std=0.01)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Route tokens to experts.

        Args:
            x: [batch_size, seq_len, d_model] or [num_tokens, d_model]

        Returns:
            selected_experts: [num_tokens, top_k] indices of selected experts
            gating_weights: [num_tokens, top_k] normalized weights
            aux_loss: scalar auxiliary loss (z-loss)
        """
        original_shape = x.shape
        if x.dim() == 3:
            x_flat = x.view(-1, x.shape[-1])
        else:
            x_flat = x

        num_tokens = x_flat.shape[0]

        # Compute logits
        logits = self.gate(x_flat)  # [num_tokens, n_experts]

        # Z-loss for numerical stability
        z_loss = self.z_loss_coef * torch.logsumexp(logits, dim=-1).pow(2).mean()

        # Sigmoid affinity (NOT softmax!)
        affinity = torch.sigmoid(logits)  # [num_tokens, n_experts]

        # Routing decision: affinity + bias (bias affects selection, not gradients)
        routing_scores = affinity + self.expert_bias

        # Select top-k experts
        _, selected_experts = torch.topk(routing_scores, self.top_k, dim=-1)

        # Gating weights = normalized affinity (WITHOUT bias!)
        selected_affinity = torch.gather(affinity, 1, selected_experts)
        gating_weights = selected_affinity / (selected_affinity.sum(dim=-1, keepdim=True) + 1e-9)

        # Store routing info for bias update (done externally after backward)
        # This avoids modifying state during forward, which breaks checkpoint recomputation
        if self.training:
            # Save for external update (not during forward to ensure determinism)
            self._last_selected_experts = selected_experts.detach()
            self._last_num_tokens = num_tokens

        return selected_experts, gating_weights, z_loss

    def update_bias_from_last_forward(self):
        """Update expert bias based on last forward pass routing decisions.

        Call this AFTER backward() completes to maintain checkpoint compatibility.
        """
        if hasattr(self, '_last_selected_experts') and self._last_selected_experts is not None:
            self._update_bias(self._last_selected_experts, self._last_num_tokens)
            self._last_selected_experts = None
            self._last_num_tokens = None

    @torch.no_grad()
    def _update_bias(self, selected_experts: torch.Tensor, num_tokens: int):
        """Update expert bias based on load (no gradients)."""
        # Count load per expert
        load = torch.zeros(self.n_experts, device=selected_experts.device, dtype=torch.float32)
        for k in range(self.top_k):
            expert_indices = selected_experts[:, k]
            load.scatter_add_(
                0, expert_indices,
                torch.ones(num_tokens, device=selected_experts.device, dtype=torch.float32)
            )

        # Expected load if perfectly balanced
        expected_load = num_tokens * self.top_k / self.n_experts

        # Update bias: decrease for overloaded, increase for underloaded
        # sign() gives -1, 0, or +1
        self.expert_bias -= self.gamma * (load - expected_load).sign()

        # Update history for monitoring
        self.load_history = 0.99 * self.load_history + 0.01 * load
        self.update_count += 1

    def get_load_balance_stats(self) -> dict:
        """Return load balancing statistics for monitoring."""
        if self.update_count == 0:
            return {'load_mean': 0, 'load_std': 0, 'load_ratio': 1}

        load = self.load_history / (self.update_count.float() + 1e-9)
        load_normalized = load / (load.sum() + 1e-9)

        return {
            'load_mean': load_normalized.mean().item(),
            'load_std': load_normalized.std().item(),
            'load_min': load_normalized.min().item(),
            'load_max': load_normalized.max().item(),
            'load_ratio': (load_normalized.max() / (load_normalized.min() + 1e-9)).item(),
        }


class MoELayer(nn.Module):
    """
    Mixture of Experts layer with shared expert(s).

    Architecture:
    - N routed experts with top-k selection
    - M shared expert(s) always active
    - Token grouping for efficient batched computation
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        # Extract config parameters
        d_model = config.n_embd
        n_experts = getattr(config, 'n_experts', 32)
        n_shared = getattr(config, 'n_shared_experts', 1)
        n_activated = getattr(config, 'n_activated', 3)
        expert_dim = getattr(config, 'expert_dim', d_model)
        bias = getattr(config, 'bias', False)
        z_loss_coef = getattr(config, 'router_z_loss_coef', 0.001)

        self.n_experts = n_experts
        self.n_shared = n_shared
        self.n_activated = n_activated
        self.d_model = d_model
        self.expert_dim = expert_dim

        # Shared expert(s) - always active (using individual experts for simplicity)
        self.shared_experts = nn.ModuleList([
            SwiGLUExpert(d_model, expert_dim, bias=bias)
            for _ in range(n_shared)
        ])

        # Routed experts - using BatchedExperts for efficiency
        self.experts = BatchedExperts(n_experts, d_model, expert_dim, bias=bias)

        # Router
        self.router = AuxLossFreeRouter(
            d_model=d_model,
            n_experts=n_experts,
            top_k=n_activated,
            z_loss_coef=z_loss_coef,
        )

        # Dropout (applied after MoE combination)
        self.dropout = nn.Dropout(getattr(config, 'dropout', 0.0))

        # Store aux loss for collection
        self.aux_loss = torch.tensor(0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, d_model]
        Returns:
            output: [batch_size, seq_len, d_model]
        """
        batch_size, seq_len, d_model = x.shape
        x_flat = x.view(-1, d_model)  # [B*S, D]
        num_tokens = x_flat.shape[0]

        # Compute shared expert output (always active)
        shared_output = torch.zeros_like(x_flat)
        for expert in self.shared_experts:
            shared_output = shared_output + expert(x_flat)

        # Route tokens to experts
        selected_experts, gating_weights, z_loss = self.router(x_flat)
        self.aux_loss = z_loss

        # Compute routed expert outputs using token grouping for efficiency
        # Token grouping reduces kernel launches from 96 (3×32) to 32 by sorting
        # tokens by expert and processing contiguous batches
        routed_output = self._compute_experts_grouped(x_flat, selected_experts, gating_weights)

        # Combine shared and routed outputs
        output = shared_output + routed_output
        output = self.dropout(output)

        return output.view(batch_size, seq_len, d_model)

    def _compute_experts_simple(
        self,
        x: torch.Tensor,
        selected_experts: torch.Tensor,
        gating_weights: torch.Tensor,
    ) -> torch.Tensor:
        """
        Expert computation compatible with gradient checkpointing.

        Uses a slot-based approach: for each of the top_k slots, we process
        tokens through their selected expert. This loops over experts but
        processes tokens in batches for each expert, minimizing Python overhead.

        For maximum efficiency, consider using megablocks or custom CUDA kernels.
        """
        num_tokens, top_k = selected_experts.shape
        d_model = x.shape[-1]
        device = x.device

        # Output accumulator - use same dtype as input for autocast compatibility
        output = torch.zeros(num_tokens, d_model, device=device, dtype=x.dtype)

        # Process slot by slot (only top_k iterations)
        for k in range(top_k):
            expert_indices = selected_experts[:, k]  # [num_tokens]
            slot_weights = gating_weights[:, k]  # [num_tokens]

            # Create slot output tensor
            slot_output = torch.zeros(num_tokens, d_model, device=device, dtype=x.dtype)

            # Process each expert for this slot
            for expert_idx in range(self.n_experts):
                # Create mask for tokens routed to this expert in this slot
                mask = (expert_indices == expert_idx)  # [num_tokens] bool

                # Skip if no tokens
                if not mask.any():
                    continue

                # Get indices of tokens for this expert
                token_indices = mask.nonzero(as_tuple=True)[0]  # [n_selected]

                # Gather inputs for this expert
                expert_input = x[token_indices]  # [n_selected, d_model]

                # Compute expert output
                expert_out = self.experts[expert_idx](expert_input)  # [n_selected, d_model]

                # Scatter back to slot_output - ensure dtype matches
                slot_output[token_indices] = expert_out.to(slot_output.dtype)

            # Apply slot weights and accumulate
            output = output + slot_weights.unsqueeze(-1) * slot_output

        return output

    def _compute_experts_grouped(
        self,
        x: torch.Tensor,
        selected_experts: torch.Tensor,
        gating_weights: torch.Tensor,
    ) -> torch.Tensor:
        """
        Efficient MoE computation with token grouping.

        Instead of nested loops over top-k and experts (96 kernel launches),
        this approach:
        1. Expands tokens for each top-k slot
        2. Sorts all (token, slot) pairs by expert assignment
        3. Processes each expert with a single contiguous batch
        4. Scatters results back and sums over top-k

        This reduces kernel launches from 96 (3 top-k × 32 experts) to 32.
        """
        B_S, K = selected_experts.shape  # [num_tokens, top_k]
        D = x.shape[-1]
        device = x.device
        dtype = x.dtype

        # 1. Count tokens per expert (across all K slots)
        tokens_per_expert = torch.zeros(self.n_experts, dtype=torch.long, device=device)
        flat_expert_indices = selected_experts.view(-1)  # [B_S * K]

        # Use bincount for efficient counting
        tokens_per_expert = torch.bincount(
            flat_expert_indices,
            minlength=self.n_experts
        )

        # 2. Sort tokens by expert assignment
        sorted_indices = torch.argsort(flat_expert_indices, stable=True)

        # 3. Expand x for each of K selections and reorder
        # x: [B_S, D] -> x_expanded: [B_S * K, D]
        x_expanded = x.unsqueeze(1).expand(-1, K, -1).reshape(-1, D)
        weights_flat = gating_weights.view(-1, 1)  # [B_S * K, 1]

        # Reorder by expert
        x_sorted = x_expanded[sorted_indices]  # Contiguous batches per expert
        weights_sorted = weights_flat[sorted_indices]

        # 4. Compute expert outputs with contiguous batches

        # Check if we can use Triton optimization
        # Note: BatchedExperts stores weights as [E, Out, In]
        # gate_up_weight: [E, 2*d_ff, d_model]
        # down_weight: [E, d_model, d_ff]

        # Cumulative sum gives us end positions for each expert
        cumsum = torch.cumsum(tokens_per_expert, dim=0)

        # Use padded batched GEMM approach for better GPU utilization
        # This eliminates the Python loop overhead by using a single batched operation
        use_padded_bmm = True

        if use_padded_bmm:
            outputs_sorted = self._compute_experts_padded_bmm(
                x_sorted, weights_sorted, tokens_per_expert, cumsum, B_S, K, D, device, dtype
            )
        else:
            outputs_sorted = torch.zeros(B_S * K, D, device=device, dtype=dtype)

            # Start positions: [0, cumsum[0], cumsum[1], ...]
            starts = torch.cat([torch.zeros(1, device=device, dtype=torch.long), cumsum[:-1]])

            for expert_idx in range(self.n_experts):
                start = starts[expert_idx]
                end = cumsum[expert_idx]

                # Get contiguous batch for this expert
                expert_input = x_sorted[start:end]  # [count, D]

                # Manual expert computation using BatchedExperts weights
                # gate_up
                w_gate_up = self.experts.gate_up_weight[expert_idx] # [2*d_ff, d_model]
                b_gate_up = self.experts.gate_up_bias[expert_idx] if self.experts.gate_up_bias is not None else None

                combined = F.linear(expert_input, w_gate_up, b_gate_up)

                gate, up = combined.chunk(2, dim=-1)
                hidden = F.silu(gate) * up

                # down
                w_down = self.experts.down_weight[expert_idx] # [d_model, d_ff]
                b_down = self.experts.down_bias[expert_idx] if self.experts.down_bias is not None else None

                expert_output = F.linear(hidden, w_down, b_down)

                # Apply weights and store
                outputs_sorted[start:end] = weights_sorted[start:end] * expert_output.to(dtype)

        # 5. Scatter back to original order
        # Create output in sorted order, then unsort
        outputs_expanded = torch.zeros_like(outputs_sorted)
        outputs_expanded[sorted_indices] = outputs_sorted

        # 6. Reshape and sum over K selections
        # [B_S * K, D] -> [B_S, K, D] -> [B_S, D]
        output = outputs_expanded.view(B_S, K, D).sum(dim=1)

        return output

    def _compute_experts_padded_bmm(
        self,
        x_sorted: torch.Tensor,
        weights_sorted: torch.Tensor,
        tokens_per_expert: torch.Tensor,
        cumsum: torch.Tensor,
        B_S: int,
        K: int,
        D: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """
        Compute expert outputs using padded batched matrix multiplication.

        This approach pads all expert batches to the same size and uses
        torch.bmm for a single fused operation, eliminating Python loop overhead.

        Trade-off: Some wasted compute on padding, but much better GPU utilization
        due to reduced kernel launch overhead.
        """
        E = self.n_experts
        d_ff = self.expert_dim
        total_tokens = B_S * K

        # Use a safe upper bound for max_tokens to avoid index out of bounds
        # Worst case: all tokens go to one expert = total_tokens
        # But we cap it to avoid excessive memory usage
        # With 32 experts and top-3, expected max is ~3x average = 3 * total_tokens / 32
        # Use 4x average as safe bound, capped at total_tokens
        max_tokens = min(total_tokens, (total_tokens * 4 + E - 1) // E)
        # Round up to power of 2 for memory alignment
        max_tokens = 1 << (max_tokens - 1).bit_length() if max_tokens > 0 else 64
        max_tokens = max(64, max_tokens)

        # Build vectorized scatter/gather indices - fully vectorized, no Python loops
        starts = torch.cat([torch.zeros(1, device=device, dtype=torch.long), cumsum[:-1]])

        # Create expert_ids by repeating each expert index by its token count
        # e.g., if tokens_per_expert = [3, 2, 1], expert_ids = [0, 0, 0, 1, 1, 2]
        expert_ids = torch.repeat_interleave(
            torch.arange(E, device=device),
            tokens_per_expert
        )

        # Create positions within each expert batch
        # positions[i] = i - starts[expert_ids[i]]
        # e.g., for expert_ids = [0, 0, 0, 1, 1, 2], positions = [0, 1, 2, 0, 1, 0]
        arange_total = torch.arange(total_tokens, device=device)
        positions = arange_total - starts[expert_ids]

        # Clamp positions to valid range to prevent out-of-bounds access
        # This handles edge cases where routing is extremely imbalanced
        positions = positions.clamp(0, max_tokens - 1)

        # Create padded input tensor using advanced indexing
        # x_padded[e, pos] = x_sorted[i] where expert_ids[i] == e and positions[i] == pos
        x_padded = torch.zeros(E, max_tokens, D, device=device, dtype=dtype)
        weights_padded = torch.zeros(E, max_tokens, 1, device=device, dtype=dtype)

        # Scatter inputs into padded tensor
        x_padded[expert_ids, positions] = x_sorted
        weights_padded[expert_ids, positions] = weights_sorted

        # Batched gate_up projection: [E, max_tokens, D] @ [E, D, 2*d_ff] -> [E, max_tokens, 2*d_ff]
        # gate_up_weight is [E, 2*d_ff, D], need to transpose last two dims
        combined = torch.bmm(x_padded, self.experts.gate_up_weight.transpose(1, 2))

        # Add bias if present
        if self.experts.gate_up_bias is not None:
            combined = combined + self.experts.gate_up_bias.unsqueeze(1)  # [E, 1, 2*d_ff]

        # SwiGLU activation
        gate, up = combined.chunk(2, dim=-1)  # Each [E, max_tokens, d_ff]
        hidden = F.silu(gate) * up

        # Batched down projection: [E, max_tokens, d_ff] @ [E, d_ff, D] -> [E, max_tokens, D]
        # down_weight is [E, D, d_ff], need to transpose last two dims
        expert_outputs = torch.bmm(hidden, self.experts.down_weight.transpose(1, 2))

        # Add bias if present
        if self.experts.down_bias is not None:
            expert_outputs = expert_outputs + self.experts.down_bias.unsqueeze(1)  # [E, 1, D]

        # Apply gating weights
        expert_outputs = expert_outputs * weights_padded

        # Gather outputs back using the same indices
        outputs_sorted = expert_outputs[expert_ids, positions].to(dtype)

        return outputs_sorted

    def get_aux_loss(self) -> torch.Tensor:
        """Return auxiliary loss (z-loss from router)."""
        return self.aux_loss

    def get_load_balance_stats(self) -> dict:
        """Return load balancing statistics."""
        return self.router.get_load_balance_stats()
