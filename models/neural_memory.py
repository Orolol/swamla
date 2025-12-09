"""Neural Long-term Memory module based on Titans (Google Research, 2025).

This module implements a neural memory with test-time learning via gradient descent.
The memory operates on the latent KV space from MLA for efficiency.

Reference: arXiv:2501.00663 (Titans: Learning to Memorize at Test Time)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class MemoryState:
    """Stateful memory representation for sequence processing.

    Each sample in the batch has independent memory state consisting of:
    - weights: The current memory MLP weights (modified at test time)
    - momentum: Surprise accumulators for momentum-based updates

    Attributes:
        weights: List of (weight, bias) tuples for each memory layer
        momentum: List of (weight_momentum, bias_momentum) tuples
    """
    weights: List[Tuple[torch.Tensor, torch.Tensor]]
    momentum: List[Tuple[torch.Tensor, torch.Tensor]]

    def detach(self) -> "MemoryState":
        """Detach state from computation graph for truncated BPTT.

        Returns:
            New MemoryState with all tensors detached and cloned.
        """
        return MemoryState(
            weights=[(w.detach().clone(), b.detach().clone()) for w, b in self.weights],
            momentum=[(mw.detach().clone(), mb.detach().clone()) for mw, mb in self.momentum]
        )

    def to(self, device: torch.device) -> "MemoryState":
        """Move state to specified device.

        Args:
            device: Target device

        Returns:
            New MemoryState on target device.
        """
        return MemoryState(
            weights=[(w.to(device), b.to(device)) for w, b in self.weights],
            momentum=[(mw.to(device), mb.to(device)) for mw, mb in self.momentum]
        )

    @classmethod
    def init_from_module(
        cls,
        module: "NeuralLongTermMemory",
        batch_size: int,
        device: torch.device = None
    ) -> "MemoryState":
        """Initialize fresh memory state from module parameters.

        Each sample in the batch gets a copy of the module's initial weights.
        Momentum is initialized to zero.

        Args:
            module: The NeuralLongTermMemory module
            batch_size: Number of samples in batch
            device: Device for state tensors (defaults to module device)

        Returns:
            Initialized MemoryState.
        """
        if device is None:
            device = next(module.parameters()).device

        weights = []
        momentum = []

        for layer in module.layers:
            # Expand weights to batch dimension: [out, in] -> [B, out, in]
            w = layer.weight.data.unsqueeze(0).expand(batch_size, -1, -1).clone().to(device)
            b = layer.bias.data.unsqueeze(0).expand(batch_size, -1).clone().to(device)
            weights.append((w, b))

            # Initialize momentum to zero
            momentum.append((torch.zeros_like(w), torch.zeros_like(b)))

        return cls(weights=weights, momentum=momentum)


class NeuralLongTermMemory(nn.Module):
    """Neural memory with test-time learning via gradient descent.

    This module implements the memory mechanism from the Titans paper.
    The memory is an MLP that learns to memorize via gradient descent
    on an associative memory loss during inference.

    Key features:
    - Operates on latent KV space (d_input) for efficiency
    - Uses manual gradient computation for meta-learning
    - Data-dependent dynamics (learning rate, momentum, forget rate)
    - Momentum-based weight updates with forget gate

    The update equations are:
        θ_t = softplus(MLP_θ(x_t))           # Learning rate (momentary surprise)
        η_t = sigmoid(MLP_η(x_t))            # Momentum decay (past surprise)
        α_t = sigmoid(MLP_α(x_t))            # Forget gate (weight decay)

        S_t = η_t * S_{t-1} - θ_t * ∇L       # Momentum update
        M_t = (1 - α_t) * M_{t-1} + S_t      # Weight update

    Args:
        d_input: Input dimension (typically kv_lora_rank from MLA)
        memory_dim: Hidden dimension of memory MLP
        memory_depth: Number of layers in memory MLP (minimum 2)
        d_output: Output dimension (typically same as d_input)
        activation: Activation function name (default: "silu")
    """

    def __init__(
        self,
        d_input: int,
        memory_dim: int,
        memory_depth: int = 2,
        d_output: Optional[int] = None,
        activation: str = "silu"
    ):
        super().__init__()

        self.d_input = d_input
        self.memory_dim = memory_dim
        self.memory_depth = max(2, memory_depth)  # Minimum 2 layers
        self.d_output = d_output if d_output is not None else d_input

        # Build memory MLP architecture
        # d_input -> memory_dim -> ... -> memory_dim -> d_output
        self.layers = nn.ModuleList()
        dims = [d_input] + [memory_dim] * (self.memory_depth - 1) + [self.d_output]

        for i in range(len(dims) - 1):
            self.layers.append(nn.Linear(dims[i], dims[i + 1], bias=True))

        # Activation function
        if activation == "silu":
            self.activation = F.silu
            self.activation_deriv = self._silu_derivative
        elif activation == "relu":
            self.activation = F.relu
            self.activation_deriv = self._relu_derivative
        elif activation == "gelu":
            self.activation = F.gelu
            self.activation_deriv = self._gelu_derivative
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        # Data-dependent dynamics networks
        # These compute per-token learning rate, momentum, and forget rate

        # θ (theta): Learning rate - how much to learn from current token
        self.theta_net = nn.Sequential(
            nn.Linear(d_input, memory_dim),
            nn.SiLU(),
            nn.Linear(memory_dim, 1)
        )

        # η (eta): Momentum decay - how much to retain past surprise
        self.eta_net = nn.Sequential(
            nn.Linear(d_input, memory_dim),
            nn.SiLU(),
            nn.Linear(memory_dim, 1)
        )

        # α (alpha): Forget rate - how much to decay old memory weights
        self.alpha_net = nn.Sequential(
            nn.Linear(d_input, memory_dim),
            nn.SiLU(),
            nn.Linear(memory_dim, 1)
        )

        # Initialize dynamics for stable training
        self._init_dynamics()

    def _init_dynamics(self):
        """Initialize dynamics networks for stable training.

        Initial values:
        - θ: softplus(-2) ≈ 0.13 (small learning rate)
        - η: sigmoid(2) ≈ 0.88 (high momentum retention)
        - α: sigmoid(-4) ≈ 0.02 (low forgetting)
        """
        # Initialize biases of final layer for desired initial values
        nn.init.constant_(self.theta_net[-1].bias, -2.0)
        nn.init.constant_(self.eta_net[-1].bias, 2.0)
        nn.init.constant_(self.alpha_net[-1].bias, -4.0)

        # Initialize weights with small values for stability
        for net in [self.theta_net, self.eta_net, self.alpha_net]:
            for module in net:
                if isinstance(module, nn.Linear):
                    nn.init.normal_(module.weight, std=0.02)

    @staticmethod
    def _silu_derivative(x: torch.Tensor) -> torch.Tensor:
        """Derivative of SiLU: silu'(x) = sigmoid(x) * (1 + x * (1 - sigmoid(x)))"""
        sig = torch.sigmoid(x)
        return sig * (1 + x * (1 - sig))

    @staticmethod
    def _relu_derivative(x: torch.Tensor) -> torch.Tensor:
        """Derivative of ReLU: 1 if x > 0 else 0"""
        return (x > 0).float()

    @staticmethod
    def _gelu_derivative(x: torch.Tensor) -> torch.Tensor:
        """Approximate derivative of GELU."""
        # Using tanh approximation derivative
        c = 0.044715
        inner = (2 / 3.14159) ** 0.5 * (x + c * x ** 3)
        tanh_inner = torch.tanh(inner)
        sech2 = 1 - tanh_inner ** 2
        inner_deriv = (2 / 3.14159) ** 0.5 * (1 + 3 * c * x ** 2)
        return 0.5 * (1 + tanh_inner) + 0.5 * x * sech2 * inner_deriv

    def _forward_memory(
        self,
        x: torch.Tensor,
        weights: List[Tuple[torch.Tensor, torch.Tensor]]
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Forward pass through memory MLP with given weights.

        Args:
            x: Input tensor [B, d_input]
            weights: List of (weight, bias) tuples for each layer

        Returns:
            output: Memory output [B, d_output]
            activations: List of pre-activation tensors for gradient computation
        """
        activations = []
        h = x

        for i, (w, b) in enumerate(weights):
            # Linear: h = x @ W.T + b
            # w has shape [B, out_dim, in_dim], h has shape [B, in_dim]
            # We need batched matrix-vector multiplication
            pre_act = torch.bmm(w, h.unsqueeze(-1)).squeeze(-1) + b
            activations.append(pre_act)

            # Apply activation for all but last layer
            if i < len(weights) - 1:
                h = self.activation(pre_act)
            else:
                h = pre_act  # No activation on output layer

        return h, activations

    def _compute_gradients(
        self,
        x: torch.Tensor,
        target: torch.Tensor,
        weights: List[Tuple[torch.Tensor, torch.Tensor]]
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Compute gradients manually for meta-learning.

        This computes gradients of the associative memory loss L = ||M(x) - target||²
        with respect to the memory weights. We use manual backpropagation instead
        of autograd because we need to update weights during the forward pass.

        Args:
            x: Input tensor [B, d_input]
            target: Target tensor [B, d_output]
            weights: Current memory weights

        Returns:
            gradients: List of (dW, db) tuples for each layer
        """
        # Ensure float32 for numerical stability
        x = x.float()
        target = target.float()
        weights_f32 = [(w.float(), b.float()) for w, b in weights]

        # Forward pass storing intermediates
        h = x
        pre_activations = []
        post_activations = [x]  # Input is "activation" of layer -1

        for i, (w, b) in enumerate(weights_f32):
            pre_act = torch.bmm(w, h.unsqueeze(-1)).squeeze(-1) + b
            pre_activations.append(pre_act)

            if i < len(weights_f32) - 1:
                h = self.activation(pre_act)
                post_activations.append(h)
            else:
                h = pre_act  # Output layer

        output = h

        # Backward pass
        # Loss gradient: ∂L/∂output = 2 * (output - target)
        delta = 2.0 * (output - target)  # [B, d_output]

        gradients = []

        # Backpropagate through layers in reverse
        for i in range(len(weights_f32) - 1, -1, -1):
            w, b = weights_f32[i]
            h_prev = post_activations[i]  # Activation from previous layer

            # Gradient w.r.t. bias: ∂L/∂b = delta
            db = delta  # [B, out_dim]

            # Gradient w.r.t. weight: ∂L/∂W = delta ⊗ h_prev
            # dw[b, i, j] = delta[b, i] * h_prev[b, j]
            dw = torch.bmm(delta.unsqueeze(-1), h_prev.unsqueeze(1))  # [B, out_dim, in_dim]

            # Clamp gradients for stability
            dw = torch.clamp(dw, -1e4, 1e4)
            db = torch.clamp(db, -1e4, 1e4)

            gradients.append((dw, db))

            # Backpropagate delta to previous layer (if not first layer)
            if i > 0:
                # delta_prev = W.T @ delta * activation_derivative
                delta_prev = torch.bmm(w.transpose(-2, -1), delta.unsqueeze(-1)).squeeze(-1)
                delta_prev = delta_prev * self.activation_deriv(pre_activations[i - 1])
                delta = delta_prev

        # Reverse to get gradients in layer order
        gradients = gradients[::-1]

        return gradients

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[MemoryState] = None
    ) -> Tuple[torch.Tensor, MemoryState]:
        """Process input through memory with test-time learning.

        For each token t:
        1. Compute dynamics: θ_t, η_t, α_t from x_t
        2. Retrieve: y_t = M_{t-1}(x_t) (forward without update)
        3. Compute gradient: ∇L = gradient of associative memory loss
        4. Update momentum: S_t = η_t * S_{t-1} - θ_t * ∇L
        5. Update weights: M_t = (1 - α_t) * M_{t-1} + S_t

        Args:
            x: Input tensor [B, S, d_input]
            state: Previous memory state (None for fresh start)

        Returns:
            output: Memory output [B, S, d_output]
            new_state: Updated memory state
        """
        B, S, _ = x.shape
        device = x.device

        # Initialize state if needed
        if state is None:
            state = MemoryState.init_from_module(self, B, device)

        # Ensure state is on correct device
        if state.weights[0][0].device != device:
            state = state.to(device)

        outputs = []
        current_weights = state.weights
        current_momentum = state.momentum

        for t in range(S):
            x_t = x[:, t, :]  # [B, d_input]

            # Compute data-dependent dynamics
            # θ (learning rate): positive via softplus
            theta = F.softplus(self.theta_net(x_t))  # [B, 1]
            theta = torch.clamp(theta, min=1e-6, max=1.0)  # Stability

            # η (momentum): [0, 1] via sigmoid
            eta = torch.sigmoid(self.eta_net(x_t))  # [B, 1]

            # α (forget rate): [0, 1] via sigmoid
            alpha = torch.sigmoid(self.alpha_net(x_t))  # [B, 1]

            # Retrieve from memory (forward without update)
            retrieved, _ = self._forward_memory(x_t, current_weights)
            outputs.append(retrieved)

            # Compute gradients for memory update
            # Target is the input itself (autoencoder-style associative memory)
            target = x_t
            grads = self._compute_gradients(x_t, target, current_weights)

            # Update memory state
            new_weights = []
            new_momentum = []

            for i, ((w, b), (m_w, m_b), (g_w, g_b)) in enumerate(
                zip(current_weights, current_momentum, grads)
            ):
                # Broadcast dynamics to weight shapes
                # theta, eta, alpha are [B, 1], need to broadcast

                # Momentum update: S_t = η * S_{t-1} - θ * ∇L
                # Expand for broadcasting: [B, 1] -> [B, 1, 1] for weights
                theta_w = theta.unsqueeze(-1)  # [B, 1, 1]
                eta_w = eta.unsqueeze(-1)  # [B, 1, 1]
                alpha_w = alpha.unsqueeze(-1)  # [B, 1, 1]

                new_m_w = eta_w * m_w - theta_w * g_w
                new_m_b = eta.squeeze(-1).unsqueeze(-1) * m_b - theta.squeeze(-1).unsqueeze(-1) * g_b

                # Weight update: M_t = (1 - α) * M_{t-1} + S_t
                new_w = (1 - alpha_w) * w + new_m_w
                new_b = (1 - alpha.squeeze(-1).unsqueeze(-1)) * b + new_m_b

                new_weights.append((new_w, new_b))
                new_momentum.append((new_m_w, new_m_b))

            current_weights = new_weights
            current_momentum = new_momentum

        # Stack outputs: [B, S, d_output]
        output = torch.stack(outputs, dim=1)

        # Create new state
        new_state = MemoryState(weights=current_weights, momentum=current_momentum)

        return output, new_state

    def forward_retrieve_only(
        self,
        x: torch.Tensor,
        state: MemoryState
    ) -> torch.Tensor:
        """Retrieve from memory without updating (for inference visualization).

        Args:
            x: Input tensor [B, S, d_input]
            state: Memory state to use

        Returns:
            output: Memory output [B, S, d_output]
        """
        B, S, _ = x.shape
        outputs = []

        for t in range(S):
            x_t = x[:, t, :]
            retrieved, _ = self._forward_memory(x_t, state.weights)
            outputs.append(retrieved)

        return torch.stack(outputs, dim=1)
