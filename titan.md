# SwamlaMemory

## Hybrid MLA-Memory Architecture

**Technical Specification v1.0**

*Combining Sliding Window Attention, Multi-head Latent Attention, and Neural Long-term Memory for efficient long-context modeling*

Based on: Titans (Google Research, 2025) + DeepSeek MLA  
Papers: arXiv:2501.00663, arXiv:2504.13173

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Architecture Overview](#2-architecture-overview)
3. [Mathematical Foundations](#3-mathematical-foundations)
4. [Component Specifications](#4-component-specifications)
5. [Implementation Guide](#5-implementation-guide)
6. [Configuration Reference](#6-configuration-reference)
7. [Training Patterns](#7-training-patterns)
8. [Testing & Validation](#8-testing--validation)
9. [Implementation Checklist](#9-implementation-checklist)

---

## 1. Executive Summary

### 1.1 Problem Statement

Standard Transformers suffer from O(N²) complexity in attention, limiting context length. Existing solutions trade off between precision (full attention) and efficiency (linear attention, SSMs). SwamlaMemory combines three complementary mechanisms to achieve both.

### 1.2 Solution Architecture

- **Sliding Window Attention (SWA):** O(N·W) local attention for precise short-range dependencies
- **Multi-head Latent Attention (MLA):** KV compression to d_latent for efficient caching
- **Neural Long-term Memory:** Gradient-based meta-learning for compressed global context
- **Learned Gating:** Dynamic interpolation between attention and memory outputs

### 1.3 Key Innovation

The hybrid approach reuses MLA's latent KV representation as input to the neural memory module. This provides two benefits:

1. **Computational efficiency** by avoiding redundant projections
2. **Conceptual coherence** since the latent space already captures "what to remember" about each token

---

## 2. Architecture Overview

### 2.1 High-Level Data Flow

```
Input x ∈ ℝ^{B×S×D}
       │
       ▼
   ┌───────────────────────────────────────┐
   │           RMSNorm(x)                  │
   └───────────────────────────────────────┘
       │
       ├──────────────────┬────────────────┤
       ▼                  ▼                │
   ┌────────┐      ┌─────────────┐         │
   │ W_q    │      │ W_kv_down   │         │
   │ Query  │      │ KV Compress │         │
   └────────┘      └─────────────┘         │
       │                  │                │
       │            kv_latent ∈ ℝ^{B×S×L}  │
       │                  │                │
       │      ┌───────────┴───────────┐    │
       │      ▼                       ▼    │
       │  ┌────────┐            ┌─────────┐│
       │  │ W_k_up │            │ Neural  ││
       │  │ W_v_up │            │ Memory  ││
       │  └────────┘            └─────────┘│
       │      │                      │     │
       │      ▼                      ▼     │
       │  ┌────────┐            memory_out │
       │  │ SWA    │                 │     │
       │  │ RoPE   │                 │     │
       │  └────────┘                 │     │
       │      │                      │     │
       │  attn_out                   │     │
       │      │                      │     │
       └──────┼──────────────────────┘     │
              ▼                            │
   ┌───────────────────────────────────────┐
   │  Gate = σ(W_g · [attn || memory])     │
   │  Output = gate · memory + (1-g) · attn│
   └───────────────────────────────────────┘
              │
              ▼
         + residual
              │
              ▼
   ┌───────────────────────────────────────┐
   │         FFN / MoE Layer               │
   └───────────────────────────────────────┘
              │
              ▼
         + residual → Output
```

### 2.2 Memory Types Comparison

| Aspect | SWA (Short-term) | Neural Memory (Long-term) | Persistent Memory |
|--------|------------------|---------------------------|-------------------|
| **Scope** | Local window (W tokens) | Entire history (compressed) | Task-level (learned) |
| **Update** | Sliding (FIFO) | Gradient descent + forget | Training only |
| **Precision** | High (exact tokens) | Medium (compressed) | Low (general patterns) |
| **Complexity** | O(N · W) | O(N · D² · L) | O(P · D) |

*Where: W = window size, D = memory dim, L = memory depth, P = persistent tokens*

---

## 3. Mathematical Foundations

### 3.1 Multi-head Latent Attention (MLA)

MLA compresses KV pairs into a low-dimensional latent space for efficient caching:

```python
# KV Compression (d_model → d_latent)
c_t = W_kv_down · x_t                    # c_t ∈ ℝ^{d_latent}

# Query projection (full dimension)  
q_t = W_q · x_t                          # q_t ∈ ℝ^{n_heads × d_head}

# KV expansion for attention
k_t = W_k_up · c_t                       # k_t ∈ ℝ^{n_kv_heads × d_head}
v_t = W_v_up · c_t                       # v_t ∈ ℝ^{n_kv_heads × d_head}

# Cache stores compressed representation
cache = [c_1, c_2, ..., c_t]             # Size: O(S × d_latent) vs O(S × 2 × d_model)
```

### 3.2 Neural Long-term Memory

The memory module is a meta-learning system that learns to memorize at test time via gradient descent on an associative memory loss:

```python
# Associative Memory Objective
L(M; k, v) = ||M(k) - v||²               # L2 loss between prediction and target

# Memory Update with Momentum (Surprise Accumulator)
S_t = η_t · S_{t-1} - θ_t · ∇L(M_{t-1}; k_t, v_t)
M_t = (1 - α_t) · M_{t-1} + S_t

# Data-dependent Dynamics
θ_t = softplus(MLP_θ(x_t))               # Learning rate (momentary surprise)
η_t = sigmoid(MLP_η(x_t))                # Momentum decay (past surprise)  
α_t = sigmoid(MLP_α(x_t))                # Forget gate (weight decay)
```

#### 3.2.1 Gradient Computation

For a 2-layer MLP memory M(k) = W₂ · σ(W₁ · k), the gradients are computed manually (required for meta-learning):

```python
# Forward pass
h = σ(W₁ · k)                            # Hidden activation
ŷ = W₂ · h                               # Prediction

# Backward pass (manual, not autograd)
δ = 2(ŷ - v)                             # Loss gradient
∂L/∂W₂ = δ ⊗ h                          # Output layer gradient
∂L/∂W₁ = (W₂ᵀ · δ) ⊙ σ'(W₁ · k) ⊗ k    # Hidden layer gradient
```

### 3.3 Gating Mechanism

The gate learns when to rely on precise local attention vs compressed global memory:

```python
# Gate computation
g = σ(W_gate · [attn_out || memory_out])

# Fused output  
output = g · memory_out + (1 - g) · attn_out

# Interpretation:
# - g → 1: Trust long-term memory (distant context needed)
# - g → 0: Trust attention (recent context sufficient)
```

### 3.4 Sliding Window Attention

Each token attends only to its local window plus optional persistent memory tokens:

```python
# Attention mask structure
For query at position t:
  Visible keys: [persistent_tokens] ∪ [max(0, t-W+1), ..., t]
  
# Complexity reduction
Full attention: O(N²)
Sliding window: O(N · W)
With persistent: O(N · (W + P))
```

---

## 4. Component Specifications

### 4.1 SwamlaHybridBlock

Main transformer block combining all components:

```python
class SwamlaHybridBlock(nn.Module):
    """
    Hybrid block: SWA + MLA + Neural Memory + Gating
    
    Input:  x ∈ ℝ^{B × S × D}
    Output: y ∈ ℝ^{B × S × D}, kv_cache, memory_state
    """
    def __init__(self, config: SwamlaConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.d_model = config.d_model
        self.d_latent = config.d_latent
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.d_head = config.d_model // config.n_heads
        self.window_size = config.window_size
        
        # Layer norms
        self.norm1 = nn.RMSNorm(config.d_model)
        self.norm2 = nn.RMSNorm(config.d_model)
        
        # === MLA Projections ===
        self.W_q = nn.Linear(config.d_model, config.d_model, bias=False)
        self.W_kv_down = nn.Linear(config.d_model, config.d_latent, bias=False)
        self.W_k_up = nn.Linear(config.d_latent, 
                                 config.n_kv_heads * self.d_head, bias=False)
        self.W_v_up = nn.Linear(config.d_latent,
                                 config.n_kv_heads * self.d_head, bias=False)
        self.W_o = nn.Linear(config.d_model, config.d_model, bias=False)
        
        # === Neural Memory (operates on latent space) ===
        self.memory = NeuralLongTermMemory(
            d_input=config.d_latent,      # Input from KV compression
            memory_dim=config.memory_dim,
            memory_depth=config.memory_depth,
            d_output=config.d_latent,     # Output in latent space
        )
        self.memory_up = nn.Linear(config.d_latent, config.d_model, bias=False)
        
        # === Gating ===
        self.gate_proj = nn.Linear(config.d_model * 2, config.d_model, bias=False)
        
        # === Positional encoding ===
        self.rotary = RotaryEmbedding(self.d_head, config.max_seq_len)
        
        # === FFN/MoE ===
        if config.use_moe and layer_idx >= config.n_dense_layers:
            self.ffn = MoELayer(config)
        else:
            self.ffn = SwiGLU(config.d_model, config.d_ff)
```

### 4.2 NeuralLongTermMemory

Memory module with gradient-based updates:

```python
class NeuralLongTermMemory(nn.Module):
    """
    Neural memory with test-time learning via gradient descent.
    
    Key insight: Memory operates on latent KV space (d_latent),
    not full model dimension, for efficiency.
    """
    def __init__(self, d_input: int, memory_dim: int, memory_depth: int,
                 d_output: int, activation: str = "silu"):
        super().__init__()
        self.d_input = d_input
        self.memory_dim = memory_dim
        self.memory_depth = memory_depth
        self.d_output = d_output
        
        # Memory MLP architecture
        self.layers = nn.ModuleList()
        dims = [d_input] + [memory_dim] * (memory_depth - 1) + [d_output]
        for i in range(len(dims) - 1):
            self.layers.append(nn.Linear(dims[i], dims[i+1], bias=True))
        
        self.activation = getattr(F, activation)
        
        # Data-dependent dynamics networks
        self.theta_net = nn.Sequential(
            nn.Linear(d_input, memory_dim),
            nn.SiLU(),
            nn.Linear(memory_dim, 1)      # Scalar learning rate per token
        )
        self.eta_net = nn.Sequential(
            nn.Linear(d_input, memory_dim),
            nn.SiLU(),  
            nn.Linear(memory_dim, 1)      # Scalar momentum per token
        )
        self.alpha_net = nn.Sequential(
            nn.Linear(d_input, memory_dim),
            nn.SiLU(),
            nn.Linear(memory_dim, 1)      # Scalar forget rate per token
        )
        
        self._init_dynamics()
    
    def _init_dynamics(self):
        """Initialize dynamics for stable training."""
        # θ: small learning rate
        nn.init.constant_(self.theta_net[-1].bias, -2.0)  # softplus(-2) ≈ 0.13
        # η: high momentum  
        nn.init.constant_(self.eta_net[-1].bias, 2.0)     # sigmoid(2) ≈ 0.88
        # α: low forgetting
        nn.init.constant_(self.alpha_net[-1].bias, -4.0)  # sigmoid(-4) ≈ 0.02
```

#### 4.2.1 Memory State Management

```python
@dataclass
class MemoryState:
    """Stateful memory representation for sequence processing."""
    weights: List[torch.Tensor]      # Per-layer weight matrices
    momentum: List[torch.Tensor]     # Surprise accumulators (S_t)
    
    def detach(self) -> "MemoryState":
        """Detach for truncated BPTT."""
        return MemoryState(
            weights=[w.detach().clone() for w in self.weights],
            momentum=[m.detach().clone() for m in self.momentum]
        )
    
    @classmethod
    def init_from_module(cls, module: NeuralLongTermMemory, 
                         batch_size: int) -> "MemoryState":
        """Initialize fresh memory state from module parameters."""
        weights = []
        momentum = []
        for layer in module.layers:
            w = layer.weight.data.unsqueeze(0).expand(batch_size, -1, -1).clone()
            b = layer.bias.data.unsqueeze(0).expand(batch_size, -1).clone()
            weights.append((w, b))
            momentum.append((torch.zeros_like(w), torch.zeros_like(b)))
        return cls(weights=weights, momentum=momentum)
```

#### 4.2.2 Forward Pass Implementation

```python
def forward(self, x: torch.Tensor, state: Optional[MemoryState] = None
           ) -> Tuple[torch.Tensor, MemoryState]:
    """
    Process input through memory with test-time learning.
    
    Args:
        x: Input tensor [B, S, d_input] (from KV latent space)
        state: Previous memory state (None for fresh start)
    
    Returns:
        output: Memory output [B, S, d_output]
        new_state: Updated memory state
    """
    B, S, _ = x.shape
    
    # Initialize state if needed
    if state is None:
        state = MemoryState.init_from_module(self, B)
    
    outputs = []
    current_state = state
    
    for t in range(S):
        x_t = x[:, t, :]  # [B, d_input]
        
        # === Compute dynamics ===
        theta = F.softplus(self.theta_net(x_t))  # [B, 1]
        eta = torch.sigmoid(self.eta_net(x_t))    # [B, 1]
        alpha = torch.sigmoid(self.alpha_net(x_t)) # [B, 1]
        
        # === Retrieve (forward without update) ===
        retrieved = self._forward_memory(x_t, current_state.weights)
        outputs.append(retrieved)
        
        # === Compute gradient for memory update ===
        # Target: input itself (autoencoder-style) or projected value
        target = x_t  # Can be modified for different objectives
        grads = self._compute_gradients(x_t, target, current_state.weights)
        
        # === Update memory state ===
        new_weights = []
        new_momentum = []
        for i, ((w, b), (m_w, m_b), (g_w, g_b)) in enumerate(
            zip(current_state.weights, current_state.momentum, grads)):
            
            # Momentum update: S_t = η·S_{t-1} - θ·∇L
            new_m_w = eta * m_w - theta * g_w
            new_m_b = eta.squeeze(-1) * m_b - theta.squeeze(-1) * g_b
            
            # Weight update: M_t = (1-α)·M_{t-1} + S_t
            new_w = (1 - alpha) * w + new_m_w
            new_b = (1 - alpha.squeeze(-1)) * b + new_m_b
            
            new_weights.append((new_w, new_b))
            new_momentum.append((new_m_w, new_m_b))
        
        current_state = MemoryState(weights=new_weights, momentum=new_momentum)
    
    output = torch.stack(outputs, dim=1)  # [B, S, d_output]
    return output, current_state
```

---

## 5. Implementation Guide

### 5.1 Forward Pass (Complete Block)

```python
def forward(self, x: torch.Tensor, 
            kv_cache: Optional[torch.Tensor] = None,
            memory_state: Optional[MemoryState] = None,
            position_ids: Optional[torch.Tensor] = None
           ) -> Tuple[torch.Tensor, torch.Tensor, MemoryState]:
    """
    Complete forward pass for SwamlaHybridBlock.
    """
    B, S, D = x.shape
    residual = x
    x_norm = self.norm1(x)
    
    # ============================================
    # STEP 1: MLA KV Compression
    # ============================================
    q = self.W_q(x_norm)  # [B, S, d_model]
    q = q.view(B, S, self.n_heads, self.d_head).transpose(1, 2)
    
    kv_latent = self.W_kv_down(x_norm)  # [B, S, d_latent]
    
    # ============================================
    # STEP 2: KV Cache Management  
    # ============================================
    if kv_cache is not None:
        kv_latent_full = torch.cat([kv_cache, kv_latent], dim=1)
    else:
        kv_latent_full = kv_latent
    
    # Keep only window for next iteration
    new_kv_cache = kv_latent_full[:, -self.window_size:]
    
    # ============================================
    # STEP 3: Expand K, V for Attention
    # ============================================
    k = self.W_k_up(kv_latent_full)  # [B, S_full, n_kv_heads * d_head]
    v = self.W_v_up(kv_latent_full)
    
    k = k.view(B, -1, self.n_kv_heads, self.d_head).transpose(1, 2)
    v = v.view(B, -1, self.n_kv_heads, self.d_head).transpose(1, 2)
    
    # Apply RoPE
    cos, sin = self.rotary(position_ids)
    q = apply_rotary_pos_emb(q, cos, sin)
    k = apply_rotary_pos_emb(k, cos, sin)
    
    # GQA expansion if needed
    if self.n_kv_heads < self.n_heads:
        k = k.repeat_interleave(self.n_heads // self.n_kv_heads, dim=1)
        v = v.repeat_interleave(self.n_heads // self.n_kv_heads, dim=1)
    
    # ============================================
    # STEP 4: Sliding Window Attention
    # ============================================
    attn_out = self._sliding_window_attention(q, k, v)  # [B, S, d_model]
    
    # ============================================
    # STEP 5: Neural Memory (on latent space)
    # ============================================
    memory_latent, new_memory_state = self.memory(kv_latent, memory_state)
    memory_out = self.memory_up(memory_latent)  # [B, S, d_model]
    
    # ============================================
    # STEP 6: Gating
    # ============================================
    combined = torch.cat([attn_out, memory_out], dim=-1)
    gate = torch.sigmoid(self.gate_proj(combined))
    fused = gate * memory_out + (1 - gate) * attn_out
    
    # ============================================
    # STEP 7: Residual + FFN
    # ============================================
    x = residual + fused
    x = x + self.ffn(self.norm2(x))
    
    return x, new_kv_cache, new_memory_state
```

### 5.2 Sliding Window Attention Implementation

```python
def _sliding_window_attention(self, q, k, v):
    """
    Compute sliding window attention with causal mask.
    
    Args:
        q: [B, n_heads, S_q, d_head]
        k: [B, n_heads, S_kv, d_head]  
        v: [B, n_heads, S_kv, d_head]
    """
    B, H, S_q, D = q.shape
    S_kv = k.shape[2]
    
    # Compute attention scores
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(D)
    
    # Build sliding window + causal mask
    mask = self._build_sliding_causal_mask(S_q, S_kv)
    scores = scores + mask.to(scores.device)
    
    # Softmax and weighted sum
    attn_weights = F.softmax(scores, dim=-1)
    attn_out = torch.matmul(attn_weights, v)
    
    # Reshape and project
    attn_out = attn_out.transpose(1, 2).contiguous().view(B, S_q, -1)
    return self.W_o(attn_out)


def _build_sliding_causal_mask(self, S_q, S_kv):
    """Build combined causal + sliding window mask."""
    # Query and key positions
    q_pos = torch.arange(S_kv - S_q, S_kv)
    k_pos = torch.arange(S_kv)
    
    # Causal: q_pos >= k_pos
    causal = q_pos.unsqueeze(1) >= k_pos.unsqueeze(0)
    
    # Window: distance <= window_size
    distance = q_pos.unsqueeze(1) - k_pos.unsqueeze(0)
    window = distance <= self.window_size
    
    # Combined mask
    valid = causal & window
    mask = torch.zeros(S_q, S_kv)
    mask.masked_fill_(~valid, float('-inf'))
    
    return mask.unsqueeze(0).unsqueeze(0)  # [1, 1, S_q, S_kv]
```

---

## 6. Configuration Reference

### 6.1 SwamlaConfig Dataclass

```python
@dataclass
class SwamlaConfig:
    """Complete configuration for SwamlaMemory model."""
    
    # === Model Architecture ===
    vocab_size: int = 32000
    d_model: int = 2048
    n_layers: int = 24
    n_heads: int = 16
    n_kv_heads: int = 4           # GQA ratio
    d_ff: int = 5504              # FFN intermediate (2.67x d_model)
    max_seq_len: int = 32768
    
    # === MLA Configuration ===
    d_latent: int = 512           # KV compression dimension
    # Compression ratio: d_model / d_latent = 4x
    
    # === Sliding Window ===
    window_size: int = 4096
    n_persistent: int = 0         # Optional persistent memory tokens
    
    # === Neural Memory ===
    use_memory: bool = True
    memory_dim: int = 256         # Internal MLP dimension
    memory_depth: int = 2         # Number of MLP layers
    memory_layers: Optional[List[int]] = None  # Which layers have memory
                                               # None = all layers
    
    # === MoE Configuration ===
    use_moe: bool = False
    n_dense_layers: int = 1       # Dense layers before MoE starts
    n_experts: int = 64
    n_activated: int = 6
    n_shared_experts: int = 2
    
    # === Training ===
    dropout: float = 0.0
    tie_embeddings: bool = True
    rope_base: float = 10000.0
    
    # === Stability ===
    use_qk_norm: bool = True
    qk_clip: float = 10.0
```

### 6.2 Recommended Configurations

#### Small Model (~170M parameters)

```python
config_small = SwamlaConfig(
    d_model=512,
    n_layers=12,
    n_heads=8,
    n_kv_heads=2,
    d_latent=128,
    window_size=512,
    memory_dim=128,
    memory_depth=2,
    use_moe=False,
)
```

#### Medium Model (~760M parameters)

```python
config_medium = SwamlaConfig(
    d_model=1024,
    n_layers=24,
    n_heads=16,
    n_kv_heads=4,
    d_latent=256,
    window_size=2048,
    memory_dim=256,
    memory_depth=2,
    use_moe=False,
)
```

#### Large MoE Model (~4B total, ~1B active)

```python
config_large_moe = SwamlaConfig(
    d_model=2048,
    n_layers=32,
    n_heads=32,
    n_kv_heads=4,
    d_latent=512,
    window_size=4096,
    memory_dim=512,
    memory_depth=3,
    use_moe=True,
    n_dense_layers=2,
    n_experts=64,
    n_activated=6,
    n_shared_experts=2,
)
```

---

## 7. Training Patterns

### 7.1 Truncated BPTT for Long Sequences

For sequences longer than memory, use truncated backpropagation through time with detached memory states:

```python
def train_long_sequence(model, sequence, chunk_size=2048):
    """
    Train on long sequence with truncated BPTT.
    
    Memory state persists across chunks but gradients are truncated.
    """
    model.train()
    optimizer.zero_grad()
    
    # Split sequence into chunks
    chunks = sequence.split(chunk_size, dim=1)
    
    # Initialize states
    kv_caches = [None] * model.n_layers
    memory_states = [None] * model.n_layers
    
    total_loss = 0
    for chunk_idx, chunk in enumerate(chunks):
        # Forward pass with persistent states
        output, new_kv_caches, new_memory_states = model(
            chunk,
            kv_caches=kv_caches,
            memory_states=memory_states
        )
        
        # Compute loss for this chunk
        loss = compute_loss(output, chunk)
        total_loss += loss.item()
        
        # Backward for this chunk
        loss.backward()
        
        # === CRITICAL: Detach states for truncated BPTT ===
        kv_caches = [cache.detach() if cache is not None else None 
                     for cache in new_kv_caches]
        memory_states = [state.detach() if state is not None else None
                        for state in new_memory_states]
    
    # Gradient clipping and optimizer step
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    
    return total_loss / len(chunks)
```

### 7.2 Memory Warmup Strategy

For optimal performance, warm up memory dynamics gradually:

```python
class MemoryWarmupScheduler:
    """
    Gradually enable memory contribution during training.
    
    Rationale: Let attention branch stabilize first, then 
    introduce memory to avoid early training instabilities.
    """
    def __init__(self, warmup_steps=10000, target_memory_weight=1.0):
        self.warmup_steps = warmup_steps
        self.target = target_memory_weight
        self.current_step = 0
    
    def get_memory_weight(self):
        if self.current_step >= self.warmup_steps:
            return self.target
        return self.target * (self.current_step / self.warmup_steps)
    
    def step(self):
        self.current_step += 1

# Usage in forward pass:
# gate = gate * memory_scheduler.get_memory_weight()
```

### 7.3 Gradient Checkpointing

Enable gradient checkpointing for memory-efficient training:

```python
class SwamlaModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gradient_checkpointing = False
        # ... rest of init
    
    def forward(self, x, ...):
        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                x, kv_cache, mem_state = torch.utils.checkpoint.checkpoint(
                    layer, x, kv_cache, mem_state,
                    use_reentrant=False
                )
            else:
                x, kv_cache, mem_state = layer(x, kv_cache, mem_state)
        return x
```

---

## 8. Testing & Validation

### 8.1 Unit Tests

```python
import pytest
import torch

class TestSwamlaMemory:
    
    @pytest.fixture
    def config(self):
        return SwamlaConfig(
            d_model=256, n_layers=2, n_heads=4, n_kv_heads=2,
            d_latent=64, window_size=128, memory_dim=64, memory_depth=2
        )
    
    def test_memory_state_persistence(self, config):
        """Memory state should change across forward passes."""
        model = SwamlaHybridBlock(config, layer_idx=0)
        x = torch.randn(2, 32, config.d_model)
        
        # First pass
        _, _, state1 = model(x)
        
        # Second pass with same input
        _, _, state2 = model(x, memory_state=state1)
        
        # States should differ
        for w1, w2 in zip(state1.weights, state2.weights):
            assert not torch.allclose(w1[0], w2[0]), "Memory should update"
    
    def test_sliding_window_mask(self, config):
        """Verify attention mask respects window size."""
        model = SwamlaHybridBlock(config, layer_idx=0)
        mask = model._build_sliding_causal_mask(S_q=32, S_kv=64)
        
        # Check that positions outside window are masked
        # Query at position 0 should only see keys in [0, window_size]
        assert mask[0, 0, 0, config.window_size + 1] == float('-inf')
    
    def test_kv_cache_shape(self, config):
        """KV cache should have correct dimensions."""
        model = SwamlaHybridBlock(config, layer_idx=0)
        x = torch.randn(2, 32, config.d_model)
        
        _, kv_cache, _ = model(x)
        
        assert kv_cache.shape == (2, 32, config.d_latent)
    
    def test_gate_range(self, config):
        """Gate values should be in [0, 1]."""
        model = SwamlaHybridBlock(config, layer_idx=0)
        x = torch.randn(2, 32, config.d_model)
        
        # Hook to capture gate values
        gate_values = []
        def hook(module, input, output):
            gate_values.append(torch.sigmoid(output))
        
        model.gate_proj.register_forward_hook(hook)
        model(x)
        
        assert gate_values[0].min() >= 0
        assert gate_values[0].max() <= 1
```

### 8.2 Integration Tests

```python
def test_full_model_forward():
    """Test complete model forward pass."""
    config = SwamlaConfig(
        d_model=512, n_layers=4, n_heads=8, n_kv_heads=2,
        d_latent=128, memory_dim=128, memory_depth=2
    )
    model = SwamlaModel(config)
    
    # Random input
    input_ids = torch.randint(0, config.vocab_size, (2, 256))
    
    # Forward pass
    output = model(input_ids)
    
    assert output.logits.shape == (2, 256, config.vocab_size)
    print("✓ Full model forward pass successful")


def test_generation():
    """Test autoregressive generation."""
    config = SwamlaConfig(d_model=256, n_layers=2, ...)
    model = SwamlaModel(config)
    model.eval()
    
    prompt = torch.randint(0, config.vocab_size, (1, 10))
    
    generated = model.generate(
        prompt,
        max_new_tokens=50,
        temperature=0.8,
        top_k=50
    )
    
    assert generated.shape[1] == 60  # prompt + generated
    print("✓ Generation test successful")
```

### 8.3 Benchmark: Needle-in-Haystack

Validate long-context capability with needle retrieval test:

```python
def needle_in_haystack_test(model, tokenizer, context_lengths=[1024, 4096, 16384]):
    """
    Test retrieval of specific fact buried in long context.
    
    Expected results (from Titans paper):
    - 4K context: >95% accuracy
    - 16K context: >90% accuracy
    """
    results = {}
    
    needle = "The secret password is RAINBOW42."
    question = "What is the secret password?"
    
    for ctx_len in context_lengths:
        # Generate haystack (random text)
        haystack = generate_random_text(ctx_len - len(needle))
        
        # Insert needle at random position
        insert_pos = random.randint(0, len(haystack))
        full_context = haystack[:insert_pos] + needle + haystack[insert_pos:]
        
        # Query model
        prompt = f"{full_context}\n\nQuestion: {question}\nAnswer:"
        response = model.generate(tokenizer.encode(prompt), max_new_tokens=20)
        
        # Check if correct
        correct = "RAINBOW42" in tokenizer.decode(response)
        results[ctx_len] = correct
    
    return results
```

---

## 9. Implementation Checklist

### 9.1 Core Components

- [ ] `SwamlaConfig` dataclass with all parameters
- [ ] `RMSNorm` implementation
- [ ] `RotaryEmbedding` (RoPE) with `apply_rotary_pos_emb`
- [ ] MLA projections: `W_q`, `W_kv_down`, `W_k_up`, `W_v_up`, `W_o`
- [ ] Sliding window attention with causal mask
- [ ] `NeuralLongTermMemory` module
- [ ] `MemoryState` dataclass with `detach()` method
- [ ] Gate projection and fusion logic
- [ ] `SwiGLU` FFN layer
- [ ] `SwamlaHybridBlock` combining all components
- [ ] `SwamlaModel` with embedding and LM head

### 9.2 Memory Module

- [ ] MLP architecture (configurable depth)
- [ ] Dynamics networks: `theta_net`, `eta_net`, `alpha_net`
- [ ] Dynamics initialization (`_init_dynamics`)
- [ ] Manual gradient computation for meta-learning
- [ ] Momentum-based weight update
- [ ] Forget gate application
- [ ] Memory retrieval (forward without update)

### 9.3 Training Infrastructure

- [ ] Truncated BPTT for long sequences
- [ ] Memory state persistence across chunks
- [ ] Gradient checkpointing support
- [ ] Memory warmup scheduler (optional)
- [ ] KV cache management for inference

### 9.4 Testing

- [ ] Unit tests for each component
- [ ] Memory state persistence test
- [ ] Sliding window mask validation
- [ ] Full model forward/backward pass
- [ ] Generation test
- [ ] Needle-in-haystack benchmark

### 9.5 Optional Enhancements

- [ ] MoE layer integration
- [ ] Persistent memory tokens
- [ ] Triton kernels for memory operations
- [ ] FlashAttention integration
- [ ] Multi-GPU / FSDP support

---

*— End of Document —*