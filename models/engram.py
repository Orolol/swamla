"""
Engram: Conditional Memory via Scalable Lookup

Implementation based on DeepSeek-AI's paper:
"Conditional Memory via Scalable Lookup: A New Axis of Sparsity for Large Language Models"

Engram complements MoE by performing O(1) lookups in N-gram embedding tables,
allowing deep layers to focus on complex reasoning while Engram handles static pattern reconstruction.
"""

import unicodedata
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .normalization import RMSNorm
except ImportError:
    from normalization import RMSNorm


# =============================================================================
# Tokenizer Compression
# =============================================================================

def build_tokenizer_compression_mapping(tokenizer) -> torch.Tensor:
    """
    Build compression mapping V -> V' via NFKC + NFD + accent removal + lowercase.

    Expected compression ratio: ~25-30% for vocab ~128k -> ~90-95k canonical IDs.
    Aligns with DeepSeek paper specification for accent-insensitive hashing.

    Args:
        tokenizer: Tokenizer with get_vocab() method

    Returns:
        mapping: Tensor [vocab_size] where mapping[original_id] = canonical_id
    """
    vocab = tokenizer.get_vocab()  # {token_str: token_id}
    vocab_size = len(vocab)

    # Group tokens by normalized form
    normalized_groups: Dict[str, List[int]] = {}

    for token_str, token_id in vocab.items():
        # 1. NFKC normalization (decomposition + recomposition)
        normalized = unicodedata.normalize('NFKC', token_str)

        # 2. NFD decomposition for accent handling (separates base chars from accents)
        normalized = unicodedata.normalize('NFD', normalized)

        # 3. Remove diacritics (accents) - keep only non-combining chars
        # Mn = Mark, Nonspacing (combining diacritical marks)
        normalized = ''.join(
            c for c in normalized
            if unicodedata.category(c) != 'Mn'
        )

        # 4. Lowercase
        normalized = normalized.lower()

        # 5. Strip leading space markers (SentencePiece: '▁', GPT-style: 'Ġ')
        normalized = normalized.lstrip('▁Ġ ')

        # 6. Collapse whitespace
        normalized = ' '.join(normalized.split())

        if normalized not in normalized_groups:
            normalized_groups[normalized] = []
        normalized_groups[normalized].append(token_id)

    # Assign canonical ID per group
    mapping = torch.zeros(vocab_size, dtype=torch.long)
    canonical_id = 0

    for normalized_form, token_ids in normalized_groups.items():
        for tid in token_ids:
            mapping[tid] = canonical_id
        canonical_id += 1

    return mapping


@dataclass
class TokenizerCompression:
    """
    Manages vocabulary compression V -> V'.

    Maps original token IDs to canonical IDs for consistent N-gram hashing.
    """
    original_vocab_size: int
    compressed_vocab_size: int
    mapping: torch.Tensor  # [original_vocab_size] -> canonical_id

    def compress(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Compress token IDs to canonical form.

        Args:
            token_ids: [batch, seq_len] or [seq_len]

        Returns:
            canonical_ids: same shape as input
        """
        # Ensure mapping is on the same device
        if self.mapping.device != token_ids.device:
            self.mapping = self.mapping.to(token_ids.device)
        return self.mapping[token_ids]

    @classmethod
    def from_tokenizer(cls, tokenizer) -> 'TokenizerCompression':
        """Create TokenizerCompression from a tokenizer."""
        mapping = build_tokenizer_compression_mapping(tokenizer)
        return cls(
            original_vocab_size=len(tokenizer.get_vocab()),
            compressed_vocab_size=mapping.max().item() + 1,
            mapping=mapping
        )

    @classmethod
    def identity(cls, vocab_size: int) -> 'TokenizerCompression':
        """Create identity compression (no compression) for testing."""
        return cls(
            original_vocab_size=vocab_size,
            compressed_vocab_size=vocab_size,
            mapping=torch.arange(vocab_size)
        )


# =============================================================================
# Engram Table Configuration
# =============================================================================

# Prime numbers for hash table sizes (good distribution)
DEFAULT_PRIMES = [
    6553, 6563, 6569, 6571, 6577, 6581, 6599, 6607,
    6619, 6637, 6653, 6659, 6661, 6673, 6679, 6689,
    65521, 65537, 65539, 65543, 65551, 65557, 65563, 65579,
    131071, 131101, 131111, 131113, 131129, 131137, 131141, 131143,
    262139, 262147, 262151, 262153, 262187, 262193, 262217, 262231,
]


@dataclass
class EngramTableConfig:
    """
    Configuration for Engram embedding tables.

    Each (n-gram order, head) pair has its own embedding table.
    Total params = sum over all tables of (table_size * slot_dim)
    where slot_dim = embed_dim / (len(n_gram_orders) * num_heads)

    Supports dynamic table sizing based on vocab_size (paper: vocab_size * 5).
    """
    n_gram_orders: List[int] = field(default_factory=lambda: [2, 3])
    num_heads: int = 8
    embed_dim: int = 1280  # d_mem total

    # Dynamic table sizing based on vocabulary (paper specification)
    vocab_size: Optional[int] = None  # If provided, tables are scaled to ~vocab_size * vocab_multiplier
    vocab_multiplier: float = 5.0  # Paper recommends 5x vocabulary size for total table capacity

    # Table sizes (prime numbers for better hash distribution)
    # Format: {(n, k): prime_size} where n=ngram order, k=head index
    table_sizes: Optional[Dict[Tuple[int, int], int]] = None

    @staticmethod
    def _nearest_prime(n: int) -> int:
        """Find the nearest prime number >= n for better hash distribution."""
        def is_prime(num: int) -> bool:
            if num < 2:
                return False
            if num == 2:
                return True
            if num % 2 == 0:
                return False
            for i in range(3, int(num**0.5) + 1, 2):
                if num % i == 0:
                    return False
            return True

        while not is_prime(n):
            n += 1
        return n

    def __post_init__(self):
        if self.table_sizes is None:
            # Auto-compute table sizes
            self.table_sizes = {}
            num_tables = len(self.n_gram_orders) * self.num_heads

            if self.vocab_size is not None:
                # Dynamic scaling based on vocabulary size (paper specification)
                # Target total table capacity = vocab_size * multiplier
                target_total_capacity = int(self.vocab_size * self.vocab_multiplier)
                size_per_table = max(1024, target_total_capacity // num_tables)

                for n in self.n_gram_orders:
                    for k in range(self.num_heads):
                        # Use prime numbers for better hash distribution
                        self.table_sizes[(n, k)] = self._nearest_prime(size_per_table)
            else:
                # Fallback: pick primes based on embed_dim to get reasonable param count
                if self.embed_dim <= 512:
                    base_primes = DEFAULT_PRIMES[:16]  # Smaller tables
                elif self.embed_dim <= 1280:
                    base_primes = DEFAULT_PRIMES[8:24]  # Medium tables
                else:
                    base_primes = DEFAULT_PRIMES[16:32]  # Larger tables

                idx = 0
                for n in self.n_gram_orders:
                    for k in range(self.num_heads):
                        self.table_sizes[(n, k)] = base_primes[idx % len(base_primes)]
                        idx += 1

    @property
    def slot_dim(self) -> int:
        """Dimension of each embedding slot."""
        return self.embed_dim // (len(self.n_gram_orders) * self.num_heads)

    def compute_total_params(self) -> int:
        """Compute total embedding parameters."""
        total = 0
        for (n, k), size in self.table_sizes.items():
            total += size * self.slot_dim
        return total


# =============================================================================
# Engram Embeddings with Multi-Head Hashing
# =============================================================================

class EngramEmbeddings(nn.Module):
    """
    N-gram embedding tables with multi-head hashing.

    For each n-gram order and hash head, maintains a separate embedding table.
    Uses multiplicative-XOR hashing for index computation.
    """

    def __init__(self, config: EngramTableConfig):
        super().__init__()
        self.config = config
        self.slot_dim = config.slot_dim

        # Create embedding tables
        self.tables = nn.ModuleDict()
        for n in config.n_gram_orders:
            for k in range(config.num_heads):
                table_size = config.table_sizes[(n, k)]
                table = nn.Embedding(table_size, self.slot_dim)

                # Standard normal initialization
                nn.init.normal_(table.weight, mean=0.0, std=0.02)

                self.tables[f"n{n}_h{k}"] = table

        # Hash seeds (different per position and head)
        max_n = max(config.n_gram_orders)
        self.register_buffer(
            'hash_seeds',
            torch.randint(1, 2**31, (max_n, config.num_heads), dtype=torch.long)
        )

        # Pre-compute vectorization info: if all tables for a given n-gram order
        # have the same size, we can batch all K heads into a single F.embedding call
        self._uniform_table_size: Dict[int, int] = {}
        for n in config.n_gram_orders:
            sizes = {config.table_sizes[(n, k)] for k in range(config.num_heads)}
            if len(sizes) == 1:
                self._uniform_table_size[n] = sizes.pop()

    def extract_ngrams(
        self,
        canonical_ids: torch.Tensor,  # [batch, seq_len]
        n: int
    ) -> torch.Tensor:
        """
        Extract suffix N-grams for each position.

        For position t, extracts [x_{t-n+1}, ..., x_t].
        Positions < n-1 are padded with 0.

        Returns:
            ngrams: [batch, seq_len, n]
        """
        device = canonical_ids.device

        # Left padding for causal extraction
        padded = F.pad(canonical_ids, (n - 1, 0), value=0)  # [batch, seq_len + n - 1]

        # Unfold to extract windows
        ngrams = padded.unfold(dimension=1, size=n, step=1)  # [batch, seq_len, n]

        return ngrams

    def hash_ngram(
        self,
        ngram_ids: torch.Tensor,  # [batch, seq_len, n]
        n: int,
        head: int,
        table_size: int
    ) -> torch.Tensor:
        """
        Multiplicative-XOR hash for N-grams.

        h = XOR_i(seed_i * id_i) mod table_size

        Returns:
            indices: [batch, seq_len] indices into the embedding table
        """
        # Get seeds for this head
        seeds = self.hash_seeds[:n, head]  # [n], on same device via register_buffer

        # Vectorized: [B, T, n] * [n] -> [B, T, n], then XOR-reduce
        terms = ngram_ids.long() * seeds
        # XOR-fold: reduce over last dim via cumulative XOR
        hash_val = terms[..., 0]
        if n > 1:
            hash_val = hash_val ^ terms[..., 1]
        if n > 2:
            hash_val = hash_val ^ terms[..., 2]

        # Modulo by table size (prime for good distribution)
        indices = hash_val % table_size

        return indices

    def retrieve(
        self,
        canonical_ids: torch.Tensor,  # [batch, seq_len] canonical IDs
    ) -> torch.Tensor:
        """
        Retrieve and concatenate all N-gram embeddings.

        Uses vectorized hash + packed F.embedding when all tables for a given
        n-gram order have the same size (production config). Falls back to
        per-head loop otherwise.

        Returns:
            embeddings: [batch, seq_len, d_mem]
        """
        all_embeddings = []
        K = self.config.num_heads
        B, T = canonical_ids.shape
        device = canonical_ids.device

        for n in self.config.n_gram_orders:
            ngrams = self.extract_ngrams(canonical_ids, n)  # [B, T, n]

            if n in self._uniform_table_size:
                # Vectorized path: all K heads at once
                table_size = self._uniform_table_size[n]
                seeds = self.hash_seeds[:n, :K]  # [n, K]

                # Batched hash: [B, T, n, 1] * [n, K] → [B, T, n, K]
                terms = ngrams.unsqueeze(-1).long() * seeds

                # XOR-fold along n-gram dimension → [B, T, K]
                hash_val = terms[:, :, 0, :]
                for i in range(1, n):
                    hash_val = hash_val ^ terms[:, :, i, :]

                indices = hash_val % table_size  # [B, T, K]

                # Pack all K table weights into one tensor for single F.embedding
                packed = torch.cat(
                    [self.tables[f"n{n}_h{k}"].weight for k in range(K)],
                    dim=0,
                )  # [K * table_size, slot_dim]

                # Offset indices so each head indexes its own slice of packed
                offsets = torch.arange(K, device=device, dtype=torch.long) * table_size
                embs = F.embedding(indices + offsets, packed)  # [B, T, K, slot_dim]
                all_embeddings.append(embs.reshape(B, T, -1))
            else:
                # Fallback: per-head loop (different table sizes)
                for k in range(K):
                    table_key = f"n{n}_h{k}"
                    table = self.tables[table_key]
                    table_size = self.config.table_sizes[(n, k)]
                    indices = self.hash_ngram(ngrams, n, k, table_size)
                    all_embeddings.append(table(indices))

        return torch.cat(all_embeddings, dim=-1)  # [B, T, d_mem]


# =============================================================================
# Context-Aware Gating
# =============================================================================

class EngramGating(nn.Module):
    """
    Context-aware gating mechanism.

    Computes gate = sigmoid(RMSNorm(h)^T @ RMSNorm(k) / sqrt(d))
    where h is the hidden state and k is projected from memory.

    Supports multi-branch mode for mHC (multi-head connections).
    """

    def __init__(
        self,
        hidden_dim: int,      # d from backbone
        memory_dim: int,      # d_mem from embeddings
        num_branches: int = 1,  # M for multi-branch (mHC)
        gate_bias_init: float = 0.0,  # Learnable bias before sigmoid (0.0 = neutral start)
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.memory_dim = memory_dim
        self.num_branches = num_branches

        # Value projection (shared across branches)
        self.w_v = nn.Linear(memory_dim, hidden_dim, bias=False)

        # Key projections (one per branch)
        self.w_k = nn.ModuleList([
            nn.Linear(memory_dim, hidden_dim, bias=False)
            for _ in range(num_branches)
        ])

        # RMSNorm for query and key
        self.query_norm = RMSNorm(hidden_dim)
        self.key_norm = RMSNorm(hidden_dim)

        # Scaling factor
        self.scale = hidden_dim ** -0.5

        # Learnable gate bias: sigmoid(element_product + gate_bias)
        # Default 0.0: sigmoid(0)=0.5, neutral start. Aux loss pushes gates open.
        # For old checkpoint resume, train.py sets this to -3.0 (conservative).
        self.gate_bias = nn.Parameter(torch.tensor(gate_bias_init))

    def forward(
        self,
        hidden_states: torch.Tensor,   # [B, T, d] or list of [B, T, d] for multi-branch
        memory: torch.Tensor           # [B, T, d_mem]
    ) -> torch.Tensor:
        """
        Apply context-aware gating.

        Returns:
            gated_values: [B, T, d] or list if multi-branch
        """
        # Value projection (shared)
        v = self.w_v(memory)  # [B, T, d]

        if self.num_branches == 1:
            # Single branch mode
            h = hidden_states
            k = self.w_k[0](memory)  # [B, T, d]

            # Normalized dot product
            q_norm = self.query_norm(h)
            k_norm = self.key_norm(k)

            # Element-wise gate: sigmoid of per-dimension product + learnable bias
            # Each q_i*k_i ∈ [-3, 3] after RMSNorm → sigmoid stays in [0.05, 0.95]
            # Resistant to collapse: even anti-correlated dims give sigmoid(-3+1)=0.12
            gate = torch.sigmoid(
                q_norm * k_norm + self.gate_bias
            )  # [B, T, d]

            # Gate stored by Engram._store_monitoring via self.gating.last_gate
            self.last_gate = gate

            return gate * v  # [B, T, d]

        else:
            # Multi-branch mode (mHC)
            outputs = []
            for m in range(self.num_branches):
                h_m = hidden_states[m]  # [B, T, d]
                k_m = self.w_k[m](memory)  # [B, T, d]

                q_norm = self.query_norm(h_m)
                k_norm = self.key_norm(k_m)

                gate_m = torch.sigmoid(
                    q_norm * k_norm + self.gate_bias
                )

                outputs.append(gate_m * v)

            return outputs  # List of [B, T, d]


# =============================================================================
# Causal Convolution
# =============================================================================

class EngramConv(nn.Module):
    """
    Depthwise causal convolution with zero-initialization.

    CRITICAL: Zero-init ensures identity mapping at start for stable training.
    Output = SiLU(Conv(RMSNorm(x))) + x (residual)
    """

    def __init__(
        self,
        dim: int,
        kernel_size: int = 4,
        dilation: int = 3,  # = max N-gram order
    ):
        super().__init__()
        self.dim = dim
        self.kernel_size = kernel_size
        self.dilation = dilation

        # Causal padding (left only)
        self.padding = (kernel_size - 1) * dilation

        # Depthwise conv (groups = dim)
        self.conv = nn.Conv1d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=kernel_size,
            dilation=dilation,
            groups=dim,  # Depthwise
            bias=False
        )

        # RMSNorm before conv
        self.norm = RMSNorm(dim)

        # CRITICAL: Zero initialization for identity mapping at start
        nn.init.zeros_(self.conv.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, dim] (gated values)

        Returns:
            y: [batch, seq_len, dim]
        """
        # Normalize
        x_norm = self.norm(x)

        # Reshape for conv1d: [B, T, D] -> [B, D, T]
        x_conv = x_norm.transpose(1, 2)

        # Causal padding (left only)
        x_padded = F.pad(x_conv, (self.padding, 0))

        # Convolution
        conv_out = self.conv(x_padded)  # [B, D, T]

        # Reshape back: [B, D, T] -> [B, T, D]
        conv_out = conv_out.transpose(1, 2)

        # SiLU activation + residual
        y = F.silu(conv_out) + x

        return y


# =============================================================================
# Main Engram Module
# =============================================================================

class Engram(nn.Module):
    """
    Complete Engram module for conditional memory lookup.

    Architecture:
        1. Compress token IDs to canonical form
        2. Extract N-grams and hash to embedding indices
        3. Lookup embeddings from multi-head tables
        4. Apply context-aware gating
        5. Apply causal convolution with residual

    Position in transformer block: BEFORE attention
    Integration: H = H + Engram(H, input_ids)
    """

    def __init__(
        self,
        hidden_dim: int,
        config: EngramTableConfig,
        tokenizer_compression: Optional[TokenizerCompression] = None,
        num_branches: int = 1,
        conv_kernel_size: int = 4,
        gate_bias_init: float = 0.0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.config = config

        # Tokenizer compression (can be set later via set_tokenizer_compression)
        self._tokenizer_compression = tokenizer_compression
        # Register mapping as buffer so it moves with model.to(device)
        # Avoids DeviceCopy in forward() that breaks CUDA graphs
        if tokenizer_compression is not None:
            self.register_buffer('_compression_mapping', tokenizer_compression.mapping, persistent=False)
        else:
            self.register_buffer('_compression_mapping', None, persistent=False)

        # 1. Embedding tables with multi-head hashing
        self.embeddings = EngramEmbeddings(config)

        # 2. Context-aware gating
        self.gating = EngramGating(
            hidden_dim=hidden_dim,
            memory_dim=config.embed_dim,
            num_branches=num_branches,
            gate_bias_init=gate_bias_init,
        )

        # 3. Causal convolution (zero-init for identity at start)
        self.conv = EngramConv(
            dim=hidden_dim,
            kernel_size=conv_kernel_size,
            dilation=max(config.n_gram_orders)
        )

        # For monitoring/debugging
        self.last_gate_values = None
        self.last_memory = None
        self.last_scalar_gate = None  # Scalar gate values (0-1) for accurate activation_rate

    @property
    def tokenizer_compression(self) -> TokenizerCompression:
        """Get tokenizer compression, raising error if not set."""
        if self._tokenizer_compression is None:
            raise RuntimeError(
                "TokenizerCompression not set. Call set_tokenizer_compression() "
                "or pass tokenizer_compression to __init__()."
            )
        return self._tokenizer_compression

    def set_tokenizer_compression(self, compression: TokenizerCompression):
        """Set tokenizer compression (can be done after model creation)."""
        self._tokenizer_compression = compression
        # Move mapping to the module's current device (set_tokenizer_compression
        # is called after model.to(device), so we must explicitly move)
        device = next(self.parameters()).device
        self._compression_mapping = compression.mapping.to(device)

    def forward(
        self,
        hidden_states: torch.Tensor,  # [B, T, d] or list for multi-branch
        input_ids: torch.Tensor       # [B, T] original token IDs
    ) -> torch.Tensor:
        """
        Forward pass of Engram module.

        Returns:
            output: [B, T, d] or list for multi-branch
        """
        # 1. Compress token IDs to canonical form
        # Use the registered buffer (already on correct device) to avoid DeviceCopy
        if self._compression_mapping is None:
            canonical_ids = input_ids
        else:
            canonical_ids = self._compression_mapping[input_ids]

        # 2. Retrieve N-gram embeddings
        memory = self.embeddings.retrieve(canonical_ids)  # [B, T, d_mem]

        # 3. Apply context-aware gating
        gated = self.gating(hidden_states, memory)  # [B, T, d]

        # 4. Apply causal convolution
        output = self.conv(gated)

        # 5. Store monitoring data (outside compiled graph to avoid graph breaks)
        self._store_monitoring(memory, gated)

        return output

    @torch.compiler.disable
    def _store_monitoring(self, memory: torch.Tensor, gated: torch.Tensor):
        """Store monitoring data outside compiled region."""
        self.last_memory = memory.detach()
        self.last_gate_values = gated.detach()
        self.last_scalar_gate = self.gating.last_gate.detach() if hasattr(self.gating, 'last_gate') else None

    def get_gate_loss(self) -> torch.Tensor:
        """Compute auxiliary gate anti-collapse loss.

        Returns -mean(log(gate + eps)) using the live (non-detached) gate tensor.
        For collapsed gates (0.01): -log(0.01) = 4.6, providing gradient signal
        to push gates open. For healthy gates (0.5): -log(0.5) = 0.69.

        Must be called after forward() in the same computation graph.
        """
        if not hasattr(self.gating, 'last_gate') or self.gating.last_gate is None:
            return torch.tensor(0.0, device=next(self.parameters()).device)
        # last_gate is the live tensor (not detached) — gradients flow through
        return -torch.log(self.gating.last_gate + 1e-6).mean()

    def get_num_params(self) -> int:
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters())

    def extra_repr(self) -> str:
        return (
            f"hidden_dim={self.hidden_dim}, "
            f"d_mem={self.config.embed_dim}, "
            f"n_gram_orders={self.config.n_gram_orders}, "
            f"num_heads={self.config.num_heads}, "
            f"params={self.get_num_params():,}"
        )

    def get_metrics(self) -> Dict[str, float]:
        """
        Return Engram metrics for logging and monitoring.

        Should be called after forward() to get stats from the last batch.

        Returns:
            Dict with:
            - engram/gate_mean: Mean norm of gated values
            - engram/gate_std: Std dev of gate norms
            - engram/activation_rate: Proportion of scalar gates > 0.5
            - engram/gate_scalar_mean: Mean of scalar gate values (0-1)
            - engram/memory_norm: Mean norm of retrieved memory embeddings
        """
        metrics = {}

        if self.last_gate_values is not None:
            gate_vals = self.last_gate_values
            # Compute gate norms (relative activation strength)
            gate_norms = gate_vals.norm(dim=-1)  # [B, T]
            metrics['engram/gate_mean'] = gate_norms.mean().item()
            metrics['engram/gate_std'] = gate_norms.std().item()

        # Activation rate: proportion of gates > 0.5 (on scalar gate, not norms)
        if self.last_scalar_gate is not None:
            metrics['engram/activation_rate'] = (self.last_scalar_gate > 0.5).float().mean().item()
            metrics['engram/gate_scalar_mean'] = self.last_scalar_gate.mean().item()
        elif self.last_gate_values is not None:
            # Fallback: use normalized gate norms (legacy, less accurate)
            gate_norms = self.last_gate_values.norm(dim=-1)
            max_norm = (self.hidden_dim ** 0.5)
            normalized_gates = gate_norms / max_norm
            metrics['engram/activation_rate'] = (normalized_gates > 0.5).float().mean().item()

        if self.last_memory is not None:
            memory_norms = self.last_memory.norm(dim=-1)  # [B, T]
            metrics['engram/memory_norm'] = memory_norms.mean().item()

        # Gate logit mean: inverse sigmoid for debugging (how far into collapse)
        if self.last_scalar_gate is not None:
            gate_clamped = self.last_scalar_gate.clamp(1e-6, 1 - 1e-6)
            metrics['engram/gate_logit_mean'] = torch.log(gate_clamped / (1 - gate_clamped)).mean().item()

        # Gate bias value (learnable parameter)
        if hasattr(self.gating, 'gate_bias'):
            metrics['engram/gate_bias'] = self.gating.gate_bias.item()

        return metrics


# =============================================================================
# Factory Functions
# =============================================================================

def create_engram(
    hidden_dim: int,
    d_mem: int = 512,
    n_gram_orders: List[int] = None,
    num_heads: int = 8,
    num_branches: int = 1,
    conv_kernel_size: int = 4,
    table_sizes: Optional[Dict[Tuple[int, int], int]] = None,
    tokenizer_compression: Optional[TokenizerCompression] = None,
    gate_bias_init: float = 0.0,
) -> Engram:
    """
    Factory function to create Engram module with custom configuration.

    Args:
        hidden_dim: Model hidden dimension
        d_mem: Memory embedding dimension
        n_gram_orders: List of N-gram orders (default: [2, 3])
        num_heads: Number of hash heads per N-gram order
        num_branches: Number of branches for mHC (default: 1)
        conv_kernel_size: Kernel size for causal conv (default: 4)
        table_sizes: Custom table sizes {(n, k): size}
        tokenizer_compression: Pre-built compression mapping
        gate_bias_init: Initial value for learnable gate bias (default: 0.0)

    Returns:
        Configured Engram module
    """
    if n_gram_orders is None:
        n_gram_orders = [2, 3]

    config = EngramTableConfig(
        n_gram_orders=n_gram_orders,
        num_heads=num_heads,
        embed_dim=d_mem,
        table_sizes=table_sizes,
    )

    return Engram(
        hidden_dim=hidden_dim,
        config=config,
        tokenizer_compression=tokenizer_compression,
        num_branches=num_branches,
        conv_kernel_size=conv_kernel_size,
        gate_bias_init=gate_bias_init,
    )


def create_engram_for_config(config, layer_id: int) -> Optional[Engram]:
    """
    Create Engram module based on model config and layer ID.

    Returns None if Engram is not enabled for this layer.

    Args:
        config: SWAMLAConfig or similar with engram settings
        layer_id: Current layer index

    Returns:
        Engram module or None
    """
    # Check if Engram is enabled
    if not getattr(config, 'use_engram', False):
        return None

    # Check if this layer should have Engram
    engram_layers = getattr(config, 'engram_layers', [])
    if layer_id not in engram_layers:
        return None

    # Get Engram configuration from model config
    d_mem = getattr(config, 'engram_d_mem', 512)
    n_gram_orders = getattr(config, 'engram_ngram_orders', [2, 3])
    num_heads = getattr(config, 'engram_n_hash_heads', 8)
    conv_kernel_size = getattr(config, 'engram_conv_kernel', 4)
    table_sizes = getattr(config, 'engram_table_sizes', None)
    gate_bias_init = getattr(config, 'engram_gate_bias_init', 0.0)

    # Get vocab_size for dynamic table scaling (paper specification)
    vocab_size = getattr(config, 'vocab_size', None)

    engram_config = EngramTableConfig(
        n_gram_orders=n_gram_orders,
        num_heads=num_heads,
        embed_dim=d_mem,
        vocab_size=vocab_size,
        table_sizes=table_sizes,
    )

    return Engram(
        hidden_dim=config.n_embd,
        config=engram_config,
        tokenizer_compression=None,  # Will be set by model
        num_branches=1,
        conv_kernel_size=conv_kernel_size,
        gate_bias_init=gate_bias_init,
    )
