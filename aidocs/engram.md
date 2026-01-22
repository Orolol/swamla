# Engram: Spécification Technique d'Implémentation

> **Paper**: "Conditional Memory via Scalable Lookup: A New Axis of Sparsity for Large Language Models"  
> **Auteurs**: DeepSeek-AI & Peking University  
> **Repo**: https://github.com/deepseek-ai/Engram

---

## 1. Vue d'ensemble

### 1.1 Concept

Engram introduit une **mémoire conditionnelle** comme axe de sparsité complémentaire au MoE. Le module effectue des lookups O(1) dans des tables d'embeddings N-gram pour récupérer des représentations statiques, libérant ainsi les couches profondes du réseau pour le raisonnement complexe.

### 1.2 Position dans l'architecture

```
Input IDs
    │
    ▼
┌─────────────────┐
│ Vocab Embedding │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Transformer     │  ← Couche 0-1 (sans Engram)
│ Block           │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Transformer     │  ← Couche 2 (avec Engram)
│ Block + Engram  │
└────────┬────────┘
         │
         ▼
       [...] 
         │
         ▼
┌─────────────────┐
│ Transformer     │  ← Couche 15 (avec Engram)
│ Block + Engram  │
└────────┬────────┘
         │
         ▼
       [...]
         │
         ▼
┌─────────────────┐
│ LM Head         │
└─────────────────┘
```

### 1.3 Ordre des opérations dans un bloc avec Engram

```
H_in
  │
  ├──────────────────┐
  │                  │
  ▼                  ▼
Engram(H_in, IDs) ──►(+)
                     │
                     ▼
               Attention(H)
                     │
                     ▼
                 MoE/FFN(H)
                     │
                     ▼
                  H_out
```

---

## 2. Compression du Tokenizer

### 2.1 Objectif

Réduire la cardinalité effective du vocabulaire en fusionnant les tokens sémantiquement équivalents (casse, espaces, diacritiques).

### 2.2 Algorithme

```python
import unicodedata
from typing import Dict

def build_tokenizer_compression(tokenizer) -> Dict[int, int]:
    """
    Construit le mapping de compression V → V'.
    
    Returns:
        Dict[original_token_id, canonical_token_id]
    """
    vocab = tokenizer.get_vocab()  # {token_str: token_id}
    
    # Grouper par forme normalisée
    normalized_groups: Dict[str, list] = {}
    
    for token_str, token_id in vocab.items():
        # 1. Normalisation NFKC (décomposition compatible puis recomposition)
        normalized = unicodedata.normalize('NFKC', token_str)
        
        # 2. Lowercase
        normalized = normalized.lower()
        
        # 3. Strip leading space marker (optionnel, dépend du tokenizer)
        # Pour SentencePiece: '▁' → ''
        # Pour GPT-style: 'Ġ' → ''
        normalized = normalized.lstrip('▁Ġ ')
        
        # 4. Collapse whitespace
        normalized = ' '.join(normalized.split())
        
        if normalized not in normalized_groups:
            normalized_groups[normalized] = []
        normalized_groups[normalized].append(token_id)
    
    # Assigner un ID canonique par groupe
    compression_map = {}
    canonical_id = 0
    
    for normalized_form, token_ids in normalized_groups.items():
        for tid in token_ids:
            compression_map[tid] = canonical_id
        canonical_id += 1
    
    return compression_map

# Compression ratio attendu: ~23% pour vocab 128k
# Résultat: ~98k IDs canoniques
```

### 2.3 Structure de données

```python
@dataclass
class TokenizerCompression:
    original_vocab_size: int          # e.g., 128000
    compressed_vocab_size: int        # e.g., 98560
    mapping: torch.Tensor             # [original_vocab_size] -> canonical_id
    
    def compress(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            token_ids: [batch, seq_len] ou [seq_len]
        Returns:
            canonical_ids: même shape
        """
        return self.mapping[token_ids]
```

---

## 3. Tables d'Embeddings N-gram

### 3.1 Structure

Pour chaque ordre N et chaque head K, on maintient une table d'embeddings séparée.

```python
@dataclass
class EngramTableConfig:
    n_gram_orders: List[int] = field(default_factory=lambda: [2, 3])
    num_heads: int = 8
    embed_dim: int = 1280  # d_mem total
    
    # Tailles des tables (nombres premiers pour meilleure distribution hash)
    # Ces valeurs doivent être ajustées selon le budget paramètres
    table_sizes: Dict[Tuple[int, int], int] = None  # (n, k) -> prime_size
    
    def __post_init__(self):
        if self.table_sizes is None:
            # Valeurs par défaut pour ~5.7B params
            # dim par slot = embed_dim / (len(n_gram_orders) * num_heads)
            self.table_sizes = {}
            slot_dim = self.embed_dim // (len(self.n_gram_orders) * self.num_heads)
            
            # Pour 5.7B params avec dim=1280:
            # 5.7e9 / 1280 ≈ 4.45M slots total
            # Répartis sur 2 ordres × 8 heads = 16 tables
            # ~278k slots par table
            
            primes = [278447, 278459, 278479, 278489,  # Quelques premiers proches
                      278491, 278497, 278503, 278543,
                      278549, 278557, 278561, 278563,
                      278567, 278581, 278587, 278593]
            
            idx = 0
            for n in self.n_gram_orders:
                for k in range(self.num_heads):
                    self.table_sizes[(n, k)] = primes[idx % len(primes)]
                    idx += 1
```

### 3.2 Initialisation des tables

```python
class EngramEmbeddings(nn.Module):
    def __init__(self, config: EngramTableConfig):
        super().__init__()
        self.config = config
        
        # Dimension par embedding individuel
        self.slot_dim = config.embed_dim // (
            len(config.n_gram_orders) * config.num_heads
        )
        
        # Créer les tables d'embeddings
        self.tables = nn.ModuleDict()
        for n in config.n_gram_orders:
            for k in range(config.num_heads):
                table_size = config.table_sizes[(n, k)]
                table = nn.Embedding(table_size, self.slot_dim)
                
                # Initialisation normale standard
                nn.init.normal_(table.weight, mean=0.0, std=0.02)
                
                self.tables[f"n{n}_h{k}"] = table
        
        # Hash seeds (différents par head)
        self.register_buffer(
            'hash_seeds',
            torch.randint(1, 2**31, (max(config.n_gram_orders) + 1, config.num_heads))
        )
```

### 3.3 Fonction de hashage

```python
def hash_ngram(
    self,
    ngram_ids: torch.Tensor,  # [batch, seq_len, n]
    n: int,
    head: int,
    table_size: int
) -> torch.Tensor:
    """
    Hash multiplicatif-XOR pour N-grams.
    
    Args:
        ngram_ids: IDs canoniques du N-gram
        n: ordre du N-gram
        head: index du head
        table_size: taille de la table (nombre premier)
    
    Returns:
        indices: [batch, seq_len] indices dans la table
    """
    # Récupérer les seeds pour ce head
    seeds = self.hash_seeds[:n, head]  # [n]
    
    # Hash multiplicatif par position
    # h = (Σ_i seed_i * id_i) mod table_size
    # Avec XOR pour meilleure distribution
    
    hash_val = torch.zeros(ngram_ids.shape[:-1], dtype=torch.long, device=ngram_ids.device)
    
    for i in range(n):
        # Multiplication avec overflow handling
        term = ngram_ids[..., i] * seeds[i].item()
        hash_val = hash_val ^ term
    
    # Modulo par taille de table (premier)
    indices = hash_val % table_size
    
    return indices
```

---

## 4. Module de Retrieval

### 4.1 Extraction des N-grams

```python
def extract_ngrams(
    self,
    canonical_ids: torch.Tensor,  # [batch, seq_len]
    n: int
) -> torch.Tensor:
    """
    Extrait les suffix N-grams pour chaque position.
    
    Pour position t, extrait [x_{t-n+1}, ..., x_t]
    Les positions < n-1 sont paddées avec 0.
    
    Returns:
        ngrams: [batch, seq_len, n]
    """
    batch_size, seq_len = canonical_ids.shape
    device = canonical_ids.device
    
    # Padding à gauche
    padded = F.pad(canonical_ids, (n - 1, 0), value=0)  # [batch, seq_len + n - 1]
    
    # Unfold pour extraire les fenêtres
    ngrams = padded.unfold(dimension=1, size=n, step=1)  # [batch, seq_len, n]
    
    return ngrams
```

### 4.2 Retrieval complet

```python
def retrieve(
    self,
    token_ids: torch.Tensor,  # [batch, seq_len] IDs originaux
    tokenizer_compression: TokenizerCompression
) -> torch.Tensor:
    """
    Récupère et concatène tous les embeddings N-gram.
    
    Returns:
        embeddings: [batch, seq_len, d_mem]
    """
    # 1. Compression du tokenizer
    canonical_ids = tokenizer_compression.compress(token_ids)
    
    # 2. Récupération pour chaque ordre et head
    all_embeddings = []
    
    for n in self.config.n_gram_orders:
        # Extraire N-grams
        ngrams = self.extract_ngrams(canonical_ids, n)  # [B, T, n]
        
        for k in range(self.config.num_heads):
            table_key = f"n{n}_h{k}"
            table = self.tables[table_key]
            table_size = self.config.table_sizes[(n, k)]
            
            # Hash vers indices
            indices = self.hash_ngram(ngrams, n, k, table_size)  # [B, T]
            
            # Lookup
            emb = table(indices)  # [B, T, slot_dim]
            all_embeddings.append(emb)
    
    # 3. Concaténation
    output = torch.cat(all_embeddings, dim=-1)  # [B, T, d_mem]
    
    return output
```

---

## 5. Module de Gating Context-Aware

### 5.1 Architecture

```python
class EngramGating(nn.Module):
    def __init__(
        self,
        hidden_dim: int,      # d du backbone
        memory_dim: int,      # d_mem des embeddings récupérés
        num_branches: int = 1  # M pour multi-branch (mHC)
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.memory_dim = memory_dim
        self.num_branches = num_branches
        
        # Projection Value (partagée entre branches)
        self.w_v = nn.Linear(memory_dim, hidden_dim, bias=False)
        
        # Projections Key (une par branche)
        self.w_k = nn.ModuleList([
            nn.Linear(memory_dim, hidden_dim, bias=False)
            for _ in range(num_branches)
        ])
        
        # RMSNorm pour query et key
        self.query_norm = RMSNorm(hidden_dim)
        self.key_norm = RMSNorm(hidden_dim)
        
        # Scaling factor
        self.scale = hidden_dim ** -0.5
    
    def forward(
        self,
        hidden_states: torch.Tensor,   # [B, T, d] ou list de [B, T, d] si multi-branch
        memory: torch.Tensor           # [B, T, d_mem]
    ) -> torch.Tensor:
        """
        Applique le gating context-aware.
        
        Returns:
            gated_values: [B, T, d] ou list si multi-branch
        """
        # Value projection (partagée)
        v = self.w_v(memory)  # [B, T, d]
        
        if self.num_branches == 1:
            # Single branch
            h = hidden_states
            k = self.w_k[0](memory)  # [B, T, d]
            
            # Normalized dot product
            q_norm = self.query_norm(h)
            k_norm = self.key_norm(k)
            
            # Gate: sigmoid of scaled dot product
            # [B, T, d] @ [B, T, d] -> [B, T] (sum over d)
            gate = torch.sigmoid(
                (q_norm * k_norm).sum(dim=-1, keepdim=True) * self.scale
            )  # [B, T, 1]
            
            return gate * v  # [B, T, d]
        
        else:
            # Multi-branch (mHC)
            outputs = []
            for m in range(self.num_branches):
                h_m = hidden_states[m]  # [B, T, d]
                k_m = self.w_k[m](memory)  # [B, T, d]
                
                q_norm = self.query_norm(h_m)
                k_norm = self.key_norm(k_m)
                
                gate_m = torch.sigmoid(
                    (q_norm * k_norm).sum(dim=-1, keepdim=True) * self.scale
                )
                
                outputs.append(gate_m * v)
            
            return outputs  # List of [B, T, d]
```

### 5.2 RMSNorm

```python
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # RMS normalization
        rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * rms * self.weight
```

---

## 6. Convolution Causale

### 6.1 Spécification

```python
class EngramConv(nn.Module):
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
        
        # Padding causal
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
        
        # RMSNorm avant conv
        self.norm = RMSNorm(dim)
        
        # IMPORTANT: Zero initialization pour identity mapping au départ
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
        
        # Reshape pour conv1d: [B, T, D] -> [B, D, T]
        x_conv = x_norm.transpose(1, 2)
        
        # Causal padding (à gauche seulement)
        x_padded = F.pad(x_conv, (self.padding, 0))
        
        # Convolution
        conv_out = self.conv(x_padded)  # [B, D, T]
        
        # Reshape back: [B, D, T] -> [B, T, D]
        conv_out = conv_out.transpose(1, 2)
        
        # SiLU activation + residual
        y = F.silu(conv_out) + x
        
        return y
```

---

## 7. Module Engram Complet

### 7.1 Assemblage

```python
class Engram(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        config: EngramTableConfig,
        tokenizer_compression: TokenizerCompression,
        num_branches: int = 1,
        conv_kernel_size: int = 4,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.config = config
        self.tokenizer_compression = tokenizer_compression
        
        # 1. Tables d'embeddings
        self.embeddings = EngramEmbeddings(config)
        
        # 2. Gating
        self.gating = EngramGating(
            hidden_dim=hidden_dim,
            memory_dim=config.embed_dim,
            num_branches=num_branches
        )
        
        # 3. Convolution
        self.conv = EngramConv(
            dim=hidden_dim,
            kernel_size=conv_kernel_size,
            dilation=max(config.n_gram_orders)
        )
    
    def forward(
        self,
        hidden_states: torch.Tensor,  # [B, T, d] ou list pour multi-branch
        input_ids: torch.Tensor       # [B, T] token IDs originaux
    ) -> torch.Tensor:
        """
        Forward pass complet du module Engram.
        
        Returns:
            output: [B, T, d] ou list pour multi-branch
        """
        # 1. Retrieval
        memory = self.embeddings.retrieve(
            input_ids, 
            self.tokenizer_compression
        )  # [B, T, d_mem]
        
        # 2. Gating
        gated = self.gating(hidden_states, memory)  # [B, T, d] ou list
        
        # 3. Convolution (appliquée à chaque branche si multi-branch)
        if isinstance(gated, list):
            output = [self.conv(g) for g in gated]
        else:
            output = self.conv(gated)
        
        return output
```

### 7.2 Intégration dans un Transformer Block

```python
class TransformerBlockWithEngram(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        ffn_dim: int,
        engram_config: Optional[EngramTableConfig] = None,
        tokenizer_compression: Optional[TokenizerCompression] = None,
        # ... autres params
    ):
        super().__init__()
        
        # Engram (optionnel - seulement sur certaines couches)
        self.engram = None
        if engram_config is not None:
            self.engram = Engram(
                hidden_dim=hidden_dim,
                config=engram_config,
                tokenizer_compression=tokenizer_compression
            )
        
        # Attention
        self.attention = MultiHeadAttention(hidden_dim, num_heads)
        self.attn_norm = RMSNorm(hidden_dim)
        
        # FFN/MoE
        self.ffn = FeedForward(hidden_dim, ffn_dim)  # ou MoE
        self.ffn_norm = RMSNorm(hidden_dim)
    
    def forward(
        self,
        x: torch.Tensor,           # [B, T, d]
        input_ids: torch.Tensor,   # [B, T] pour Engram
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        
        # 1. Engram (si présent)
        if self.engram is not None:
            engram_out = self.engram(x, input_ids)
            x = x + engram_out  # Residual connection
        
        # 2. Attention
        x = x + self.attention(self.attn_norm(x), attention_mask)
        
        # 3. FFN/MoE
        x = x + self.ffn(self.ffn_norm(x))
        
        return x
```

---

## 8. Configuration de l'Optimiseur

### 8.1 Groupes de paramètres séparés

```python
def configure_optimizers(
    model: nn.Module,
    backbone_lr: float = 4e-4,
    engram_embed_lr_multiplier: float = 5.0,
    weight_decay: float = 0.1
) -> torch.optim.Optimizer:
    """
    Configure les optimiseurs avec traitements différenciés:
    - Backbone: Muon ou AdamW avec weight decay
    - Engram embeddings: Adam sans weight decay, LR × 5
    - Engram conv: inclus dans backbone (zero-init)
    """
    
    # Séparer les paramètres
    engram_embed_params = []
    other_params_decay = []
    other_params_no_decay = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
            
        if 'engram' in name and 'embeddings' in name:
            # Embeddings Engram: traitement spécial
            engram_embed_params.append(param)
        elif 'bias' in name or 'norm' in name or 'embedding' in name:
            # Pas de weight decay pour bias, norms, embeddings vocab
            other_params_no_decay.append(param)
        else:
            other_params_decay.append(param)
    
    # Groupes d'optimisation
    param_groups = [
        {
            'params': other_params_decay,
            'lr': backbone_lr,
            'weight_decay': weight_decay
        },
        {
            'params': other_params_no_decay,
            'lr': backbone_lr,
            'weight_decay': 0.0
        },
        {
            'params': engram_embed_params,
            'lr': backbone_lr * engram_embed_lr_multiplier,
            'weight_decay': 0.0  # IMPORTANT: pas de weight decay
        }
    ]
    
    # Utiliser Adam pour tout (ou Muon pour backbone si disponible)
    optimizer = torch.optim.AdamW(param_groups, betas=(0.9, 0.95))
    
    return optimizer
```

---

## 9. Configurations de Référence

### 9.1 Engram-27B (config paper)

```python
ENGRAM_27B_CONFIG = {
    # Backbone
    'hidden_dim': 2560,
    'num_layers': 30,
    'num_heads': 32,
    'vocab_size': 129280,
    
    # MoE
    'num_routed_experts': 55,  # Réduit de 72
    'num_shared_experts': 2,
    'top_k': 6,
    
    # Engram
    'engram_layers': [2, 15],  # Couches où Engram est appliqué
    'engram_config': EngramTableConfig(
        n_gram_orders=[2, 3],
        num_heads=8,
        embed_dim=1280,
        # ~5.7B params pour embeddings
    ),
    
    # Convolution
    'conv_kernel_size': 4,
    'conv_dilation': 3,  # = max(n_gram_orders)
    
    # Multi-branch (mHC)
    'num_branches': 4,
    'mhc_expansion_rate': 4,
    
    # Training
    'batch_size': 1280,
    'seq_len': 4096,
    'total_tokens': 262e9,
    'backbone_lr': 4e-4,
    'engram_lr_multiplier': 5.0,
    'weight_decay': 0.1,
}
```

### 9.2 Configuration minimale pour expérimentation

```python
ENGRAM_SMALL_CONFIG = {
    # Backbone (style GPT-2 small)
    'hidden_dim': 768,
    'num_layers': 12,
    'num_heads': 12,
    'vocab_size': 50257,
    
    # Pas de MoE pour simplicité
    'ffn_dim': 3072,
    
    # Engram
    'engram_layers': [2, 6],
    'engram_config': EngramTableConfig(
        n_gram_orders=[2, 3],
        num_heads=4,
        embed_dim=512,
        # ~100M params pour embeddings
        table_sizes={
            (2, 0): 6553, (2, 1): 6563, (2, 2): 6569, (2, 3): 6571,
            (3, 0): 6577, (3, 1): 6581, (3, 2): 6599, (3, 3): 6607,
        }
    ),
    
    # Training
    'batch_size': 64,
    'seq_len': 1024,
}
```

---

## 10. Considérations d'Implémentation

### 10.1 Efficacité mémoire

```python
# Pour le training: sharding des tables sur GPUs
class ShardedEngramEmbeddings(EngramEmbeddings):
    def __init__(self, config: EngramTableConfig, world_size: int, rank: int):
        # Chaque GPU ne stocke qu'une partie des tables
        self.world_size = world_size
        self.rank = rank
        
        # Shard les tables par head
        self.local_heads = list(range(rank, config.num_heads, world_size))
        
        # Ne créer que les tables locales
        # ...
    
    def retrieve(self, token_ids, compression):
        # All-to-All pour gather les embeddings des autres GPUs
        # ...
```

### 10.2 Prefetching pour inference

```python
class EngramWithPrefetch(Engram):
    """Version avec prefetching asynchrone pour inference."""
    
    def prefetch(self, input_ids: torch.Tensor):
        """
        Pré-calcule les indices et lance le transfert depuis host memory.
        Appelé avant le forward du bloc précédent.
        """
        canonical_ids = self.tokenizer_compression.compress(input_ids)
        
        # Calculer tous les indices de hash
        indices_to_fetch = {}
        for n in self.config.n_gram_orders:
            ngrams = self.extract_ngrams(canonical_ids, n)
            for k in range(self.config.num_heads):
                indices = self.hash_ngram(ngrams, n, k, self.config.table_sizes[(n, k)])
                indices_to_fetch[(n, k)] = indices
        
        # Lancer transfert asynchrone depuis CPU
        self._prefetch_future = self._async_gather(indices_to_fetch)
    
    def forward(self, hidden_states, input_ids):
        # Attendre le prefetch si pas encore terminé
        if hasattr(self, '_prefetch_future'):
            memory = self._prefetch_future.result()
        else:
            memory = self.embeddings.retrieve(input_ids, self.tokenizer_compression)
        
        # Suite du forward normal
        # ...
```

### 10.3 Tests unitaires recommandés

```python
def test_tokenizer_compression():
    """Vérifier que les tokens équivalents sont mappés au même ID."""
    compression = build_tokenizer_compression(tokenizer)
    
    # "Apple" et "apple" doivent avoir le même ID canonique
    id_apple = tokenizer.encode("Apple")[0]
    id_apple_lower = tokenizer.encode("apple")[0]
    
    assert compression[id_apple] == compression[id_apple_lower]

def test_ngram_extraction():
    """Vérifier l'extraction correcte des N-grams."""
    ids = torch.tensor([[1, 2, 3, 4, 5]])
    
    # 2-grams: [0,1], [1,2], [2,3], [3,4], [4,5]
    ngrams_2 = extract_ngrams(ids, n=2)
    assert ngrams_2.shape == (1, 5, 2)
    assert ngrams_2[0, 2].tolist() == [2, 3]

def test_hash_collision_rate():
    """Mesurer le taux de collision sur un corpus."""
    # Devrait être < 1% pour des tables correctement dimensionnées
    pass

def test_zero_init_conv():
    """Vérifier que la conv est initialisée à zéro."""
    conv = EngramConv(dim=768)
    x = torch.randn(2, 100, 768)
    
    # Au début, output = input (identity)
    y = conv(x)
    assert torch.allclose(y, x, atol=1e-5)

def test_gating_suppression():
    """Vérifier que le gate supprime les embeddings non pertinents."""
    # Si memory est orthogonal à hidden_state, gate → 0
    pass
```

---

## 11. Métriques de Monitoring

### 11.1 Métriques à logger pendant le training

```python
def log_engram_metrics(engram_module, step):
    metrics = {}
    
    # 1. Distribution des gates
    # Moyenne et variance des α_t sur le batch
    metrics['engram/gate_mean'] = engram_module.last_gate_values.mean()
    metrics['engram/gate_std'] = engram_module.last_gate_values.std()
    
    # 2. Taux d'activation (gates > 0.5)
    metrics['engram/activation_rate'] = (engram_module.last_gate_values > 0.5).float().mean()
    
    # 3. Norme des embeddings récupérés
    metrics['engram/memory_norm'] = engram_module.last_memory.norm(dim=-1).mean()
    
    # 4. Norme des gradients des tables
    for name, param in engram_module.embeddings.named_parameters():
        if param.grad is not None:
            metrics[f'engram/grad_norm_{name}'] = param.grad.norm()
    
    return metrics
```

### 11.2 Analyse qualitative

```python
def visualize_gating(model, tokenizer, text: str):
    """
    Visualise les patterns de gating comme dans Figure 7 du paper.
    """
    input_ids = tokenizer.encode(text, return_tensors='pt')
    
    with torch.no_grad():
        # Forward avec capture des gates
        _, gate_values = model.forward_with_gate_capture(input_ids)
    
    # gate_values: [num_engram_layers, batch, seq_len, num_branches]
    
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    
    # Créer heatmap
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=(20, 4))
    
    # Moyenne sur les branches et couches
    gates = gate_values.mean(dim=(0, 3))[0]  # [seq_len]
    
    # Plot
    for i, (token, gate) in enumerate(zip(tokens, gates)):
        color = plt.cm.Reds(gate.item())
        ax.text(i, 0, token, fontsize=10, ha='center', 
                bbox=dict(boxstyle='round', facecolor=color))
    
    plt.show()
```

---

## 12. Checklist d'Implémentation

### Phase 1: Core Components
- [ ] `TokenizerCompression`: mapping V → V'
- [ ] `EngramEmbeddings`: tables avec hash multi-head
- [ ] `hash_ngram`: fonction de hashage
- [ ] `extract_ngrams`: extraction des suffix N-grams

### Phase 2: Gating & Fusion
- [ ] `RMSNorm`: normalisation
- [ ] `EngramGating`: mécanisme de gate
- [ ] `EngramConv`: convolution causale (zero-init!)

### Phase 3: Integration
- [ ] `Engram`: module complet
- [ ] `TransformerBlockWithEngram`: intégration dans backbone
- [ ] Configuration optimiseur (LR ×5, no weight decay pour embeds)

### Phase 4: Validation
- [ ] Tests unitaires
- [ ] Benchmark sur tâche simple (language modeling)
- [ ] Comparaison avec baseline sans Engram

### Phase 5: Optimisation (optionnel)
- [ ] Sharding pour multi-GPU
- [ ] Prefetching pour inference
- [ ] Cache hiérarchique (Zipfian)

---

## Annexe A: Calcul du Budget Paramètres

```python
def compute_engram_params(config: EngramTableConfig) -> int:
    """Calcule le nombre total de paramètres Engram."""
    
    total = 0
    slot_dim = config.embed_dim // (len(config.n_gram_orders) * config.num_heads)
    
    # Tables d'embeddings
    for n in config.n_gram_orders:
        for k in range(config.num_heads):
            table_size = config.table_sizes[(n, k)]
            total += table_size * slot_dim
    
    return total

# Exemple: vérification pour config 27B
config = EngramTableConfig(
    n_gram_orders=[2, 3],
    num_heads=8,
    embed_dim=1280,
    # Tables de ~2.26M slots chacune
    # slot_dim = 1280 / (2 * 8) = 80
    # 16 tables × 2.26M × 80 = ~2.9B params juste pour embeddings
    # + W_K (M=4), W_V, conv = reste pour atteindre 5.7B
)
```

---

## Annexe B: Références Utiles

- **Paper original**: https://arxiv.org/abs/XXXX.XXXXX
- **Code officiel**: https://github.com/deepseek-ai/Engram
- **DeepSeekMoE**: https://arxiv.org/abs/2401.06066
- **mHC (Hyper-Connections)**: https://arxiv.org/abs/2512.24880
- **Hash Embeddings**: Tito Svenstrup et al., NeurIPS 2017