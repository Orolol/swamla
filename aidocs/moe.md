# Avancées MoE 2025 pour Small LLMs - Guide Technique Approfondi

## Table des matières
1. [Fondamentaux MoE et évolutions 2025](#1-fondamentaux-moe-et-évolutions-2025)
2. [Mécanismes de Routing en profondeur](#2-mécanismes-de-routing-en-profondeur)
3. [Load Balancing: Théorie et Pratique](#3-load-balancing-théorie-et-pratique)
4. [Shared Experts: Analyse et Dimensionnement](#4-shared-experts-analyse-et-dimensionnement)
5. [Attention Sparse + MoE](#5-attention-sparse--moe)
6. [Optimisations GPU et Mémoire](#6-optimisations-gpu-et-mémoire)
7. [Compression et Inference](#7-compression-et-inference)
8. [Implémentation de Référence](#8-implémentation-de-référence)
9. [Debugging et Diagnostic](#9-debugging-et-diagnostic)
10. [Configuration Finale pour Small LLMs](#10-configuration-finale-pour-small-llms)

---

## 1. Fondamentaux MoE et évolutions 2025

### 1.1 Formulation mathématique

Un layer MoE remplace le FFN dense par un ensemble d'experts avec routing conditionnel:

**FFN Dense classique:**
```
y = W₂ · σ(W₁ · x)
```

**MoE Layer:**
```
y = Σᵢ gᵢ(x) · Eᵢ(x)
```

où:
- `Eᵢ(x)` = output de l'expert i pour l'input x
- `gᵢ(x)` = poids de gating pour l'expert i (déterminé par le router)
- La somme est sur les K experts sélectionnés (sparse)

### 1.2 Pourquoi MoE fonctionne: perspective théorique

**Hypothèse de spécialisation:** Les données d'entraînement contiennent des clusters naturels (syntaxe, sémantique, domaines, langues). Un expert peut se spécialiser sur un cluster avec moins de paramètres qu'un modèle dense généraliste.

**Scaling laws MoE (2024-2025):**
- Avec budget compute fixe C, un MoE atteint la même loss qu'un dense avec **20-40x moins de FLOPs**
- Cet avantage **ne plafonne pas** avec plus de tokens (contrairement aux findings de 2022)
- Formule empirique: `Loss_MoE(N_active, N_total) ≈ Loss_dense(N_active × √(N_total/N_active))`

### 1.3 Évolution architecturale 2024→2025

| Aspect | 2024 (DeepSeek-V3) | 2025 (Kimi K2) | Pourquoi le changement |
|--------|-------------------|----------------|------------------------|
| Experts totaux | 256 | 384 | Plus de spécialisation |
| Dense init layers | 3 | 1 | K2 utilise MuonClip → moins besoin de stabilisation |
| Shared experts | 1 | 1 | Convergence vers minimum efficace |
| Optimizer | AdamW | MuonClip | Token-efficiency + stabilité native |
| Params activés | 37B | 32B | Efficacité accrue |

### 1.4 Modèles de référence 2025

**Kimi K2 (Juillet 2025)** - Le nouveau benchmark:
```
Total params:     1.04T
Active params:    32B
Experts:          384 routed + 1 shared
Top-K:            8
Attention heads:  64
Context:          128K
Training tokens:  15.5T
```

**Innovations architecturales K2:**
1. **MuonClip optimizer**: Muon + QK-Clip pour stabilité sans loss spikes
2. **Single dense layer**: Réduit de 3 à 1 (suffisant avec MuonClip)
3. **Agentic training pipeline**: Synthèse de données tool-use à grande échelle

---

## 2. Mécanismes de Routing en profondeur

### 2.1 Router classique: TopK + Softmax

**Formulation:**
```python
# Scores de gating
scores = x @ W_gate  # [batch, seq, n_experts]

# Softmax pour normalisation
probs = softmax(scores, dim=-1)

# Sélection TopK
topk_probs, topk_indices = torch.topk(probs, k=top_k, dim=-1)

# Re-normalisation des poids sélectionnés
weights = topk_probs / topk_probs.sum(dim=-1, keepdim=True)
```

**Problèmes:**
1. **Non-différentiabilité du TopK**: Gradient ne passe pas à travers la sélection
2. **Winner-take-all**: Softmax amplifie les différences → routing collapse
3. **Load imbalance**: Certains experts sur-utilisés, d'autres ignorés

### 2.2 Sigmoid Routing (DeepSeek-V3, 2024)

**Insight clé:** Séparer la décision de routing de la valeur de gating.

```python
# Affinité sigmoid (pas softmax!)
affinity = torch.sigmoid(x @ W_gate)  # [B, S, E], valeurs dans [0,1]

# Décision de routing: affinity + bias learnable
routing_scores = affinity + expert_bias  # bias ajusté dynamiquement

# Sélection des top-K
_, selected = torch.topk(routing_scores, k=top_k, dim=-1)

# CRUCIAL: Gating weights = affinity normalisée, SANS le bias
selected_affinity = torch.gather(affinity, -1, selected)
weights = selected_affinity / selected_affinity.sum(dim=-1, keepdim=True)
```

**Pourquoi ça marche mieux:**
1. **Sigmoid vs Softmax**: Sigmoid traite chaque expert indépendamment (pas de compétition)
2. **Bias séparé**: Influence le routing mais pas les gradients des weights
3. **Pas de winner-take-all**: Distribution plus uniforme naturellement

### 2.3 ReMoE: ReLU Routing (ICLR 2025)

**Innovation:** Éliminer complètement le TopK avec du gating ReLU.

```python
class ReMoERouter(nn.Module):
    def __init__(self, d_model, n_experts, target_sparsity=0.125):
        self.gate = nn.Linear(d_model, n_experts)
        self.bias = nn.Parameter(torch.zeros(n_experts))
        self.target_sparsity = target_sparsity  # e.g., 8/64 = 0.125
        
    def forward(self, x):
        # ReLU gating - pas de TopK!
        raw_scores = self.gate(x)
        gates = F.relu(raw_scores - self.bias)  # Sparse naturellement
        
        # Normalisation (somme des gates actifs)
        gates = gates / (gates.sum(dim=-1, keepdim=True) + 1e-6)
        
        return gates  # [B, S, E] avec beaucoup de zéros
    
    def compute_sparsity_loss(self, gates):
        # L1 adaptative pour atteindre la sparsité cible
        actual_sparsity = (gates > 0).float().mean()
        return F.mse_loss(actual_sparsity, torch.tensor(self.target_sparsity))
```

**Avantages:**
- Entièrement différentiable (pas de STE nécessaire)
- Sparsité variable par token (certains tokens utilisent plus d'experts)
- Compatible auto-régressif

**Trade-off:** Nécessite régularisation L1 pour contrôler la sparsité.

### 2.4 MoNE: Mixture of Neuron Experts (Octobre 2025)

**Observation:** Même à l'intérieur des experts MoE, les activations sont très sparses. 60% des neurones peuvent être prunés sans dégradation.

**Concept:** Appliquer TopK **dans** chaque expert, au niveau des neurones.

```python
class MoNEExpert(nn.Module):
    """Expert avec sparsité interne au niveau des neurones"""
    def __init__(self, d_model, d_ff, neuron_sparsity=0.5):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)  # gate
        self.w2 = nn.Linear(d_ff, d_model, bias=False)  # down
        self.w3 = nn.Linear(d_model, d_ff, bias=False)  # up
        self.k = int(d_ff * (1 - neuron_sparsity))  # neurons to keep
        
    def forward(self, x):
        # Gate projection (détermine l'importance des neurones)
        gate = self.w1(x)  # [B, S, d_ff]
        
        # TopK selection sur les neurones (pas les experts!)
        gate_abs = gate.abs()
        topk_values, topk_indices = torch.topk(gate_abs, self.k, dim=-1)
        
        # Créer masque sparse
        mask = torch.zeros_like(gate)
        mask.scatter_(-1, topk_indices, 1.0)
        
        # Appliquer SwiGLU avec masque
        gate_masked = F.silu(gate) * mask
        up = self.w3(x) * mask
        
        return self.w2(gate_masked * up)
```

**Gains:**
- 2x speedup inference avec même performance
- Pas de paramètres de routing additionnels
- Complètement orthogonal au routing MoE (combinable)

### 2.5 Router Upcycling (Août 2025)

**Problème:** Les routers linéaires simples peinent lors de l'upcycling dense→MoE.

**Solution:** Initialiser les routers depuis les attention heads pré-entraînées.

```python
class UpcycledRouter(nn.Module):
    """Router initialisé depuis les attention heads"""
    def __init__(self, pretrained_attn, n_experts):
        super().__init__()
        # Utiliser les heads d'attention comme base
        self.n_routers = pretrained_attn.n_heads
        
        # Chaque "router" est une projection Q initialisée depuis une head
        self.query_projs = nn.ModuleList([
            nn.Linear(pretrained_attn.d_model, pretrained_attn.d_head)
            for _ in range(self.n_routers)
        ])
        
        # Initialiser depuis les poids pré-entraînés
        for i, proj in enumerate(self.query_projs):
            proj.weight.data = pretrained_attn.W_Q.weight.data[
                i*pretrained_attn.d_head:(i+1)*pretrained_attn.d_head
            ]
        
        # Keys pour chaque expert (learnable)
        self.expert_keys = nn.Parameter(
            torch.randn(n_experts, pretrained_attn.d_head)
        )
        
    def forward(self, x):
        # Multi-router scoring
        scores_list = []
        for proj in self.query_projs:
            query = proj(x)  # [B, S, d_head]
            scores = query @ self.expert_keys.T  # [B, S, n_experts]
            scores_list.append(scores)
        
        # Aggregate scores across routers
        combined_scores = torch.stack(scores_list, dim=-1).mean(dim=-1)
        
        return combined_scores
```

**Intuition:** Les attention heads ont déjà appris à "router" l'attention vers les tokens pertinents. Cette capacité transfère bien au routing d'experts.

---

## 3. Load Balancing: Théorie et Pratique

### 3.1 Le problème du routing collapse

**Symptômes:**
- Quelques experts reçoivent 90%+ des tokens
- Les autres experts ne s'entraînent pas (gradients ~0)
- Performance dégradée (modèle équivalent à un petit dense)

**Causes:**
1. **Rich-get-richer**: Experts bien initialisés attirent plus de tokens → meilleurs gradients → encore plus de tokens
2. **Softmax amplification**: Petites différences de score → grandes différences de probabilité
3. **Batch effects**: Certains experts "matchent" mieux les données courantes

### 3.2 Auxiliary Loss traditionnelle

**Formulation (Switch Transformer, GShard):**
```python
def load_balancing_loss(router_probs, expert_mask, n_experts):
    """
    router_probs: [B, S, E] - probabilités du router
    expert_mask: [B, S, E] - masque binaire des experts sélectionnés
    """
    # Fraction de tokens routés vers chaque expert
    tokens_per_expert = expert_mask.float().mean(dim=[0, 1])  # [E]
    
    # Probabilité moyenne de chaque expert
    router_prob_per_expert = router_probs.mean(dim=[0, 1])  # [E]
    
    # Loss = N * Σ (fraction_i * prob_i)
    # Encourage: fraction uniforme ET probs uniformes
    loss = n_experts * (tokens_per_expert * router_prob_per_expert).sum()
    
    return loss
```

**Problème:** Le coefficient α est un hyperparamètre critique.
- α trop petit (< 0.001): Load imbalance persiste
- α trop grand (> 0.1): Dégrade significativement la loss principale

### 3.3 Auxiliary-Loss-Free Balancing (DeepSeek-V3)

**Insight:** Utiliser un bias dynamique qui affecte SEULEMENT le routing, pas les gradients.

```python
class AuxLossFreeRouter(nn.Module):
    def __init__(self, d_model, n_experts, top_k, gamma=0.001):
        super().__init__()
        self.gate = nn.Linear(d_model, n_experts, bias=False)
        self.expert_bias = nn.Parameter(torch.zeros(n_experts), requires_grad=False)
        self.gamma = gamma
        self.top_k = top_k
        self.n_experts = n_experts
        
    def forward(self, x):
        B, S, D = x.shape
        x_flat = x.view(-1, D)
        N = x_flat.shape[0]  # total tokens
        
        # 1. Compute affinity (sigmoid, not softmax)
        affinity = torch.sigmoid(self.gate(x_flat))  # [N, E]
        
        # 2. Routing decision = affinity + bias
        routing_scores = affinity + self.expert_bias
        _, selected_experts = torch.topk(routing_scores, self.top_k, dim=-1)
        
        # 3. Gating weights = normalized affinity (WITHOUT bias!)
        selected_affinity = torch.gather(affinity, 1, selected_experts)
        gating_weights = selected_affinity / (selected_affinity.sum(-1, keepdim=True) + 1e-9)
        
        # 4. Update bias based on load (training only, no gradients)
        if self.training:
            with torch.no_grad():
                # Count tokens per expert
                load = torch.zeros(self.n_experts, device=x.device)
                for k in range(self.top_k):
                    expert_indices = selected_experts[:, k]
                    load.scatter_add_(
                        0, expert_indices,
                        torch.ones(N, device=x.device)
                    )
                
                # Expected load if perfectly balanced
                expected_load = N * self.top_k / self.n_experts
                
                # Update bias: reduce for overloaded, increase for underloaded
                # sign() donne -1, 0, ou +1
                self.expert_bias.data -= self.gamma * (load - expected_load).sign()
        
        return selected_experts, gating_weights
```

**Pourquoi ça marche:**

1. **Séparation des préoccupations:**
   - `affinity` → détermine la contribution de chaque expert (gradient flows)
   - `bias` → détermine la sélection (no gradient)

2. **Feedback loop stable:**
   - Expert surchargé → bias diminue → moins sélectionné
   - Expert sous-utilisé → bias augmente → plus sélectionné
   - Converge vers équilibre

3. **Pas d'interférence avec la loss:**
   - La loss principale optimise `affinity` pour la tâche
   - Le balancing est géré séparément par `bias`

**Hyperparamètres:**
- `gamma = 0.001`: Vitesse de mise à jour du bias
- Annealing: `gamma → 0` pour les derniers ~500B tokens (fige le routing)

### 3.4 Sequence-wise Auxiliary Loss (complémentaire)

DeepSeek-V3 ajoute une loss auxiliaire minimale pour prévenir les cas extrêmes:

```python
def sequence_auxiliary_loss(router_probs, alpha=0.0001):
    """
    Loss auxiliaire au niveau séquence (pas token)
    alpha beaucoup plus petit que les approches traditionnelles
    """
    # Moyenne sur la séquence
    seq_probs = router_probs.mean(dim=1)  # [B, E]
    
    # Variance across experts (encourage uniformité)
    variance = seq_probs.var(dim=-1).mean()
    
    return alpha * variance
```

**Coefficient:** α = 0.0001 (100x plus petit que traditionnel)

### 3.5 Router Z-Loss pour la stabilité numérique

**Problème:** Les logits du router peuvent devenir très grands → instabilité softmax/sigmoid.

```python
def router_z_loss(router_logits, coef=0.001):
    """
    Pénalise les logits larges pour éviter l'instabilité numérique
    
    router_logits: [B, S, E] - logits bruts avant sigmoid/softmax
    """
    # Log-sum-exp est une approximation smooth du max
    logsumexp = torch.logsumexp(router_logits, dim=-1)  # [B, S]
    
    # Pénaliser les grandes valeurs
    z_loss = logsumexp.pow(2).mean()
    
    return coef * z_loss
```

**Effet:** Force les logits à rester dans une plage raisonnable → gradients stables.

---

## 4. Shared Experts: Analyse et Dimensionnement

### 4.1 Rôle des shared experts

**Intuition:** Certaines connaissances sont universelles (grammaire, logique basique, patterns communs). Les shared experts capturent cette "base commune" tandis que les routed experts se spécialisent.

**Formulation:**
```
y = Σⱼ FFN_shared_j(x) + Σᵢ gᵢ(x) · FFN_routed_i(x)
```

### 4.2 Dimensionnement optimal

**Études de scaling (2024-2025):**

| Étude | Ratio optimal shared/activés | Notes |
|-------|------------------------------|-------|
| DeepSeekMoE | 1/8 (12.5%) | 1 shared + 7 routed |
| Scaling Laws MoE | 13-31% | 446 expériences |
| Kimi K2 | 1/9 (11%) | 1 shared + 8 routed |
| Qwen1.5-MoE | 4/8 (50%) | Plus agressif |

**Ablation DeepSeekMoE (critique):**
```
Configuration                    | Pile Loss
--------------------------------|----------
1 shared + 7 routed (baseline)  | 1.808
0 shared + 8 routed             | 2.414 (+33%!)
2 shared + 6 routed             | 1.802 (légèrement mieux)
```

→ Au moins 1 shared expert est **critique**. Au-delà, gains marginaux.

### 4.3 Implémentation efficace

```python
class MoEWithSharedExperts(nn.Module):
    def __init__(self, d_model, d_ff, n_experts, n_shared, top_k):
        super().__init__()
        self.n_shared = n_shared
        self.top_k = top_k
        
        # Shared experts (toujours activés)
        self.shared_experts = nn.ModuleList([
            SwiGLUExpert(d_model, d_ff) for _ in range(n_shared)
        ])
        
        # Routed experts
        self.routed_experts = nn.ModuleList([
            SwiGLUExpert(d_model, d_ff) for _ in range(n_experts)
        ])
        
        # Router (seulement pour routed experts)
        self.router = AuxLossFreeRouter(d_model, n_experts, top_k)
        
    def forward(self, x):
        # Shared experts: toujours calculés
        # Optimization: peut être fusionné en un seul matmul si même taille
        shared_output = torch.zeros_like(x)
        for expert in self.shared_experts:
            shared_output = shared_output + expert(x)
        
        # Routed experts
        selected, weights = self.router(x)
        routed_output = self._compute_routed(x, selected, weights)
        
        return shared_output + routed_output
```

**Optimisation mémoire:**
```python
# Si tous les shared experts ont la même taille, fusionner les poids
class FusedSharedExperts(nn.Module):
    def __init__(self, d_model, d_ff, n_shared):
        super().__init__()
        # Un seul set de poids, dimension augmentée
        self.w1 = nn.Linear(d_model, d_ff * n_shared, bias=False)
        self.w2 = nn.Linear(d_ff * n_shared, d_model, bias=False)
        self.w3 = nn.Linear(d_model, d_ff * n_shared, bias=False)
        self.n_shared = n_shared
        self.d_ff = d_ff
        
    def forward(self, x):
        # Un seul forward, puis reshape + sum
        gate = F.silu(self.w1(x))  # [B, S, d_ff * n_shared]
        up = self.w3(x)
        hidden = gate * up
        
        # Reshape to [B, S, n_shared, d_ff] et sum
        hidden = hidden.view(*x.shape[:-1], self.n_shared, self.d_ff)
        hidden = hidden.sum(dim=-2)  # [B, S, d_ff]
        
        # Project back (ajuster w2 dimensions)
        # Note: nécessite adaptation de w2
        return self.w2_adapted(hidden)
```

---

## 5. Attention Sparse + MoE

### 5.1 Native Sparse Attention (NSA) - DeepSeek, Février 2025

**Motivation:** L'attention full est O(n²). Pour les longs contextes (128K+), c'est prohibitif.

**Architecture NSA: 3 branches parallèles**

```
                    ┌─────────────────┐
                    │   Compressed    │ → Contexte global (coarse)
                    │   Attention     │
        ┌───────────┼─────────────────┤
Query ──┤           │   Selected      │ → Tokens importants (fine)
        │           │   Attention     │
        └───────────┼─────────────────┤
                    │   Sliding       │ → Contexte local
                    │   Attention     │
                    └─────────────────┘
```

```python
class NativeSparseAttention(nn.Module):
    def __init__(self, d_model, n_heads, window_size=512, 
                 n_compressed=64, n_selected_blocks=8, block_size=64):
        super().__init__()
        self.d_head = d_model // n_heads
        self.n_heads = n_heads
        self.window_size = window_size
        self.n_selected_blocks = n_selected_blocks
        self.block_size = block_size
        
        # Projections Q, K, V
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        # Compression pour branch 1
        self.compressor = nn.Linear(d_model, n_compressed)
        
        # Gating pour combiner les branches
        self.branch_gate = nn.Linear(d_model, 3)
        
    def forward(self, x, cache=None):
        B, S, D = x.shape
        
        q = self.W_q(x).view(B, S, self.n_heads, self.d_head)
        k = self.W_k(x).view(B, S, self.n_heads, self.d_head)
        v = self.W_v(x).view(B, S, self.n_heads, self.d_head)
        
        # Branch 1: Compressed attention (global context)
        attn_compressed = self._compressed_attention(q, k, v)
        
        # Branch 2: Selected attention (important tokens)
        attn_selected = self._selected_attention(q, k, v)
        
        # Branch 3: Sliding window (local)
        attn_sliding = self._sliding_attention(q, k, v)
        
        # Combine branches (learnable gating)
        gate = F.softmax(self.branch_gate(x), dim=-1)  # [B, S, 3]
        
        output = (
            gate[..., 0:1] * attn_compressed +
            gate[..., 1:2] * attn_selected +
            gate[..., 2:3] * attn_sliding
        )
        
        return self.W_o(output.view(B, S, D))
    
    def _compressed_attention(self, q, k, v):
        """Compresse les K,V pour contexte global"""
        B, S, H, D = k.shape
        
        # Compresser: [B, S, H, D] → [B, n_compressed, H, D]
        k_compressed = self.compressor(
            k.view(B, S, -1)
        ).view(B, -1, H, D)
        v_compressed = self.compressor(
            v.view(B, S, -1)
        ).view(B, -1, H, D)
        
        # Attention standard sur séquence compressée
        attn = F.scaled_dot_product_attention(q, k_compressed, v_compressed)
        return attn
    
    def _selected_attention(self, q, k, v):
        """Sélectionne les blocks les plus importants"""
        B, S, H, D = k.shape
        n_blocks = S // self.block_size
        
        # Reshape en blocks
        k_blocks = k.view(B, n_blocks, self.block_size, H, D)
        v_blocks = v.view(B, n_blocks, self.block_size, H, D)
        
        # Compute block importance (centroid-based)
        k_centroids = k_blocks.mean(dim=2)  # [B, n_blocks, H, D]
        
        # Score blocks par rapport à chaque query
        q_for_routing = q.mean(dim=2)  # Average over sequence for routing
        block_scores = torch.einsum('bhd,bnhd->bhn', q_for_routing, k_centroids)
        
        # Select top-k blocks
        _, top_blocks = torch.topk(block_scores, self.n_selected_blocks, dim=-1)
        
        # Gather selected blocks et compute attention
        # (simplified - real implementation uses custom CUDA kernels)
        selected_k = self._gather_blocks(k_blocks, top_blocks)
        selected_v = self._gather_blocks(v_blocks, top_blocks)
        
        return F.scaled_dot_product_attention(q, selected_k, selected_v)
    
    def _sliding_attention(self, q, k, v):
        """Attention sur fenêtre glissante locale"""
        # Utilise flash_attn avec masque causal + window
        # En pratique: flash_attn_varlen ou implémentation custom
        return sliding_window_attention(q, k, v, self.window_size)
```

**Résultats NSA:**
- Surpasse Full Attention sur benchmarks généraux malgré ~75% sparsité
- 2-3x speedup sur séquences 64K (tous les stages: decode, forward, backward)
- Particulièrement efficace pour chain-of-thought reasoning

### 5.2 MoBA: Mixture of Block Attention (Février 2025, NeurIPS Spotlight)

**Concept:** Appliquer les principes MoE à l'attention elle-même.

```python
class MixtureOfBlockAttention(nn.Module):
    """
    Au lieu de TopK experts, on a TopK blocks de K,V
    Le "router" détermine quels blocks sont pertinents pour chaque query
    """
    def __init__(self, d_model, n_heads, block_size=64, top_k_blocks=8):
        super().__init__()
        self.d_head = d_model // n_heads
        self.n_heads = n_heads
        self.block_size = block_size
        self.top_k = top_k_blocks
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
    def forward(self, x):
        B, S, D = x.shape
        n_blocks = S // self.block_size
        
        q = self.W_q(x).view(B, S, self.n_heads, self.d_head)
        k = self.W_k(x).view(B, S, self.n_heads, self.d_head)
        v = self.W_v(x).view(B, S, self.n_heads, self.d_head)
        
        # Reshape K, V into blocks
        k_blocks = k.view(B, n_blocks, self.block_size, self.n_heads, self.d_head)
        v_blocks = v.view(B, n_blocks, self.block_size, self.n_heads, self.d_head)
        
        # Compute block centroids for routing
        k_centroids = k_blocks.mean(dim=2)  # [B, n_blocks, H, D]
        
        # Route each query to top-k blocks
        # scores[b, s, h, n] = affinity of query (b,s,h) to block n
        scores = torch.einsum('bshd,bnhd->bshn', q, k_centroids)
        scores = scores / (self.d_head ** 0.5)
        
        # Select top-k blocks per query
        topk_scores, topk_indices = torch.topk(scores, self.top_k, dim=-1)
        
        # Attention sur blocks sélectionnés (sparse)
        # En pratique: utilise FlashMoBA kernel pour efficacité
        output = self._sparse_block_attention(
            q, k_blocks, v_blocks, topk_indices, topk_scores
        )
        
        return self.W_o(output.view(B, S, D))
    
    def _sparse_block_attention(self, q, k_blocks, v_blocks, indices, scores):
        """
        Compute attention only on selected blocks
        
        Real implementation uses custom CUDA kernels (FlashMoBA)
        This is a reference implementation
        """
        B, S, H, D = q.shape
        _, n_blocks, block_size, _, _ = k_blocks.shape
        
        output = torch.zeros_like(q)
        
        for b in range(B):
            for s in range(S):
                for h in range(H):
                    # Get selected block indices for this query
                    block_ids = indices[b, s, h]  # [top_k]
                    block_weights = F.softmax(scores[b, s, h], dim=-1)  # [top_k]
                    
                    query = q[b, s, h]  # [D]
                    
                    weighted_output = torch.zeros(D, device=q.device)
                    for k, (bid, weight) in enumerate(zip(block_ids, block_weights)):
                        # Attention on this block
                        keys = k_blocks[b, bid, :, h]  # [block_size, D]
                        values = v_blocks[b, bid, :, h]  # [block_size, D]
                        
                        attn_scores = query @ keys.T / (D ** 0.5)
                        attn_probs = F.softmax(attn_scores, dim=-1)
                        block_output = attn_probs @ values
                        
                        weighted_output += weight * block_output
                    
                    output[b, s, h] = weighted_output
        
        return output
```

**FlashMoBA (Novembre 2025):**
- Kernel CUDA optimisé: jusqu'à 14.7x speedup vs FlashAttention-2
- Analyse signal-to-noise pour guider le design (petits blocks = meilleur SNR)

### 5.3 Intégration NSA/MoBA avec MoE

**Pattern recommandé (DeepSeek-V3, Kimi K2):**

```python
class TransformerBlockWithSparsity(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        
        # Normalization
        self.norm1 = nn.RMSNorm(config.d_model)
        self.norm2 = nn.RMSNorm(config.d_model)
        
        # Attention: choisir selon le layer
        if config.use_sparse_attention and layer_idx >= config.sparse_attn_start_layer:
            self.attn = NativeSparseAttention(
                config.d_model, config.n_heads,
                window_size=config.window_size
            )
        else:
            self.attn = MultiHeadLatentAttention(
                config.d_model, config.n_heads,
                n_kv_heads=config.n_kv_heads,
                d_latent=config.d_latent
            )
        
        # FFN: MoE sauf pour les premières couches
        if layer_idx < config.n_dense_init_layers:
            self.ffn = SwiGLUFFN(config.d_model, config.d_ff)
        else:
            self.ffn = MoELayer(config)
    
    def forward(self, x, cache=None):
        x = x + self.attn(self.norm1(x), cache)
        x = x + self.ffn(self.norm2(x))
        return x
```

---

## 6. Optimisations GPU et Mémoire

### 6.1 Batched Expert Computation

**Problème:** Boucle naïve sur les experts = séquentiel = lent.

```python
# ❌ Mauvais: séquentiel
def naive_moe(x, experts, selected, weights):
    output = torch.zeros_like(x)
    for i, expert in enumerate(experts):
        mask = (selected == i).any(dim=-1)
        if mask.any():
            output[mask] += weights[mask, selected[mask] == i] * expert(x[mask])
    return output
```

**Solution: Regrouper les tokens par expert**

```python
def efficient_moe(x, experts, selected_experts, weights):
    """
    Efficient MoE computation with token grouping
    
    x: [B*S, D] - flattened input
    selected_experts: [B*S, K] - indices of selected experts
    weights: [B*S, K] - gating weights
    """
    B_S, K = selected_experts.shape
    D = x.shape[-1]
    n_experts = len(experts)
    
    # 1. Count tokens per expert
    tokens_per_expert = torch.zeros(n_experts, dtype=torch.long, device=x.device)
    for k in range(K):
        tokens_per_expert.scatter_add_(
            0, selected_experts[:, k],
            torch.ones(B_S, dtype=torch.long, device=x.device)
        )
    
    # 2. Create permutation indices to group tokens by expert
    # Sort by expert assignment
    flat_expert_indices = selected_experts.view(-1)  # [B*S*K]
    sorted_indices = torch.argsort(flat_expert_indices)
    
    # 3. Prepare batched inputs for each expert
    # Expand x for each of K selections
    x_expanded = x.unsqueeze(1).expand(-1, K, -1).reshape(-1, D)  # [B*S*K, D]
    weights_flat = weights.view(-1, 1)  # [B*S*K, 1]
    
    # Reorder by expert
    x_sorted = x_expanded[sorted_indices]
    weights_sorted = weights_flat[sorted_indices]
    
    # 4. Compute expert outputs (can be parallelized with grouped GEMM)
    outputs_sorted = torch.zeros_like(x_sorted)
    
    cumsum = torch.cumsum(tokens_per_expert, dim=0)
    starts = torch.cat([torch.zeros(1, device=x.device, dtype=torch.long), cumsum[:-1]])
    
    for i, expert in enumerate(experts):
        if tokens_per_expert[i] > 0:
            start, end = starts[i].item(), cumsum[i].item()
            expert_input = x_sorted[start:end]
            expert_output = expert(expert_input)
            outputs_sorted[start:end] = weights_sorted[start:end] * expert_output
    
    # 5. Scatter back to original order and sum
    outputs_expanded = torch.zeros_like(x_sorted)
    outputs_expanded[sorted_indices] = outputs_sorted
    
    # Reshape and sum over K
    outputs = outputs_expanded.view(B_S, K, D).sum(dim=1)
    
    return outputs
```

### 6.2 Grouped GEMM avec Triton

Pour vraiment optimiser, il faut un kernel custom qui traite tous les experts en un seul lancement:

```python
import triton
import triton.language as tl

@triton.jit
def grouped_gemm_kernel(
    # Pointers
    x_ptr, w1_ptr, w2_ptr, w3_ptr, out_ptr,
    # Shapes
    expert_offsets_ptr, token_counts_ptr,
    D, D_FF, n_experts,
    # Block sizes
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    """
    Grouped GEMM for MoE: process all experts in one kernel launch
    Each program instance handles one tile of one expert's computation
    """
    # Get expert and tile indices
    pid = tl.program_id(0)
    expert_id = tl.program_id(1)
    
    # Load expert-specific offsets
    offset = tl.load(expert_offsets_ptr + expert_id)
    count = tl.load(token_counts_ptr + expert_id)
    
    if pid * BLOCK_M >= count:
        return  # This tile has no work
    
    # Compute tile boundaries
    m_start = pid * BLOCK_M
    m_end = min(m_start + BLOCK_M, count)
    
    # Load input tile
    x_tile = tl.load(x_ptr + offset + m_start * D + tl.arange(0, BLOCK_M)[:, None] * D + tl.arange(0, BLOCK_K)[None, :])
    
    # Load expert weights (shared across tiles of same expert)
    w1 = tl.load(w1_ptr + expert_id * D * D_FF + ...)  # Gate weights
    w3 = tl.load(w3_ptr + expert_id * D * D_FF + ...)  # Up weights
    w2 = tl.load(w2_ptr + expert_id * D_FF * D + ...)  # Down weights
    
    # SwiGLU: gate * up, then down projection
    gate = tl.dot(x_tile, w1)
    gate = gate * tl.sigmoid(gate)  # SiLU
    up = tl.dot(x_tile, w3)
    hidden = gate * up
    output = tl.dot(hidden, w2)
    
    # Store result
    tl.store(out_ptr + offset + m_start * D + ..., output)
```

### 6.3 Memory Layout pour MoE

**Problème:** Les experts ont des poids séparés → cache misses.

**Solution: Interleaved layout**

```python
class InterleavedMoEWeights:
    """
    Instead of: [n_experts, d_model, d_ff]
    Use:        [d_model, n_experts, d_ff]
    
    This improves cache locality when processing batches
    where adjacent tokens go to different experts
    """
    def __init__(self, n_experts, d_model, d_ff):
        # Standard layout
        self.w1_standard = nn.Parameter(torch.randn(n_experts, d_model, d_ff))
        
        # Interleaved layout (transpose for inference)
        self.w1_interleaved = nn.Parameter(
            self.w1_standard.data.permute(1, 0, 2).contiguous()
        )
    
    def forward_interleaved(self, x, expert_ids):
        """
        x: [batch, d_model]
        expert_ids: [batch]
        
        Access pattern: pour chaque token, on accède à w1[expert_id]
        Avec interleaved: accès contigus si tokens adjacents ont experts différents
        """
        # Gather weights for each token's expert
        batch_size = x.shape[0]
        
        # [batch, d_model, d_ff] via advanced indexing
        selected_weights = self.w1_interleaved[:, expert_ids, :]  # [d_model, batch, d_ff]
        selected_weights = selected_weights.permute(1, 0, 2)  # [batch, d_model, d_ff]
        
        # Batched matmul
        return torch.bmm(x.unsqueeze(1), selected_weights).squeeze(1)
```

### 6.4 Expert Parallelism (EP)

Pour les modèles large-scale, distribuer les experts sur plusieurs GPUs:

```python
class ExpertParallelMoE(nn.Module):
    """
    Distribute experts across GPUs
    Each GPU holds a subset of experts
    Requires All-to-All communication
    """
    def __init__(self, config, rank, world_size):
        super().__init__()
        self.rank = rank
        self.world_size = world_size
        
        # Each GPU gets n_experts // world_size experts
        self.experts_per_gpu = config.n_experts // world_size
        self.local_expert_start = rank * self.experts_per_gpu
        
        # Local experts only
        self.local_experts = nn.ModuleList([
            SwiGLUExpert(config.d_model, config.expert_dim)
            for _ in range(self.experts_per_gpu)
        ])
        
        # Router is replicated
        self.router = AuxLossFreeRouter(config.d_model, config.n_experts, config.top_k)
        
    def forward(self, x):
        B, S, D = x.shape
        x_flat = x.view(-1, D)
        
        # 1. Route tokens (replicated computation)
        selected_experts, weights = self.router(x_flat)
        
        # 2. All-to-All: send tokens to the GPU that owns their expert
        tokens_to_send, send_counts = self._prepare_all_to_all(
            x_flat, selected_experts
        )
        
        received_tokens, recv_counts = torch.distributed.all_to_all(
            tokens_to_send, send_counts
        )
        
        # 3. Local expert computation
        local_outputs = self._compute_local_experts(
            received_tokens, recv_counts
        )
        
        # 4. All-to-All: send results back
        results = torch.distributed.all_to_all(local_outputs, recv_counts)
        
        # 5. Combine with weights
        output = self._combine_results(results, weights)
        
        return output.view(B, S, D)
```

**Node-limited routing (Kimi K2):**
```python
# Limite le nombre de nodes (pas GPUs) pour réduire communication
MAX_NODES_PER_TOKEN = 4

def node_limited_routing(scores, top_k, nodes_per_gpu, max_nodes=4):
    """
    Contrainte: un token peut aller à max 4 nodes différents
    Réduit drastiquement le coût All-to-All
    """
    n_experts = scores.shape[-1]
    experts_per_node = n_experts // (n_experts // nodes_per_gpu)
    
    # D'abord, sélectionner les top nodes
    node_scores = scores.view(*scores.shape[:-1], -1, experts_per_node).max(dim=-1).values
    top_nodes = torch.topk(node_scores, max_nodes, dim=-1).indices
    
    # Ensuite, sélectionner les top-k experts PARMI ces nodes
    mask = torch.zeros_like(scores, dtype=torch.bool)
    for node_idx in range(max_nodes):
        node = top_nodes[..., node_idx]
        start = node * experts_per_node
        for e in range(experts_per_node):
            mask[..., start + e] = True
    
    masked_scores = scores.masked_fill(~mask, float('-inf'))
    return torch.topk(masked_scores, top_k, dim=-1)
```

### 6.5 KV Cache avec MoE + MLA

**MLA (Multi-head Latent Attention) compression:**

```python
class MLAWithMoE(nn.Module):
    """
    MLA compresses KV cache via low-rank projection
    Combined with MoE in the same block
    """
    def __init__(self, d_model, n_heads, d_latent, n_kv_heads):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.d_latent = d_latent  # Compressed dimension
        self.n_kv_heads = n_kv_heads
        
        # Q projection (full dimension)
        self.W_q = nn.Linear(d_model, d_model)
        
        # KV compression: d_model → d_latent
        self.W_kv_down = nn.Linear(d_model, d_latent)
        
        # KV decompression: d_latent → 2 * n_kv_heads * d_head
        self.W_k_up = nn.Linear(d_latent, n_kv_heads * self.d_head)
        self.W_v_up = nn.Linear(d_latent, n_kv_heads * self.d_head)
        
        self.W_o = nn.Linear(d_model, d_model)
        
    def forward(self, x, cache=None):
        B, S, D = x.shape
        
        # Query: full projection
        q = self.W_q(x).view(B, S, self.n_heads, self.d_head)
        
        # KV: compress to latent space
        kv_latent = self.W_kv_down(x)  # [B, S, d_latent]
        
        # Cache the COMPRESSED representation
        if cache is not None:
            kv_latent = torch.cat([cache, kv_latent], dim=1)
        
        # Decompress for attention
        k = self.W_k_up(kv_latent).view(B, -1, self.n_kv_heads, self.d_head)
        v = self.W_v_up(kv_latent).view(B, -1, self.n_kv_heads, self.d_head)
        
        # GQA: repeat KV heads
        k = k.repeat_interleave(self.n_heads // self.n_kv_heads, dim=2)
        v = v.repeat_interleave(self.n_heads // self.n_kv_heads, dim=2)
        
        # Attention
        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        
        return self.W_o(attn_out.view(B, S, D)), kv_latent
```

**Gains mémoire:**
```
Standard MHA KV cache:  2 * n_layers * seq_len * n_heads * d_head
MLA KV cache:           n_layers * seq_len * d_latent

Exemple (d_model=4096, n_heads=32, d_latent=512):
- MHA: 2 * seq_len * 4096 = 8192 * seq_len bytes (bf16)
- MLA: seq_len * 512 = 512 * seq_len bytes (bf16)
→ Compression 16x!
```

---

## 7. Compression et Inference

### 7.1 Expert Pruning vs Merging

**Observation 2025 (REAP paper):** Le pruning outperform le merging sur les tâches génératives.

**Pourquoi?**
- Merging perturbe la coordination router-experts
- Les poids merged ne correspondent plus aux patterns appris par le router
- Le pruning préserve les experts intacts, juste moins nombreux

```python
def reap_expert_scores(model, calibration_data, n_samples=1024):
    """
    REAP: Router-weighted Expert Activation Pruning
    Score = router_weight * activation_norm (par expert)
    """
    expert_scores = defaultdict(list)
    
    model.eval()
    with torch.no_grad():
        for batch in calibration_data[:n_samples]:
            x = batch['input_ids']
            
            for layer in model.moe_layers:
                # Get router decisions
                selected, weights = layer.router(x)
                
                # Compute activation norms for each expert
                for expert_idx in range(layer.n_experts):
                    mask = (selected == expert_idx)
                    if mask.any():
                        expert_input = x[mask.any(dim=-1)]
                        
                        # Activation norm (L2)
                        with torch.enable_grad():
                            output = layer.experts[expert_idx](expert_input)
                            activation_norm = output.norm(dim=-1).mean()
                        
                        # REAP score
                        router_weight = weights[mask].mean()
                        score = router_weight * activation_norm
                        
                        expert_scores[expert_idx].append(score.item())
    
    # Average scores
    return {k: sum(v)/len(v) for k, v in expert_scores.items()}

def prune_experts(model, keep_ratio=0.5):
    """Prune experts with lowest REAP scores"""
    scores = reap_expert_scores(model, calibration_data)
    
    n_keep = int(len(scores) * keep_ratio)
    experts_to_keep = sorted(scores.keys(), key=lambda k: scores[k], reverse=True)[:n_keep]
    
    for layer in model.moe_layers:
        # Update router to only route to kept experts
        layer.update_expert_mask(experts_to_keep)
        
        # Optionally: physically remove pruned experts
        layer.experts = nn.ModuleList([
            layer.experts[i] for i in experts_to_keep
        ])
    
    return model
```

### 7.2 Quantization MoE-aware

**FP8 Training (DeepSeek-V3, Kimi K2):**

```python
class FP8Expert(nn.Module):
    """Expert avec quantization FP8 pour training et inference"""
    def __init__(self, d_model, d_ff):
        super().__init__()
        # Poids stockés en FP8
        self.w1 = nn.Parameter(torch.randn(d_model, d_ff).to(torch.float8_e4m3fn))
        self.w2 = nn.Parameter(torch.randn(d_ff, d_model).to(torch.float8_e4m3fn))
        self.w3 = nn.Parameter(torch.randn(d_model, d_ff).to(torch.float8_e4m3fn))
        
        # Scales pour dequantization
        self.register_buffer('w1_scale', torch.ones(1))
        self.register_buffer('w2_scale', torch.ones(1))
        self.register_buffer('w3_scale', torch.ones(1))
        
    def forward(self, x):
        # Dequantize on-the-fly
        w1 = self.w1.to(x.dtype) * self.w1_scale
        w2 = self.w2.to(x.dtype) * self.w2_scale
        w3 = self.w3.to(x.dtype) * self.w3_scale
        
        gate = F.silu(x @ w1)
        up = x @ w3
        return (gate * up) @ w2
```

**INT4 avec QAT (Kimi K2 Thinking):**

```python
def quantization_aware_training(model, train_data, epochs=1):
    """
    QAT pour INT4: entraîne le modèle à être robuste à la quantization
    """
    from torch.quantization import prepare_qat, convert
    
    # Configure QAT
    model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
    
    # Prepare model
    model_prepared = prepare_qat(model)
    
    # Train with fake quantization
    optimizer = torch.optim.AdamW(model_prepared.parameters(), lr=1e-5)
    
    for epoch in range(epochs):
        for batch in train_data:
            loss = model_prepared(batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
    
    # Convert to quantized model
    model_quantized = convert(model_prepared)
    
    return model_quantized
```

### 7.3 Expert Offloading

Pour inference sur GPU limité:

```python
class OffloadedMoE(nn.Module):
    """
    Keep only active experts on GPU
    Offload others to CPU
    Prefetch based on router predictions
    """
    def __init__(self, config, max_gpu_experts=8):
        super().__init__()
        self.max_gpu_experts = max_gpu_experts
        
        # All experts start on CPU
        self.experts_cpu = nn.ModuleList([
            SwiGLUExpert(config.d_model, config.expert_dim).cpu()
            for _ in range(config.n_experts)
        ])
        
        # GPU cache for active experts
        self.gpu_cache = {}
        self.cache_order = []
        
        # Router stays on GPU
        self.router = AuxLossFreeRouter(config.d_model, config.n_experts, config.top_k).cuda()
        
    def forward(self, x):
        # Route to determine needed experts
        selected, weights = self.router(x)
        needed_experts = selected.unique().tolist()
        
        # Ensure needed experts are on GPU
        self._ensure_on_gpu(needed_experts)
        
        # Compute with GPU experts
        output = self._compute_with_cache(x, selected, weights)
        
        return output
    
    def _ensure_on_gpu(self, expert_ids):
        for eid in expert_ids:
            if eid not in self.gpu_cache:
                # Evict if cache full
                while len(self.gpu_cache) >= self.max_gpu_experts:
                    evict_id = self.cache_order.pop(0)
                    self.gpu_cache[evict_id].cpu()
                    del self.gpu_cache[evict_id]
                
                # Load to GPU
                self.gpu_cache[eid] = self.experts_cpu[eid].cuda()
                self.cache_order.append(eid)
            else:
                # Move to end (LRU)
                self.cache_order.remove(eid)
                self.cache_order.append(eid)
    
    def prefetch_next_layer(self, next_layer_router, x):
        """Prefetch experts pour le layer suivant pendant le compute actuel"""
        with torch.no_grad():
            next_selected, _ = next_layer_router(x)
            next_needed = next_selected.unique().tolist()
        
        # Async copy to GPU
        for eid in next_needed:
            if eid not in self.gpu_cache:
                # Non-blocking copy
                self.experts_cpu[eid].cuda(non_blocking=True)
```

---

## 8. Implémentation de Référence

### 8.1 Configuration complète

```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class SwamlaMoEConfig:
    """Configuration pour Small LLM avec MoE + SWA + MLA"""
    
    # Model dimensions
    d_model: int = 4096
    n_layers: int = 32
    vocab_size: int = 128256
    
    # Attention (Sliding Window + MLA)
    n_heads: int = 32
    n_kv_heads: int = 4  # GQA ratio 8:1
    d_latent: int = 512  # MLA compression
    window_size: int = 4096
    max_seq_len: int = 131072
    
    # MoE configuration
    n_experts: int = 64
    n_shared_experts: int = 2
    n_activated: int = 6  # + shared = 8 total
    expert_dim: int = 1536
    
    # Routing
    routing_type: str = "sigmoid"  # "sigmoid", "softmax", "relu"
    use_bias_balancing: bool = True
    bias_update_gamma: float = 0.001
    aux_loss_coef: float = 0.0001
    router_z_loss_coef: float = 0.001
    
    # Stability (Kimi K2 style)
    use_qk_clip: bool = True
    qk_clip_threshold: float = 10.0
    
    # Architecture choices
    n_dense_init_layers: int = 1
    use_mone: bool = False  # Internal neuron sparsity
    mone_ratio: float = 0.5
    
    # Training
    tie_word_embeddings: bool = False
    use_flash_attention: bool = True
    
    # Compute dtype
    dtype: str = "bfloat16"
    
    def __post_init__(self):
        self.d_head = self.d_model // self.n_heads
        self.total_activated = self.n_activated + self.n_shared_experts
        
        # Validate
        assert self.d_model % self.n_heads == 0
        assert self.n_heads % self.n_kv_heads == 0
```

### 8.2 Modules de base

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x):
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * rms * self.weight


class SwiGLUExpert(nn.Module):
    """SwiGLU FFN expert avec option MoNE"""
    def __init__(self, d_model: int, d_ff: int, use_mone: bool = False, mone_ratio: float = 0.5):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)  # gate
        self.w2 = nn.Linear(d_ff, d_model, bias=False)  # down
        self.w3 = nn.Linear(d_model, d_ff, bias=False)  # up
        
        self.use_mone = use_mone
        self.mone_k = int(d_ff * mone_ratio) if use_mone else d_ff
    
    def forward(self, x):
        gate = self.w1(x)
        
        if self.use_mone and not self.training:
            # MoNE: keep only top-k neurons during inference
            _, topk_indices = torch.topk(gate.abs(), self.mone_k, dim=-1)
            mask = torch.zeros_like(gate).scatter_(-1, topk_indices, 1.0)
            gate = gate * mask
            up = self.w3(x) * mask
        else:
            up = self.w3(x)
        
        return self.w2(F.silu(gate) * up)


class AuxLossFreeRouter(nn.Module):
    """Router avec auxiliary-loss-free balancing (DeepSeek-V3 style)"""
    def __init__(self, d_model: int, n_experts: int, top_k: int, 
                 gamma: float = 0.001, z_loss_coef: float = 0.001):
        super().__init__()
        self.gate = nn.Linear(d_model, n_experts, bias=False)
        self.expert_bias = nn.Parameter(torch.zeros(n_experts), requires_grad=False)
        self.top_k = top_k
        self.gamma = gamma
        self.z_loss_coef = z_loss_coef
        self.n_experts = n_experts
        
        # For logging
        self.register_buffer('load_history', torch.zeros(n_experts))
        self.register_buffer('update_count', torch.tensor(0))
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            selected_experts: [N, top_k] indices
            gating_weights: [N, top_k] normalized weights
            aux_loss: scalar auxiliary loss
        """
        original_shape = x.shape
        x_flat = x.view(-1, x.shape[-1])
        N = x_flat.shape[0]
        
        # Compute logits
        logits = self.gate(x_flat)  # [N, E]
        
        # Z-loss for numerical stability
        z_loss = self.z_loss_coef * torch.logsumexp(logits, dim=-1).pow(2).mean()
        
        # Sigmoid affinity (not softmax!)
        affinity = torch.sigmoid(logits)  # [N, E]
        
        # Routing decision with bias
        routing_scores = affinity + self.expert_bias
        _, selected_experts = torch.topk(routing_scores, self.top_k, dim=-1)
        
        # Gating weights = normalized affinity WITHOUT bias
        selected_affinity = torch.gather(affinity, 1, selected_experts)
        gating_weights = selected_affinity / (selected_affinity.sum(-1, keepdim=True) + 1e-9)
        
        # Update bias during training
        if self.training:
            self._update_bias(selected_experts, N)
        
        return selected_experts, gating_weights, z_loss
    
    def _update_bias(self, selected_experts: torch.Tensor, N: int):
        with torch.no_grad():
            # Count load per expert
            load = torch.zeros(self.n_experts, device=selected_experts.device)
            for k in range(self.top_k):
                load.scatter_add_(
                    0, selected_experts[:, k],
                    torch.ones(N, device=selected_experts.device)
                )
            
            expected_load = N * self.top_k / self.n_experts
            
            # Update bias
            self.expert_bias.data -= self.gamma * (load - expected_load).sign()
            
            # Update history for monitoring
            self.load_history = 0.99 * self.load_history + 0.01 * load
            self.update_count += 1
    
    def get_load_balance_stats(self) -> dict:
        """Pour monitoring"""
        load = self.load_history / (self.update_count + 1e-9)
        return {
            'load_mean': load.mean().item(),
            'load_std': load.std().item(),
            'load_min': load.min().item(),
            'load_max': load.max().item(),
            'load_ratio': (load.max() / (load.min() + 1e-9)).item(),
        }


class MoELayer(nn.Module):
    """MoE Layer complet avec shared experts et routing efficace"""
    def __init__(self, config: SwamlaMoEConfig):
        super().__init__()
        self.config = config
        
        # Shared experts
        self.shared_experts = nn.ModuleList([
            SwiGLUExpert(config.d_model, config.expert_dim)
            for _ in range(config.n_shared_experts)
        ])
        
        # Routed experts
        self.experts = nn.ModuleList([
            SwiGLUExpert(
                config.d_model, config.expert_dim,
                use_mone=config.use_mone, mone_ratio=config.mone_ratio
            )
            for _ in range(config.n_experts)
        ])
        
        # Router
        self.router = AuxLossFreeRouter(
            config.d_model, config.n_experts, config.n_activated,
            gamma=config.bias_update_gamma,
            z_loss_coef=config.router_z_loss_coef
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            output: [B, S, D]
            aux_loss: scalar
        """
        B, S, D = x.shape
        x_flat = x.view(-1, D)
        
        # Shared experts (always computed)
        shared_output = sum(expert(x_flat) for expert in self.shared_experts)
        
        # Routing
        selected_experts, weights, aux_loss = self.router(x_flat)
        
        # Routed experts computation
        routed_output = self._compute_experts_grouped(x_flat, selected_experts, weights)
        
        output = shared_output + routed_output
        return output.view(B, S, D), aux_loss
    
    def _compute_experts_grouped(self, x: torch.Tensor, 
                                  selected: torch.Tensor, 
                                  weights: torch.Tensor) -> torch.Tensor:
        """Compute experts with token grouping for efficiency"""
        N, K = selected.shape
        D = x.shape[-1]
        output = torch.zeros(N, D, device=x.device, dtype=x.dtype)
        
        # Group tokens by expert
        for expert_idx in range(self.config.n_experts):
            # Find tokens routed to this expert
            mask = (selected == expert_idx)  # [N, K]
            
            if not mask.any():
                continue
            
            # Get indices and weights for tokens going to this expert
            token_indices = mask.any(dim=-1).nonzero(as_tuple=True)[0]
            
            if len(token_indices) == 0:
                continue
            
            # Get expert weights for these tokens
            expert_weights = torch.zeros(len(token_indices), device=x.device)
            for k in range(K):
                k_mask = mask[token_indices, k]
                expert_weights[k_mask] += weights[token_indices[k_mask], k]
            
            # Compute expert output
            expert_input = x[token_indices]
            expert_output = self.experts[expert_idx](expert_input)
            
            # Weighted contribution
            output[token_indices] += expert_weights.unsqueeze(-1) * expert_output
        
        return output
```

### 8.3 Attention avec MLA et Sliding Window

```python
class SlidingWindowMLA(nn.Module):
    """Multi-head Latent Attention with Sliding Window"""
    def __init__(self, config: SwamlaMoEConfig):
        super().__init__()
        self.config = config
        self.d_head = config.d_head
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.d_latent = config.d_latent
        self.window_size = config.window_size
        
        # Q projection (full dimension)
        self.W_q = nn.Linear(config.d_model, config.d_model, bias=False)
        
        # KV compression
        self.W_kv_down = nn.Linear(config.d_model, config.d_latent, bias=False)
        self.W_k_up = nn.Linear(config.d_latent, config.n_kv_heads * self.d_head, bias=False)
        self.W_v_up = nn.Linear(config.d_latent, config.n_kv_heads * self.d_head, bias=False)
        
        # Output
        self.W_o = nn.Linear(config.d_model, config.d_model, bias=False)
        
        # RoPE
        self.rotary_emb = RotaryEmbedding(self.d_head, config.max_seq_len)
        
        # QK clipping for stability
        self.use_qk_clip = config.use_qk_clip
        self.qk_clip_threshold = config.qk_clip_threshold
    
    def forward(self, x: torch.Tensor, 
                cache: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            output: [B, S, D]
            new_cache: [B, S, d_latent] (compressed KV for caching)
        """
        B, S, D = x.shape
        
        # Query projection
        q = self.W_q(x).view(B, S, self.n_heads, self.d_head)
        
        # KV compression
        kv_latent = self.W_kv_down(x)  # [B, S, d_latent]
        
        # Handle cache
        if cache is not None:
            kv_latent_full = torch.cat([cache, kv_latent], dim=1)
        else:
            kv_latent_full = kv_latent
        
        # Decompress KV
        total_len = kv_latent_full.shape[1]
        k = self.W_k_up(kv_latent_full).view(B, total_len, self.n_kv_heads, self.d_head)
        v = self.W_v_up(kv_latent_full).view(B, total_len, self.n_kv_heads, self.d_head)
        
        # Apply RoPE
        if position_ids is None:
            position_ids = torch.arange(total_len, device=x.device).unsqueeze(0)
        cos, sin = self.rotary_emb(position_ids)
        q = apply_rotary_pos_emb(q, cos[:, -S:], sin[:, -S:])
        k = apply_rotary_pos_emb(k, cos, sin)
        
        # QK clipping for stability (Kimi K2 style)
        if self.use_qk_clip:
            q = self._qk_clip(q)
            k = self._qk_clip(k)
        
        # GQA: expand KV heads
        n_rep = self.n_heads // self.n_kv_heads
        k = k.repeat_interleave(n_rep, dim=2)
        v = v.repeat_interleave(n_rep, dim=2)
        
        # Sliding window attention mask
        attn_mask = self._make_sliding_window_mask(S, total_len)
        
        # Compute attention
        q = q.transpose(1, 2)  # [B, H, S, D]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        if self.config.use_flash_attention and x.is_cuda:
            # Use Flash Attention if available
            attn_output = F.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask, is_causal=(cache is None)
            )
        else:
            attn_output = self._manual_attention(q, k, v, attn_mask)
        
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, S, D)
        output = self.W_o(attn_output)
        
        return output, kv_latent
    
    def _qk_clip(self, x: torch.Tensor) -> torch.Tensor:
        """QK clipping à la Kimi K2 pour stabilité"""
        # Compute max attention score per head
        # Scale down if exceeds threshold
        norm = x.norm(dim=-1, keepdim=True)
        scale = torch.clamp(self.qk_clip_threshold / (norm + 1e-6), max=1.0)
        return x * scale
    
    def _make_sliding_window_mask(self, query_len: int, kv_len: int) -> torch.Tensor:
        """Create sliding window causal mask"""
        # Positions des queries et keys
        q_pos = torch.arange(kv_len - query_len, kv_len)
        k_pos = torch.arange(kv_len)
        
        # Causal mask
        causal = q_pos.unsqueeze(1) >= k_pos.unsqueeze(0)
        
        # Sliding window mask
        distance = q_pos.unsqueeze(1) - k_pos.unsqueeze(0)
        window = distance <= self.window_size
        
        mask = causal & window
        return mask.float().masked_fill(~mask, float('-inf'))
    
    def _manual_attention(self, q, k, v, mask):
        """Fallback attention sans Flash"""
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.d_head ** 0.5)
        if mask is not None:
            scores = scores + mask
        attn_probs = F.softmax(scores, dim=-1)
        return torch.matmul(attn_probs, v)


class RotaryEmbedding(nn.Module):
    """RoPE embeddings"""
    def __init__(self, dim: int, max_seq_len: int, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        self.max_seq_len = max_seq_len
        
    def forward(self, position_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        seq_len = position_ids.shape[-1]
        freqs = torch.outer(position_ids.float().squeeze(), self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos(), emb.sin()


def apply_rotary_pos_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply rotary position embeddings"""
    x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
    return torch.cat([
        x1 * cos - x2 * sin,
        x2 * cos + x1 * sin
    ], dim=-1)
```

### 8.4 Transformer Block complet

```python
class SwamlaTransformerBlock(nn.Module):
    """Transformer block avec MoE + MLA + Sliding Window"""
    def __init__(self, config: SwamlaMoEConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        
        # Pre-norm
        self.attn_norm = RMSNorm(config.d_model)
        self.ffn_norm = RMSNorm(config.d_model)
        
        # Attention
        self.attn = SlidingWindowMLA(config)
        
        # FFN: dense pour les premières couches, MoE ensuite
        if layer_idx < config.n_dense_init_layers:
            self.ffn = SwiGLUExpert(config.d_model, config.d_model * 4)
            self.is_moe = False
        else:
            self.ffn = MoELayer(config)
            self.is_moe = True
    
    def forward(self, x: torch.Tensor, 
                cache: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        """
        Returns:
            output: [B, S, D]
            new_cache: KV cache for this layer
            aux_loss: auxiliary loss (0 if dense FFN)
        """
        # Attention with residual
        attn_out, new_cache = self.attn(self.attn_norm(x), cache, position_ids)
        x = x + attn_out
        
        # FFN with residual
        if self.is_moe:
            ffn_out, aux_loss = self.ffn(self.ffn_norm(x))
        else:
            ffn_out = self.ffn(self.ffn_norm(x))
            aux_loss = torch.tensor(0.0, device=x.device)
        
        x = x + ffn_out
        
        return x, new_cache, aux_loss


class SwamlaModel(nn.Module):
    """Modèle complet"""
    def __init__(self, config: SwamlaMoEConfig):
        super().__init__()
        self.config = config
        
        # Embeddings
        self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model)
        
        # Transformer blocks
        self.layers = nn.ModuleList([
            SwamlaTransformerBlock(config, i) for i in range(config.n_layers)
        ])
        
        # Final norm
        self.norm = RMSNorm(config.d_model)
        
        # LM head
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        # Tie embeddings optionnel
        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, input_ids: torch.Tensor,
                cache: Optional[list] = None,
                position_ids: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, list, torch.Tensor]:
        """
        Returns:
            logits: [B, S, vocab_size]
            new_cache: list of KV caches per layer
            total_aux_loss: sum of auxiliary losses
        """
        B, S = input_ids.shape
        
        # Embeddings
        x = self.embed_tokens(input_ids)
        
        # Prepare cache
        if cache is None:
            cache = [None] * len(self.layers)
        new_cache = []
        
        # Position IDs
        if position_ids is None:
            if cache[0] is not None:
                past_len = cache[0].shape[1]
            else:
                past_len = 0
            position_ids = torch.arange(past_len, past_len + S, device=x.device).unsqueeze(0)
        
        # Forward through layers
        total_aux_loss = torch.tensor(0.0, device=x.device)
        
        for i, layer in enumerate(self.layers):
            x, layer_cache, aux_loss = layer(x, cache[i], position_ids)
            new_cache.append(layer_cache)
            total_aux_loss = total_aux_loss + aux_loss
        
        # Final norm
        x = self.norm(x)
        
        # LM head
        logits = self.lm_head(x)
        
        return logits, new_cache, total_aux_loss
    
    def compute_loss(self, input_ids: torch.Tensor, 
                     labels: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """Compute total loss with auxiliary losses"""
        logits, _, aux_loss = self.forward(input_ids)
        
        # Shift for next-token prediction
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        # Cross-entropy loss
        ce_loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100
        )
        
        # Total loss
        total_loss = ce_loss + self.config.aux_loss_coef * aux_loss
        
        return total_loss, {
            'ce_loss': ce_loss.item(),
            'aux_loss': aux_loss.item(),
            'total_loss': total_loss.item()
        }
```

---

## 9. Debugging et Diagnostic

### 9.1 Monitoring du load balancing

```python
def monitor_load_balance(model: SwamlaModel) -> dict:
    """Collecter les stats de load balancing de tous les MoE layers"""
    stats = {}
    
    for i, layer in enumerate(model.layers):
        if hasattr(layer.ffn, 'router'):
            router = layer.ffn.router
            layer_stats = router.get_load_balance_stats()
            stats[f'layer_{i}'] = layer_stats
    
    # Aggregate stats
    if stats:
        all_ratios = [s['load_ratio'] for s in stats.values()]
        stats['aggregate'] = {
            'mean_load_ratio': sum(all_ratios) / len(all_ratios),
            'max_load_ratio': max(all_ratios),
            'min_load_ratio': min(all_ratios),
        }
    
    return stats

def plot_expert_usage(model: SwamlaModel, save_path: str = None):
    """Visualiser l'utilisation des experts par layer"""
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(
        len([l for l in model.layers if hasattr(l.ffn, 'router')]),
        1, figsize=(12, 3 * model.config.n_layers)
    )
    
    for i, layer in enumerate(model.layers):
        if hasattr(layer.ffn, 'router'):
            load = layer.ffn.router.load_history.cpu().numpy()
            load = load / (load.sum() + 1e-9)
            
            ax = axes[i] if hasattr(axes, '__len__') else axes
            ax.bar(range(len(load)), load)
            ax.set_title(f'Layer {i} Expert Usage')
            ax.set_xlabel('Expert ID')
            ax.set_ylabel('Usage Fraction')
            ax.axhline(y=1/len(load), color='r', linestyle='--', label='Uniform')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()
```

### 9.2 Détection du routing collapse

```python
def detect_routing_collapse(model: SwamlaModel, threshold: float = 0.1) -> list:
    """
    Détecte les layers avec routing collapse
    threshold: fraction minimale d'utilisation attendue
    """
    collapsed_layers = []
    
    for i, layer in enumerate(model.layers):
        if not hasattr(layer.ffn, 'router'):
            continue
        
        load = layer.ffn.router.load_history
        load_frac = load / (load.sum() + 1e-9)
        
        # Compter les experts sous-utilisés
        underused = (load_frac < threshold / model.config.n_experts).sum().item()
        
        if underused > model.config.n_experts * 0.5:  # Plus de 50% sous-utilisés
            collapsed_layers.append({
                'layer': i,
                'underused_experts': underused,
                'min_usage': load_frac.min().item(),
                'max_usage': load_frac.max().item(),
                'usage_ratio': (load_frac.max() / (load_frac.min() + 1e-9)).item()
            })
    
    return collapsed_layers

def fix_routing_collapse(model: SwamlaModel, reset_bias: bool = True):
    """Tente de corriger un routing collapse détecté"""
    for layer in model.layers:
        if hasattr(layer.ffn, 'router'):
            router = layer.ffn.router
            
            if reset_bias:
                # Reset les bias à zéro
                router.expert_bias.data.zero_()
            
            # Augmenter temporairement gamma pour forcer le rebalancing
            router.gamma = router.gamma * 10
```

### 9.3 Profiling des experts

```python
def profile_expert_computation(model: SwamlaModel, 
                               sample_input: torch.Tensor) -> dict:
    """Profile le temps de calcul par expert"""
    import time
    
    profile_results = {}
    
    for layer_idx, layer in enumerate(model.layers):
        if not hasattr(layer.ffn, 'experts'):
            continue
        
        layer_results = {}
        x = sample_input
        
        # Forward jusqu'à ce layer
        with torch.no_grad():
            for prev_layer in model.layers[:layer_idx]:
                x, _, _ = prev_layer(x)
            
            x_normed = layer.ffn_norm(x)
            
            # Profile chaque expert
            for expert_idx, expert in enumerate(layer.ffn.experts):
                torch.cuda.synchronize()
                start = time.time()
                
                for _ in range(100):  # Average over runs
                    _ = expert(x_normed)
                
                torch.cuda.synchronize()
                elapsed = (time.time() - start) / 100
                
                layer_results[f'expert_{expert_idx}'] = elapsed * 1000  # ms
        
        profile_results[f'layer_{layer_idx}'] = layer_results
    
    return profile_results
```

### 9.4 Validation de la configuration

```python
def validate_config(config: SwamlaMoEConfig) -> list:
    """Valide la configuration et retourne les warnings"""
    warnings = []
    
    # Check expert ratio
    active_ratio = config.total_activated / config.n_experts
    if active_ratio > 0.25:
        warnings.append(
            f"High activation ratio ({active_ratio:.1%}). "
            f"Consider more experts for better sparsity."
        )
    
    # Check shared experts
    shared_ratio = config.n_shared_experts / config.total_activated
    if shared_ratio > 0.5:
        warnings.append(
            f"High shared expert ratio ({shared_ratio:.1%}). "
            f"May reduce specialization benefits."
        )
    elif shared_ratio < 0.1:
        warnings.append(
            f"Low shared expert ratio ({shared_ratio:.1%}). "
            f"Consider at least 1 shared expert for stability."
        )
    
    # Check MLA compression
    compression_ratio = config.d_model / config.d_latent
    if compression_ratio > 16:
        warnings.append(
            f"Very high MLA compression ({compression_ratio}x). "
            f"May lose important information."
        )
    
    # Check expert dimension
    if config.expert_dim < config.d_model:
        warnings.append(
            f"Expert dim ({config.expert_dim}) < model dim ({config.d_model}). "
            f"Experts may be bottlenecked."
        )
    
    # Check dense init layers
    if config.n_dense_init_layers > 3:
        warnings.append(
            f"Many dense init layers ({config.n_dense_init_layers}). "
            f"Consider reducing if using MuonClip optimizer."
        )
    
    return warnings
```

---

## 10. Configuration Finale pour Small LLMs

### 10.1 Configurations recommandées par taille

```python
# ~4B total params, ~1B active
CONFIG_4B = SwamlaMoEConfig(
    d_model=2048,
    n_layers=24,
    n_heads=16,
    n_kv_heads=4,
    d_latent=256,
    n_experts=32,
    n_shared_experts=1,
    n_activated=4,  # +1 shared = 5 total
    expert_dim=1024,
    window_size=2048,
)

# ~8B total params, ~2B active
CONFIG_8B = SwamlaMoEConfig(
    d_model=4096,
    n_layers=32,
    n_heads=32,
    n_kv_heads=4,
    d_latent=512,
    n_experts=64,
    n_shared_experts=2,
    n_activated=6,  # +2 shared = 8 total
    expert_dim=1536,
    window_size=4096,
)

# ~16B total params, ~3B active
CONFIG_16B = SwamlaMoEConfig(
    d_model=4096,
    n_layers=40,
    n_heads=32,
    n_kv_heads=8,
    d_latent=512,
    n_experts=128,
    n_shared_experts=2,
    n_activated=6,
    expert_dim=2048,
    window_size=4096,
)
```

### 10.2 Checklist d'implémentation

```markdown
## Checklist MoE 2025

### Architecture
- [ ] Sigmoid routing (pas softmax) avec bias balancing
- [ ] 1-2 shared experts (au moins 1 obligatoire)
- [ ] 1 couche dense initiale (pas 3)
- [ ] MLA avec compression 8-16x
- [ ] Sliding window attention
- [ ] QK-Clip pour stabilité

### Routing
- [ ] Auxiliary-loss-free balancing (gamma=0.001)
- [ ] Router Z-loss (coef=0.001)
- [ ] Minimal aux loss (coef=0.0001)
- [ ] Bias annealing (gamma→0 fin de training)

### Optimisation
- [ ] Token grouping pour experts
- [ ] Flash Attention
- [ ] FP8/BF16 mixed precision
- [ ] Gradient checkpointing si nécessaire

### Monitoring
- [ ] Load balance stats par layer
- [ ] Détection routing collapse
- [ ] Expert usage histograms

### Optionnel (gains supplémentaires)
- [ ] MoNE pour 2x inference speedup
- [ ] NSA/MoBA pour long context
- [ ] Expert pruning pour deployment
```

### 10.3 Ordre de priorité des optimisations

1. **Critique (implémenter d'abord):**
   - Sigmoid routing + bias balancing
   - Au moins 1 shared expert
   - Router Z-loss

2. **Important (gains significatifs):**
   - MLA compression
   - QK-Clip
   - Token grouping efficace

3. **Optimisation (gains marginaux):**
   - MoNE internal sparsity
   - Expert offloading
   - INT4 quantization

4. **Long context (si nécessaire):**
   - NSA ou MoBA
   - Sliding window

---

## Références

### Papers 2025 essentiels
1. Kimi K2 Technical Report (arXiv:2507.20534)
2. MoNE: Mixture of Neuron Experts (arXiv:2510.05781)
3. Native Sparse Attention (arXiv:2502.11089)
4. MoBA: Mixture of Block Attention (arXiv:2502.13189)
5. REAP: Expert Pruning (arXiv:2510.13999)
6. R3: Rollout Routing Replay (arXiv:2510.11370)

### Surveys
1. Comprehensive MoE Survey (arXiv:2503.07137)
2. MoE in LLMs (arXiv:2507.11181)
3. MoE for LLMs Survey (arXiv:2407.06204)

### Implémentations de référence
- DeepSeek-V3: github.com/deepseek-ai/DeepSeek-V3
- Kimi K2: github.com/MoonshotAI/Kimi-K2
- MoBA: github.com/MoonshotAI/MoBA