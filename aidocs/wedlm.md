# WeDLM : Documentation Technique Complète

## Reconciling Diffusion Language Models with Standard Causal Attention for Fast Inference

**Version** : 1.0  
**Date** : Décembre 2024  
**Référence** : [Paper WeDLM](https://wedlm.github.io) | [GitHub](https://github.com/tencent/WeDLM)

---

## Table des Matières

1. [Introduction et Motivation](#1-introduction-et-motivation)
2. [Concepts Fondamentaux](#2-concepts-fondamentaux)
3. [Architecture WeDLM](#3-architecture-wedlm)
4. [Contribution 1 : Topological Reordering](#4-contribution-1--topological-reordering)
5. [Contribution 2 : Dual-Stream Masking](#5-contribution-2--dual-stream-masking)
6. [Contribution 3 : Streaming Parallel Decoding](#6-contribution-3--streaming-parallel-decoding)
7. [Implémentation Détaillée](#7-implémentation-détaillée)
8. [Guide d'Intégration](#8-guide-dintégration)
9. [Optimisations et Performance](#9-optimisations-et-performance)
10. [API Reference](#10-api-reference)
11. [Troubleshooting](#11-troubleshooting)

---

## 1. Introduction et Motivation

### 1.1 Problème Adressé

Les **Large Language Models (LLMs)** autorégressifs (AR) génèrent du texte token par token, ce qui :
- Sous-utilise les accélérateurs modernes (GPU/TPU)
- Crée un goulot d'étranglement mémoire (memory-bound)
- Limite le débit de génération

Les **Diffusion Language Models (DLLMs)** promettent du parallélisme en prédisant plusieurs tokens masqués simultanément. **Cependant**, en pratique, ils ne surpassent pas les moteurs AR optimisés comme vLLM.

### 1.2 Pourquoi les DLLMs Existants Échouent

| Problème | Cause | Impact |
|----------|-------|--------|
| **Attention bidirectionnelle** | Chaque token dépend des tokens futurs | KV cache invalidé à chaque modification |
| **Résolution out-of-order** | Tokens résolus dans un ordre non-séquentiel | Préfixe non-contigu, recomputation nécessaire |
| **Block synchronization** | Attente de la finalisation du bloc entier | Pipeline bubbles, sous-utilisation GPU |

### 1.3 Solution WeDLM

WeDLM réconcilie le décodage parallèle avec l'attention causale standard via :

```
┌─────────────────────────────────────────────────────────────────┐
│                         WeDLM Framework                          │
├─────────────────────────────────────────────────────────────────┤
│  TRAINING                          │  INFERENCE                  │
│  ├─ Topological Reordering         │  ├─ Streaming Parallel      │
│  └─ Dual-Stream Masking            │  │   Decoding                │
│                                    │  └─ Distance-Penalized      │
│                                    │      Selection              │
└─────────────────────────────────────────────────────────────────┘
```

### 1.4 Résultats Clés

- **~3× speedup** sur GSM8K vs vLLM-optimized Qwen3-8B
- **Jusqu'à 10×** sur tâches à basse entropie
- **Qualité préservée** : +2.4 points sur le score moyen vs baseline
- **Compatible** avec FlashAttention, PagedAttention, CUDA Graphs

---

## 2. Concepts Fondamentaux

### 2.1 Autoregressive vs Diffusion

#### Modèle Autorégressif (AR)

```
P(x) = ∏ P(xₜ | x<t; θ)
         t=1

Génération : x₁ → x₂ → x₃ → ... → xₜ (séquentiel)
```

**Avantages** : KV caching natif, infrastructure mature  
**Inconvénients** : 1 token/forward, sous-utilisation GPU

#### Masked Diffusion Language Model (MDLM)

```
Noising  : x₀ → x_γ  (masquer γL positions aléatoires)
Denoising: x_γ → x̂₀  (prédire tokens masqués en parallèle)

Loss: L(θ) = -E[1/γ ∑ log p_θ(x₀⁽ⁱ⁾ | x_γ)]
                    i∈M
```

**Avantages** : Parallélisme, contexte bidirectionnel  
**Inconvénients** : Incompatible KV cache, recomputation

### 2.2 Métrique Clé : Prefix Cacheability (p_cache)

WeDLM introduit une métrique fondamentale pour évaluer l'efficacité du décodage :

```
              N_gen
p_cache = ─────────── ∈ (0, 1]
              N_fwd

Où:
- N_gen = nombre de tokens finalement générés
- N_fwd = nombre total d'instances de tokens traités par le réseau
```

**Interprétation** :
- `p_cache = 1.0` : Chaque token traité devient un token final (optimal)
- `p_cache = 0.2` : En moyenne, chaque token est traité 5 fois (inefficace)

**Comparaison des méthodes** :

| Méthode | p_cache typique | Recomputation Factor |
|---------|-----------------|---------------------|
| AR (baseline) | 1.0 | 1× |
| LLaDA (bidir) | ~0.15-0.25 | 4-7× |
| Block Diffusion | ~0.3-0.5 | 2-3× |
| **WeDLM** | **~0.7-0.9** | **1.1-1.4×** |

### 2.3 Positional Encoding Décorrélé

WeDLM exploite la séparation entre :
- **Position physique** : Index dans le tenseur d'entrée
- **Position logique** : Position sémantique dans la séquence (via RoPE position IDs)

```python
# Standard AR : positions physiques = positions logiques
input  = [x₁, x₂, x₃, x₄]
pos_id = [0,  1,  2,  3 ]

# WeDLM : positions découplées
input  = [x₁, x₄, x₂, x₃]  # Réordonné physiquement
pos_id = [0,  3,  1,  2 ]  # Positions logiques préservées
```

Avec RoPE, les scores d'attention sont calculés sur les positions logiques :

```
Attention(Q, K) = softmax(Q·K^T / √d) où Q, K incluent RoPE(pos_id)
```

---

## 3. Architecture WeDLM

### 3.1 Vue d'Ensemble

```
                              WeDLM Architecture
┌────────────────────────────────────────────────────────────────────┐
│                                                                    │
│   TRAINING PHASE                                                   │
│   ══════════════                                                   │
│                                                                    │
│   Input: x₀ = [x₁, x₂, x₃, x₄, x₅, x₆, x₇, x₈]                    │
│                     │                                              │
│                     ▼                                              │
│   ┌─────────────────────────────────────────┐                     │
│   │      Dual-Stream Construction           │                     │
│   │  ┌─────────────┐  ┌─────────────────┐  │                     │
│   │  │Memory Stream│  │Prediction Stream│  │                     │
│   │  │   (clean)   │  │   (masked)      │  │                     │
│   │  └─────────────┘  └─────────────────┘  │                     │
│   └─────────────────────────────────────────┘                     │
│                     │                                              │
│                     ▼                                              │
│   ┌─────────────────────────────────────────┐                     │
│   │   Block-wise Topological Reordering     │                     │
│   │   (within each prediction block)        │                     │
│   └─────────────────────────────────────────┘                     │
│                     │                                              │
│                     ▼                                              │
│   ┌─────────────────────────────────────────┐                     │
│   │   Causal Transformer + Block Attention  │                     │
│   └─────────────────────────────────────────┘                     │
│                     │                                              │
│                     ▼                                              │
│   Loss = -∑∑ (1/γₖ) log P(x₀⁽ʲ⁾ | x_o^(<k), x̃_t^(k,<j))          │
│           k j∈Mₖ                                                   │
│                                                                    │
├────────────────────────────────────────────────────────────────────┤
│                                                                    │
│   INFERENCE PHASE                                                  │
│   ═══════════════                                                  │
│                                                                    │
│   Prompt: [p₁, p₂, ..., pₙ]                                       │
│                     │                                              │
│                     ▼                                              │
│   ┌─────────────────────────────────────────┐                     │
│   │           Prefill (KV Cache Init)       │                     │
│   └─────────────────────────────────────────┘                     │
│                     │                                              │
│                     ▼                                              │
│   ┌─────────────────────────────────────────┐                     │
│   │     Streaming Parallel Decoding Loop    │◄────────┐          │
│   │  ┌───────────────────────────────────┐  │         │          │
│   │  │ 1. Reorder window (obs → masks)   │  │         │          │
│   │  │ 2. Forward with KV cache          │  │         │          │
│   │  │ 3. Commit contiguous prefix       │  │         │          │
│   │  │ 4. Predict (distance-penalized)   │  │         │          │
│   │  │ 5. Refill window with new [M]     │──┼─────────┘          │
│   │  └───────────────────────────────────┘  │                     │
│   └─────────────────────────────────────────┘                     │
│                     │                                              │
│                     ▼                                              │
│   Output: [p₁, ..., pₙ, y₁, y₂, ..., yₘ]                         │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
```

### 3.2 Composants Principaux

| Composant | Phase | Rôle |
|-----------|-------|------|
| Topological Reordering | Train + Infer | Expose contexte observé sous attention causale |
| Dual-Stream Masking | Train | Simule le conditioning prefix-based |
| Streaming Parallel Decoding | Infer | Maximise p_cache avec fenêtre glissante |
| Distance Penalty | Infer | Favorise résolution gauche-droite |

---

## 4. Contribution 1 : Topological Reordering

### 4.1 Motivation

Pour que la récupération de masques fonctionne, chaque position masquée doit voir **tout le contexte observé**. Avec l'attention bidirectionnelle, c'est trivial. Avec l'attention causale, un token à la position `i` ne voit que les positions `< i`.

**Solution** : Réordonner physiquement la séquence pour placer tous les tokens observés AVANT les tokens masqués.

### 4.2 Algorithme

```
ALGORITHME: Topological Reordering
═══════════════════════════════════

ENTRÉE:
  - tokens: [x₁, x₂, ..., xₗ]     # Séquence de tokens
  - M ⊂ {1,...,L}                  # Indices des positions masquées
  - O = {1,...,L} \ M              # Indices des positions observées

SORTIE:
  - x̃: séquence réordonnée
  - p̃: positions logiques correspondantes

ÉTAPES:
  1. Séparer indices observés et masqués
     O_sorted = sort(O)  # [o₁, o₂, ..., oₙₒ] en ordre croissant
     M_sorted = sort(M)  # [m₁, m₂, ..., mₙₘ] en ordre croissant

  2. Construire séquence réordonnée
     x̃ = [x_o₁, x_o₂, ..., x_oₙₒ, [M], [M], ..., [M]]
         └────────────────────┘  └─────────────────┘
              N_o tokens              N_m masks

  3. Construire positions logiques
     p̃ = [o₁, o₂, ..., oₙₒ, m₁, m₂, ..., mₙₘ]

  4. Retourner (x̃, p̃)
```

### 4.3 Exemple Visuel

```
AVANT REORDERING:
═════════════════
Position physique:  [0]   [1]   [2]   [3]   [4]   [5]
Position logique:   [0]   [1]   [2]   [3]   [4]   [5]
Tokens:             x₁    x₂    [M]   x₄    [M]   x₆
                    obs   obs   mask  obs   mask  obs

Masque causal standard:
     0   1   2   3   4   5
   ┌───┬───┬───┬───┬───┬───┐
 0 │ ✓ │   │   │   │   │   │  x₁ voit: x₁
 1 │ ✓ │ ✓ │   │   │   │   │  x₂ voit: x₁,x₂
 2 │ ✓ │ ✓ │ ✓ │   │   │   │  [M] voit: x₁,x₂ (manque x₄,x₆!)
 3 │ ✓ │ ✓ │ ✓ │ ✓ │   │   │  x₄ voit: x₁,x₂,[M],x₄
 4 │ ✓ │ ✓ │ ✓ │ ✓ │ ✓ │   │  [M] voit: x₁,x₂,[M],x₄ (manque x₆!)
 5 │ ✓ │ ✓ │ ✓ │ ✓ │ ✓ │ ✓ │  x₆ voit: tout
   └───┴───┴───┴───┴───┴───┘

⚠️  PROBLÈME: Les [M] ne voient pas tout le contexte observé!


APRÈS REORDERING:
═════════════════
Position physique:  [0]   [1]   [2]   [3]   [4]   [5]
Position logique:   [0]   [1]   [3]   [5]   [2]   [4]
Tokens:             x₁    x₂    x₄    x₆    [M]   [M]
                    obs   obs   obs   obs   mask  mask

Masque causal standard:
     0   1   2   3   4   5
   ┌───┬───┬───┬───┬───┬───┐
 0 │ ✓ │   │   │   │   │   │  x₁ voit: x₁
 1 │ ✓ │ ✓ │   │   │   │   │  x₂ voit: x₁,x₂
 2 │ ✓ │ ✓ │ ✓ │   │   │   │  x₄ voit: x₁,x₂,x₄
 3 │ ✓ │ ✓ │ ✓ │ ✓ │   │   │  x₆ voit: x₁,x₂,x₄,x₆
 4 │ ✓ │ ✓ │ ✓ │ ✓ │ ✓ │   │  [M] voit: x₁,x₂,x₄,x₆ ✓ TOUT!
 5 │ ✓ │ ✓ │ ✓ │ ✓ │ ✓ │ ✓ │  [M] voit: x₁,x₂,x₄,x₆,[M] ✓ TOUT!
   └───┴───┴───┴───┴───┴───┘

✓ SOLUTION: Tous les [M] voient tout le contexte observé!

Note: Les scores d'attention utilisent les positions LOGIQUES (via RoPE),
      donc x₄ à position physique 2 utilise quand même pos_id=3.
```

### 4.4 Implémentation PyTorch

```python
def topological_reordering(
    tokens: torch.Tensor,        # [batch, seq_len]
    mask_positions: torch.Tensor, # [batch, seq_len] bool
    mask_token_id: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Réordonne les tokens: observés d'abord, masqués ensuite.
    
    Returns:
        reordered_tokens: [batch, seq_len] - Tokens réordonnés
        reordered_positions: [batch, seq_len] - Positions logiques
        inverse_perm: [batch, seq_len] - Pour restaurer l'ordre original
    """
    batch_size, seq_len = tokens.shape
    device = tokens.device
    
    # Positions logiques originales
    positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
    
    # Clé de tri: observés (0) avant masqués (1), puis par position
    # mask_positions.float() donne 0 pour obs, 1 pour mask
    # Multiplier par seq_len assure que tous les obs viennent avant tous les masks
    sort_keys = mask_positions.float() * seq_len + positions.float()
    
    # Obtenir la permutation
    _, permutation = sort_keys.sort(dim=1, stable=True)
    
    # Appliquer la permutation
    batch_idx = torch.arange(batch_size, device=device).unsqueeze(1)
    reordered_tokens = tokens[batch_idx, permutation]
    reordered_positions = positions[batch_idx, permutation]
    
    # Calculer permutation inverse
    inverse_perm = torch.zeros_like(permutation)
    inverse_perm.scatter_(1, permutation, positions)
    
    return reordered_tokens, reordered_positions, inverse_perm
```

### 4.5 Propriétés Mathématiques

**Théorème (Visibilité Complète)** : Soit `x̃` la séquence réordonnée avec `N_o` tokens observés. Alors pour tout token masqué à la position physique `i ≥ N_o`, sous attention causale :

```
Attend(i) = {0, 1, ..., i-1} ⊇ {0, 1, ..., N_o-1} = tous les observés
```

**Preuve** : Par construction, tous les tokens observés occupent les positions `{0, ..., N_o-1}`. Tout token masqué est à une position `i ≥ N_o`, donc voit au moins `{0, ..., N_o-1}`. ∎

---

## 5. Contribution 2 : Dual-Stream Masking

### 5.1 Motivation

Le Topological Reordering permet l'entraînement avec attention causale, mais crée un **gap train/inference** :

| Aspect | Entraînement Standard | Inférence |
|--------|----------------------|-----------|
| Masquage | Uniforme sur toute la séquence | Concentré en suffixe (gauche→droite) |
| Contexte | Peut inclure des prédictions bruitées | Contexte toujours "clean" |
| Pattern | Aléatoire | Bloc par bloc |

**Solution** : Simuler le régime d'inférence pendant l'entraînement via deux flux.

### 5.2 Architecture Dual-Stream

```
DUAL-STREAM MASKING
═══════════════════

Entrée originale: x₀ = [x₁, x₂, x₃, x₄, x₅, x₆, x₇, x₈]
                        └──Block 1──┘  └──Block 2──┘

Construction:
┌────────────────────────────────────────────────────────────────┐
│                                                                │
│  MEMORY STREAM (x_o) - Tokens clean, pas de masquage          │
│  ════════════════════════════════════════════════             │
│  Tokens:    [x₁] [x₂] [x₃] [x₄] [x₅] [x₆] [x₇] [x₈]          │
│  Positions: [0]  [1]  [2]  [3]  [4]  [5]  [6]  [7]            │
│                                                                │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  PREDICTION STREAM (x_t) - Masqué + réordonné par bloc        │
│  ═══════════════════════════════════════════════════          │
│                                                                │
│  Block 1 (masque pos 1,2):                                    │
│    Original: [x₁] [x₂] [x₃] [x₄]                              │
│    Masqué:   [x₁] [M]  [M]  [x₄]                              │
│    Reorder:  [x₁] [x₄] [M]  [M]   (obs first)                 │
│    Pos IDs:  [0]  [3]  [1]  [2]                               │
│                                                                │
│  Block 2 (masque pos 5,7):                                    │
│    Original: [x₅] [x₆] [x₇] [x₈]                              │
│    Masqué:   [M]  [x₆] [M]  [x₈]                              │
│    Reorder:  [x₆] [x₈] [M]  [M]   (obs first)                 │
│    Pos IDs:  [5]  [7]  [4]  [6]                               │
│                                                                │
│  Prediction Stream final:                                      │
│  Tokens:    [x₁] [x₄] [M]  [M]  [x₆] [x₈] [M]  [M]            │
│  Positions: [0]  [3]  [1]  [2]  [5]  [7]  [4]  [6]            │
│             └───Block 1────┘   └───Block 2────┘               │
│                                                                │
└────────────────────────────────────────────────────────────────┘

Séquence combinée: [Memory Stream ; Prediction Stream]
Longueur totale: 2L (ici 16 tokens)
```

### 5.3 Masque d'Attention

Le masque d'attention est **crucial** pour simuler l'inférence :

```
ATTENTION MASK (16 x 16)
════════════════════════

                    MEMORY STREAM              PREDICTION STREAM
                    Block 1    Block 2         Block 1    Block 2
                    0 1 2 3    4 5 6 7         8 9 10 11  12 13 14 15

Memory      B1  0   ■ □ □ □    □ □ □ □         □ □ □  □   □  □  □  □
Stream      B1  1   ■ ■ □ □    □ □ □ □         □ □ □  □   □  □  □  □
            B1  2   ■ ■ ■ □    □ □ □ □         □ □ □  □   □  □  □  □
            B1  3   ■ ■ ■ ■    □ □ □ □         □ □ □  □   □  □  □  □
            B2  4   ■ ■ ■ ■    ■ □ □ □         □ □ □  □   □  □  □  □
            B2  5   ■ ■ ■ ■    ■ ■ □ □         □ □ □  □   □  □  □  □
            B2  6   ■ ■ ■ ■    ■ ■ ■ □         □ □ □  □   □  □  □  □
            B2  7   ■ ■ ■ ■    ■ ■ ■ ■         □ □ □  □   □  □  □  □

Pred        B1  8   □ □ □ □    □ □ □ □         ■ □ □  □   □  □  □  □
Stream      B1  9   □ □ □ □    □ □ □ □         ■ ■ □  □   □  □  □  □
            B1  10  □ □ □ □    □ □ □ □         ■ ■ ■  □   □  □  □  □
            B1  11  □ □ □ □    □ □ □ □         ■ ■ ■  ■   □  □  □  □
            B2  12  ■ ■ ■ ■    □ □ □ □         □ □ □  □   ■  □  □  □
            B2  13  ■ ■ ■ ■    □ □ □ □         □ □ □  □   ■  ■  □  □
            B2  14  ■ ■ ■ ■    □ □ □ □         □ □ □  □   ■  ■  ■  □
            B2  15  ■ ■ ■ ■    □ □ □ □         □ □ □  □   ■  ■  ■  ■

■ = Attention autorisée    □ = Attention bloquée

RÈGLES:
1. Memory Stream: attention causale standard (triangulaire inférieure)
2. Pred Block k → Memory: voit positions AVANT le block k (clean context)
3. Pred Block k → Pred Block k: attention causale intra-bloc
4. Pred Block k → Pred Block j (j≠k): BLOQUÉ (pas de cross-block leakage)
5. Pred Stream → Memory après block k: BLOQUÉ
```

### 5.4 Fonction de Perte

```
            K    1      
L(θ) = -E  [ Σ  ──── Σ  log P_θ(x₀⁽ʲ⁾ | x_o^(<k), x̃_t^(k,<j)) ]
            k=1  γₖ  j∈Mₖ

Où:
- K = nombre de blocs
- γₖ = ratio de masquage du bloc k
- Mₖ = ensemble des positions masquées dans le bloc k
- x_o^(<k) = tokens du memory stream avant le bloc k
- x̃_t^(k,<j) = tokens du prediction stream bloc k avant position j
```

**Interprétation** :
- Le facteur `1/γₖ` compense les blocs avec plus/moins de masques
- Chaque bloc prédit ses masques conditionnés sur le contexte "clean"
- Pas de dépendance aux prédictions (potentiellement bruitées) des autres blocs

### 5.5 Implémentation

```python
class DualStreamMasking(nn.Module):
    def __init__(self, block_size: int = 32, mask_token_id: int = 32000):
        super().__init__()
        self.block_size = block_size
        self.mask_token_id = mask_token_id
    
    def forward(self, input_ids: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            input_ids: [batch, seq_len] clean tokens
        
        Returns:
            - combined_input: [batch, 2*seq_len]
            - position_ids: [batch, 2*seq_len]  
            - attention_mask: [2*seq_len, 2*seq_len]
            - targets: [batch, seq_len]
            - loss_weights: [batch, seq_len]
        """
        B, L = input_ids.shape
        device = input_ids.device
        K = math.ceil(L / self.block_size)
        
        # Memory stream (clean)
        memory_stream = input_ids.clone()
        
        # Prediction stream (masked + reordered)
        pred_stream = input_ids.clone()
        loss_weights = torch.zeros(B, L, device=device)
        
        for k in range(K):
            start = k * self.block_size
            end = min((k + 1) * self.block_size, L)
            block_len = end - start
            
            # Sample mask ratio
            gamma = torch.empty(1).uniform_(0.1, 0.9).item()
            num_mask = max(1, int(gamma * block_len))
            
            # Random mask positions
            mask_idx = torch.randperm(block_len)[:num_mask]
            block_mask = torch.zeros(block_len, dtype=torch.bool, device=device)
            block_mask[mask_idx] = True
            
            # Apply topological reordering within block
            block_tokens = pred_stream[:, start:end]
            reordered, positions, _ = topological_reordering(
                block_tokens,
                block_mask.unsqueeze(0).expand(B, -1),
                self.mask_token_id
            )
            
            # Replace masks
            reordered[~block_mask.unsqueeze(0).expand(B, -1)] = self.mask_token_id
            pred_stream[:, start:end] = reordered
            
            # Loss weights: 1/gamma for masked positions
            loss_weights[:, start:end] = block_mask.float() / gamma
        
        # Combine streams
        combined = torch.cat([memory_stream, pred_stream], dim=1)
        
        # Position IDs (shared)
        pos = torch.arange(L, device=device)
        position_ids = torch.cat([pos, pos]).unsqueeze(0).expand(B, -1)
        
        # Build attention mask
        attention_mask = self._build_attention_mask(L, K, device)
        
        return {
            'input_ids': combined,
            'position_ids': position_ids,
            'attention_mask': attention_mask,
            'labels': input_ids,
            'loss_weights': loss_weights,
        }
    
    def _build_attention_mask(self, L: int, K: int, device) -> torch.Tensor:
        """Build the dual-stream attention mask."""
        total = 2 * L
        mask = torch.zeros(total, total, dtype=torch.bool, device=device)
        
        # Memory: causal
        mask[:L, :L] = torch.tril(torch.ones(L, L, dtype=torch.bool, device=device))
        
        # Prediction blocks
        for k in range(K):
            start = k * self.block_size
            end = min((k + 1) * self.block_size, L)
            pred_start = L + start
            pred_end = L + end
            block_len = end - start
            
            # Attend to memory before block k
            if start > 0:
                mask[pred_start:pred_end, :start] = True
            
            # Causal within block
            mask[pred_start:pred_end, pred_start:pred_end] = torch.tril(
                torch.ones(block_len, block_len, dtype=torch.bool, device=device)
            )
        
        return mask
```

---

## 6. Contribution 3 : Streaming Parallel Decoding

### 6.1 Motivation

Les méthodes block-wise existantes souffrent de **stop-and-wait** :

```
BLOCK DECODING (problématique)
══════════════════════════════

Step 1: [M][M][M][M]     → Predict all
Step 2: [A][B][?][?]     → 2 confident, 2 uncertain
Step 3: [A][B][C][?]     → Wait... still uncertain
Step 4: [A][B][C][D]     → Finally complete! Cache all
Step 5: [M][M][M][M]     → New block

⚠️ Problèmes:
- Tokens A,B sont prêts au step 2 mais on attend step 4
- GPU sous-utilisé pendant l'attente
- Bidirectional attention = pas de cache progressif
```

### 6.2 Streaming Parallel Decoding

```
STREAMING DECODING (WeDLM)
══════════════════════════

Fenêtre fixe W=6, commit progressif:

Step 1: [M][M][M][M][M][M]     Initial
        └─────────────────→ Forward
        
Step 2: [A][B][M][M][M][M]     Predict A,B (low entropy)
        [■][■]                 Commit A,B (contiguous prefix!)
        
Step 3: [M][M][M][M][M][M]     Slide + Refill
           └─prev─┘└─new─┘
        [C][M][E][M][M][M]     Predict C,E
        [■]                    Commit C only (E not contiguous)
        
Step 4: [M][E][M][M][M][M]     Slide + Refill
        [D][E][F][M][M][M]     Predict D,F (E already filled)
        [■][■][■]              Commit D,E,F (now contiguous!)

✓ Avantages:
- Commit immédiat dès qu'un préfixe est contiguë
- Pas d'attente de bloc complet
- GPU toujours occupé (fenêtre fixe)
- KV cache valide immédiatement (causal attention)
```

### 6.3 Distance Penalty

Pour favoriser la résolution gauche→droite et maximiser le préfixe contigu :

```
DISTANCE-PENALIZED ENTROPY (Eq. 11)
═══════════════════════════════════

H̃ᵢ = Hᵢ + λ · dᵢ

Où:
- Hᵢ = entropie de la distribution prédite à la position i
- dᵢ = distance de i au premier [M] dans la fenêtre
- λ = coefficient de pénalité (hyperparamètre, ~0.05-0.1)

Exemple avec fenêtre [M₀][M₁][M₂][M₃]:
- Position 0: d=0, H̃₀ = H₀
- Position 1: d=1, H̃₁ = H₁ + λ
- Position 2: d=2, H̃₂ = H₂ + 2λ
- Position 3: d=3, H̃₃ = H₃ + 3λ

→ À entropie égale, on préfère les positions de gauche!

SÉLECTION:
  Fill position i if H̃ᵢ < τ (threshold)
```

**Effet sur p_cache** :

```
Sans distance penalty:           Avec distance penalty:
─────────────────────           ─────────────────────
Step 1: [A][M][C][M]            Step 1: [A][B][M][M]
        Commit: A seulement             Commit: A,B (contiguous!)
        p_cache: 1/4 = 0.25             p_cache: 2/4 = 0.50

La distance penalty double p_cache dans cet exemple!
```

### 6.4 Algorithme Complet

```
ALGORITHM 1: Streaming Parallel Decoding
════════════════════════════════════════

REQUIRE: 
  - prompt x
  - window size W
  - entropy threshold τ  
  - distance penalty λ

ENSURE: Generated sequence y

 1: y ← []
 2: (K, V) ← PREFILL(x)                    // Initialize KV cache
 3: W ← [[M]]^W                             // Window of W masks
 4: next_pos ← len(x)                       // Next global position
 5: 
 6: WHILE W ≠ ∅ AND len(y) < max_tokens DO
 7:     
 8:     // ══ REORDER ══
 9:     W_filled ← [slot for slot in W if slot.is_filled]
10:     W_masks ← [slot for slot in W if not slot.is_filled]  
11:     W ← W_filled + W_masks              // Filled first, masks last
12:     
13:     // ══ FORWARD ══
14:     tokens ← [slot.token for slot in W]
15:     positions ← [slot.position for slot in W]
16:     (logits, K_w, V_w) ← FORWARD(tokens, positions, K, V)
17:     
18:     // ══ COMMIT ══
19:     // Find leftmost contiguous filled prefix (by position)
20:     committed ← []
21:     sorted_filled ← SORT(W_filled, key=position)
22:     expected_pos ← sorted_filled[0].position
23:     FOR slot IN sorted_filled DO
24:         IF slot.position == expected_pos THEN
25:             committed.APPEND(slot.token)
26:             expected_pos ← expected_pos + 1
27:         ELSE
28:             BREAK
29:     
30:     // Update output and cache
31:     y.EXTEND(committed)
32:     (K, V).EXTEND(K_w[:len(committed)], V_w[:len(committed)])
33:     
34:     IF EOS IN committed THEN BREAK
35:     
36:     // ══ PREDICT ══
37:     FOR i, slot IN ENUMERATE(W_masks) DO
38:         d_i ← slot.position - MIN_POS(W_masks)
39:         H_i ← ENTROPY(logits[len(W_filled) + i])
40:         H̃_i ← H_i + λ * d_i
41:         
42:         IF H̃_i < τ THEN
43:             token ← SAMPLE(logits[len(W_filled) + i])
44:             slot.token ← token
45:             slot.is_filled ← True
46:     
47:     // ══ SLIDE & REFILL ══
48:     W ← W[len(committed):]               // Remove committed
49:     FOR i IN RANGE(len(committed)) DO
50:         W.APPEND(Slot(token=[M], position=next_pos, is_filled=False))
51:         next_pos ← next_pos + 1
52:     
53: RETURN CONCAT(x, y)
```

### 6.5 Implémentation PyTorch

```python
@dataclass
class DecodingConfig:
    window_size: int = 6
    entropy_threshold: float = 0.5
    distance_penalty: float = 0.1
    mask_token_id: int = 32000
    eos_token_id: int = 2
    max_new_tokens: int = 512


class StreamingDecoder:
    def __init__(self, model: nn.Module, config: DecodingConfig):
        self.model = model
        self.config = config
    
    @torch.no_grad()
    def generate(self, prompt_ids: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        device = prompt_ids.device
        cfg = self.config
        
        # Prefill
        outputs = self.model(prompt_ids, use_cache=True)
        past_kv = outputs.past_key_values
        
        # Initialize window
        window = [
            {'token': cfg.mask_token_id, 'position': prompt_ids.shape[1] + i, 'filled': False}
            for i in range(cfg.window_size)
        ]
        next_pos = prompt_ids.shape[1] + cfg.window_size
        
        generated = []
        
        while len(generated) < cfg.max_new_tokens and window:
            # 1. REORDER
            filled = [s for s in window if s['filled']]
            masks = [s for s in window if not s['filled']]
            window = filled + masks
            n_filled = len(filled)
            
            # 2. FORWARD
            input_ids = torch.tensor([[s['token'] for s in window]], device=device)
            position_ids = torch.tensor([[s['position'] for s in window]], device=device)
            
            outputs = self.model(
                input_ids=input_ids,
                position_ids=position_ids,
                past_key_values=past_kv,
                use_cache=True
            )
            logits = outputs.logits[0]  # [window_size, vocab]
            new_kv = outputs.past_key_values
            
            # 3. COMMIT contiguous prefix
            committed = []
            if n_filled > 0:
                sorted_filled = sorted(enumerate(filled), key=lambda x: x[1]['position'])
                expected = sorted_filled[0][1]['position']
                for idx, slot in sorted_filled:
                    if slot['position'] == expected:
                        committed.append(slot['token'])
                        expected += 1
                    else:
                        break
            
            generated.extend(committed)
            
            # Update KV cache (simplified - real impl needs proper slicing)
            if committed:
                past_kv = self._update_cache(past_kv, new_kv, len(committed))
            
            if cfg.eos_token_id in committed:
                break
            
            # 4. PREDICT with distance penalty
            if masks:
                mask_logits = logits[n_filled:]  # [n_masks, vocab]
                probs = F.softmax(mask_logits / temperature, dim=-1)
                entropy = -(probs * probs.log()).sum(dim=-1)  # [n_masks]
                
                min_pos = min(s['position'] for s in masks)
                
                for i, slot in enumerate(masks):
                    distance = slot['position'] - min_pos
                    adjusted_entropy = entropy[i].item() + cfg.distance_penalty * distance
                    
                    if adjusted_entropy < cfg.entropy_threshold:
                        token = torch.multinomial(probs[i], 1).item()
                        slot['token'] = token
                        slot['filled'] = True
            
            # 5. SLIDE & REFILL
            window = window[len(committed):]
            for _ in range(len(committed)):
                if len(generated) + len(window) < cfg.max_new_tokens:
                    window.append({
                        'token': cfg.mask_token_id,
                        'position': next_pos,
                        'filled': False
                    })
                    next_pos += 1
        
        result = torch.cat([
            prompt_ids,
            torch.tensor([generated], device=device)
        ], dim=1)
        
        return result
    
    def _update_cache(self, old_kv, new_kv, n_commit):
        """Update KV cache with committed tokens."""
        # Implementation depends on model architecture
        # Basic idea: append KV states for committed positions
        updated = []
        for (old_k, old_v), (new_k, new_v) in zip(old_kv, new_kv):
            # new_k/v shape: [batch, heads, window_size, dim]
            # Take only first n_commit positions
            updated.append((
                torch.cat([old_k, new_k[:, :, :n_commit]], dim=2),
                torch.cat([old_v, new_v[:, :, :n_commit]], dim=2)
            ))
        return tuple(updated)
```

---

## 7. Implémentation Détaillée

### 7.1 Structure des Fichiers

```
wedlm/
├── __init__.py
├── config.py                 # Configuration classes
├── modeling/
│   ├── __init__.py
│   ├── reordering.py         # Topological Reordering
│   ├── dual_stream.py        # Dual-Stream Masking
│   └── attention.py          # Custom attention masks
├── training/
│   ├── __init__.py
│   ├── trainer.py            # Training loop
│   ├── loss.py               # WeDLM loss function
│   └── data.py               # Data preparation
├── inference/
│   ├── __init__.py
│   ├── decoder.py            # Streaming Parallel Decoder
│   └── cache.py              # KV cache management
└── utils/
    ├── __init__.py
    └── metrics.py            # p_cache and other metrics
```

### 7.2 Configuration Complète

```python
from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class WeDLMConfig:
    """Configuration complète pour WeDLM."""
    
    # Model
    base_model: str = "Qwen/Qwen2.5-7B"
    mask_token_id: int = 32000
    eos_token_id: int = 2
    pad_token_id: int = 0
    
    # Training - Dual Stream
    block_size: int = 32
    min_mask_ratio: float = 0.1
    max_mask_ratio: float = 0.9
    ar_loss_weight: float = 0.1  # Auxiliary AR loss
    
    # Training - Optimization
    learning_rate: float = 3e-6
    min_learning_rate: float = 3e-7
    warmup_steps: int = 1000
    total_steps: int = 100000
    batch_size: int = 32
    gradient_accumulation: int = 4
    
    # Inference - Streaming Decoding
    window_size: int = 6
    entropy_threshold: float = 0.5
    distance_penalty: float = 0.1
    max_new_tokens: int = 512
    temperature: float = 1.0
    
    # System
    dtype: str = "bfloat16"
    use_flash_attention: bool = True
    compile_model: bool = False


@dataclass 
class TrainingState:
    """État d'entraînement pour checkpointing."""
    step: int = 0
    epoch: int = 0
    best_loss: float = float('inf')
    tokens_seen: int = 0
```

### 7.3 Training Loop

```python
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm


class WeDLMTrainer:
    def __init__(self, config: WeDLMConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load base model
        self.model = AutoModelForCausalLM.from_pretrained(
            config.base_model,
            torch_dtype=getattr(torch, config.dtype),
            attn_implementation="flash_attention_2" if config.use_flash_attention else "eager"
        ).to(self.device)
        
        self.tokenizer = AutoTokenizer.from_pretrained(config.base_model)
        
        # Add mask token if not present
        if self.tokenizer.mask_token is None:
            self.tokenizer.add_special_tokens({'mask_token': '[MASK]'})
            self.model.resize_token_embeddings(len(self.tokenizer))
        
        # Training components
        self.dual_stream = DualStreamMasking(
            block_size=config.block_size,
            mask_token_id=self.tokenizer.mask_token_id
        )
        
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=0.01
        )
        
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=config.total_steps,
            eta_min=config.min_learning_rate
        )
    
    def train_step(self, batch: torch.Tensor) -> dict:
        """Single training step."""
        self.model.train()
        
        # Prepare dual-stream input
        ds_output = self.dual_stream(batch)
        
        input_ids = ds_output['input_ids'].to(self.device)
        position_ids = ds_output['position_ids'].to(self.device)
        attention_mask = ds_output['attention_mask'].to(self.device)
        labels = ds_output['labels'].to(self.device)
        loss_weights = ds_output['loss_weights'].to(self.device)
        
        # Forward pass
        outputs = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            output_hidden_states=False
        )
        
        # Extract prediction stream logits (second half)
        seq_len = labels.shape[1]
        pred_logits = outputs.logits[:, seq_len:, :]  # [B, L, V]
        
        # WeDLM loss
        diffusion_loss = self._compute_wedlm_loss(
            pred_logits, labels, loss_weights
        )
        
        # Auxiliary AR loss (optional, helps preserve AR capability)
        ar_loss = torch.tensor(0.0, device=self.device)
        if self.config.ar_loss_weight > 0:
            memory_logits = outputs.logits[:, :seq_len-1, :]
            ar_targets = batch[:, 1:].to(self.device)
            ar_loss = F.cross_entropy(
                memory_logits.reshape(-1, memory_logits.size(-1)),
                ar_targets.reshape(-1),
                ignore_index=self.config.pad_token_id
            )
        
        # Total loss
        total_loss = diffusion_loss + self.config.ar_loss_weight * ar_loss
        
        # Backward
        total_loss.backward()
        
        return {
            'loss': total_loss.item(),
            'diffusion_loss': diffusion_loss.item(),
            'ar_loss': ar_loss.item(),
        }
    
    def _compute_wedlm_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        weights: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute WeDLM loss with per-block weighting.
        
        Args:
            logits: [B, L, V] - Predictions from prediction stream
            targets: [B, L] - Ground truth tokens
            weights: [B, L] - 1/gamma weights (0 for non-masked)
        """
        B, L, V = logits.shape
        
        # Flatten
        logits_flat = logits.reshape(-1, V)
        targets_flat = targets.reshape(-1)
        
        # Per-token CE
        ce_loss = F.cross_entropy(logits_flat, targets_flat, reduction='none')
        ce_loss = ce_loss.reshape(B, L)
        
        # Apply weights
        weighted_loss = ce_loss * weights
        
        # Average over masked positions
        mask = weights > 0
        if mask.sum() > 0:
            return weighted_loss.sum() / mask.sum()
        return torch.tensor(0.0, device=logits.device)
    
    def train(self, dataloader, num_epochs: int = 1):
        """Full training loop."""
        state = TrainingState()
        
        for epoch in range(num_epochs):
            state.epoch = epoch
            pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
            
            for batch in pbar:
                metrics = self.train_step(batch)
                
                # Gradient accumulation
                if (state.step + 1) % self.config.gradient_accumulation == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                
                state.step += 1
                state.tokens_seen += batch.numel()
                
                pbar.set_postfix({
                    'loss': f"{metrics['loss']:.4f}",
                    'lr': f"{self.scheduler.get_last_lr()[0]:.2e}"
                })
                
                # Checkpoint
                if state.step % 1000 == 0:
                    self.save_checkpoint(state)
        
        return state
    
    def save_checkpoint(self, state: TrainingState):
        """Save model checkpoint."""
        torch.save({
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'state': state,
            'config': self.config,
        }, f"checkpoint_step_{state.step}.pt")
```

---

## 8. Guide d'Intégration

### 8.1 Intégration avec Hugging Face Transformers

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from wedlm import WeDLMConfig, StreamingDecoder


def load_wedlm_model(model_path: str):
    """Charger un modèle WeDLM pré-entraîné."""
    
    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="flash_attention_2"
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    return model, tokenizer


def generate_with_wedlm(
    model,
    tokenizer,
    prompt: str,
    config: Optional[WeDLMConfig] = None
) -> str:
    """Génération avec WeDLM Streaming Decoding."""
    
    if config is None:
        config = WeDLMConfig()
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Create decoder
    decoder = StreamingDecoder(model, config)
    
    # Generate
    output_ids = decoder.generate(
        inputs.input_ids,
        temperature=config.temperature
    )
    
    # Decode
    response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    return response


# Usage
model, tokenizer = load_wedlm_model("tencent/WeDLM-8B-Instruct")
response = generate_with_wedlm(model, tokenizer, "Explain quantum computing:")
print(response)
```

### 8.2 Intégration avec vLLM

WeDLM est conçu pour être compatible avec vLLM. Voici comment l'intégrer :

```python
# Note: Nécessite des modifications à vLLM pour supporter
# les position_ids personnalisés et la fenêtre glissante

from vllm import LLM, SamplingParams
from wedlm.vllm_integration import WeDLMSampler


class WeDLMvLLM:
    """Wrapper pour utiliser WeDLM avec vLLM."""
    
    def __init__(self, model_path: str):
        self.llm = LLM(
            model=model_path,
            dtype="bfloat16",
            gpu_memory_utilization=0.9,
        )
        self.sampler = WeDLMSampler()
    
    def generate(
        self,
        prompts: List[str],
        window_size: int = 6,
        entropy_threshold: float = 0.5,
        distance_penalty: float = 0.1,
    ) -> List[str]:
        """
        Batch generation avec WeDLM.
        
        Note: L'implémentation complète nécessite de modifier
        vLLM pour supporter le streaming decoding de WeDLM.
        """
        sampling_params = SamplingParams(
            temperature=1.0,
            max_tokens=512,
            # Custom params pour WeDLM
            wedlm_window_size=window_size,
            wedlm_entropy_threshold=entropy_threshold,
            wedlm_distance_penalty=distance_penalty,
        )
        
        outputs = self.llm.generate(prompts, sampling_params)
        
        return [output.outputs[0].text for output in outputs]
```

### 8.3 Intégration avec MagiAttention

Pour les masques d'attention irréguliers du Dual-Stream Masking :

```python
# MagiAttention pour masques non-rectangulaires efficaces
# https://github.com/SandAI-org/MagiAttention/

from magi_attention import MagiAttention


class WeDLMAttention(nn.Module):
    """Attention WeDLM avec MagiAttention pour efficacité."""
    
    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.o_proj = nn.Linear(hidden_size, hidden_size)
        
        self.magi_attn = MagiAttention()
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,  # [seq, seq] irregular mask
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        B, L, _ = hidden_states.shape
        
        Q = self.q_proj(hidden_states).view(B, L, self.num_heads, self.head_dim)
        K = self.k_proj(hidden_states).view(B, L, self.num_heads, self.head_dim)
        V = self.v_proj(hidden_states).view(B, L, self.num_heads, self.head_dim)
        
        # Apply RoPE with custom position_ids
        Q, K = apply_rotary_pos_emb(Q, K, position_ids)
        
        # MagiAttention handles irregular masks efficiently
        output = self.magi_attn(Q, K, V, attention_mask)
        
        output = output.view(B, L, self.hidden_size)
        return self.o_proj(output)
```

---

## 9. Optimisations et Performance

### 9.1 Hyperparamètres Recommandés

| Paramètre | Plage | Défaut | Notes |
|-----------|-------|--------|-------|
| `window_size` (W) | 4-16 | 6 | Plus grand = plus de parallélisme, mais plus de mémoire |
| `entropy_threshold` (τ) | 0.2-0.8 | 0.5 | Plus bas = meilleure qualité, moins de speedup |
| `distance_penalty` (λ) | 0.01-0.2 | 0.1 | Plus haut = plus left-to-right, meilleur p_cache |
| `block_size` (B) | 4-64 | 32 | Pour l'entraînement; plus grand = plus flexible |
| `temperature` | 0.1-1.0 | 1.0 | Standard pour sampling |

### 9.2 Speedup Attendu par Type de Tâche

```
SPEEDUP PAR ENTROPIE DE SORTIE
═══════════════════════════════

Entropie Basse (patterns prévisibles):
├── Comptage, listes structurées
├── Code boilerplate
└── Speedup: 5-10×

Entropie Moyenne (raisonnement structuré):
├── Math step-by-step
├── Code avec logique
└── Speedup: 2-4×

Entropie Haute (génération ouverte):
├── Creative writing
├── Explanations complexes
└── Speedup: 1.2-2×
```

### 9.3 Profiling et Métriques

```python
class WeDLMProfiler:
    """Profiler pour analyser les performances WeDLM."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.n_generated = 0
        self.n_forward_tokens = 0
        self.n_forwards = 0
        self.n_committed_per_step = []
        self.entropies = []
        self.times = []
    
    def log_step(
        self,
        n_committed: int,
        window_size: int,
        entropies: List[float],
        step_time: float
    ):
        self.n_generated += n_committed
        self.n_forward_tokens += window_size
        self.n_forwards += 1
        self.n_committed_per_step.append(n_committed)
        self.entropies.extend(entropies)
        self.times.append(step_time)
    
    def get_metrics(self) -> dict:
        return {
            'p_cache': self.n_generated / max(1, self.n_forward_tokens),
            'avg_commit_per_step': sum(self.n_committed_per_step) / max(1, len(self.n_committed_per_step)),
            'total_tokens': self.n_generated,
            'total_forwards': self.n_forwards,
            'avg_entropy': sum(self.entropies) / max(1, len(self.entropies)),
            'tokens_per_second': self.n_generated / max(0.001, sum(self.times)),
            'recomputation_factor': self.n_forward_tokens / max(1, self.n_generated),
        }
    
    def print_report(self):
        metrics = self.get_metrics()
        print("\n" + "="*50)
        print("WeDLM PERFORMANCE REPORT")
        print("="*50)
        print(f"Tokens generated:      {metrics['total_tokens']}")
        print(f"Forward passes:        {metrics['total_forwards']}")
        print(f"p_cache:               {metrics['p_cache']:.2%}")
        print(f"Recomputation factor:  {metrics['recomputation_factor']:.2f}×")
        print(f"Avg commit/step:       {metrics['avg_commit_per_step']:.2f}")
        print(f"Avg entropy:           {metrics['avg_entropy']:.3f}")
        print(f"Throughput:            {metrics['tokens_per_second']:.1f} tok/s")
        print("="*50)
```

### 9.4 Optimisations Mémoire

```python
# 1. Gradient Checkpointing pour l'entraînement
model.gradient_checkpointing_enable()

# 2. Mixed Precision
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
with autocast(dtype=torch.bfloat16):
    loss = train_step(batch)
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()

# 3. KV Cache Efficient
# Pré-allouer le cache pour éviter les réallocations
class PreallocatedKVCache:
    def __init__(self, max_length: int, num_layers: int, num_heads: int, head_dim: int):
        self.max_length = max_length
        self.k_cache = torch.zeros(num_layers, 1, num_heads, max_length, head_dim)
        self.v_cache = torch.zeros(num_layers, 1, num_heads, max_length, head_dim)
        self.length = 0
    
    def append(self, k: torch.Tensor, v: torch.Tensor):
        n = k.shape[2]  # sequence length to add
        self.k_cache[:, :, :, self.length:self.length+n] = k
        self.v_cache[:, :, :, self.length:self.length+n] = v
        self.length += n
    
    def get(self):
        return (
            self.k_cache[:, :, :, :self.length],
            self.v_cache[:, :, :, :self.length]
        )
```

---

## 10. API Reference

### 10.1 Core Classes

#### `TopologicalReordering`

```python
def topological_reordering(
    tokens: torch.Tensor,
    mask_positions: torch.Tensor,
    mask_token_id: int,
) -> TopologicalReorderingResult:
    """
    Réordonne les tokens pour placer les observés avant les masqués.
    
    Parameters
    ----------
    tokens : torch.Tensor
        Shape [batch, seq_len]. Token IDs de la séquence.
    mask_positions : torch.Tensor
        Shape [batch, seq_len]. Boolean, True pour les positions masquées.
    mask_token_id : int
        ID du token [MASK].
    
    Returns
    -------
    TopologicalReorderingResult
        - reordered_tokens: [batch, seq_len] - Séquence réordonnée
        - reordered_positions: [batch, seq_len] - Positions logiques
        - observed_mask: [batch, seq_len] - True pour observés après reorder
        - inverse_permutation: [batch, seq_len] - Pour restaurer l'ordre
    
    Examples
    --------
    >>> tokens = torch.tensor([[1, 2, 0, 4, 0, 6]])  # 0 = mask
    >>> mask_pos = torch.tensor([[False, False, True, False, True, False]])
    >>> result = topological_reordering(tokens, mask_pos, mask_token_id=0)
    >>> result.reordered_tokens
    tensor([[1, 2, 4, 6, 0, 0]])
    >>> result.reordered_positions
    tensor([[0, 1, 3, 5, 2, 4]])
    """
```

#### `DualStreamMasking`

```python
class DualStreamMasking(nn.Module):
    """
    Module pour préparer les données d'entraînement dual-stream.
    
    Parameters
    ----------
    block_size : int, default=32
        Taille des blocs de prédiction.
    min_mask_ratio : float, default=0.1
        Ratio minimum de masquage par bloc.
    max_mask_ratio : float, default=0.9
        Ratio maximum de masquage par bloc.
    mask_token_id : int, default=32000
        ID du token [MASK].
    
    Methods
    -------
    forward(input_ids: torch.Tensor) -> Dict[str, torch.Tensor]
        Prépare un batch pour l'entraînement.
    
    Examples
    --------
    >>> ds = DualStreamMasking(block_size=4)
    >>> batch = torch.randint(0, 1000, (2, 8))
    >>> output = ds(batch)
    >>> output['input_ids'].shape
    torch.Size([2, 16])  # 2 * seq_len
    """
```

#### `StreamingDecoder`

```python
class StreamingDecoder:
    """
    Décodeur streaming parallèle pour l'inférence WeDLM.
    
    Parameters
    ----------
    model : nn.Module
        Modèle de langage causal (HuggingFace compatible).
    config : DecodingConfig
        Configuration du décodage.
    
    Methods
    -------
    generate(prompt_ids: torch.Tensor, temperature: float = 1.0) -> torch.Tensor
        Génère une séquence complète.
    
    generate_streaming(prompt_ids: torch.Tensor, ...) -> Iterator[str]
        Génère en mode streaming (token par token côté utilisateur).
    
    Examples
    --------
    >>> config = DecodingConfig(window_size=6, entropy_threshold=0.5)
    >>> decoder = StreamingDecoder(model, config)
    >>> output = decoder.generate(prompt_ids)
    """
```

### 10.2 Configuration Classes

```python
@dataclass
class WeDLMConfig:
    """Configuration complète WeDLM."""
    
    # Model
    base_model: str = "Qwen/Qwen2.5-7B"
    mask_token_id: int = 32000
    
    # Training
    block_size: int = 32
    min_mask_ratio: float = 0.1
    max_mask_ratio: float = 0.9
    
    # Inference  
    window_size: int = 6
    entropy_threshold: float = 0.5
    distance_penalty: float = 0.1
    max_new_tokens: int = 512


@dataclass
class DecodingConfig:
    """Configuration pour l'inférence uniquement."""
    
    window_size: int = 6
    entropy_threshold: float = 0.5
    distance_penalty: float = 0.1
    mask_token_id: int = 32000
    eos_token_id: int = 2
    max_new_tokens: int = 512
    temperature: float = 1.0
```

### 10.3 Loss Functions

```python
def wedlm_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    weights: torch.Tensor,
    ignore_index: int = -100,
) -> torch.Tensor:
    """
    Calcule la perte WeDLM avec pondération par bloc.
    
    Parameters
    ----------
    logits : torch.Tensor
        Shape [batch, seq_len, vocab_size]. Prédictions du modèle.
    targets : torch.Tensor
        Shape [batch, seq_len]. Tokens cibles.
    weights : torch.Tensor
        Shape [batch, seq_len]. Poids 1/γ_k pour chaque position.
    ignore_index : int, default=-100
        Index à ignorer dans le calcul.
    
    Returns
    -------
    torch.Tensor
        Scalaire, perte moyenne pondérée.
    """
```

---

## 11. Troubleshooting

### 11.1 Problèmes Fréquents

#### Erreur: "Position IDs mismatch"

```
RuntimeError: position_ids shape [1, 16] doesn't match input_ids shape [1, 8]
```

**Cause** : Les position_ids ne correspondent pas à la taille de l'input.

**Solution** :
```python
# Vérifier que position_ids a la même longueur que input_ids
assert position_ids.shape == input_ids.shape, \
    f"Shape mismatch: {position_ids.shape} vs {input_ids.shape}"
```

#### Erreur: "KV cache size mismatch"

```
RuntimeError: Cached key/value has length 100 but expected 106
```

**Cause** : Le cache KV n'est pas correctement mis à jour lors du commit.

**Solution** :
```python
# S'assurer que seuls les tokens committed sont ajoutés au cache
def update_cache_correctly(past_kv, new_kv, n_commit):
    # Ne prendre QUE les n_commit premiers tokens du nouveau KV
    for layer_idx in range(len(past_kv)):
        k_new = new_kv[layer_idx][0][:, :, :n_commit, :]
        v_new = new_kv[layer_idx][1][:, :, :n_commit, :]
        # Concatenate avec l'ancien cache
        ...
```

#### Performance dégradée

**Symptômes** : Speedup < 1.5× même sur tâches structurées.

**Diagnostic** :
```python
profiler = WeDLMProfiler()
# ... run generation with profiling ...
profiler.print_report()

# Vérifier p_cache
if profiler.get_metrics()['p_cache'] < 0.5:
    print("⚠️ p_cache faible - vérifier:")
    print("  1. Distance penalty trop faible?")
    print("  2. Entropy threshold trop haut?")
    print("  3. Beaucoup de résolutions out-of-order?")
```

**Solutions** :
1. Augmenter `distance_penalty` (0.1 → 0.15)
2. Baisser `entropy_threshold` (0.5 → 0.3)
3. Vérifier que l'attention est bien causale

#### Qualité dégradée

**Symptômes** : Scores benchmarks en baisse significative.

**Diagnostic** :
```python
# Comparer avec décodage AR standard
ar_output = model.generate(prompt, max_new_tokens=100)  # Standard
wedlm_output = decoder.generate(prompt)  # WeDLM

# Vérifier si les outputs divergent significativement
```

**Solutions** :
1. Baisser `entropy_threshold` pour plus de qualité
2. Vérifier que le modèle a bien été entraîné avec Dual-Stream
3. Augmenter `window_size` si truncation des contextes

### 11.2 FAQ

**Q: Puis-je utiliser WeDLM avec n'importe quel LLM?**

R: WeDLM nécessite un fine-tuning spécifique (Dual-Stream Masking). Un modèle AR standard ne fonctionnera pas directement. Cependant, le fine-tuning est relativement léger (~100B tokens selon le paper).

**Q: Quelle est la différence avec speculative decoding?**

R: Le speculative decoding utilise un modèle draft plus petit. WeDLM prédit directement plusieurs tokens avec le même modèle via diffusion. Les deux sont complémentaires.

**Q: WeDLM fonctionne-t-il en mode streaming (affichage progressif)?**

R: Oui! Les tokens sont committés progressivement. Cependant, ils arrivent par "bursts" plutôt que un par un, ce qui peut donner une UX différente.

**Q: Comment choisir window_size?**

R: Plus grand = plus de parallélisme potentiel mais plus de mémoire et risque de résolutions out-of-order. Recommandation: 4-8 pour qualité, 8-16 pour vitesse maximale.

---

## Annexes

### A. Notation Mathématique

| Symbole | Signification |
|---------|--------------|
| x₀ | Séquence clean originale |
| x_γ | Séquence avec γL positions masquées |
| M | Ensemble des indices masqués |
| O | Ensemble des indices observés |
| γ | Ratio de masquage |
| τ | Seuil d'entropie |
| λ | Coefficient de distance penalty |
| W | Taille de la fenêtre glissante |
| B | Taille de bloc (entraînement) |
| p_cache | Prefix cacheability |
| H̃ | Entropie ajustée par distance |

### B. Références

1. **WeDLM Paper**: Liu et al., "WeDLM: Reconciling Diffusion Language Models with Standard Causal Attention for Fast Inference"
2. **LLaDA**: Nie et al., "Large Language Diffusion Models"
3. **Dream**: Ye et al., "Dream 7B: Diffusion Large Language Models"
4. **FlashAttention**: Dao et al., "FlashAttention: Fast and Memory-Efficient Exact Attention"
5. **vLLM**: Kwon et al., "Efficient Memory Management for Large Language Model Serving with PagedAttention"

### C. Changelog

- **v1.0** (Dec 2024): Documentation initiale

---

*Documentation générée pour l'équipe de développement. Pour questions: [contact]*