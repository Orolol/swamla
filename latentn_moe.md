
# LatentMoE : Architecture Hardware-Aware pour l'Amélioration de la Précision par Octet

## Introduction

LatentMoE est une architecture novatrice introduite par NVIDIA dans la famille de modèles Nemotron 3 (Super et Ultra). Elle vise à améliorer la qualité du modèle sans compromettre le débit d'inférence ni la latence, en repensant fondamentalement la conception des couches Mixture-of-Experts.

## Contexte et Motivation

### Goulots d'étranglement selon le mode de déploiement

Les couches MoE traditionnelles font face à des bottlenecks différents selon le contexte de déploiement :

**Déploiement orienté latence** (dizaines à centaines de tokens par itération)
- Le calcul MoE est limité par la bande passante mémoire
- La lecture des poids experts depuis la mémoire domine le coût
- Chaque matrice expert a une taille `d × m` (dimension cachée × dimension intermédiaire FFN)
- Réduire `d` ou `m` diminue les coûts de bande passante

**Déploiement orienté débit** (milliers de tokens par itération)
- La communication all-to-all pour dispatcher les tokens aux experts devient le goulot principal
- Le volume de communication scale linéairement avec :
  - Le nombre d'experts actifs top-K
  - La dimension cachée `d`
- Indépendant de la dimension intermédiaire `m`

### Insight clé

La puissance expressive des couches FFN est principalement contrôlée par le **budget non-linéaire effectif**, approximativement proportionnel à `K × m` (experts actifs × dimension intermédiaire).

## Architecture LatentMoE

### Principe fondamental

L'idée centrale est de **réduire la dimension d'entrée routée** `d` vers une dimension latente plus petite `ℓ`, puis de **réinvestir les économies** dans l'augmentation du budget non-linéaire et de la diversité des experts.

### Pipeline de traitement

```
┌─────────────────────────────────────────────────────────────────┐
│                      Standard MoE                                │
│                                                                  │
│   Token (d) ──► Router ──► Expert₁...Expert_N (d×m) ──► Output  │
│                   │                                              │
│              Top-K routing                                       │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                      LatentMoE                                   │
│                                                                  │
│   Token (d) ──► Proj_down ──► Latent (ℓ) ──► Router             │
│                                    │                             │
│                               Top-K' routing                     │
│                                    │                             │
│                    Expert₁...Expert_N' (ℓ×m) ──► Proj_up (d)    │
└─────────────────────────────────────────────────────────────────┘
```

### Étapes détaillées

1. **Projection vers l'espace latent** : Chaque embedding de token est projeté de la dimension cachée originale `d` vers une représentation latente de dimension `ℓ < d`

2. **Routage dans l'espace latent** : Les tokens sont routés vers un ensemble élargi d'experts qui opèrent entièrement dans cet espace latent

3. **Projection de retour** : Les sorties sont projetées de retour vers la dimension cachée originale `d`

### Facteurs de scaling

Le ratio `d/ℓ` (typiquement ~4×) détermine les économies et réinvestissements :

| Paramètre | Standard MoE | LatentMoE | Facteur |
|-----------|--------------|-----------|---------|
| Dimension routée | d | ℓ | ÷ (d/ℓ) |
| Nombre total d'experts | N | N' = N × (d/ℓ) | × (d/ℓ) |
| Experts actifs (top-K) | K | K' = K × (d/ℓ) | × (d/ℓ) |
| Charge poids par expert | d × m | ℓ × m | ÷ (d/ℓ) |
| Trafic all-to-all | O(K × d) | O(K' × ℓ) | ~constant |

## Préservation de la qualité

### Opérations maintenues en haute dimension

Pour préserver la qualité du modèle, certaines opérations non-routées restent dans la dimension originale `d` :

- **Gate de routage MoE** (réseau de gating)
- **Calcul des experts partagés**
- **Couches non-experts** (attention, Mamba, etc.)

Ces composants ne contribuent pas significativement aux goulots d'étranglement ciblés.

## Configuration expérimentale

### Comparaison des architectures

| Configuration | Standard MoE | LatentMoE |
|---------------|--------------|-----------|
| Paramètres actifs | 8.09B | 8.02B |
| Paramètres totaux | 72.6B | 72.8B |
| Dimension cachée (d) | 4096 | 4096 |
| Dimension latente (ℓ) | - | 1024 |
| Ratio d/ℓ | - | 4× |
| Experts totaux | 128 | 512 |
| Experts actifs (K) | 6 | 22 |
| Tokens d'entraînement | 1T | 1T |

### Résultats de performance

Les deux modèles ont été entraînés avec des hyperparamètres identiques sur 1 trillion de tokens :

| Benchmark | Standard MoE | LatentMoE | Amélioration |
|-----------|--------------|-----------|--------------|
| MMLU-Pro | 48.30% | 52.87% | +4.57 pts |
| MMLU | 70.10% | 72.11% | +2.01 pts |
| Code (agrégé) | 51.95% | 55.14% | +3.19 pts |
| Math (agrégé) | 78.32% | 80.19% | +1.87 pts |
| Commonsense | 81.73% | 82.10% | +0.37 pts |

**Note sur les agrégations :**
- **Code** : moyenne de HumanEval, HumanEval+, MBPP, MBPP+
- **Math** : moyenne de GSM8K CoT et MATH-500
- **Commonsense** : moyenne de RACE, ARC-Challenge, HellaSwag, Winogrande

## Analyse des bénéfices

### Efficacité mémoire

La réduction de la dimension routée de `d` à `ℓ` diminue :
- Les charges de poids par expert d'un facteur `d/ℓ`
- Le payload de communication all-to-all d'un facteur `d/ℓ`

### Diversité des experts

L'augmentation du nombre d'experts de `N` à `N' = N × (d/ℓ)` et des experts actifs de `K` à `K' = K × (d/ℓ)` permet :
- Une spécialisation plus fine des experts
- Un meilleur coverage des patterns dans les données
- Un budget non-linéaire préservé voire augmenté

### Neutralité computationnelle

La réduction de dimensionnalité compense l'augmentation du nombre d'experts, maintenant un budget computationnel et de communication similaire au MoE standard.

## Intégration dans Nemotron 3

### Architecture hybride

LatentMoE s'intègre dans l'architecture hybride Mamba-Transformer MoE de Nemotron 3 :

```
[Mamba-2] ─► [LatentMoE] ─► [Mamba-2] ─► [LatentMoE] ─► ... ─► [Attention] ─► [LatentMoE]
     ×5              ×5           ×3            ×3                  ×1
```

### Combinaison avec d'autres techniques

Dans les modèles Super et Ultra, LatentMoE est combiné avec :

- **NVFP4 Training** : Les projections latentes sont maintenues en BF16 car leur impact sur le temps de step est minimal
- **Multi-Token Prediction (MTP)** : Les couches MTP restent en BF16 pour préserver leurs capacités
- **Long Context (1M tokens)** : Compatible avec l'extension de contexte sans RoPE

## Pseudo-code d'implémentation

```python
class LatentMoE(nn.Module):
    def __init__(
        self,
        hidden_dim: int,        # d = 4096
        latent_dim: int,        # ℓ = 1024
        num_experts: int,       # N' = 512
        top_k: int,             # K' = 22
        ffn_intermediate: int,  # m
    ):
        super().__init__()
        
        # Projections latentes
        self.down_proj = nn.Linear(hidden_dim, latent_dim)
        self.up_proj = nn.Linear(latent_dim, hidden_dim)
        
        # Router opère sur la dimension originale pour la qualité
        self.router = nn.Linear(hidden_dim, num_experts)
        
        # Experts opèrent dans l'espace latent
        self.experts = nn.ModuleList([
            FFNExpert(latent_dim, ffn_intermediate)
            for _ in range(num_experts)
        ])
        
        self.top_k = top_k
    
    def forward(self, x: Tensor) -> Tensor:
        # x: [batch, seq, hidden_dim]
        
        # 1. Calcul des scores de routage (dimension originale)
        router_logits = self.router(x)  # [batch, seq, num_experts]
        router_probs = F.softmax(router_logits, dim=-1)
        
        # 2. Sélection top-K
        top_k_probs, top_k_indices = torch.topk(
            router_probs, self.top_k, dim=-1
        )
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
        
        # 3. Projection vers l'espace latent
        x_latent = self.down_proj(x)  # [batch, seq, latent_dim]
        
        # 4. Dispatch aux experts (dans l'espace latent)
        expert_outputs = torch.zeros_like(x_latent)
        for i, expert in enumerate(self.experts):
            mask = (top_k_indices == i).any(dim=-1)
            if mask.any():
                expert_out = expert(x_latent[mask])
                # Pondération par probabilité de routage
                weight = top_k_probs[..., top_k_indices == i].sum(dim=-1)
                expert_outputs[mask] += weight[mask].unsqueeze(-1) * expert_out
        
        # 5. Projection de retour vers la dimension originale
        output = self.up_proj(expert_outputs)
        
        return output


class FFNExpert(nn.Module):
    """Expert FFN opérant dans l'espace latent"""
    def __init__(self, latent_dim: int, intermediate_dim: int):
        super().__init__()
        self.w1 = nn.Linear(latent_dim, intermediate_dim)
        self.w2 = nn.Linear(intermediate_dim, latent_dim)
        self.act = nn.SiLU()
    
    def forward(self, x: Tensor) -> Tensor:
        return self.w2(self.act(self.w1(x)))
```

## Considérations pratiques

### Choix du ratio d/ℓ

Le ratio de 4× utilisé par NVIDIA (4096 → 1024) représente un bon compromis :
- Suffisamment agressif pour des économies significatives
- Préserve assez de capacité représentationnelle dans l'espace latent
- Permet une multiplication par 4 du nombre d'experts actifs

### Trade-offs

| Aspect | Avantage | Inconvénient potentiel |
|--------|----------|------------------------|
| Latence | Réduction des charges mémoire | Overhead des projections |
| Débit | Réduction du trafic all-to-all | Complexité de routage accrue |
| Qualité | Plus d'experts = meilleure spécialisation | Compression de l'information |
| Mémoire | Experts plus petits | Plus d'experts à stocker |

## Conclusion

LatentMoE représente une avancée significative dans la conception des architectures MoE, démontrant qu'il est possible d'améliorer simultanément la qualité et l'efficacité en repensant les dimensions de travail. L'approche hardware-aware de compression vers un espace latent, compensée par l'augmentation de la diversité des experts, offre un nouveau point sur la frontière précision-efficacité pour les LLMs.

## Références

- NVIDIA Nemotron 3 White Paper (2025)
- DeepSeek-V3 Technical Report (2024)
- Waleffe et al., "An Empirical Study of Mamba-based Language Models" (2024)