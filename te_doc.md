# NVIDIA Transformer Engine - API PyTorch

Documentation complète de l'API PyTorch pour Transformer Engine, la bibliothèque NVIDIA pour l'entraînement et l'inférence de modèles Transformer avec support FP8.

---

## Table des matières

1. [Modules de base](#1-modules-de-base)
   - [Linear](#11-linear)
   - [GroupedLinear](#12-groupedlinear)
   - [LayerNorm](#13-layernorm)
   - [RMSNorm](#14-rmsnorm)
2. [Modules fusionnés](#2-modules-fusionnés)
   - [LayerNormLinear](#21-layernormlinear)
   - [LayerNormMLP](#22-layernormmlp)
3. [Modules d'Attention](#3-modules-dattention)
   - [DotProductAttention](#31-dotproductattention)
   - [MultiheadAttention](#32-multiheadattention)
4. [TransformerLayer](#4-transformerlayer)
5. [Gestion des états RNG](#5-gestion-des-états-rng)
6. [FP8 - Entraînement en précision mixte](#6-fp8---entraînement-en-précision-mixte)
   - [fp8_autocast](#61-fp8_autocast)
   - [fp8_model_init](#62-fp8_model_init)
7. [Utilitaires](#7-utilitaires)
   - [checkpoint](#71-checkpoint)
   - [make_graphed_callables](#72-make_graphed_callables)
   - [get_cpu_offload_context](#73-get_cpu_offload_context)
8. [MoE (Mixture of Experts)](#8-moe-mixture-of-experts)
9. [Userbuffers](#9-userbuffers)

---

## 1. Modules de base

### 1.1 Linear

```python
class transformer_engine.pytorch.Linear(in_features, out_features, bias=True, **kwargs)
```

Applique une transformation linéaire aux données entrantes : **y = xAᵀ + b**

Remplacement direct de `torch.nn.Linear` optimisé pour les GPUs NVIDIA.

#### Paramètres principaux

| Paramètre | Type | Défaut | Description |
|-----------|------|--------|-------------|
| `in_features` | int | - | Taille de chaque échantillon d'entrée |
| `out_features` | int | - | Taille de chaque échantillon de sortie |
| `bias` | bool | True | Si False, pas de biais appris |
| `init_method` | Callable | None | Méthode d'initialisation des poids. Défaut: `normal_(mean=0.0, std=0.023)` |
| `device` | Union[torch.device, str] | "cuda" | Device d'allocation des paramètres |
| `name` | str | None | Nom du module (debug) |

#### Paramètres de parallélisme

| Paramètre | Type | Défaut | Description |
|-----------|------|--------|-------------|
| `sequence_parallel` | bool | False | Active le parallélisme de séquence |
| `tp_group` | ProcessGroup | None | Groupe de processus tensor parallel |
| `tp_size` | int | 1 | Taille du monde tensor parallel |
| `parallel_mode` | {None, 'column', 'row'} | None | Mode de parallélisme (cf. [Megatron-LM](https://arxiv.org/pdf/1909.08053.pdf)) |

#### Paramètres d'optimisation

| Paramètre | Type | Défaut | Description |
|-----------|------|--------|-------------|
| `fuse_wgrad_accumulation` | bool | False | Fusionne création et accumulation du gradient des poids |
| `return_bias` | bool | False | Retourne le biais séparément (pour fusion avec ops suivantes) |
| `params_dtype` | torch.dtype | default dtype | Type des paramètres initiaux |
| `delay_wgrad_compute` | bool | False | Délai du calcul du gradient des poids |
| `symmetric_ar_type` | {None, 'multimem_all_reduce', 'two_shot', 'one_shot'} | None | Type d'all-reduce symétrique (PyTorch ≥ 2.7.0) |

#### Méthodes

```python
def forward(inp, is_first_microbatch=None, fp8_output=False, fp8_grad=False)
```

- `is_first_microbatch` : Optimisations pour le premier microbatch (cache FP8, skip accumulation)
- `fp8_output` / `fp8_grad` : Force la sortie/gradient en FP8

```python
def set_tensor_parallel_group(tp_group)
```
Configure le groupe tensor parallel avant le forward pass.

---

### 1.2 GroupedLinear

```python
class transformer_engine.pytorch.GroupedLinear(in_features, out_features, bias=True, **kwargs)
```

Applique des transformations linéaires groupées : **yᵢ = xᵢAᵢᵀ + bᵢ**

Utile pour les architectures MoE (Mixture of Experts).

#### Paramètres spécifiques

| Paramètre | Type | Description |
|-----------|------|-------------|
| `num_gemms` | int | Nombre de GEMMs exécutés simultanément |

#### Méthode forward

```python
def forward(inp, m_splits, is_first_microbatch=None)
```

- `m_splits` : Liste d'entiers représentant la division du tenseur d'entrée

> **Note** : GroupedLinear ne gère pas directement les communications TP. Les paramètres `tp_size` et `parallel_mode` déterminent les formes des poids/biais. La communication TP doit être gérée dans les phases dispatch/combine des modèles MoE.

---

### 1.3 LayerNorm

```python
class transformer_engine.pytorch.LayerNorm(hidden_size, eps=1e-5, **kwargs)
```

Normalisation de couche standard ([Layer Normalization](https://arxiv.org/abs/1607.06450)).

**Formule** :
$$y = \frac{x - E[x]}{\sqrt{Var[x] + \varepsilon}} \cdot \gamma + \beta$$

Avec `zero_centered_gamma=True` :
$$y = \frac{x - E[x]}{\sqrt{Var[x] + \varepsilon}} \cdot (1 + \gamma) + \beta$$

#### Paramètres

| Paramètre | Type | Défaut | Description |
|-----------|------|--------|-------------|
| `normalized_shape` | int ou iterable | - | Dimensions internes du tenseur |
| `eps` | float | 1e-5 | Stabilité numérique |
| `zero_centered_gamma` | bool | False | Initialise γ à 0 |
| `sm_margin` | int ou dict | 0 | SMs exclus pour overlap avec kernels de communication |

---

### 1.4 RMSNorm

```python
class transformer_engine.pytorch.RMSNorm(hidden_size, eps=1e-5, **kwargs)
```

Root Mean Square Layer Normalization ([paper](https://arxiv.org/abs/1910.07467)).

**Formule** :
$$y = \frac{x}{RMS_\varepsilon(x)} \cdot \gamma$$

où :
$$RMS_\varepsilon(x) = \sqrt{\frac{1}{n}\sum_{i=0}^n x_i^2 + \varepsilon}$$

Mêmes paramètres que LayerNorm (sans β).

---

## 2. Modules fusionnés

### 2.1 LayerNormLinear

```python
class transformer_engine.pytorch.LayerNormLinear(in_features, out_features, eps=1e-5, bias=True, **kwargs)
```

Applique la normalisation de couche suivie d'une transformation linéaire.

#### Paramètres spécifiques

| Paramètre | Type | Défaut | Description |
|-----------|------|--------|-------------|
| `normalization` | {'LayerNorm', 'RMSNorm'} | 'LayerNorm' | Type de normalisation |
| `return_layernorm_output` | bool | False | Retourne aussi la sortie du layernorm |
| `return_layernorm_output_gathered` | bool | False | Retourne la sortie après all-gather (sequence parallel) |
| `parameters_split` | Optional[Union[Tuple, Dict]] | None | Configuration pour diviser poids/biais en plusieurs paramètres |

---

### 2.2 LayerNormMLP

```python
class transformer_engine.pytorch.LayerNormMLP(hidden_size, ffn_hidden_size, eps=1e-5, bias=True, **kwargs)
```

Normalisation de couche suivie d'un MLP (2 transformations linéaires + activation).

#### Paramètres

| Paramètre | Type | Défaut | Description |
|-----------|------|--------|-------------|
| `hidden_size` | int | - | Taille d'entrée |
| `ffn_hidden_size` | int | - | Taille intermédiaire |
| `activation` | str | 'gelu' | Fonction d'activation |

#### Activations supportées

- `'gelu'`, `'geglu'`, `'qgelu'`, `'qgeglu'`
- `'relu'`, `'reglu'`
- `'srelu'`, `'sreglu'`
- `'silu'`, `'swiglu'`

#### Paramètres de parallélisme

| Paramètre | Type | Défaut | Description |
|-----------|------|--------|-------------|
| `set_parallel_mode` | bool | False | FC1 = Column Parallel, FC2 = Row Parallel |
| `sequence_parallel` | bool | False | Parallélisme de séquence |

---

## 3. Modules d'Attention

### 3.1 DotProductAttention

```python
class transformer_engine.pytorch.DotProductAttention(num_attention_heads, kv_channels, **kwargs)
```

Implémentation de l'attention "Scaled Dot-Product" ([Attention Is All You Need](https://arxiv.org/abs/1706.03762)).

#### Paramètres principaux

| Paramètre | Type | Défaut | Description |
|-----------|------|--------|-------------|
| `num_attention_heads` | int | - | Nombre de têtes d'attention |
| `kv_channels` | int ou Tuple[int,int] | - | Taille des têtes K/V |
| `num_gqa_groups` | int | None | Nombre de groupes GQA ([paper](https://arxiv.org/pdf/2305.13245.pdf)) |
| `attention_dropout` | float | 0.0 | Dropout sur l'attention |

#### Types de masques (`attn_mask_type`)

| Valeur | Description |
|--------|-------------|
| `'no_mask'` | Pas de masque |
| `'padding'` | Masque pour tokens de padding |
| `'causal'` | Masque causal (triangulaire supérieur) |
| `'padding_causal'` | Combinaison padding + causal |
| `'causal_bottom_right'` | Causal aligné en bas à droite (inférence/KV cache) |
| `'arbitrary'` | Masque arbitraire fourni par l'utilisateur |

#### Formats QKV (`qkv_format`)

| Format | Description |
|--------|-------------|
| `'sbhd'` | (sequence, batch, heads, head_dim) |
| `'bshd'` | (batch, sequence, heads, head_dim) |
| `'thd'` | (total_tokens, heads, head_dim) - séquences de longueurs variables |

#### Paramètres de Context Parallelism

| Paramètre | Type | Description |
|-----------|------|-------------|
| `cp_group` | ProcessGroup ou List | Groupe de processus CP |
| `cp_comm_type` | str | Type de communication : `'p2p'`, `'all_gather'`, `'a2a'`, `'a2a+p2p'` |
| `cp_stream` | CUDA stream | Stream pour overlap compute/communication |

#### Types de softmax (`softmax_type`)

| Type | Formule |
|------|---------|
| `'vanilla'` | Standard softmax |
| `'off-by-one'` | exp(S) / (1 + Σexp(S)) - "zero sink" |
| `'learnable'` | exp(S) / (exp(α) + Σexp(S)) - "learnable sink" |

#### Backends d'attention

1. **FlashAttention** : [flash-attn](https://arxiv.org/pdf/2305.13245.pdf)
2. **FusedAttention** : Basé sur [cuDNN Graph API](https://docs.nvidia.com/deeplearning/cudnn/developer-guide/index.html#op-fusion)
3. **UnfusedDotProductAttention** : Implémentation PyTorch native

**Variables d'environnement** :
- `NVTE_FLASH_ATTN` : Active/désactive FlashAttention
- `NVTE_FUSED_ATTN` : Active/désactive FusedAttention
- `NVTE_FUSED_ATTN_BACKEND` : Sélectionne le backend FusedAttention
- `NVTE_ALLOW_NONDETERMINISTIC_ALGO=0` : Force le déterminisme (FlashAttention ≥ 2.4.1)

---

### 3.2 MultiheadAttention

```python
class transformer_engine.pytorch.MultiheadAttention(hidden_size, num_attention_heads, **kwargs)
```

Multi-Head Attention complète incluant les projections Q, K, V et Output.

#### Paramètres spécifiques

| Paramètre | Type | Défaut | Description |
|-----------|------|--------|-------------|
| `kv_channels` | int | None | Canaux K/V (défaut: hidden_size / num_heads) |
| `attention_type` | {'self', 'cross'} | 'self' | Type d'attention |
| `input_layernorm` | bool | False | Applique layernorm à l'entrée |
| `fuse_qkv_params` | bool | False | Fusionne les paramètres QKV |
| `qkv_weight_interleaved` | bool | True | Poids QKV entrelacés par tête |

#### QK Normalization

| Paramètre | Type | Défaut | Description |
|-----------|------|--------|-------------|
| `qk_norm_type` | str | None | Type: `None`, `'L2Normalization'`, `'RMSNorm'`, `'LayerNorm'` |
| `qk_norm_eps` | float | 1e-6 | Epsilon pour la normalisation |
| `qk_norm_before_rope` | bool | False | Normalisation avant RoPE |

---

## 4. TransformerLayer

```python
class transformer_engine.pytorch.TransformerLayer(hidden_size, ffn_hidden_size, num_attention_heads, **kwargs)
```

Couche Transformer complète : bloc d'attention + réseau feedforward (MLP).

#### Paramètres architecturaux

| Paramètre | Type | Défaut | Description |
|-----------|------|--------|-------------|
| `layer_type` | {'encoder', 'decoder'} | 'encoder' | Type de couche |
| `parallel_attention_mlp` | bool | False | Attention et MLP en parallèle (architecture Falcon) |
| `output_layernorm` | bool | False | LayerNorm après la couche |
| `apply_residual_connection_post_layernorm` | bool | False | Résiduel depuis la sortie du layernorm |

#### Paramètres de dropout

| Paramètre | Type | Défaut |
|-----------|------|--------|
| `hidden_dropout` | float | 0.1 |
| `attention_dropout` | float | 0.1 |
| `drop_path_rate` | float | 0.0 |

#### Exemple d'utilisation

```python
import transformer_engine.pytorch as te

layer = te.TransformerLayer(
    hidden_size=1024,
    ffn_hidden_size=4096,
    num_attention_heads=16,
    num_gqa_groups=4,  # Grouped Query Attention
    activation='swiglu',
    normalization='RMSNorm',
)

output = layer(
    hidden_states,
    attention_mask=mask,
    rotary_pos_emb=rope_emb,
)
```

---

## 5. Gestion des états RNG

```python
class transformer_engine.pytorch.CudaRNGStatesTracker
```

Gestionnaire d'états RNG multiples pour le parallélisme de modèle.

#### Méthodes

```python
# Ajouter un état RNG
tracker.add(name: str, seed: int)

# Fork temporaire vers un état RNG
with tracker.fork(name='model-parallel-rng'):
    # Opérations avec cet état RNG
    pass

# Sauvegarder/restaurer les états
states = tracker.get_states()
tracker.set_states(states)

# Réinitialiser
tracker.reset()
```

---

## 6. FP8 - Entraînement en précision mixte

### 6.1 fp8_autocast

```python
@contextmanager
transformer_engine.pytorch.fp8_autocast(
    enabled=True,
    calibrating=False,
    fp8_recipe=None,
    fp8_group=None,
    _graph=False
)
```

Context manager pour l'utilisation du FP8.

#### Paramètres

| Paramètre | Type | Défaut | Description |
|-----------|------|--------|-------------|
| `enabled` | bool | True | Active le FP8 |
| `calibrating` | bool | False | Mode calibration (collecte amax/scale sans FP8) |
| `fp8_recipe` | recipe.Recipe | None | Recette FP8 |
| `fp8_group` | ProcessGroup | None | Groupe pour réduction des amax |

#### Exemple

```python
from transformer_engine.pytorch import fp8_autocast

with fp8_autocast(enabled=True, fp8_recipe=recipe):
    output = model(input)
```

> **Note** : Les dimensions des tenseurs doivent être divisibles par 16.

> **Warning** : Avec `fp8_recipe.reduce_amax=True`, un module ne doit pas être appelé plus d'une fois dans une région `fp8_autocast`.

---

### 6.2 fp8_model_init

```python
@contextmanager
transformer_engine.pytorch.fp8_model_init(
    enabled=True,
    recipe=None,
    preserve_high_precision_init_val=False
)
```

Context manager pour l'initialisation des paramètres en FP8.

#### Cas d'usage

- **Entraînement complet** avec optimiseur à master weights
- **Inférence** (seules les copies FP8 sont nécessaires)
- **Fine-tuning LoRA** (paramètres principaux inchangés)

#### Exemple

```python
from transformer_engine.pytorch import fp8_model_init

# Initialisation FP8 avec préservation haute précision
with fp8_model_init(enabled=True, preserve_high_precision_init_val=True):
    model = te.Linear(768, 768)

# Récupérer la valeur haute précision pour master weights
master_weight = model.weight.get_high_precision_init_val()
model.weight.clear_high_precision_init_val()  # Libérer la mémoire CPU
```

---

## 7. Utilitaires

### 7.1 checkpoint

```python
transformer_engine.pytorch.checkpoint(
    function: Callable,
    *args,
    **kwargs
) -> Tuple[torch.Tensor, ...]
```

Checkpointing d'activation (trade compute ↔ memory).

#### Paramètres

| Paramètre | Type | Défaut | Description |
|-----------|------|--------|-------------|
| `distribute_saved_activations` | bool | False | Distribue le premier tensor sur le groupe TP |
| `get_rng_state_tracker` | Callable | None | Retourne une instance CudaRNGStatesTracker |
| `tp_group` | ProcessGroup | None | Groupe tensor parallel |
| `use_reentrant` | bool | True | Mode réentrant (True) ou non-réentrant (False) |

---

### 7.2 make_graphed_callables

```python
transformer_engine.pytorch.make_graphed_callables(
    modules,
    sample_args,
    num_warmup_iters=3,
    allow_unused_input=False,
    sample_kwargs=None,
    fp8_enabled=False,
    fp8_recipe=None,
    fp8_weight_caching=False,
    ...
) -> Callable | Tuple[Callable, ...]
```

Crée des versions CUDA Graph des modules TE avec support FP8.

#### Configuration des GEMMs (`ub_cfgs`)

```python
{
    "<gemm_name>": {
        "method": "ring_exchange" | "pipeline",
        "is_reduce_scatter": bool,
        "num_sm": int,
        "cga_size": int,
        "set_sm_margin": bool,
        "num_splits": int,
        "aggregate": bool,
        "atomic_gemm": bool,
        "use_ce": bool,
        "fp8_buf": bool,
    }
}
```

Noms de GEMM disponibles : `qkv_fprop`, `qkv_dgrad`, `qkv_wgrad`, `proj_fprop`, `proj_dgrad`, `proj_wgrad`, `fc1_fprop`, `fc1_dgrad`, `fc2_dgrad`, `fc2_fprop`, `fc2_wgrad`

---

### 7.3 get_cpu_offload_context

```python
transformer_engine.pytorch.get_cpu_offload_context(
    enabled=False,
    num_layers=1,
    model_layers=1,
    offload_activations=True,
    offload_weights=False,
    double_buffering=False
)
```

Offloading CPU des activations et poids.

#### Exemple

```python
cpu_offload_context, cpu_offload_synchronizer = get_cpu_offload_context(
    enabled=True,
    num_layers=12,
    offload_activations=True,
)

for layer in transformer_layers:
    with cpu_offload_context:
        output = layer(input)
    cpu_offload_synchronizer()
```

---

## 8. MoE (Mixture of Experts)

### moe_permute

```python
transformer_engine.pytorch.moe_permute(
    inp: torch.Tensor,
    routing_map: torch.Tensor,
    num_out_tokens: int = -1,
    max_token_num: int = -1,
    map_type: str = 'mask'
) -> Tuple[torch.Tensor, torch.Tensor]
```

Permute les tokens selon la routing map. Les tokens routés vers le même expert sont groupés ensemble.

| Paramètre | Description |
|-----------|-------------|
| `inp` | Tensor [num_tokens, hidden_size] |
| `routing_map` | Si `'mask'`: [num_tokens, num_experts], Si `'index'`: [num_tokens, topK] |
| `num_out_tokens` | Nombre de tokens effectifs (-1 = aucun drop) |
| `map_type` | `'mask'` ou `'index'` |

### moe_permute_with_probs

Variante qui permute également les probabilités associées aux tokens.

### moe_unpermute

```python
transformer_engine.pytorch.moe_unpermute(
    inp: torch.Tensor,
    row_id_map: torch.Tensor,
    merging_probs: torch.Tensor = None,
    restore_shape: torch.Size = None,
    map_type: str = 'mask'
) -> torch.Tensor
```

Inverse de `moe_permute`, avec fusion optionnelle par probabilités.

### moe_sort_chunks_by_index

```python
transformer_engine.pytorch.moe_sort_chunks_by_index(
    inp: torch.Tensor,
    split_sizes: torch.Tensor,
    sorted_index: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]
```

Split et trie le tenseur d'entrée selon les indices.

---

## 9. Userbuffers

### initialize_ub

```python
transformer_engine.pytorch.initialize_ub(
    shape: list,
    tp_size: int,
    use_fp8: bool = False,
    quantization_modes: List[UserBufferQuantizationMode] = None,
    dtype: torch.dtype = torch.bfloat16,
    ub_cfgs: dict = None,
    bootstrap_backend: str = None
)
```

Initialise le communicateur Userbuffers pour l'overlap des communications TP avec les GEMMs.

#### Paramètres

| Paramètre | Type | Description |
|-----------|------|-------------|
| `shape` | list | Forme du buffer de communication : `[seq_len * batch_size, hidden_size]` |
| `tp_size` | int | Nombre de GPUs dans le groupe TP |
| `quantization_modes` | List | Modes de quantification (remplace `use_fp8`) |
| `dtype` | torch.dtype | Type de données non-FP8 |
| `bootstrap_backend` | str | Backend de bootstrap : préfère MPI, puis Gloo, puis NCCL |

### destroy_ub

```python
transformer_engine.pytorch.destroy_ub()
```

Détruit tous les communicateurs Userbuffers alloués.

---

## Ressources additionnelles

- [Documentation officielle NVIDIA Transformer Engine](https://docs.nvidia.com/deeplearning/transformer-engine/)
- [GitHub Transformer Engine](https://github.com/NVIDIA/TransformerEngine)
- [Paper Megatron-LM](https://arxiv.org/pdf/1909.08053.pdf) - Parallélisme tensor
- [Paper FlashAttention](https://arxiv.org/pdf/2305.13245.pdf)
- [Paper Grouped Query Attention](https://arxiv.org/pdf/2305.13245.pdf)

---

*Généré à partir de la documentation NVIDIA Transformer Engine v1.x*