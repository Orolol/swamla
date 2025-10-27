# SWA-MLA Inference Guide

Script d'inférence complet pour le modèle SWA-MLA, permettant la génération de texte en mode batch ou chat interactif.

## Installation

Assurez-vous d'avoir les dépendances requises :

```bash
pip install torch transformers huggingface_hub
```

## Utilisation

### Mode Batch

Génère des réponses pour des prompts pré-enregistrés :

```bash
# Utiliser les prompts par défaut
python inference.py --checkpoint monoswamla/checkpoint_20000.pt --mode batch

# Utiliser un fichier de prompts personnalisé
python inference.py --checkpoint monoswamla/checkpoint_20000.pt \
    --mode batch \
    --prompts_file my_prompts.txt \
    --max_new_tokens 100 \
    --temperature 0.8
```

**Format du fichier de prompts** : Un prompt par ligne dans un fichier texte.

Exemple (`my_prompts.txt`) :
```
Once upon a time
What is the meaning of life?
Explain quantum computing
```

### Mode Chat

Conversation interactive avec le modèle :

```bash
python inference.py --checkpoint monoswamla/checkpoint_20000.pt \
    --mode chat \
    --max_new_tokens 150 \
    --temperature 0.7
```

**Commandes disponibles en mode chat :**
- `/quit` ou `/exit` - Quitter
- `/clear` - Effacer l'historique de conversation
- `/temp <valeur>` - Changer la température (ex: `/temp 0.7`)
- `/topk <valeur>` - Changer top_k (ex: `/topk 40`)
- `/topp <valeur>` - Changer top_p (ex: `/topp 0.95`)
- `/help` - Afficher l'aide

### Charger depuis Hugging Face

```bash
python inference.py --hf_model votre_username/swamla-model \
    --mode chat
```

## Paramètres

### Paramètres de génération

- `--max_new_tokens` : Nombre maximum de tokens à générer (défaut: 256)
- `--temperature` : Température de sampling (défaut: 0.8)
  - Valeurs basses (0.1-0.5) : Plus déterministe et cohérent
  - Valeurs hautes (0.8-1.5) : Plus créatif et varié
- `--top_k` : Échantillonnage top-k (défaut: 50, 0 pour désactiver)
- `--top_p` : Échantillonnage nucleus/top-p (défaut: 0.9)
- `--repetition_penalty` : Pénalité de répétition (défaut: 1.1, >1.0 = moins de répétition)

### Paramètres du modèle

- `--checkpoint` : Chemin vers un checkpoint local
- `--hf_model` : ID du modèle sur Hugging Face Hub
- `--device` : Device à utiliser (`cuda` ou `cpu`, défaut: `cuda`)
- `--dtype` : Type de données (`float32`, `float16`, `bfloat16`, défaut: `bfloat16`)
- `--max_length` : Longueur maximale de séquence (défaut: 2048)

### Paramètres du mode chat

- `--system_prompt` : Prompt système pour initialiser la conversation

### Paramètres du mode batch

- `--prompts_file` : Fichier contenant les prompts (un par ligne)

## Exemples d'utilisation

### Génération créative

```bash
python inference.py --checkpoint checkpoint.pt \
    --mode batch \
    --temperature 1.2 \
    --top_k 100 \
    --max_new_tokens 200
```

### Génération précise et cohérente

```bash
python inference.py --checkpoint checkpoint.pt \
    --mode chat \
    --temperature 0.3 \
    --top_p 0.85 \
    --repetition_penalty 1.2
```

### Sur CPU avec float32

```bash
python inference.py --checkpoint checkpoint.pt \
    --mode chat \
    --device cpu \
    --dtype float32
```

### Avec GPU spécifique

```bash
CUDA_VISIBLE_DEVICES=1 python inference.py \
    --checkpoint checkpoint.pt \
    --mode batch \
    --prompts_file prompts.txt
```

## Fonctionnalités

### ✅ Chargement de modèles
- Depuis Hugging Face Hub
- Depuis checkpoints locaux
- Gère automatiquement les préfixes DDP et torch.compile
- Infère les dimensions du modèle depuis le checkpoint
- Compatible avec les checkpoints FP8 (avec ou sans TorchAO installé)

### ✅ Génération de texte
- Mode batch pour traiter plusieurs prompts
- Mode chat interactif
- Support de temperature, top_k, top_p
- Repetition penalty
- Gestion automatique du contexte

### ✅ Compatibilité
- CPU et CUDA
- Multiple dtypes (float32, float16, bfloat16)
- Conversion automatique des dtypes mixtes
- Mock TorchAO pour charger des checkpoints FP8 sans dépendance

## Dépannage

### CUDA out of memory

Réduisez `--max_new_tokens` ou `--max_length`, ou utilisez le CPU :

```bash
python inference.py --checkpoint checkpoint.pt \
    --mode chat \
    --device cpu \
    --dtype float32
```

### Erreur de dtype mismatch

Le script gère automatiquement la conversion des dtypes. Si vous rencontrez des erreurs, essayez `float32` :

```bash
python inference.py --checkpoint checkpoint.pt \
    --mode chat \
    --dtype float32
```

### Génération de mauvaise qualité

Ajustez les paramètres de génération :
- Diminuez `--temperature` pour plus de cohérence
- Augmentez `--repetition_penalty` pour éviter les répétitions
- Ajustez `--top_k` et `--top_p` selon vos besoins

### GPU non compatible (comme RTX 5090)

Si votre GPU n'est pas supporté par votre version de PyTorch, utilisez le CPU :

```bash
python inference.py --checkpoint checkpoint.pt \
    --mode chat \
    --device cpu \
    --dtype float32
```

### Erreur de dtype mismatch sur CUDA (Half != BFloat16)

Si vous rencontrez l'erreur `expected mat1 and mat2 to have the same dtype, but got: c10::Half != c10::BFloat16`, cela peut être dû à un checkpoint sauvegardé avec un autocast actif. Solutions :

**Solution 1 : Utiliser le CPU (recommandé pour l'inférence)**
```bash
python inference.py --checkpoint checkpoint.pt \
    --mode chat \
    --device cpu \
    --dtype float32
```

**Solution 2 : Créer un nouveau checkpoint sans autocast**
Le problème vient probablement d'un checkpoint créé pendant l'entraînement avec autocast. Créez un nouveau checkpoint dédié à l'inférence.

## Architecture du modèle

Le modèle SWA-MLA utilise une architecture hybride alternant :
- **Blocs SWA** : Attention locale avec fenêtre glissante (256 tokens par défaut)
- **Blocs MLA** : Attention globale avec compression KV low-rank

Le checkpoint chargé contient :
- 12 layers (8 SWA + 4 MLA)
- 262M paramètres
- Embedding dimension: 1024
- Vocabulaire: 50257 tokens (GPT-2)

## Notes

- Le script détecte automatiquement les dimensions du modèle depuis le checkpoint
- Les préfixes `_orig_mod.` (torch.compile) et `module.` (DDP) sont automatiquement supprimés
- Le tokenizer par défaut est GPT-2
- Le modèle est automatiquement mis en mode évaluation
