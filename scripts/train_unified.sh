#!/bin/bash

# =============================================================================
# Unified Training Script: DeltaNet+MLA with ALL Features
# =============================================================================
#
# Combines all advanced training features:
#
# ARCHITECTURE:
# - DeltaNet: O(n) linear attention for local context (GatedDeltaNet)
# - MLA: Multi-head Latent Attention with Flash Attention for global context
# - LatentMoE: Projects tokens to latent space before expert computation
# - Engram: Conditional Memory via Scalable N-gram Lookup (O(1) lookups)
#
# TRAINING OPTIMIZATIONS:
# - μP: Maximal Update Parametrization for width-independent hyperparameters
# - Progressive Training: Sequence length curriculum (512→1024→2048)
# - EMA: Exponential Moving Average for better generalization
#
# REQUIREMENTS:
# To use all features, merge from the mup-progressive-swa worktree:
#   cd .worktrees/mup-progressive-swa
#   git checkout main
#   git merge mup-progressive-swa
#
# This merges:
#   - optimization/mup.py (μP initialization + optimizer)
#   - optimization/progressive.py (sequence length curriculum)
#   - optimization/swa.py (EMA model wrapper)
#   - train.py updates for μP/Progressive/EMA integration
#   - models/swa_mla_model.py config fields for μP/Progressive/EMA
#
# Without the merge, disable these features:
#   USE_MUP=false USE_PROGRESSIVE=false USE_EMA=false ./scripts/train_unified.sh
#
# Auto-detects available GPUs and launches DDP training if multiple GPUs are found
# =============================================================================

set -e

# Add Flash Attention 3 (Hopper) to PYTHONPATH if available
FA3_PATH="$HOME/.local/flash-attention-3/flash-attention/hopper"
if [ -d "$FA3_PATH" ]; then
    export PYTHONPATH="$FA3_PATH:$PYTHONPATH"
    echo "Flash Attention 3 path added: $FA3_PATH"
fi

# =============================================================================
# Basic Configuration
# =============================================================================
BATCH_SIZE=${1:-4}
BLOCK_SIZE=${2:-2048}
OUTPUT_DIR=${3:-outputs/unified}
RESUME_FROM=${4:-false}
OPTIMIZER=${5:-muon}
HF_REPO_ID=${6:-}
USE_TENSORBOARD=${7:-true}
TENSORBOARD_PORT=${8:-6006}

# =============================================================================
# Feature Toggles (set to "true" or "false")
# =============================================================================
# μP: Width-independent hyperparameters (LR scales with model width)
USE_MUP=${USE_MUP:-true}
MUP_BASE_WIDTH=${MUP_BASE_WIDTH:-256}

# Progressive Training: Start with short sequences, increase over training
USE_PROGRESSIVE=${USE_PROGRESSIVE:-true}
PROGRESSIVE_SCHEDULE=${PROGRESSIVE_SCHEDULE:-"512:500M,1024:2B,2048:inf"}

# EMA: Exponential moving average for validation/deployment
USE_EMA=${USE_EMA:-true}
EMA_DECAY=${EMA_DECAY:-0.9999}

# Engram: Conditional memory via N-gram lookups
USE_ENGRAM=${USE_ENGRAM:-true}
ENGRAM_LAYERS=${ENGRAM_LAYERS:-"2,6"}
ENGRAM_D_MEM=${ENGRAM_D_MEM:-512}
ENGRAM_N_HASH_HEADS=${ENGRAM_N_HASH_HEADS:-8}  # Paper: K=8
ENGRAM_NGRAM_ORDERS=${ENGRAM_NGRAM_ORDERS:-"2,3"}
ENGRAM_CONV_KERNEL=${ENGRAM_CONV_KERNEL:-4}
ENGRAM_LR_MULT=${ENGRAM_LR_MULT:-5.0}

# LatentMoE: Mixture of experts with latent space projection
USE_LATENT_MOE=${USE_LATENT_MOE:-true}
LATENT_RATIO=${LATENT_RATIO:-4}
N_EXPERTS=${N_EXPERTS:-64}
N_ACTIVATED=${N_ACTIVATED:-2}

# DeltaNet Latent Compression (optional bottleneck for Q/K/V/O)
DELTANET_LATENT_DIM=${DELTANET_LATENT_DIM:-0}  # 0 = disabled
DELTANET_SHARE_QK=${DELTANET_SHARE_QK:-false}

# MLA Q LoRA (optional query compression)
MLA_Q_LORA_RANK=${MLA_Q_LORA_RANK:-0}  # 0 = disabled

# Profiling (for debugging performance)
PROFILE=${PROFILE:-false}
PROFILE_STEPS=${PROFILE_STEPS:-5}
PROFILE_WARMUP=${PROFILE_WARMUP:-2}

# =============================================================================
# Auto-detect GPUs
# =============================================================================
if command -v nvidia-smi &> /dev/null; then
    NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
else
    NUM_GPUS=0
fi

# =============================================================================
# Display Configuration
# =============================================================================
echo "==========================================="
echo "   Unified Training: DeltaNet+MLA+MoE++"
echo "==========================================="
echo ""
echo "Basic Configuration:"
echo "  Batch size: $BATCH_SIZE"
echo "  Block size: $BLOCK_SIZE"
echo "  Output dir: $OUTPUT_DIR"
echo "  Optimizer: $OPTIMIZER"
echo "  Detected GPUs: $NUM_GPUS"
echo ""

if [ "$USE_MUP" = "true" ]; then
    echo "μP (Maximal Update Parametrization): ENABLED"
    echo "  Base width: $MUP_BASE_WIDTH"
    echo ""
else
    echo "μP: disabled"
    echo ""
fi

if [ "$USE_PROGRESSIVE" = "true" ]; then
    echo "Progressive Training: ENABLED"
    echo "  Schedule: $PROGRESSIVE_SCHEDULE"
    echo ""
else
    echo "Progressive Training: disabled"
    echo ""
fi

if [ "$USE_EMA" = "true" ]; then
    echo "EMA (Stochastic Weight Averaging): ENABLED"
    echo "  Decay: $EMA_DECAY"
    echo ""
else
    echo "EMA: disabled"
    echo ""
fi

if [ "$USE_ENGRAM" = "true" ]; then
    echo "Engram (Conditional Memory): ENABLED"
    echo "  Layers: $ENGRAM_LAYERS"
    echo "  Memory dim: $ENGRAM_D_MEM"
    echo "  Hash heads: $ENGRAM_N_HASH_HEADS (paper: K=8)"
    echo "  N-gram orders: $ENGRAM_NGRAM_ORDERS"
    echo "  Conv kernel: $ENGRAM_CONV_KERNEL"
    echo "  LR multiplier: ${ENGRAM_LR_MULT}x"
    echo ""
else
    echo "Engram: disabled"
    echo ""
fi

if [ "$USE_LATENT_MOE" = "true" ]; then
    echo "LatentMoE: ENABLED"
    echo "  Latent ratio: ${LATENT_RATIO}x compression"
    echo "  Base experts: $N_EXPERTS (effective: $((N_EXPERTS * LATENT_RATIO)))"
    echo "  Base activated: $N_ACTIVATED (effective: $((N_ACTIVATED * LATENT_RATIO)))"
    echo ""
else
    echo "LatentMoE: disabled"
    echo ""
fi

if [ "$DELTANET_LATENT_DIM" != "0" ]; then
    echo "DeltaNet Latent Compression: ENABLED"
    echo "  Latent dim: $DELTANET_LATENT_DIM"
    echo "  Share Q/K: $DELTANET_SHARE_QK"
    echo ""
fi

if [ "$PROFILE" = "true" ]; then
    echo "Profiling: ENABLED"
    echo "  Warmup steps: $PROFILE_WARMUP"
    echo "  Profile steps: $PROFILE_STEPS"
    echo ""
fi

if [ -n "$HF_REPO_ID" ]; then
    echo "HuggingFace: $HF_REPO_ID (auto-push on validation)"
fi

if [ "$RESUME_FROM" = "true" ]; then
    echo "Resume: from HuggingFace"
elif [ "$RESUME_FROM" != "false" ] && [ -n "$RESUME_FROM" ]; then
    echo "Resume: from $RESUME_FROM"
fi

echo ""

# =============================================================================
# Launch TensorBoard
# =============================================================================
if [ "$USE_TENSORBOARD" = "true" ]; then
    if [ -f "./scripts/launch_tensorboard.sh" ]; then
        ./scripts/launch_tensorboard.sh "$OUTPUT_DIR/tensorboard" "$TENSORBOARD_PORT"
    fi
    echo ""
fi

# =============================================================================
# Build Command Arguments
# =============================================================================

# HuggingFace
HF_REPO_ARG=""
if [ -n "$HF_REPO_ID" ]; then
    HF_REPO_ARG="--hf_repo_id $HF_REPO_ID"
fi

# Resume
RESUME_ARG=""
if [ "$RESUME_FROM" = "true" ]; then
    RESUME_ARG="--resume_from_hf"
elif [ "$RESUME_FROM" != "false" ] && [ -n "$RESUME_FROM" ]; then
    RESUME_ARG="--resume_from $RESUME_FROM"
fi

# TensorBoard
TB_ARG=""
if [ "$USE_TENSORBOARD" = "true" ]; then
    TB_ARG="--use_tensorboard"
fi

# Profiler
PROFILE_ARG=""
if [ "$PROFILE" = "true" ]; then
    PROFILE_ARG="--profile --profile_steps $PROFILE_STEPS --profile_warmup $PROFILE_WARMUP"
fi

# μP
MUP_ARGS=""
if [ "$USE_MUP" = "true" ]; then
    MUP_ARGS="--use_mup --mup_base_width $MUP_BASE_WIDTH"
fi

# Progressive Training
PROGRESSIVE_ARGS=""
if [ "$USE_PROGRESSIVE" = "true" ]; then
    PROGRESSIVE_ARGS="--use_progressive --progressive_schedule $PROGRESSIVE_SCHEDULE"
fi

# EMA
EMA_ARGS=""
if [ "$USE_EMA" = "true" ]; then
    EMA_ARGS="--use_ema --ema_decay $EMA_DECAY"
fi

# Engram
ENGRAM_ARGS=""
if [ "$USE_ENGRAM" = "true" ]; then
    ENGRAM_ARGS="--use_engram \
        --engram_layers $ENGRAM_LAYERS \
        --engram_d_mem $ENGRAM_D_MEM \
        --engram_n_hash_heads $ENGRAM_N_HASH_HEADS \
        --engram_ngram_orders $ENGRAM_NGRAM_ORDERS \
        --engram_conv_kernel $ENGRAM_CONV_KERNEL \
        --engram_lr_multiplier $ENGRAM_LR_MULT"
fi

# LatentMoE
MOE_ARGS=""
if [ "$USE_LATENT_MOE" = "true" ]; then
    MOE_ARGS="--latent_ratio $LATENT_RATIO \
        --latent_preserve_expert_dim \
        --n_experts $N_EXPERTS \
        --n_activated $N_ACTIVATED"
fi

# DeltaNet Latent
DELTANET_ARGS="--deltanet_latent_dim $DELTANET_LATENT_DIM"
if [ "$DELTANET_SHARE_QK" = "true" ]; then
    DELTANET_ARGS="$DELTANET_ARGS --deltanet_share_qk"
fi

# =============================================================================
# Common Training Arguments
# =============================================================================
COMMON_ARGS="--size engram-moe-1b \
    --batch_size $BATCH_SIZE \
    --block_size $BLOCK_SIZE \
    --output_dir $OUTPUT_DIR \
    --optimizer_type $OPTIMIZER \
    --learning_rate 1e-4 \
    --weight_decay 0.1 \
    --warmup_iters 400 \
    --max_iters 1000000 \
    --grad_clip 1.0 \
    --gradient_accumulation_steps 1 \
    --gradient_checkpointing \
    --num_workers 8 \
    --local_layers_per_cycle 2 \
    --mla_layers_per_cycle 1 \
    --mla_q_lora_rank $MLA_Q_LORA_RANK \
    --mla_kv_lora_rank 256 \
    --mla_qk_nope_head_dim 128 \
    --mla_qk_rope_head_dim 64 \
    --mla_v_head_dim 128 \
    --tokenizer_name openai-community/gpt2 \
    --log_interval 50 \
    --eval_interval 5000 \
    --save_interval 5000 \
    --compile \
    --compile_mode max-autotune \
    $MUP_ARGS \
    $PROGRESSIVE_ARGS \
    $EMA_ARGS \
    $ENGRAM_ARGS \
    $MOE_ARGS \
    $DELTANET_ARGS \
    $HF_REPO_ARG \
    $RESUME_ARG \
    $TB_ARG \
    $PROFILE_ARG"

# =============================================================================
# Launch Training
# =============================================================================
if [ $NUM_GPUS -gt 1 ]; then
    echo "Launching DDP training on $NUM_GPUS GPUs..."
    echo ""
    torchrun --standalone --nproc_per_node=$NUM_GPUS train.py $COMMON_ARGS
else
    echo "Launching single GPU training..."
    echo ""
    python train.py $COMMON_ARGS
fi

exit 0

# =============================================================================
# Usage Examples
# =============================================================================
#
# Basic usage (all features enabled by default):
#   ./scripts/train_unified.sh
#   ./scripts/train_unified.sh 4 2048
#   ./scripts/train_unified.sh 8 2048 outputs/my_exp
#
# With HuggingFace auto-push:
#   ./scripts/train_unified.sh 4 2048 outputs/exp false muon "username/model-name"
#
# Resume from HuggingFace:
#   ./scripts/train_unified.sh 4 2048 outputs/exp true muon "username/model-name"
#
# Resume from local checkpoint:
#   ./scripts/train_unified.sh 4 2048 outputs/exp /path/to/checkpoint.pt
#
# =============================================================================
# Feature Toggle Examples
# =============================================================================
#
# Disable μP (use standard training):
#   USE_MUP=false ./scripts/train_unified.sh
#
# Disable progressive training (fixed sequence length):
#   USE_PROGRESSIVE=false ./scripts/train_unified.sh
#
# Custom progressive schedule (start at 256):
#   PROGRESSIVE_SCHEDULE="256:100M,512:500M,1024:2B,2048:inf" ./scripts/train_unified.sh
#
# Disable Engram (pure MoE):
#   USE_ENGRAM=false ./scripts/train_unified.sh
#
# Larger Engram memory:
#   ENGRAM_D_MEM=1024 ENGRAM_N_HASH_HEADS=16 ./scripts/train_unified.sh
#
# More Engram layers:
#   ENGRAM_LAYERS="2,4,6,8" ./scripts/train_unified.sh
#
# More experts:
#   N_EXPERTS=128 N_ACTIVATED=4 ./scripts/train_unified.sh
#
# Enable DeltaNet latent compression:
#   DELTANET_LATENT_DIM=256 DELTANET_SHARE_QK=true ./scripts/train_unified.sh
#
# Enable profiling:
#   PROFILE=true ./scripts/train_unified.sh
#
# =============================================================================
# Minimal Training (debugging/testing)
# =============================================================================
#
# Disable all advanced features:
#   USE_MUP=false USE_PROGRESSIVE=false USE_EMA=false USE_ENGRAM=false \
#   USE_LATENT_MOE=false ./scripts/train_unified.sh
#
# =============================================================================
# Parameters
# =============================================================================
#
#   $1: Batch size [default: 4]
#   $2: Block size (max context length) [default: 2048]
#   $3: Output directory [default: outputs/unified]
#   $4: Resume from (false/true/path) [default: false]
#   $5: Optimizer (muon/adamw/lion) [default: muon]
#   $6: HuggingFace repo ID [default: none]
#   $7: Use TensorBoard (true/false) [default: true]
#   $8: TensorBoard port [default: 6006]
#
# =============================================================================
# Architecture Notes
# =============================================================================
#
# Block Pattern (12 layers, 2 SWA + 1 MLA per cycle):
#   [DeltaNet, DeltaNet, MLA+MoE, DeltaNet, DeltaNet, MLA+MoE, ...]
#
# Engram Placement:
#   Applied BEFORE attention at specified layers
#   Uses residual: H = H + Engram(H, input_ids)
#
# μP Scaling:
#   - Embeddings: 1x LR, no decay
#   - Attention/MLP: 1/width LR
#   - LM Head: zero init, 1/width LR
#   - Engram tables: 5x LR, no decay
#
# Progressive Training Phases:
#   Phase 1 (0-500M tokens): seq_len=512, larger batch
#   Phase 2 (500M-2B tokens): seq_len=1024, medium batch
#   Phase 3 (2B+ tokens): seq_len=2048, base batch
#
# EMA:
#   Updated every step: θ_ema = 0.9999 * θ_ema + 0.0001 * θ
#   Used for validation (swapped temporarily)
#
