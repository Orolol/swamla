#!/bin/bash

# =============================================================================
# Unified Training Script for DeltaNet+MLA with All Features
# =============================================================================
#
# This consolidated script replaces all individual training scripts with a
# single configurable entrypoint that supports presets and feature toggles.
#
# QUICK START:
#   ./scripts/train.sh --preset base        # Basic DeltaNet+MLA
#   ./scripts/train.sh --preset engram-moe  # With Engram + LatentMoE
#   ./scripts/train.sh --preset full        # All features enabled
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
# =============================================================================

set -e

# =============================================================================
# Parse Command Line Arguments
# =============================================================================
show_help() {
    cat << EOF
Usage: ./scripts/train.sh [OPTIONS] [BATCH_SIZE] [BLOCK_SIZE]

OPTIONS:
  --preset NAME       Use a preset configuration (see presets below)
  --features LIST     Comma-separated features to enable (mup,progressive,ema,engram,moe)
  --size NAME         Model size (small, base, large, xl, moe-1b, engram-moe-1b)
  --output DIR        Output directory (default: outputs/train)
  --resume PATH       Resume from checkpoint (true=HF, false=none, or local path)
  --optimizer TYPE    Optimizer (adamw, muon, lion) [default: muon]
  --hf-repo ID        HuggingFace repo for auto-push
  --no-tensorboard    Disable TensorBoard
  --profile           Enable profiling
  --help              Show this help message

PRESETS:
  base          Basic DeltaNet+MLA training (no MoE, no Engram)
  moe           DeltaNet+MLA with LatentMoE
  engram        DeltaNet+MLA with Engram (no MoE)
  engram-moe    DeltaNet+MLA with Engram + LatentMoE (recommended)
  full          All features enabled (μP, Progressive, EMA, Engram, MoE)
  minimal       Minimal config for testing/debugging

FEATURES (use with --features):
  mup           Enable μP (Maximal Update Parametrization)
  progressive   Enable progressive sequence length training
  ema           Enable EMA weight averaging
  engram        Enable Engram conditional memory
  moe           Enable LatentMoE
  deltanet-latent  Enable DeltaNet latent compression

EXAMPLES:
  # Basic training with presets
  ./scripts/train.sh --preset engram-moe 8 2048

  # Custom features
  ./scripts/train.sh --features mup,progressive,engram 4 2048

  # With HuggingFace push
  ./scripts/train.sh --preset full --hf-repo username/model-name

  # Resume from checkpoint
  ./scripts/train.sh --preset engram-moe --resume /path/to/checkpoint.pt

ENVIRONMENT VARIABLES:
  Engram:
    ENGRAM_LAYERS="2,6"        Layers for Engram
    ENGRAM_D_MEM=512           Memory dimension
    ENGRAM_N_HASH_HEADS=8      Hash heads per N-gram order
    ENGRAM_NGRAM_ORDERS="2,3"  N-gram orders

  LatentMoE:
    LATENT_RATIO=4             Compression ratio
    N_EXPERTS=64               Base number of experts
    N_ACTIVATED=2              Base activated experts

  μP/Progressive/EMA:
    MUP_BASE_WIDTH=256         μP base width
    PROGRESSIVE_SCHEDULE="512:500M,1024:2B,2048:inf"
    EMA_DECAY=0.9999           EMA decay factor

EOF
    exit 0
}

# Defaults
PRESET=""
FEATURES=""
MODEL_SIZE=""
BATCH_SIZE=""
BLOCK_SIZE=""
OUTPUT_DIR=""
RESUME_FROM="false"
OPTIMIZER="muon"
HF_REPO_ID=""
USE_TENSORBOARD="true"
TENSORBOARD_PORT="6006"
PROFILE="false"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --preset)
            PRESET="$2"
            shift 2
            ;;
        --features)
            FEATURES="$2"
            shift 2
            ;;
        --size)
            MODEL_SIZE="$2"
            shift 2
            ;;
        --output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --resume)
            RESUME_FROM="$2"
            shift 2
            ;;
        --optimizer)
            OPTIMIZER="$2"
            shift 2
            ;;
        --hf-repo)
            HF_REPO_ID="$2"
            shift 2
            ;;
        --no-tensorboard)
            USE_TENSORBOARD="false"
            shift
            ;;
        --profile)
            PROFILE="true"
            shift
            ;;
        --help|-h)
            show_help
            ;;
        -*)
            echo "Unknown option: $1"
            exit 1
            ;;
        *)
            # Positional arguments: BATCH_SIZE BLOCK_SIZE
            if [ -z "$BATCH_SIZE" ]; then
                BATCH_SIZE="$1"
            elif [ -z "$BLOCK_SIZE" ]; then
                BLOCK_SIZE="$1"
            fi
            shift
            ;;
    esac
done

# =============================================================================
# Apply Preset Configurations
# =============================================================================

# Feature flags (will be set by preset or --features)
USE_MUP="${USE_MUP:-false}"
USE_PROGRESSIVE="${USE_PROGRESSIVE:-false}"
USE_EMA="${USE_EMA:-false}"
USE_ENGRAM="${USE_ENGRAM:-false}"
USE_LATENT_MOE="${USE_LATENT_MOE:-false}"
DELTANET_LATENT_DIM="${DELTANET_LATENT_DIM:-0}"
DELTANET_SHARE_QK="${DELTANET_SHARE_QK:-false}"

case "$PRESET" in
    base)
        MODEL_SIZE="${MODEL_SIZE:-moe-1b}"
        USE_LATENT_MOE="false"
        USE_ENGRAM="false"
        OUTPUT_DIR="${OUTPUT_DIR:-outputs/base}"
        ;;
    moe)
        MODEL_SIZE="${MODEL_SIZE:-moe-1b}"
        USE_LATENT_MOE="true"
        USE_ENGRAM="false"
        OUTPUT_DIR="${OUTPUT_DIR:-outputs/moe}"
        ;;
    engram)
        MODEL_SIZE="${MODEL_SIZE:-engram-moe-1b}"
        USE_LATENT_MOE="false"
        USE_ENGRAM="true"
        OUTPUT_DIR="${OUTPUT_DIR:-outputs/engram}"
        ;;
    engram-moe)
        MODEL_SIZE="${MODEL_SIZE:-engram-moe-1b}"
        USE_LATENT_MOE="true"
        USE_ENGRAM="true"
        OUTPUT_DIR="${OUTPUT_DIR:-outputs/engram-moe}"
        ;;
    full)
        MODEL_SIZE="${MODEL_SIZE:-engram-moe-1b}"
        USE_MUP="true"
        USE_PROGRESSIVE="true"
        USE_EMA="true"
        USE_ENGRAM="true"
        USE_LATENT_MOE="true"
        OUTPUT_DIR="${OUTPUT_DIR:-outputs/full}"
        ;;
    minimal)
        MODEL_SIZE="${MODEL_SIZE:-small}"
        USE_LATENT_MOE="false"
        USE_ENGRAM="false"
        OUTPUT_DIR="${OUTPUT_DIR:-outputs/minimal}"
        ;;
    "")
        # No preset, use defaults or --features
        MODEL_SIZE="${MODEL_SIZE:-engram-moe-1b}"
        OUTPUT_DIR="${OUTPUT_DIR:-outputs/train}"
        ;;
    *)
        echo "Unknown preset: $PRESET"
        echo "Available: base, moe, engram, engram-moe, full, minimal"
        exit 1
        ;;
esac

# Parse --features flag to override preset
if [ -n "$FEATURES" ]; then
    IFS=',' read -ra FEATURE_LIST <<< "$FEATURES"
    for feature in "${FEATURE_LIST[@]}"; do
        case "$feature" in
            mup) USE_MUP="true" ;;
            progressive) USE_PROGRESSIVE="true" ;;
            ema) USE_EMA="true" ;;
            engram) USE_ENGRAM="true" ;;
            moe) USE_LATENT_MOE="true" ;;
            deltanet-latent) DELTANET_LATENT_DIM="${DELTANET_LATENT_DIM:-256}" ;;
            *) echo "Unknown feature: $feature"; exit 1 ;;
        esac
    done
fi

# Apply defaults after preset/features processing
BATCH_SIZE="${BATCH_SIZE:-4}"
BLOCK_SIZE="${BLOCK_SIZE:-2048}"

# =============================================================================
# Feature Configuration (can be overridden via environment)
# =============================================================================

# μP
MUP_BASE_WIDTH="${MUP_BASE_WIDTH:-256}"

# Progressive Training
PROGRESSIVE_SCHEDULE="${PROGRESSIVE_SCHEDULE:-512:50M,1024:2B,2048:inf}"

# EMA
EMA_DECAY="${EMA_DECAY:-0.9999}"

# Engram
ENGRAM_LAYERS="${ENGRAM_LAYERS:-2,6}"
ENGRAM_D_MEM="${ENGRAM_D_MEM:-512}"
ENGRAM_N_HASH_HEADS="${ENGRAM_N_HASH_HEADS:-8}"
ENGRAM_NGRAM_ORDERS="${ENGRAM_NGRAM_ORDERS:-2,3}"
ENGRAM_CONV_KERNEL="${ENGRAM_CONV_KERNEL:-4}"
ENGRAM_LR_MULT="${ENGRAM_LR_MULT:-5.0}"

# LatentMoE
LATENT_RATIO="${LATENT_RATIO:-4}"
N_EXPERTS="${N_EXPERTS:-64}"
N_ACTIVATED="${N_ACTIVATED:-2}"

# MLA Q LoRA
MLA_Q_LORA_RANK="${MLA_Q_LORA_RANK:-0}"

# Profiling
PROFILE_STEPS="${PROFILE_STEPS:-5}"
PROFILE_WARMUP="${PROFILE_WARMUP:-2}"

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
echo "   DeltaNet+MLA Training"
echo "==========================================="
echo ""
echo "Configuration:"
if [ -n "$PRESET" ]; then
    echo "  Preset: $PRESET"
fi
echo "  Model size: $MODEL_SIZE"
echo "  Batch size: $BATCH_SIZE"
echo "  Block size: $BLOCK_SIZE"
echo "  Output dir: $OUTPUT_DIR"
echo "  Optimizer: $OPTIMIZER"
echo "  Detected GPUs: $NUM_GPUS"
echo ""

echo "Features:"
[ "$USE_MUP" = "true" ] && echo "  ✓ μP (base_width=$MUP_BASE_WIDTH)" || echo "  ✗ μP"
[ "$USE_PROGRESSIVE" = "true" ] && echo "  ✓ Progressive ($PROGRESSIVE_SCHEDULE)" || echo "  ✗ Progressive"
[ "$USE_EMA" = "true" ] && echo "  ✓ EMA (decay=$EMA_DECAY)" || echo "  ✗ EMA"
[ "$USE_ENGRAM" = "true" ] && echo "  ✓ Engram (layers=$ENGRAM_LAYERS, d_mem=$ENGRAM_D_MEM)" || echo "  ✗ Engram"
[ "$USE_LATENT_MOE" = "true" ] && echo "  ✓ LatentMoE (ratio=$LATENT_RATIO, experts=$N_EXPERTS)" || echo "  ✗ LatentMoE"
[ "$DELTANET_LATENT_DIM" != "0" ] && echo "  ✓ DeltaNet Latent (dim=$DELTANET_LATENT_DIM)"
echo ""

if [ -n "$HF_REPO_ID" ]; then
    echo "HuggingFace: $HF_REPO_ID (auto-push on validation)"
fi
if [ "$RESUME_FROM" = "true" ]; then
    echo "Resume: from HuggingFace"
elif [ "$RESUME_FROM" != "false" ] && [ -n "$RESUME_FROM" ]; then
    echo "Resume: from $RESUME_FROM"
fi
if [ "$PROFILE" = "true" ]; then
    echo "Profiling: enabled ($PROFILE_STEPS steps)"
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
COMMON_ARGS="--size $MODEL_SIZE \
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
    --fp8_backend auto \
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
