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
  full          All features + nanochat optimizations (recommended for best perf)
  minimal       Minimal config for testing/debugging

FEATURES (use with --features):
  mup           Enable μP (Maximal Update Parametrization)
  progressive   Enable progressive sequence length training
  ema           Enable EMA weight averaging
  engram        Enable Engram conditional memory
  moe           Enable LatentMoE
  deltanet-latent  Enable DeltaNet latent compression
  yarn          Enable YaRN context extension (set YARN_SCALE_FACTOR env var)
  fope          Enable FoPE (Fourier Position Embedding) for better length generalization
  nanochat      Enable all nanochat optimizations (resid-scalars + cautious-wd + wd-schedule)
  resid-scalars Enable per-layer residual scalars (x0/resid lambdas)
  cautious-wd   Enable cautious weight decay (only decay same-sign updates)
  wd-schedule   Enable linear WD schedule (decay to 0 over training)
  bestfit-crop  Enable BestFit-Crop packing (~100% seq utilization)

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

  YaRN (Context Extension):
    YARN_ENABLED=false         Enable YaRN context extension
    YARN_SCALE_FACTOR=1.0      Context extension ratio
    YARN_ORIGINAL_MAX_SEQ=2048 Original training context length
    YARN_BETA_FAST=32.0        High frequency boundary
    YARN_BETA_SLOW=1.0         Low frequency boundary

  FoPE (Fourier Position Embedding):
    FOPE_ENABLED=false         Enable FoPE (replaces RoPE with Fourier series)
    FOPE_N_HARMONICS=4         Number of harmonic components per dimension
    FOPE_FLOOR_RATIO=0.1       Fraction of low frequencies to zero out
    FOPE_COEF_INIT_STD=0.3     Std for Fourier coefficient initialization

  Nanochat (per-layer residual scalars):
    X0_LR=0.5                  LR for x0_lambdas (additive residual)
    RESID_LR=0.005             LR for resid_lambdas (multiplicative)
    X0_BETA1=0.96              Beta1 for x0 params

  Token-based Triggers:
    EVAL_TOKENS=500M           Validate every N tokens (e.g., 500M, 1B, 2.5B)
    SAVE_TOKENS=2B             Save checkpoint every N tokens (e.g., 2B, 5B)

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

# Nanochat features
USE_RESIDUAL_SCALARS="${USE_RESIDUAL_SCALARS:-false}"
USE_CAUTIOUS_WD="${USE_CAUTIOUS_WD:-false}"
USE_WD_SCHEDULE="${USE_WD_SCHEDULE:-false}"
USE_BESTFIT_CROP="${USE_BESTFIT_CROP:-true}"  # Enabled by default

# YaRN context extension
USE_YARN="${USE_YARN:-false}"

# FoPE (Fourier Position Embedding)
USE_FOPE="${USE_FOPE:-false}"

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
        USE_PROGRESSIVE="false"
        USE_EMA="true"
        USE_ENGRAM="true"
        USE_LATENT_MOE="true"
        # Position embeddings
        USE_YARN="true"
        USE_FOPE="true"
        # Nanochat features
        USE_RESIDUAL_SCALARS="true"
        USE_CAUTIOUS_WD="true"
        USE_WD_SCHEDULE="true"
        USE_BESTFIT_CROP="true"
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
            # YaRN context extension
            yarn) USE_YARN="true" ;;
            # FoPE (Fourier Position Embedding)
            fope) USE_FOPE="true" ;;
            # Nanochat features
            resid-scalars) USE_RESIDUAL_SCALARS="true" ;;
            cautious-wd) USE_CAUTIOUS_WD="true" ;;
            wd-schedule) USE_WD_SCHEDULE="true" ;;
            bestfit-crop) USE_BESTFIT_CROP="true" ;;
            nanochat)
                USE_RESIDUAL_SCALARS="true"
                USE_CAUTIOUS_WD="true"
                USE_WD_SCHEDULE="true"
                USE_BESTFIT_CROP="true"
                ;;
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
PROGRESSIVE_SCHEDULE="${PROGRESSIVE_SCHEDULE:-2048:inf}"

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

# YaRN: Context Extension
YARN_ENABLED="${YARN_ENABLED:-$USE_YARN}"
YARN_SCALE_FACTOR="${YARN_SCALE_FACTOR:-1.0}"
YARN_ORIGINAL_MAX_SEQ="${YARN_ORIGINAL_MAX_SEQ:-2048}"
YARN_BETA_FAST="${YARN_BETA_FAST:-32.0}"
YARN_BETA_SLOW="${YARN_BETA_SLOW:-1.0}"

# FoPE: Fourier Position Embedding
FOPE_ENABLED="${FOPE_ENABLED:-$USE_FOPE}"
FOPE_N_HARMONICS="${FOPE_N_HARMONICS:-4}"
FOPE_FLOOR_RATIO="${FOPE_FLOOR_RATIO:-0.1}"
FOPE_COEF_INIT_STD="${FOPE_COEF_INIT_STD:-0.3}"

# Nanochat: Per-layer residual scalars
X0_LR="${X0_LR:-0.5}"
RESID_LR="${RESID_LR:-0.005}"
X0_BETA1="${X0_BETA1:-0.96}"

# Token-based validation and save triggers
EVAL_TOKENS="${EVAL_TOKENS:-500M}"
SAVE_TOKENS="${SAVE_TOKENS:-500M}"

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
[ "$YARN_ENABLED" = "true" ] && echo "  ✓ YaRN (scale=$YARN_SCALE_FACTOR, orig_len=$YARN_ORIGINAL_MAX_SEQ)" || echo "  ✗ YaRN"
[ "$FOPE_ENABLED" = "true" ] && echo "  ✓ FoPE (harmonics=$FOPE_N_HARMONICS, floor=$FOPE_FLOOR_RATIO)" || echo "  ✗ FoPE"

echo ""
echo "Nanochat Optimizations:"
[ "$USE_RESIDUAL_SCALARS" = "true" ] && echo "  ✓ Residual Scalars (x0_lr=$X0_LR, resid_lr=$RESID_LR)" || echo "  ✗ Residual Scalars"
[ "$USE_CAUTIOUS_WD" = "true" ] && echo "  ✓ Cautious Weight Decay" || echo "  ✗ Cautious Weight Decay"
[ "$USE_WD_SCHEDULE" = "true" ] && echo "  ✓ WD Schedule (linear decay to 0)" || echo "  ✗ WD Schedule"
[ "$USE_BESTFIT_CROP" = "true" ] && echo "  ✓ BestFit-Crop Packing" || echo "  ✗ BestFit-Crop Packing"
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

# Nanochat: Residual Scalars
RESID_SCALAR_ARGS=""
if [ "$USE_RESIDUAL_SCALARS" = "true" ]; then
    RESID_SCALAR_ARGS="--use_residual_scalars \
        --x0_lr $X0_LR \
        --resid_lr $RESID_LR \
        --x0_beta1 $X0_BETA1"
fi

# Nanochat: Muon Optimizer Upgrades
MUON_UPGRADE_ARGS=""
if [ "$USE_CAUTIOUS_WD" = "true" ]; then
    MUON_UPGRADE_ARGS="$MUON_UPGRADE_ARGS --cautious_wd"
fi
if [ "$USE_WD_SCHEDULE" = "true" ]; then
    MUON_UPGRADE_ARGS="$MUON_UPGRADE_ARGS --wd_schedule"
fi

# Nanochat: BestFit-Crop Packing
BESTFIT_ARGS=""
if [ "$USE_BESTFIT_CROP" = "true" ]; then
    BESTFIT_ARGS="--use_bestfit_crop"
fi

# YaRN: Context Extension
YARN_ARGS=""
if [ "$YARN_ENABLED" = "true" ]; then
    YARN_ARGS="--yarn_enabled \
        --yarn_scale_factor $YARN_SCALE_FACTOR \
        --yarn_original_max_seq_len $YARN_ORIGINAL_MAX_SEQ \
        --yarn_beta_fast $YARN_BETA_FAST \
        --yarn_beta_slow $YARN_BETA_SLOW"
fi

# FoPE: Fourier Position Embedding
FOPE_ARGS=""
if [ "$FOPE_ENABLED" = "true" ]; then
    FOPE_ARGS="--fope_enabled \
        --fope_n_harmonics $FOPE_N_HARMONICS \
        --fope_floor_ratio $FOPE_FLOOR_RATIO \
        --fope_coef_init_std $FOPE_COEF_INIT_STD"
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
    --cudnn_compatible_heads \
    --mla_qk_nope_head_dim 128 \
    --mla_qk_rope_head_dim 64 \
    --mla_v_head_dim 128 \
    --tokenizer_name openai-community/gpt2 \
    --log_interval 50 \
    --eval_tokens $EVAL_TOKENS \
    --save_tokens $SAVE_TOKENS \
    --fp8_backend auto \
    --compile \
    --compile_mode max-autotune \
    $MUP_ARGS \
    $PROGRESSIVE_ARGS \
    $EMA_ARGS \
    $ENGRAM_ARGS \
    $MOE_ARGS \
    $DELTANET_ARGS \
    $RESID_SCALAR_ARGS \
    $MUON_UPGRADE_ARGS \
    $BESTFIT_ARGS \
    $YARN_ARGS \
    $FOPE_ARGS \
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
