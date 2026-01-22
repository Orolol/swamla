#!/bin/bash

# Train with μP + Progressive Training + EMA
#
# Architecture:
# - μP: Maximal Update Parametrization for width-independent hyperparameters
# - Progressive: Sequence length curriculum (512→1024→2048)
# - EMA: Exponential moving average for better generalization
#
# Auto-detects available GPUs and launches DDP training if multiple GPUs are found

# Add Flash Attention 3 (Hopper) to PYTHONPATH if available
FA3_PATH="$HOME/.local/flash-attention-3/flash-attention/hopper"
if [ -d "$FA3_PATH" ]; then
    export PYTHONPATH="$FA3_PATH:$PYTHONPATH"
    echo "Flash Attention 3 path added: $FA3_PATH"
fi

BATCH_SIZE=${1:-4}
BLOCK_SIZE=${2:-2048}
OUTPUT_DIR=${3:-outputs/mup_progressive}
RESUME_FROM=${4:-false}
OPTIMIZER=${5:-muon}
HF_REPO_ID=${6:-}
USE_TENSORBOARD=${7:-true}

# μP configuration
MUP_BASE_WIDTH=${MUP_BASE_WIDTH:-256}

# Progressive training configuration
PROGRESSIVE_SCHEDULE=${PROGRESSIVE_SCHEDULE:-"512:500M,1024:2B,2048:inf"}

# EMA configuration
EMA_DECAY=${EMA_DECAY:-0.9999}

# Auto-detect number of GPUs
if command -v nvidia-smi &> /dev/null; then
    NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
else
    NUM_GPUS=0
fi

echo "==========================================="
echo "Training with μP + Progressive + EMA"
echo "==========================================="
echo "Batch size: $BATCH_SIZE"
echo "Block size: $BLOCK_SIZE"
echo "Output dir: $OUTPUT_DIR"
echo "Optimizer: $OPTIMIZER"
echo "Detected GPUs: $NUM_GPUS"
echo ""
echo "μP Configuration:"
echo "  Base width: $MUP_BASE_WIDTH"
echo ""
echo "Progressive Training:"
echo "  Schedule: $PROGRESSIVE_SCHEDULE"
echo ""
echo "EMA:"
echo "  Decay: $EMA_DECAY"
echo ""

# Build arguments
HF_REPO_ARG=""
if [ -n "$HF_REPO_ID" ]; then
    HF_REPO_ARG="--hf_repo_id $HF_REPO_ID"
fi

RESUME_ARG=""
if [ "$RESUME_FROM" = "true" ]; then
    RESUME_ARG="--resume_from_hf"
elif [ "$RESUME_FROM" != "false" ] && [ -n "$RESUME_FROM" ]; then
    RESUME_ARG="--resume_from $RESUME_FROM"
fi

TB_ARG=""
if [ "$USE_TENSORBOARD" = "true" ]; then
    TB_ARG="--use_tensorboard"
fi

COMMON_ARGS="--size mup-1b \
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
    --use_mup \
    --mup_base_width $MUP_BASE_WIDTH \
    --use_progressive \
    --progressive_schedule "$PROGRESSIVE_SCHEDULE" \
    --use_ema \
    --ema_decay $EMA_DECAY \
    --compile \
    --compile_mode max-autotune \
    --log_interval 50 \
    --eval_interval 5000 \
    --save_interval 5000 \
    $HF_REPO_ARG \
    $RESUME_ARG \
    $TB_ARG"

# Launch training
if [ $NUM_GPUS -gt 1 ]; then
    echo "Launching DDP training on $NUM_GPUS GPUs..."
    echo ""
    torchrun --standalone --nproc_per_node=$NUM_GPUS train.py $COMMON_ARGS
else
    echo "Launching single GPU training..."
    echo ""
    python train.py $COMMON_ARGS
fi

# Usage examples:
# ./scripts/train_mup_progressive.sh                                             # Train with defaults (batch=4, block=2048)
# ./scripts/train_mup_progressive.sh 8 2048                                      # Custom batch/block size
# ./scripts/train_mup_progressive.sh 4 2048 outputs/exp1                         # Custom output directory
# ./scripts/train_mup_progressive.sh 4 2048 outputs/exp1 checkpoint.pt           # Resume from local checkpoint
# ./scripts/train_mup_progressive.sh 4 2048 outputs/exp1 true adamw "user/repo"  # Resume from HF + custom optimizer
#
# Parameters:
#   $1: Batch size [default: 4]
#   $2: Block size (context length) [default: 2048]
#   $3: Output directory [default: outputs/mup_progressive]
#   $4: Resume from (false/true/path) [default: false]
#       - false: start fresh
#       - true: resume from HuggingFace (requires $6)
#       - path: resume from local checkpoint
#   $5: Optimizer (muon/adamw/lion) [default: muon]
#   $6: HuggingFace repo ID (e.g., "username/repo") [default: none]
#   $7: Use TensorBoard (true/false) [default: true]
#
# Environment variables:
#   MUP_BASE_WIDTH: μP base width for LR scaling [default: 256]
#   PROGRESSIVE_SCHEDULE: Sequence length curriculum [default: "512:500M,1024:2B,2048:inf"]
#   EMA_DECAY: EMA decay factor [default: 0.9999]
#
# Example with custom schedule:
#   PROGRESSIVE_SCHEDULE="256:100M,512:500M,1024:2B,2048:inf" ./scripts/train_mup_progressive.sh 8 2048
