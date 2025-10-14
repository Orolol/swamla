#!/bin/bash

# Train SWA+MLA hybrid model with optimized defaults
# This script is a standalone version that doesn't depend on the main project

MODEL_SIZE=${1:-small}
BATCH_SIZE=${2:-8}
BLOCK_SIZE=${3:-2048}
OUTPUT_DIR=${4:-outputs/swa_mla}
RESUME=${5:-false}
OPTIMIZER=${6:-adamw}  # adamw or lion

echo "Training SWA+MLA hybrid model..."
echo "Model size: $MODEL_SIZE"
echo "Batch size: $BATCH_SIZE"
echo "Block size: $BLOCK_SIZE"
echo "Output dir: $OUTPUT_DIR"
echo "Optimizer: $OPTIMIZER"
echo ""

# Determine if we should use FP8
USE_FP8=""
if command -v nvidia-smi &> /dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1)
    if [[ $GPU_NAME == *"H100"* ]] || [[ $GPU_NAME == *"H200"* ]]; then
        echo "Detected H100/H200 GPU - enabling FP8 training"
        USE_FP8="--use_fp8"
    else
        echo "GPU: $GPU_NAME - using BF16 training"
    fi
fi

python train.py \
    --size $MODEL_SIZE \
    --batch_size $BATCH_SIZE \
    --block_size $BLOCK_SIZE \
    --output_dir $OUTPUT_DIR \
    --optimizer_type $OPTIMIZER \
    --learning_rate 1e-4 \
    --weight_decay 0.1 \
    --warmup_iters 400 \
    --max_iters 100000 \
    --grad_clip 1.0 \
    --gradient_accumulation_steps 1 \
    --num_workers 8 \
    --swa_layers_per_cycle 2 \
    --mla_layers_per_cycle 1 \
    --swa_window 256 \
    --swa_sink_size 4 \
    --mla_q_lora_rank 0 \
    --mla_kv_lora_rank 256 \
    --mla_qk_nope_head_dim 128 \
    --mla_qk_rope_head_dim 64 \
    --mla_v_head_dim 128 \
    --tokenizer_name "openai-community/gpt2" \
    --log_interval 10 \
    --eval_interval 1000 \
    --save_interval 5000 \
    --compile \
    --gradient_checkpointing \
    $USE_FP8

# Usage examples:
# ./train_swa_mla.sh small 16 2048                                 # Train small model
# ./train_swa_mla.sh medium 8 2048 outputs/my_model                # Train medium model with custom output
# ./train_swa_mla.sh small 16 2048 outputs/my_model false lion     # Use Lion optimizer

exit 0
