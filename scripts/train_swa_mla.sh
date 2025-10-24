#!/bin/bash

# Train SWA+MLA hybrid model with optimized defaults
# This script is a standalone version that doesn't depend on the main project
# Auto-detects available GPUs and launches DDP training if multiple GPUs are found

MODEL_SIZE=${1:-small}
BATCH_SIZE=${2:-8}
BLOCK_SIZE=${3:-2048}
OUTPUT_DIR=${4:-outputs/swa_mla}
RESUME=${5:-false}
OPTIMIZER=${6:-adamw}  # adamw or lion
HF_REPO_ID=${7:-}  # e.g., "Orosius/swamla"

# Auto-detect number of GPUs
if command -v nvidia-smi &> /dev/null; then
    NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
else
    NUM_GPUS=0
fi

echo "Training SWA+MLA hybrid model..."
echo "Model size: $MODEL_SIZE"
echo "Batch size: $BATCH_SIZE"
echo "Block size: $BLOCK_SIZE"
echo "Output dir: $OUTPUT_DIR"
echo "Optimizer: $OPTIMIZER"
echo "Detected GPUs: $NUM_GPUS"
if [ -n "$HF_REPO_ID" ]; then
    echo "HF Repo: $HF_REPO_ID (automatic push on val loss improvement)"
fi
echo ""

# Build HF repo argument if provided
HF_REPO_ARG=""
if [ -n "$HF_REPO_ID" ]; then
    HF_REPO_ARG="--hf_repo_id $HF_REPO_ID"
fi

# Launch training with DDP if multiple GPUs detected
if [ $NUM_GPUS -gt 1 ]; then
    echo "Launching DDP training on $NUM_GPUS GPUs..."
    echo ""
    torchrun --standalone --nproc_per_node=$NUM_GPUS train.py \
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
    --use_fp8 \
    --gradient_checkpointing \
    $HF_REPO_ARG
else
    echo "Launching single GPU training..."
    echo ""
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
    --use_fp8 \
    --gradient_checkpointing \
    $HF_REPO_ARG
fi

# Usage examples:
# ./train_swa_mla.sh small 16 2048                                                  # Train small model (auto-detects GPUs)
# ./train_swa_mla.sh medium 8 2048 outputs/my_model                                 # Train medium model with custom output
# ./train_swa_mla.sh small 16 2048 outputs/my_model false lion                      # Use Lion optimizer
# ./train_swa_mla.sh small 8 2048 outputs/my_model false adamw "Orosius/swamla"    # Auto-push to HuggingFace
#
# Note: The script automatically detects available GPUs and launches:
#   - DDP training with torchrun if multiple GPUs are detected
#   - Single GPU training if only one GPU is detected
#   - CPU training if no GPUs are detected

exit 0
