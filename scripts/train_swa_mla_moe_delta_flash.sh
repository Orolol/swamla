#!/bin/bash

# Train SWA+MLA hybrid model with optimized defaults
# This script is a standalone version that doesn't depend on the main project
# Auto-detects available GPUs and launches DDP training if multiple GPUs are found

# Add Flash Attention 3 (Hopper) to PYTHONPATH if available
FA3_PATH="$HOME/.local/flash-attention-3/flash-attention/hopper"
if [ -d "$FA3_PATH" ]; then
    export PYTHONPATH="$FA3_PATH:$PYTHONPATH"
    echo "Flash Attention 3 path added: $FA3_PATH"
fi

BATCH_SIZE=${1:-4}
BLOCK_SIZE=${2:-2048}
OUTPUT_DIR=${3:-outputs/swa_mla_moe}
RESUME_FROM_HF=${4:-false}  # Set to 'true' to resume from HuggingFace
OPTIMIZER=${5:-muon}  # adamw or lion
HF_REPO_ID=${6:-}  # e.g., "Orosius/swamla"
USE_TENSORBOARD=${7:-true}  # Enable TensorBoard by default
TENSORBOARD_PORT=${8:-6006}  # Default TensorBoard port

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
    echo "HF Repo: $HF_REPO_ID (automatic push every validation)"
    if [ "$RESUME_FROM_HF" = "true" ]; then
        echo "Resume: true (will load latest checkpoint from HF)"
    fi
fi

# Launch TensorBoard if enabled
if [ "$USE_TENSORBOARD" = "true" ]; then
    echo ""
    ./scripts/launch_tensorboard.sh "$OUTPUT_DIR/tensorboard" "$TENSORBOARD_PORT"
    echo ""
fi

# Build HF repo argument if provided
HF_REPO_ARG=""
if [ -n "$HF_REPO_ID" ]; then
    HF_REPO_ARG="--hf_repo_id $HF_REPO_ID"
    if [ "$RESUME_FROM_HF" = "true" ]; then
        HF_REPO_ARG="$HF_REPO_ARG --resume_from_hf"
    fi
fi

# Build TensorBoard argument if enabled
TB_ARG=""
if [ "$USE_TENSORBOARD" = "true" ]; then
    TB_ARG="--use_tensorboard"
fi

# Launch training with DDP if multiple GPUs detected
if [ $NUM_GPUS -gt 1 ]; then
    echo "Launching DDP training on $NUM_GPUS GPUs..."
    echo ""
    torchrun --standalone --nproc_per_node=$NUM_GPUS train.py \
    --size moe-1b \
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
    --save_interval 50000 \
    --use_flash_attention \
    --use_gated_deltanet \
    --use_tensorboard \
    --compile \ 
    --use_moe \
    $HF_REPO_ARG \
    $TB_ARG
else
    echo "Launching single GPU training..."
    echo ""
    python train.py \
    --size moe-1b \
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
    --eval_interval 5000 \
    --save_interval 20000 \
    --use_flash_attention \
    --use_gated_deltanet \
    --use_tensorboard \
    --compile \
    --use_moe \
    $HF_REPO_ARG \
    $TB_ARG
fi

# Usage examples:
# ./train_swa_mla.sh small 16 2048                                                    # Train small model (auto-detects GPUs)
# ./train_swa_mla.sh medium 8 2048 outputs/my_model                                   # Train medium model with custom output
# ./train_swa_mla.sh small 16 2048 outputs/my_model false lion                        # Use Lion optimizer
# ./train_swa_mla.sh small 8 2048 outputs/my_model false adamw "Orosius/swamla"      # Auto-push to HuggingFace every validation
# ./train_swa_mla.sh small 8 2048 outputs/my_model true adamw "Orosius/swamla"       # Resume from latest HF checkpoint
#
# Parameters:
#   $1: Model size (small, base, large, xl) [default: small]
#   $2: Batch size [default: 8]
#   $3: Block size (context length) [default: 2048]
#   $4: Output directory [default: outputs/swa_mla]
#   $5: Resume from HuggingFace (true/false) [default: false]
#   $6: Optimizer (adamw/lion) [default: adamw]
#   $7: HuggingFace repo ID (e.g., "username/repo") [default: none]
#   $8: Use TensorBoard (true/false) [default: true]
#   $9: TensorBoard port [default: 6006]
#
# Note: The script automatically detects available GPUs and launches:
#   - DDP training with torchrun if multiple GPUs are detected
#   - Single GPU training if only one GPU is detected
#   - CPU training if no GPUs are detected

exit 0
