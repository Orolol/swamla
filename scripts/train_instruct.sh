#!/bin/bash

# Instruction fine-tune SWA+MLA hybrid model with SlimOrca dataset
# Uses ChatML format with loss masking on prompts
# Auto-detects available GPUs and launches DDP training if multiple GPUs are found

MODEL_SIZE=${1:-small}
BATCH_SIZE=${2:-16}  # Higher batch size for fine-tuning (shorter sequences typically)
BLOCK_SIZE=${3:-2048}
OUTPUT_DIR=${4:-outputs/swa_mla_instruct}
RESUME_FROM_HF=${5:-true}  # Default to true: resume from pre-trained checkpoint
OPTIMIZER=${6:-adamw}
HF_REPO_ID=${7:-}  # e.g., "Orosius/swamla"

# Auto-detect number of GPUs
if command -v nvidia-smi &> /dev/null; then
    NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
else
    NUM_GPUS=0
fi

echo "Instruction Fine-Tuning SWA+MLA hybrid model..."
echo "Model size: $MODEL_SIZE"
echo "Batch size: $BATCH_SIZE"
echo "Block size: $BLOCK_SIZE"
echo "Output dir: $OUTPUT_DIR"
echo "Optimizer: $OPTIMIZER"
echo "Detected GPUs: $NUM_GPUS"
if [ -n "$HF_REPO_ID" ]; then
    echo "HF Repo: $HF_REPO_ID (automatic push to instruct/ subdirectory)"
    if [ "$RESUME_FROM_HF" = "true" ]; then
        echo "Resume: true (will load latest pre-trained checkpoint from HF)"
    fi
fi
echo ""
echo "Dataset: Open-Orca/SlimOrca"
echo "Format: ChatML with loss masking on prompts"
echo ""

# Build HF repo argument if provided
HF_REPO_ARG=""
if [ -n "$HF_REPO_ID" ]; then
    HF_REPO_ARG="--hf_repo_id $HF_REPO_ID"
    if [ "$RESUME_FROM_HF" = "true" ]; then
        HF_REPO_ARG="$HF_REPO_ARG --resume_from_hf"
    fi
fi

# Launch training with DDP if multiple GPUs detected
if [ $NUM_GPUS -gt 1 ]; then
    echo "Launching DDP training on $NUM_GPUS GPUs..."
    echo ""
    torchrun --standalone --nproc_per_node=$NUM_GPUS train_instruct.py \
    --size $MODEL_SIZE \
    --batch_size $BATCH_SIZE \
    --block_size $BLOCK_SIZE \
    --output_dir $OUTPUT_DIR \
    --optimizer_type $OPTIMIZER \
    --learning_rate 5e-5 \
    --min_lr 5e-6 \
    --weight_decay 0.1 \
    --warmup_iters 100 \
    --max_iters 10000 \
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
    --eval_interval 100 \
    --save_interval 500 \
    --wandb_project "swamla-instruct" \
    --use_fp8 \
    --gradient_checkpointing \
    $HF_REPO_ARG
else
    echo "Launching single GPU training..."
    echo ""
    python train_instruct.py \
    --size $MODEL_SIZE \
    --batch_size $BATCH_SIZE \
    --block_size $BLOCK_SIZE \
    --output_dir $OUTPUT_DIR \
    --optimizer_type $OPTIMIZER \
    --learning_rate 5e-5 \
    --min_lr 5e-6 \
    --weight_decay 0.1 \
    --warmup_iters 100 \
    --max_iters 10000 \
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
    --save_interval 1000 \
    --wandb_project "swamla-instruct" \
    --use_fp8 \
    --gradient_checkpointing \
    $HF_REPO_ARG
fi

# Usage examples:
# ./train_instruct.sh small 16 2048                                                    # Fine-tune small model (auto-resumes from HF)
# ./train_instruct.sh medium 8 2048 outputs/my_model                                   # Fine-tune medium model with custom output
# ./train_instruct.sh small 16 2048 outputs/my_model false lion                        # Use Lion optimizer, no resume
# ./train_instruct.sh small 8 2048 outputs/my_model true adamw "Orosius/swamla"       # Auto-push to HuggingFace instruct/ subdirectory
#
# Parameters:
#   $1: Model size (small, base, large, xl) [default: small]
#   $2: Batch size [default: 16]
#   $3: Block size (context length) [default: 2048]
#   $4: Output directory [default: outputs/swa_mla_instruct]
#   $5: Resume from HuggingFace (true/false) [default: true - loads pre-trained checkpoint]
#   $6: Optimizer (adamw/lion) [default: adamw]
#   $7: HuggingFace repo ID (e.g., "username/repo") [default: none]
#
# Note: The script automatically detects available GPUs and launches:
#   - DDP training with torchrun if multiple GPUs are detected
#   - Single GPU training if only one GPU is detected
#   - CPU training if no GPUs are detected
#
# Important: This script is designed for instruction fine-tuning:
#   - Loads pre-trained checkpoint from HuggingFace by default
#   - Uses SlimOrca dataset with ChatML format
#   - Applies loss masking (loss only on assistant responses)
#   - Saves checkpoints to instruct/ subdirectory on HuggingFace

exit 0
