#!/bin/bash

# Train SWA+MLA hybrid model with LatentMoE (NVIDIA Nemotron-3 style)
# LatentMoE projects tokens to latent space before expert computation,
# allowing more experts with the same compute budget for better quality.
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
OUTPUT_DIR=${3:-outputs/swa_mla_latent_moe}
RESUME_FROM_HF=${4:-false}  # Set to 'true' to resume from HuggingFace
OPTIMIZER=${5:-muon}  # adamw, lion, or muon
HF_REPO_ID=${6:-}  # e.g., "Orosius/swamla-latent-moe"
USE_TENSORBOARD=${7:-true}  # Enable TensorBoard by default
TENSORBOARD_PORT=${8:-6006}  # Default TensorBoard port

# LatentMoE configuration
LATENT_RATIO=${LATENT_RATIO:-4}  # d_model / latent_dim (default: 4x compression)
N_EXPERTS=${N_EXPERTS:-32}  # Base experts (will be scaled by latent_ratio -> 128)
N_ACTIVATED=${N_ACTIVATED:-3}  # Base activated (will be scaled by latent_ratio -> 12)

# Auto-detect number of GPUs
if command -v nvidia-smi &> /dev/null; then
    NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
else
    NUM_GPUS=0
fi

echo "==========================================="
echo "Training SWA+MLA with LatentMoE"
echo "==========================================="
echo "Batch size: $BATCH_SIZE"
echo "Block size: $BLOCK_SIZE"
echo "Output dir: $OUTPUT_DIR"
echo "Optimizer: $OPTIMIZER"
echo "Detected GPUs: $NUM_GPUS"
echo ""
echo "LatentMoE Configuration:"
echo "  Latent ratio: ${LATENT_RATIO}x compression"
echo "  Base experts: $N_EXPERTS (effective: $((N_EXPERTS * LATENT_RATIO)))"
echo "  Base activated: $N_ACTIVATED (effective: $((N_ACTIVATED * LATENT_RATIO)))"
echo ""
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

# Common training arguments
COMMON_ARGS="--size moe-1b \
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
    --tokenizer_name openai-community/gpt2 \
    --log_interval 10 \
    --eval_interval 1000 \
    --save_interval 50000 \
    --use_flash_attention \
    --use_gated_deltanet \
    --compile \
    --use_moe \
    --use_latent_moe \
    --latent_ratio $LATENT_RATIO \
    --latent_preserve_expert_dim \
    --n_experts $N_EXPERTS \
    --n_activated $N_ACTIVATED \
    $HF_REPO_ARG \
    $TB_ARG"

# Launch training with DDP if multiple GPUs detected
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
# ./scripts/train_swa_mla_latent_moe.sh 4 2048                                    # Train with defaults
# ./scripts/train_swa_mla_latent_moe.sh 8 2048 outputs/my_model                   # Custom output dir
# ./scripts/train_swa_mla_latent_moe.sh 4 2048 outputs/my_model false muon        # Use Muon optimizer
# ./scripts/train_swa_mla_latent_moe.sh 4 2048 outputs/my_model false adamw "Orosius/swamla-latent"  # Auto-push to HF
#
# Environment variables for LatentMoE configuration:
#   LATENT_RATIO=4     # Compression ratio (default: 4, meaning latent_dim = n_embd/4)
#   N_EXPERTS=32       # Base number of experts (scaled by LATENT_RATIO -> 128)
#   N_ACTIVATED=3      # Base activated experts (scaled by LATENT_RATIO -> 12)
#
# Example with custom LatentMoE config:
#   LATENT_RATIO=2 N_EXPERTS=64 N_ACTIVATED=4 ./scripts/train_swa_mla_latent_moe.sh 4 2048
#
# Parameters:
#   $1: Batch size [default: 4]
#   $2: Block size (context length) [default: 2048]
#   $3: Output directory [default: outputs/swa_mla_latent_moe]
#   $4: Resume from HuggingFace (true/false) [default: false]
#   $5: Optimizer (adamw/lion/muon) [default: muon]
#   $6: HuggingFace repo ID (e.g., "username/repo") [default: none]
#   $7: Use TensorBoard (true/false) [default: true]
#   $8: TensorBoard port [default: 6006]

exit 0
