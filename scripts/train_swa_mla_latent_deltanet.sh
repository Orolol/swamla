#!/bin/bash

# Train DeltaNet+MLA hybrid model with LatentMoE AND DeltaNet Latent Compression
#
# Architecture:
# - GatedDeltaNet: O(n) linear attention for local context
# - MLA: Multi-head Latent Attention with Flash Attention for global context
# - LatentMoE: Projects tokens to latent space before expert computation
# - DeltaNet Latent Projections: Compresses Q/K/V/O projections via bottleneck
# - DeltaNet Shared Q/K: Q and K share projection, K is normalized Q
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
OUTPUT_DIR=${3:-outputs/deltanet_mla_latent}
RESUME_FROM=${4:-false}  # 'true' = resume from HF, 'false' = no resume, or path to local checkpoint
OPTIMIZER=${5:-muon}  # adamw or muon
HF_REPO_ID=${6:-Orosius/deltanet-mla-latent}  # e.g., "Orosius/deltanet-mla-latent"
USE_TENSORBOARD=${7:-true}  # Enable TensorBoard by default
TENSORBOARD_PORT=${8:-6006}  # Default TensorBoard port

# Profiling configuration (via environment variables)
PROFILE=${PROFILE:-false}  # Enable profiler
PROFILE_STEPS=${PROFILE_STEPS:-5}  # Number of steps to profile
PROFILE_WARMUP=${PROFILE_WARMUP:-2}  # Warmup steps before profiling

# LatentMoE configuration
LATENT_RATIO=${LATENT_RATIO:-4}  # d_model / latent_dim (default: 4x compression)
N_EXPERTS=${N_EXPERTS:-32}  # Base experts (will be scaled by latent_ratio -> 128)
N_ACTIVATED=${N_ACTIVATED:-3}  # Base activated (will be scaled by latent_ratio -> 12)

# DeltaNet Latent Compression configuration
# For moe-1b preset: n_embd=1024, so 4x compression -> 256
DELTANET_LATENT_DIM=${DELTANET_LATENT_DIM:-256}  # 0 = disabled, >0 = latent dimension
DELTANET_SHARE_QK=${DELTANET_SHARE_QK:-true}  # Share Q/K projection (K is normalized Q)

# MLA Q LoRA configuration (additional compression for MLA blocks)
MLA_Q_LORA_RANK=${MLA_Q_LORA_RANK:-256}  # 0 = disabled, >0 = Q LoRA rank

# Auto-detect number of GPUs
if command -v nvidia-smi &> /dev/null; then
    NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
else
    NUM_GPUS=0
fi

echo "==========================================="
echo "Training DeltaNet+MLA with LatentMoE"
echo "==========================================="
echo "Batch size: $BATCH_SIZE"
echo "Block size: $BLOCK_SIZE"
echo "Output dir: $OUTPUT_DIR"
echo "Optimizer: $OPTIMIZER"
echo "Detected GPUs: $NUM_GPUS"
echo ""
echo "LatentMoE Configuration (MLA blocks):"
echo "  Latent ratio: ${LATENT_RATIO}x compression"
echo "  Base experts: $N_EXPERTS (effective: $((N_EXPERTS * LATENT_RATIO)))"
echo "  Base activated: $N_ACTIVATED (effective: $((N_ACTIVATED * LATENT_RATIO)))"
echo ""
echo "DeltaNet Latent Compression:"
echo "  Latent dim: $DELTANET_LATENT_DIM (0 = disabled)"
echo "  Share Q/K: $DELTANET_SHARE_QK"
echo ""
echo "MLA Q LoRA:"
echo "  Q LoRA rank: $MLA_Q_LORA_RANK (0 = disabled)"
echo ""
if [ "$PROFILE" = "true" ]; then
    echo "Profiling:"
    echo "  Enabled: yes"
    echo "  Warmup steps: $PROFILE_WARMUP"
    echo "  Profile steps: $PROFILE_STEPS"
    echo ""
fi
if [ -n "$HF_REPO_ID" ]; then
    echo "HF Repo: $HF_REPO_ID (automatic push every validation)"
fi
if [ "$RESUME_FROM" = "true" ]; then
    echo "Resume: from HuggingFace (will load latest checkpoint from HF)"
elif [ "$RESUME_FROM" != "false" ] && [ -n "$RESUME_FROM" ]; then
    echo "Resume: from local path ($RESUME_FROM)"
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
fi

# Build resume argument (HF or local path)
RESUME_ARG=""
if [ "$RESUME_FROM" = "true" ]; then
    # Resume from HuggingFace
    RESUME_ARG="--resume_from_hf"
elif [ "$RESUME_FROM" != "false" ] && [ -n "$RESUME_FROM" ]; then
    # Resume from local checkpoint path
    RESUME_ARG="--resume_from $RESUME_FROM"
fi

# Build TensorBoard argument if enabled
TB_ARG=""
if [ "$USE_TENSORBOARD" = "true" ]; then
    TB_ARG="--use_tensorboard"
fi

# Build profiler arguments if enabled
PROFILE_ARG=""
if [ "$PROFILE" = "true" ]; then
    PROFILE_ARG="--profile --profile_steps $PROFILE_STEPS --profile_warmup $PROFILE_WARMUP"
fi

# Build DeltaNet latent compression arguments
DELTANET_ARGS="--deltanet_latent_dim $DELTANET_LATENT_DIM"
if [ "$DELTANET_SHARE_QK" = "true" ]; then
    DELTANET_ARGS="$DELTANET_ARGS --deltanet_share_qk"
fi

# Common training arguments
COMMON_ARGS="--size moe-2b \
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
    --swa_layers_per_cycle 2 \
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
    $DELTANET_ARGS \
    --compile \
    --compile_mode max-autotune \
    --latent_ratio $LATENT_RATIO \
    --latent_preserve_expert_dim \
    --n_experts $N_EXPERTS \
    --n_activated $N_ACTIVATED \
    $HF_REPO_ARG \
    $RESUME_ARG \
    $TB_ARG \
    $PROFILE_ARG"

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
# ./scripts/train_swa_mla_latent_deltanet.sh 4 2048                                    # Train with defaults
# ./scripts/train_swa_mla_latent_deltanet.sh 8 2048 outputs/my_model                   # Custom output dir
# ./scripts/train_swa_mla_latent_deltanet.sh 4 2048 outputs/my_model false muon        # Use Muon optimizer
# ./scripts/train_swa_mla_latent_deltanet.sh 4 2048 outputs/my_model false adamw "Orosius/deltanet-mla"  # Auto-push to HF
#
# Environment variables for configuration:
#
# LatentMoE (MLA blocks):
#   LATENT_RATIO=4         # Compression ratio (default: 4, meaning latent_dim = n_embd/4)
#   N_EXPERTS=32           # Base number of experts (scaled by LATENT_RATIO -> 128)
#   N_ACTIVATED=3          # Base activated experts (scaled by LATENT_RATIO -> 12)
#
# DeltaNet Latent Compression:
#   DELTANET_LATENT_DIM=256  # Latent dimension for Q/K/V/O bottleneck (0 = disabled)
#   DELTANET_SHARE_QK=true   # Share Q and K projection (K is normalized Q)
#
# MLA Q LoRA:
#   MLA_Q_LORA_RANK=256    # Q LoRA rank for MLA blocks (0 = disabled)
#
# Profiling:
#   PROFILE=true           # Enable PyTorch profiler
#   PROFILE_STEPS=5        # Number of steps to profile (default: 5)
#   PROFILE_WARMUP=2       # Warmup steps before profiling (default: 2)
#
# Example with profiler:
#   PROFILE=true ./scripts/train_swa_mla_latent_deltanet.sh 4 2048
#   # Then view results: tensorboard --logdir=outputs/deltanet_mla_latent/profiler
#
# Example with custom DeltaNet config:
#   DELTANET_LATENT_DIM=384 DELTANET_SHARE_QK=false ./scripts/train_swa_mla_latent_deltanet.sh 4 2048
#
# Example disabling DeltaNet latent compression (only LatentMoE):
#   DELTANET_LATENT_DIM=0 DELTANET_SHARE_QK=false ./scripts/train_swa_mla_latent_deltanet.sh 4 2048
#
# Parameters:
#   $1: Batch size [default: 4]
#   $2: Block size (context length) [default: 2048]
#   $3: Output directory [default: outputs/deltanet_mla_latent]
#   $4: Resume from (true=HF, false=none, or local path) [default: false]
#   $5: Optimizer (adamw/muon) [default: muon]
#   $6: HuggingFace repo ID (e.g., "username/repo") [default: none]
#   $7: Use TensorBoard (true/false) [default: true]
#   $8: TensorBoard port [default: 6006]
#
# Resume examples:
#   ./scripts/train_swa_mla_latent_deltanet.sh 4 2048 outputs/model false      # No resume
#   ./scripts/train_swa_mla_latent_deltanet.sh 4 2048 outputs/model true       # Resume from HF
#   ./scripts/train_swa_mla_latent_deltanet.sh 4 2048 outputs/model /path/to/checkpoint.pt  # Resume from local

exit 0
