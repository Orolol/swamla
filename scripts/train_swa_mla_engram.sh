#!/bin/bash

# Train DeltaNet+MLA hybrid model with LatentMoE AND Engram
#
# Architecture:
# - GatedDeltaNet: O(n) linear attention for local context
# - MLA: Multi-head Latent Attention with Flash Attention for global context
# - LatentMoE: Projects tokens to latent space before expert computation
# - Engram: Conditional Memory via Scalable N-gram Lookup
#   - Performs O(1) lookups in N-gram embedding tables
#   - Complements MoE by handling static pattern reconstruction
#   - Applied BEFORE attention at specified layers
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
OUTPUT_DIR=${3:-outputs/deltanet_mla_engram}
RESUME_FROM=${4:-false}  # 'true' = resume from HF, 'false' = no resume, or path to local checkpoint
OPTIMIZER=${5:-muon}  # adamw or muon
HF_REPO_ID=${6:-}  # e.g., "Orosius/deltanet-mla-engram"
USE_TENSORBOARD=${7:-true}  # Enable TensorBoard by default
TENSORBOARD_PORT=${8:-6006}  # Default TensorBoard port

# Profiling configuration (via environment variables)
PROFILE=${PROFILE:-false}  # Enable profiler
PROFILE_STEPS=${PROFILE_STEPS:-5}  # Number of steps to profile
PROFILE_WARMUP=${PROFILE_WARMUP:-2}  # Warmup steps before profiling

# LatentMoE configuration (reduced experts to make room for Engram params)
LATENT_RATIO=${LATENT_RATIO:-4}  # d_model / latent_dim (default: 4x compression)
N_EXPERTS=${N_EXPERTS:-64}  # Reduced from 32 to allocate params to Engram
N_ACTIVATED=${N_ACTIVATED:-2}  # Base activated (will be scaled by latent_ratio)

# Engram configuration
# Paper recommendation: layers 2 and 15 for 27B model
# For smaller models (12 layers): layers 2 and 6
ENGRAM_LAYERS=${ENGRAM_LAYERS:-"2,6"}  # Comma-separated layer indices
ENGRAM_D_MEM=${ENGRAM_D_MEM:-512}  # Memory embedding dimension
ENGRAM_N_HASH_HEADS=${ENGRAM_N_HASH_HEADS:-8}  # Hash heads per N-gram order (paper: K=8)
ENGRAM_NGRAM_ORDERS=${ENGRAM_NGRAM_ORDERS:-"2,3"}  # N-gram orders (bigrams, trigrams)
ENGRAM_CONV_KERNEL=${ENGRAM_CONV_KERNEL:-4}  # Causal conv kernel size
ENGRAM_LR_MULT=${ENGRAM_LR_MULT:-5.0}  # LR multiplier for Engram embeddings (paper: 5x)

# DeltaNet Latent Compression configuration (optional, can be combined)
DELTANET_LATENT_DIM=${DELTANET_LATENT_DIM:-0}  # 0 = disabled by default for Engram config
DELTANET_SHARE_QK=${DELTANET_SHARE_QK:-false}

# MLA Q LoRA configuration
MLA_Q_LORA_RANK=${MLA_Q_LORA_RANK:-0}  # 0 = disabled

# Auto-detect number of GPUs
if command -v nvidia-smi &> /dev/null; then
    NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
else
    NUM_GPUS=0
fi

echo "==========================================="
echo "Training DeltaNet+MLA with LatentMoE + Engram"
echo "==========================================="
echo "Batch size: $BATCH_SIZE"
echo "Block size: $BLOCK_SIZE"
echo "Output dir: $OUTPUT_DIR"
echo "Optimizer: $OPTIMIZER"
echo "Detected GPUs: $NUM_GPUS"
echo ""
echo "Engram Configuration (Conditional Memory):"
echo "  Layers: $ENGRAM_LAYERS"
echo "  Memory dim (d_mem): $ENGRAM_D_MEM"
echo "  Hash heads: $ENGRAM_N_HASH_HEADS"
echo "  N-gram orders: $ENGRAM_NGRAM_ORDERS"
echo "  Conv kernel: $ENGRAM_CONV_KERNEL"
echo "  LR multiplier: ${ENGRAM_LR_MULT}x (for embedding tables)"
echo ""
echo "LatentMoE Configuration (MLA blocks):"
echo "  Latent ratio: ${LATENT_RATIO}x compression"
echo "  Base experts: $N_EXPERTS (effective: $((N_EXPERTS * LATENT_RATIO)))"
echo "  Base activated: $N_ACTIVATED (effective: $((N_ACTIVATED * LATENT_RATIO)))"
echo ""
if [ "$DELTANET_LATENT_DIM" != "0" ]; then
    echo "DeltaNet Latent Compression:"
    echo "  Latent dim: $DELTANET_LATENT_DIM"
    echo "  Share Q/K: $DELTANET_SHARE_QK"
    echo ""
fi
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
    if [ -f "./scripts/launch_tensorboard.sh" ]; then
        ./scripts/launch_tensorboard.sh "$OUTPUT_DIR/tensorboard" "$TENSORBOARD_PORT"
    fi
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

# Build Engram arguments
ENGRAM_ARGS="--use_engram \
    --engram_layers $ENGRAM_LAYERS \
    --engram_d_mem $ENGRAM_D_MEM \
    --engram_n_hash_heads $ENGRAM_N_HASH_HEADS \
    --engram_ngram_orders $ENGRAM_NGRAM_ORDERS \
    --engram_conv_kernel $ENGRAM_CONV_KERNEL \
    --engram_lr_multiplier $ENGRAM_LR_MULT"

# Common training arguments
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
    $ENGRAM_ARGS \
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
# ./scripts/train_swa_mla_engram.sh 4 2048                                    # Train with defaults
# ./scripts/train_swa_mla_engram.sh 8 2048 outputs/my_model                   # Custom output dir
# ./scripts/train_swa_mla_engram.sh 4 2048 outputs/my_model false muon        # Use Muon optimizer
# ./scripts/train_swa_mla_engram.sh 4 2048 outputs/my_model false adamw "Orosius/engram-model"  # Auto-push to HF
#
# Environment variables for Engram configuration:
#
#   ENGRAM_LAYERS="2,6"        # Layers where Engram is applied (comma-separated)
#   ENGRAM_D_MEM=512           # Memory embedding dimension
#   ENGRAM_N_HASH_HEADS=8      # Number of hash heads per N-gram order (paper: K=8)
#   ENGRAM_NGRAM_ORDERS="2,3"  # N-gram orders (bigrams, trigrams)
#   ENGRAM_CONV_KERNEL=4       # Causal convolution kernel size
#   ENGRAM_LR_MULT=5.0         # LR multiplier for Engram embeddings (paper: 5x)
#
# Environment variables for LatentMoE (MLA blocks):
#
#   LATENT_RATIO=4             # Compression ratio (default: 4, meaning latent_dim = n_embd/4)
#   N_EXPERTS=24               # Base number of experts (reduced to make room for Engram)
#   N_ACTIVATED=2              # Base activated experts (scaled by LATENT_RATIO)
#
# Environment variables for DeltaNet Latent Compression (optional):
#
#   DELTANET_LATENT_DIM=256    # Latent dimension for Q/K/V/O bottleneck (0 = disabled)
#   DELTANET_SHARE_QK=true     # Share Q and K projection (K is normalized Q)
#
# Profiling:
#   PROFILE=true               # Enable PyTorch profiler
#   PROFILE_STEPS=5            # Number of steps to profile (default: 5)
#   PROFILE_WARMUP=2           # Warmup steps before profiling (default: 2)
#
# Example with custom Engram config (larger memory):
#   ENGRAM_D_MEM=1024 ENGRAM_N_HASH_HEADS=8 ./scripts/train_swa_mla_engram.sh 4 2048
#
# Example with Engram on more layers (for larger models):
#   ENGRAM_LAYERS="2,8,15,22" ./scripts/train_swa_mla_engram.sh 4 2048
#
# Example with profiler:
#   PROFILE=true ./scripts/train_swa_mla_engram.sh 4 2048
#
# Example combining Engram with DeltaNet latent compression:
#   DELTANET_LATENT_DIM=256 DELTANET_SHARE_QK=true ./scripts/train_swa_mla_engram.sh 4 2048
#
# Parameters:
#   $1: Batch size [default: 4]
#   $2: Block size (context length) [default: 2048]
#   $3: Output directory [default: outputs/deltanet_mla_engram]
#   $4: Resume from (true=HF, false=none, or local path) [default: false]
#   $5: Optimizer (adamw/muon) [default: muon]
#   $6: HuggingFace repo ID (e.g., "username/repo") [default: none]
#   $7: Use TensorBoard (true/false) [default: true]
#   $8: TensorBoard port [default: 6006]
#
# Resume examples:
#   ./scripts/train_swa_mla_engram.sh 4 2048 outputs/model false      # No resume
#   ./scripts/train_swa_mla_engram.sh 4 2048 outputs/model true       # Resume from HF
#   ./scripts/train_swa_mla_engram.sh 4 2048 outputs/model /path/to/checkpoint.pt  # Resume from local
#
# Architecture notes:
#   - Engram is applied BEFORE attention at specified layers
#   - Uses residual connection: H = H + Engram(H, input_ids)
#   - Embedding tables use 5x higher learning rate (no weight decay)
#   - Zero-initialized causal conv ensures identity mapping at start
#   - Recommended allocation: ~75-80% MoE params, ~20-25% Engram params

exit 0
