#!/bin/bash

# Compare SWA+FlashAttn vs DeltaNet+FlashAttn on 2000 steps
# This script runs both training configurations and compares results

set -e

BATCH_SIZE=${1:-4}
BLOCK_SIZE=${2:-2048}
MAX_ITERS=${3:-2000}
MODEL_SIZE=${4:-small}

echo "============================================================"
echo "SWA vs DeltaNet Comparison Training"
echo "============================================================"
echo "Model size: $MODEL_SIZE"
echo "Batch size: $BATCH_SIZE"
echo "Block size: $BLOCK_SIZE"
echo "Max iterations: $MAX_ITERS"
echo "============================================================"
echo ""

# Output directories
OUTPUT_SWA="outputs/compare_swa_flash_${MAX_ITERS}steps"
OUTPUT_DELTANET="outputs/compare_deltanet_flash_${MAX_ITERS}steps"

# Clean previous runs
rm -rf "$OUTPUT_SWA" "$OUTPUT_DELTANET"

echo "============================================================"
echo "PHASE 1: Training SWA + FlashAttention"
echo "============================================================"
echo ""

START_SWA=$(date +%s)

python train.py \
    --size $MODEL_SIZE \
    --batch_size $BATCH_SIZE \
    --block_size $BLOCK_SIZE \
    --output_dir $OUTPUT_SWA \
    --optimizer_type lion \
    --learning_rate 1e-4 \
    --weight_decay 0.1 \
    --warmup_iters 200 \
    --max_iters $MAX_ITERS \
    --grad_clip 1.0 \
    --gradient_accumulation_steps 1 \
    --num_workers 4 \
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
    --log_interval 50 \
    --eval_interval 500 \
    --save_interval 10000 \
    --use_flash_attention \
    --no_moe \
    --use_tensorboard

END_SWA=$(date +%s)
TIME_SWA=$((END_SWA - START_SWA))

echo ""
echo "============================================================"
echo "SWA + FlashAttention completed in ${TIME_SWA}s"
echo "============================================================"
echo ""

echo "============================================================"
echo "PHASE 2: Training DeltaNet + FlashAttention (MLA)"
echo "============================================================"
echo ""

START_DELTANET=$(date +%s)

python train.py \
    --size $MODEL_SIZE \
    --batch_size $BATCH_SIZE \
    --block_size $BLOCK_SIZE \
    --output_dir $OUTPUT_DELTANET \
    --optimizer_type lion \
    --learning_rate 1e-4 \
    --weight_decay 0.1 \
    --warmup_iters 200 \
    --max_iters $MAX_ITERS \
    --grad_clip 1.0 \
    --gradient_accumulation_steps 1 \
    --num_workers 4 \
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
    --log_interval 50 \
    --eval_interval 500 \
    --save_interval 10000 \
python train.py \
    --size small \
    --batch_size 4 \
    --block_size 2048 \
    --output_dir outputs/compare_deltanet_flash_2000steps \
    --optimizer_type lion \
    --learning_rate 1e-4 \
    --weight_decay 0.1 \
    --warmup_iters 200 \
    --max_iters 2000 \
    --grad_clip 1.0 \
    --num_workers 4 \
    --swa_layers_per_cycle 2 \
    --mla_layers_per_cycle 1 \
    --tokenizer_name "openai-community/gpt2" \
    --log_interval 50 \
    --eval_interval 500 \
    --use_flash_attention \
    --use_gated_deltanet \
    --use_tensorboardpython train.py \
    --size small \
    --batch_size 4 \
    --block_size 2048 \
    --output_dir outputs/compare_deltanet_flash_2000steps \
    --optimizer_type lion \
    --learning_rate 1e-4 \
    --weight_decay 0.1 \
    --warmup_iters 200 \
    --max_iters 2000 \
    --grad_clip 1.0 \
    --num_workers 4 \
    --swa_layers_per_cycle 2 \
    --mla_layers_per_cycle 1 \
    --tokenizer_name "openai-community/gpt2" \
    --log_interval 50 \
    --eval_interval 500 \
    --use_flash_attention \
    --use_gated_deltanet \
    --use_tensorboard

END_DELTANET=$(date +%s)
TIME_DELTANET=$((END_DELTANET - START_DELTANET))

echo ""
echo "============================================================"
echo "DeltaNet + FlashAttention completed in ${TIME_DELTANET}s"
echo "============================================================"
echo ""

# Summary
echo ""
echo "============================================================"
echo "COMPARISON SUMMARY"
echo "============================================================"
echo ""
echo "Training Time:"
echo "  SWA + FlashAttn:      ${TIME_SWA}s"
echo "  DeltaNet + FlashAttn: ${TIME_DELTANET}s"
echo ""
if [ $TIME_SWA -lt $TIME_DELTANET ]; then
    SPEEDUP=$(echo "scale=2; $TIME_DELTANET / $TIME_SWA" | bc)
    echo "  Winner: SWA + FlashAttn (${SPEEDUP}x faster)"
else
    SPEEDUP=$(echo "scale=2; $TIME_SWA / $TIME_DELTANET" | bc)
    echo "  Winner: DeltaNet + FlashAttn (${SPEEDUP}x faster)"
fi
echo ""
echo "TensorBoard logs:"
echo "  SWA:      $OUTPUT_SWA/tensorboard"
echo "  DeltaNet: $OUTPUT_DELTANET/tensorboard"
echo ""
echo "To compare loss curves:"
echo "  tensorboard --logdir_spec swa:$OUTPUT_SWA/tensorboard,deltanet:$OUTPUT_DELTANET/tensorboard"
echo ""
echo "============================================================"
