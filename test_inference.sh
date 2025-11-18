#!/bin/bash

# Script to test inference with the SWA-MLA model

echo "==================================================================="
echo "SWA-MLA Model Inference Testing"
echo "==================================================================="

# Configuration
HF_REPO="Orosius/swamla"  # Your HuggingFace repo
DEVICE="cuda"
DTYPE="bfloat16"
MAX_TOKENS=256
TEMPERATURE=0.8

# Check if HF token is set
if [ -z "$HF_TOKEN" ]; then
    echo "Note: HF_TOKEN not set. If the repo is private, set it with: export HF_TOKEN=your_token"
fi

echo ""
echo "1. Testing BATCH MODE with standard prompts..."
echo "-------------------------------------------------------------------"
python inference.py \
    --hf_repo_id "$HF_REPO" \
    --mode batch \
    --prompts_file prompts_standard.txt \
    --max_new_tokens $MAX_TOKENS \
    --temperature $TEMPERATURE \
    --device $DEVICE \
    --dtype $DTYPE

echo ""
echo "2. Testing CHAT MODE with ChatML format (for instruction-tuned model)..."
echo "-------------------------------------------------------------------"
echo "To test chat mode interactively, run:"
echo ""
echo "python inference.py \\"
echo "    --hf_repo_id $HF_REPO \\"
echo "    --mode chat \\"
echo "    --chatml \\"
echo "    --max_new_tokens $MAX_TOKENS \\"
echo "    --temperature $TEMPERATURE \\"
echo "    --device $DEVICE \\"
echo "    --dtype $DTYPE"

echo ""
echo "3. Testing with specific checkpoint from instruct/ subdirectory..."
echo "-------------------------------------------------------------------"
echo "To load a specific instruction-tuned checkpoint, run:"
echo ""
echo "python inference.py \\"
echo "    --hf_repo_id $HF_REPO \\"
echo "    --hf_checkpoint instruct/checkpoint_tokens_XXX_loss_Y.YYY \\"
echo "    --mode chat \\"
echo "    --chatml \\"
echo "    --max_new_tokens $MAX_TOKENS \\"
echo "    --device $DEVICE"

echo ""
echo "==================================================================="
echo "Quick test commands:"
echo ""
echo "# Standard generation (pre-trained model):"
echo "python inference.py --hf_repo_id $HF_REPO --mode batch"
echo ""
echo "# Interactive chat (instruction-tuned model):"
echo "python inference.py --hf_repo_id $HF_REPO --mode chat --chatml"
echo ""
echo "# With custom prompts:"
echo "python inference.py --hf_repo_id $HF_REPO --mode batch --prompts_file my_prompts.txt"
echo "==================================================================="