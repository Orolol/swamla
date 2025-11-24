#!/bin/bash
set -e

echo "Starting installation..."

# Install system dependencies if needed
apt-get update && apt-get install -y git build-essential

git clone https://github.com/orosius/swamla.git

pip install -r swamla/requirements.txt

# Environment variables HF_TOKEN and WANDB_API_KEY should be set before running this script

echo "Installation complete!"

cd swamla
./scripts/train_swa_mla.sh small 48 2048 monoswamla true lion orosius/swamla

