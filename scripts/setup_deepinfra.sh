#!/bin/bash
# =============================================================================
# Setup script for DeepInfra instances (Ubuntu 22.04 + CUDA driver only)
# Installs all dependencies including Transformer Engine
# =============================================================================
set -e

echo "=== DeepInfra Setup for SWA-MLA ==="

# 1. Fix PATH for pip --user installs and python alias
export PATH=$HOME/.local/bin:$PATH
if ! command -v python &>/dev/null; then
    echo "Creating python -> python3 symlink..."
    sudo ln -sf /usr/bin/python3 /usr/bin/python 2>/dev/null || alias python=python3
fi

# 2. Install system build dependencies for Transformer Engine
echo "=== Installing system dependencies ==="
sudo apt-get update -qq
sudo apt-get install -y -qq \
    ninja-build \
    libcudnn8-dev libcudnn-dev \
    libnccl-dev \
    2>/dev/null || echo "Warning: some system packages may have failed, continuing..."

# 3. Install CUDA toolkit if nvcc is missing
if ! command -v nvcc &>/dev/null && [ ! -f /usr/local/cuda/bin/nvcc ]; then
    echo "=== Installing CUDA toolkit ==="
    wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
    sudo dpkg -i cuda-keyring_1.1-1_all.deb
    sudo apt-get update -qq
    sudo apt-get install -y -qq cuda-toolkit-12-8
    rm -f cuda-keyring_1.1-1_all.deb
fi

# 4. Set environment variables
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
export PATH=$CUDA_HOME/bin:$PATH

# Add NVIDIA Python package lib paths for cuDNN/NCCL shared objects
NVIDIA_SITE_PKGS=$(python -c "import site; print([p for p in site.getsitepackages() + [site.getusersitepackages()] if 'nvidia' not in p][0])" 2>/dev/null || echo "/usr/local/lib/python3.10/dist-packages")
CUDNN_LIB=$(find "$NVIDIA_SITE_PKGS" "$HOME/.local/lib" /usr/local/lib -path "*/nvidia/cudnn/lib" -type d 2>/dev/null | head -1)
NCCL_LIB=$(find "$NVIDIA_SITE_PKGS" "$HOME/.local/lib" /usr/local/lib -path "*/nvidia/nccl/lib" -type d 2>/dev/null | head -1)

if [ -n "$CUDNN_LIB" ]; then
    export LD_LIBRARY_PATH=$CUDNN_LIB:$LD_LIBRARY_PATH
fi
if [ -n "$NCCL_LIB" ]; then
    export LD_LIBRARY_PATH=$NCCL_LIB:$LD_LIBRARY_PATH
fi

# 5. Install Python requirements
echo "=== Installing Python requirements ==="
pip install -r requirements.txt

# 6. Install Transformer Engine separately (heavy C++ build)
echo "=== Installing Transformer Engine ==="
pip install "transformer-engine[pytorch]>=2.0.0"

# 7. Persist environment in .bashrc
echo "=== Persisting environment ==="
grep -q 'swamla setup' ~/.bashrc 2>/dev/null || cat >> ~/.bashrc << 'BASHRC'

# --- swamla setup ---
export PATH=$HOME/.local/bin:/usr/local/cuda/bin:$PATH
export CUDA_HOME=/usr/local/cuda
# Auto-detect cuDNN/NCCL lib paths from pip packages
for pkg_dir in "$HOME/.local/lib" "/usr/local/lib"; do
    for lib_name in cudnn nccl; do
        lib_path=$(find "$pkg_dir" -path "*/nvidia/$lib_name/lib" -type d 2>/dev/null | head -1)
        [ -n "$lib_path" ] && export LD_LIBRARY_PATH=$lib_path:$LD_LIBRARY_PATH
    done
done
BASHRC

echo ""
echo "=== Setup complete ==="
echo "Run: ./scripts/train.sh --preset full 48 2048"
