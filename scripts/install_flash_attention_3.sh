#!/bin/bash

# =============================================================================
# Flash Attention 3 Installation Script for H100/H800 GPUs
# =============================================================================
# Requirements:
#   - H100 or H800 GPU
#   - CUDA >= 12.3 (CUDA 12.8 recommended for best performance)
#   - Python 3.8+
#   - PyTorch 2.0+
#
# Features supported:
#   - FP16 / BF16 forward and backward
#   - FP8 forward
# =============================================================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}================================================${NC}"
echo -e "${BLUE}  Flash Attention 3 Installer for H100/H800${NC}"
echo -e "${BLUE}================================================${NC}"
echo ""

# -----------------------------------------------------------------------------
# Check prerequisites
# -----------------------------------------------------------------------------

echo -e "${YELLOW}[1/6] Checking prerequisites...${NC}"

# Check for NVIDIA GPU
if ! command -v nvidia-smi &> /dev/null; then
    echo -e "${RED}ERROR: nvidia-smi not found. Please install NVIDIA drivers.${NC}"
    exit 1
fi

# Get GPU name
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1)
echo "  GPU detected: $GPU_NAME"

# Check for H100/H800
if [[ ! "$GPU_NAME" =~ "H100" ]] && [[ ! "$GPU_NAME" =~ "H800" ]]; then
    echo -e "${YELLOW}  WARNING: Flash Attention 3 is optimized for H100/H800 GPUs.${NC}"
    echo -e "${YELLOW}  Your GPU ($GPU_NAME) may not be fully supported.${NC}"
    read -p "  Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check CUDA version
CUDA_VERSION=$(nvcc --version 2>/dev/null | grep "release" | sed -n 's/.*release \([0-9]*\.[0-9]*\).*/\1/p')
if [ -z "$CUDA_VERSION" ]; then
    echo -e "${RED}ERROR: nvcc not found. Please install CUDA toolkit.${NC}"
    exit 1
fi

echo "  CUDA version: $CUDA_VERSION"

# Parse CUDA version
CUDA_MAJOR=$(echo $CUDA_VERSION | cut -d. -f1)
CUDA_MINOR=$(echo $CUDA_VERSION | cut -d. -f2)

if [ "$CUDA_MAJOR" -lt 12 ] || ([ "$CUDA_MAJOR" -eq 12 ] && [ "$CUDA_MINOR" -lt 3 ]); then
    echo -e "${RED}ERROR: CUDA >= 12.3 required. Found $CUDA_VERSION${NC}"
    exit 1
fi

if [ "$CUDA_MAJOR" -eq 12 ] && [ "$CUDA_MINOR" -lt 8 ]; then
    echo -e "${YELLOW}  NOTE: CUDA 12.8+ recommended for best performance.${NC}"
fi

# Check Python
PYTHON_VERSION=$(python3 --version 2>&1 | sed 's/Python //')
echo "  Python version: $PYTHON_VERSION"

# Check PyTorch
PYTORCH_VERSION=$(python3 -c "import torch; print(torch.__version__)" 2>/dev/null || echo "NOT FOUND")
if [ "$PYTORCH_VERSION" = "NOT FOUND" ]; then
    echo -e "${RED}ERROR: PyTorch not found. Please install PyTorch first.${NC}"
    exit 1
fi
echo "  PyTorch version: $PYTORCH_VERSION"

echo -e "${GREEN}  Prerequisites OK!${NC}"
echo ""

# -----------------------------------------------------------------------------
# Setup installation directory
# -----------------------------------------------------------------------------

echo -e "${YELLOW}[2/6] Setting up installation directory...${NC}"

INSTALL_DIR="${HOME}/.local/flash-attention-3"
mkdir -p "$INSTALL_DIR"
cd "$INSTALL_DIR"

echo "  Installation directory: $INSTALL_DIR"
echo ""

# -----------------------------------------------------------------------------
# Clone or update repository
# -----------------------------------------------------------------------------

echo -e "${YELLOW}[3/6] Cloning Flash Attention repository...${NC}"

if [ -d "flash-attention" ]; then
    echo "  Repository exists, updating..."
    cd flash-attention
    git fetch origin
    git pull origin main
else
    echo "  Cloning repository..."
    git clone https://github.com/Dao-AILab/flash-attention.git
    cd flash-attention
fi

echo -e "${GREEN}  Repository ready!${NC}"
echo ""

# -----------------------------------------------------------------------------
# Build Flash Attention 3 (Hopper)
# -----------------------------------------------------------------------------

echo -e "${YELLOW}[4/6] Building Flash Attention 3 (Hopper)...${NC}"
echo "  This may take several minutes..."

cd hopper

# Clean previous builds
if [ -d "build" ]; then
    echo "  Cleaning previous build..."
    rm -rf build
fi

# Set environment for build
export TORCH_CUDA_ARCH_LIST="9.0"  # H100 compute capability
export MAX_JOBS=${MAX_JOBS:-$(nproc)}

echo "  Building with $MAX_JOBS parallel jobs..."
python setup.py install

echo -e "${GREEN}  Build complete!${NC}"
echo ""

# -----------------------------------------------------------------------------
# Verify installation
# -----------------------------------------------------------------------------

echo -e "${YELLOW}[5/6] Verifying installation...${NC}"

cd "$INSTALL_DIR/flash-attention/hopper"
export PYTHONPATH="$INSTALL_DIR/flash-attention/hopper:$PYTHONPATH"

python3 -c "
import flash_attn_interface
print('  flash_attn_interface imported successfully!')
print(f'  Available functions: {[x for x in dir(flash_attn_interface) if not x.startswith(\"_\")]}')
" || {
    echo -e "${RED}ERROR: Failed to import flash_attn_interface${NC}"
    exit 1
}

echo -e "${GREEN}  Installation verified!${NC}"
echo ""

# -----------------------------------------------------------------------------
# Run tests (optional)
# -----------------------------------------------------------------------------

echo -e "${YELLOW}[6/6] Running tests (optional)...${NC}"
read -p "  Run pytest tests? This may take a while. (y/n) " -n 1 -r
echo

if [[ $REPLY =~ ^[Yy]$ ]]; then
    cd "$INSTALL_DIR/flash-attention/hopper"
    export PYTHONPATH="$INSTALL_DIR/flash-attention/hopper:$PYTHONPATH"
    pytest -q -s test_flash_attn.py || echo -e "${YELLOW}  Some tests may have failed (this is expected for beta)${NC}"
else
    echo "  Skipping tests."
fi

echo ""

# -----------------------------------------------------------------------------
# Print usage instructions
# -----------------------------------------------------------------------------

echo -e "${GREEN}================================================${NC}"
echo -e "${GREEN}  Installation Complete!${NC}"
echo -e "${GREEN}================================================${NC}"
echo ""
echo -e "${BLUE}To use Flash Attention 3 in your code:${NC}"
echo ""
echo "  # Add to your Python script or shell environment:"
echo "  export PYTHONPATH=\"$INSTALL_DIR/flash-attention/hopper:\$PYTHONPATH\""
echo ""
echo "  # Or add to ~/.bashrc for permanent access:"
echo "  echo 'export PYTHONPATH=\"$INSTALL_DIR/flash-attention/hopper:\$PYTHONPATH\"' >> ~/.bashrc"
echo ""
echo -e "${BLUE}Example usage in Python:${NC}"
echo ""
echo "  import flash_attn_interface"
echo "  flash_attn_interface.flash_attn_func(q, k, v, causal=True)"
echo ""
echo -e "${BLUE}For use with swamla, add --use_flash_attention to your training command.${NC}"
echo ""

# Create activation script
cat > "$INSTALL_DIR/activate_fa3.sh" << 'EOF'
#!/bin/bash
# Source this script to activate Flash Attention 3
export PYTHONPATH="$HOME/.local/flash-attention-3/flash-attention/hopper:$PYTHONPATH"
echo "Flash Attention 3 activated!"
EOF
chmod +x "$INSTALL_DIR/activate_fa3.sh"

echo -e "${YELLOW}Quick activation script created:${NC}"
echo "  source $INSTALL_DIR/activate_fa3.sh"
echo ""
