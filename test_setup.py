"""
Quick test to verify the SWA-MLA setup is working correctly.
"""

import sys
from pathlib import Path

# Add modules to path
sys.path.insert(0, str(Path(__file__).parent / 'models'))
sys.path.insert(0, str(Path(__file__).parent / 'data'))
sys.path.insert(0, str(Path(__file__).parent / 'optimization'))

print("Testing SWA-MLA standalone setup...\n")

# Test 1: Model imports
print("1. Testing model imports...")
try:
    from swa_mla_model import create_swa_mla_model, SWAMLAConfig
    from mla import MLA
    from attention import CausalSelfAttention
    from mlp import MLP
    from normalization import RMSNorm, DynamicTanh
    from positional_encoding import RoPE, precompute_freqs_cis
    print("   ✓ All model imports successful")
except Exception as e:
    print(f"   ✗ Model import failed: {e}")
    sys.exit(1)

# Test 2: Data loader imports
print("2. Testing data loader imports...")
try:
    from data_loader_packed import PackedFinewebDataset
    print("   ✓ Data loader import successful")
except Exception as e:
    print(f"   ✗ Data loader import failed: {e}")
    sys.exit(1)

# Test 3: Optimization imports
print("3. Testing optimization imports...")
try:
    from fp8_trainer import FP8AdamW, FP8Lion
    print("   ✓ Optimization imports successful")
except Exception as e:
    print(f"   ✗ Optimization import failed: {e}")
    sys.exit(1)

# Test 4: PyTorch availability
print("4. Testing PyTorch...")
try:
    import torch
    print(f"   ✓ PyTorch {torch.__version__} available")
    if torch.cuda.is_available():
        print(f"   ✓ CUDA available: {torch.cuda.get_device_name(0)}")
    else:
        print("   ⚠ CUDA not available (CPU only)")
except Exception as e:
    print(f"   ✗ PyTorch test failed: {e}")
    sys.exit(1)

# Test 5: Create a small model
print("5. Testing model creation...")
try:
    import torch
    model = create_swa_mla_model(
        size='small',
        vocab_size=1000,
        block_size=512,
        dropout=0.0,
    )
    param_count = sum(p.numel() for p in model.parameters())
    print(f"   ✓ Model created successfully ({param_count/1e6:.1f}M parameters)")
except Exception as e:
    print(f"   ✗ Model creation failed: {e}")
    sys.exit(1)

# Test 6: Test forward pass
print("6. Testing forward pass...")
try:
    x = torch.randint(0, 1000, (2, 128))  # batch_size=2, seq_len=128
    with torch.no_grad():
        logits, loss = model(x, targets=x)
    print(f"   ✓ Forward pass successful (logits shape: {logits.shape})")
except Exception as e:
    print(f"   ✗ Forward pass failed: {e}")
    sys.exit(1)

# Test 7: Check optional dependencies
print("7. Checking optional dependencies...")
optional_deps = {
    'wandb': 'Experiment tracking',
    'flash_attn': 'Flash Attention 2',
    'xformers': 'Memory efficient attention',
    'transformer_engine': 'FP8 training (H100/H200)',
    'lion_pytorch': 'Lion optimizer',
}

for dep, description in optional_deps.items():
    try:
        __import__(dep)
        print(f"   ✓ {dep}: {description}")
    except ImportError:
        print(f"   ⚠ {dep}: {description} (not installed)")

print("\n" + "="*60)
print("✓ All core tests passed! Setup is working correctly.")
print("="*60)
print("\nYou can now start training with:")
print("  ./scripts/train_swa_mla.sh small 8 2048")
print("\nOr with Python directly:")
print("  python train.py --size small --batch_size 8 --block_size 2048")
