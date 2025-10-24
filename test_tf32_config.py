#!/usr/bin/env python3
"""
Test script to verify TF32 configuration and GPU auto-detection.
"""

import torch
import sys
from pathlib import Path

# Add models path
sys.path.insert(0, str(Path(__file__).parent / 'models'))
sys.path.insert(0, str(Path(__file__).parent))

from train import configure_tf32

def test_gpu_detection():
    """Test GPU detection and capability."""
    print("=" * 60)
    print("GPU Detection Test")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("❌ No CUDA available")
        return False

    num_gpus = torch.cuda.device_count()
    print(f"✓ Detected {num_gpus} GPU(s)")

    for i in range(num_gpus):
        name = torch.cuda.get_device_name(i)
        capability = torch.cuda.get_device_capability(i)
        compute_cap = f"{capability[0]}.{capability[1]}"
        supports_tf32 = capability[0] >= 8

        print(f"\nGPU {i}:")
        print(f"  - Name: {name}")
        print(f"  - Compute Capability: {compute_cap}")
        print(f"  - TF32 Support: {'✓ Yes' if supports_tf32 else '❌ No (requires CC 8.0+)'}")

    print("")
    return True


def test_tf32_configuration():
    """Test TF32 configuration."""
    print("=" * 60)
    print("TF32 Configuration Test")
    print("=" * 60)
    print("")

    # Test enabling TF32
    print("Test 1: Enabling TF32")
    print("-" * 40)
    configure_tf32(enable_tf32=True, verbose=True)
    print("")

    # Test disabling TF32
    print("Test 2: Disabling TF32")
    print("-" * 40)
    configure_tf32(enable_tf32=False, verbose=True)
    print("")

    # Re-enable for actual training
    print("Test 3: Re-enabling TF32 (for training)")
    print("-" * 40)
    configure_tf32(enable_tf32=True, verbose=True)
    print("")


def test_ddp_simulation():
    """Simulate DDP detection logic."""
    print("=" * 60)
    print("DDP Auto-Detection Simulation")
    print("=" * 60)

    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0

    print(f"\nDetected {num_gpus} GPU(s)")

    if num_gpus > 1:
        print(f"✓ Would launch DDP training with: torchrun --nproc_per_node={num_gpus}")
        print(f"  - {num_gpus} processes will be spawned")
        print(f"  - Each process will use 1 GPU")
    elif num_gpus == 1:
        print("✓ Would launch single-GPU training with: python train.py")
    else:
        print("✓ Would launch CPU training with: python train.py")

    print("")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("SWA-MLA Training Environment Test")
    print("=" * 60)
    print("")

    # Run tests
    gpu_ok = test_gpu_detection()

    if gpu_ok:
        test_tf32_configuration()
        test_ddp_simulation()

    print("=" * 60)
    print("Test Complete!")
    print("=" * 60)
    print("\nYou can now run training with:")
    print("  ./scripts/train_swa_mla.sh small 8 2048")
    print("\nThe script will automatically:")
    print("  - Detect available GPUs")
    print("  - Launch DDP if multiple GPUs are found")
    print("  - Configure TF32 for optimal performance on Ampere+ GPUs")
    print("")
