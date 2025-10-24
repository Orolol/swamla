#!/usr/bin/env python3
"""Quick test for TF32 configuration."""

import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from train import configure_tf32

print("Testing TF32 configuration...")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    cap = torch.cuda.get_device_capability(0)
    print(f"Compute Capability: {cap[0]}.{cap[1]}")
    print("")

    print("Enabling TF32...")
    configure_tf32(enable_tf32=True, verbose=True)
    print("")

    print("Legacy API values after configuration:")
    print(f"  matmul.allow_tf32: {torch.backends.cuda.matmul.allow_tf32}")
    print(f"  cudnn.allow_tf32: {torch.backends.cudnn.allow_tf32}")
    print("")
    print("✓ TF32 configuration successful!")
else:
    print("No CUDA available")
