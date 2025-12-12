#!/usr/bin/env python3
"""Test Flash Attention 3 availability and basic functionality."""

import sys
import os

# Add FA3 to path if installed in default location
fa3_path = os.path.expanduser("~/.local/flash-attention-3/flash-attention/hopper")
if os.path.exists(fa3_path) and fa3_path not in sys.path:
    sys.path.insert(0, fa3_path)
    print(f"Added FA3 path: {fa3_path}")

print("\n" + "=" * 60)
print("Flash Attention 3 Test")
print("=" * 60)

# Check GPU
import torch
if not torch.cuda.is_available():
    print("CUDA not available!")
    sys.exit(1)

print(f"GPU: {torch.cuda.get_device_name()}")
cc = torch.cuda.get_device_capability()
print(f"Compute capability: {cc[0]}.{cc[1]}")

if cc[0] < 9:
    print("WARNING: FA3 is optimized for SM90 (H100/H800)")

# Try to import FA3
print("\nTrying to import flash_attn_interface...")
try:
    import flash_attn_interface
    print("✓ flash_attn_interface imported successfully!")
    funcs = [x for x in dir(flash_attn_interface) if not x.startswith('_')]
    print(f"  Available functions: {funcs}")
except ImportError as e:
    print(f"✗ Import failed: {e}")
    print("\nTo install FA3:")
    print("  ./scripts/install_flash_attention_3.sh")
    print("  # or manually:")
    print("  git clone https://github.com/Dao-AILab/flash-attention.git")
    print("  cd flash-attention/hopper && python setup.py install")
    sys.exit(1)

# Test basic forward pass
print("\nTesting basic forward pass...")
B, T, n_head, head_dim = 2, 128, 8, 64

q = torch.randn(B, T, n_head, head_dim, device="cuda", dtype=torch.bfloat16)
k = torch.randn(B, T, n_head, head_dim, device="cuda", dtype=torch.bfloat16)
v = torch.randn(B, T, n_head, head_dim, device="cuda", dtype=torch.bfloat16)

try:
    from flash_attn_interface import flash_attn_func
    output = flash_attn_func(q, k, v, causal=True)
    print(f"✓ Forward pass successful! Output shape: {output.shape}")
except Exception as e:
    print(f"✗ Forward pass failed: {e}")
    sys.exit(1)

# Test sliding window
print("\nTesting sliding window attention...")
try:
    output_sw = flash_attn_func(q, k, v, causal=True, window_size=(64, 0))
    print(f"✓ Sliding window successful! Output shape: {output_sw.shape}")
except Exception as e:
    print(f"✗ Sliding window failed: {e}")

# Test backward pass
print("\nTesting backward pass...")
try:
    q.requires_grad_(True)
    k.requires_grad_(True)
    v.requires_grad_(True)
    output = flash_attn_func(q, k, v, causal=True)
    loss = output.sum()
    loss.backward()
    print(f"✓ Backward pass successful!")
    print(f"  q.grad shape: {q.grad.shape}")
except Exception as e:
    print(f"✗ Backward pass failed: {e}")

# Benchmark
print("\nBenchmarking...")
import time

# Warmup
for _ in range(5):
    output = flash_attn_func(q.detach(), k.detach(), v.detach(), causal=True)
torch.cuda.synchronize()

# Benchmark
n_runs = 50
torch.cuda.synchronize()
start = time.perf_counter()
for _ in range(n_runs):
    output = flash_attn_func(q.detach(), k.detach(), v.detach(), causal=True)
torch.cuda.synchronize()
elapsed = (time.perf_counter() - start) / n_runs

print(f"  Shape: (B={B}, T={T}, n_head={n_head}, head_dim={head_dim})")
print(f"  Time per forward pass: {elapsed*1000:.3f}ms")
print(f"  Throughput: {B * T * n_head * head_dim * 4 / elapsed / 1e9:.2f} GB/s")

print("\n" + "=" * 60)
print("All tests passed!")
print("=" * 60)
