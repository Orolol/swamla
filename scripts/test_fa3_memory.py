#!/usr/bin/env python3
"""Compare memory usage between FA3 and SDPA."""

import torch
import gc

B, T, H, D = 8, 2048, 12, 128  # Dimensions typiques MLA

print(f"Test config: B={B}, T={T}, H={H}, D={D}")
print(f"GPU: {torch.cuda.get_device_name()}")
print()

q = torch.randn(B, T, H, D, device='cuda', dtype=torch.bfloat16)
k = torch.randn(B, T, H, D, device='cuda', dtype=torch.bfloat16)
v = torch.randn(B, T, H, D, device='cuda', dtype=torch.bfloat16)

torch.cuda.reset_peak_memory_stats()
torch.cuda.empty_cache()
gc.collect()

print(f"Avant: {torch.cuda.memory_allocated()/1e9:.2f} GB")

# Test FA3
print("\n--- Test FA3 ---")
from flash_attn_interface import flash_attn_func
for _ in range(10):
    out = flash_attn_func(q, k, v, causal=True)

print(f"Apres FA3: {torch.cuda.memory_allocated()/1e9:.2f} GB")
print(f"Peak FA3: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")

del out
torch.cuda.reset_peak_memory_stats()
torch.cuda.empty_cache()
gc.collect()

# Test SDPA
print("\n--- Test SDPA ---")
print(f"Avant SDPA: {torch.cuda.memory_allocated()/1e9:.2f} GB")

q_sdpa = q.transpose(1, 2)  # BTHD -> BHTD
k_sdpa = k.transpose(1, 2)
v_sdpa = v.transpose(1, 2)
for _ in range(10):
    out = torch.nn.functional.scaled_dot_product_attention(q_sdpa, k_sdpa, v_sdpa, is_causal=True)

print(f"Apres SDPA: {torch.cuda.memory_allocated()/1e9:.2f} GB")
print(f"Peak SDPA: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")

print("\n--- Done ---")
