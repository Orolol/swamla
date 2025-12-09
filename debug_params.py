import torch
import sys
import os
sys.path.insert(0, os.path.abspath('models'))
from swa_mla_model import create_swa_mla_model

def count_params(model):
    return sum(p.numel() for p in model.parameters())

def print_model_stats(model):
    print(f"Total params: {count_params(model):,}")
    
    # Count by component
    embeddings = sum(p.numel() for n, p in model.named_parameters() if 'wte' in n or 'lm_head' in n)
    print(f"Embeddings: {embeddings:,}")
    
    # Count MoE
    moe_params = sum(p.numel() for n, p in model.named_parameters() if 'experts' in n)
    print(f"MoE Experts: {moe_params:,}")
    
    # Count others
    other = count_params(model) - embeddings - moe_params
    print(f"Other: {other:,}")

    # Check config
    print(f"Config: n_experts={model.config.n_experts}, use_moe={model.config.use_moe}")
    print(f"Config: n_layer={model.config.n_layer}, n_embd={model.config.n_embd}")

print("--- moe-1b default ---")
model = create_swa_mla_model(size='moe-1b')
print_model_stats(model)

print("\n--- moe-1b with explicit args (simulating train.py) ---")
# Simulate train.py args
model2 = create_swa_mla_model(
    size='moe-1b',
    use_moe=True,
    n_experts=32,
    n_shared_experts=1,
    n_activated=3
)
print_model_stats(model2)
