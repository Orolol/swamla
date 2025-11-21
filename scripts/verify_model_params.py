
import sys
import os
import torch

# Add the current directory to the path so we can import the modules
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), "models"))

from models.swa_mla_model import create_swa_mla_model, SWALocalBlock, MLASelectiveBlock, MLABlock

def verify_scaling():
    print("Verifying SWA Window and MLA Rank Scaling...")
    
    # Create a model with known config
    # Base window: 256
    # Base rank: 512
    # Layers: 10 (for easy calculation)
    # SWA per cycle: 1
    # MLA per cycle: 1
    # Cycle length: 2
    # Layers: 0 (SWA), 1 (MLA), 2 (SWA), 3 (MLA), ...
    
    n_layer = 10
    base_window = 256
    base_rank = 512
    
    model = create_swa_mla_model(
        size="small", # will be overridden
        n_layer=n_layer,
        n_embd=256,
        n_head=4,
        swa_window=base_window,
        kv_lora_rank=base_rank,
        swa_layers_per_cycle=1,
        mla_layers_per_cycle=1
    )
    
    print(f"Model created with {n_layer} layers.")
    print(f"Base Window: {base_window}")
    print(f"Base Rank: {base_rank}")
    
    for i, block in enumerate(model.transformer.h):
        # Expected factor
        # factor = 0.5 + 1.5 * (i / (n_layer - 1))
        expected_factor = 0.5 + 1.5 * (i / (n_layer - 1))
        
        if isinstance(block, SWALocalBlock):
            actual_window = block.attn.attention_window
            expected_window = int(base_window * expected_factor)
            print(f"Layer {i} (SWA): Factor={expected_factor:.4f}, Window={actual_window} (Expected: {expected_window})")
            
            if actual_window != expected_window:
                print(f"ERROR: Layer {i} SWA Window mismatch!")
                return False
                
        elif isinstance(block, (MLABlock, MLASelectiveBlock)):
            # For MLABlock, the rank is in block.attn.kv_lora_rank
            # For MLASelectiveBlock, it might be different, but we are using standard MLA here effectively
            actual_rank = block.attn.kv_lora_rank
            expected_rank = int(base_rank * expected_factor)
            print(f"Layer {i} (MLA): Factor={expected_factor:.4f}, Rank={actual_rank} (Expected: {expected_rank})")
            
            if actual_rank != expected_rank:
                print(f"ERROR: Layer {i} MLA Rank mismatch!")
                return False
        else:
            print(f"Layer {i}: Unknown block type")
            
    print("Verification PASSED!")
    return True

if __name__ == "__main__":
    success = verify_scaling()
    if not success:
        sys.exit(1)
