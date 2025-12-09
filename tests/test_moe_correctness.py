
import torch
import sys
from pathlib import Path

# Add models to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'models'))

from moe import MoELayer

# Config mock
class Config:
    n_embd = 128
    n_experts = 4
    n_shared_experts = 1
    n_activated = 2
    expert_dim = 128
    bias = False
    router_z_loss_coef = 0.001
    dropout = 0.0

def test_correctness():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    torch.manual_seed(42)
    
    config = Config()
    model = MoELayer(config).to(device).float() # FP32 for precision check
    model.eval()
    
    batch_size = 2
    seq_len = 10
    x = torch.randn(batch_size, seq_len, config.n_embd, device=device)
    
    # Run once (should use Triton if available)
    print("Running forward pass...")
    output_triton = model(x)
    
    # Force disable Triton (hack: set TRITON_AVAILABLE to False in moe module)
    import moe as moe_module
    moe_module.TRITON_AVAILABLE = False
    print("Running forward pass (Python loop)...")
    output_python = model(x)
    
    # Compare
    diff = (output_triton - output_python).abs().max().item()
    print(f"Max difference: {diff}")
    
    if diff < 1e-4:
        print("PASS: Outputs match")
    else:
        print("FAIL: Outputs mismatch")

if __name__ == "__main__":
    test_correctness()
