
import torch
import time
import sys
from pathlib import Path

# Add models to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'models'))

from moe import MoELayer

# Config mock
class Config:
    n_embd = 4096
    n_experts = 64
    n_shared_experts = 1
    n_activated = 6
    expert_dim = 4096
    bias = False
    router_z_loss_coef = 0.001
    dropout = 0.0

def benchmark():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    config = Config()
    model = MoELayer(config).to(device).half() # FP16
    
    batch_size = 16
    seq_len = 1024
    x = torch.randn(batch_size, seq_len, config.n_embd, device=device, dtype=torch.float16)
    
    print(f"Input shape: {x.shape}")
    print(f"Experts: {config.n_experts}, Activated: {config.n_activated}")
    
    # Warmup
    for _ in range(10):
        _ = model(x)
    
    torch.cuda.synchronize()
    start = time.time()
    iters = 50
    for _ in range(iters):
        _ = model(x)
    torch.cuda.synchronize()
    end = time.time()
    
    avg_time = (end - start) / iters
    print(f"Average time per forward pass: {avg_time*1000:.2f} ms")
    
    # Estimate TFLOPS (very rough)
    # 2 * batch * seq * top_k * (2 * d_model * expert_dim + expert_dim * d_model)
    # = 2 * B * S * K * 3 * D^2
    flops = 2 * batch_size * seq_len * config.n_activated * 3 * (config.n_embd ** 2)
    tflops = (flops / avg_time) / 1e12
    print(f"Estimated TFLOPS: {tflops:.2f}")

if __name__ == "__main__":
    benchmark()
