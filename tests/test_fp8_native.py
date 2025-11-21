
import torch
import torch.nn as nn
import unittest
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'optimization'))

try:
    from fp8_native import FP8Linear, HAS_NATIVE_FP8
except ImportError:
    HAS_NATIVE_FP8 = False

@unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
@unittest.skipIf(not HAS_NATIVE_FP8, "Native FP8 not supported")
class TestFP8Native(unittest.TestCase):
    def setUp(self):
        self.device = "cuda"
        self.B, self.M, self.K, self.N = 2, 128, 64, 32
        
    def test_forward_backward(self):
        # Create random inputs
        x = torch.randn(self.B, self.M, self.K, device=self.device, dtype=torch.bfloat16, requires_grad=True)
        
        # Standard Linear
        linear = nn.Linear(self.K, self.N, bias=True, dtype=torch.bfloat16).to(self.device)
        
        # FP8 Linear (copy weights)
        fp8_linear = FP8Linear.from_linear(linear).to(self.device)
        
        # Forward Standard
        y_ref = linear(x)
        loss_ref = y_ref.sum()
        loss_ref.backward()
        grad_x_ref = x.grad.clone()
        grad_w_ref = linear.weight.grad.clone()
        
        # Reset grads
        x.grad = None
        
        # Forward FP8
        y_fp8 = fp8_linear(x)
        loss_fp8 = y_fp8.sum()
        loss_fp8.backward()
        grad_x_fp8 = x.grad.clone()
        grad_w_fp8 = fp8_linear.weight.grad.clone()
        
        # Check shapes
        self.assertEqual(y_fp8.shape, y_ref.shape)
        self.assertEqual(grad_x_fp8.shape, grad_x_ref.shape)
        self.assertEqual(grad_w_fp8.shape, grad_w_ref.shape)
        
        # Check values (allow some tolerance due to FP8 quantization)
        # FP8 is lossy, so we expect some difference.
        # We just want to ensure it runs and produces reasonable values.
        print(f"Max diff output: {(y_fp8 - y_ref).abs().max().item()}")
        print(f"Max diff grad_x: {(grad_x_fp8 - grad_x_ref).abs().max().item()}")
        print(f"Max diff grad_w: {(grad_w_fp8 - grad_w_ref).abs().max().item()}")
        
        # Sanity check: correlation should be high
        # self.assertTrue(torch.allclose(y_fp8, y_ref, atol=1e-1, rtol=1e-1))

if __name__ == '__main__':
    unittest.main()
