
import torch
import unittest
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'optimization'))

try:
    from muon import Muon
except ImportError:
    Muon = None

@unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
@unittest.skipIf(Muon is None, "Muon module not found")
class TestMuon(unittest.TestCase):
    def setUp(self):
        self.device = "cuda"
        
    def test_step(self):
        # Simple quadratic problem: min (x-1)^2
        # Optimum at x=1
        
        # Muon requires 2D parameters usually, let's try a 2D parameter
        # f(X) = ||X - I||^2
        
        N = 10
        X = torch.zeros(N, N, device=self.device, requires_grad=True)
        target = torch.eye(N, device=self.device)
        
        optimizer = Muon([X], lr=0.02, momentum=0.95)
        
        # Run a few steps
        initial_loss = (X - target).pow(2).sum().item()
        
        for _ in range(20):
            optimizer.zero_grad()
            loss = (X - target).pow(2).sum()
            loss.backward()
            optimizer.step()
            
        final_loss = (X - target).pow(2).sum().item()
        
        print(f"Initial Loss: {initial_loss}, Final Loss: {final_loss}")
        self.assertLess(final_loss, initial_loss)

if __name__ == '__main__':
    unittest.main()
