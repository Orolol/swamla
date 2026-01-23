"""Tests for Neural Long-term Memory module (Titans paper implementation)."""

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

# Add models to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'models'))

from neural_memory import NeuralLongTermMemory, MemoryState


class TestMemoryState:
    """Tests for MemoryState dataclass."""

    def test_init_from_module(self):
        """Test MemoryState initialization from module."""
        memory = NeuralLongTermMemory(
            d_input=64, memory_dim=32, memory_depth=2, d_output=64
        )
        batch_size = 4
        state = MemoryState.init_from_module(memory, batch_size)

        # Check that we have correct number of layers
        assert len(state.weights) == 2  # depth=2 means 2 layers
        assert len(state.momentum) == 2

        # Check shapes
        # Layer 0: 64 -> 32 (d_input -> memory_dim)
        assert state.weights[0][0].shape == (batch_size, 32, 64)  # weight
        assert state.weights[0][1].shape == (batch_size, 32)  # bias
        # Layer 1: 32 -> 64 (memory_dim -> d_output)
        assert state.weights[1][0].shape == (batch_size, 64, 32)
        assert state.weights[1][1].shape == (batch_size, 64)

        # Check momentum is zero-initialized
        assert torch.allclose(state.momentum[0][0], torch.zeros_like(state.momentum[0][0]))

    def test_detach(self):
        """Test that detach creates independent tensors."""
        memory = NeuralLongTermMemory(d_input=32, memory_dim=16, memory_depth=2)
        state = MemoryState.init_from_module(memory, batch_size=2)

        # Make weights require grad
        for w, b in state.weights:
            w.requires_grad = True
            b.requires_grad = True

        detached = state.detach()

        # Check that detached tensors don't require grad
        for w, b in detached.weights:
            assert not w.requires_grad
            assert not b.requires_grad

        # Check that tensors are different objects
        assert state.weights[0][0] is not detached.weights[0][0]

    def test_to_device(self):
        """Test moving state to different device."""
        memory = NeuralLongTermMemory(d_input=32, memory_dim=16, memory_depth=2)
        state = MemoryState.init_from_module(memory, batch_size=2)

        # Move to CPU (should work even if already on CPU)
        state_cpu = state.to(torch.device('cpu'))
        assert state_cpu.weights[0][0].device.type == 'cpu'


class TestNeuralLongTermMemory:
    """Tests for NeuralLongTermMemory module."""

    @pytest.fixture
    def memory_module(self):
        """Create a small memory module for testing."""
        return NeuralLongTermMemory(
            d_input=64,
            memory_dim=32,
            memory_depth=2,
            d_output=64,
            activation="silu"
        )

    def test_output_shape(self, memory_module):
        """Test that output shape matches expected."""
        batch_size, seq_len = 4, 16
        x = torch.randn(batch_size, seq_len, 64)

        output, state = memory_module(x)

        assert output.shape == (batch_size, seq_len, 64)
        assert len(state.weights) == 2

    def test_memory_state_updates(self, memory_module):
        """Test that memory weights change after processing."""
        batch_size = 2
        x = torch.randn(batch_size, 8, 64)

        # Get initial state
        state1 = MemoryState.init_from_module(memory_module, batch_size)
        initial_weights = state1.weights[0][0].clone()

        # Process through memory
        _, state2 = memory_module(x, state=state1)

        # Weights should have changed
        assert not torch.allclose(initial_weights, state2.weights[0][0])

    def test_dynamics_range(self, memory_module):
        """Test that dynamics outputs are in expected ranges."""
        x = torch.randn(4, 64)

        # theta (learning rate) should be positive via softplus
        theta = torch.nn.functional.softplus(memory_module.theta_net(x))
        assert (theta > 0).all()

        # eta (momentum) should be in [0, 1] via sigmoid
        eta = torch.sigmoid(memory_module.eta_net(x))
        assert (eta >= 0).all() and (eta <= 1).all()

        # alpha (forget) should be in [0, 1] via sigmoid
        alpha = torch.sigmoid(memory_module.alpha_net(x))
        assert (alpha >= 0).all() and (alpha <= 1).all()

    def test_forward_without_state(self, memory_module):
        """Test forward pass with no initial state."""
        x = torch.randn(2, 8, 64)
        output, state = memory_module(x, state=None)

        assert output.shape == x.shape
        assert state is not None

    def test_state_persistence(self, memory_module):
        """Test that state persists correctly across calls."""
        x1 = torch.randn(2, 8, 64)
        x2 = torch.randn(2, 8, 64)

        # First pass
        _, state1 = memory_module(x1)

        # Second pass with state from first
        output2, state2 = memory_module(x2, state=state1)

        # State should continue evolving
        assert not torch.allclose(state1.weights[0][0], state2.weights[0][0])

    def test_gradient_flow(self, memory_module):
        """Test that gradients flow through the module."""
        x = torch.randn(2, 8, 64, requires_grad=True)
        output, _ = memory_module(x)
        loss = output.sum()
        loss.backward()

        # Input should have gradients
        assert x.grad is not None
        assert x.grad.shape == x.shape

    def test_different_depths(self):
        """Test memory with different depths."""
        for depth in [2, 3, 4]:
            memory = NeuralLongTermMemory(
                d_input=32, memory_dim=16, memory_depth=depth, d_output=32
            )
            x = torch.randn(2, 4, 32)
            output, state = memory(x)

            assert output.shape == x.shape
            assert len(state.weights) == depth

    def test_different_activations(self):
        """Test different activation functions."""
        for activation in ["silu", "relu", "gelu"]:
            memory = NeuralLongTermMemory(
                d_input=32, memory_dim=16, memory_depth=2, activation=activation
            )
            x = torch.randn(2, 4, 32)
            output, _ = memory(x)
            assert output.shape == x.shape


class TestMLAMemoryBlock:
    """Tests for MLAMemoryBlock integration."""

    @pytest.fixture
    def config(self):
        """Create a minimal config for testing."""
        from dataclasses import dataclass

        @dataclass
        class TestConfig:
            n_embd: int = 256
            n_head: int = 4
            dropout: float = 0.0
            bias: bool = False
            use_gradient_checkpointing: bool = False
            use_dyt: bool = False
            dyt_alpha_init: float = 0.5
            q_lora_rank: int = 0
            kv_lora_rank: int = 64
            qk_nope_head_dim: int = 32
            qk_rope_head_dim: int = 16
            v_head_dim: int = 32
            attn_impl: str = "naive"
            world_size: int = 1
            dropout_p: float = 0.0
            max_seq_len: int = 512
            original_max_seq_len: int = 512
            rope_scaling: None = None
            rope_theta: float = 10000.0
            rope_factor: float = 1.0
            mscale: float = 1.0
            use_fp8: bool = False
            fp8_mla_params: bool = False
            fp8_tile_size: int = 128
            use_te_fp8: bool = False
            memory_dim: int = 64
            memory_depth: int = 2

        return TestConfig()

    def test_block_forward(self, config):
        """Test MLAMemoryBlock forward pass."""
        from mla_memory_block import MLAMemoryBlock

        block = MLAMemoryBlock(config, layer_id=0)
        x = torch.randn(2, 16, config.n_embd)

        # Create dummy freqs_cis
        freqs_cis = torch.randn(16, config.qk_rope_head_dim // 2, 2)

        output, state = block(x, start_pos=0, freqs_cis=freqs_cis, mask=None)

        assert output.shape == x.shape
        assert state is not None

    def test_gate_output_range(self, config):
        """Test that gate values are in [0, 1]."""
        from mla_memory_block import MLAMemoryBlock

        block = MLAMemoryBlock(config, layer_id=0)

        # Hook to capture gate values
        gate_values = []

        def hook(module, input, output):
            gate_values.append(torch.sigmoid(output))

        block.gate_proj.register_forward_hook(hook)

        x = torch.randn(2, 8, config.n_embd)
        freqs_cis = torch.randn(8, config.qk_rope_head_dim // 2, 2)
        block(x, start_pos=0, freqs_cis=freqs_cis, mask=None)

        assert len(gate_values) > 0
        for g in gate_values:
            assert (g >= 0).all() and (g <= 1).all()


class TestSWAMLAWithMemory:
    """Integration tests for SWAMLAModel with neural memory."""

    @pytest.fixture
    def small_config(self):
        """Create a small model config for testing."""
        from swa_mla_model import SWAMLAConfig

        return SWAMLAConfig(
            vocab_size=1000,
            block_size=128,
            n_layer=4,
            n_head=4,
            n_embd=128,
            local_layers_per_cycle=2,
            mla_layers_per_cycle=1,
            kv_lora_rank=32,
            use_neural_memory=True,
            memory_dim=32,
            memory_depth=2,
            use_gradient_checkpointing=False,
        )

    def test_model_creation(self, small_config):
        """Test model instantiation with memory enabled."""
        from swa_mla_model import SWAMLAModel, MLAMemoryBlock

        model = SWAMLAModel(small_config)

        # Count memory blocks
        memory_blocks = sum(
            1 for block in model.transformer.h
            if isinstance(block, MLAMemoryBlock)
        )

        # With 4 layers and 2 SWA + 1 MLA per cycle, we should have some MLA blocks
        assert memory_blocks > 0

    def test_forward_with_memory(self, small_config):
        """Test forward pass with memory states."""
        from swa_mla_model import SWAMLAModel

        model = SWAMLAModel(small_config)
        input_ids = torch.randint(0, 1000, (2, 32))
        targets = torch.randint(0, 1000, (2, 32))

        # Forward with memory
        logits, loss, states = model(
            input_ids, targets=targets,
            return_memory_states=True
        )

        assert logits.shape == (2, 32, 1000)
        assert loss is not None
        assert len(states) > 0

    def test_truncated_bptt(self, small_config):
        """Test truncated BPTT training pattern."""
        from swa_mla_model import SWAMLAModel

        model = SWAMLAModel(small_config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        memory_states = None

        # Simulate two training steps
        for step in range(2):
            input_ids = torch.randint(0, 1000, (2, 32))
            targets = torch.randint(0, 1000, (2, 32))

            optimizer.zero_grad()
            logits, loss, new_states = model(
                input_ids, targets=targets,
                memory_states=memory_states,
                return_memory_states=True
            )
            loss.backward()
            optimizer.step()

            # Detach states for truncated BPTT
            memory_states = [s.detach() for s in new_states]

        # States should persist and be valid
        assert memory_states is not None
        assert len(memory_states) > 0

    def test_generation_with_memory(self, small_config):
        """Test generation with persistent memory."""
        from swa_mla_model import SWAMLAModel

        model = SWAMLAModel(small_config)
        model.eval()

        prompt = torch.randint(0, 1000, (1, 10))
        generated, _ = model.generate(prompt, max_new_tokens=5, use_memory=True)

        assert generated.shape[1] == 15  # prompt + 5 new tokens


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
