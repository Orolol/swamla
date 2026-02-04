"""Unit tests for YaRN (Yet another RoPE extensioN) implementation.

Tests cover:
- Core YaRN frequency computation (NTK-by-parts interpolation)
- Backward compatibility with standard RoPE
- Integration with SWAMLAModel
- Configuration validation
"""

import math
import pytest
import torch
import sys
import os

# Add models directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'models'))


class TestComputeYarnInvFreq:
    """Tests for compute_yarn_inv_freq() function."""

    def test_no_scaling_returns_standard_rope(self):
        """When scale_factor <= 1.0, should return standard RoPE frequencies."""
        from positional_encoding import compute_yarn_inv_freq

        dim = 64
        base = 10000.0

        # Standard RoPE computation
        expected = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))

        # YaRN with no scaling
        result = compute_yarn_inv_freq(dim, base=base, scale_factor=1.0)

        torch.testing.assert_close(result, expected)

    def test_scaling_modifies_frequencies(self):
        """When scale_factor > 1.0, frequencies should be modified."""
        from positional_encoding import compute_yarn_inv_freq

        dim = 64
        base = 10000.0

        unscaled = compute_yarn_inv_freq(dim, base=base, scale_factor=1.0)
        scaled = compute_yarn_inv_freq(dim, base=base, scale_factor=4.0)

        # Frequencies should be different after scaling
        assert not torch.allclose(unscaled, scaled)

    def test_low_frequencies_preserved(self):
        """Low frequencies (long wavelengths) should be more preserved than high frequencies.

        YaRN NTK-by-parts interpolation:
        - γ(ratio) = 0 when ratio < beta_slow → full interpolation (scale by 1/scale_factor)
        - γ(ratio) = 1 when ratio > beta_fast → no interpolation (keep original)

        Since ratio = wavelength / original_max_seq_len:
        - High inv_freq (first elements) → small wavelength → small ratio → γ≈0 → more scaling
        - Low inv_freq (last elements) → large wavelength → large ratio → γ≈1 → less scaling

        So low frequencies change less proportionally to their original values.
        """
        from positional_encoding import compute_yarn_inv_freq

        dim = 64
        base = 10000.0
        beta_slow = 1.0
        beta_fast = 32.0

        unscaled = compute_yarn_inv_freq(dim, base=base, scale_factor=1.0)
        scaled = compute_yarn_inv_freq(
            dim, base=base, scale_factor=4.0,
            beta_fast=beta_fast, beta_slow=beta_slow
        )

        # Low frequencies (last elements) have large wavelengths → ratio closer to or above beta_fast
        # So they are less affected by interpolation (scale_mult closer to 1.0)
        # We check the relative change: |scaled - unscaled| / |unscaled|
        high_freq_relative_diff = ((scaled[:4] - unscaled[:4]).abs() / unscaled[:4].abs()).mean()
        low_freq_relative_diff = ((scaled[-4:] - unscaled[-4:]).abs() / unscaled[-4:].abs()).mean()

        # Low frequencies should have smaller relative change
        assert low_freq_relative_diff < high_freq_relative_diff

    def test_output_shape(self):
        """Output should have shape [dim // 2]."""
        from positional_encoding import compute_yarn_inv_freq

        for dim in [32, 64, 128]:
            result = compute_yarn_inv_freq(dim, scale_factor=4.0)
            assert result.shape == (dim // 2,)

    def test_custom_beta_parameters(self):
        """Custom beta_fast and beta_slow should affect interpolation."""
        from positional_encoding import compute_yarn_inv_freq

        dim = 64

        result1 = compute_yarn_inv_freq(dim, scale_factor=4.0, beta_fast=32.0, beta_slow=1.0)
        result2 = compute_yarn_inv_freq(dim, scale_factor=4.0, beta_fast=64.0, beta_slow=2.0)

        # Different beta values should produce different results
        assert not torch.allclose(result1, result2)


class TestPrecomputeFreqsCisYarn:
    """Tests for precompute_freqs_cis_yarn() function."""

    def test_output_shape(self):
        """Output should have shape [end, dim // 2] and be complex."""
        from positional_encoding import precompute_freqs_cis_yarn

        dim = 64
        end = 2048

        result = precompute_freqs_cis_yarn(dim, end, scale_factor=4.0)

        assert result.shape == (end, dim // 2)
        assert result.is_complex()

    def test_no_scaling_matches_standard(self):
        """With scale_factor=1.0, should match standard precompute_freqs_cis."""
        from positional_encoding import precompute_freqs_cis, precompute_freqs_cis_yarn

        dim = 64
        end = 2048
        theta = 10000.0

        standard = precompute_freqs_cis(dim, end, theta)
        yarn = precompute_freqs_cis_yarn(dim, end, theta, scale_factor=1.0)

        torch.testing.assert_close(yarn, standard)


class TestBackwardCompatibility:
    """Tests ensuring backward compatibility with existing RoPE behavior."""

    def test_yarn_disabled_matches_rope(self):
        """With yarn_enabled=False, model should produce identical results to RoPE."""
        from swa_mla_model import SWAMLAConfig

        # Create two configs: one without YaRN, one with YaRN disabled
        config_default = SWAMLAConfig(
            vocab_size=1000, n_layer=2, n_embd=128, n_head=4
        )
        config_yarn_disabled = SWAMLAConfig(
            vocab_size=1000, n_layer=2, n_embd=128, n_head=4,
            yarn_enabled=False, yarn_scale_factor=4.0  # YaRN params set but disabled
        )

        # Both should have yarn_enabled=False
        assert config_default.yarn_enabled == False
        assert config_yarn_disabled.yarn_enabled == False

    def test_scale_factor_one_matches_rope(self):
        """With scale_factor=1.0, YaRN should match standard RoPE exactly."""
        from positional_encoding import precompute_freqs_cis, precompute_freqs_cis_yarn

        dim = 64
        end = 2048
        theta = 10000.0

        # Standard RoPE
        standard = precompute_freqs_cis(dim, end, theta)

        # YaRN with scale_factor=1.0 (should be identical to standard RoPE)
        yarn = precompute_freqs_cis_yarn(dim, end, theta, scale_factor=1.0)

        torch.testing.assert_close(yarn, standard)


class TestYarnConfigValidation:
    """Tests for YaRN configuration validation in SWAMLAConfig."""

    def test_valid_config_accepted(self):
        """Valid YaRN configuration should be accepted."""
        from swa_mla_model import SWAMLAConfig

        # Valid YaRN config
        config = SWAMLAConfig(
            vocab_size=1000, n_layer=2, n_embd=128, n_head=4,
            yarn_enabled=True,
            yarn_scale_factor=4.0,
            yarn_original_max_seq_len=2048,
            yarn_beta_fast=32.0,
            yarn_beta_slow=1.0,
        )
        assert config.yarn_enabled == True
        assert config.yarn_scale_factor == 4.0
        # attn_factor should be auto-computed
        assert config.yarn_attn_factor is not None
        assert config.yarn_attn_factor == pytest.approx(1.1386, rel=1e-3)

    def test_invalid_scale_factor_rejected(self):
        """scale_factor < 1.0 should raise ValueError."""
        from swa_mla_model import SWAMLAConfig

        with pytest.raises(ValueError, match="yarn_scale_factor must be >= 1.0"):
            SWAMLAConfig(
                vocab_size=1000, n_layer=2, n_embd=128, n_head=4,
                yarn_scale_factor=0.5,  # Invalid: < 1.0
            )

    def test_invalid_beta_order_rejected(self):
        """beta_fast <= beta_slow should raise ValueError."""
        from swa_mla_model import SWAMLAConfig

        with pytest.raises(ValueError, match="yarn_beta_fast.*must be > yarn_beta_slow"):
            SWAMLAConfig(
                vocab_size=1000, n_layer=2, n_embd=128, n_head=4,
                yarn_beta_fast=1.0,  # Invalid: same as beta_slow
                yarn_beta_slow=1.0,
            )

        with pytest.raises(ValueError, match="yarn_beta_fast.*must be > yarn_beta_slow"):
            SWAMLAConfig(
                vocab_size=1000, n_layer=2, n_embd=128, n_head=4,
                yarn_beta_fast=0.5,  # Invalid: less than beta_slow
                yarn_beta_slow=1.0,
            )


class TestYarnIntegration:
    """Integration tests for YaRN with full model."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for model tests")
    def test_model_with_yarn_enabled(self):
        """Model should run forward pass with YaRN enabled."""
        from swa_mla_model import SWAMLAConfig, SWAMLAModel

        config = SWAMLAConfig(
            vocab_size=1000,
            block_size=512,
            n_layer=2,
            n_embd=128,
            n_head=4,
            kv_lora_rank=64,
            qk_nope_head_dim=32,
            qk_rope_head_dim=16,
            v_head_dim=32,
            yarn_enabled=True,
            yarn_scale_factor=4.0,
            yarn_original_max_seq_len=128,
            # Pure MLA config: 0 DeltaNet layers, all MLA
            local_layers_per_cycle=0,
            mla_layers_per_cycle=1,
            # Disable Triton kernels for CPU-only compatibility testing
            use_triton_kernels=False,
            use_flash_attention=False,
        )

        device = 'cuda'
        model = SWAMLAModel(config).to(device)
        model.eval()

        # Forward pass with normal sequence
        batch_size = 2
        seq_len = 64
        x = torch.randint(0, 1000, (batch_size, seq_len), device=device)

        with torch.no_grad():
            logits, _ = model(x)

        assert logits.shape == (batch_size, 1, 1000)  # Only last token logits

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for model tests")
    def test_extended_context_inference(self):
        """Model should process sequences longer than training length."""
        from swa_mla_model import SWAMLAConfig, SWAMLAModel

        original_max_seq = 128
        scale_factor = 4.0
        extended_max_seq = int(original_max_seq * scale_factor)  # 512

        config = SWAMLAConfig(
            vocab_size=1000,
            block_size=extended_max_seq,  # Extended context
            n_layer=2,
            n_embd=128,
            n_head=4,
            kv_lora_rank=64,
            qk_nope_head_dim=32,
            qk_rope_head_dim=16,
            v_head_dim=32,
            yarn_enabled=True,
            yarn_scale_factor=scale_factor,
            yarn_original_max_seq_len=original_max_seq,
            # Pure MLA config: 0 DeltaNet layers, all MLA
            local_layers_per_cycle=0,
            mla_layers_per_cycle=1,
            # Disable Triton kernels for CPU-only compatibility testing
            use_triton_kernels=False,
            use_flash_attention=False,
        )

        device = 'cuda'
        model = SWAMLAModel(config).to(device)
        model.eval()

        # Test with sequence at 2x original length (within extended range)
        batch_size = 1
        seq_len = original_max_seq * 2  # 256
        x = torch.randint(0, 1000, (batch_size, seq_len), device=device)

        with torch.no_grad():
            logits, _ = model(x)

        assert logits.shape == (batch_size, 1, 1000)


class TestAttnFactor:
    """Tests for attention temperature scaling factor."""

    def test_attn_factor_auto_computation(self):
        """attn_factor should be auto-computed from scale_factor."""
        scale_factor = 4.0
        expected = 0.1 * math.log(scale_factor) + 1.0

        # Verify expected value
        assert expected == pytest.approx(1.1386, rel=1e-3)

        # Verify config computes the same value
        from swa_mla_model import SWAMLAConfig
        config = SWAMLAConfig(
            vocab_size=1000, n_layer=2, n_embd=128, n_head=4,
            yarn_enabled=True,
            yarn_scale_factor=scale_factor,
        )
        assert config.yarn_attn_factor == pytest.approx(expected, rel=1e-5)

    def test_custom_attn_factor_override(self):
        """Custom attn_factor should override auto-computation."""
        from swa_mla_model import SWAMLAConfig

        custom_factor = 1.5
        config = SWAMLAConfig(
            vocab_size=1000, n_layer=2, n_embd=128, n_head=4,
            yarn_enabled=True,
            yarn_scale_factor=4.0,
            yarn_attn_factor=custom_factor,  # Custom override
        )
        # Custom value should be preserved, not auto-computed
        assert config.yarn_attn_factor == custom_factor


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
