#!/usr/bin/env python3
"""Test checkpoint compatibility with inference.py"""
import torch
import sys
from pathlib import Path

# Add models directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'models'))

def test_checkpoint(checkpoint_path):
    print(f"Testing checkpoint: {checkpoint_path}")

    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    print("\n" + "="*60)
    print("CHECKPOINT STRUCTURE")
    print("="*60)
    print(f"Top-level keys: {list(ckpt.keys())}")

    # Check which key has the state dict
    state_dict_key = None
    if 'model' in ckpt:
        state_dict_key = 'model'
    elif 'model_state_dict' in ckpt:
        state_dict_key = 'model_state_dict'

    if state_dict_key:
        print(f"State dict key: '{state_dict_key}'")
        state_dict = ckpt[state_dict_key]

        # Check for _orig_mod prefix
        has_orig_mod = any(k.startswith('_orig_mod.') for k in state_dict.keys())
        print(f"Has _orig_mod. prefix: {has_orig_mod}")
    else:
        print("WARNING: No model state dict found!")
        return

    # Check config
    config = ckpt.get('config', {})
    print(f"\n" + "="*60)
    print(f"CONFIG ({len(config)} keys)")
    print("="*60)

    # Essential architecture params
    essential = ['vocab_size', 'block_size', 'n_layer', 'n_embd', 'n_head']
    print("\nEssential params:")
    for k in essential:
        val = config.get(k, 'MISSING!')
        status = "✓" if k in config else "✗"
        print(f"  {status} {k}: {val}")

    # Feature flags
    features = ['use_moe', 'use_engram', 'use_gated_deltanet', 'fope_enabled', 'yarn_enabled',
                'local_layers_per_cycle', 'mla_layers_per_cycle', 'q_lora_rank', 'kv_lora_rank']
    print("\nFeature params:")
    for k in features:
        if k in config:
            print(f"  ✓ {k}: {config[k]}")

    # Check args (training params)
    args = ckpt.get('args', {})
    print(f"\n" + "="*60)
    print(f"ARGS ({len(args)} keys)")
    print("="*60)

    # Check for size preset
    if 'size' in args:
        print(f"Size preset: {args['size']}")
    elif 'size' in config:
        print(f"Size preset (in config): {config['size']}")
    else:
        print("No size preset found - will need full config")

    # Check inference.py compatibility issues
    print(f"\n" + "="*60)
    print("INFERENCE.PY COMPATIBILITY CHECK")
    print("="*60)

    issues = []

    # Issue 1: inference.py checks for 'model' key, not 'model_state_dict'
    if state_dict_key == 'model_state_dict':
        # Check if inference.py handles this
        issues.append("inference.py line 504-528 checks for 'model' key but checkpoint uses 'model_state_dict'")

    # Issue 2: Check if config has all needed params for SWAMLAConfig
    from swa_mla_model import SWAMLAConfig
    import dataclasses
    config_fields = {f.name for f in dataclasses.fields(SWAMLAConfig)}

    # Parameters that have defaults so they're optional
    optional_params = config_fields  # All have defaults in dataclass

    # Check for training-only params that need to be removed
    training_only = ['batch_size', 'max_iters', 'learning_rate', 'optimizer_type',
                     'use_te_fp8', 'fp8_backend', 'compile', 'compile_mode']
    found_training_params = [k for k in config.keys() if k in training_only]
    if found_training_params:
        print(f"\nTraining params in config (will be filtered): {found_training_params}")

    # Check for unknown params
    unknown = set(config.keys()) - config_fields - set(training_only)
    if unknown:
        print(f"\nUnknown params in config (may cause issues): {list(unknown)[:10]}")

    if issues:
        print("\n⚠ POTENTIAL ISSUES:")
        for i, issue in enumerate(issues, 1):
            print(f"  {i}. {issue}")
    else:
        print("\n✓ No obvious compatibility issues found")

    # Test actual model creation
    print(f"\n" + "="*60)
    print("MODEL CREATION TEST")
    print("="*60)

    try:
        from swa_mla_model import SWAMLAModel, SWAMLAConfig, create_swa_mla_model

        # Clean config
        clean_config = config.copy()

        # Remove training-only params
        for param in training_only:
            clean_config.pop(param, None)

        # Check for size preset
        model_size = clean_config.pop('size', None)

        if model_size:
            print(f"Creating model with size preset: {model_size}")
            model = create_swa_mla_model(size=model_size, **clean_config)
        else:
            print("Creating model with direct config")
            model_config = SWAMLAConfig(**clean_config)
            model = SWAMLAModel(model_config)

        print(f"✓ Model created: {model.param_count/1e6:.2f}M params")

        # Try loading state dict
        if has_orig_mod:
            state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}

        result = model.load_state_dict(state_dict, strict=False)
        if result.missing_keys:
            print(f"⚠ Missing keys: {result.missing_keys[:5]}...")
        if result.unexpected_keys:
            print(f"⚠ Unexpected keys: {result.unexpected_keys[:5]}...")
        if not result.missing_keys and not result.unexpected_keys:
            print("✓ State dict loaded successfully (all keys matched)")

    except Exception as e:
        print(f"✗ Model creation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        # Default checkpoint
        checkpoint_path = "outputs/full/checkpoint_1_00B_step6103.pt"
    else:
        checkpoint_path = sys.argv[1]

    test_checkpoint(checkpoint_path)
