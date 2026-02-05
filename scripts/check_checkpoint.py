#!/usr/bin/env python3
"""Check checkpoint contents for debugging resume issues."""

import sys
import torch

def check_checkpoint(path):
    print(f"Loading checkpoint: {path}")
    ckpt = torch.load(path, map_location='cpu', weights_only=False)

    print("\n" + "="*60)
    print("CHECKPOINT KEYS")
    print("="*60)
    print(list(ckpt.keys()))

    print("\n" + "="*60)
    print("METADATA")
    print("="*60)
    print(f"  step: {ckpt.get('step', 'N/A')}")
    print(f"  total_tokens: {ckpt.get('total_tokens', 'N/A')}")
    print(f"  last_eval_tokens: {ckpt.get('last_eval_tokens', 'N/A')}")
    print(f"  last_save_tokens: {ckpt.get('last_save_tokens', 'N/A')}")

    print("\n" + "="*60)
    print("MODEL STATE DICT")
    print("="*60)
    state_dict = ckpt.get('model_state_dict', {})
    keys = list(state_dict.keys())
    print(f"Total keys: {len(keys)}")
    print("\nFirst 20 keys:")
    for k in keys[:20]:
        v = state_dict[k]
        print(f"  {k}: {v.shape} {v.dtype}")

    # Check for FP8-related keys
    fp8_keys = [k for k in keys if 'float8' in k.lower() or 'fp8' in k.lower()]
    scale_keys = [k for k in keys if 'scale' in k.lower() and 'scalar' not in k.lower()]
    print(f"\nFP8-related keys: {len(fp8_keys)}")
    if fp8_keys:
        for k in fp8_keys[:10]:
            print(f"  {k}")
    print(f"\nScale keys: {len(scale_keys)}")
    if scale_keys:
        for k in scale_keys[:10]:
            print(f"  {k}")

    # Check for _orig_mod prefix (torch.compile)
    orig_mod_keys = [k for k in keys if k.startswith('_orig_mod.')]
    print(f"\n_orig_mod. prefixed keys: {len(orig_mod_keys)}")

    print("\n" + "="*60)
    print("OPTIMIZER STATE")
    print("="*60)
    opt_state = ckpt.get('optimizer_state_dict')
    if opt_state is None:
        print("NO OPTIMIZER STATE!")
    elif isinstance(opt_state, list):
        print(f"List of {len(opt_state)} optimizers")
        for i, s in enumerate(opt_state):
            print(f"\n  Optimizer [{i}]:")
            print(f"    Keys: {list(s.keys())}")
            if 'param_groups' in s:
                print(f"    Param groups: {len(s['param_groups'])}")
                for j, pg in enumerate(s['param_groups']):
                    n_params = len(pg.get('params', []))
                    lr = pg.get('lr', 'N/A')
                    print(f"      Group {j}: {n_params} params, lr={lr}")
    else:
        print(f"Single optimizer")
        print(f"  Keys: {list(opt_state.keys())}")
        if 'param_groups' in opt_state:
            print(f"  Param groups: {len(opt_state['param_groups'])}")

    print("\n" + "="*60)
    print("EMA STATE")
    print("="*60)
    ema_state = ckpt.get('ema')
    if ema_state:
        print(f"EMA present with keys: {list(ema_state.keys())[:5]}...")
    else:
        print("No EMA state")

    print("\n" + "="*60)
    print("CONFIG/ARGS")
    print("="*60)
    config = ckpt.get('config', {})
    args = ckpt.get('args', {})
    print(f"Config keys: {list(config.keys())[:10]}...")
    print(f"Args keys: {list(args.keys())[:10]}...")

    # Check specific important args
    important_args = ['use_fp8', 'fp8_backend', 'optimizer_type', 'learning_rate', 'batch_size', 'block_size']
    print("\nImportant args:")
    for arg in important_args:
        val = args.get(arg, config.get(arg, 'N/A'))
        print(f"  {arg}: {val}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python check_checkpoint.py <checkpoint_path>")
        print("Example: python check_checkpoint.py outputs/full/checkpoint_1_00B_step6103.pt")
        sys.exit(1)

    check_checkpoint(sys.argv[1])
