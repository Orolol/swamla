"""Debug checkpoint: inspect weights, config, EMA state."""
import sys
import torch

if len(sys.argv) < 2:
    print("Usage: python debug_checkpoint.py <checkpoint.pt>")
    sys.exit(1)

path = sys.argv[1]
print(f"Loading {path}...")
ckpt = torch.load(path, map_location="cpu", weights_only=False)

# Config
print("\n=== Config ===")
config = ckpt.get("config", {})
for k, v in sorted(config.items()):
    print(f"  {k}: {v}")

# State dict
sd_key = "model" if "model" in ckpt else "model_state_dict" if "model_state_dict" in ckpt else None
if sd_key:
    sd = ckpt[sd_key]
    print(f"\n=== State dict (key='{sd_key}'): {len(sd)} tensors ===")
    prefixes = set(k.split(".")[0] for k in sd.keys())
    print(f"  Top-level prefixes: {prefixes}")

    # Check for prefix issues
    has_orig_mod = any(k.startswith("_orig_mod.") for k in sd.keys())
    has_module = any(k.startswith("module.") for k in sd.keys())
    print(f"  Has _orig_mod. prefix: {has_orig_mod}")
    print(f"  Has module. prefix: {has_module}")

    # Weight statistics
    print("\n  --- First 15 tensors ---")
    for k, v in list(sd.items())[:15]:
        if v.is_floating_point():
            vf = v.float()
            print(f"  {k}: shape={list(v.shape)}, dtype={v.dtype}, "
                  f"mean={vf.mean():.6f}, std={vf.std():.6f}, "
                  f"min={vf.min():.4f}, max={vf.max():.4f}, "
                  f"nan={v.isnan().any().item()}, inf={v.isinf().any().item()}")
        else:
            print(f"  {k}: shape={list(v.shape)}, dtype={v.dtype}")

    # Check for NaN/Inf across all tensors
    print("\n  --- NaN/Inf scan ---")
    nan_keys = []
    inf_keys = []
    for k, v in sd.items():
        if v.is_floating_point():
            if v.isnan().any():
                nan_keys.append(k)
            if v.isinf().any():
                inf_keys.append(k)
    print(f"  Tensors with NaN: {len(nan_keys)}")
    for k in nan_keys[:10]:
        print(f"    - {k}")
    print(f"  Tensors with Inf: {len(inf_keys)}")
    for k in inf_keys[:10]:
        print(f"    - {k}")

    # Embedding and lm_head stats
    print("\n  --- Key layers ---")
    for pattern in ["transformer.wte.weight", "transformer.ln_f.weight", "lm_head.weight"]:
        for prefix in ["_orig_mod.", ""]:
            key = prefix + pattern
            if key in sd:
                v = sd[key].float()
                print(f"  {key}: mean={v.mean():.6f}, std={v.std():.6f}, "
                      f"min={v.min():.4f}, max={v.max():.4f}")
                break

    # Residual scalars (if present)
    print("\n  --- Residual scalars ---")
    for k, v in sd.items():
        if "resid" in k or "x0_lambda" in k:
            print(f"  {k}: {v.float().mean():.6f} (shape={list(v.shape)})")
else:
    print("\nNo state dict found in checkpoint!")

# EMA
print("\n=== EMA ===")
has_ema = "ema" in ckpt and ckpt["ema"] is not None
print(f"  Present: {has_ema}")
if has_ema:
    ema_state = ckpt["ema"]
    print(f"  Decay: {ema_state.get('decay')}")
    ep = ema_state.get("ema_params", {})
    print(f"  Param count: {len(ep)}")

    # Check prefix mismatch with state dict
    if sd_key:
        ema_keys = set(ep.keys())
        sd_keys = set(sd.keys())
        direct_match = len(ema_keys & sd_keys)
        strip = lambda k: k.replace("_orig_mod.", "").replace("module.", "")
        stripped_sd = {strip(k) for k in sd_keys}
        stripped_match = len({strip(k) for k in ema_keys} & stripped_sd)
        print(f"  Direct key matches with state_dict: {direct_match}/{len(ep)}")
        print(f"  Matches after stripping prefixes: {stripped_match}/{len(ep)}")
        if direct_match == 0 and stripped_match > 0:
            print("  >>> PREFIX MISMATCH DETECTED - EMA keys lack _orig_mod. prefix!")

    # EMA vs raw weight divergence
    if sd_key:
        print("\n  --- EMA vs Raw divergence (first 10 params) ---")
        count = 0
        for ema_name, ema_val in ep.items():
            # Find matching key in state_dict
            sd_name = None
            if ema_name in sd:
                sd_name = ema_name
            else:
                stripped = ema_name.replace("_orig_mod.", "").replace("module.", "")
                for sk in sd:
                    if sk.replace("_orig_mod.", "").replace("module.", "") == stripped:
                        sd_name = sk
                        break
            if sd_name and ema_val.is_floating_point():
                diff = (ema_val.float() - sd[sd_name].float()).abs()
                print(f"  {ema_name}: mean_diff={diff.mean():.6f}, max_diff={diff.max():.6f}, "
                      f"rel_diff={diff.mean() / (sd[sd_name].float().abs().mean() + 1e-8):.4f}")
                count += 1
                if count >= 10:
                    break

# Metadata
print("\n=== Metadata ===")
print(f"  Step: {ckpt.get('step')}")
print(f"  Total tokens: {ckpt.get('total_tokens')}")
print(f"  Val loss: {ckpt.get('val_loss')}")
print(f"  Checkpoint keys: {list(ckpt.keys())}")
