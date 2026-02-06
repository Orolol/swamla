"""Minimal forward pass test — bypasses inference.py entirely."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "models"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "optimization"))
import torch
from models.swa_mla_model import SWAMLAModel, SWAMLAConfig
from transformers import AutoTokenizer

if len(sys.argv) < 2:
    print("Usage: python debug_forward.py <checkpoint.pt>")
    sys.exit(1)

device = "cuda" if torch.cuda.is_available() else "cpu"
path = sys.argv[1]
print(f"Loading {path}...")
ckpt = torch.load(path, map_location="cpu", weights_only=False)

# --- Step 1: Create model from config ---
config_dict = ckpt["config"].copy()

# Strip training-only params that aren't SWAMLAConfig fields
# (try creating config, catch errors)
print("\n=== Step 1: Create model ===")
print(f"Config keys: {sorted(config_dict.keys())}")

# Remove known training-only params
for k in list(config_dict.keys()):
    if k in ('batch_size', 'max_iters', 'learning_rate', 'min_lr',
             'weight_decay', 'beta1', 'beta2', 'warmup_iters', 'grad_clip',
             'gradient_accumulation_steps', 'optimizer_type', 'enable_tf32',
             'disable_tf32', 'tokenizer_name', 'num_workers', 'output_dir',
             'log_interval', 'eval_interval', 'save_interval', 'wandb_project',
             'wandb_run_name', 'hf_repo_id', 'resume_from_hf', 'compile', 'compile_mode',
             'resume_from', 'profile', 'profile_steps', 'profile_warmup',
             'memory_reset_interval', 'use_tensorboard', 'use_te_fp8', 'use_varlen_attn',
             'use_wedlm', 'no_moe'):
        config_dict.pop(k, None)

# Force FP8 off
config_dict['use_fp8'] = False
config_dict.pop('fp8_backend', None)

# Try creating config
try:
    config = SWAMLAConfig(**config_dict)
    print(f"SWAMLAConfig created successfully")
except TypeError as e:
    print(f"SWAMLAConfig creation failed: {e}")
    # Remove invalid keys and retry
    import re
    bad_key = re.search(r"unexpected keyword argument '(\w+)'", str(e))
    while bad_key:
        k = bad_key.group(1)
        print(f"  Removing invalid key: {k}")
        config_dict.pop(k, None)
        try:
            config = SWAMLAConfig(**config_dict)
            print(f"SWAMLAConfig created after removing invalid keys")
            bad_key = None
        except TypeError as e2:
            bad_key = re.search(r"unexpected keyword argument '(\w+)'", str(e2))

model = SWAMLAModel(config)
print(f"Model created: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params")

# --- Step 2: Load state dict ---
print("\n=== Step 2: Load state dict ===")
sd = ckpt.get("model_state_dict") or ckpt.get("model")

# Strip _orig_mod. prefix
if any(k.startswith("_orig_mod.") for k in sd.keys()):
    sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
    print("Stripped _orig_mod. prefix")

# Strip module. prefix
if any(k.startswith("module.") for k in sd.keys()):
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    print("Stripped module. prefix")

# Check key alignment BEFORE loading
model_keys = set(model.state_dict().keys())
ckpt_keys = set(sd.keys())
missing = model_keys - ckpt_keys
unexpected = ckpt_keys - model_keys
print(f"Model expects {len(model_keys)} keys")
print(f"Checkpoint has {len(ckpt_keys)} keys")
print(f"Missing from checkpoint: {len(missing)}")
for k in sorted(missing)[:20]:
    print(f"  MISSING: {k}")
print(f"Unexpected in checkpoint: {len(unexpected)}")
for k in sorted(unexpected)[:20]:
    print(f"  UNEXPECTED: {k}")

# Load
result = model.load_state_dict(sd, strict=False)
print(f"\nload_state_dict result:")
print(f"  missing_keys: {len(result.missing_keys)}")
print(f"  unexpected_keys: {len(result.unexpected_keys)}")

# --- Step 3: Forward pass test ---
print("\n=== Step 3: Forward pass test ===")
model = model.to(device=device, dtype=torch.bfloat16)
model.eval()

tokenizer = AutoTokenizer.from_pretrained("gpt2")
text = "The capital of France is"
input_ids = tokenizer.encode(text, return_tensors="pt").to(device)
print(f"Input: '{text}' -> {input_ids.shape}")

# Test 1: With autocast (like training)
with torch.inference_mode():
    with torch.amp.autocast(device, dtype=torch.bfloat16):
        logits, _ = model(input_ids, return_all_logits=True)
    print(f"Logits shape: {logits.shape}")
    last_logits = logits[0, -1, :].float()
    probs = torch.softmax(last_logits, dim=-1)
    topk = torch.topk(probs, 10)
    print("\nTop-10 next token predictions:")
    for i in range(10):
        token = tokenizer.decode([topk.indices[i].item()])
        print(f"  {i+1}. '{token}' (p={topk.values[i].item():.4f})")

# Test 2: Perplexity on a longer text
test_text = "The French Revolution began in 1789 when the people of France rose up against the monarchy."
test_ids = tokenizer.encode(test_text, return_tensors="pt").to(device)
with torch.inference_mode():
    with torch.amp.autocast(device, dtype=torch.bfloat16):
        logits, _ = model(test_ids, return_all_logits=True)
    shift_logits = logits[:, :-1, :].float()
    shift_labels = test_ids[:, 1:]
    loss = torch.nn.functional.cross_entropy(
        shift_logits.reshape(-1, shift_logits.size(-1)),
        shift_labels.reshape(-1)
    )
    print(f"\nPerplexity test: '{test_text[:50]}...'")
    print(f"  Loss: {loss.item():.4f}")
    print(f"  Perplexity: {torch.exp(loss).item():.2f}")

# Test 3: Check if model is in "always predict EOS" mode
print(f"\nEOS token analysis:")
eos_id = tokenizer.eos_token_id
print(f"  EOS token id: {eos_id}")
print(f"  EOS logit: {last_logits[eos_id].item():.4f}")
print(f"  EOS prob: {probs[eos_id].item():.6f}")
print(f"  Max logit: {last_logits.max().item():.4f} (token: '{tokenizer.decode([last_logits.argmax().item()])}')")
print(f"  Logit mean: {last_logits.mean().item():.4f}, std: {last_logits.std().item():.4f}")
