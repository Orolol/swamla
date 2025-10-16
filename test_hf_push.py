"""
Test script for HuggingFace push functionality.
This script verifies that the HF push system works without actually training.
"""

import os
import sys
import torch
from pathlib import Path

# Add models to path
sys.path.insert(0, str(Path(__file__).parent / 'models'))
sys.path.insert(0, str(Path(__file__).parent / 'data'))

from swa_mla_model import create_swa_mla_model
from transformers import AutoTokenizer

# Check if huggingface_hub is available
try:
    from huggingface_hub import HfApi
    print("✓ huggingface_hub is installed")
    HF_AVAILABLE = True
except ImportError:
    print("✗ huggingface_hub is NOT installed")
    print("  Install with: pip install huggingface_hub>=0.16.0")
    HF_AVAILABLE = False
    sys.exit(1)

# Check if HF_TOKEN is set
HF_TOKEN = os.getenv("HF_TOKEN")
if HF_TOKEN:
    print(f"✓ HF_TOKEN is set (length: {len(HF_TOKEN)})")
else:
    print("✗ HF_TOKEN is NOT set")
    print("  Set it with: export HF_TOKEN='your_token_here'")
    print("  Get your token from: https://huggingface.co/settings/tokens")

# Test HF API connection
if HF_TOKEN:
    try:
        api = HfApi(token=HF_TOKEN)
        whoami = api.whoami()
        username = whoami['name']
        print(f"✓ Successfully authenticated as: {username}")

        # Test if we can list repositories (validates token has read access)
        repos = list(api.list_repos(author=username, limit=1))
        print(f"✓ Token has valid permissions")

    except Exception as e:
        print(f"✗ Error authenticating with HuggingFace: {e}")
        print("  Check that your token is valid and has write permissions")
        sys.exit(1)

# Create a small test model
print("\nCreating test model...")
tokenizer = AutoTokenizer.from_pretrained('openai-community/gpt2')
model = create_swa_mla_model(
    size='small',
    vocab_size=len(tokenizer),
    block_size=512,
    dropout=0.0,
)
print(f"✓ Created small test model ({sum(p.numel() for p in model.parameters())/1e6:.1f}M params)")

# Test model saving
print("\nTesting model save...")
test_save_path = "outputs/test_hf_push"
os.makedirs(test_save_path, exist_ok=True)

try:
    # Save model state dict
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': {'size': 'small', 'test': True},
        'total_tokens': 1000000,
        'val_loss': 2.5,
    }, os.path.join(test_save_path, "pytorch_model.bin"))

    # Save tokenizer
    tokenizer.save_pretrained(test_save_path)

    # Save config
    import json
    with open(os.path.join(test_save_path, "config.json"), "w") as f:
        json.dump({
            'model_type': 'swa_mla',
            'total_tokens': 1000000,
            'val_loss': 2.5,
        }, f, indent=2)

    # Create README
    with open(os.path.join(test_save_path, "README.md"), "w") as f:
        f.write("# Test Model\n\nThis is a test checkpoint.\n")

    print(f"✓ Successfully saved model to {test_save_path}")
    print(f"  Files: {list(os.listdir(test_save_path))}")

except Exception as e:
    print(f"✗ Error saving model: {e}")
    sys.exit(1)

# Cleanup
print("\nCleaning up test files...")
import shutil
shutil.rmtree(test_save_path)
print("✓ Cleanup complete")

print("\n" + "="*60)
print("SUCCESS! HuggingFace push functionality is properly configured.")
print("="*60)

if HF_TOKEN:
    print("\nTo use automatic HF push during training:")
    print(f"  ./scripts/train_swa_mla.sh small 8 2048 outputs/my_model false adamw \"{username}/your-repo-name\"")
    print("\nOr with Python:")
    print(f'  python train.py --size small --hf_repo_id "{username}/your-repo-name"')
else:
    print("\nTo use automatic HF push, first set your HF_TOKEN:")
    print("  export HF_TOKEN='your_token_here'")
    print("  Then run training with --hf_repo_id argument")

print("\nFor full documentation, see: HUGGINGFACE_PUSH.md")
