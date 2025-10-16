# Automatic Hugging Face Push

The training script now supports automatic pushing of model checkpoints to Hugging Face Hub whenever the validation loss improves.

## Setup

### 1. Install requirements
Make sure `huggingface_hub` is installed:
```bash
pip install -r requirements.txt
```

### 2. Set your Hugging Face token
Export your HF token as an environment variable:
```bash
export HF_TOKEN="your_huggingface_token_here"
```

You can get your token from: https://huggingface.co/settings/tokens

### 3. Create your repository (optional)
If you haven't created a repository yet, create it on Hugging Face:
```bash
# Using the CLI
huggingface-cli repo create swamla --type model

# Or create it manually at https://huggingface.co/new
```

## Usage

### Using the shell script
```bash
# Basic usage - push to HuggingFace when validation loss improves
./scripts/train_swa_mla.sh small 8 2048 outputs/my_model false adamw "Orosius/swamla"

# With different parameters
./scripts/train_swa_mla.sh base 16 4096 outputs/base_model false lion "YourUsername/your-repo"
```

### Using Python directly
```bash
python train.py \
    --size small \
    --batch_size 8 \
    --block_size 2048 \
    --hf_repo_id "Orosius/swamla" \
    --output_dir outputs/my_model
```

## How it works

1. **Best model tracking**: The script tracks the best validation loss seen during training
2. **Automatic push**: When validation loss improves, the model is automatically pushed to HuggingFace
3. **Informative naming**: Checkpoints are named with total tokens and validation loss:
   - Format: `checkpoint_tokens_{XXX}k_loss_{Y.YYYY}`
   - Example: `checkpoint_tokens_512M_loss_2.3456`
4. **Complete package**: Each push includes:
   - Model state dict (`pytorch_model.bin`)
   - Tokenizer files
   - Model configuration (`config.json`)
   - Comprehensive README with training details

## Checkpoint Structure

Each checkpoint pushed to HuggingFace contains:

```
checkpoint_tokens_512M_loss_2.3456/
├── pytorch_model.bin        # Model weights and training state
├── config.json               # Model configuration
├── tokenizer_config.json     # Tokenizer configuration
├── vocab.json                # Vocabulary
├── merges.txt                # BPE merges
└── README.md                 # Documentation with training details
```

## Loading a checkpoint

```python
import torch
from transformers import AutoTokenizer
from models.swa_mla_model import create_swa_mla_model

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    "Orosius/swamla",
    subfolder="checkpoint_tokens_512M_loss_2.3456"
)

# Load checkpoint
checkpoint = torch.hub.load_state_dict_from_url(
    "https://huggingface.co/Orosius/swamla/resolve/main/checkpoint_tokens_512M_loss_2.3456/pytorch_model.bin",
    map_location="cpu"
)

# Extract config and create model
config = checkpoint['config']
model = create_swa_mla_model(
    size=config['size'],
    vocab_size=len(tokenizer),
    block_size=config['block_size'],
    # ... other config parameters
)

# Load weights
model.load_state_dict(checkpoint['model_state_dict'])
```

## Environment Variables

- `HF_TOKEN`: Your Hugging Face authentication token (required)
  - Get it from: https://huggingface.co/settings/tokens
  - Must have write access to the repository

## Notes

- **Frequency**: Models are only pushed when validation loss improves (not every validation)
- **Storage**: Local upload directories are cleaned up after successful push to save disk space
- **Error handling**: Push failures are logged but don't stop training
- **Multi-GPU**: Only the master process (rank 0) performs the push in distributed training
- **Security**: Never commit your HF_TOKEN to git - always use environment variables

## Troubleshooting

### "HF_TOKEN not set"
- Make sure you've exported the token: `export HF_TOKEN="your_token"`
- Verify it's set: `echo $HF_TOKEN`

### "huggingface_hub not available"
- Install the package: `pip install huggingface_hub>=0.16.0`

### Permission denied errors
- Verify your token has write permissions
- Check that the repository exists and you have access to it
- Try creating the repo first: `huggingface-cli repo create your-repo-name --type model`

### Upload failures
- Check your internet connection
- Verify the repository ID is correct (format: "username/repo-name")
- Check HuggingFace status: https://status.huggingface.co/

## Example Training Session

```bash
# Set up environment
export HF_TOKEN="hf_xxxxxxxxxxxxxxxxxxxx"

# Start training with automatic HF push
./scripts/train_swa_mla.sh small 8 2048 outputs/swamla_v1 false adamw "Orosius/swamla"

# Output will show:
# Training SWA+MLA hybrid model...
# Model size: small
# Batch size: 8
# Block size: 2048
# Output dir: outputs/swamla_v1
# Optimizer: adamw
# HF Repo: Orosius/swamla (automatic push on val loss improvement)
#
# Training...
# Step 1000 | Loss: 3.2456 | ...
#
# Validation | Loss: 3.1234 | Perplexity: 22.71
# New best validation loss: 3.1234 (previous: inf)
# Pushing model to Hugging Face...
# Saving model to outputs/swamla_v1/hf_upload/checkpoint_tokens_16M_loss_3.1234...
# Uploading to Hugging Face: Orosius/swamla/checkpoint_tokens_16M_loss_3.1234...
# Successfully uploaded to https://huggingface.co/Orosius/swamla/tree/main/checkpoint_tokens_16M_loss_3.1234
```
