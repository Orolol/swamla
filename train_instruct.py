"""
Instruction fine-tuning script for SWA-MLA model.
Uses SlimOrca dataset with ChatML format and loss masking on prompts.
Supports FP8, Lion optimizer, wandb logging, and packed conversation loading.
"""

import os
import sys
import time
import math
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

# Add models and data to path
sys.path.insert(0, str(Path(__file__).parent / 'models'))
sys.path.insert(0, str(Path(__file__).parent / 'data'))
sys.path.insert(0, str(Path(__file__).parent / 'optimization'))

from swa_mla_model import create_swa_mla_model, SWAMLAConfig
from data_loader_instruct import PackedInstructDataset  # Changed from data_loader_packed
from fp8_native import convert_to_native_fp8

# Try to import wandb
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("wandb not available - logging to console only")

# Try to import huggingface_hub
try:
    from huggingface_hub import HfApi
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    print("huggingface_hub not available - HF push disabled")

# Try to import Lion optimizer
try:
    from lion_pytorch import Lion
    LION_AVAILABLE = True
except ImportError:
    try:
        import torch_optimizer as extra_optim
        if hasattr(extra_optim, 'Lion'):
            Lion = extra_optim.Lion
            LION_AVAILABLE = True
        else:
            LION_AVAILABLE = False
    except ImportError:
        LION_AVAILABLE = False

from transformers import AutoTokenizer


def setup_distributed():
    """Setup distributed training if available."""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])

        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)

        return True, rank, local_rank, world_size
    return False, 0, 0, 1


def configure_tf32(enable_tf32=True, verbose=True):
    """Configure TensorFloat-32 (TF32) precision for Ampere+ GPUs.

    TF32 provides ~7x speedup on A100/H100 with minimal accuracy loss.
    IMPORTANT: Uses new API (fp32_precision) if available, otherwise legacy API.
    NEVER mixes both APIs to avoid conflicts.

    Args:
        enable_tf32: Whether to enable TF32 (default: True for speed)
        verbose: Whether to print configuration details
    """
    if not torch.cuda.is_available():
        return

    # Check if GPU supports TF32 (Ampere or later, compute capability >= 8.0)
    device_capability = torch.cuda.get_device_capability()
    supports_tf32 = device_capability[0] >= 8

    # Detect which API is available (new fp32_precision or legacy allow_tf32)
    # Try to access fp32_precision to check if new API exists
    has_new_api = False
    try:
        # Just try to read the attribute to see if it exists
        _ = torch.backends.cuda.matmul.fp32_precision
        has_new_api = True
    except AttributeError:
        has_new_api = False

    if enable_tf32 and supports_tf32:
        if has_new_api:
            # Use ONLY new PyTorch 2.9+ API
            # DO NOT touch allow_tf32 (legacy API) to avoid mixing
            torch.backends.cuda.matmul.fp32_precision = "tf32"
            torch.backends.cudnn.fp32_precision = "tf32"
            api_used = "new API (fp32_precision)"
            
        else:
            # Use ONLY legacy API for PyTorch < 2.9
            # DO NOT touch fp32_precision to avoid mixing
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            api_used = "legacy API (allow_tf32)"

        if verbose:
            print(f"✓ TF32 enabled for FP32 operations ({api_used})")
            print("  - Matmul operations: TF32 (includes attention, linear layers)")
            print("  - cuDNN operations: TF32 (includes convolutions)")
            print("  - Expected speedup: ~3-7x on A100/H100 for FP32 operations")
            print("  - Note: TF32 reduces mantissa from 23 to 10 bits (minimal accuracy loss)")

    elif enable_tf32 and not supports_tf32:
        if verbose:
            print(f"⚠ TF32 requested but GPU doesn't support it (compute capability {device_capability[0]}.{device_capability[1]} < 8.0)")
            print("  Falling back to standard FP32 precision")

    else:
        # Disable TF32 for full IEEE FP32 precision
        if has_new_api:
            # Use ONLY new API
            torch.backends.cuda.matmul.fp32_precision = "ieee"
            torch.backends.cudnn.fp32_precision = "ieee"
            api_used = "new API"
        else:
            # Use ONLY legacy API
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            api_used = "legacy API"

        if verbose:
            print(f"✓ TF32 disabled - using full IEEE FP32 precision ({api_used})")


def get_lr(it, warmup_iters, max_iters, learning_rate, min_lr):
    """Learning rate schedule with warmup and cosine decay."""
    # Linear warmup
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # Cosine decay
    if it > max_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (max_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)


def configure_optimizer(model, learning_rate, weight_decay, betas, device_type, optimizer_type='adamw', use_fp8=False):
    """Configure optimizer with proper parameter grouping.

    Note: FP8 training uses standard optimizers with GradScaler for gradient scaling.
    """
    # Separate parameters into decay and no_decay groups
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # No weight decay for biases, norms, and embeddings
        if any(nd in name for nd in ['.bias', 'norm', 'ln_', 'wte', 'wpe']):
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    param_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': no_decay_params, 'weight_decay': 0.0}
    ]

    # Choose optimizer based on type
    if optimizer_type == 'lion' and LION_AVAILABLE:
        print("Using Lion optimizer")
        optimizer = Lion(param_groups, lr=learning_rate, betas=betas)
    else:
        print("Using AdamW optimizer")
        if device_type == 'cuda':
            optimizer = torch.optim.AdamW(param_groups, lr=learning_rate, betas=betas, fused=True)
        else:
            optimizer = torch.optim.AdamW(param_groups, lr=learning_rate, betas=betas)

    return optimizer


def load_latest_from_huggingface(repo_id, hf_token=None):
    """Load the latest checkpoint from HuggingFace Hub instruct/ subdirectory.

    Returns:
        dict: Checkpoint data with keys 'model_state_dict', 'optimizer_state_dict', 'step', 'total_tokens', 'config', 'val_loss'
        None: If loading fails
    """
    if not HF_AVAILABLE:
        print("huggingface_hub not available - cannot load from HF")
        return None

    try:
        from huggingface_hub import list_repo_files
        import re

        print(f"Loading latest checkpoint from {repo_id} (instruct/ subdirectory only)...")

        # List all files in the repo
        files = list_repo_files(repo_id, token=hf_token)

        # Find all checkpoint directories in instruct/ subdirectory ONLY
        # Pattern: instruct/checkpoint_tokens_XXX_loss_Y.YYYY/pytorch_model.bin
        checkpoint_pattern = re.compile(r'instruct/checkpoint_tokens_(\d+[kKmMbB])_loss_([\d.]+)/pytorch_model\.bin')
        checkpoints = []

        for file in files:
            match = checkpoint_pattern.match(file)
            if match:
                tokens_str = match.group(1)
                loss_str = match.group(2)

                # Parse tokens (convert k/M/B to actual number)
                tokens_multiplier = {'k': 1000, 'K': 1000, 'm': 1_000_000, 'M': 1_000_000, 'b': 1_000_000_000, 'B': 1_000_000_000}
                tokens_value = int(tokens_str[:-1])
                tokens_suffix = tokens_str[-1]
                total_tokens = tokens_value * tokens_multiplier.get(tokens_suffix, 1)

                checkpoints.append({
                    'file': file,
                    'total_tokens': total_tokens,
                    'loss': float(loss_str),
                    'dir': file.rsplit('/', 1)[0]
                })

        if not checkpoints:
            print(f"No instruction fine-tuning checkpoints found in {repo_id}/instruct/")
            print(f"  Looking for pattern: instruct/checkpoint_tokens_XXX_loss_Y.YYYY/")
            return None

        # Sort by total tokens (most recent training)
        checkpoints.sort(key=lambda x: x['total_tokens'], reverse=True)
        latest = checkpoints[0]

        print(f"Found {len(checkpoints)} checkpoints")
        print(f"Loading latest: {latest['dir']} (tokens: {latest['total_tokens']:,}, loss: {latest['loss']:.4f})")

        # Download the checkpoint file
        from huggingface_hub import hf_hub_download
        checkpoint_path = hf_hub_download(
            repo_id=repo_id,
            filename=latest['file'],
            token=hf_token
        )

        # Load checkpoint
        # Note: Using weights_only=False because we trust our own checkpoints
        # and they may contain custom optimizer states
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        except TypeError:
            # Fallback for older PyTorch versions that don't have weights_only parameter
            checkpoint = torch.load(checkpoint_path, map_location='cpu')

        print(f"✓ Successfully loaded checkpoint from HuggingFace")

        return checkpoint

    except Exception as e:
        print(f"Error loading from HuggingFace: {e}")
        import traceback
        traceback.print_exc()
        return None


def push_to_huggingface(model, tokenizer, config_args, output_dir, total_tokens, val_loss, repo_id, hf_token, optimizer=None, step=None):
    """Push instruction-tuned model to Hugging Face Hub in instruct/ subdirectory."""
    if not HF_AVAILABLE:
        print("huggingface_hub not available - skipping HF push")
        return

    if not hf_token:
        print("HF_TOKEN not set - skipping HF push")
        return

    try:
        # Create temporary save directory with informative name
        # Format: instruct/checkpoint_tokens_XXXk_loss_Y.YYYY (in instruct/ subfolder)
        tokens_str = f"{total_tokens // 1000}k" if total_tokens < 1_000_000 else f"{total_tokens // 1_000_000}M"
        model_name = f"instruct/checkpoint_tokens_{tokens_str}_loss_{val_loss:.4f}"
        save_path = os.path.join(output_dir, "hf_upload", model_name)
        os.makedirs(save_path, exist_ok=True)

        # Save model state dict and config
        print(f"Saving model to {save_path}...")
        checkpoint_data = {
            'model_state_dict': model.state_dict(),
            'config': config_args,
            'total_tokens': total_tokens,
            'val_loss': val_loss,
        }

        # Add optimizer state and step if provided (for resuming)
        if optimizer is not None:
            checkpoint_data['optimizer_state_dict'] = optimizer.state_dict()
        if step is not None:
            checkpoint_data['step'] = step

        torch.save(checkpoint_data, os.path.join(save_path, "pytorch_model.bin"))

        # Save tokenizer
        tokenizer.save_pretrained(save_path)

        # Save model config as JSON
        import json
        with open(os.path.join(save_path, "config.json"), "w") as f:
            json.dump({
                'model_type': 'swa_mla',
                'total_tokens': total_tokens,
                'val_loss': val_loss,
                'training_config': config_args,
            }, f, indent=2)

        # Create a comprehensive README
        readme_content = f"""---
license: apache-2.0
tags:
- swamla
- sliding-window-attention
- multi-head-latent-attention
- pytorch
- causal-lm
---

# SWA-MLA Model Checkpoint (Instruction Fine-Tuned)

**Training Progress:**
- Total instruction tokens processed: {total_tokens:,}
- Validation loss: {val_loss:.4f}
- Perplexity: {math.exp(val_loss):.2f}

This is an instruction fine-tuned checkpoint trained on SlimOrca dataset with ChatML format.
The model has been fine-tuned to follow instructions and engage in helpful conversations.

## Model Architecture
Hybrid architecture combining:
- **Sliding Window Attention (SWA)** blocks for efficient local context
- **Multi-head Latent Attention (MLA)** blocks for global context with KV compression

## Fine-Tuning Details
- Dataset: Open-Orca/SlimOrca
- Format: ChatML with special tokens (`<|im_start|>`, `<|im_end|>`)
- Loss masking: Loss calculated only on assistant responses (not on system/user prompts)

## Model Configuration
- Size: {config_args.get('size', 'unknown')}
- Block size (context length): {config_args.get('block_size', 'unknown')}
- Local layers per cycle: {config_args.get('local_layers_per_cycle', 'unknown')}
- MLA layers per cycle: {config_args.get('mla_layers_per_cycle', 'unknown')}
- SWA window size: {config_args.get('swa_window', 'unknown')}
- MLA KV LoRA rank: {config_args.get('mla_kv_lora_rank', 'unknown')}

## Loading the Model
```python
import torch
from transformers import AutoTokenizer

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("{repo_id}", subfolder="{model_name}")

# Load model checkpoint
checkpoint = torch.hub.load_state_dict_from_url(
    f"https://huggingface.co/{repo_id}/resolve/main/{model_name}/pytorch_model.bin",
    map_location="cpu"
)

# You'll need the SWA-MLA model code to instantiate the model
# See: https://github.com/yourusername/swamla for the model implementation
```

## Training Details
- Optimizer: {config_args.get('optimizer_type', 'unknown')}
- Learning rate: {config_args.get('learning_rate', 'unknown')}
- Batch size: {config_args.get('batch_size', 'unknown')}
- Gradient accumulation: {config_args.get('gradient_accumulation_steps', 'unknown')}

---
Generated with [SWA-MLA](https://github.com/yourusername/swamla)
"""

        with open(os.path.join(save_path, "README.md"), "w") as f:
            f.write(readme_content)

        # Upload to HF
        print(f"Uploading to Hugging Face: {repo_id}/{model_name}...")
        api = HfApi(token=hf_token)
        api.upload_folder(
            folder_path=save_path,
            repo_id=repo_id,
            repo_type="model",
            path_in_repo=model_name,
            commit_message=f"Add checkpoint: {total_tokens:,} tokens, val_loss={val_loss:.4f}"
        )

        print(f"Successfully uploaded to https://huggingface.co/{repo_id}/tree/main/{model_name}")

        # Clean up local upload directory (optional, to save disk space)
        import shutil
        shutil.rmtree(save_path)
        print(f"Cleaned up local upload directory")

    except Exception as e:
        print(f"Error pushing to Hugging Face: {e}")
        import traceback
        traceback.print_exc()


def train(args):
    """Main training function."""
    # Setup distributed training
    is_ddp, rank, local_rank, world_size = setup_distributed()
    master_process = rank == 0
    device = f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu'

    if master_process:
        print(f"Training SWA-MLA model")
        print(f"Device: {device}")
        print(f"DDP: {is_ddp}, World size: {world_size}")

    # Configure TF32 precision (only on master process to avoid spam)
    # enable_tf32 = not args.disable_tf32 if hasattr(args, 'disable_tf32') else True
    enable_tf32 = False
    if master_process:
        print("\nConfiguring TF32 precision...")
        configure_tf32(enable_tf32=enable_tf32, verbose=True)
        print("")

    # Setup wandb (will be updated with model stats later)
    wandb_run = None
    if master_process and WANDB_AVAILABLE and args.wandb_project and hasattr(wandb, 'init'):
        if hasattr(wandb, 'login'):
            wandb.login()
        wandb_run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name or f"swa_mla_instruct_{args.size}",
            config=vars(args)
        )

    # Setup TensorBoard
    tb_writer = None
    if master_process:
        try:
            from torch.utils.tensorboard import SummaryWriter
            # Use a descriptive run name for TensorBoard
            run_name = args.wandb_run_name or f"swa_mla_instruct_{args.size}_{int(time.time())}"
            tb_writer = SummaryWriter(log_dir=f"runs/{run_name}")
            print(f"TensorBoard logging enabled in runs/{run_name}")
        except ImportError:
            print("TensorBoard not available - logging to console only")

    # Load tokenizer and add ChatML special tokens
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token

    # Configure tokenizer to support longer sequences BEFORE adding tokens
    # This suppresses the warning about sequence length
    tokenizer.model_max_length = args.block_size

    # Add ChatML special tokens if not already present
    special_tokens = {"additional_special_tokens": ["<|im_start|>", "<|im_end|>"]}
    num_added = tokenizer.add_special_tokens(special_tokens)
    if master_process:
        if wandb_run is not None:
            wandb.finish()
        if tb_writer is not None:
            tb_writer.close()
        print("Training completed.")
        if num_added > 0:
            print(f"Added {num_added} ChatML special tokens to tokenizer")

    vocab_size = len(tokenizer)

    if master_process:
        print(f"Tokenizer vocabulary size: {vocab_size} (includes {num_added} new tokens)")

    # Try to load checkpoint from HuggingFace if requested
    resume_checkpoint = None
    resume_step = 0
    resume_tokens = 0
    if args.resume_from_hf and args.hf_repo_id:
        if master_process:
            print("\n" + "="*80)
            print("LOADING PRE-TRAINED CHECKPOINT FROM HUGGINGFACE")
            print("="*80)

        hf_token = os.getenv("HF_TOKEN")
        resume_checkpoint = load_latest_from_huggingface(args.hf_repo_id, hf_token)

        if resume_checkpoint:
            # Get pre-training stats for logging
            pretrain_step = resume_checkpoint.get('step', 0)
            pretrain_tokens = resume_checkpoint.get('total_tokens', 0)
            if master_process:
                print(f"✓ Loaded pre-trained checkpoint (step {pretrain_step:,}, {pretrain_tokens:,} tokens)")
                print(f"✓ Starting instruction fine-tuning from step 0")
                print("="*80 + "\n")
            # DON'T set resume_step - we're starting fresh for instruction fine-tuning
            resume_step = 0
            resume_tokens = 0
        else:
            if master_process:
                print("⚠ Failed to load checkpoint from HuggingFace, starting from scratch")
                print("="*80 + "\n")

    # Create model
    if master_process:
        print(f"\nCreating {args.size} SWA-MLA model...")

    # IMPORTANT: If resuming from checkpoint, create model with ORIGINAL vocab_size
    # We'll resize embeddings AFTER loading the checkpoint
    model_vocab_size = vocab_size
    if resume_checkpoint and num_added > 0:
        # Create model with original vocab size to match checkpoint
        model_vocab_size = vocab_size - num_added
        if master_process:
            print(f"Creating model with original vocab_size={model_vocab_size} (will resize after loading)")

    model = create_swa_mla_model(
        size=args.size,
        vocab_size=model_vocab_size,
        block_size=args.block_size,
        dropout=args.dropout,
        local_layers_per_cycle=args.local_layers_per_cycle,
        mla_layers_per_cycle=args.mla_layers_per_cycle,
        swa_window=args.swa_window,
        swa_sink_size=args.swa_sink_size,
        q_lora_rank=args.mla_q_lora_rank,
        kv_lora_rank=args.mla_kv_lora_rank,
        qk_nope_head_dim=args.mla_qk_nope_head_dim,
        qk_rope_head_dim=args.mla_qk_rope_head_dim,
        v_head_dim=args.mla_v_head_dim,
        use_fp8=False,  # FP8 conversion handled separately via fp8_native
        use_gradient_checkpointing=args.gradient_checkpointing,
    )

    model = model.to(device)

    # Load model weights if resuming (BEFORE resizing embeddings!)
    if resume_checkpoint:
        if master_process:
            print("Loading model weights from checkpoint...")

        # Handle state_dict from compiled models (removes _orig_mod. prefix)
        state_dict = resume_checkpoint['model_state_dict']
        if any(key.startswith('_orig_mod.') for key in state_dict.keys()):
            if master_process:
                print("Detected compiled model checkpoint, removing _orig_mod. prefix...")
            state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}

        model.load_state_dict(state_dict)  # Should match exactly now
        if master_process:
            print("✓ Model weights loaded")

    # Resize token embeddings if ChatML tokens were added (AFTER loading checkpoint!)
    if num_added > 0:
        if master_process:
            print(f"\nResizing model embeddings from {model.config.vocab_size} to {vocab_size} (adding {num_added} ChatML tokens)...")

        # Get current embedding weights
        old_wte_weight = model.transformer.wte.weight.data
        old_lm_head_weight = model.lm_head.weight.data

        # Create new embeddings with random initialization for new tokens
        new_wte_weight = torch.zeros(vocab_size, model.config.n_embd, device=device, dtype=old_wte_weight.dtype)
        new_lm_head_weight = torch.zeros(vocab_size, model.config.n_embd, device=device, dtype=old_lm_head_weight.dtype)

        # Copy old weights
        new_wte_weight[:old_wte_weight.size(0)] = old_wte_weight
        new_lm_head_weight[:old_lm_head_weight.size(0)] = old_lm_head_weight

        # Initialize new token embeddings (small random values)
        torch.nn.init.normal_(new_wte_weight[old_wte_weight.size(0):], mean=0.0, std=0.02)
        torch.nn.init.normal_(new_lm_head_weight[old_lm_head_weight.size(0):], mean=0.0, std=0.02)

        # Replace embeddings
        model.transformer.wte.weight = torch.nn.Parameter(new_wte_weight)
        model.lm_head.weight = torch.nn.Parameter(new_lm_head_weight)

        # Update config
        model.config.vocab_size = vocab_size
        if master_process:
            print(f"✓ Model embeddings resized to {vocab_size}")
            print(f"  - Added {num_added} new token embeddings with random initialization")

    # Convert model to FP8 using native implementation if requested
    if args.use_fp8:
        if master_process:
            print("\nConverting model to FP8 with Native PyTorch...")
        try:
            # Convert model, excluding lm_head to avoid dimension alignment issues
            model = convert_to_native_fp8(model)

            if master_process:
                print("✓ Model converted to FP8 for training")
                print("  - Linear layers converted to FP8Linear")
                print("  - Using torch._scaled_mm for acceleration")
                print("  - Dynamic scaling enabled")

        except ImportError:
            if master_process:
                print("Warning: fp8_native module not found")
                print("Falling back to BF16 training")
            args.use_fp8 = False

        except Exception as e:
            if master_process:
                print(f"Warning: Failed to convert model to FP8: {e}")
                print("Falling back to BF16 training")
            args.use_fp8 = False


               
    # Print model info and log to wandb
    if master_process:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\nModel Statistics:")
        print(f"  Total parameters: {total_params:,} ({total_params/1e6:.2f}M)")
        print(f"  Trainable parameters: {trainable_params:,} ({trainable_params/1e6:.2f}M)")

        # Memory estimation
        param_memory = total_params * 4 / (1024**3)  # FP32 in GB
        print(f"  Estimated memory (FP32): {param_memory:.2f} GB")
        print(f"  Estimated memory (BF16): {param_memory/2:.2f} GB")

        # Log to wandb
        if wandb_run is not None:
            wandb.config.update({
                'total_params': total_params,
                'total_params_M': total_params / 1e6,
                'trainable_params': trainable_params,
                'trainable_params_M': trainable_params / 1e6,
                'param_memory_fp32_gb': param_memory,
                'param_memory_bf16_gb': param_memory / 2,
            }, allow_val_change=True)

    # Compile model if requested
    if args.compile:
        if master_process:
            print("Compiling model with torch.compile()...")

        # Use different compilation mode for FP8 to avoid FlexibleLayout issues
        if args.use_fp8:
            # 'reduce-overhead' mode works better with FP8 dynamic scaling
            model = torch.compile(model, mode='reduce-overhead')
            if master_process:
                print("  Using 'reduce-overhead' mode (optimized for FP8)")
        else:
            # 'max-autotune' for non-FP8 training
            model = torch.compile(model, mode='max-autotune')
            if master_process:
                print("  Using 'max-autotune' mode")

    # Wrap with DDP
    if is_ddp:
        model = DDP(model, device_ids=[local_rank])
        raw_model = model.module
    else:
        raw_model = model

    # Setup data loader for instruction fine-tuning
    if master_process:
        print("\nSetting up instruction data loader...")

    # IMPORTANT: Instruction dataset is DIFFERENT from pre-training dataset
    # Always start from offset 0 (we're not resuming from the same dataset)
    # OPTIMIZATION: Use num_workers=1 for streaming datasets to avoid conflicts
    optimal_workers = 1  # Streaming datasets don't benefit from multiple workers
    data_loader = PackedInstructDataset(
        split='train',
        max_length=args.block_size,
        batch_size=args.batch_size,
        buffer_docs=50,  # Number of conversations to buffer before building batch (reduced for faster loading)
        prefetch_batches=4,  # Number of batches to prefetch
        tokenizer=tokenizer,
        shuffle=True,
        num_workers=optimal_workers,
        start_offset=0,  # Always start from beginning for instruction fine-tuning
    )

    if master_process:
        print("✓ Using PackedInstructDataset with SlimOrca")
        print(f"  - Format: ChatML with loss masking on prompts")
        print(f"  - Loss calculated only on assistant responses")

    # Setup optimizer
    optimizer = configure_optimizer(
        raw_model,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(args.beta1, args.beta2),
        device_type='cuda' if torch.cuda.is_available() else 'cpu',
        optimizer_type=args.optimizer_type,
        use_fp8=args.use_fp8
    )

    # DON'T load optimizer state for instruction fine-tuning
    # Reasons:
    # 1. Vocabulary changed (embeddings resized)
    # 2. Different dataset (SlimOrca vs FineWeb)
    # 3. Fresh start is better for fine-tuning stability
    if master_process:
        print("Using fresh optimizer state (not loading from pre-training checkpoint)")
        print("  - This is normal for instruction fine-tuning")

    # Training loop
    if master_process:
        print("\nStarting training...")
        if resume_step > 0:
            print(f"Resuming from step: {resume_step:,}")
            print(f"Starting tokens: {resume_tokens:,}")
        print(f"Max iterations: {args.max_iters:,}")
        print(f"Gradient accumulation steps: {args.gradient_accumulation_steps}")
        print(f"Effective batch size: {args.batch_size * args.gradient_accumulation_steps * world_size:,}")

    data_iter = iter(data_loader)
    # Native FP8 implementation needs GradScaler for proper gradient scaling
    # BF16 training doesn't need it (enabled=False)
    scaler = torch.amp.GradScaler('cuda', enabled=args.use_fp8)

    running_loss = 0.0
    t0 = time.time()

    # Track best validation loss for HF push
    best_val_loss = float('inf')
    total_tokens_seen = resume_tokens  # Start from resumed token count

    # Adjust starting step
    start_step = resume_step
    for step in range(start_step, args.max_iters):
        # Update learning rate
        lr = get_lr(step, args.warmup_iters, args.max_iters, args.learning_rate, args.min_lr)

        # Update learning rate (handle both tensor and float lr)
        for param_group in optimizer.param_groups:
            if isinstance(param_group['lr'], torch.Tensor):
                # Tensor lr (some optimizers): use .fill_()
                param_group['lr'].fill_(lr)
            else:
                # Standard optimizer: direct assignment
                param_group['lr'] = lr

        # Training step with gradient accumulation
        model.train()
        optimizer.zero_grad()
        accum_loss = 0.0

        # Profiling timestamps
        if master_process and step % args.log_interval == 0:
            step_start = time.perf_counter()
            timings = {}

        for micro_step in range(args.gradient_accumulation_steps):
            if master_process and step % args.log_interval == 0:
                prof_t0 = time.perf_counter()

            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(data_loader)
                batch = next(data_iter)

            if master_process and step % args.log_interval == 0:
                timings['data_loading'] = timings.get('data_loading', 0) + (time.perf_counter() - prof_t0)
                prof_t0 = time.perf_counter()

            input_ids = batch['input_ids'].to(device)
            loss_mask = batch['loss_mask'].to(device)

            # Create labels (shift input_ids by 1 for next token prediction)
            labels = input_ids.clone()

            if master_process and step % args.log_interval == 0:
                timings['data_transfer'] = timings.get('data_transfer', 0) + (time.perf_counter() - prof_t0)
                prof_t0 = time.perf_counter()

            # Forward pass with mixed precision
            # IMPORTANT: Use return_all_logits=True to get logits for ALL positions without computing loss
            with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=True):
                logits, _ = model(input_ids, return_all_logits=True)  # Avoids redundant loss computation

                if master_process and step % args.log_interval == 0:
                    timings['forward'] = timings.get('forward', 0) + (time.perf_counter() - prof_t0)
                    prof_t0 = time.perf_counter()

                # Compute masked loss manually
                # Shift logits and labels for next token prediction
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = labels[:, 1:].contiguous()
                shift_mask = loss_mask[:, 1:].contiguous()

                # CRITICAL: Check for valid shapes before any computation
                # If shift_logits has batch_size=0 or seq_len=0, create zero loss and skip
                if shift_logits.size(0) == 0 or shift_logits.size(1) == 0:
                    loss = torch.tensor(0.0, device=device, dtype=torch.float32)
                    if master_process and step % args.log_interval == 0:
                        print(f"  Warning: Skipping batch with invalid shape: {shift_logits.shape}")
                else:
                    # Check if there are any valid tokens to compute loss on
                    num_valid_tokens = shift_mask.sum()

                    if num_valid_tokens > 0:
                        # Flatten for cross_entropy
                        loss = F.cross_entropy(
                            shift_logits.view(-1, shift_logits.size(-1)),
                            shift_labels.view(-1),
                            reduction='none'
                        )

                        # Apply mask and compute mean only over unmasked tokens
                        loss = (loss * shift_mask.view(-1)).sum() / num_valid_tokens
                        loss = loss / args.gradient_accumulation_steps
                    else:
                        # No valid tokens in this batch (all masked), skip
                        # This can happen with very short conversations or edge cases
                        loss = torch.tensor(0.0, device=device, dtype=torch.float32)
                        if master_process and step % args.log_interval == 0:
                            print(f"  Warning: Batch with no valid loss tokens (all masked), skipping...")

                if master_process and step % args.log_interval == 0:
                    timings['loss_compute'] = timings.get('loss_compute', 0) + (time.perf_counter() - prof_t0)
                    prof_t0 = time.perf_counter()

            # Backward pass (only if loss has gradients)
            if loss.requires_grad:
                scaler.scale(loss).backward()
            accum_loss += loss.item()

            if master_process and step % args.log_interval == 0:
                timings['backward'] = timings.get('backward', 0) + (time.perf_counter() - prof_t0)

        if master_process and step % args.log_interval == 0:
            prof_t0 = time.perf_counter()

        # Gradient clipping
        if args.grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

        # Optimizer step
        scaler.step(optimizer)
        scaler.update()

        if master_process and step % args.log_interval == 0:
            timings['optimizer'] = time.perf_counter() - prof_t0
            timings['total'] = time.perf_counter() - step_start

        running_loss += accum_loss

        # Track total tokens processed
        total_tokens_seen += args.batch_size * args.block_size * args.gradient_accumulation_steps * world_size

        # Logging
        if step % args.log_interval == 0 and master_process:
            t1 = time.time()
            dt = t1 - t0
            t0 = t1

            lossf = running_loss / args.log_interval if step > 0 else accum_loss
            running_loss = 0.0

            tokens_per_sec = (args.batch_size * args.block_size * args.gradient_accumulation_steps * world_size * args.log_interval) / dt

            # Format total tokens for display
            if total_tokens_seen < 1_000_000:
                tokens_str = f"{total_tokens_seen // 1000}K"
            elif total_tokens_seen < 1_000_000_000:
                tokens_str = f"{total_tokens_seen / 1_000_000:.2f}M"
            else:
                tokens_str = f"{total_tokens_seen / 1_000_000_000:.2f}B"

            print(f"Step {step:6d} | Loss: {lossf:.4f} | LR: {lr:.2e} | Tokens/sec: {tokens_per_sec:,.0f} | Total: {tokens_str}")

            # # Print detailed timings breakdown
            # if 'timings' in locals():
            #     total_time = timings.get('total', 1.0)
            #     print(f"  Timing breakdown (ms):")
            #     print(f"    Data loading:  {timings.get('data_loading', 0)*1000:6.2f} ({timings.get('data_loading', 0)/total_time*100:5.1f}%)")
            #     print(f"    Data transfer: {timings.get('data_transfer', 0)*1000:6.2f} ({timings.get('data_transfer', 0)/total_time*100:5.1f}%)")
            #     print(f"    Forward pass:  {timings.get('forward', 0)*1000:6.2f} ({timings.get('forward', 0)/total_time*100:5.1f}%)")
            #     print(f"    Loss compute:  {timings.get('loss_compute', 0)*1000:6.2f} ({timings.get('loss_compute', 0)/total_time*100:5.1f}%)")
            #     print(f"    Backward pass: {timings.get('backward', 0)*1000:6.2f} ({timings.get('backward', 0)/total_time*100:5.1f}%)")
            #     print(f"    Optimizer:     {timings.get('optimizer', 0)*1000:6.2f} ({timings.get('optimizer', 0)/total_time*100:5.1f}%)")
            #     print(f"    TOTAL:         {total_time*1000:6.2f} ms")

            if wandb_run is not None:
                wandb.log({
                    'train/loss': lossf,
                    'train/lr': lr,
                    'train/tokens_per_sec': tokens_per_sec,
                    'train/total_tokens': total_tokens_seen,
                    'step': step
                })
            
            if tb_writer is not None:
                tb_writer.add_scalar('train/loss', lossf, step)
                tb_writer.add_scalar('train/lr', lr, step)
                tb_writer.add_scalar('train/tokens_per_sec', tokens_per_sec, step)
                tb_writer.add_scalar('train/total_tokens', total_tokens_seen, step)

        # Validation (using next batches from same data loader)
        # Skip validation at the exact resume step to avoid immediate validation after loading
        if step % args.eval_interval == 0 and step > start_step and master_process:
            model.eval()
            val_loss = 0.0
            val_steps = 50

            with torch.no_grad():
                for _ in range(val_steps):
                    try:
                        batch = next(data_iter)
                    except StopIteration:
                        data_iter = iter(data_loader)
                        batch = next(data_iter)

                    input_ids = batch['input_ids'].to(device)
                    loss_mask = batch['loss_mask'].to(device)
                    labels = input_ids.clone()

                    with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=True):
                        # IMPORTANT: Use return_all_logits=True to avoid redundant loss computation
                        logits, _ = model(input_ids, return_all_logits=True)

                        # Compute masked loss (same as training)
                        shift_logits = logits[:, :-1, :].contiguous()
                        shift_labels = labels[:, 1:].contiguous()
                        shift_mask = loss_mask[:, 1:].contiguous()

                        # CRITICAL: Check for valid shapes before any computation
                        if shift_logits.size(0) == 0 or shift_logits.size(1) == 0:
                            loss = torch.tensor(0.0, device=device, dtype=torch.float32)
                        else:
                            # Check if there are any valid tokens
                            num_valid_tokens = shift_mask.sum()

                            if num_valid_tokens > 0:
                                loss = F.cross_entropy(
                                    shift_logits.view(-1, shift_logits.size(-1)),
                                    shift_labels.view(-1),
                                    reduction='none'
                                )
                                loss = (loss * shift_mask.view(-1)).sum() / num_valid_tokens
                            else:
                                # Skip this batch if no valid tokens
                                loss = torch.tensor(0.0, device=device, dtype=torch.float32)

                    val_loss += loss.item()

            val_loss /= val_steps
            perplexity = math.exp(val_loss) if val_loss < 10 else float('inf')

            print(f"\nValidation | Loss: {val_loss:.4f} | Perplexity: {perplexity:.2f}\n")

            if wandb_run is not None:
                wandb.log({
                    'val/loss': val_loss,
                    'val/perplexity': perplexity,
                    'step': step
                })
            
            if tb_writer is not None:
                tb_writer.add_scalar('val/loss', val_loss, step)
                tb_writer.add_scalar('val/perplexity', perplexity, step)

            # Track best validation loss
            if val_loss < best_val_loss:
                print(f"New best validation loss: {val_loss:.4f} (previous: {best_val_loss:.4f})")
                best_val_loss = val_loss

            # Push to HF at every validation if repo_id is set
            # IMPORTANT: Only master process should upload to avoid NCCL timeout in multi-GPU training
            if args.hf_repo_id and master_process:
                # Get HF token from environment
                hf_token = os.getenv("HF_TOKEN")

                if hf_token:
                    print(f"Pushing model to Hugging Face...")
                    push_to_huggingface(
                        model=raw_model,
                        tokenizer=tokenizer,
                        config_args=vars(args),
                        output_dir=args.output_dir,
                        total_tokens=total_tokens_seen,
                        val_loss=val_loss,
                        repo_id=args.hf_repo_id,
                        hf_token=hf_token,
                        optimizer=optimizer,
                        step=step
                    )
                else:
                    print("HF_TOKEN not set - skipping HF push. Set HF_TOKEN environment variable to enable automatic uploads.")

        # Checkpointing
        if step % args.save_interval == 0 and step > 0 and master_process:
            checkpoint = {
                'model': raw_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'step': step,
                'config': vars(args),
            }

            checkpoint_path = os.path.join(args.output_dir, f'checkpoint_{step}.pt')
            torch.save(checkpoint, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")

    # Cleanup
    if is_ddp:
        dist.destroy_process_group()

    if master_process:
        print("\nTraining complete!")


def main():
    parser = argparse.ArgumentParser(description='Instruction fine-tune SWA-MLA model with SlimOrca')

    # Model parameters
    parser.add_argument('--size', type=str, default='small', choices=['small', 'base', 'large', 'xl'])
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--block_size', type=int, default=2048)
    parser.add_argument('--dropout', type=float, default=0.0)

    # SWA-MLA specific parameters
    parser.add_argument('--local_layers_per_cycle', '--swa_layers_per_cycle', type=int, default=2,
                        dest='local_layers_per_cycle')
    parser.add_argument('--mla_layers_per_cycle', type=int, default=1)
    parser.add_argument('--swa_window', type=int, default=256)
    parser.add_argument('--swa_sink_size', type=int, default=4)
    parser.add_argument('--mla_q_lora_rank', type=int, default=0)
    parser.add_argument('--mla_kv_lora_rank', type=int, default=256)
    parser.add_argument('--mla_qk_nope_head_dim', type=int, default=128)
    parser.add_argument('--mla_qk_rope_head_dim', type=int, default=64)
    parser.add_argument('--mla_v_head_dim', type=int, default=128)

    # Training parameters (adjusted for instruction fine-tuning)
    parser.add_argument('--max_iters', type=int, default=10000)  # Fewer iterations for fine-tuning
    parser.add_argument('--learning_rate', type=float, default=5e-5)  # Lower LR for fine-tuning
    parser.add_argument('--min_lr', type=float, default=5e-6)  # Lower min LR
    parser.add_argument('--weight_decay', type=float, default=0.1)
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.95)
    parser.add_argument('--warmup_iters', type=int, default=100)  # Less warmup for fine-tuning
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--gradient_checkpointing', action='store_true')

    # Optimizer
    parser.add_argument('--optimizer_type', type=str, default='adamw', choices=['adamw', 'lion'])
    parser.add_argument('--use_fp8', action='store_true')

    # TF32 precision control (Ampere+ GPUs)
    parser.add_argument('--enable_tf32', action='store_true', default=True,
                        help='Enable TF32 for ~3-7x speedup on A100/H100 (default: True)')
    parser.add_argument('--disable_tf32', action='store_true',
                        help='Disable TF32 for full IEEE FP32 precision')

    # Data parameters
    parser.add_argument('--tokenizer_name', type=str, default='openai-community/gpt2')
    parser.add_argument('--num_workers', type=int, default=8)

    # Logging and checkpointing (more frequent for fine-tuning)
    parser.add_argument('--output_dir', type=str, default='outputs/swa_mla_instruct')
    parser.add_argument('--log_interval', type=int, default=10)
    parser.add_argument('--eval_interval', type=int, default=100)  # Validate more often
    parser.add_argument('--save_interval', type=int, default=500)  # Save more often
    parser.add_argument('--wandb_project', type=str, default="swamla-instruct")
    parser.add_argument('--wandb_run_name', type=str, default=None)

    # Hugging Face integration
    parser.add_argument('--hf_repo_id', type=str, default=None, help='HuggingFace repo ID (e.g., "username/model-name"). Set HF_TOKEN env var for authentication.')
    parser.add_argument('--resume_from_hf', action='store_true', help='Resume training from the latest checkpoint in HF repo (requires --hf_repo_id)')

    # Performance
    parser.add_argument('--compile', action='store_true')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Run training
    train(args)


if __name__ == '__main__':
    main()
