"""
Training script for DeltaNet-MLA model with LatentMoE.

Supported features:
- Muon + AdamW optimizer (PyTorch native)
- LatentMoE (NVIDIA Nemotron-3 style)
- GatedDeltaNet (O(n) linear attention)
- MLA with Flash Attention
- Gradient checkpointing
- torch.compile with max-autotune
- WeDLM training (Causal Diffusion Language Model)
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

from swa_mla_model import create_swa_mla_model, SWAMLAConfig
from data_loader_packed import PackedFinewebDataset

# Try to import wandb
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("wandb not available - logging to console only")

# Try to import tensorboard
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    print("tensorboard not available - tensorboard logging disabled")

# Try to import huggingface_hub
try:
    from huggingface_hub import HfApi
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    print("huggingface_hub not available - HF push disabled")

from transformers import AutoTokenizer

# WeDLM imports (lazy to avoid import errors if not using WeDLM)
WEDLM_AVAILABLE = False
try:
    from wedlm import WeDLMConfig, DualStreamMasker, WeDLMLoss, adapt_deltanet_for_wedlm
    from wedlm.loss import compute_accuracy
    WEDLM_AVAILABLE = True
except ImportError:
    pass

# Transformer Engine FP8 imports
TE_AVAILABLE = False
te = None
try:
    import transformer_engine.pytorch as te
    from transformer_engine.common.recipe import Format, DelayedScaling
    TE_AVAILABLE = True
except ImportError:
    pass

# For context manager fallback
from contextlib import nullcontext

# μP imports
MUP_AVAILABLE = False
try:
    sys.path.insert(0, str(Path(__file__).parent / 'optimization'))
    from mup import MuPConfig, mup_init, configure_mup_optimizer, mup_scale_output
    MUP_AVAILABLE = True
except ImportError:
    pass

# Progressive Training imports (independent of μP)
PROGRESSIVE_AVAILABLE = False
try:
    from progressive import ProgressiveScheduler
    PROGRESSIVE_AVAILABLE = True
except ImportError:
    pass

# EMA imports (independent of μP)
EMA_AVAILABLE = False
try:
    from swa import EMAModel
    EMA_AVAILABLE = True
except ImportError:
    pass


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


def format_params(n):
    """Format parameter count with appropriate suffix."""
    if n >= 1e9:
        return f"{n/1e9:.2f}B"
    elif n >= 1e6:
        return f"{n/1e6:.1f}M"
    elif n >= 1e3:
        return f"{n/1e3:.0f}K"
    return str(n)


def print_training_banner(args, model, world_size, device, resume_step=0, resume_tokens=0):
    """Print compact training summary banner."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Calculate effective batch size
    eff_batch = args.batch_size * args.gradient_accumulation_steps * world_size

    # GPU info
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_short = gpu_name.split()[-1] if 'NVIDIA' in gpu_name else gpu_name[:20]
    else:
        gpu_short = "CPU"

    # Run identifier
    run_id = args.wandb_run_name or f"swa_mla_{args.size}"
    timestamp = time.strftime('%Y-%m-%d %H:%M')

    # Build summary
    print("\n" + "═" * 70)
    print(f"  SWA-MLA Training │ {run_id}")
    print("═" * 70)

    # Model line
    arch_pattern = f"{args.local_layers_per_cycle}×DeltaNet + {args.mla_layers_per_cycle}×MLA"
    print(f"  Model: {args.size} ({format_params(total_params)} params) │ {arch_pattern}")

    # MoE line if enabled
    if args.use_moe:
        if args.use_latent_moe:
            eff_experts = args.latent_n_experts or (args.n_experts * args.latent_ratio)
            eff_active = args.latent_n_activated or (args.n_activated * args.latent_ratio)
            moe_info = f"LatentMoE {eff_experts}E/{eff_active}A (ratio={args.latent_ratio})"
        else:
            moe_info = f"MoE {args.n_experts}E/{args.n_activated}A"
        if args.n_shared_experts > 0:
            moe_info += f" +{args.n_shared_experts}shared"
        print(f"  MoE: {moe_info}")

    # Training config line
    mode_str = " [WeDLM]" if args.use_wedlm else ""
    print(f"  Training: bs={eff_batch} (×{args.gradient_accumulation_steps} accum) │ seq={args.block_size} │ lr={args.learning_rate:.0e}{mode_str}")

    # Optimizer and device line
    compile_str = f"compile={args.compile_mode}" if args.compile else "no-compile"
    grad_ckpt = " +ckpt" if args.gradient_checkpointing else ""
    print(f"  Optim: {args.optimizer_type} │ {compile_str}{grad_ckpt} │ {world_size}×{gpu_short}")

    # Resume info if applicable
    if resume_step > 0:
        tokens_str = format_params(resume_tokens).replace('.0', '')
        print(f"  Resume: step {resume_step:,} ({tokens_str} tokens)")

    print("═" * 70)
    print(f"  Started: {timestamp} │ max_iters: {args.max_iters:,}")
    print("═" * 70 + "\n")


def configure_cuda_optimizations():
    """Configure CUDA optimizations for maximum throughput."""
    if not torch.cuda.is_available():
        return
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False


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


def get_wsd_sched(it, warmup_iters, max_iters, learning_rate, min_lr, stable_ratio=0.8):
    """
    Warmup-Stable-Decay (WSD) scheduler.
    
    Phases:
    1. Warmup: Linear increase from 0 to learning_rate
    2. Stable: Constant learning_rate
    3. Decay: Linear/Cosine decay to min_lr
    """
    # 1. Warmup
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    
    # 2. Stable
    decay_start = int(max_iters * stable_ratio)
    if it < decay_start:
        return learning_rate
        
    # 3. Decay
    if it >= max_iters:
        return min_lr
        
    # 1-cosine decay for the last phase
    decay_steps = max_iters - decay_start
    step_in_decay = it - decay_start
    decay_ratio = step_in_decay / decay_steps
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)


def get_lr(it, warmup_iters, max_iters, learning_rate, min_lr):
    """Wrapper for scheduler selection."""
    # Use WSD by default for this high-performance run
    return get_wsd_sched(it, warmup_iters, max_iters, learning_rate, min_lr)


def configure_optimizer(model, learning_rate, weight_decay, betas, device_type, optimizer_type='adamw', engram_lr_multiplier=5.0):
    """Configure optimizer with proper parameter grouping.

    Engram parameters are handled specially:
    - Engram embedding tables: 5x LR (engram_lr_multiplier), no weight decay
    - Engram other params (w_k, w_v, conv): normal LR, weight decay
    """

    if optimizer_type == 'muon':
        # Use native PyTorch Muon (requires PyTorch 2.6+)
        if not hasattr(torch.optim, 'Muon'):
            optimizer_type = 'adamw'
        else:
            Muon = torch.optim.Muon
            muon_lr = learning_rate * 200
            adamw_lr = learning_rate
            engram_embed_lr = learning_rate * engram_lr_multiplier

            muon_params = []
            adamw_params = []
            engram_embed_params = []  # Engram embeddings: high LR, no decay

            for name, param in model.named_parameters():
                if not param.requires_grad:
                    continue
                # Engram embedding tables: special treatment (5x LR, no decay)
                if 'engram' in name and 'embeddings' in name and 'tables' in name:
                    engram_embed_params.append(param)
                elif any(nd in name for nd in ['wte', 'wpe', 'lm_head', 'embed']):
                    adamw_params.append(param)
                elif param.ndim == 2:
                    muon_params.append(param)
                else:
                    adamw_params.append(param)

            optimizers = []
            if muon_params:
                optimizers.append(Muon(
                    muon_params,
                    lr=muon_lr,
                    momentum=0.95,
                    weight_decay=weight_decay,
                    nesterov=True,
                    ns_steps=5,
                ))
            if adamw_params:
                optimizers.append(torch.optim.AdamW(adamw_params, lr=adamw_lr, betas=betas, weight_decay=weight_decay, fused=True))
            # Engram embeddings: separate optimizer with high LR and no weight decay
            if engram_embed_params:
                optimizers.append(torch.optim.AdamW(
                    [{'params': engram_embed_params, 'weight_decay': 0.0}],
                    lr=engram_embed_lr, betas=betas, fused=True
                ))

            return optimizers

    # Standard AdamW configuration with Engram support
    decay_params = []
    no_decay_params = []
    engram_embed_params = []  # Engram embeddings: high LR, no decay
    engram_other_params = []  # Engram w_k, w_v, conv: normal LR, with decay

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # Engram embedding tables: 5x LR, no weight decay (paper spec)
        if 'engram' in name and 'embeddings' in name and 'tables' in name:
            engram_embed_params.append(param)
        # Engram other params (w_k, w_v, conv weights): normal LR with decay
        elif 'engram' in name:
            if any(nd in name for nd in ['.bias', 'norm']):
                no_decay_params.append(param)
            else:
                engram_other_params.append(param)
        # Standard no-decay params
        elif any(nd in name for nd in ['.bias', 'norm', 'ln_', 'wte', 'wpe']):
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    param_groups = [
        {'params': decay_params, 'weight_decay': weight_decay, 'lr': learning_rate},
        {'params': no_decay_params, 'weight_decay': 0.0, 'lr': learning_rate},
    ]

    # Add Engram param groups
    if engram_embed_params:
        param_groups.append({
            'params': engram_embed_params,
            'weight_decay': 0.0,  # No weight decay for embeddings
            'lr': learning_rate * engram_lr_multiplier,  # 5x LR
        })
    if engram_other_params:
        param_groups.append({
            'params': engram_other_params,
            'weight_decay': weight_decay,
            'lr': learning_rate,
        })

    if device_type == 'cuda':
        optimizer = torch.optim.AdamW(param_groups, lr=learning_rate, betas=betas, fused=True)
    else:
        optimizer = torch.optim.AdamW(param_groups, lr=learning_rate, betas=betas)

    return optimizer


def load_latest_from_huggingface(repo_id, hf_token=None):
    """Load the latest checkpoint from HuggingFace Hub.

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

        print(f"Loading latest checkpoint from {repo_id}...")

        # List all files in the repo
        files = list_repo_files(repo_id, token=hf_token)

        # Find all checkpoint directories (format: checkpoint_tokens_XXX_loss_Y.YYYY)
        checkpoint_pattern = re.compile(r'checkpoint_tokens_(\d+[kKmMbB])_loss_([\d.]+)/pytorch_model\.bin')
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
            print(f"No checkpoints found in {repo_id}")
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
    """Push model to Hugging Face Hub with automatic naming."""
    if not HF_AVAILABLE:
        print("huggingface_hub not available - skipping HF push")
        return

    if not hf_token:
        print("HF_TOKEN not set - skipping HF push")
        return

    try:
        # Create temporary save directory with informative name
        # Format: tokens_XXXk_loss_Y.YYYY
        tokens_str = f"{total_tokens // 1000}k" if total_tokens < 1_000_000 else f"{total_tokens // 1_000_000}M"
        model_name = f"checkpoint_tokens_{tokens_str}_loss_{val_loss:.4f}"
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
            if isinstance(optimizer, list):
                checkpoint_data['optimizer_state_dict'] = [opt.state_dict() for opt in optimizer]
            else:
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

# SWA-MLA Model Checkpoint

**Training Progress:**
- Total tokens processed: {total_tokens:,}
- Validation loss: {val_loss:.4f}
- Perplexity: {math.exp(val_loss):.2f}

This is an automatic checkpoint uploaded during training when validation loss improved.

## Model Architecture
Hybrid architecture combining:
- **GatedDeltaNet** blocks for O(n) linear attention (local context)
- **Multi-head Latent Attention (MLA)** blocks for global context with KV compression
- **LatentMoE** for efficient mixture-of-experts in latent space

## Model Configuration
- Size: {config_args.get('size', 'unknown')}
- Block size (context length): {config_args.get('block_size', 'unknown')}
- DeltaNet layers per cycle: {config_args.get('local_layers_per_cycle', 'unknown')}
- MLA layers per cycle: {config_args.get('mla_layers_per_cycle', 'unknown')}
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

    # Configure CUDA optimizations (silent)
    configure_cuda_optimizations()

    # Configure TF32 precision (silent)
    enable_tf32 = False
    configure_tf32(enable_tf32=enable_tf32, verbose=False)

    # Setup wandb
    wandb_run = None
    if master_process and WANDB_AVAILABLE and args.wandb_project and hasattr(wandb, 'init'):
        if hasattr(wandb, 'login'):
            wandb.login()
        wandb_run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name or f"swa_mla_{args.size}",
            config=vars(args)
        )

    # Setup TensorBoard (silent)
    tb_writer = None
    if master_process and TENSORBOARD_AVAILABLE and args.use_tensorboard:
        tb_dir = os.path.join(args.output_dir, 'tensorboard', args.wandb_run_name or f"swa_mla_{args.size}_{time.strftime('%Y%m%d_%H%M%S')}")
        tb_writer = SummaryWriter(tb_dir)
        tb_writer.add_text('config', str(vars(args)), 0)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    # Configure tokenizer to support longer sequences (suppress warning)
    # GPT-2 tokenizer defaults to 1024, but our model supports longer sequences
    tokenizer.model_max_length = args.block_size
    vocab_size = len(tokenizer)

    # Try to load checkpoint from HuggingFace or local path if requested
    resume_checkpoint = None
    resume_step = 0
    resume_tokens = 0

    if args.resume_from:
        # Resume from local checkpoint path
        checkpoint_path = args.resume_from
        if os.path.isdir(checkpoint_path):
            import glob
            checkpoint_files = glob.glob(os.path.join(checkpoint_path, "checkpoint_*.pt"))
            if checkpoint_files:
                def get_step(f):
                    try:
                        return int(os.path.basename(f).replace("checkpoint_", "").replace(".pt", ""))
                    except:
                        return 0
                checkpoint_files.sort(key=get_step, reverse=True)
                checkpoint_path = checkpoint_files[0]
            else:
                checkpoint_path = None

        if checkpoint_path and os.path.exists(checkpoint_path):
            try:
                resume_checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
                resume_step = resume_checkpoint.get('step', 0)
                resume_tokens = resume_checkpoint.get('total_tokens', 0)
            except Exception as e:
                if master_process:
                    print(f"⚠ Failed to load checkpoint: {e}")

    elif args.resume_from_hf and args.hf_repo_id:
        # Resume from HuggingFace
        hf_token = os.getenv("HF_TOKEN")
        resume_checkpoint = load_latest_from_huggingface(args.hf_repo_id, hf_token)
        if resume_checkpoint:
            resume_step = resume_checkpoint.get('step', 0)
            resume_tokens = resume_checkpoint.get('total_tokens', 0)

    # Setup FP8 training if requested
    fp8_recipe = None
    if args.use_fp8:
        if not TE_AVAILABLE:
            raise RuntimeError(
                "--use_fp8 requires transformer-engine. "
                "Install with: pip install transformer-engine"
            )
        # Create FP8 recipe with HYBRID format (E4M3 forward, E5M2 backward)
        fp8_recipe = DelayedScaling(
            fp8_format=Format.HYBRID,
            amax_history_len=16,
            amax_compute_algo="max",
        )
        if master_process:
            print(f"FP8 training enabled via Transformer Engine (HYBRID format)")

    # Prepare MoE kwargs
    moe_kwargs = {}
    if args.use_moe:
        moe_kwargs = {
            'use_moe': True,
            'n_experts': args.n_experts,
            'n_shared_experts': args.n_shared_experts,
            'n_activated': args.n_activated,
            'router_z_loss_coef': args.router_z_loss_coef,
        }
        if args.expert_dim is not None:
            moe_kwargs['expert_dim'] = args.expert_dim

        # LatentMoE configuration
        if args.use_latent_moe:
            moe_kwargs['use_latent_moe'] = True
            moe_kwargs['latent_ratio'] = args.latent_ratio
            moe_kwargs['latent_preserve_expert_dim'] = args.latent_preserve_expert_dim
            if args.latent_dim is not None:
                moe_kwargs['latent_dim'] = args.latent_dim
            if args.latent_n_experts is not None:
                moe_kwargs['latent_n_experts'] = args.latent_n_experts
            if args.latent_n_activated is not None:
                moe_kwargs['latent_n_activated'] = args.latent_n_activated

    # Parse Engram arguments
    engram_layers = []
    engram_ngram_orders = []
    if args.use_engram:
        engram_layers = [int(x.strip()) for x in args.engram_layers.split(',') if x.strip()]
        engram_ngram_orders = [int(x.strip()) for x in args.engram_ngram_orders.split(',') if x.strip()]

    # Common model kwargs
    model_kwargs = dict(
        size=args.size,
        vocab_size=vocab_size,
        block_size=args.block_size,
        dropout=args.dropout,
        local_layers_per_cycle=args.local_layers_per_cycle,
        mla_layers_per_cycle=args.mla_layers_per_cycle,
        q_lora_rank=args.mla_q_lora_rank,
        kv_lora_rank=args.mla_kv_lora_rank,
        qk_nope_head_dim=args.mla_qk_nope_head_dim,
        qk_rope_head_dim=args.mla_qk_rope_head_dim,
        v_head_dim=args.mla_v_head_dim,
        use_gradient_checkpointing=args.gradient_checkpointing,
        # Attention backend options
        use_flash_attention=args.use_flash_attention,
        use_varlen_attn=args.use_varlen_attn,
        use_triton_mla=args.use_triton_mla,
        use_triton_kernels=args.use_triton_kernels,
        use_gated_deltanet=args.use_gated_deltanet,
        # DeltaNet latent compression options
        deltanet_latent_dim=args.deltanet_latent_dim,
        deltanet_share_qk=args.deltanet_share_qk,
        # FP8 training via Transformer Engine
        use_te_fp8=args.use_fp8,
        # Engram: Conditional Memory via N-gram Lookup
        use_engram=args.use_engram,
        engram_layers=engram_layers,
        engram_d_mem=args.engram_d_mem,
        engram_n_hash_heads=args.engram_n_hash_heads,
        engram_ngram_orders=engram_ngram_orders,
        engram_conv_kernel=args.engram_conv_kernel,
        # MoE parameters
        **moe_kwargs,
    )

    model = create_swa_mla_model(**model_kwargs)
    model = model.to(device)

    # Set tokenizer compression for Engram if enabled (must be done on all processes)
    if args.use_engram:
        model.set_engram_tokenizer_compression(tokenizer)

    # Load model weights if resuming
    if resume_checkpoint:
        state_dict = resume_checkpoint['model_state_dict']
        if any(key.startswith('_orig_mod.') for key in state_dict.keys()):
            state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)

    # Setup WeDLM training if enabled
    wedlm_masker = None
    wedlm_loss_fn = None
    wedlm_config = None
    mask_token_id = None

    if args.use_wedlm:
        if not WEDLM_AVAILABLE:
            raise RuntimeError("WeDLM training requested but wedlm module not found.")

        # Determine mask token ID
        if args.wedlm_mask_token_id is not None:
            mask_token_id = args.wedlm_mask_token_id
        elif hasattr(tokenizer, 'mask_token_id') and tokenizer.mask_token_id is not None:
            mask_token_id = tokenizer.mask_token_id
        else:
            mask_token_id = vocab_size - 1

        wedlm_config = WeDLMConfig(
            use_wedlm_training=True,
            block_size=args.wedlm_block_size,
            min_mask_ratio=args.wedlm_min_mask_ratio,
            max_mask_ratio=args.wedlm_max_mask_ratio,
            ar_loss_weight=args.wedlm_ar_loss_weight,
            mask_token_id=mask_token_id,
        )

        wedlm_masker = DualStreamMasker(wedlm_config, mask_token_id)
        wedlm_loss_fn = WeDLMLoss(
            ar_loss_weight=args.wedlm_ar_loss_weight,
            label_smoothing=0.0,
            ignore_index=-100,
        )

        if args.use_gated_deltanet:
            model = adapt_deltanet_for_wedlm(model, max_seq_len=args.block_size * 2, d_model=None)

    # Initialize μP if enabled
    mup_config = None
    if args.use_mup and MUP_AVAILABLE:
        mup_config = MuPConfig(
            base_width=args.mup_base_width,
            width=model.config.n_embd,
        )
        mup_init(model, mup_config)
        if rank == 0:
            print(f"μP initialized: base_width={mup_config.base_width}, width_mult={mup_config.width_mult:.1f}x")

    # Initialize Progressive Scheduler if enabled
    progressive = None
    if args.use_progressive and PROGRESSIVE_AVAILABLE:
        progressive = ProgressiveScheduler.from_schedule(
            args.progressive_schedule,
            base_batch_size=args.batch_size,
            target_seq_len=args.block_size,
        )
        if rank == 0:
            print(f"Progressive training: {args.progressive_schedule}")

    # Initialize EMA if enabled
    ema = None
    if args.use_ema and EMA_AVAILABLE:
        ema = EMAModel(model, decay=args.ema_decay)
        if rank == 0:
            print(f"EMA enabled: decay={args.ema_decay}")

    # Log to wandb (silent)
    if master_process and wandb_run is not None:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        param_memory = total_params * 4 / (1024**3)
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
        compile_mode = args.compile_mode
        if args.gradient_accumulation_steps > 1 and compile_mode == 'max-autotune':
            compile_mode = 'reduce-overhead'
        model = torch.compile(model, mode=compile_mode)
        # Update args for banner display
        args.compile_mode = compile_mode

    # Wrap with DDP
    if is_ddp:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
        raw_model = model.module
    else:
        raw_model = model

    # Print compact training banner
    if master_process:
        print_training_banner(args, raw_model, world_size, device, resume_step, resume_tokens)

    # Get initial seq_len and batch_size (may be from progressive scheduler)
    initial_seq_len = args.block_size
    initial_batch_size = args.batch_size
    if progressive is not None:
        initial_seq_len, initial_batch_size = progressive.get_current_config(resume_tokens)
        if rank == 0:
            print(f"Progressive training: starting at seq_len={initial_seq_len}, batch_size={initial_batch_size}")

    data_loader = PackedFinewebDataset(
        split='train',
        max_length=initial_seq_len,
        batch_size=initial_batch_size,
        tokenizer=tokenizer,
        shuffle=True,
        num_workers=args.num_workers,
    )

    # Configure optimizer (μP-aware if enabled)
    if mup_config is not None:
        optimizer = configure_mup_optimizer(
            raw_model, mup_config,
            base_lr=args.learning_rate,
            weight_decay=args.weight_decay,
            betas=(args.beta1, args.beta2),
            engram_lr_mult=args.engram_lr_multiplier if args.use_engram else 1.0,
            device_type='cuda' if torch.cuda.is_available() else 'cpu',
            optimizer_type=args.optimizer_type,
        )
    else:
        optimizer = configure_optimizer(
            raw_model,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            betas=(args.beta1, args.beta2),
            device_type='cuda' if torch.cuda.is_available() else 'cpu',
            optimizer_type=args.optimizer_type,
            engram_lr_multiplier=args.engram_lr_multiplier if args.use_engram else 1.0,
        )

    # Load optimizer state if resuming
    if resume_checkpoint and 'optimizer_state_dict' in resume_checkpoint:
        try:
            saved_state = resume_checkpoint['optimizer_state_dict']
            if isinstance(optimizer, list) and isinstance(saved_state, list):
                for opt, state in zip(optimizer, saved_state):
                    opt.load_state_dict(state)
            elif not isinstance(optimizer, list) and not isinstance(saved_state, list):
                optimizer.load_state_dict(saved_state)
            # Silently skip mismatched structures
        except Exception:
            pass  # Silently continue with fresh optimizer state

    # Load EMA state if resuming
    if resume_checkpoint and 'ema' in resume_checkpoint and ema is not None:
        try:
            ema.load_state_dict(resume_checkpoint['ema'])
            if rank == 0:
                print("Restored EMA state from checkpoint")
        except Exception as e:
            if rank == 0:
                print(f"Warning: Could not restore EMA state: {e}")

    # Load progressive scheduler state if resuming
    if resume_checkpoint and 'progressive' in resume_checkpoint and progressive is not None:
        try:
            prog_state = resume_checkpoint['progressive']
            progressive.current_phase_idx = prog_state.get('current_phase_idx', 0)
            progressive._last_phase_idx = prog_state.get('_last_phase_idx', 0)
            if rank == 0:
                print(f"Restored progressive scheduler state: phase {progressive.current_phase_idx}")
        except Exception as e:
            if rank == 0:
                print(f"Warning: Could not restore progressive state: {e}")

    # Setup profiler if enabled
    profiler = None
    profiler_results = []  # Store results for final summary

    def profiler_trace_handler(prof):
        """Custom handler that prints summary tables instead of TensorBoard."""
        profiler_results.append(prof.key_averages())

    if args.profile and master_process:
        profiler = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(
                wait=0,
                warmup=args.profile_warmup,
                active=args.profile_steps,
                repeat=1
            ),
            on_trace_ready=profiler_trace_handler,
            record_shapes=True,
            profile_memory=True,
            with_stack=False,  # Disable stack for cleaner output
            with_flops=True,
        )
        profiler.start()

    data_iter = iter(data_loader)
    scaler = torch.amp.GradScaler('cuda', enabled=False)

    # Create a separate CUDA stream for async data prefetching
    prefetch_stream = torch.cuda.Stream() if torch.cuda.is_available() else None

    running_loss = 0.0
    t0 = time.time()

    # Track best validation loss for HF push
    best_val_loss = float('inf')
    total_tokens_seen = resume_tokens  # Start from resumed token count

    # Adjust starting step
    start_step = resume_step

    # Store initial LR ratios for each optimizer to maintain proportions during scheduling
    # This is important for Muon which uses a much higher LR than AdamW
    optimizers_list = optimizer if isinstance(optimizer, list) else [optimizer]
    initial_lrs = []
    for opt in optimizers_list:
        opt_lrs = [pg['lr'] for pg in opt.param_groups]
        initial_lrs.append(opt_lrs)

    # Prefetch first batch
    next_batch = None
    try:
        next_batch = next(data_iter)
    except StopIteration:
        data_iter = iter(data_loader)
        next_batch = next(data_iter)

    # Track current progressive config for detecting changes
    current_seq_len = args.block_size
    current_batch_size = args.batch_size

    for step in range(start_step, args.max_iters):
        # Progressive training: check for phase transition
        if progressive is not None:
            new_seq_len, new_batch_size = progressive.get_current_config(total_tokens_seen)
            if progressive.check_phase_transition(total_tokens_seen):
                if rank == 0:
                    print(f"\n>>> Progressive transition: seq_len={new_seq_len}, batch_size={new_batch_size}")

                # Recreate data loader with new configuration
                try:
                    data_loader.close()  # Clean up existing data loader
                except Exception:
                    pass

                data_loader = PackedFinewebDataset(
                    split='train',
                    max_length=new_seq_len,
                    batch_size=new_batch_size,
                    tokenizer=tokenizer,
                    shuffle=True,
                    num_workers=args.num_workers,
                    start_offset=total_tokens_seen // new_seq_len,  # Approximate position
                )
                data_iter = iter(data_loader)
                current_seq_len = new_seq_len
                current_batch_size = new_batch_size

                # Prefetch first batch from new loader
                try:
                    next_batch = next(data_iter)
                except StopIteration:
                    data_iter = iter(data_loader)
                    next_batch = next(data_iter)

                if rank == 0:
                    print(f">>> Data loader reconfigured: seq_len={new_seq_len}, batch_size={new_batch_size}")

        # Update learning rate - compute the schedule ratio
        scheduled_lr = get_lr(step, args.warmup_iters, args.max_iters, args.learning_rate, args.min_lr)
        lr_ratio = scheduled_lr / args.learning_rate if args.learning_rate > 0 else 1.0

        # Apply ratio to each optimizer's initial LR to maintain proportions
        for opt_idx, opt in enumerate(optimizers_list):
            for pg_idx, param_group in enumerate(opt.param_groups):
                new_lr = initial_lrs[opt_idx][pg_idx] * lr_ratio
                if isinstance(param_group['lr'], torch.Tensor):
                    param_group['lr'].fill_(new_lr)
                else:
                    param_group['lr'] = new_lr

        # For logging, use the base scheduled LR
        lr = scheduled_lr

        # Training step with gradient accumulation
        model.train()

        # Zero grad for all optimizers (set_to_none is faster than zero_grad)
        if isinstance(optimizer, list):
            for opt in optimizer:
                opt.zero_grad(set_to_none=True)
        else:
            optimizer.zero_grad(set_to_none=True)

        accum_loss = 0.0

        for micro_step in range(args.gradient_accumulation_steps):
            # Use prefetched batch
            batch = next_batch

            # Prefetch next batch while current one is being processed
            if prefetch_stream is not None:
                with torch.cuda.stream(prefetch_stream):
                    try:
                        next_batch = next(data_iter)
                    except StopIteration:
                        data_iter = iter(data_loader)
                        next_batch = next(data_iter)
            else:
                try:
                    next_batch = next(data_iter)
                except StopIteration:
                    data_iter = iter(data_loader)
                    next_batch = next(data_iter)

            input_ids = batch['input_ids'].to(device, non_blocking=True)
            labels = batch['labels'].to(device, non_blocking=True)

            # Extract varlen metadata if available
            cu_seqlens = None
            max_seqlen = None
            if args.use_varlen_attn and 'cu_seqlens' in batch:
                cu_seqlens = batch['cu_seqlens'].to(device, non_blocking=True)
                max_seqlen = batch.get('max_seqlen', None)

            # Create FP8 autocast context if enabled
            # For DDP, pass fp8_group for synchronized scaling across GPUs
            fp8_group = dist.group.WORLD if is_ddp and args.use_fp8 else None
            fp8_ctx = te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe, fp8_group=fp8_group) if args.use_fp8 else nullcontext()

            # Forward pass with mixed precision (BF16 + optional FP8 via TE)
            with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=True):
                with fp8_ctx:
                    if args.use_wedlm and wedlm_masker is not None:
                        # WeDLM training: dual-stream forward pass
                        wedlm_batch = wedlm_masker(input_ids)

                        dual_input_ids = wedlm_batch['dual_input_ids']
                        dual_position_ids = wedlm_batch['dual_position_ids']
                        dual_attention_mask = wedlm_batch['dual_attention_mask']
                        target_ids = wedlm_batch['target_ids']
                        target_mask = wedlm_batch['target_mask']
                        mask_ratios = wedlm_batch['mask_ratios']

                        # Forward pass with dual-stream inputs
                        logits, _ = model(
                            dual_input_ids,
                            position_ids=dual_position_ids,
                            attention_mask_2d=dual_attention_mask,
                            return_all_logits=True,
                        )

                        # Extract prediction stream logits (second half)
                        L = input_ids.size(1)
                        pred_logits = logits[:, L:, :]  # [B, L, V]

                        # Compute WeDLM loss
                        loss, loss_dict = wedlm_loss_fn(
                            pred_logits,
                            target_ids,
                            target_mask,
                            mask_ratios,
                            wedlm_config.block_size,
                        )
                    else:
                        # Standard AR training
                        logits, loss = model(
                            input_ids,
                            targets=labels,
                            cu_seqlens=cu_seqlens,
                            max_seqlen=max_seqlen,
                        )

                    # Add MoE auxiliary loss if model has MoE layers
                    if hasattr(raw_model, 'get_moe_aux_loss'):
                        moe_aux_loss = raw_model.get_moe_aux_loss()
                        loss = loss + moe_aux_loss
                    loss = loss / args.gradient_accumulation_steps

            # Backward pass
            scaler.scale(loss).backward()
            accum_loss += loss.item()

        # Update MoE router biases after backward (must be after backward for checkpoint compatibility)
        if hasattr(raw_model, 'update_moe_bias'):
            raw_model.update_moe_bias()

        # Gradient clipping
        if args.grad_clip > 0:
            # Unscale all optimizers for gradient clipping
            if isinstance(optimizer, list):
                for opt in optimizer:
                    scaler.unscale_(opt)
            else:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

        # Optimizer step
        if isinstance(optimizer, list):
            for opt in optimizer:
                scaler.step(opt)
        else:
            scaler.step(optimizer)
        scaler.update()

        # Update EMA weights
        if ema is not None:
            ema.update(raw_model)

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

            # Include current seq_len/batch_size for progressive training tracking
            prog_info = f" | BS={current_batch_size}x{current_seq_len}" if progressive is not None else ""
            print(f"Step {step:6d} | Loss: {lossf:.4f} | LR: {lr:.2e} | Tokens/sec: {tokens_per_sec:,.0f} | Total: {tokens_str}{prog_info}")

            if wandb_run is not None:
                log_dict = {
                    'train/loss': lossf,
                    'train/lr': lr,
                    'train/tokens_per_sec': tokens_per_sec,
                    'train/total_tokens': total_tokens_seen,
                    'step': step
                }
                if progressive is not None:
                    log_dict['train/seq_len'] = current_seq_len
                    log_dict['train/batch_size'] = current_batch_size
                wandb.log(log_dict)
            
            if tb_writer is not None:
                tb_writer.add_scalar('train/loss', lossf, step)
                tb_writer.add_scalar('train/lr', lr, step)
                tb_writer.add_scalar('train/tokens_per_sec', tokens_per_sec, step)
                tb_writer.add_scalar('train/total_tokens', total_tokens_seen, step)
                tb_writer.add_scalar('train/perplexity', math.exp(lossf) if lossf < 10 else float('inf'), step)
                if progressive is not None:
                    tb_writer.add_scalar('train/seq_len', current_seq_len, step)
                    tb_writer.add_scalar('train/batch_size', current_batch_size, step)

            # Log Engram metrics if enabled
            if args.use_engram:
                engram_metrics = {}
                # Get the raw model (unwrap DDP if needed)
                raw_model = model.module if hasattr(model, 'module') else model
                for block in raw_model.transformer.h:
                    if hasattr(block, 'engram') and block.engram is not None:
                        layer_metrics = block.engram.get_metrics()
                        for k, v in layer_metrics.items():
                            if k not in engram_metrics:
                                engram_metrics[k] = []
                            engram_metrics[k].append(v)

                # Average metrics across all Engram layers and log
                if engram_metrics:
                    for k, values in engram_metrics.items():
                        avg_value = sum(values) / len(values)
                        if wandb_run is not None:
                            wandb.log({k: avg_value, 'step': step})
                        if tb_writer is not None:
                            tb_writer.add_scalar(k, avg_value, step)

        # Validation (using next batches from same data loader)
        # Skip validation at the exact resume step to avoid immediate validation after loading
        if step % args.eval_interval == 0 and step > start_step and master_process:
            model.eval()
            val_loss = 0.0
            val_steps = 50

            # Use EMA weights for validation if available
            ema_ctx = ema.apply(raw_model) if ema is not None else nullcontext()
            with ema_ctx, torch.no_grad():
                for _ in range(val_steps):
                    try:
                        batch = next(data_iter)
                    except StopIteration:
                        data_iter = iter(data_loader)
                        batch = next(data_iter)

                    input_ids = batch['input_ids'].to(device, non_blocking=True)
                    labels = batch['labels'].to(device, non_blocking=True)

                    # Extract varlen metadata if available
                    val_cu_seqlens = None
                    val_max_seqlen = None
                    if args.use_varlen_attn and 'cu_seqlens' in batch:
                        val_cu_seqlens = batch['cu_seqlens'].to(device, non_blocking=True)
                        val_max_seqlen = batch.get('max_seqlen', None)

                    # FP8 context for validation (no DDP sync needed in eval)
                    val_fp8_ctx = te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe) if args.use_fp8 else nullcontext()

                    with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=True):
                        with val_fp8_ctx:
                            if args.use_wedlm and wedlm_masker is not None:
                                # WeDLM validation
                                wedlm_batch = wedlm_masker(input_ids)
                                dual_input_ids = wedlm_batch['dual_input_ids']
                                dual_position_ids = wedlm_batch['dual_position_ids']
                                dual_attention_mask = wedlm_batch['dual_attention_mask']
                                target_ids = wedlm_batch['target_ids']
                                target_mask = wedlm_batch['target_mask']
                                mask_ratios = wedlm_batch['mask_ratios']

                                logits, _ = model(
                                    dual_input_ids,
                                    position_ids=dual_position_ids,
                                    attention_mask_2d=dual_attention_mask,
                                    return_all_logits=True,
                                )

                                L = input_ids.size(1)
                                pred_logits = logits[:, L:, :]

                                loss, _ = wedlm_loss_fn(
                                    pred_logits,
                                    target_ids,
                                    target_mask,
                                    mask_ratios,
                                    wedlm_config.block_size,
                                )
                            else:
                                logits, loss = model(
                                    input_ids,
                                    targets=labels,
                                    cu_seqlens=val_cu_seqlens,
                                    max_seqlen=val_max_seqlen,
                                )

                    val_loss += loss.item()

            val_loss /= val_steps
            perplexity = math.exp(val_loss) if val_loss < 10 else float('inf')

            mode_str = " (WeDLM)" if args.use_wedlm else ""
            print(f"\nValidation{mode_str} | Loss: {val_loss:.4f} | Perplexity: {perplexity:.2f}\n")

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
            # Handle list of optimizers (e.g., Muon setup)
            if isinstance(optimizer, list):
                optimizer_state = [opt.state_dict() for opt in optimizer]
            else:
                optimizer_state = optimizer.state_dict()

            # Build config dict with effective values (not just args)
            config_dict = vars(args).copy()

            # For WeDLM, save the actual mask_token_id used (not None)
            if args.use_wedlm and mask_token_id is not None:
                config_dict['wedlm_mask_token_id'] = mask_token_id

            checkpoint = {
                'model_state_dict': raw_model.state_dict(),
                'optimizer_state_dict': optimizer_state,
                'step': step,
                'config': config_dict,
                'total_tokens': total_tokens_seen,
            }

            # Save EMA state if enabled
            if ema is not None:
                checkpoint['ema'] = ema.state_dict()

            # Save progressive scheduler state if enabled
            if progressive is not None:
                checkpoint['progressive'] = {
                    'current_phase_idx': progressive.current_phase_idx,
                    '_last_phase_idx': progressive._last_phase_idx,
                }

            checkpoint_path = os.path.join(args.output_dir, f'checkpoint_{step}.pt')
            torch.save(checkpoint, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")

        # Profiler step
        if profiler is not None:
            profiler.step()
            # Check if profiling is complete
            total_profile_steps = args.profile_warmup + args.profile_steps
            if step >= start_step + total_profile_steps - 1:
                profiler.stop()
                print(f"\n{'='*80}")
                print(f"✅ PROFILING COMPLETE - {args.profile_steps} steps recorded")
                print(f"{'='*80}\n")

                # Print summary tables from collected results
                if profiler_results:
                    key_avg = profiler_results[-1]  # Use last collected results

                    # Table 1: Top operations by CUDA time
                    print("📊 TOP 20 OPERATIONS BY CUDA TIME:")
                    print("-" * 80)
                    print(key_avg.table(sort_by="cuda_time_total", row_limit=20))

                    # Table 2: Top operations by CPU time
                    print("\n📊 TOP 15 OPERATIONS BY CPU TIME:")
                    print("-" * 80)
                    print(key_avg.table(sort_by="cpu_time_total", row_limit=15))

                    # Table 3: Memory usage
                    print("\n📊 TOP 15 OPERATIONS BY CUDA MEMORY:")
                    print("-" * 80)
                    print(key_avg.table(sort_by="cuda_memory_usage", row_limit=15))

                    # Summary statistics (use cpu_time as cuda_time may not be available)
                    total_cpu_time = sum(e.self_cpu_time_total for e in key_avg)
                    print(f"\n{'='*80}")
                    print(f"📈 SUMMARY:")
                    print(f"   Total CPU time: {total_cpu_time / 1e6:.2f} s")
                    print(f"{'='*80}\n")

                break  # Exit training loop after profiling

    # Cleanup profiler if still running
    if profiler is not None:
        try:
            profiler.stop()
        except Exception:
            pass  # Already stopped

    # Cleanup
    if tb_writer is not None:
        tb_writer.close()

    if is_ddp:
        dist.destroy_process_group()

    if master_process:
        print("\nTraining complete!")


def main():
    parser = argparse.ArgumentParser(description='Train DeltaNet-MLA model with LatentMoE')

    # Model parameters
    parser.add_argument('--size', type=str, default='moe-1b', choices=['small', 'base', 'large', 'xl', 'moe-1b', 'moe-2b', 'engram-moe-1b', 'mup-1b'])
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--block_size', type=int, default=2048)
    parser.add_argument('--dropout', type=float, default=0.0)

    # DeltaNet-MLA architecture parameters
    parser.add_argument('--local_layers_per_cycle', '--swa_layers_per_cycle', type=int, default=2,
                        help='DeltaNet/local attention blocks per cycle', dest='local_layers_per_cycle')
    parser.add_argument('--mla_layers_per_cycle', type=int, default=1, help='MLA blocks per cycle')
    parser.add_argument('--mla_q_lora_rank', type=int, default=0, help='MLA Q LoRA rank (0=disabled)')
    parser.add_argument('--mla_kv_lora_rank', type=int, default=256, help='MLA KV LoRA rank')
    parser.add_argument('--mla_qk_nope_head_dim', type=int, default=128)
    parser.add_argument('--mla_qk_rope_head_dim', type=int, default=64)
    parser.add_argument('--mla_v_head_dim', type=int, default=128)

    # DeltaNet options (always enabled)
    parser.add_argument('--use_flash_attention', action='store_true', default=True,
                        help='Use Flash Attention for MLA blocks')
    parser.add_argument('--use_varlen_attn', action='store_true', default=True,
                        help='Use varlen_attn (PyTorch 2.10+) for packed sequences without padding waste')
    parser.add_argument('--use_triton_mla', action='store_true', default=True,
                        help='Use custom Triton MLA kernel (H100 compatible, avoids FA2 CUDA graph issues)')
    parser.add_argument('--use_triton_kernels', action='store_true', default=True,
                        help='Use fused Triton kernels for SwiGLU and RMSNorm (15-25%% speedup)')
    parser.add_argument('--use_gated_deltanet', action='store_true', default=True,
                        help='Use GatedDeltaNet for local attention (O(n) linear)')

    # DeltaNet latent compression options
    parser.add_argument('--deltanet_latent_dim', type=int, default=0,
                        help='Latent dimension for DeltaNet projections (0=disabled, >0=compress Q/K/V/O)')
    parser.add_argument('--deltanet_share_qk', action='store_true', default=False,
                        help='Share Q and K projection in DeltaNet (K is normalized Q)')

    # Engram: Conditional Memory via N-gram Lookup
    parser.add_argument('--use_engram', action='store_true', default=False,
                        help='Enable Engram conditional memory module')
    parser.add_argument('--engram_layers', type=str, default='2,6',
                        help='Comma-separated layer indices for Engram (e.g., "2,6")')
    parser.add_argument('--engram_d_mem', type=int, default=512,
                        help='Engram memory embedding dimension')
    parser.add_argument('--engram_n_hash_heads', type=int, default=4,
                        help='Number of hash heads per N-gram order')
    parser.add_argument('--engram_ngram_orders', type=str, default='2,3',
                        help='Comma-separated N-gram orders (e.g., "2,3")')
    parser.add_argument('--engram_conv_kernel', type=int, default=4,
                        help='Engram causal convolution kernel size')
    parser.add_argument('--engram_lr_multiplier', type=float, default=5.0,
                        help='Learning rate multiplier for Engram embedding tables')

    # μP arguments
    parser.add_argument('--use_mup', action='store_true', help='Enable μP (Maximal Update Parametrization)')
    parser.add_argument('--mup_base_width', type=int, default=256, help='μP base width for LR scaling')

    # Progressive training arguments
    parser.add_argument('--use_progressive', action='store_true', help='Enable progressive sequence length training')
    parser.add_argument('--progressive_schedule', type=str, default='512:500M,1024:2B,2048:inf',
                        help='Progressive schedule: seq_len:tokens,... (e.g., "512:500M,1024:2B,2048:inf")')

    # EMA arguments
    parser.add_argument('--use_ema', action='store_true', help='Enable EMA weight averaging')
    parser.add_argument('--ema_decay', type=float, default=0.9999, help='EMA decay factor')

    # LatentMoE parameters (always enabled for moe-* sizes)
    parser.add_argument('--use_moe', action='store_true', default=True, help='Enable MoE for MLA blocks')
    parser.add_argument('--use_latent_moe', action='store_true', default=True,
                        help='Use LatentMoE: projects to latent space before expert computation')
    parser.add_argument('--n_experts', type=int, default=32, help='Number of routed experts')
    parser.add_argument('--n_shared_experts', type=int, default=1, help='Number of shared experts (always active)')
    parser.add_argument('--n_activated', type=int, default=3, help='Number of routed experts activated per token')
    parser.add_argument('--expert_dim', type=int, default=None, help='Expert hidden dimension (default: n_embd)')
    parser.add_argument('--router_z_loss_coef', type=float, default=0.001, help='Router Z-loss coefficient')
    parser.add_argument('--latent_ratio', type=int, default=4,
                        help='d_model/latent_dim ratio (default: 4, i.e., latent_dim = n_embd/4)')
    parser.add_argument('--latent_dim', type=int, default=None,
                        help='Explicit latent dimension (overrides latent_ratio if set)')
    parser.add_argument('--latent_n_experts', type=int, default=None,
                        help='Total experts for LatentMoE (default: n_experts * latent_ratio)')
    parser.add_argument('--latent_n_activated', type=int, default=None,
                        help='Activated experts for LatentMoE (default: n_activated * latent_ratio)')
    parser.add_argument('--latent_preserve_expert_dim', action='store_true', default=False,
                        help='Keep full expert_dim in LatentMoE (more params, more capacity)')

    # Training parameters
    parser.add_argument('--max_iters', type=int, default=100000)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--min_lr', type=float, default=1e-5)
    parser.add_argument('--weight_decay', type=float, default=0.1)
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.95)
    parser.add_argument('--warmup_iters', type=int, default=400)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--gradient_checkpointing', action='store_true')

    # Optimizer (Muon + AdamW or pure AdamW)
    parser.add_argument('--optimizer_type', type=str, default='muon', choices=['adamw', 'muon'])

    # Data parameters
    parser.add_argument('--tokenizer_name', type=str, default='openai-community/gpt2')
    parser.add_argument('--num_workers', type=int, default=8)

    # Logging and checkpointing
    parser.add_argument('--output_dir', type=str, default='outputs/deltanet_mla')
    parser.add_argument('--log_interval', type=int, default=10)
    parser.add_argument('--eval_interval', type=int, default=1000)
    parser.add_argument('--save_interval', type=int, default=5000)
    parser.add_argument('--wandb_project', type=str, default="swamla")
    parser.add_argument('--wandb_run_name', type=str, default=None)

    # Hugging Face integration
    parser.add_argument('--hf_repo_id', type=str, default=None, help='HuggingFace repo ID')
    parser.add_argument('--resume_from_hf', action='store_true', help='Resume from HF checkpoint')
    parser.add_argument('--resume_from', type=str, default=None, help='Resume from local checkpoint path')

    # Performance
    parser.add_argument('--compile', action='store_true', default=True)
    parser.add_argument('--compile_mode', type=str, default='max-autotune',
                        choices=['reduce-overhead', 'max-autotune', 'default'],
                        help='torch.compile mode')

    # FP8 Training (Transformer Engine)
    parser.add_argument('--use_fp8', action='store_true', default=True,
                        help='Enable FP8 training via Transformer Engine (requires H100/H200)')

    # TensorBoard
    parser.add_argument('--use_tensorboard', action='store_true', default=True)

    # WeDLM (Causal Diffusion Language Model) training
    parser.add_argument('--use_wedlm', action='store_true', default=False,
                        help='Enable WeDLM training with dual-stream masking')
    parser.add_argument('--wedlm_block_size', type=int, default=32,
                        help='WeDLM prediction block size')
    parser.add_argument('--wedlm_min_mask_ratio', type=float, default=0.1,
                        help='Minimum masking ratio per block')
    parser.add_argument('--wedlm_max_mask_ratio', type=float, default=1.0,
                        help='Maximum masking ratio per block')
    parser.add_argument('--wedlm_ar_loss_weight', type=float, default=0.5,
                        help='Weight for auxiliary AR loss (0 to disable)')
    parser.add_argument('--wedlm_mask_token_id', type=int, default=None,
                        help='Token ID for [MASK] (default: use tokenizer.mask_token_id or vocab_size-1)')

    # Profiling
    parser.add_argument('--profile', action='store_true', default=False,
                        help='Enable PyTorch profiler for performance analysis')
    parser.add_argument('--profile_steps', type=int, default=5,
                        help='Number of steps to profile (default: 5)')
    parser.add_argument('--profile_warmup', type=int, default=2,
                        help='Number of warmup steps before profiling (default: 2)')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Run training
    train(args)


if __name__ == '__main__':
    main()
