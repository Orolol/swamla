"""
Standalone training script for SWA-MLA model.
Supports FP8, Lion optimizer, wandb logging, and packed data loading.
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
from data_loader_packed import PackedFinewebDataset
from fp8_trainer import FP8AdamW, FP8Lion

# Try to import wandb
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("wandb not available - logging to console only")

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
    """Configure optimizer with proper parameter grouping."""
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

    # Choose optimizer
    if use_fp8:
        if optimizer_type == 'lion':
            print("Using FP8Lion optimizer")
            optimizer = FP8Lion(param_groups, lr=learning_rate, betas=betas)
        else:
            print("Using FP8AdamW optimizer")
            optimizer = FP8AdamW(param_groups, lr=learning_rate, betas=betas)
    elif optimizer_type == 'lion' and LION_AVAILABLE:
        print("Using Lion optimizer")
        optimizer = Lion(param_groups, lr=learning_rate, betas=betas)
    else:
        print("Using AdamW optimizer")
        if device_type == 'cuda':
            optimizer = torch.optim.AdamW(param_groups, lr=learning_rate, betas=betas, fused=True)
        else:
            optimizer = torch.optim.AdamW(param_groups, lr=learning_rate, betas=betas)

    return optimizer


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

    # Setup wandb (will be updated with model stats later)
    wandb_run = None
    if master_process and WANDB_AVAILABLE and args.wandb_project:
        wandb_run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name or f"swa_mla_{args.size}",
            config=vars(args)
        )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    vocab_size = len(tokenizer)

    # Create model
    if master_process:
        print(f"\nCreating {args.size} SWA-MLA model...")

    model = create_swa_mla_model(
        size=args.size,
        vocab_size=vocab_size,
        block_size=args.block_size,
        dropout=args.dropout,
        swa_layers_per_cycle=args.swa_layers_per_cycle,
        mla_layers_per_cycle=args.mla_layers_per_cycle,
        swa_window=args.swa_window,
        swa_sink_size=args.swa_sink_size,
        q_lora_rank=args.mla_q_lora_rank,
        kv_lora_rank=args.mla_kv_lora_rank,
        qk_nope_head_dim=args.mla_qk_nope_head_dim,
        qk_rope_head_dim=args.mla_qk_rope_head_dim,
        v_head_dim=args.mla_v_head_dim,
        use_fp8=args.use_fp8,
        use_gradient_checkpointing=args.gradient_checkpointing,
    )

    model = model.to(device)

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
        model = torch.compile(model)

    # Wrap with DDP
    if is_ddp:
        model = DDP(model, device_ids=[local_rank])
        raw_model = model.module
    else:
        raw_model = model

    # Setup data loader (single instance for both train and val)
    if master_process:
        print("\nSetting up data loader...")

    data_loader = PackedFinewebDataset(
        split='train',
        max_length=args.block_size,
        batch_size=args.batch_size,
        tokenizer=tokenizer,
        shuffle=True,
        num_workers=args.num_workers,
    )

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

    # Training loop
    if master_process:
        print("\nStarting training...")
        print(f"Max iterations: {args.max_iters:,}")
        print(f"Gradient accumulation steps: {args.gradient_accumulation_steps}")
        print(f"Effective batch size: {args.batch_size * args.gradient_accumulation_steps * world_size:,}")

    data_iter = iter(data_loader)
    scaler = torch.amp.GradScaler('cuda', enabled=args.use_fp8)

    running_loss = 0.0
    t0 = time.time()

    for step in range(args.max_iters):
        # Update learning rate
        lr = get_lr(step, args.warmup_iters, args.max_iters, args.learning_rate, args.min_lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # Training step with gradient accumulation
        model.train()
        optimizer.zero_grad()
        accum_loss = 0.0

        for micro_step in range(args.gradient_accumulation_steps):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(data_loader)
                batch = next(data_iter)

            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)

            # Forward pass with mixed precision
            with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=True):    
                logits, loss = model(input_ids, targets=labels)
                loss = loss / args.gradient_accumulation_steps

            # Backward pass
            scaler.scale(loss).backward()
            accum_loss += loss.item()

        # Gradient clipping
        if args.grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

        # Optimizer step
        scaler.step(optimizer)
        scaler.update()

        running_loss += accum_loss

        # Logging
        if step % args.log_interval == 0 and master_process:
            t1 = time.time()
            dt = t1 - t0
            t0 = t1

            lossf = running_loss / args.log_interval if step > 0 else accum_loss
            running_loss = 0.0

            tokens_per_sec = (args.batch_size * args.block_size * args.gradient_accumulation_steps * world_size * args.log_interval) / dt

            print(f"Step {step:6d} | Loss: {lossf:.4f} | LR: {lr:.2e} | Tokens/sec: {tokens_per_sec:,.0f}")

            if WANDB_AVAILABLE and args.wandb_project:
                wandb.log({
                    'train/loss': lossf,
                    'train/lr': lr,
                    'train/tokens_per_sec': tokens_per_sec,
                    'step': step
                })

        # Validation (using next batches from same data loader)
        if step % args.eval_interval == 0 and step > 0 and master_process:
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
                    labels = batch['labels'].to(device)

                    with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=True):
                        logits, loss = model(input_ids, targets=labels)

                    val_loss += loss.item()

            val_loss /= val_steps
            perplexity = math.exp(val_loss) if val_loss < 10 else float('inf')

            print(f"\nValidation | Loss: {val_loss:.4f} | Perplexity: {perplexity:.2f}\n")

            if WANDB_AVAILABLE and args.wandb_project:
                wandb.log({
                    'val/loss': val_loss,
                    'val/perplexity': perplexity,
                    'step': step
                })

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
    parser = argparse.ArgumentParser(description='Train SWA-MLA model')

    # Model parameters
    parser.add_argument('--size', type=str, default='small', choices=['small', 'base', 'large', 'xl'])
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--block_size', type=int, default=2048)
    parser.add_argument('--dropout', type=float, default=0.0)

    # SWA-MLA specific parameters
    parser.add_argument('--swa_layers_per_cycle', type=int, default=2)
    parser.add_argument('--mla_layers_per_cycle', type=int, default=1)
    parser.add_argument('--swa_window', type=int, default=256)
    parser.add_argument('--swa_sink_size', type=int, default=4)
    parser.add_argument('--mla_q_lora_rank', type=int, default=0)
    parser.add_argument('--mla_kv_lora_rank', type=int, default=256)
    parser.add_argument('--mla_qk_nope_head_dim', type=int, default=128)
    parser.add_argument('--mla_qk_rope_head_dim', type=int, default=64)
    parser.add_argument('--mla_v_head_dim', type=int, default=128)

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

    # Optimizer
    parser.add_argument('--optimizer_type', type=str, default='adamw', choices=['adamw', 'lion'])
    parser.add_argument('--use_fp8', action='store_true')

    # Data parameters
    parser.add_argument('--tokenizer_name', type=str, default='openai-community/gpt2')
    parser.add_argument('--num_workers', type=int, default=8)

    # Logging and checkpointing
    parser.add_argument('--output_dir', type=str, default='outputs/swa_mla')
    parser.add_argument('--log_interval', type=int, default=10)
    parser.add_argument('--eval_interval', type=int, default=1000)
    parser.add_argument('--save_interval', type=int, default=5000)
    parser.add_argument('--wandb_project', type=str, default="swamla")
    parser.add_argument('--wandb_run_name', type=str, default=None)

    # Performance
    parser.add_argument('--compile', action='store_true')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Run training
    train(args)


if __name__ == '__main__':
    main()
