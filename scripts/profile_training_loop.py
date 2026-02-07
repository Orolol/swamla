"""
Profile each phase of the training loop to find CPU-GPU sync bottlenecks.

Measures: data loading, H2D transfer, forward, backward, grad clip,
cautious WD, optimizer step, EMA update, and Python overhead.
"""

import sys
import time
from pathlib import Path
from contextlib import nullcontext

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent / 'models'))
sys.path.insert(0, str(Path(__file__).parent.parent / 'data'))
sys.path.insert(0, str(Path(__file__).parent.parent / 'optimization'))

from swa_mla_model import create_swa_mla_model
from transformers import AutoTokenizer

# TE FP8
TE_AVAILABLE = False
te = None
try:
    import transformer_engine.pytorch as te
    from transformer_engine.common.recipe import Format, DelayedScaling
    TE_AVAILABLE = True
except ImportError:
    pass

# EMA
try:
    from swa import EMAModel
    EMA_AVAILABLE = True
except ImportError:
    EMA_AVAILABLE = False


def cuda_event_timer():
    """Create start/end CUDA events for timing."""
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    return start, end


def profile_training_loop():
    device = torch.device('cuda')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    print("Setting up model (full preset: engram-moe-1b)...")

    # Model kwargs matching full preset
    tokenizer = AutoTokenizer.from_pretrained('openai-community/gpt2')
    model_kwargs = {
        'use_moe': True,
        'use_latent_moe': True,
        'latent_ratio': 4,
        'latent_preserve_expert_dim': True,
        'n_experts': 64,
        'n_activated': 2,
        'use_engram': True,
        'engram_layers': [2, 6],
        'engram_d_mem': 512,
        'engram_n_hash_heads': 8,
        'engram_ngram_orders': [2, 3],
        'engram_conv_kernel': 4,
        'use_residual_scalars': True,
        'yarn_enabled': True,
        'yarn_scale_factor': 1.0,
        'yarn_original_max_seq_len': 2048,
        'fope_enabled': True,
        'fope_n_harmonics': 4,
        'fope_floor_ratio': 0.1,
        'fope_coef_init_std': 0.3,
        'use_gradient_checkpointing': True,
    }
    use_te_fp8 = TE_AVAILABLE
    model = create_swa_mla_model(
        'engram-moe-1b',
        vocab_size=len(tokenizer),
        block_size=2048,
        use_te_fp8=use_te_fp8,
        **model_kwargs,
    )
    model = model.to(device)

    # Set Engram tokenizer compression
    from engram import TokenizerCompression
    for block in model.transformer.h:
        if hasattr(block, 'engram') and block.engram is not None:
            block.engram.set_tokenizer_compression(
                TokenizerCompression.identity(len(tokenizer))
            )

    # No torch.compile — profiling eager mode for clean absolute timings
    # model = torch.compile(model, mode='default')

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {total_params/1e6:.0f}M params, TE FP8: {use_te_fp8}")

    # Count trainable tensors for overhead estimation
    n_trainable = sum(1 for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameter tensors: {n_trainable}")

    # Optimizers (Muon + friends, matching full preset)
    raw_model = model._orig_mod if hasattr(model, '_orig_mod') else model
    no_wd_keywords = ['.bias', 'norm', 'ln_', 'wte', 'wpe', 'lambdas', 'engram']

    if hasattr(torch.optim, 'Muon'):
        muon_params, adamw_params, engram_embed_params = [], [], []
        x0_params, resid_params = [], []
        for name, param in raw_model.named_parameters():
            if not param.requires_grad:
                continue
            if 'x0_lambdas' in name:
                x0_params.append(param)
            elif 'resid_lambdas' in name:
                resid_params.append(param)
            elif 'engram' in name and 'embeddings' in name and 'tables' in name:
                engram_embed_params.append(param)
            elif any(nd in name for nd in ['wte', 'wpe', 'lm_head', 'embed']):
                adamw_params.append(param)
            elif param.ndim == 2:
                muon_params.append(param)
            else:
                adamw_params.append(param)

        optimizers = []
        if muon_params:
            optimizers.append(torch.optim.Muon(muon_params, lr=0.02, momentum=0.95, weight_decay=0.1, nesterov=True, ns_steps=5))
        # Merge all AdamW groups into single optimizer
        adamw_groups = []
        if adamw_params:
            adamw_groups.append({'params': adamw_params, 'lr': 1e-4, 'weight_decay': 0.1, 'betas': (0.9, 0.95)})
        if engram_embed_params:
            adamw_groups.append({'params': engram_embed_params, 'lr': 5e-4, 'weight_decay': 0.0, 'betas': (0.9, 0.95)})
        if x0_params:
            adamw_groups.append({'params': x0_params, 'lr': 0.5, 'weight_decay': 0.0, 'betas': (0.96, 0.95)})
        if resid_params:
            adamw_groups.append({'params': resid_params, 'lr': 0.005, 'weight_decay': 0.0, 'betas': (0.9, 0.95)})
        if adamw_groups:
            optimizers.append(torch.optim.AdamW(adamw_groups, lr=1e-4, betas=(0.9, 0.95), fused=True))
        print(f"Optimizers: {len(optimizers)} (Muon + merged AdamW)")
    else:
        optimizers = [torch.optim.AdamW(model.parameters(), lr=1e-4, fused=True)]
        print("Optimizers: 1 (AdamW only, Muon not available)")

    # EMA
    ema = EMAModel(raw_model, decay=0.9999) if EMA_AVAILABLE else None
    print(f"EMA: {'enabled' if ema else 'disabled'}")

    # Cautious WD params
    cautious_wd_params = [
        param for name, param in raw_model.named_parameters()
        if param.requires_grad and not any(nd in name for nd in no_wd_keywords)
    ]
    print(f"Cautious WD params: {len(cautious_wd_params)} tensors")

    # GradScaler
    scaler = torch.amp.GradScaler('cuda', enabled=use_te_fp8)

    # FP8 recipe
    fp8_recipe = None
    if use_te_fp8:
        fp8_recipe = DelayedScaling(fp8_format=Format.HYBRID, amax_history_len=16, amax_compute_algo="max")

    # Fake batch
    B, T = 6, 2048
    fake_input = torch.randint(0, len(tokenizer), (B, T), device=device)
    fake_labels = torch.randint(0, len(tokenizer), (B, T), device=device)

    print(f"\nBatch: {B}x{T} = {B*T:,} tokens")
    print(f"Warmup: 8 steps, Profile: 10 steps\n")

    # Warmup (let torch.compile JIT fully — needs 5+ steps for all code paths)
    print("Warming up (torch.compile JIT)...")
    for i in range(8):
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)
        fp8_ctx = te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe) if use_te_fp8 else nullcontext()
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            with fp8_ctx:
                _, loss = model(fake_input, targets=fake_labels)
                if hasattr(raw_model, 'get_moe_aux_loss'):
                    loss = loss + raw_model.get_moe_aux_loss()
        scaler.scale(loss).backward()
        for opt in optimizers:
            scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, foreach=True)
        for opt in optimizers:
            scaler.step(opt)
        scaler.update()
        if ema:
            ema.update(raw_model)
        print(f"  warmup step {i+1}/8 done")

    torch.cuda.synchronize()
    print("Warmup complete.\n")

    # Profile
    N_STEPS = 10
    timings = {
        'zero_grad': [],
        'forward': [],
        'backward': [],
        'unscale': [],
        'grad_clip': [],
        'cautious_wd': [],
        'optimizer_step': [],
        'scaler_update': [],
        'ema_update': [],
        'moe_bias_update': [],
        'total_step': [],
    }

    for step in range(N_STEPS):
        events = {}
        for key in timings:
            events[key] = cuda_event_timer()

        torch.cuda.synchronize()

        # Total step start
        events['total_step'][0].record()

        # Zero grad
        events['zero_grad'][0].record()
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)
        events['zero_grad'][1].record()

        # Forward
        events['forward'][0].record()
        fp8_ctx = te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe) if use_te_fp8 else nullcontext()
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            with fp8_ctx:
                _, loss = model(fake_input, targets=fake_labels)
                if hasattr(raw_model, 'get_moe_aux_loss'):
                    loss = loss + raw_model.get_moe_aux_loss()
        events['forward'][1].record()

        # Backward
        events['backward'][0].record()
        scaler.scale(loss).backward()
        events['backward'][1].record()

        # MoE bias update
        events['moe_bias_update'][0].record()
        if hasattr(raw_model, 'update_moe_bias'):
            raw_model.update_moe_bias()
        events['moe_bias_update'][1].record()

        # Unscale
        events['unscale'][0].record()
        for opt in optimizers:
            scaler.unscale_(opt)
        events['unscale'][1].record()

        # Grad clip
        events['grad_clip'][0].record()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, foreach=True)
        events['grad_clip'][1].record()

        # Cautious WD (per-param loop, 1 temp tensor at a time)
        events['cautious_wd'][0].record()
        wd_factor = 1e-4 * 0.1
        with torch.no_grad():
            for p in cautious_wd_params:
                if p.grad is not None:
                    mask = (p.grad * p.data > 0).to(p.data.dtype)
                    p.data.mul_(1.0 - wd_factor * mask)
        events['cautious_wd'][1].record()

        # Optimizer step
        events['optimizer_step'][0].record()
        for opt in optimizers:
            scaler.step(opt)
        events['optimizer_step'][1].record()

        # Scaler update
        events['scaler_update'][0].record()
        scaler.update()
        events['scaler_update'][1].record()

        # EMA update
        events['ema_update'][0].record()
        if ema:
            ema.update(raw_model)
        events['ema_update'][1].record()

        # Total step end
        events['total_step'][1].record()

        torch.cuda.synchronize()

        # Collect timings
        for key in timings:
            ms = events[key][0].elapsed_time(events[key][1])
            timings[key].append(ms)

    # Report
    print("=" * 70)
    print(f"  TRAINING LOOP PHASE BREAKDOWN (avg over {N_STEPS} steps)")
    print(f"  Batch: {B}x{T}, Full preset, {'TE FP8' if use_te_fp8 else 'BF16'}")
    print("=" * 70)

    total_avg = sum(t for t in [sum(v)/len(v) for k, v in timings.items() if k != 'total_step'])

    for key in ['zero_grad', 'forward', 'backward', 'moe_bias_update', 'unscale',
                'grad_clip', 'cautious_wd', 'optimizer_step', 'scaler_update', 'ema_update']:
        vals = timings[key]
        avg = sum(vals) / len(vals)
        pct = avg / (sum(timings['total_step']) / len(timings['total_step'])) * 100
        bar = "█" * int(pct / 2) + "░" * (50 - int(pct / 2))
        print(f"  {key:<20s} {avg:8.2f} ms  {pct:5.1f}%  {bar}")

    total_step_avg = sum(timings['total_step']) / len(timings['total_step'])
    phases_sum = total_avg
    overhead = total_step_avg - phases_sum
    tokens_per_sec = B * T / (total_step_avg / 1000)

    print("-" * 70)
    print(f"  {'Sum of phases':<20s} {phases_sum:8.2f} ms")
    print(f"  {'Python overhead':<20s} {overhead:8.2f} ms  {overhead/total_step_avg*100:5.1f}%")
    print(f"  {'Total step':<20s} {total_step_avg:8.2f} ms")
    print(f"  {'Tokens/sec':<20s} {tokens_per_sec:8.0f}")
    print("=" * 70)

    # Highlight bottlenecks
    print("\nBOTTLENECK ANALYSIS:")
    non_compute = 0
    for key in ['cautious_wd', 'ema_update', 'optimizer_step', 'unscale', 'scaler_update', 'moe_bias_update']:
        avg = sum(timings[key]) / len(timings[key])
        non_compute += avg
    compute = sum(timings['forward'])/len(timings['forward']) + sum(timings['backward'])/len(timings['backward'])

    print(f"  Compute (fwd+bwd):        {compute:.1f} ms ({compute/total_step_avg*100:.0f}%)")
    print(f"  Non-compute overhead:     {non_compute:.1f} ms ({non_compute/total_step_avg*100:.0f}%)")
    print(f"  GPU utilization estimate: {compute/total_step_avg*100:.0f}%")


if __name__ == '__main__':
    profile_training_loop()
