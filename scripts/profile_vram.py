#!/usr/bin/env python3
"""
VRAM Profiling Script for Progressive Training

Empirically measures VRAM usage for different training configurations.
More accurate than theoretical calculations (typically 20-40% off).

Usage:
    # Test a specific config
    python scripts/profile_vram.py --model-size small --batch-size 8 --seq-len 2048

    # Find optimal batch size
    python scripts/profile_vram.py --model-size moe-1b --seq-len 2048 --search-optimal

    # Generate progressive schedule
    python scripts/profile_vram.py --model-size mup-1b --seq-len 2048 --generate-schedule

    # With gradient checkpointing
    python scripts/profile_vram.py --model-size large --search-optimal --gradient-checkpointing
"""

import argparse
import gc
import json
import os
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, List, Tuple

# Add parent directory to path for imports (same pattern as other scripts)
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "models"))  # For relative imports in models/

import torch

from models.swa_mla_model import create_swa_mla_model


# ============================================================================
# GPU Detection
# ============================================================================

@dataclass
class GPUInfo:
    """GPU information container."""
    name: str
    total_vram_gb: float
    compute_capability: str
    supports_fp8: bool
    supports_tf32: bool
    cuda_version: str


def get_gpu_info() -> GPUInfo:
    """Detect GPU and its capabilities."""
    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA GPU available")

    props = torch.cuda.get_device_properties(0)
    cuda_version = torch.version.cuda or "unknown"

    return GPUInfo(
        name=props.name,
        total_vram_gb=props.total_memory / (1024**3),
        compute_capability=f"{props.major}.{props.minor}",
        supports_fp8=props.major >= 9,  # Hopper+
        supports_tf32=props.major >= 8,  # Ampere+
        cuda_version=cuda_version,
    )


# ============================================================================
# Synthetic Batch Creation
# ============================================================================

def create_synthetic_batch(
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    device: str,
) -> dict:
    """Create a synthetic batch for profiling (no I/O overhead)."""
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    return {
        'input_ids': input_ids,
        'labels': input_ids.clone(),
    }


# ============================================================================
# Profiling Core
# ============================================================================

@dataclass
class ProfileResult:
    """Result of a profiling run."""
    success: bool
    peak_vram_gb: float
    time_per_step_ms: float
    error: Optional[str]
    model_params_m: float = 0.0
    tokens_per_step: int = 0


def profile_config(
    model_size: str,
    batch_size: int,
    seq_len: int,
    gradient_checkpointing: bool = False,
    use_compile: bool = False,
    use_fp8: bool = False,
    warmup_steps: int = 2,
    vocab_size: int = 50257,
    suppress_model_output: bool = True,
) -> ProfileResult:
    """
    Measure actual VRAM for a given configuration.

    Args:
        model_size: Model preset (small, base, large, xl, moe-1b, etc.)
        batch_size: Batch size to test
        seq_len: Sequence length
        gradient_checkpointing: Enable gradient checkpointing
        use_compile: Enable torch.compile
        use_fp8: Enable FP8 training
        warmup_steps: Number of warmup steps before measurement
        vocab_size: Vocabulary size
        suppress_model_output: Suppress model creation output

    Returns:
        ProfileResult with VRAM usage and timing info
    """
    # Ensure clean slate
    gc.collect()
    try:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass  # Ignore cleanup errors

    model = None
    optimizer = None
    scaler = None
    params_m = 0.0

    # Redirect stdout to suppress model creation output
    if suppress_model_output:
        import io
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()

    try:
        # 1. Create model
        model = create_swa_mla_model(
            model_size,
            vocab_size=vocab_size,
            block_size=seq_len,
            use_gradient_checkpointing=gradient_checkpointing,
        )
        model = model.cuda()

        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        params_m = total_params / 1e6

        # 2. Optional FP8 conversion
        if use_fp8:
            try:
                from optimization.fp8_native import convert_to_native_fp8
                model = convert_to_native_fp8(model)
            except ImportError:
                pass  # FP8 not available, continue without

        # 3. Optional compile
        if use_compile:
            model = torch.compile(model, mode='reduce-overhead')

        # 4. Create optimizer (takes memory for optimizer states!)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        # 5. GradScaler for FP8 or mixed precision
        scaler = torch.amp.GradScaler('cuda', enabled=use_fp8)

        # 6. Warmup phase (important for torch.compile and CUDA lazy init)
        model.train()
        for _ in range(warmup_steps):
            batch = create_synthetic_batch(batch_size, seq_len, vocab_size, 'cuda')
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                _, loss = model(batch['input_ids'], targets=batch['labels'])
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            # Clean up batch
            del batch, loss

        # 7. Reset stats for actual measurement
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

        # 8. Measure a single training step
        start = time.perf_counter()

        batch = create_synthetic_batch(batch_size, seq_len, vocab_size, 'cuda')
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            _, loss = model(batch['input_ids'], targets=batch['labels'])
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - start) * 1000

        peak_vram = torch.cuda.max_memory_allocated() / (1024**3)

        return ProfileResult(
            success=True,
            peak_vram_gb=peak_vram,
            time_per_step_ms=elapsed_ms,
            error=None,
            model_params_m=params_m,
            tokens_per_step=batch_size * seq_len,
        )

    except Exception as e:
        error_str = str(e).lower()
        error_type = type(e).__name__
        # Catch all CUDA/memory related errors
        if any(keyword in error_str for keyword in ['out of memory', 'cuda', 'oom', 'accelerator']):
            return ProfileResult(
                success=False,
                peak_vram_gb=float('inf'),
                time_per_step_ms=float('inf'),
                error='OOM',
                model_params_m=params_m,
                tokens_per_step=batch_size * seq_len,
            )
        if error_type in ['OutOfMemoryError', 'AcceleratorError']:
            return ProfileResult(
                success=False,
                peak_vram_gb=float('inf'),
                time_per_step_ms=float('inf'),
                error='OOM',
                model_params_m=params_m,
                tokens_per_step=batch_size * seq_len,
            )
        # Re-raise unexpected errors
        raise

    finally:
        # Restore stdout
        if suppress_model_output:
            sys.stdout = old_stdout

        # Cleanup - wrap in try/except to handle post-OOM state
        try:
            if model is not None:
                del model
            if optimizer is not None:
                del optimizer
            if scaler is not None:
                del scaler
            gc.collect()
            torch.cuda.empty_cache()
        except Exception:
            # Force reset CUDA on OOM
            try:
                torch.cuda.synchronize()
                gc.collect()
                torch.cuda.empty_cache()
            except Exception:
                pass


# ============================================================================
# Search Algorithms
# ============================================================================

def reset_cuda_context():
    """Attempt to reset CUDA context after OOM."""
    try:
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        # Reset memory stats
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.reset_accumulated_memory_stats()
    except Exception:
        pass


def find_max_batch_size(
    model_size: str,
    seq_len: int,
    vram_budget_gb: float,
    gradient_checkpointing: bool = False,
    use_fp8: bool = False,
    verbose: bool = True,
) -> Tuple[int, ProfileResult]:
    """
    Find maximum batch size that fits in VRAM budget using binary search.

    Returns:
        Tuple of (max_batch_size, ProfileResult for that config)
    """
    # Start with a reasonable upper bound based on VRAM budget
    # Rough heuristic: ~1GB per batch item for large models at seq_len=2048
    estimated_max = max(1, int(vram_budget_gb * 2))  # Conservative estimate
    low, high = 1, min(estimated_max, 64)  # Cap at 64 for initial search
    best_batch = 1
    best_result = None

    if verbose:
        print(f"\nSearching for optimal batch_size (seq_len={seq_len}, budget={vram_budget_gb:.1f}GB)...")

    # First, find a working batch size with linear probing from 1
    working_batch = None
    for test_batch in [1, 2, 4, 8]:
        if verbose:
            print(f"  Probe batch_size={test_batch}...", end=" ", flush=True)

        result = profile_config(
            model_size, test_batch, seq_len,
            gradient_checkpointing=gradient_checkpointing,
            use_fp8=use_fp8,
            warmup_steps=1,
        )

        if result.success and result.peak_vram_gb <= vram_budget_gb:
            if verbose:
                print(f"✓ VRAM: {result.peak_vram_gb:.2f}GB")
            working_batch = test_batch
            best_batch = test_batch
            best_result = result
        else:
            if verbose:
                print(f"✗ OOM" if not result.success else f"✗ Exceeds budget ({result.peak_vram_gb:.2f}GB)")
            reset_cuda_context()
            break

    if working_batch is None:
        if verbose:
            print("  ⚠ Could not find any working batch size")
        return 1, None

    # Now binary search upward from working batch
    # Estimate memory per item and set upper bound
    vram_per_item = best_result.peak_vram_gb / working_batch
    estimated_max_batch = int(vram_budget_gb / vram_per_item * 0.9)  # 90% safety margin
    low = working_batch + 1
    high = min(estimated_max_batch, 128)

    while low <= high:
        mid = (low + high) // 2

        if verbose:
            print(f"  Testing batch_size={mid}...", end=" ", flush=True)

        result = profile_config(
            model_size, mid, seq_len,
            gradient_checkpointing=gradient_checkpointing,
            use_fp8=use_fp8,
            warmup_steps=1,
        )

        if result.success and result.peak_vram_gb <= vram_budget_gb:
            if verbose:
                print(f"✓ VRAM: {result.peak_vram_gb:.2f}GB")
            best_batch = mid
            best_result = result
            low = mid + 1
        else:
            if verbose:
                print(f"✗ OOM" if not result.success else f"✗ Exceeds budget ({result.peak_vram_gb:.2f}GB)")
            reset_cuda_context()
            high = mid - 1

    # Re-profile with full warmup for accurate timing
    if best_result is not None:
        if verbose:
            print(f"\n  Final validation with batch_size={best_batch}...")
        best_result = profile_config(
            model_size, best_batch, seq_len,
            gradient_checkpointing=gradient_checkpointing,
            use_fp8=use_fp8,
            warmup_steps=2,
        )

    return best_batch, best_result


def profile_sequence_lengths(
    model_size: str,
    seq_lens: List[int],
    vram_budget_gb: float,
    gradient_checkpointing: bool = False,
    use_fp8: bool = False,
    verbose: bool = True,
) -> List[Tuple[int, int, ProfileResult]]:
    """
    Profile multiple sequence lengths and find optimal batch sizes.

    Returns:
        List of (seq_len, batch_size, ProfileResult)
    """
    results = []

    for seq_len in seq_lens:
        if verbose:
            print(f"\n{'='*60}")
            print(f"Profiling seq_len={seq_len}")
            print('='*60)

        batch_size, result = find_max_batch_size(
            model_size, seq_len, vram_budget_gb,
            gradient_checkpointing=gradient_checkpointing,
            use_fp8=use_fp8,
            verbose=verbose,
        )

        if result is not None:
            results.append((seq_len, batch_size, result))
        else:
            if verbose:
                print(f"  ⚠ Could not find valid config for seq_len={seq_len}")

    return results


# ============================================================================
# Progressive Schedule Generation
# ============================================================================

def generate_progressive_schedule(
    model_size: str,
    target_seq_len: int,
    vram_budget_gb: float,
    gradient_checkpointing: bool = False,
    use_fp8: bool = False,
    verbose: bool = True,
) -> Tuple[str, List[dict]]:
    """
    Generate a progressive training schedule.

    Returns:
        Tuple of (schedule_string, detailed_phases)
    """
    # Standard sequence length progression
    all_seq_lens = [256, 512, 1024, 2048, 4096, 8192]
    seq_lens = [s for s in all_seq_lens if s <= target_seq_len]

    if not seq_lens:
        seq_lens = [target_seq_len]

    # Profile each sequence length
    results = profile_sequence_lengths(
        model_size, seq_lens, vram_budget_gb,
        gradient_checkpointing=gradient_checkpointing,
        use_fp8=use_fp8,
        verbose=verbose,
    )

    if not results:
        return "", []

    # Build schedule based on constant tokens per batch
    # Use the final config's tokens_per_batch as reference
    final_tokens = results[-1][2].tokens_per_step

    phases = []
    schedule_parts = []

    for i, (seq_len, batch_size, result) in enumerate(results):
        phase = {
            'seq_len': seq_len,
            'batch_size': batch_size,
            'peak_vram_gb': result.peak_vram_gb,
            'time_per_step_ms': result.time_per_step_ms,
            'tokens_per_step': result.tokens_per_step,
        }

        if i == len(results) - 1:
            phase['tokens_until'] = float('inf')
            schedule_parts.append(f"{seq_len}:inf")
        else:
            # Progressive transition points
            # Each phase trains for increasing amounts
            phase_tokens = int(500e6 * (2 ** i))  # 500M, 1B, 2B, 4B, etc.
            phase['tokens_until'] = phase_tokens

            if phase_tokens >= 1e9:
                schedule_parts.append(f"{seq_len}:{phase_tokens / 1e9:.0f}B")
            else:
                schedule_parts.append(f"{seq_len}:{phase_tokens / 1e6:.0f}M")

        phases.append(phase)

    schedule_str = ",".join(schedule_parts)

    return schedule_str, phases


# ============================================================================
# Output Formatting
# ============================================================================

def print_table(results: List[Tuple[int, int, ProfileResult]], gpu_info: GPUInfo):
    """Print results as a formatted table."""
    print("\n" + "="*75)
    print(f"VRAM Profile Results - {gpu_info.name}")
    print("="*75)

    # Header
    print(f"{'seq_len':>10} | {'batch_size':>10} | {'peak_vram_gb':>13} | {'time_ms':>10} | {'tokens/step':>12}")
    print("-"*75)

    for seq_len, batch_size, result in results:
        if result.success:
            print(f"{seq_len:>10} | {batch_size:>10} | {result.peak_vram_gb:>13.2f} | "
                  f"{result.time_per_step_ms:>10.1f} | {result.tokens_per_step:>12,}")
        else:
            print(f"{seq_len:>10} | {batch_size:>10} | {'OOM':>13} | {'N/A':>10} | {'-':>12}")

    print("="*75)


def print_schedule(schedule_str: str, phases: List[dict]):
    """Print progressive schedule details."""
    print("\n" + "="*75)
    print("Progressive Training Schedule")
    print("="*75)
    print(f"\nSchedule string: {schedule_str}")
    print("\nPhase details:")

    for i, phase in enumerate(phases):
        tokens_str = f"{phase['tokens_until']/1e9:.1f}B" if phase['tokens_until'] < float('inf') else "∞"
        print(f"\n  Phase {i+1}: seq_len={phase['seq_len']}")
        print(f"    - batch_size: {phase['batch_size']}")
        print(f"    - tokens_per_step: {phase['tokens_per_step']:,}")
        print(f"    - peak_vram: {phase['peak_vram_gb']:.2f}GB")
        print(f"    - time_per_step: {phase['time_per_step_ms']:.1f}ms")
        print(f"    - train_until: {tokens_str} tokens")


# ============================================================================
# Main CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Profile VRAM usage for training configurations',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test specific config
  python scripts/profile_vram.py --model-size small --batch-size 8 --seq-len 2048

  # Find optimal batch size
  python scripts/profile_vram.py --model-size moe-1b --seq-len 2048 --search-optimal

  # Generate progressive schedule
  python scripts/profile_vram.py --model-size mup-1b --seq-len 4096 --generate-schedule

  # With gradient checkpointing
  python scripts/profile_vram.py --model-size large --search-optimal --gradient-checkpointing
        """
    )

    parser.add_argument('--model-size', type=str, default='small',
                        help='Model size preset (small, base, large, xl, moe-1b, moe-2b, engram-moe-1b)')
    parser.add_argument('--batch-size', type=int,
                        help='Test specific batch size')
    parser.add_argument('--seq-len', type=int, default=2048,
                        help='Sequence length (default: 2048)')
    parser.add_argument('--vram-budget', type=float,
                        help='VRAM budget in GB (default: 90%% of available)')
    parser.add_argument('--gradient-checkpointing', action='store_true',
                        help='Enable gradient checkpointing')
    parser.add_argument('--use-fp8', action='store_true',
                        help='Enable FP8 training (H100/H200 only)')
    parser.add_argument('--use-compile', action='store_true',
                        help='Enable torch.compile')
    parser.add_argument('--search-optimal', action='store_true',
                        help='Search for optimal batch size')
    parser.add_argument('--generate-schedule', action='store_true',
                        help='Generate progressive training schedule')
    parser.add_argument('--output', type=str,
                        help='Output JSON file for results')
    parser.add_argument('--quiet', action='store_true',
                        help='Minimal output')

    args = parser.parse_args()

    # Detect GPU
    try:
        gpu_info = get_gpu_info()
    except RuntimeError as e:
        print(f"Error: {e}")
        sys.exit(1)

    if not args.quiet:
        print(f"\nGPU: {gpu_info.name}")
        print(f"VRAM: {gpu_info.total_vram_gb:.1f} GB")
        print(f"Compute Capability: {gpu_info.compute_capability}")
        print(f"CUDA: {gpu_info.cuda_version}")
        print(f"FP8 Support: {'Yes' if gpu_info.supports_fp8 else 'No'}")
        print(f"TF32 Support: {'Yes' if gpu_info.supports_tf32 else 'No'}")

    # Set VRAM budget (default: 90% of available)
    vram_budget = args.vram_budget or (gpu_info.total_vram_gb * 0.90)

    if not args.quiet:
        print(f"\nVRAM Budget: {vram_budget:.1f} GB")
        print(f"Model: {args.model_size}")
        print(f"Gradient Checkpointing: {'Yes' if args.gradient_checkpointing else 'No'}")
        if args.use_fp8:
            print(f"FP8: Enabled")

    results_data = {
        'gpu': asdict(gpu_info),
        'model_size': args.model_size,
        'vram_budget_gb': vram_budget,
        'gradient_checkpointing': args.gradient_checkpointing,
        'use_fp8': args.use_fp8,
    }

    if args.batch_size:
        # Single test mode
        print(f"\nProfiling batch_size={args.batch_size}, seq_len={args.seq_len}...")

        result = profile_config(
            args.model_size,
            args.batch_size,
            args.seq_len,
            gradient_checkpointing=args.gradient_checkpointing,
            use_compile=args.use_compile,
            use_fp8=args.use_fp8,
        )

        if result.success:
            print(f"\n✓ Success")
            print(f"  Peak VRAM: {result.peak_vram_gb:.2f} GB")
            print(f"  Time per step: {result.time_per_step_ms:.1f} ms")
            print(f"  Model params: {result.model_params_m:.1f}M")
            print(f"  Tokens per step: {result.tokens_per_step:,}")
        else:
            print(f"\n✗ Failed: {result.error}")

        results_data['single_test'] = asdict(result)

    elif args.search_optimal:
        # Search for optimal batch size
        batch_size, result = find_max_batch_size(
            args.model_size,
            args.seq_len,
            vram_budget,
            gradient_checkpointing=args.gradient_checkpointing,
            use_fp8=args.use_fp8,
            verbose=not args.quiet,
        )

        if result:
            print(f"\n{'='*60}")
            print(f"Optimal Configuration")
            print(f"{'='*60}")
            print(f"  batch_size: {batch_size}")
            print(f"  seq_len: {args.seq_len}")
            print(f"  Peak VRAM: {result.peak_vram_gb:.2f} GB")
            print(f"  Time per step: {result.time_per_step_ms:.1f} ms")
            print(f"  Tokens per step: {result.tokens_per_step:,}")
            print(f"  Throughput: {result.tokens_per_step / (result.time_per_step_ms / 1000):,.0f} tokens/sec")
        else:
            print(f"\n✗ Could not find valid configuration")

        results_data['optimal'] = {
            'batch_size': batch_size,
            'seq_len': args.seq_len,
            'result': asdict(result) if result else None,
        }

    elif args.generate_schedule:
        # Generate progressive schedule
        schedule_str, phases = generate_progressive_schedule(
            args.model_size,
            args.seq_len,
            vram_budget,
            gradient_checkpointing=args.gradient_checkpointing,
            use_fp8=args.use_fp8,
            verbose=not args.quiet,
        )

        if schedule_str:
            print_schedule(schedule_str, phases)

            # Also print the config flag to use
            print(f"\n{'='*75}")
            print("Usage with train.py:")
            print(f"{'='*75}")
            print(f"\npython train.py --model-size {args.model_size} \\")
            print(f"    --progressive-schedule \"{schedule_str}\"")
            if args.gradient_checkpointing:
                print("    --gradient-checkpointing")
            if args.use_fp8:
                print("    --use-fp8")
        else:
            print("\n✗ Could not generate schedule")

        results_data['schedule'] = {
            'schedule_string': schedule_str,
            'phases': phases,
        }

    else:
        # Default: profile common sequence lengths
        seq_lens = [512, 1024, 2048, 4096]
        seq_lens = [s for s in seq_lens if s <= 8192]  # Reasonable max

        results = profile_sequence_lengths(
            args.model_size,
            seq_lens,
            vram_budget,
            gradient_checkpointing=args.gradient_checkpointing,
            use_fp8=args.use_fp8,
            verbose=not args.quiet,
        )

        if results:
            print_table(results, gpu_info)

        results_data['profiles'] = [
            {
                'seq_len': seq_len,
                'batch_size': batch_size,
                'result': asdict(result),
            }
            for seq_len, batch_size, result in results
        ]

    # Export to JSON if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results_data, f, indent=2, default=str)
        print(f"\nResults saved to: {args.output}")


if __name__ == '__main__':
    main()
