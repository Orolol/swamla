"""
Benchmark script to compare training speed between PackedFinewebDataset and FastFinewebDataset.
"""

import os
import sys
import time
import argparse
from pathlib import Path
from datetime import datetime

import torch

# Add models and data to path
sys.path.insert(0, str(Path(__file__).parent / 'models'))
sys.path.insert(0, str(Path(__file__).parent / 'data'))

from swa_mla_model import create_swa_mla_model
from data_loader_packed import PackedFinewebDataset
from data_loader_fast import FastFinewebDataset
from transformers import AutoTokenizer


def benchmark_dataloader(
    dataloader_class,
    dataloader_name: str,
    model,
    optimizer,
    device,
    num_steps: int = 100,
    batch_size: int = 4,
    block_size: int = 512,
    tokenizer=None,
):
    """Run benchmark for a single dataloader."""
    print(f"\n{'='*60}")
    print(f"Benchmarking: {dataloader_name}")
    print(f"{'='*60}")

    # Create dataloader
    print(f"Creating {dataloader_name}...")

    if dataloader_class == PackedFinewebDataset:
        dataset = dataloader_class(
            split="train",
            max_length=block_size,
            batch_size=batch_size,
            buffer_docs=1024,
            prefetch_batches=8,
            shuffle=True,
            shuffle_buffer_size=5000,
            tokenizer=tokenizer,
            num_workers=1,
            start_offset=0,
        )
    else:  # FastFinewebDataset
        dataset = dataloader_class(
            split="train",
            max_length=block_size,
            batch_size=batch_size,
            buffer_docs=1024,
            prefetch_batches=8,
            shuffle=True,
            tokenizer=tokenizer,
            num_workers=1,
            start_offset=0,
        )

    # Warmup: wait for prefetch buffer to fill
    print("Waiting for prefetch buffer to fill...")
    time.sleep(3)

    # Reset model gradients
    optimizer.zero_grad()

    # Benchmark
    total_tokens = 0
    total_loss = 0.0
    step_times = []
    data_times = []
    forward_times = []
    backward_times = []

    data_iter = iter(dataset)

    print(f"Running {num_steps} training steps...")

    # Warmup steps (not counted)
    print("Warmup (5 steps)...")
    for _ in range(5):
        batch = next(data_iter)
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            # Model returns (logits, loss) when targets are provided
            _, loss = model(input_ids, targets=labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    torch.cuda.synchronize()

    # Actual benchmark
    print("Benchmarking...")
    start_time = time.perf_counter()

    for step in range(num_steps):
        step_start = time.perf_counter()

        # Data loading
        data_start = time.perf_counter()
        batch = next(data_iter)
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        torch.cuda.synchronize()
        data_end = time.perf_counter()

        # Forward pass
        forward_start = time.perf_counter()
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            # Model returns (logits, loss) when targets are provided
            _, loss = model(input_ids, targets=labels)
        torch.cuda.synchronize()
        forward_end = time.perf_counter()

        # Backward pass
        backward_start = time.perf_counter()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        torch.cuda.synchronize()
        backward_end = time.perf_counter()

        step_end = time.perf_counter()

        # Record times
        step_times.append(step_end - step_start)
        data_times.append(data_end - data_start)
        forward_times.append(forward_end - forward_start)
        backward_times.append(backward_end - backward_start)

        # Count tokens
        tokens_in_batch = (labels != -100).sum().item()
        total_tokens += tokens_in_batch
        total_loss += loss.item()

        if (step + 1) % 20 == 0:
            avg_step_time = sum(step_times[-20:]) / 20
            tokens_per_sec = tokens_in_batch / avg_step_time
            print(f"  Step {step+1}/{num_steps}: loss={loss.item():.4f}, "
                  f"step_time={avg_step_time*1000:.1f}ms, "
                  f"tokens/s={tokens_per_sec:.0f}")

    end_time = time.perf_counter()
    total_time = end_time - start_time

    # Close dataset
    dataset.close()

    # Calculate stats
    avg_step_time = sum(step_times) / len(step_times)
    avg_data_time = sum(data_times) / len(data_times)
    avg_forward_time = sum(forward_times) / len(forward_times)
    avg_backward_time = sum(backward_times) / len(backward_times)
    tokens_per_sec = total_tokens / total_time
    avg_loss = total_loss / num_steps

    # Get padding stats
    stats = dataset.get_stats()
    padding_ratio = stats.get('avg_padding_ratio', stats.get('total_padding', 0) / max(stats.get('total_tokens', 1), 1))

    results = {
        'name': dataloader_name,
        'total_time': total_time,
        'total_tokens': total_tokens,
        'tokens_per_sec': tokens_per_sec,
        'avg_step_time_ms': avg_step_time * 1000,
        'avg_data_time_ms': avg_data_time * 1000,
        'avg_forward_time_ms': avg_forward_time * 1000,
        'avg_backward_time_ms': avg_backward_time * 1000,
        'avg_loss': avg_loss,
        'padding_ratio': padding_ratio,
    }

    print(f"\n{dataloader_name} Results:")
    print(f"  Total time: {total_time:.2f}s")
    print(f"  Total tokens: {total_tokens:,}")
    print(f"  Tokens/sec: {tokens_per_sec:,.0f}")
    print(f"  Avg step time: {avg_step_time*1000:.2f}ms")
    print(f"  Avg data time: {avg_data_time*1000:.2f}ms ({avg_data_time/avg_step_time*100:.1f}%)")
    print(f"  Avg forward time: {avg_forward_time*1000:.2f}ms ({avg_forward_time/avg_step_time*100:.1f}%)")
    print(f"  Avg backward time: {avg_backward_time*1000:.2f}ms ({avg_backward_time/avg_step_time*100:.1f}%)")
    print(f"  Avg loss: {avg_loss:.4f}")
    print(f"  Padding ratio: {padding_ratio:.2%}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Benchmark data loaders")
    parser.add_argument("--num_steps", type=int, default=100, help="Number of training steps")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--block_size", type=int, default=512, help="Sequence length")
    parser.add_argument("--size", type=str, default="small", choices=["small", "base"], help="Model size")
    args = parser.parse_args()

    print(f"\n{'#'*60}")
    print(f"# Data Loader Benchmark")
    print(f"# {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#'*60}")
    print(f"\nConfiguration:")
    print(f"  Model size: {args.size}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Block size: {args.block_size}")
    print(f"  Num steps: {args.num_steps}")

    # Check CUDA
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available!")
        return

    device = torch.device("cuda")
    print(f"  Device: {torch.cuda.get_device_name()}")

    # Load tokenizer
    print("\nLoading tokenizer...")
    access_token = os.getenv("HF_TOKEN")
    tokenizer = AutoTokenizer.from_pretrained(
        "meta-llama/Llama-3.2-1B-Instruct",
        use_fast=True,
        access_token=access_token
    )
    tokenizer.pad_token = tokenizer.eos_token
    vocab_size = len(tokenizer)

    # Create model
    print(f"\nCreating {args.size} model...")
    model = create_swa_mla_model(
        size=args.size,
        vocab_size=vocab_size,
        block_size=args.block_size,
    )
    model = model.to(device)
    model = torch.compile(model, mode='reduce-overhead')

    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {num_params/1e6:.1f}M")

    # Create optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=True)

    # Benchmark both dataloaders
    results = []

    # 1. Packed dataloader
    results.append(benchmark_dataloader(
        PackedFinewebDataset,
        "PackedFinewebDataset",
        model,
        optimizer,
        device,
        num_steps=args.num_steps,
        batch_size=args.batch_size,
        block_size=args.block_size,
        tokenizer=tokenizer,
    ))

    # Clear GPU memory between tests
    torch.cuda.empty_cache()
    time.sleep(2)

    # 2. Fast dataloader
    results.append(benchmark_dataloader(
        FastFinewebDataset,
        "FastFinewebDataset",
        model,
        optimizer,
        device,
        num_steps=args.num_steps,
        batch_size=args.batch_size,
        block_size=args.block_size,
        tokenizer=tokenizer,
    ))

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"\n{'Metric':<25} {'Packed':<20} {'Fast':<20} {'Diff':<15}")
    print("-" * 80)

    packed = results[0]
    fast = results[1]

    def fmt_diff(packed_val, fast_val, higher_better=True):
        diff = (fast_val - packed_val) / packed_val * 100
        sign = "+" if diff > 0 else ""
        better = (diff > 0) == higher_better
        indicator = "✓" if better else "✗"
        return f"{sign}{diff:.1f}% {indicator}"

    print(f"{'Tokens/sec':<25} {packed['tokens_per_sec']:>15,.0f} {fast['tokens_per_sec']:>15,.0f} {fmt_diff(packed['tokens_per_sec'], fast['tokens_per_sec'], True):>15}")
    print(f"{'Avg step time (ms)':<25} {packed['avg_step_time_ms']:>15.2f} {fast['avg_step_time_ms']:>15.2f} {fmt_diff(packed['avg_step_time_ms'], fast['avg_step_time_ms'], False):>15}")
    print(f"{'Avg data time (ms)':<25} {packed['avg_data_time_ms']:>15.2f} {fast['avg_data_time_ms']:>15.2f} {fmt_diff(packed['avg_data_time_ms'], fast['avg_data_time_ms'], False):>15}")
    print(f"{'Padding ratio':<25} {packed['padding_ratio']:>15.2%} {fast['padding_ratio']:>15.2%} {fmt_diff(packed['padding_ratio'], fast['padding_ratio'], False):>15}")
    print(f"{'Avg loss':<25} {packed['avg_loss']:>15.4f} {fast['avg_loss']:>15.4f}")

    print(f"\n{'='*60}")
    winner = "PackedFinewebDataset" if packed['tokens_per_sec'] > fast['tokens_per_sec'] else "FastFinewebDataset"
    speed_diff = abs(packed['tokens_per_sec'] - fast['tokens_per_sec']) / min(packed['tokens_per_sec'], fast['tokens_per_sec']) * 100
    print(f"Winner: {winner} ({speed_diff:.1f}% faster)")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
