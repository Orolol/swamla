"""
Benchmark script to compare training speed between PackedFinewebDataset, FastFinewebDataset,
and fast_loader (external package).
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

# Try to import external fast_loader
try:
    from fast_loader import FastFinewebDataset as ExternalFastLoader
    EXTERNAL_FAST_LOADER_AVAILABLE = True
except ImportError:
    EXTERNAL_FAST_LOADER_AVAILABLE = False
    print("Note: fast_loader not installed. Install with: pip install git+https://github.com/Orolol/data-loader-fast.git")


def create_model_and_optimizer(size, vocab_size, block_size, device, use_flash_attention=False, use_varlen_masking=False, use_gated_deltanet=False):
    """Create a fresh model and optimizer.

    Args:
        use_flash_attention: If True, use FlashCausalSelfAttention (supports sliding window natively)
        use_varlen_masking: If True, use varlen to skip padded tokens (slower with dynamic shapes)
        use_gated_deltanet: If True, use GatedDeltaNet (linear attention O(n)) instead of SWA
    """
    model = create_swa_mla_model(
        size=size,
        vocab_size=vocab_size,
        block_size=block_size,
        use_flash_attention=use_flash_attention,
        use_varlen_masking=use_varlen_masking,
        use_gated_deltanet=use_gated_deltanet,
    )
    model = model.to(device)
    model = torch.compile(model, mode='reduce-overhead')
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=True)
    return model, optimizer


def benchmark_dataloader(
    dataloader_class,
    dataloader_name: str,
    size: str,
    vocab_size: int,
    device,
    num_steps: int = 100,
    batch_size: int = 4,
    block_size: int = 512,
    tokenizer=None,
    use_flash_attention: bool = False,
    use_gated_deltanet: bool = False,
):
    """Run benchmark for a single dataloader."""
    print(f"\n{'='*60}")
    print(f"Benchmarking: {dataloader_name}")
    print(f"{'='*60}")

    # Create dataloader
    print(f"Creating {dataloader_name}...")
    
    
    dataset = dataloader_class(
            split="train",
            max_length=block_size,
            batch_size=batch_size,
            buffer_docs=10240,
            prefetch_batches=512,
            shuffle=True,
            shuffle_buffer_size=5000,
            tokenizer=tokenizer,
            num_workers=1,
            start_offset=0,
        )

    # if dataloader_class == PackedFinewebDataset:
    #     dataset = dataloader_class(
    #         split="train",
    #         max_length=block_size,
    #         batch_size=batch_size,
    #         buffer_docs=10240,
    #         prefetch_batches=512,
    #         shuffle=True,
    #         shuffle_buffer_size=5000,
    #         tokenizer=tokenizer,
    #         num_workers=1,
    #         start_offset=0,
    #     )
    # elif dataloader_class == FastFinewebDataset:
    #     dataset = dataloader_class(
    #         split="train",
    #         max_length=block_size,
    #         batch_size=batch_size,
    #         buffer_docs=10240,
    #         prefetch_batches=512,
    #         shuffle=True,
    #         tokenizer=tokenizer,
    #         num_workers=1,
    #         start_offset=0,
    #     )
    # elif EXTERNAL_FAST_LOADER_AVAILABLE and dataloader_class == ExternalFastLoader:
    #     # External fast_loader from github.com/Orolol/data-loader-fast
    #     dataset = dataloader_class(
    #         split="train",
    #         max_length=block_size,
    #         batch_size=batch_size,
    #         tokenizer=tokenizer,
    #     )
    # else:
    #     raise ValueError(f"Unknown dataloader class: {dataloader_class}")

    # Create fresh model and optimizer for this benchmark
    if use_gated_deltanet:
        attn_type = "GatedDeltaNet"
    elif use_flash_attention:
        attn_type = "FlashAttention"
    else:
        attn_type = "SDPA"
    print(f"Creating fresh model ({size}) with {attn_type}...")
    model, optimizer = create_model_and_optimizer(
        size, vocab_size, block_size, device,
        use_flash_attention=use_flash_attention,
        use_gated_deltanet=use_gated_deltanet
    )

    # Warmup: wait for prefetch buffer to fill
    print("Waiting for prefetch buffer to fill...")
    time.sleep(3)

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

        # Mark step begin for CUDAGraphs compatibility with torch.compile
        torch.compiler.cudagraph_mark_step_begin()

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

        # Mark step begin for CUDAGraphs compatibility with torch.compile
        torch.compiler.cudagraph_mark_step_begin()

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

    # Close dataset (if method exists)
    if hasattr(dataset, 'close'):
        dataset.close()

    # Calculate stats
    avg_step_time = sum(step_times) / len(step_times)
    avg_data_time = sum(data_times) / len(data_times)
    avg_forward_time = sum(forward_times) / len(forward_times)
    avg_backward_time = sum(backward_times) / len(backward_times)
    tokens_per_sec = total_tokens / total_time
    avg_loss = total_loss / num_steps

    # Get padding stats (if method exists)
    if hasattr(dataset, 'get_stats'):
        stats = dataset.get_stats()
        padding_ratio = stats.get('avg_padding_ratio', stats.get('total_padding', 0) / max(stats.get('total_tokens', 1), 1))
    else:
        padding_ratio = 0.0  # Unknown for external loader

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

    # Benchmark configurations:
    # 1. Packed + SDPA (standard PyTorch attention, compiled)
    # 2. Fast + FlashAttention (FlashCausalSelfAttention with native sliding window)
    results = []

    # # 1. Packed dataloader + SDPA
    # results.append(benchmark_dataloader(
    #     PackedFinewebDataset,
    #     "Packed + SDPA",
    #     size=args.size,
    #     vocab_size=vocab_size,
    #     device=device,
    #     num_steps=args.num_steps,
    #     batch_size=args.batch_size,
    #     block_size=args.block_size,
    #     tokenizer=tokenizer,
    #     use_flash_attention=False,
    # ))

    # # Clear GPU memory between tests
    # torch.cuda.empty_cache()
    # time.sleep(2)

    # 2. Packed dataloader + FlashAttention
    results.append(benchmark_dataloader(
        PackedFinewebDataset,
        "SWA + FlashAttn",
        size=args.size,
        vocab_size=vocab_size,
        device=device,
        num_steps=args.num_steps,
        batch_size=args.batch_size,
        block_size=args.block_size,
        tokenizer=tokenizer,
        use_flash_attention=True,  # Uses FlashCausalSelfAttention with native window_size
    ))

    # # Clear GPU memory between tests
    # torch.cuda.empty_cache()
    # time.sleep(2)

    # # 3. Packed dataloader + GatedDeltaNet (linear O(n) attention)
    # results.append(benchmark_dataloader(
    #     PackedFinewebDataset,
    #     "SPDA + DeltaNet",
    #     size=args.size,
    #     vocab_size=vocab_size,
    #     device=device,
    #     num_steps=args.num_steps,
    #     batch_size=args.batch_size,
    #     block_size=args.block_size,
    #     tokenizer=tokenizer,
    #     use_flash_attention=False,
    #     use_gated_deltanet=True,  # Uses GatedDeltaNet linear attention
    # ))
    
    # torch.cuda.empty_cache()
    # time.sleep(2)

    # 4. Flash + GatedDeltaNet (linear O(n) attention)
    results.append(benchmark_dataloader(
        PackedFinewebDataset,
        "Flash + DeltaNet",
        size=args.size,
        vocab_size=vocab_size,
        device=device,
        num_steps=args.num_steps,
        batch_size=args.batch_size,
        block_size=args.block_size,
        tokenizer=tokenizer,
        use_flash_attention=True,
        use_gated_deltanet=True,  # Uses GatedDeltaNet linear attention
    ))


    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    # Dynamic column headers based on results
    headers = [r['name'] for r in results]
    col_width = 18

    # Header row
    print(f"\n{'Metric':<22}", end="")
    for h in headers:
        # Shorten names for display
        short_name = h.replace("FinewebDataset", "").replace("Dataset", "")
        print(f"{short_name:>{col_width}}", end="")
    print()
    print("-" * (22 + col_width * len(results)))

    # Tokens/sec
    print(f"{'Tokens/sec':<22}", end="")
    for r in results:
        print(f"{r['tokens_per_sec']:>{col_width},.0f}", end="")
    print()

    # Avg step time
    print(f"{'Avg step time (ms)':<22}", end="")
    for r in results:
        print(f"{r['avg_step_time_ms']:>{col_width}.2f}", end="")
    print()

    # Avg data time
    print(f"{'Avg data time (ms)':<22}", end="")
    for r in results:
        print(f"{r['avg_data_time_ms']:>{col_width}.2f}", end="")
    print()

    # Padding ratio
    print(f"{'Padding ratio':<22}", end="")
    for r in results:
        print(f"{r['padding_ratio']*100:>{col_width-1}.2f}%", end="")
    print()

    # Avg loss
    print(f"{'Avg loss':<22}", end="")
    for r in results:
        print(f"{r['avg_loss']:>{col_width}.4f}", end="")
    print()

    # Find winner
    print(f"\n{'='*60}")
    best = max(results, key=lambda x: x['tokens_per_sec'])
    worst = min(results, key=lambda x: x['tokens_per_sec'])
    speed_diff = (best['tokens_per_sec'] - worst['tokens_per_sec']) / worst['tokens_per_sec'] * 100
    print(f"Winner: {best['name']} ({speed_diff:.1f}% faster than {worst['name']})")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
