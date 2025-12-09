"""
Analyze batch content from different dataloaders to understand token distribution.
"""

import os
import sys
import time
from pathlib import Path
from collections import defaultdict

import torch

# Add models and data to path
sys.path.insert(0, str(Path(__file__).parent / 'models'))
sys.path.insert(0, str(Path(__file__).parent / 'data'))

from data_loader_packed import PackedFinewebDataset
from data_loader_fast import FastFinewebDataset
from transformers import AutoTokenizer

# Try to import external fast_loader
try:
    from fast_loader import FastFinewebDataset as ExternalFastLoader
    EXTERNAL_FAST_LOADER_AVAILABLE = True
except ImportError:
    EXTERNAL_FAST_LOADER_AVAILABLE = False
    print("Note: fast_loader not installed")


def analyze_batch(batch, batch_idx):
    """Analyze a single batch and return statistics."""
    input_ids = batch["input_ids"]
    labels = batch["labels"]

    batch_size, seq_len = input_ids.shape

    # Count non-padding tokens per sequence
    # Labels == -100 means padding (ignored in loss)
    non_pad_per_seq = (labels != -100).sum(dim=1).tolist()

    # Also check input_ids padding (usually pad_token_id)
    # We can infer pad_id from the most common token at the end
    pad_id = input_ids[0, -1].item() if (labels[0, -1] == -100) else None

    stats = {
        'batch_idx': batch_idx,
        'batch_size': batch_size,
        'seq_len': seq_len,
        'total_positions': batch_size * seq_len,
        'non_pad_tokens': non_pad_per_seq,
        'total_non_pad': sum(non_pad_per_seq),
        'min_seq_tokens': min(non_pad_per_seq),
        'max_seq_tokens': max(non_pad_per_seq),
        'mean_seq_tokens': sum(non_pad_per_seq) / len(non_pad_per_seq),
        'padding_ratio': 1 - (sum(non_pad_per_seq) / (batch_size * seq_len)),
        'pad_id': pad_id,
    }

    return stats


def print_batch_analysis(stats):
    """Pretty print batch analysis."""
    print(f"\n  Batch {stats['batch_idx']}:")
    print(f"    Shape: ({stats['batch_size']}, {stats['seq_len']})")
    print(f"    Total positions: {stats['total_positions']:,}")
    print(f"    Non-pad tokens: {stats['total_non_pad']:,} ({100*(1-stats['padding_ratio']):.1f}%)")
    print(f"    Padding ratio: {stats['padding_ratio']*100:.1f}%")
    print(f"    Tokens per sequence: min={stats['min_seq_tokens']}, max={stats['max_seq_tokens']}, mean={stats['mean_seq_tokens']:.1f}")
    print(f"    Per-sequence breakdown: {stats['non_pad_tokens']}")


def analyze_dataloader(name, dataset, num_batches=20):
    """Analyze multiple batches from a dataloader."""
    print(f"\n{'='*70}")
    print(f"Analyzing: {name}")
    print(f"{'='*70}")

    # Wait for prefetch
    print("Waiting for prefetch buffer...")
    time.sleep(2)

    data_iter = iter(dataset)
    all_stats = []
    all_seq_lengths = []  # Track all sequence lengths across batches

    print(f"\nAnalyzing {num_batches} batches...")

    for i in range(num_batches):
        try:
            batch = next(data_iter)
            stats = analyze_batch(batch, i)
            all_stats.append(stats)
            all_seq_lengths.extend(stats['non_pad_tokens'])

            # Print first 5 batches in detail
            if i < 5:
                print_batch_analysis(stats)
        except StopIteration:
            print(f"  Dataset exhausted after {i} batches")
            break

    # Close dataset
    if hasattr(dataset, 'close'):
        dataset.close()

    # Summary statistics
    print(f"\n  {'='*60}")
    print(f"  SUMMARY for {name}")
    print(f"  {'='*60}")

    if all_stats:
        total_tokens = sum(s['total_non_pad'] for s in all_stats)
        total_positions = sum(s['total_positions'] for s in all_stats)
        avg_padding = sum(s['padding_ratio'] for s in all_stats) / len(all_stats)

        # Check if sequences are sorted by length within batches
        sorted_batches = 0
        for s in all_stats:
            tokens = s['non_pad_tokens']
            if tokens == sorted(tokens) or tokens == sorted(tokens, reverse=True):
                sorted_batches += 1

        # Check variance within batches (low = good sorting)
        within_batch_variances = []
        for s in all_stats:
            tokens = s['non_pad_tokens']
            mean = sum(tokens) / len(tokens)
            variance = sum((t - mean) ** 2 for t in tokens) / len(tokens)
            within_batch_variances.append(variance)
        avg_within_variance = sum(within_batch_variances) / len(within_batch_variances)

        # Check variance across batches
        batch_means = [s['mean_seq_tokens'] for s in all_stats]
        overall_mean = sum(batch_means) / len(batch_means)
        across_batch_variance = sum((m - overall_mean) ** 2 for m in batch_means) / len(batch_means)

        print(f"  Total batches analyzed: {len(all_stats)}")
        print(f"  Total tokens (non-pad): {total_tokens:,}")
        print(f"  Total positions: {total_positions:,}")
        print(f"  Average padding ratio: {avg_padding*100:.2f}%")
        print(f"  Sorted batches: {sorted_batches}/{len(all_stats)}")
        print(f"  Avg within-batch variance: {avg_within_variance:.1f}")
        print(f"  Across-batch variance: {across_batch_variance:.1f}")

        # Sequence length distribution
        print(f"\n  Sequence length distribution:")
        length_buckets = defaultdict(int)
        for l in all_seq_lengths:
            bucket = (l // 256) * 256  # 256-token buckets
            length_buckets[bucket] += 1

        for bucket in sorted(length_buckets.keys()):
            count = length_buckets[bucket]
            pct = count / len(all_seq_lengths) * 100
            bar = '█' * int(pct / 2)
            print(f"    {bucket:4d}-{bucket+255:4d}: {count:4d} ({pct:5.1f}%) {bar}")

        # Show batch means progression (to see if batches are ordered by length)
        print(f"\n  Batch mean tokens progression (first 10):")
        for i, s in enumerate(all_stats[:10]):
            bar_len = int(s['mean_seq_tokens'] / 50)
            bar = '█' * bar_len
            print(f"    Batch {i:2d}: {s['mean_seq_tokens']:6.1f} tokens {bar}")

    return all_stats


def main():
    print("\n" + "#" * 70)
    print("# Batch Content Analysis")
    print("#" * 70)

    # Configuration
    batch_size = 4
    block_size = 2048
    num_batches = 20

    print(f"\nConfiguration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Block size (max_length): {block_size}")
    print(f"  Batches to analyze: {num_batches}")

    # Load tokenizer
    print("\nLoading tokenizer...")
    access_token = os.getenv("HF_TOKEN")
    tokenizer = AutoTokenizer.from_pretrained(
        "meta-llama/Llama-3.2-1B-Instruct",
        use_fast=True,
        access_token=access_token
    )
    tokenizer.pad_token = tokenizer.eos_token

    results = {}

    # 1. PackedFinewebDataset
    print("\n" + "=" * 70)
    print("Creating PackedFinewebDataset...")
    packed_dataset = PackedFinewebDataset(
        split="train",
        max_length=block_size,
        batch_size=batch_size,
        buffer_docs=4096,
        prefetch_batches=32,
        shuffle=True,
        shuffle_buffer_size=5000,
        tokenizer=tokenizer,
        num_workers=1,
        start_offset=0,
    )
    results['Packed'] = analyze_dataloader("PackedFinewebDataset", packed_dataset, num_batches)

    # 2. FastFinewebDataset (local)
    print("\n" + "=" * 70)
    print("Creating FastFinewebDataset (local)...")
    fast_dataset = FastFinewebDataset(
        split="train",
        max_length=block_size,
        batch_size=batch_size,
        buffer_docs=4096,
        prefetch_batches=32,
        shuffle=True,
        tokenizer=tokenizer,
        num_workers=1,
        start_offset=0,
        fixed_length=True,
    )
    results['Fast'] = analyze_dataloader("FastFinewebDataset", fast_dataset, num_batches)

    # 3. External FastLoader (if available)
    if EXTERNAL_FAST_LOADER_AVAILABLE:
        print("\n" + "=" * 70)
        print("Creating ExternalFastLoader...")
        external_dataset = ExternalFastLoader(
            split="train",
            max_length=block_size,
            batch_size=batch_size,
            tokenizer=tokenizer,
        )
        results['External'] = analyze_dataloader("ExternalFastLoader", external_dataset, num_batches)

    # Final comparison
    print("\n" + "=" * 70)
    print("FINAL COMPARISON")
    print("=" * 70)

    print(f"\n{'Loader':<20} {'Avg Padding':<15} {'Within-Var':<15} {'Across-Var':<15}")
    print("-" * 65)

    for name, stats in results.items():
        if stats:
            avg_padding = sum(s['padding_ratio'] for s in stats) / len(stats)

            within_vars = []
            for s in stats:
                tokens = s['non_pad_tokens']
                mean = sum(tokens) / len(tokens)
                var = sum((t - mean) ** 2 for t in tokens) / len(tokens)
                within_vars.append(var)
            avg_within = sum(within_vars) / len(within_vars)

            batch_means = [s['mean_seq_tokens'] for s in stats]
            overall_mean = sum(batch_means) / len(batch_means)
            across_var = sum((m - overall_mean) ** 2 for m in batch_means) / len(batch_means)

            print(f"{name:<20} {avg_padding*100:>12.2f}% {avg_within:>14.1f} {across_var:>14.1f}")

    print("\n" + "=" * 70)
    print("Interpretation:")
    print("  - Low within-batch variance = sequences in same batch have similar lengths")
    print("  - High across-batch variance = batches contain different length ranges")
    print("  - Ideal for training: Low within-batch, High across-batch variance")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
