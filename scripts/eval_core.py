#!/usr/bin/env python3
"""
CORE Benchmark Evaluation for DeltaNet-MLA models.

Evaluates models on the CORE metric (22 ICL tasks) from the DCLM paper.
GPT-2 baseline: 0.256525

Usage:
    # Evaluate a checkpoint
    python scripts/eval_core.py --checkpoint outputs/engram-moe/checkpoint_step_10000.pt

    # Evaluate with DDP (faster)
    torchrun --nproc_per_node=4 scripts/eval_core.py --checkpoint outputs/engram-moe/checkpoint_step_10000.pt

    # Quick evaluation (subset of examples)
    python scripts/eval_core.py --checkpoint outputs/engram-moe/checkpoint_step_10000.pt --max_per_task 100

    # Evaluate a HuggingFace model (e.g., GPT-2 for comparison)
    python scripts/eval_core.py --hf_model openai-community/gpt2

References:
    - DCLM paper: https://arxiv.org/abs/2406.11794
    - nanochat: https://github.com/karpathy/nanochat
"""

import os
import sys
import csv
import json
import yaml
import time
import random
import shutil
import zipfile
import tempfile
import argparse
from pathlib import Path
from contextlib import nullcontext
from filelock import FileLock

import torch
import torch.distributed as dist
from jinja2 import Template

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / 'models'))

from models.swa_mla_model import SWAMLAModel, SWAMLAConfig
from transformers import AutoTokenizer, AutoModelForCausalLM

# -----------------------------------------------------------------------------
# Constants

EVAL_BUNDLE_URL = "https://karpathy-public.s3.us-west-2.amazonaws.com/eval_bundle.zip"
GPT2_CORE_BASELINE = 0.256525

# -----------------------------------------------------------------------------
# Distributed setup

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


def print0(*args, **kwargs):
    """Print only on rank 0."""
    rank = dist.get_rank() if dist.is_initialized() else 0
    if rank == 0:
        print(*args, **kwargs)


# -----------------------------------------------------------------------------
# Tokenizer wrapper

class TokenizerWrapper:
    """Wrapper to provide consistent interface for tokenizers."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, texts, prepend=None):
        """Tokenize texts, optionally prepending a token."""
        if isinstance(texts, str):
            texts = [texts]
        results = []
        for text in texts:
            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            if prepend is not None:
                tokens = [prepend] + tokens
            results.append(tokens)
        return results if len(results) > 1 else results[0]

    def get_bos_token_id(self):
        return self.tokenizer.bos_token_id or self.tokenizer.eos_token_id

    def decode(self, tokens):
        return self.tokenizer.decode(tokens)


# -----------------------------------------------------------------------------
# Model wrappers

class SWAMLAWrapper:
    """Wrapper for SWAMLAModel to provide consistent interface."""

    def __init__(self, model, max_seq_len=None):
        self.model = model
        self.max_seq_len = max_seq_len or getattr(model.config, 'block_size', 2048)

    def __call__(self, input_ids):
        """Forward pass returning logits."""
        with torch.inference_mode():
            logits, _ = self.model(input_ids)
        return logits

    def get_device(self):
        return next(self.model.parameters()).device


class HFModelWrapper:
    """Wrapper for HuggingFace models."""

    def __init__(self, model, max_seq_len=None):
        self.model = model
        self.max_seq_len = max_seq_len

    def __call__(self, input_ids):
        """Forward pass returning logits."""
        with torch.inference_mode():
            outputs = self.model(input_ids)
        return outputs.logits

    def get_device(self):
        return next(self.model.parameters()).device


# -----------------------------------------------------------------------------
# Evaluation bundle management

def download_file(url: str, dest_path: str):
    """Download a file from URL."""
    import urllib.request
    print0(f"Downloading {url}...")
    urllib.request.urlretrieve(url, dest_path)
    print0(f"Downloaded to {dest_path}")


def get_eval_bundle_dir():
    """Get or create the eval bundle directory."""
    cache_dir = Path.home() / ".cache" / "swamla"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / "eval_bundle"


def ensure_eval_bundle():
    """Download and extract eval bundle if needed."""
    eval_bundle_dir = get_eval_bundle_dir()

    if eval_bundle_dir.exists():
        return eval_bundle_dir

    # Use file lock to prevent concurrent downloads
    lock_path = eval_bundle_dir.parent / "eval_bundle.lock"
    with FileLock(str(lock_path)):
        # Check again after acquiring lock
        if eval_bundle_dir.exists():
            return eval_bundle_dir

        # Download and extract
        zip_path = eval_bundle_dir.parent / "eval_bundle.zip"
        download_file(EVAL_BUNDLE_URL, str(zip_path))

        with tempfile.TemporaryDirectory() as tmpdir:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(tmpdir)
            extracted_dir = Path(tmpdir) / "eval_bundle"
            shutil.move(str(extracted_dir), str(eval_bundle_dir))

        # Clean up zip file
        zip_path.unlink()
        print0(f"Eval bundle extracted to {eval_bundle_dir}")

    return eval_bundle_dir


# -----------------------------------------------------------------------------
# Prompt rendering utilities (adapted from nanochat)

def render_prompts_mc(item, continuation_delimiter, fewshot_examples=None):
    """Render prompts for multiple choice questions."""
    template_str = """
{%- for example in fewshot_examples -%}
{{ example.query }}{{ continuation_delimiter }}{{ example.choices[example.gold] }}

{% endfor -%}
{{ item.query }}{{ continuation_delimiter }}{{ choice }}""".strip()
    template = Template(template_str)
    fewshot_examples = fewshot_examples or []
    context = {
        'fewshot_examples': fewshot_examples,
        'continuation_delimiter': continuation_delimiter,
        'item': item
    }
    prompts = [template.render(choice=choice, **context) for choice in item['choices']]
    return prompts


def render_prompts_schema(item, continuation_delimiter, fewshot_examples=None):
    """Render prompts for schema questions."""
    template_str = """
{%- for example in fewshot_examples -%}
{{ example.context_options[example.gold] }}{{ continuation_delimiter }}{{ example.continuation }}

{% endfor -%}
{{ context }}{{ continuation_delimiter }}{{ item.continuation }}""".strip()
    template = Template(template_str)
    fewshot_examples = fewshot_examples or []
    context = {
        'fewshot_examples': fewshot_examples,
        'continuation_delimiter': continuation_delimiter,
        'item': item
    }
    prompts = [template.render(context=context_option, **context)
               for context_option in item['context_options']]
    return prompts


def render_prompts_lm(item, continuation_delimiter, fewshot_examples=None):
    """Render prompts for language modeling tasks."""
    template_str = """
{%- for example in fewshot_examples -%}
{{ example.context | trim }}{{ continuation_delimiter }}{{ example.continuation }}

{% endfor -%}
{{ item.context | trim }}{{ continuation_delimiter }}{% if include_continuation %}{{ item.continuation }}{% endif %}""".strip()
    template = Template(template_str)
    fewshot_examples = fewshot_examples or []
    context = {
        'fewshot_examples': fewshot_examples,
        'continuation_delimiter': continuation_delimiter,
        'item': item
    }
    prompt_without = template.render(include_continuation=False, **context).strip()
    prompt_with = template.render(include_continuation=True, **context)
    return [prompt_without, prompt_with]


# -----------------------------------------------------------------------------
# Sequence utilities

def find_common_length(token_sequences, direction='left'):
    """Find length of common prefix or suffix across sequences."""
    min_len = min(len(seq) for seq in token_sequences)
    indices = {
        'left': range(min_len),
        'right': range(-1, -min_len-1, -1)
    }[direction]
    for i, idx in enumerate(indices):
        token = token_sequences[0][idx]
        if not all(seq[idx] == token for seq in token_sequences):
            return i
    return min_len


def stack_sequences(tokens, pad_token_id):
    """Stack token sequences into padded tensor."""
    bsz, seq_len = len(tokens), max(len(x) for x in tokens)
    input_ids = torch.full((bsz, seq_len), pad_token_id, dtype=torch.long)
    for i, x in enumerate(tokens):
        input_ids[i, :len(x)] = torch.tensor(x, dtype=torch.long)
    return input_ids


def batch_sequences_mc(tokenizer, prompts):
    """Batch sequences for multiple choice tasks."""
    tokens = [tokenizer(p, prepend=tokenizer.get_bos_token_id()) for p in prompts]
    answer_start_idx = find_common_length(tokens, direction='left')
    start_indices = [answer_start_idx] * len(prompts)
    end_indices = [len(x) for x in tokens]
    return tokens, start_indices, end_indices


def batch_sequences_schema(tokenizer, prompts):
    """Batch sequences for schema tasks."""
    tokens = [tokenizer(p, prepend=tokenizer.get_bos_token_id()) for p in prompts]
    suffix_length = find_common_length(tokens, direction='right')
    end_indices = [len(x) for x in tokens]
    start_indices = [ei - suffix_length for ei in end_indices]
    return tokens, start_indices, end_indices


def batch_sequences_lm(tokenizer, prompts):
    """Batch sequences for language modeling tasks."""
    tokens_without = tokenizer(prompts[0], prepend=tokenizer.get_bos_token_id())
    tokens_with = tokenizer(prompts[1], prepend=tokenizer.get_bos_token_id())
    start_idx, end_idx = len(tokens_without), len(tokens_with)
    assert start_idx < end_idx, "prompt without should be prefix of prompt with"
    return [tokens_with], [start_idx], [end_idx]


# -----------------------------------------------------------------------------
# Evaluation core

@torch.no_grad()
def forward_model(model, input_ids):
    """Forward model and compute losses and predictions."""
    batch_size, seq_len = input_ids.size()
    outputs = model(input_ids)

    # Roll tensor left to get autoregressive targets
    target_ids = torch.roll(input_ids, shifts=-1, dims=1)

    # Calculate cross entropy at all positions
    losses = torch.nn.functional.cross_entropy(
        outputs.view(batch_size * seq_len, -1),
        target_ids.view(batch_size * seq_len),
        reduction='none'
    ).view(batch_size, seq_len)

    # Last column has no target
    losses[:, -1] = float('nan')

    # Argmax predictions
    predictions = outputs.argmax(dim=-1)

    return losses, predictions


@torch.no_grad()
def evaluate_example(idx, model, tokenizer, data, device, task_meta):
    """Evaluate a single example."""
    item = data[idx]
    task_type = task_meta['task_type']
    num_fewshot = task_meta['num_fewshot']
    continuation_delimiter = task_meta['continuation_delimiter']

    # Sample few-shot examples
    fewshot_examples = []
    if num_fewshot > 0:
        rng = random.Random(1234 + idx)
        available_indices = [i for i in range(len(data)) if i != idx]
        fewshot_indices = rng.sample(available_indices, min(num_fewshot, len(available_indices)))
        fewshot_examples = [data[i] for i in fewshot_indices]

    # Render prompts based on task type
    if task_type == 'multiple_choice':
        prompts = render_prompts_mc(item, continuation_delimiter, fewshot_examples)
        tokens, start_idxs, end_idxs = batch_sequences_mc(tokenizer, prompts)
    elif task_type == 'schema':
        prompts = render_prompts_schema(item, continuation_delimiter, fewshot_examples)
        tokens, start_idxs, end_idxs = batch_sequences_schema(tokenizer, prompts)
    elif task_type == 'language_modeling':
        prompts = render_prompts_lm(item, continuation_delimiter, fewshot_examples)
        tokens, start_idxs, end_idxs = batch_sequences_lm(tokenizer, prompts)
    else:
        raise ValueError(f"Unsupported task type: {task_type}")

    # Truncate if needed
    if hasattr(model, 'max_seq_len') and model.max_seq_len is not None:
        max_tokens = model.max_seq_len
        new_tokens, new_start_idxs, new_end_idxs = [], [], []
        for t, s, e in zip(tokens, start_idxs, end_idxs):
            if len(t) > max_tokens:
                num_to_crop = len(t) - max_tokens
                new_tokens.append(t[-max_tokens:])
                new_start_idxs.append(max(0, s - num_to_crop))
                new_end_idxs.append(e - num_to_crop)
            else:
                new_tokens.append(t)
                new_start_idxs.append(s)
                new_end_idxs.append(e)
        tokens, start_idxs, end_idxs = new_tokens, new_start_idxs, new_end_idxs

    # Stack and move to device
    pad_token_id = tokenizer.get_bos_token_id()
    input_ids = stack_sequences(tokens, pad_token_id).to(device)

    # Forward pass
    losses, predictions = forward_model(model, input_ids)

    # Evaluate correctness
    if task_type == 'language_modeling':
        si, ei = start_idxs[0], end_idxs[0]
        predicted_tokens = predictions[0, si-1:ei-1]
        actual_tokens = input_ids[0, si:ei]
        is_correct = torch.all(predicted_tokens == actual_tokens).item()
    elif task_type in ['multiple_choice', 'schema']:
        mean_losses = [losses[i, si-1:ei-1].mean().item()
                       for i, (si, ei) in enumerate(zip(start_idxs, end_idxs))]
        pred_idx = mean_losses.index(min(mean_losses))
        is_correct = pred_idx == item['gold']
    else:
        raise ValueError(f"Unsupported task type: {task_type}")

    return is_correct


def evaluate_task(model, tokenizer, data, device, task_meta):
    """Evaluate one task across all examples with DDP support."""
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1

    correct = torch.zeros(len(data), dtype=torch.float32, device=device)

    # Stride examples across ranks
    for idx in range(rank, len(data), world_size):
        is_correct = evaluate_example(idx, model, tokenizer, data, device, task_meta)
        correct[idx] = float(is_correct)

    # Sync results across processes
    if world_size > 1:
        dist.barrier()
        dist.all_reduce(correct, op=dist.ReduceOp.SUM)

    return correct.mean().item()


def evaluate_core(model, tokenizer, device, max_per_task=-1):
    """
    Evaluate model on CORE benchmark.

    Returns dict with:
        - results: raw accuracy per task
        - centered_results: centered accuracy per task
        - core_metric: average centered accuracy (CORE score)
    """
    eval_bundle_dir = ensure_eval_bundle()

    config_path = eval_bundle_dir / "core.yaml"
    data_base_path = eval_bundle_dir / "eval_data"
    eval_meta_path = eval_bundle_dir / "eval_meta_data.csv"

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    tasks = config['icl_tasks']

    # Load random baselines
    random_baselines = {}
    with open(eval_meta_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            task_name = row['Eval Task']
            random_baseline = row['Random baseline']
            random_baselines[task_name] = float(random_baseline)

    # Evaluate each task
    results = {}
    centered_results = {}

    for task in tasks:
        start_time = time.time()
        label = task['label']
        task_meta = {
            'task_type': task['icl_task_type'],
            'dataset_uri': task['dataset_uri'],
            'num_fewshot': task['num_fewshot'][0],
            'continuation_delimiter': task.get('continuation_delimiter', ' ')
        }

        print0(f"Evaluating: {label} ({task_meta['num_fewshot']}-shot, {task_meta['task_type']})... ", end='', flush=True)

        data_path = data_base_path / task_meta['dataset_uri']
        with open(data_path, 'r', encoding='utf-8') as f:
            data = [json.loads(line.strip()) for line in f]

        # Shuffle for consistent subsampling
        shuffle_rng = random.Random(1337)
        shuffle_rng.shuffle(data)
        if max_per_task > 0:
            data = data[:max_per_task]

        accuracy = evaluate_task(model, tokenizer, data, device, task_meta)
        results[label] = accuracy

        random_baseline = random_baselines.get(label, 0.0)
        centered_result = (accuracy - 0.01 * random_baseline) / (1.0 - 0.01 * random_baseline)
        centered_results[label] = centered_result

        elapsed = time.time() - start_time
        print0(f"acc: {accuracy:.4f} | centered: {centered_result:.4f} | {elapsed:.1f}s")

    core_metric = sum(centered_results.values()) / len(centered_results)

    return {
        "results": results,
        "centered_results": centered_results,
        "core_metric": core_metric
    }


# -----------------------------------------------------------------------------
# Model loading

def load_swamla_checkpoint(checkpoint_path: str, device):
    """Load a SWAMLA model from checkpoint."""
    from dataclasses import fields, asdict
    print0(f"Loading checkpoint: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    # Get config from checkpoint
    if 'config' in checkpoint:
        config = checkpoint['config']

        # Handle both dict and SWAMLAConfig objects
        if isinstance(config, SWAMLAConfig):
            # Already a config object, use as-is
            pass
        elif isinstance(config, dict):
            # Filter out unknown keys that aren't in SWAMLAConfig
            valid_fields = {f.name for f in fields(SWAMLAConfig)}
            filtered_config = {k: v for k, v in config.items() if k in valid_fields}

            # Convert string lists to actual lists (from CLI args like "2,6")
            list_fields = ['engram_layers', 'engram_ngram_orders']
            for field_name in list_fields:
                if field_name in filtered_config:
                    val = filtered_config[field_name]
                    if isinstance(val, str):
                        filtered_config[field_name] = [int(x) for x in val.split(',')]

            config = SWAMLAConfig(**filtered_config)
        else:
            raise ValueError(f"Unknown config type: {type(config)}")
    else:
        raise ValueError("Checkpoint does not contain config")

    # Debug: print key config values
    print0(f"Config: n_embd={config.n_embd}, n_layer={config.n_layer}, vocab_size={config.vocab_size}, block_size={config.block_size}")

    # Create model directly from config
    model = SWAMLAModel(config)

    # Load weights
    state_dict = checkpoint.get('model', checkpoint.get('model_state_dict', checkpoint))

    # Handle DDP prefix
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        elif k.startswith('_orig_mod.'):
            new_state_dict[k[10:]] = v
        else:
            new_state_dict[k] = v

    model.load_state_dict(new_state_dict, strict=False)
    model.to(device)
    model.eval()

    return SWAMLAWrapper(model, max_seq_len=config.block_size)


def load_hf_model(model_name: str, device):
    """Load a HuggingFace model."""
    print0(f"Loading HuggingFace model: {model_name}")

    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to(device)
    model.eval()

    # Determine max sequence length
    max_seq_len = None
    if "gpt2" in model_name.lower():
        max_seq_len = 1024
    elif hasattr(model.config, 'max_position_embeddings'):
        max_seq_len = model.config.max_position_embeddings

    return HFModelWrapper(model, max_seq_len=max_seq_len)


# -----------------------------------------------------------------------------
# Main

def main():
    parser = argparse.ArgumentParser(description="CORE Benchmark Evaluation")
    parser.add_argument('--checkpoint', type=str, help='Path to SWAMLA checkpoint')
    parser.add_argument('--hf_model', type=str, help='HuggingFace model name (e.g., openai-community/gpt2)')
    parser.add_argument('--tokenizer', type=str, default='openai-community/gpt2', help='Tokenizer to use')
    parser.add_argument('--max_per_task', type=int, default=-1, help='Max examples per task (-1 = all)')
    parser.add_argument('--output', type=str, help='Output JSON file for results')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
    args = parser.parse_args()

    if not args.checkpoint and not args.hf_model:
        parser.error("Must specify either --checkpoint or --hf_model")

    # Setup distributed
    is_distributed, rank, local_rank, world_size = setup_distributed()

    if is_distributed:
        device = torch.device(f'cuda:{local_rank}')
    else:
        device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    print0(f"Using device: {device}")
    if is_distributed:
        print0(f"Distributed: {world_size} GPUs")

    # Load tokenizer
    hf_tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    tokenizer = TokenizerWrapper(hf_tokenizer)

    # Load model
    if args.checkpoint:
        model = load_swamla_checkpoint(args.checkpoint, device)
    else:
        model = load_hf_model(args.hf_model, device)

    # Run evaluation
    print0("\n" + "=" * 60)
    print0("CORE Benchmark Evaluation")
    print0("=" * 60 + "\n")

    results = evaluate_core(model, tokenizer, device, max_per_task=args.max_per_task)

    # Print summary
    print0("\n" + "=" * 60)
    print0("Results Summary")
    print0("=" * 60)
    print0(f"\nCORE Score: {results['core_metric']:.6f}")
    print0(f"GPT-2 Baseline: {GPT2_CORE_BASELINE:.6f}")

    delta = results['core_metric'] - GPT2_CORE_BASELINE
    if delta > 0:
        print0(f"Delta vs GPT-2: +{delta:.6f} ✓")
    else:
        print0(f"Delta vs GPT-2: {delta:.6f}")

    print0("\nPer-task results:")
    for task, acc in sorted(results['results'].items()):
        centered = results['centered_results'][task]
        print0(f"  {task:40s} acc={acc:.4f}  centered={centered:.4f}")

    # Save results
    if args.output and rank == 0:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print0(f"\nResults saved to: {output_path}")

    # Cleanup distributed
    if is_distributed:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
