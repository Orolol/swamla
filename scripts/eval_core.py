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

    # Load latest checkpoint from HuggingFace repo
    python scripts/eval_core.py --hf_repo_id username/swamla-model

    # Load specific checkpoint from HuggingFace repo
    python scripts/eval_core.py --hf_repo_id username/swamla-model --hf_checkpoint checkpoint_tokens_500k_loss_2.3456

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

from models.swa_mla_model import SWAMLAModel, SWAMLAConfig, create_swa_mla_model
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
        """Forward pass returning logits for all positions."""
        with torch.inference_mode():
            logits, _ = self.model(input_ids, return_all_logits=True)
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
# HuggingFace repo loading

def load_swamla_from_hf(
    repo_id: str,
    checkpoint_name: str = None,
    device: torch.device = None,
    hf_token: str = None,
):
    """Load SWAMLA model from HuggingFace repo.

    Args:
        repo_id: HF model repo ID (e.g., "username/swamla-model")
        checkpoint_name: Optional specific checkpoint folder name.
                        If None, loads the latest checkpoint automatically.
        device: Device to load model on
        hf_token: Optional HuggingFace token for private repos

    Returns:
        SWAMLAWrapper instance
    """
    from huggingface_hub import hf_hub_download, list_repo_files
    from dataclasses import fields
    import re
    import ast

    # If no checkpoint specified, find the latest one
    if checkpoint_name is None:
        print0(f"Finding latest checkpoint in {repo_id}...")

        files = list_repo_files(repo_id, token=hf_token)

        # Find all checkpoint directories
        checkpoint_pattern = re.compile(r'checkpoint_tokens_(\d+[kKmMbB])_loss_([\d.]+)/pytorch_model\.bin')
        checkpoints = []

        for file in files:
            match = checkpoint_pattern.match(file)
            if match:
                tokens_str = match.group(1)
                loss_str = match.group(2)

                # Parse tokens
                tokens_multiplier = {'k': 1000, 'K': 1000, 'm': 1_000_000, 'M': 1_000_000, 'b': 1_000_000_000, 'B': 1_000_000_000}
                tokens_value = int(tokens_str[:-1])
                tokens_suffix = tokens_str[-1]
                total_tokens = tokens_value * tokens_multiplier.get(tokens_suffix, 1)

                checkpoints.append({
                    'name': file.rsplit('/', 1)[0],
                    'total_tokens': total_tokens,
                    'loss': float(loss_str),
                })

        if not checkpoints:
            raise ValueError(f"No checkpoints found in {repo_id}")

        # Sort by total tokens (most recent training)
        checkpoints.sort(key=lambda x: x['total_tokens'], reverse=True)
        checkpoint_name = checkpoints[0]['name']

        print0(f"Found {len(checkpoints)} checkpoints")
        print0(f"Loading latest: {checkpoint_name} (tokens: {checkpoints[0]['total_tokens']:,}, loss: {checkpoints[0]['loss']:.4f})")
    else:
        print0(f"Loading checkpoint {checkpoint_name} from {repo_id}...")

    # Download config
    config_path = hf_hub_download(
        repo_id=repo_id,
        filename=f"{checkpoint_name}/config.json",
        token=hf_token
    )
    with open(config_path, 'r') as f:
        config_dict = json.load(f)

    # Download weights
    weights_path = hf_hub_download(
        repo_id=repo_id,
        filename=f"{checkpoint_name}/pytorch_model.bin",
        token=hf_token
    )

    checkpoint_data = torch.load(weights_path, map_location="cpu", weights_only=False)

    # Extract state_dict
    if 'model_state_dict' in checkpoint_data:
        state_dict = checkpoint_data['model_state_dict']
    else:
        state_dict = checkpoint_data

    # Remove DDP/compile wrapper prefixes
    if any(key.startswith("module.") for key in state_dict.keys()):
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    if any(key.startswith("_orig_mod.") for key in state_dict.keys()):
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}

    # Extract vocab_size and block_size from state_dict
    vocab_size = None
    block_size = None
    if "transformer.wte.weight" in state_dict:
        vocab_size = state_dict["transformer.wte.weight"].shape[0]
        print0(f"Extracted vocab_size={vocab_size} from checkpoint")
    if "transformer.wpe.weight" in state_dict:
        block_size = state_dict["transformer.wpe.weight"].shape[0]
        print0(f"Extracted block_size={block_size} from checkpoint")

    # Process training_config if present
    if 'training_config' in config_dict:
        training_config = config_dict['training_config'].copy()
        model_size = training_config.get('size', None)

        # Map parameter names
        param_mapping = {
            'mla_q_lora_rank': 'q_lora_rank',
            'mla_kv_lora_rank': 'kv_lora_rank',
            'mla_qk_nope_head_dim': 'qk_nope_head_dim',
            'mla_qk_rope_head_dim': 'qk_rope_head_dim',
            'mla_v_head_dim': 'v_head_dim',
        }
        for old_name, new_name in param_mapping.items():
            if old_name in training_config:
                training_config[new_name] = training_config[old_name]

        if 'gradient_checkpointing' in training_config:
            training_config['use_gradient_checkpointing'] = training_config.pop('gradient_checkpointing')

        # Filter to valid SWAMLAConfig fields
        valid_fields = {f.name for f in fields(SWAMLAConfig)}
        training_config = {k: v for k, v in training_config.items() if k in valid_fields}

        # Convert string lists
        list_fields = ['engram_layers', 'engram_ngram_orders']
        for field_name in list_fields:
            if field_name in training_config and isinstance(training_config[field_name], str):
                try:
                    training_config[field_name] = ast.literal_eval(training_config[field_name])
                except (ValueError, SyntaxError):
                    pass

        # Force use_fp8 to False for inference
        training_config['use_fp8'] = False

        # Override from checkpoint
        if vocab_size is not None:
            training_config['vocab_size'] = vocab_size
        if block_size is not None:
            training_config['block_size'] = block_size

        # Create model
        if model_size:
            training_config.pop('size', None)
            model = create_swa_mla_model(size=model_size, **training_config)
        else:
            config = SWAMLAConfig(**training_config)
            model = SWAMLAModel(config)
    else:
        # Fallback to old format
        if vocab_size is not None:
            config_dict['vocab_size'] = vocab_size
        if block_size is not None:
            config_dict['block_size'] = block_size
        valid_fields = {f.name for f in fields(SWAMLAConfig)}
        config_dict = {k: v for k, v in config_dict.items() if k in valid_fields}
        config = SWAMLAConfig(**config_dict)
        model = SWAMLAModel(config)

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print0(f"  Missing keys (will use default init): {missing}")
    if unexpected:
        print0(f"  Unexpected keys (ignored): {unexpected}")
    model.to(device)
    model.eval()

    print0(f"Model loaded successfully ({model.param_count / 1e6:.2f}M parameters)")
    return SWAMLAWrapper(model, max_seq_len=model.config.block_size)


# -----------------------------------------------------------------------------
# Model loading

def infer_config_from_weights(state_dict):
    """Infer model config from weight shapes."""
    inferred = {}

    # Infer vocab_size and n_embd from embedding or lm_head
    if 'transformer.wte.weight' in state_dict:
        inferred['vocab_size'], inferred['n_embd'] = state_dict['transformer.wte.weight'].shape
    elif 'lm_head.weight' in state_dict:
        inferred['vocab_size'], inferred['n_embd'] = state_dict['lm_head.weight'].shape

    # Infer n_layer by counting transformer blocks
    n_layer = 0
    for key in state_dict.keys():
        if key.startswith('transformer.h.'):
            layer_idx = int(key.split('.')[2])
            n_layer = max(n_layer, layer_idx + 1)
    inferred['n_layer'] = n_layer

    # Infer n_head from DeltaNet g_proj (first DeltaNet block)
    for i in range(n_layer):
        key = f'transformer.h.{i}.attn.g_proj.weight'
        if key in state_dict:
            inferred['n_head'] = state_dict[key].shape[0]
            break

    # Infer MLA parameters from first MLA block
    for i in range(n_layer):
        wkv_a_key = f'transformer.h.{i}.attn.wkv_a.weight'
        wkv_b_key = f'transformer.h.{i}.attn.wkv_b.weight'
        wo_key = f'transformer.h.{i}.attn.wo.weight'
        kv_norm_key = f'transformer.h.{i}.attn.kv_norm.weight'

        if wkv_a_key in state_dict and wkv_b_key in state_dict:
            # This is an MLA block
            wkv_a_shape = state_dict[wkv_a_key].shape  # [kv_lora_rank + rope_dim, n_embd]
            wkv_b_shape = state_dict[wkv_b_key].shape  # [n_head * (nope + v), kv_lora_rank + rope_dim]
            wo_shape = state_dict[wo_key].shape  # [n_embd, n_head * v_head_dim]
            kv_norm_shape = state_dict[kv_norm_key].shape  # [kv_lora_rank + rope_dim - some offset]

            # wo gives us n_head * v_head_dim
            n_head_times_v = wo_shape[1]

            # Try to infer rope_head_dim from wkv_a
            # wkv_a.shape[0] = kv_lora_rank + rope_head_dim
            # kv_norm.shape[0] = kv_lora_rank + rope_head_dim (with possible offset)

            # For now, just store raw dimensions - let the model figure it out
            break

    return inferred


def load_swamla_checkpoint(checkpoint_path: str, device, size='engram-moe-1b', n_experts=None, latent_ratio=None):
    """Load a SWAMLA model from checkpoint."""
    from dataclasses import fields, asdict
    print0(f"Loading checkpoint: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    # Get state dict first to infer dimensions
    state_dict = checkpoint.get('model', checkpoint.get('model_state_dict', checkpoint))

    # Handle DDP/compile prefixes
    clean_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            clean_state_dict[k[7:]] = v
        elif k.startswith('_orig_mod.'):
            clean_state_dict[k[10:]] = v
        else:
            clean_state_dict[k] = v
    state_dict = clean_state_dict

    # Get config from checkpoint
    config = None
    use_fallback = False

    if 'config' in checkpoint:
        ckpt_config = checkpoint['config']

        # Handle both dict and SWAMLAConfig objects
        if isinstance(ckpt_config, SWAMLAConfig):
            config = ckpt_config
            print0(f"Config from checkpoint (SWAMLAConfig): n_embd={config.n_embd}, n_layer={config.n_layer}, n_head={config.n_head}")
        elif isinstance(ckpt_config, dict):
            # Check if config has actual model dimensions (not just CLI args)
            has_dimensions = ckpt_config.get('n_embd') is not None and ckpt_config.get('n_layer') is not None
            print0(f"Config from checkpoint (dict): n_embd={ckpt_config.get('n_embd')}, n_layer={ckpt_config.get('n_layer')}, n_head={ckpt_config.get('n_head')}")

            if has_dimensions:
                # Filter out unknown keys that aren't in SWAMLAConfig
                valid_fields = {f.name for f in fields(SWAMLAConfig)}
                filtered_config = {k: v for k, v in ckpt_config.items() if k in valid_fields}

                # Convert string lists to actual lists (from CLI args like "2,6")
                list_fields = ['engram_layers', 'engram_ngram_orders']
                for field_name in list_fields:
                    if field_name in filtered_config:
                        val = filtered_config[field_name]
                        if isinstance(val, str):
                            filtered_config[field_name] = [int(x) for x in val.split(',')]

                config = SWAMLAConfig(**filtered_config)
            else:
                use_fallback = True
        else:
            use_fallback = True
    else:
        use_fallback = True

    # Fallback to preset if config is incomplete
    if use_fallback:
        print0(f"Config incomplete - using fallback preset '{size}'")
        # Build config overrides from arguments
        config_override = {}
        if n_experts is not None:
            config_override['n_experts'] = n_experts
        if latent_ratio is not None:
            config_override['latent_ratio'] = latent_ratio

        model = create_swa_mla_model(
            size=size,
            vocab_size=50257,
            block_size=2048,
            config_override=config_override if config_override else None
        )
        config = model.config
    else:
        # Debug: print final config values
        print0(f"Final config: n_embd={config.n_embd}, n_layer={config.n_layer}, n_head={config.n_head}, vocab_size={config.vocab_size}, block_size={config.block_size}")
        # Create model directly from config
        model = SWAMLAModel(config)

    # Load weights (state_dict already cleaned above)
    model.load_state_dict(state_dict, strict=False)
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
    parser.add_argument('--checkpoint', type=str, help='Path to local SWAMLA checkpoint')
    parser.add_argument('--hf_repo_id', type=str, help='HuggingFace repo ID for SWAMLA model (e.g., username/swamla-model)')
    parser.add_argument('--hf_checkpoint', type=str, default=None,
                        help='Specific checkpoint folder in HF repo (e.g., checkpoint_tokens_500k_loss_2.3456). If not specified, loads latest.')
    parser.add_argument('--hf_model', type=str, help='HuggingFace model name for comparison (e.g., openai-community/gpt2)')
    parser.add_argument('--size', type=str, default='engram-moe-1b', help='Model size preset for fallback (default: engram-moe-1b)')
    parser.add_argument('--n_experts', type=int, default=None, help='Override number of experts')
    parser.add_argument('--latent_ratio', type=int, default=None, help='Override latent ratio for MoE')
    parser.add_argument('--tokenizer', type=str, default='openai-community/gpt2', help='Tokenizer to use')
    parser.add_argument('--max_per_task', type=int, default=-1, help='Max examples per task (-1 = all)')
    parser.add_argument('--output', type=str, help='Output JSON file for results')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
    args = parser.parse_args()

    if not args.checkpoint and not args.hf_repo_id and not args.hf_model:
        parser.error("Must specify one of: --checkpoint, --hf_repo_id, or --hf_model")

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
        model = load_swamla_checkpoint(
            args.checkpoint, device,
            size=args.size,
            n_experts=args.n_experts,
            latent_ratio=args.latent_ratio
        )
    elif args.hf_repo_id:
        hf_token = os.getenv("HF_TOKEN")
        model = load_swamla_from_hf(
            repo_id=args.hf_repo_id,
            checkpoint_name=args.hf_checkpoint,
            device=device,
            hf_token=hf_token,
        )
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
