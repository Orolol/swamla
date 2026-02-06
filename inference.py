"""
Inference script for SWA-MLA model.
Supports loading from Hugging Face and two modes:
- Batch mode: Generate responses for pre-registered prompts
- Chat mode: Interactive conversation with the model

Examples:
    # Load latest checkpoint from HuggingFace repo
    python inference.py --hf_repo_id username/swamla-model --mode chat

    # Load specific checkpoint from HuggingFace repo
    python inference.py --hf_repo_id username/swamla-model --hf_checkpoint checkpoint_tokens_500k_loss_2.3456 --mode chat

    # Load from local checkpoint file
    python inference.py --checkpoint outputs/swa_mla/checkpoint_1000.pt --mode batch

    # Batch mode with custom prompts file
    python inference.py --hf_repo_id username/swamla-model --mode batch --prompts_file my_prompts.txt
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Optional, List, Dict

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

# Add models directory to path
sys.path.insert(0, str(Path(__file__).parent / 'models'))

from swa_mla_model import SWAMLAModel, SWAMLAConfig, create_swa_mla_model


class InferenceEngine:
    """Inference engine for SWA-MLA model."""

    def __init__(
        self,
        model: SWAMLAModel,
        tokenizer: AutoTokenizer,
        device: str = "cuda",
        max_length: int = 2048,
        use_autocast: bool = True,
    ):
        self.model = model.to(device)
        self.model.eval()
        self.tokenizer = tokenizer
        self.device = device
        self.max_length = max_length
        self.use_autocast = use_autocast

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.8,
        top_k: Optional[int] = 50,
        top_p: Optional[float] = 0.9,
        repetition_penalty: float = 1.1,
        stop_tokens: Optional[List[int]] = None,
        debug: bool = False,
        return_full_text: bool = True,
        do_sample: bool = True,
    ) -> str:
        """Generate text from a prompt.

        Args:
            prompt: Input text prompt
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_k: Keep only top k tokens with highest probability
            top_p: Keep tokens with cumulative probability >= top_p (nucleus sampling)
            repetition_penalty: Penalty for repeating tokens (>1.0 = less repetition)
            stop_tokens: List of token IDs that stop generation
            debug: Print top-5 token probabilities for first few tokens
            return_full_text: If True, return prompt + continuation; else continuation only
            do_sample: If False, use greedy decoding (argmax)

        Returns:
            Generated text (prompt + continuation)
        """
        # Encode prompt
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)

        if input_ids.size(1) > self.max_length:
            print(f"Warning: Prompt length ({input_ids.size(1)}) exceeds max_length ({self.max_length}). Truncating.")
            input_ids = input_ids[:, -self.max_length:]

        # Track token frequencies for repetition penalty
        token_counts = {}

        # Generate tokens one by one
        generated_ids = input_ids.clone()
        debug_token_count = 0
        for _ in range(max_new_tokens):
            # Truncate to max context length
            context_ids = generated_ids if generated_ids.size(1) <= self.model.config.block_size else generated_ids[:, -self.model.config.block_size:]

            # Forward pass (autocast matches training: bf16 compute, fp32 for sensitive ops)
            with torch.amp.autocast(self.device, dtype=torch.bfloat16, enabled=(self.use_autocast and self.device == 'cuda')):
                logits, _ = self.model(context_ids)
            logits = logits[:, -1, :].float()  # Always use fp32 for sampling

            # Debug: print top-5 token probabilities for first few tokens
            if debug and debug_token_count < 5:
                raw_probs = F.softmax(logits, dim=-1)
                top5_probs, top5_ids = torch.topk(raw_probs, 5, dim=-1)
                tokens = [self.tokenizer.decode([tid]) for tid in top5_ids[0].tolist()]
                entropy = -(raw_probs * torch.log(raw_probs + 1e-10)).sum().item()
                print(f"  [DEBUG token {debug_token_count}] entropy={entropy:.2f} top5: ", end="")
                for tok, prob in zip(tokens, top5_probs[0].tolist()):
                    print(f"'{tok}'({prob:.4f}) ", end="")
                print()
                debug_token_count += 1

            # Apply repetition penalty
            if repetition_penalty != 1.0:
                for token_id, count in token_counts.items():
                    penalty = repetition_penalty ** count
                    token_logit = logits[0, token_id]
                    # Match standard repetition penalty behavior:
                    # negative logits are multiplied, positive are divided.
                    if token_logit < 0:
                        logits[0, token_id] = token_logit * penalty
                    else:
                        logits[0, token_id] = token_logit / penalty

            if do_sample:
                # Apply temperature
                logits = logits / max(temperature, 1e-6)

                # Apply top-k filtering
                if top_k is not None and top_k > 0:
                    top_k_actual = min(top_k, logits.size(-1))
                    values, _ = torch.topk(logits, top_k_actual)
                    logits[logits < values[:, [-1]]] = -float("inf")

                # Apply top-p (nucleus) filtering
                if top_p is not None and top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    # Keep at least one token
                    sorted_indices_to_remove[..., 0] = False

                    # Scatter back to original indexing
                    indices_to_remove = torch.zeros_like(logits, dtype=torch.bool)
                    indices_to_remove.scatter_(1, sorted_indices, sorted_indices_to_remove)
                    logits[indices_to_remove] = -float("inf")

                # Sample from the filtered distribution
                probs = F.softmax(logits, dim=-1)
                # Safety fallback if filtering produced an invalid distribution
                if (not torch.isfinite(probs).all()) or torch.any(probs.sum(dim=-1) <= 0):
                    next_token = torch.argmax(logits, dim=-1, keepdim=True)
                else:
                    next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)

            # Update token counts for repetition penalty
            token_id = next_token.item()
            token_counts[token_id] = token_counts.get(token_id, 0) + 1

            # Append to generated sequence
            generated_ids = torch.cat([generated_ids, next_token], dim=1)

            # Check for stop tokens
            if stop_tokens and token_id in stop_tokens:
                break

            # Check if we've hit max length
            if generated_ids.size(1) >= self.max_length:
                break

        # Decode generated text
        generated_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        if return_full_text:
            return generated_text

        continuation_ids = generated_ids[0, input_ids.size(1):]
        continuation_text = self.tokenizer.decode(continuation_ids, skip_special_tokens=True)
        return continuation_text


def _setup_torchao_mock():
    """Setup mock TorchAO modules for loading old checkpoints.

    Legacy function for backward compatibility with checkpoints that may have been
    trained with TorchAO optimizers. Now that we use native FP8, this is mainly
    for loading historical checkpoints.
    """
    try:
        from torchao.optim.subclass_fp8 import OptimStateFp8
        print("TorchAO detected - using native OptimStateFp8")
    except ImportError:
        print("Creating mock OptimStateFp8 for legacy checkpoint loading")
        # Create a dummy class to allow unpickling without TorchAO
        import sys
        from types import ModuleType

        # Create fake torchao modules
        if 'torchao' not in sys.modules:
            torchao_module = ModuleType('torchao')
            sys.modules['torchao'] = torchao_module

            optim_module = ModuleType('torchao.optim')
            sys.modules['torchao.optim'] = optim_module

            subclass_fp8_module = ModuleType('torchao.optim.subclass_fp8')
            sys.modules['torchao.optim.subclass_fp8'] = subclass_fp8_module

            # Create dummy OptimStateFp8 class that mimics torch.Tensor subclass
            class OptimStateFp8(torch.Tensor):
                """Mock OptimStateFp8 for loading legacy checkpoints."""

                @staticmethod
                def __new__(cls, data, *args, **kwargs):
                    if isinstance(data, torch.Tensor):
                        return data.as_subclass(cls)
                    return torch.as_tensor(data).as_subclass(cls)

                def __init__(self, *args, **kwargs):
                    pass

                @classmethod
                def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
                    """Required for torch.Tensor subclasses."""
                    kwargs = kwargs or {}

                    def unwrap(x):
                        if isinstance(x, OptimStateFp8):
                            return x.as_subclass(torch.Tensor)
                        return x

                    def unwrap_args(a):
                        if isinstance(a, (list, tuple)):
                            return type(a)(unwrap_args(x) for x in a)
                        return unwrap(a)

                    args = unwrap_args(args)
                    kwargs = {k: unwrap_args(v) for k, v in kwargs.items()}

                    result = func(*args, **kwargs)
                    return result

            subclass_fp8_module.OptimStateFp8 = OptimStateFp8


def _restore_precision_buffers(model: SWAMLAModel, device: str = "cuda"):
    """Restore numerically sensitive buffers to float32 after model.to(bf16).

    FoPE's inv_freq/harmonic_mult and RoPE's cos_cached/sin_cached are computed in
    float32 during model construction but get downcast by model.to(dtype=bf16).
    This recomputes them in float32 to preserve Fourier frequency precision.

    Args:
        model: The SWA-MLA model (already on device with bf16 weights).
        device: Target device string.
    """
    restored = []
    for name, module in model.named_modules():
        # Restore FoPE buffers
        if module.__class__.__name__ == 'FoPE':
            dim = module.dim
            base = module.base
            # Recompute inv_freq in float32
            inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
            module.register_buffer('inv_freq', inv_freq.to(device=device), persistent=False)
            # Recompute harmonic_mult in float32
            if module.harmonic_mult is not None:
                harmonic_mult = torch.arange(1, module.n_harmonics + 1).float()
                module.register_buffer('harmonic_mult', harmonic_mult.to(device=device), persistent=False)
            restored.append(f"{name} (FoPE: inv_freq, harmonic_mult)")

        # Restore RoPE buffers
        elif module.__class__.__name__ == 'RoPE':
            dim = module.dim
            max_seq_len = module.max_seq_len
            base = module.base
            # Recompute cos/sin cache in float32
            inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
            t = torch.arange(max_seq_len).type_as(inv_freq)
            freqs = torch.einsum('i,j->ij', t, inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos_cached = emb.cos().view(1, 1, max_seq_len, dim)
            sin_cached = emb.sin().view(1, 1, max_seq_len, dim)
            module.register_buffer('cos_cached', cos_cached.to(device=device), persistent=False)
            module.register_buffer('sin_cached', sin_cached.to(device=device), persistent=False)
            restored.append(f"{name} (RoPE: cos_cached, sin_cached)")

    if restored:
        print(f"Restored {len(restored)} precision buffer(s) to float32:")
        for r in restored:
            print(f"  - {r}")


def _parse_token_count(s: str) -> int:
    """Parse a human-readable token count string like '3.5B', '500M', '100k'.

    Args:
        s: Token count string with optional suffix (k/M/B/T).

    Returns:
        Integer token count.
    """
    s = s.strip()
    multipliers = {'k': 1_000, 'K': 1_000, 'm': 1_000_000, 'M': 1_000_000,
                   'b': 1_000_000_000, 'B': 1_000_000_000,
                   't': 1_000_000_000_000, 'T': 1_000_000_000_000}
    if s[-1] in multipliers:
        return int(float(s[:-1]) * multipliers[s[-1]])
    return int(float(s))


def load_model_from_hf(
    repo_id: str,
    checkpoint_name: Optional[str] = None,
    checkpoint_strategy: str = "latest",
    target_tokens: Optional[int] = None,
    device: str = "cuda",
    torch_dtype: torch.dtype = torch.bfloat16,
    hf_token: Optional[str] = None,
) -> tuple[SWAMLAModel, AutoTokenizer]:
    """Load SWA-MLA model and tokenizer from Hugging Face.

    Args:
        repo_id: HF model repo ID (e.g., "username/swamla-model")
        checkpoint_name: Optional specific checkpoint folder name (e.g., "checkpoint_tokens_500k_loss_2.3456")
                        If None, loads the latest checkpoint automatically
        checkpoint_strategy: Strategy when checkpoint_name is None:
                             "latest" (highest token count) or "best_loss" (lowest loss)
        target_tokens: Optional target token count. Selects the checkpoint closest to this value.
                      Overrides checkpoint_strategy when set.
        device: Device to load model on
        torch_dtype: Data type for model weights
        hf_token: Optional HuggingFace token for private repos

    Returns:
        Tuple of (model, tokenizer)
    """
    from huggingface_hub import hf_hub_download, list_repo_files
    import json
    import re

    # Setup mock for legacy checkpoint compatibility (old FP8 optimizer states)
    _setup_torchao_mock()

    # If no checkpoint specified, find the latest one
    if checkpoint_name is None:
        print(f"Finding latest checkpoint in {repo_id}...")

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
                    'name': file.rsplit('/', 1)[0],
                    'total_tokens': total_tokens,
                    'loss': float(loss_str),
                })

        if not checkpoints:
            raise ValueError(f"No checkpoints found in {repo_id}")

        if target_tokens is not None:
            checkpoints.sort(key=lambda x: abs(x['total_tokens'] - target_tokens))
            selected = checkpoints[0]
            selection_label = f"closest to {target_tokens:,} tokens"
        elif checkpoint_strategy == "best_loss":
            checkpoints.sort(key=lambda x: (x['loss'], -x['total_tokens']))
            selected = checkpoints[0]
            selection_label = "best_loss"
        else:
            checkpoints.sort(key=lambda x: x['total_tokens'], reverse=True)
            selected = checkpoints[0]
            selection_label = "latest"

        checkpoint_name = selected['name']

        print(f"Found {len(checkpoints)} checkpoints")
        print(f"Loading {selection_label}: {checkpoint_name} (tokens: {selected['total_tokens']:,}, loss: {selected['loss']:.4f})")
    else:
        print(f"Loading checkpoint {checkpoint_name} from {repo_id}...")

    # Download config from checkpoint subfolder
    config_path = hf_hub_download(
        repo_id=repo_id,
        filename=f"{checkpoint_name}/config.json",
        token=hf_token
    )
    with open(config_path, 'r') as f:
        config_dict = json.load(f)

    # Download and load weights FIRST to extract vocab_size
    weights_path = hf_hub_download(
        repo_id=repo_id,
        filename=f"{checkpoint_name}/pytorch_model.bin",
        token=hf_token
    )

    checkpoint_data = torch.load(weights_path, map_location="cpu", weights_only=False)

    # Extract state_dict (could be directly in checkpoint or under 'model_state_dict' key)
    if 'model_state_dict' in checkpoint_data:
        state_dict = checkpoint_data['model_state_dict']
    else:
        state_dict = checkpoint_data

    # Remove DDP wrapper prefix if present
    if any(key.startswith("module.") for key in state_dict.keys()):
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    # Remove torch.compile wrapper prefix if present
    if any(key.startswith("_orig_mod.") for key in state_dict.keys()):
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}

    # Extract vocab_size from state_dict BEFORE creating model
    vocab_size = None
    block_size = None
    if "transformer.wte.weight" in state_dict:
        vocab_size = state_dict["transformer.wte.weight"].shape[0]
        print(f"Extracted vocab_size={vocab_size} from checkpoint")
    if "transformer.wpe.weight" in state_dict:
        block_size = state_dict["transformer.wpe.weight"].shape[0]
        print(f"Extracted block_size={block_size} from checkpoint")

    # Extract training_config if it exists (new format from train.py)
    if 'training_config' in config_dict:
        training_config = config_dict['training_config'].copy()

        # Check if 'size' preset is specified (small, base, large, xl)
        model_size = training_config.get('size', None)

        # Map training config parameter names to model config parameter names
        param_mapping = {
            'mla_q_lora_rank': 'q_lora_rank',
            'mla_kv_lora_rank': 'kv_lora_rank',
            'mla_qk_nope_head_dim': 'qk_nope_head_dim',
            'mla_qk_rope_head_dim': 'qk_rope_head_dim',
            'mla_v_head_dim': 'v_head_dim',
        }

        # Apply mapping
        for old_name, new_name in param_mapping.items():
            if old_name in training_config:
                training_config[new_name] = training_config[old_name]

        # Rename gradient_checkpointing to use_gradient_checkpointing if needed
        if 'gradient_checkpointing' in training_config:
            training_config['use_gradient_checkpointing'] = training_config.pop('gradient_checkpointing')

        # Filter to only valid SWAMLAConfig fields (robust against checkpoint extras)
        from dataclasses import fields
        import ast
        valid_fields = {f.name for f in fields(SWAMLAConfig)}
        training_config = {k: v for k, v in training_config.items() if k in valid_fields}

        # Convert string representations of lists back to actual lists
        list_fields = ['engram_layers', 'engram_ngram_orders']
        for field_name in list_fields:
            if field_name in training_config and isinstance(training_config[field_name], str):
                try:
                    training_config[field_name] = ast.literal_eval(training_config[field_name])
                except (ValueError, SyntaxError):
                    pass  # Keep as-is if parsing fails

        # Force FP8 off for inference (use correct SWAMLAConfig field names)
        training_config['use_te_fp8'] = False
        training_config['fp8_backend'] = 'none'

        # Override vocab_size and block_size from checkpoint if extracted
        if vocab_size is not None:
            training_config['vocab_size'] = vocab_size
        if block_size is not None:
            training_config['block_size'] = block_size

        # Create model using the appropriate method
        if model_size:
            # Use create_swa_mla_model() with size preset
            # Remove 'size' from training_config as it's passed separately
            training_config.pop('size', None)
            model = create_swa_mla_model(size=model_size, **training_config)
        else:
            # Direct config creation (fallback)
            config = SWAMLAConfig(**training_config)
            model = SWAMLAModel(config)
    else:
        # Fallback to old format (direct config)
        if vocab_size is not None:
            config_dict['vocab_size'] = vocab_size
        if block_size is not None:
            config_dict['block_size'] = block_size
        # Filter to only valid SWAMLAConfig fields
        from dataclasses import fields
        import ast
        valid_fields = {f.name for f in fields(SWAMLAConfig)}
        config_dict = {k: v for k, v in config_dict.items() if k in valid_fields}
        # Convert string representations of lists back to actual lists
        list_fields = ['engram_layers', 'engram_ngram_orders']
        for field_name in list_fields:
            if field_name in config_dict and isinstance(config_dict[field_name], str):
                try:
                    config_dict[field_name] = ast.literal_eval(config_dict[field_name])
                except (ValueError, SyntaxError):
                    pass
        config = SWAMLAConfig(**config_dict)
        model = SWAMLAModel(config)

    model.load_state_dict(state_dict)

    # Preserve complex-valued buffers (e.g., freqs_cis) before dtype conversion
    # Converting complex64 to bfloat16 discards the imaginary part
    complex_buffers = {}
    for name, buf in model.named_buffers():
        if buf is not None and buf.is_complex():
            complex_buffers[name] = buf.clone()

    model = model.to(device=device, dtype=torch_dtype)

    # Restore complex-valued buffers (they must stay complex64)
    for name, buf in complex_buffers.items():
        parts = name.split('.')
        obj = model
        for part in parts[:-1]:
            obj = getattr(obj, part)
        # Use register_buffer to properly restore tracked buffers
        obj.register_buffer(parts[-1], buf.to(device=device), persistent=False)

    # Restore FoPE/RoPE buffers to float32 (they lose precision from model.to(bf16))
    _restore_precision_buffers(model, device)

    model.eval()

    # Load tokenizer from checkpoint subfolder
    print("Loading tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            repo_id,
            subfolder=checkpoint_name,
            token=hf_token
        )
    except Exception as e:
        print(f"Failed to load tokenizer from checkpoint: {e}")
        print("Falling back to default GPT-2 tokenizer")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")

    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Set tokenizer compression mapping for Engram if the model uses it
    if hasattr(model, 'set_engram_tokenizer_compression') and getattr(model.config, 'use_engram', False):
        model.set_engram_tokenizer_compression(tokenizer)
        print("Engram tokenizer compression mapping initialized")

    print(f"Model loaded successfully ({model.param_count / 1e6:.2f}M parameters)")
    return model, tokenizer


def load_model_from_checkpoint(
    checkpoint_path: str,
    device: str = "cuda",
    torch_dtype: torch.dtype = torch.bfloat16,
    use_ema: bool = True,
) -> tuple[SWAMLAModel, AutoTokenizer]:
    """Load SWA-MLA model from a local checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load model on
        torch_dtype: Data type for model weights
        use_ema: If True, prefer EMA weights when available in checkpoint

    Returns:
        Tuple of (model, tokenizer)
    """
    print(f"Loading checkpoint from {checkpoint_path}...")

    # Setup mock for legacy checkpoint compatibility (old FP8 optimizer states)
    _setup_torchao_mock()

    # Load checkpoint with weights_only=False to allow optimizer states
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Extract config
    if "config" in checkpoint:
        config_dict = checkpoint["config"].copy()

        # Check if 'size' preset is specified (small, base, large, xl)
        model_size = config_dict.get('size', None)

        # Map training config parameter names to model config parameter names
        # Training config uses 'mla_' prefix, model config doesn't
        param_mapping = {
            'mla_q_lora_rank': 'q_lora_rank',
            'mla_kv_lora_rank': 'kv_lora_rank',
            'mla_qk_nope_head_dim': 'qk_nope_head_dim',
            'mla_qk_rope_head_dim': 'qk_rope_head_dim',
            'mla_v_head_dim': 'v_head_dim',
        }

        # Apply mapping
        for old_name, new_name in param_mapping.items():
            if old_name in config_dict:
                config_dict[new_name] = config_dict.pop(old_name)

        # Remove training-specific parameters that don't exist in SWAMLAConfig
        training_only_params = [
            # Optimizer and training loop params
            'batch_size', 'max_iters', 'learning_rate', 'min_lr',
            'weight_decay', 'beta1', 'beta2', 'warmup_iters', 'grad_clip',
            'gradient_accumulation_steps', 'optimizer_type', 'enable_tf32',
            'disable_tf32', 'tokenizer_name', 'num_workers', 'output_dir',
            'log_interval', 'eval_interval', 'save_interval', 'wandb_project',
            'wandb_run_name', 'hf_repo_id', 'resume_from_hf', 'compile', 'compile_mode',
            'resume_from', 'profile', 'profile_steps', 'profile_warmup',
            # Neural memory training params
            'memory_reset_interval',
            # Other training-only params
            'use_tensorboard', 'use_te_fp8', 'use_varlen_attn',
            # WeDLM training params
            'use_wedlm', 'wedlm_block_size', 'wedlm_min_mask_ratio',
            'wedlm_max_mask_ratio', 'wedlm_ar_loss_weight', 'wedlm_mask_token_id',
            # Deprecated/removed params
            'no_moe',
        ]

        for param in training_only_params:
            config_dict.pop(param, None)

        # Convert string lists to actual lists (e.g., "2,6" -> [2, 6])
        for key in ['engram_layers', 'engram_ngram_orders']:
            if key in config_dict and isinstance(config_dict[key], str):
                config_dict[key] = [int(x.strip()) for x in config_dict[key].split(',') if x.strip()]

        # Handle config parameters that have changed defaults between versions
        # If not explicitly set in checkpoint, use safe defaults to match training
        if 'use_value_embeds' not in config_dict:
            config_dict['use_value_embeds'] = False  # Feature didn't exist in older checkpoints

        # Disable Triton kernels and cuDNN for CPU inference (they only work on CUDA)
        if device == 'cpu' or device == torch.device('cpu'):
            config_dict['use_triton_kernels'] = False
            config_dict['use_triton_mla'] = False
            config_dict['use_cudnn_sdpa'] = False  # cuDNN SDPA only works on CUDA
            config_dict['attn_impl'] = 'sdpa'  # Force standard SDPA for CPU

        # Rename gradient_checkpointing to use_gradient_checkpointing if needed
        if 'gradient_checkpointing' in config_dict:
            config_dict['use_gradient_checkpointing'] = config_dict.pop('gradient_checkpointing')

        # Force use_fp8 to False for inference to avoid dtype issues
        # FP8 is for training only, we'll convert everything to the target dtype
        if 'use_fp8' in config_dict:
            config_dict['use_fp8'] = False
            print("Forcing use_fp8=False for inference (FP8 is training-only)")

        # Extract dimensions from state_dict to override preset values
        # This ensures the model matches the actual checkpoint shapes
        if "model" in checkpoint or "model_state_dict" in checkpoint:
            state_dict_for_inspection = checkpoint.get("model") or checkpoint.get("model_state_dict")
            # Remove prefixes for easier key lookup
            def get_key(base_key):
                for prefix in ["_orig_mod.", "module.", ""]:
                    key = prefix + base_key
                    if key in state_dict_for_inspection:
                        return key
                return None

            # Extract vocab_size from embedding
            wte_key = get_key("transformer.wte.weight")
            if wte_key:
                vocab_size = state_dict_for_inspection[wte_key].shape[0]
                config_dict['vocab_size'] = vocab_size
                print(f"Extracted vocab_size={vocab_size} from checkpoint")

            # Extract expert_dim from MoE layer if present
            # SwiGLU: gate_up_proj has shape [expert_dim * 2, n_embd]
            expert_key = get_key("transformer.h.2.ffn.shared_experts.0.gate_up_proj.weight")
            if expert_key:
                gate_up_dim = state_dict_for_inspection[expert_key].shape[0]
                expert_dim = gate_up_dim // 2  # SwiGLU doubles the dimension
                config_dict['expert_dim'] = expert_dim
                print(f"Extracted expert_dim={expert_dim} from checkpoint (gate_up_dim={gate_up_dim})")

        # Create model using the appropriate method
        if model_size:
            # Use create_swa_mla_model() with size preset
            # Remove 'size' from config_dict as it's passed separately
            config_dict.pop('size', None)
            print(f"Creating model with size preset: {model_size}")
            model = create_swa_mla_model(size=model_size, **config_dict)
        else:
            # Direct config creation (fallback)
            # Need to infer n_layer and n_embd from the actual checkpoint
            state_dict_ref = checkpoint.get("model") or checkpoint.get("model_state_dict")
            if state_dict_ref:
                # Try to infer n_layer from state dict keys
                layer_keys = [k for k in state_dict_ref.keys() if "transformer.h." in k]
                if layer_keys:
                    # Extract layer indices
                    layer_indices = set()
                    for key in layer_keys:
                        parts = key.split(".")
                        for i, part in enumerate(parts):
                            if part == "h" and i + 1 < len(parts):
                                try:
                                    layer_indices.add(int(parts[i + 1]))
                                except ValueError:
                                    pass
                    if layer_indices:
                        n_layer = max(layer_indices) + 1
                        config_dict['n_layer'] = n_layer
                        print(f"Inferred n_layer={n_layer} from checkpoint")

                # Try to infer n_embd from embedding weights
                wte_key = None
                for prefix in ["_orig_mod.", "module.", ""]:
                    candidate = prefix + "transformer.wte.weight"
                    if candidate in state_dict_ref:
                        wte_key = candidate
                        break
                if wte_key:
                    n_embd = state_dict_ref[wte_key].shape[1]
                    vocab_size = state_dict_ref[wte_key].shape[0]
                    config_dict['n_embd'] = n_embd
                    config_dict['vocab_size'] = vocab_size
                    print(f"Inferred n_embd={n_embd}, vocab_size={vocab_size} from checkpoint")

            print(f"Creating model with config: {list(config_dict.keys())}")
            config = SWAMLAConfig(**config_dict)
            model = SWAMLAModel(config)
    else:
        raise ValueError("Checkpoint must contain 'config' key")

    # Load state dict — prefer EMA weights if available (they match validation loss)
    has_ema = use_ema and 'ema' in checkpoint and checkpoint['ema'] is not None
    if has_ema:
        print("EMA weights found in checkpoint — using EMA weights for inference")
        ema_state = checkpoint['ema']
        ema_params = ema_state.get('ema_params', {})

        # Start with the base model state dict, then overlay EMA params
        if "model" in checkpoint:
            state_dict = checkpoint["model"]
        elif "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint

        # Overlay EMA parameters onto base state dict
        # EMA keys may lack prefixes (_orig_mod., module.) that are in the state_dict
        # from torch.compile or DDP wrapping. Build a mapping to handle this.
        def _strip_prefixes(key: str) -> str:
            """Strip _orig_mod. and module. prefixes from a key."""
            for prefix in ("_orig_mod.", "module."):
                if key.startswith(prefix):
                    key = key[len(prefix):]
            return key

        # Build reverse mapping: stripped_key -> original state_dict key
        stripped_to_sd_key = {}
        for sd_key in state_dict:
            stripped_to_sd_key[_strip_prefixes(sd_key)] = sd_key

        ema_count = 0
        for ema_name, tensor in ema_params.items():
            ema_stripped = _strip_prefixes(ema_name)
            if ema_name in state_dict:
                # Direct match (same prefix)
                state_dict[ema_name] = tensor
                ema_count += 1
            elif ema_stripped in stripped_to_sd_key:
                # Match after stripping prefixes
                state_dict[stripped_to_sd_key[ema_stripped]] = tensor
                ema_count += 1
        print(f"  Applied {ema_count} EMA parameters over base state dict")
        if ema_count == 0 and len(ema_params) > 0:
            # Show sample keys to help diagnose prefix mismatch
            ema_sample = list(ema_params.keys())[:3]
            sd_sample = list(state_dict.keys())[:3]
            print(f"  WARNING: EMA has {len(ema_params)} params but 0 matched!")
            print(f"  EMA key samples: {ema_sample}")
            print(f"  State dict key samples: {sd_sample}")
    else:
        if "model" in checkpoint:
            state_dict = checkpoint["model"]
        elif "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint

    # Remove DDP wrapper prefix if present
    if any(key.startswith("module.") for key in state_dict.keys()):
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    # Remove torch.compile wrapper prefix if present (_orig_mod.)
    if any(key.startswith("_orig_mod.") for key in state_dict.keys()):
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
        print("Removed _orig_mod. prefix from compiled model state dict")

    # IMPORTANT: Convert state_dict tensors to target dtype BEFORE loading
    # This prevents dtype mismatch issues where checkpoint has float16 but we want bfloat16
    print(f"Converting state_dict to {torch_dtype}...")
    for key in state_dict.keys():
        if state_dict[key].dtype.is_floating_point:
            state_dict[key] = state_dict[key].to(dtype=torch_dtype)

    # Load the converted state dict with strict=False to handle version differences
    load_result = model.load_state_dict(state_dict, strict=False)
    if load_result.missing_keys:
        print(f"Note: {len(load_result.missing_keys)} missing keys (new features not in checkpoint)")
        for key in load_result.missing_keys[:5]:
            print(f"  - {key}")
        if len(load_result.missing_keys) > 5:
            print(f"  ... and {len(load_result.missing_keys) - 5} more")
    if load_result.unexpected_keys:
        print(f"Warning: {len(load_result.unexpected_keys)} unexpected keys in checkpoint")

    # Move model to device and dtype (should already be correct dtype, but ensures device is set)
    model = model.to(device=device, dtype=torch_dtype)

    # Restore FoPE/RoPE buffers to float32 (they lose precision from model.to(bf16))
    _restore_precision_buffers(model, device)

    # Final verification: check all parameters and buffers are in correct dtype
    # Note: FoPE/RoPE buffers are intentionally float32 for numerical precision
    mismatched_params = []
    for name, param in model.named_parameters():
        if param.dtype != torch_dtype:
            mismatched_params.append((name, param.dtype))

    mismatched_buffers = []
    precision_buffer_names = {'inv_freq', 'harmonic_mult', 'cos_cached', 'sin_cached'}
    for name, buffer in model.named_buffers():
        if buffer is not None and buffer.dtype.is_floating_point and buffer.dtype != torch_dtype:
            # Skip intentionally float32 precision buffers
            if name.split('.')[-1] not in precision_buffer_names:
                mismatched_buffers.append((name, buffer.dtype))

    if mismatched_params or mismatched_buffers:
        print(f"WARNING: Found mismatched dtypes after conversion:")
        for name, dtype in mismatched_params[:5]:
            print(f"  Parameter {name}: {dtype} (expected {torch_dtype})")
        for name, dtype in mismatched_buffers[:5]:
            print(f"  Buffer {name}: {dtype} (expected {torch_dtype})")

    model.eval()
    print(f"Model loaded and moved to {device} with dtype {torch_dtype}")

    # Load tokenizer (assume GPT-2 tokenizer if not specified)
    tokenizer_name = checkpoint.get("tokenizer_name", "gpt2")
    print(f"Loading tokenizer: {tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Set tokenizer compression mapping for Engram if the model uses it
    if hasattr(model, 'set_engram_tokenizer_compression') and config_dict.get('use_engram', False):
        model.set_engram_tokenizer_compression(tokenizer)
        print("Engram tokenizer compression mapping initialized")

    print(f"Model loaded successfully ({model.param_count / 1e6:.2f}M parameters)")
    return model, tokenizer


# Pre-registered prompts for batch mode
DEFAULT_PROMPTS = [
    "Once upon a time, in a land far away,",
    "The future of artificial intelligence is",
    "In a world where technology has advanced beyond our wildest dreams,",
    "The most important lesson I learned was",
    "Science and magic are not as different as you might think.",
]


def batch_mode(
    engine: InferenceEngine,
    prompts: Optional[List[str]] = None,
    max_new_tokens: int = 256,
    temperature: float = 0.8,
    top_k: Optional[int] = 50,
    top_p: Optional[float] = 0.9,
    repetition_penalty: float = 1.1,
    debug: bool = False,
    do_sample: bool = True,
):
    """Run batch generation on pre-registered prompts.

    Args:
        engine: Inference engine
        prompts: List of prompts (uses DEFAULT_PROMPTS if None)
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_k: Top-k sampling parameter
        top_p: Top-p (nucleus) sampling parameter
        repetition_penalty: Penalty for repeating tokens (>1.0 = less repetition)
        debug: Print top-5 token probabilities for first tokens
        do_sample: If False, use greedy decoding (argmax)
    """
    if prompts is None:
        prompts = DEFAULT_PROMPTS

    print("\n" + "="*80)
    print("BATCH MODE - Generating responses for pre-registered prompts")
    print("="*80 + "\n")

    for i, prompt in enumerate(prompts, 1):
        print(f"\n[Prompt {i}/{len(prompts)}]")
        print(f"Input: {prompt}")
        print("-" * 80)

        # Generate
        output = engine.generate(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            debug=debug,
            return_full_text=False,
            do_sample=do_sample,
        )
        print(f"Generated: {output}")
        print("="*80)


def chat_mode(
    engine: InferenceEngine,
    max_new_tokens: int = 256,
    temperature: float = 0.8,
    top_k: Optional[int] = 50,
    top_p: Optional[float] = 0.9,
    repetition_penalty: float = 1.1,
    system_prompt: Optional[str] = None,
    use_chatml: bool = False,
    do_sample: bool = True,
):
    """Interactive chat mode with the model.

    Args:
        engine: Inference engine
        max_new_tokens: Maximum tokens to generate per response
        temperature: Sampling temperature
        top_k: Top-k sampling parameter
        top_p: Top-p (nucleus) sampling parameter
        repetition_penalty: Penalty for repeating tokens (>1.0 = less repetition)
        system_prompt: Optional system prompt to prepend to conversation
        use_chatml: Whether to use ChatML format (for instruction-tuned models)
        do_sample: If False, use greedy decoding (argmax)
    """
    print("\n" + "="*80)
    print("CHAT MODE - Interactive conversation with the model")
    if use_chatml:
        print("Format: ChatML (instruction-tuned model)")
    print("="*80)
    print("\nCommands:")
    print("  /quit or /exit - Exit chat mode")
    print("  /clear - Clear conversation history")
    print("  /temp <value> - Change temperature (e.g., /temp 0.7)")
    print("  /topk <value> - Change top_k (e.g., /topk 40)")
    print("  /topp <value> - Change top_p (e.g., /topp 0.95)")
    print("  /help - Show this help message")
    print("\n" + "="*80 + "\n")

    # Initialize conversation history
    conversation_history = ""
    if system_prompt is None and use_chatml:
        system_prompt = "You are a helpful assistant."

    if system_prompt:
        if use_chatml:
            conversation_history = f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        else:
            conversation_history = system_prompt + "\n\n"
        print(f"[System prompt set: {system_prompt}]\n")

    # Chat loop
    while True:
        try:
            # Get user input
            user_input = input("You: ").strip()

            if not user_input:
                continue

            # Handle commands
            if user_input.startswith("/"):
                cmd_parts = user_input.split(maxsplit=1)
                cmd = cmd_parts[0].lower()

                if cmd in ["/quit", "/exit"]:
                    print("Exiting chat mode...")
                    break

                elif cmd == "/clear":
                    conversation_history = ""
                    if system_prompt:
                        if use_chatml:
                            conversation_history = f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
                        else:
                            conversation_history = system_prompt + "\n\n"
                    print("[Conversation history cleared]\n")
                    continue

                elif cmd == "/temp":
                    if len(cmd_parts) > 1:
                        try:
                            temperature = float(cmd_parts[1])
                            print(f"[Temperature set to {temperature}]\n")
                        except ValueError:
                            print("[Error: Invalid temperature value]\n")
                    else:
                        print(f"[Current temperature: {temperature}]\n")
                    continue

                elif cmd == "/topk":
                    if len(cmd_parts) > 1:
                        try:
                            top_k = int(cmd_parts[1])
                            print(f"[Top-k set to {top_k}]\n")
                        except ValueError:
                            print("[Error: Invalid top_k value]\n")
                    else:
                        print(f"[Current top_k: {top_k}]\n")
                    continue

                elif cmd == "/topp":
                    if len(cmd_parts) > 1:
                        try:
                            top_p = float(cmd_parts[1])
                            print(f"[Top-p set to {top_p}]\n")
                        except ValueError:
                            print("[Error: Invalid top_p value]\n")
                    else:
                        print(f"[Current top_p: {top_p}]\n")
                    continue

                elif cmd == "/help":
                    print("\nCommands:")
                    print("  /quit or /exit - Exit chat mode")
                    print("  /clear - Clear conversation history")
                    print("  /temp <value> - Change temperature")
                    print("  /topk <value> - Change top_k")
                    print("  /topp <value> - Change top_p")
                    print("  /help - Show this help message\n")
                    continue

                else:
                    print(f"[Unknown command: {cmd}. Type /help for available commands]\n")
                    continue

            # Add user input to conversation history
            if use_chatml:
                conversation_history += f"<|im_start|>user\n{user_input}<|im_end|>\n"
                prompt = conversation_history + "<|im_start|>assistant\n"
            else:
                conversation_history += f"You: {user_input}\n"
                prompt = conversation_history + "Assistant:"

            # Prepare stop tokens for ChatML
            stop_tokens_list = None
            if use_chatml:
                # Get <|im_end|> token ID
                im_end_id = engine.tokenizer.convert_tokens_to_ids("<|im_end|>")
                if im_end_id != engine.tokenizer.unk_token_id:
                    stop_tokens_list = [im_end_id]

            # Generate response
            output = engine.generate(
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                stop_tokens=stop_tokens_list,
                return_full_text=False,
                do_sample=do_sample,
            )

            # Extract assistant response
            if use_chatml:
                # Extract until <|im_end|>
                assistant_response = output.strip()
                if "<|im_end|>" in assistant_response:
                    assistant_response = assistant_response.split("<|im_end|>")[0].strip()
            else:
                assistant_response = output.strip()
                if assistant_response.startswith("Assistant:"):
                    assistant_response = assistant_response[len("Assistant:"):].strip()

                # Stop at next "You:" if model hallucinates continuation
                if "\nYou:" in assistant_response:
                    assistant_response = assistant_response.split("\nYou:")[0].strip()

            print(f"Assistant: {assistant_response}\n")

            # Add assistant response to history
            if use_chatml:
                conversation_history += f"<|im_start|>assistant\n{assistant_response}<|im_end|>\n"
            else:
                conversation_history += f"Assistant: {assistant_response}\n"

        except KeyboardInterrupt:
            print("\n\n[Interrupted. Type /quit to exit or continue chatting]\n")
            continue
        except Exception as e:
            print(f"\n[Error: {e}]\n")
            continue


@torch.inference_mode()
def _eval_perplexity(engine: InferenceEngine, text: str):
    """Compute and print perplexity of the model on given text.

    Args:
        engine: Inference engine with loaded model
        text: Text to evaluate perplexity on
    """
    import math

    print("\n" + "=" * 80)
    print("PERPLEXITY EVALUATION")
    print("=" * 80)

    input_ids = engine.tokenizer.encode(text, return_tensors="pt").to(engine.device)
    n_tokens = input_ids.size(1)
    print(f"Text length: {n_tokens} tokens")

    if n_tokens < 2:
        print("Text too short for perplexity evaluation (need >= 2 tokens)")
        return

    # Truncate to model's block_size
    block_size = engine.model.config.block_size
    if n_tokens > block_size:
        print(f"Truncating to block_size={block_size}")
        input_ids = input_ids[:, :block_size]
        n_tokens = block_size

    with torch.amp.autocast(engine.device, dtype=torch.bfloat16, enabled=(engine.use_autocast and engine.device == 'cuda')):
        logits, _ = engine.model(input_ids, return_all_logits=True)

    # Shift: predict token t+1 from position t
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()

    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
    )
    perplexity = math.exp(loss.item())

    print(f"Cross-entropy loss: {loss.item():.4f}")
    print(f"Perplexity: {perplexity:.2f}")
    print("=" * 80 + "\n")


@torch.inference_mode()
def run_diagnostics(model: SWAMLAModel, tokenizer, device: str = "cuda"):
    """Run diagnostics to verify model precision and quality.

    Prints:
    1. Full reconstructed SWAMLAConfig
    2. Parameter/buffer dtypes for key layers
    3. Weight statistics (mean, std, min, max, NaN/Inf check)
    4. Perplexity on a test sentence
    5. Logit divergence: pure bf16 vs autocast

    Args:
        model: Loaded SWA-MLA model.
        tokenizer: Tokenizer for the model.
        device: Device string.
    """
    import math
    from dataclasses import fields

    print("\n" + "=" * 80)
    print("DIAGNOSTICS MODE")
    print("=" * 80)

    # 1. Full config
    print("\n--- Model Configuration ---")
    config = model.config
    for f in fields(config):
        print(f"  {f.name}: {getattr(config, f.name)}")

    # 2. Parameter/buffer dtypes for key layers
    print("\n--- Key Layer Dtypes ---")
    key_patterns = [
        'transformer.wte.weight',
        'transformer.ln_f.weight',
        'lm_head.weight',
    ]
    # Also find first MLA and DeltaNet block params
    for name, param in model.named_parameters():
        short = name.split('.')
        if len(short) >= 4 and short[1] == 'h' and short[2] == '0':
            key_patterns.append(name)
            if len(key_patterns) > 12:
                break

    seen = set()
    for name, param in model.named_parameters():
        for pattern in key_patterns:
            if name == pattern and name not in seen:
                seen.add(name)
                print(f"  [param] {name}: {param.dtype}, shape={tuple(param.shape)}")

    for name, buf in model.named_buffers():
        if buf is not None:
            dtype_str = str(buf.dtype)
            if 'complex' in dtype_str or buf.dtype == torch.float32:
                print(f"  [buffer] {name}: {buf.dtype}, shape={tuple(buf.shape)}")

    # 3. Weight statistics
    print("\n--- Weight Statistics (critical params) ---")
    critical_params = []
    for name, param in model.named_parameters():
        if any(k in name for k in ['wte.weight', 'lm_head.weight', 'ln_f.weight',
                                     'w_qkv', 'w_o', 'gate_up_proj', 'down_proj']):
            critical_params.append((name, param))
            if len(critical_params) >= 10:
                break

    has_issues = False
    for name, param in critical_params:
        data = param.float()
        nan_count = torch.isnan(data).sum().item()
        inf_count = torch.isinf(data).sum().item()
        flag = ""
        if nan_count > 0 or inf_count > 0:
            flag = " *** ISSUE ***"
            has_issues = True
        print(f"  {name}: mean={data.mean().item():.6f}, std={data.std().item():.6f}, "
              f"min={data.min().item():.6f}, max={data.max().item():.6f}, "
              f"NaN={nan_count}, Inf={inf_count}{flag}")

    if not has_issues:
        print("  All checked parameters look healthy (no NaN/Inf).")

    # 3b. Residual scalars (if present — key diagnostic for weight collapse)
    if hasattr(model, 'resid_lambdas') and model.resid_lambdas is not None:
        print("\n--- Residual Scalars ---")
        resid = model.resid_lambdas.float()
        x0 = model.x0_lambdas.float()
        print(f"  resid_lambdas: shape={tuple(resid.shape)}")
        for i in range(resid.shape[0]):
            print(f"    layer {i:2d}: resid={resid[i].item():.6f}, x0={x0[i].item():.6f}, "
                  f"ratio(resid/x0)={resid[i].item() / (x0[i].item() + 1e-10):.4f}")
    else:
        print("\n--- Residual Scalars ---")
        print("  Not present (use_residual_scalars=False or not trained)")

    # 4. Perplexity on test sentence
    print("\n--- Test Perplexity ---")
    test_text = "The quick brown fox jumps over the lazy dog. In a world where technology advances rapidly, humans must adapt to constant change."
    input_ids = tokenizer.encode(test_text, return_tensors="pt").to(device)
    n_tokens = input_ids.size(1)
    shift_labels = input_ids[:, 1:].contiguous()

    # With autocast (matches training) — need return_all_logits for perplexity
    with torch.amp.autocast(device, dtype=torch.bfloat16, enabled=(device == 'cuda')):
        logits_ac_full, _ = model(input_ids, return_all_logits=True)
    shift_logits = logits_ac_full[:, :-1, :].float().contiguous()
    loss_ac = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    ppl_ac = math.exp(loss_ac.item())
    print(f"  Autocast (bf16):  loss={loss_ac.item():.4f}, perplexity={ppl_ac:.2f} ({n_tokens} tokens)")

    # Without autocast (pure bf16)
    logits_bf_full, _ = model(input_ids, return_all_logits=True)
    shift_logits_bf = logits_bf_full[:, :-1, :].float().contiguous()
    loss_bf = F.cross_entropy(shift_logits_bf.view(-1, shift_logits_bf.size(-1)), shift_labels.view(-1))
    ppl_bf = math.exp(loss_bf.item())
    print(f"  Pure bf16:        loss={loss_bf.item():.4f}, perplexity={ppl_bf:.2f} ({n_tokens} tokens)")

    # 5. Logit divergence (compare last-token logits, which is what generate() uses)
    print("\n--- Logit Divergence (autocast vs pure bf16) ---")
    logits_ac_f = logits_ac_full[:, -1, :].float()
    logits_bf_f = logits_bf_full[:, -1, :].float()

    abs_diff = (logits_ac_f - logits_bf_f).abs()
    rel_diff = abs_diff / (logits_ac_f.abs() + 1e-8)

    print(f"  Absolute diff: mean={abs_diff.mean().item():.6f}, max={abs_diff.max().item():.6f}")
    print(f"  Relative diff: mean={rel_diff.mean().item():.6f}, max={rel_diff.max().item():.6f}")

    # Compare top-k predictions
    probs_ac = F.softmax(logits_ac_f, dim=-1)
    probs_bf = F.softmax(logits_bf_f, dim=-1)
    top5_ac = torch.topk(probs_ac, 5, dim=-1)
    top5_bf = torch.topk(probs_bf, 5, dim=-1)

    print(f"\n  Top-5 predictions (autocast):")
    for i in range(5):
        tok = tokenizer.decode([top5_ac.indices[0, i].item()])
        print(f"    {i+1}. '{tok}' ({top5_ac.values[0, i].item():.4f})")

    print(f"  Top-5 predictions (pure bf16):")
    for i in range(5):
        tok = tokenizer.decode([top5_bf.indices[0, i].item()])
        print(f"    {i+1}. '{tok}' ({top5_bf.values[0, i].item():.4f})")

    # KL divergence between the two distributions
    kl_div = F.kl_div(probs_bf.log().clamp(min=-100), probs_ac, reduction='sum').item()
    print(f"\n  KL divergence (bf16 || autocast): {kl_div:.6f}")

    if abs_diff.mean().item() > 0.1:
        print("\n  ** Significant divergence detected. Autocast is recommended. **")
    else:
        print("\n  Divergence is minimal. Both modes should produce similar quality.")

    print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(description="SWA-MLA Inference Script")

    # Model loading options
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument("--hf_repo_id", type=str, help="Hugging Face model repo ID (e.g., 'username/swamla-model')")
    model_group.add_argument("--checkpoint", type=str, help="Path to local checkpoint file")

    # HuggingFace specific options
    parser.add_argument("--hf_checkpoint", type=str, default=None,
                        help="Specific checkpoint folder name (e.g., 'checkpoint_tokens_500k_loss_2.3456'). If not specified, loads the latest checkpoint automatically.")
    parser.add_argument("--hf_checkpoint_strategy", type=str, default="latest", choices=["latest", "best_loss"],
                        help="Checkpoint selection strategy when --hf_checkpoint is not specified")
    parser.add_argument("--nb_tok", type=str, default=None,
                        help="Target token count for checkpoint selection (e.g., '3.5B', '500M', '100k'). Selects the closest checkpoint.")

    # Inference mode
    parser.add_argument("--mode", type=str, choices=["batch", "chat"], default="chat",
                        help="Inference mode: 'batch' for pre-registered prompts, 'chat' for interactive")

    # Generation parameters
    parser.add_argument("--max_new_tokens", type=int, default=256,
                        help="Maximum number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="Sampling temperature (higher = more random)")
    parser.add_argument("--top_k", type=int, default=50,
                        help="Top-k sampling parameter (0 to disable)")
    parser.add_argument("--top_p", type=float, default=0.9,
                        help="Top-p (nucleus) sampling parameter")
    parser.add_argument("--repetition_penalty", type=float, default=1.1,
                        help="Repetition penalty (>1.0 = less repetition)")
    parser.add_argument("--greedy", action="store_true",
                        help="Use greedy decoding (disables sampling)")

    # Chat mode specific
    parser.add_argument("--system_prompt", type=str, default=None,
                        help="System prompt for chat mode")
    parser.add_argument("--chatml", action="store_true",
                        help="Use ChatML format (for instruction-tuned models)")

    # Batch mode specific
    parser.add_argument("--prompts_file", type=str, default=None,
                        help="File containing prompts (one per line) for batch mode")

    # Diagnostics
    parser.add_argument("--debug", action="store_true",
                        help="Print top-5 token probabilities for first tokens (diagnose model quality)")
    parser.add_argument("--eval_perplexity", type=str, default=None,
                        help="Compute perplexity on the given text or file path before generating")

    # Device options
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to run inference on")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float32", "float16", "bfloat16"],
                        help="Data type for model weights")
    parser.add_argument("--max_length", type=int, default=2048,
                        help="Maximum sequence length")

    # Precision control
    parser.add_argument("--no_autocast", action="store_true",
                        help="Disable autocast (pure bf16 inference, may degrade quality)")
    parser.add_argument("--no_ema", action="store_true",
                        help="Skip EMA weights even if available in local checkpoint (use raw model weights)")
    parser.add_argument("--diagnose", action="store_true",
                        help="Run diagnostics: print config, dtypes, weight stats, and precision comparison")

    args = parser.parse_args()

    # Convert dtype string to torch dtype
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    torch_dtype = dtype_map[args.dtype]

    # Check device availability
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = "cpu"
        if args.dtype == "bfloat16":
            print("BFloat16 not well supported on CPU, using float32")
            torch_dtype = torch.float32

    # Load model
    if args.hf_repo_id:
        # Get HF token from environment if available
        hf_token = os.getenv("HF_TOKEN")

        target_tokens = _parse_token_count(args.nb_tok) if args.nb_tok else None

        model, tokenizer = load_model_from_hf(
            repo_id=args.hf_repo_id,
            checkpoint_name=args.hf_checkpoint,
            checkpoint_strategy=args.hf_checkpoint_strategy,
            target_tokens=target_tokens,
            device=args.device,
            torch_dtype=torch_dtype,
            hf_token=hf_token,
        )
    else:
        model, tokenizer = load_model_from_checkpoint(
            checkpoint_path=args.checkpoint,
            device=args.device,
            torch_dtype=torch_dtype,
            use_ema=not args.no_ema,
        )

    # Create inference engine
    engine = InferenceEngine(
        model=model,
        tokenizer=tokenizer,
        device=args.device,
        max_length=args.max_length,
        use_autocast=not args.no_autocast,
    )

    # Run diagnostics if requested
    if args.diagnose:
        run_diagnostics(model, tokenizer, args.device)
        return

    # Evaluate perplexity if requested
    if args.eval_perplexity:
        eval_text = args.eval_perplexity
        # If it looks like a file path, read the file
        if Path(eval_text).is_file():
            eval_text = Path(eval_text).read_text()
        _eval_perplexity(engine, eval_text)

    # Run inference mode
    if args.mode == "batch":
        # Load prompts from file if specified
        prompts = None
        if args.prompts_file:
            with open(args.prompts_file, 'r') as f:
                prompts = [line.strip() for line in f if line.strip() and not line.strip().startswith('#')]
            print(f"Loaded {len(prompts)} prompts from {args.prompts_file}")

        batch_mode(
            engine=engine,
            prompts=prompts,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k if args.top_k > 0 else None,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            debug=args.debug,
            do_sample=not args.greedy,
        )

    elif args.mode == "chat":
        chat_mode(
            engine=engine,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k if args.top_k > 0 else None,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            system_prompt=args.system_prompt,
            use_chatml=args.chatml,
            do_sample=not args.greedy,
        )


if __name__ == "__main__":
    main()
