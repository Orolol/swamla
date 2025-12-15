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
    ):
        self.model = model.to(device)
        self.model.eval()
        self.tokenizer = tokenizer
        self.device = device
        self.max_length = max_length

        # Disable any global autocast that might interfere with inference
        if torch.is_autocast_enabled():
            print("WARNING: Global autocast is enabled, this may cause dtype issues")
        torch.set_autocast_enabled(False)

    @torch.no_grad()
    @torch.amp.autocast('cuda', enabled=False)  # Explicitly disable autocast for inference
    @torch.amp.autocast('cpu', enabled=False)   # Also disable CPU autocast
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.8,
        top_k: Optional[int] = 50,
        top_p: Optional[float] = 0.9,
        repetition_penalty: float = 1.1,
        stop_tokens: Optional[List[int]] = None,
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
        for _ in range(max_new_tokens):
            # Truncate to max context length
            context_ids = generated_ids if generated_ids.size(1) <= self.model.config.block_size else generated_ids[:, -self.model.config.block_size:]

            # Forward pass
            logits, _ = self.model(context_ids)
            logits = logits[:, -1, :]  # Get last token logits

            # Apply repetition penalty
            if repetition_penalty != 1.0:
                for token_id, count in token_counts.items():
                    logits[0, token_id] /= (repetition_penalty ** count)

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
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = -float("inf")

            # Sample from the filtered distribution
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

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
        return generated_text


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


def load_model_from_hf(
    repo_id: str,
    checkpoint_name: Optional[str] = None,
    device: str = "cuda",
    torch_dtype: torch.dtype = torch.bfloat16,
    hf_token: Optional[str] = None,
) -> tuple[SWAMLAModel, AutoTokenizer]:
    """Load SWA-MLA model and tokenizer from Hugging Face.

    Args:
        repo_id: HF model repo ID (e.g., "username/swamla-model")
        checkpoint_name: Optional specific checkpoint folder name (e.g., "checkpoint_tokens_500k_loss_2.3456")
                        If None, loads the latest checkpoint automatically
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

        # Sort by total tokens (most recent training)
        checkpoints.sort(key=lambda x: x['total_tokens'], reverse=True)
        checkpoint_name = checkpoints[0]['name']

        print(f"Found {len(checkpoints)} checkpoints")
        print(f"Loading latest: {checkpoint_name} (tokens: {checkpoints[0]['total_tokens']:,}, loss: {checkpoints[0]['loss']:.4f})")
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

        # Remove training-specific parameters (but keep model architecture params)
        training_only_params = [
            'batch_size', 'max_iters', 'learning_rate', 'min_lr',
            'weight_decay', 'beta1', 'beta2', 'warmup_iters', 'grad_clip',
            'gradient_accumulation_steps', 'optimizer_type', 'enable_tf32',
            'disable_tf32', 'tokenizer_name', 'num_workers', 'output_dir',
            'log_interval', 'eval_interval', 'save_interval', 'wandb_project',
            'wandb_run_name', 'hf_repo_id', 'resume_from_hf', 'compile',
            'mla_q_lora_rank', 'mla_kv_lora_rank', 'mla_qk_nope_head_dim',
            'mla_qk_rope_head_dim', 'mla_v_head_dim',
        ]

        for param in training_only_params:
            training_config.pop(param, None)

        # Rename gradient_checkpointing to use_gradient_checkpointing if needed
        if 'gradient_checkpointing' in training_config:
            training_config['use_gradient_checkpointing'] = training_config.pop('gradient_checkpointing')

        # Force use_fp8 to False for inference
        training_config['use_fp8'] = False

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
        config = SWAMLAConfig(**config_dict)
        model = SWAMLAModel(config)

    model.load_state_dict(state_dict)
    model = model.to(device=device, dtype=torch_dtype)
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

    print(f"Model loaded successfully ({model.param_count / 1e6:.2f}M parameters)")
    return model, tokenizer


def load_model_from_checkpoint(
    checkpoint_path: str,
    device: str = "cuda",
    torch_dtype: torch.dtype = torch.bfloat16,
) -> tuple[SWAMLAModel, AutoTokenizer]:
    """Load SWA-MLA model from a local checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load model on
        torch_dtype: Data type for model weights

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
            'wandb_run_name', 'hf_repo_id', 'resume_from_hf', 'compile',
            # Neural memory training params
            'memory_reset_interval',
            # Other training-only params
            'use_tensorboard', 'use_te_fp8',
            # Deprecated/removed params
            'no_moe',
        ]

        for param in training_only_params:
            config_dict.pop(param, None)

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
        if "model" in checkpoint:
            state_dict_for_inspection = checkpoint["model"]
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
            if "model" in checkpoint:
                # Try to infer n_layer from state dict keys
                layer_keys = [k for k in checkpoint["model"].keys() if k.startswith("_orig_mod.transformer.h.")]
                if layer_keys:
                    # Extract layer indices
                    layer_indices = set()
                    for key in layer_keys:
                        parts = key.split(".")
                        if len(parts) > 3 and parts[2] == "h":
                            try:
                                layer_indices.add(int(parts[3]))
                            except ValueError:
                                pass
                    if layer_indices:
                        n_layer = max(layer_indices) + 1
                        config_dict['n_layer'] = n_layer
                        print(f"Inferred n_layer={n_layer} from checkpoint")

                # Try to infer n_embd from embedding weights
                wte_key = "_orig_mod.transformer.wte.weight"
                if wte_key in checkpoint["model"]:
                    n_embd = checkpoint["model"][wte_key].shape[1]
                    vocab_size = checkpoint["model"][wte_key].shape[0]
                    config_dict['n_embd'] = n_embd
                    config_dict['vocab_size'] = vocab_size
                    print(f"Inferred n_embd={n_embd}, vocab_size={vocab_size} from checkpoint")

            print(f"Creating model with config: {list(config_dict.keys())}")
            config = SWAMLAConfig(**config_dict)
            model = SWAMLAModel(config)
    else:
        raise ValueError("Checkpoint must contain 'config' key")

    # Load state dict
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

    # Load the converted state dict
    model.load_state_dict(state_dict)

    # Move model to device and dtype (should already be correct dtype, but ensures device is set)
    model = model.to(device=device, dtype=torch_dtype)

    # Final verification: check all parameters and buffers are in correct dtype
    mismatched_params = []
    for name, param in model.named_parameters():
        if param.dtype != torch_dtype:
            mismatched_params.append((name, param.dtype))

    mismatched_buffers = []
    for name, buffer in model.named_buffers():
        if buffer is not None and buffer.dtype.is_floating_point and buffer.dtype != torch_dtype:
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
):
    """Run batch generation on pre-registered prompts.

    Args:
        engine: Inference engine
        prompts: List of prompts (uses DEFAULT_PROMPTS if None)
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_k: Top-k sampling parameter
        top_p: Top-p (nucleus) sampling parameter
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
        )

        # Extract only the generated part (remove prompt)
        generated = output[len(prompt):]

        print(f"Generated: {generated}")
        print("="*80)


def chat_mode(
    engine: InferenceEngine,
    max_new_tokens: int = 256,
    temperature: float = 0.8,
    top_k: Optional[int] = 50,
    top_p: Optional[float] = 0.9,
    system_prompt: Optional[str] = None,
    use_chatml: bool = False,
):
    """Interactive chat mode with the model.

    Args:
        engine: Inference engine
        max_new_tokens: Maximum tokens to generate per response
        temperature: Sampling temperature
        top_k: Top-k sampling parameter
        top_p: Top-p (nucleus) sampling parameter
        system_prompt: Optional system prompt to prepend to conversation
        use_chatml: Whether to use ChatML format (for instruction-tuned models)
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
                stop_tokens=stop_tokens_list,
            )

            # Extract assistant response
            if use_chatml:
                # Remove the prompt and extract until <|im_end|>
                assistant_response = output[len(prompt):].strip()
                if "<|im_end|>" in assistant_response:
                    assistant_response = assistant_response.split("<|im_end|>")[0].strip()
            else:
                if "Assistant:" in output:
                    assistant_response = output.split("Assistant:")[-1].strip()
                else:
                    # Fallback: take everything after the prompt
                    assistant_response = output[len(prompt):].strip()

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


def main():
    parser = argparse.ArgumentParser(description="SWA-MLA Inference Script")

    # Model loading options
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument("--hf_repo_id", type=str, help="Hugging Face model repo ID (e.g., 'username/swamla-model')")
    model_group.add_argument("--checkpoint", type=str, help="Path to local checkpoint file")

    # HuggingFace specific options
    parser.add_argument("--hf_checkpoint", type=str, default=None,
                        help="Specific checkpoint folder name (e.g., 'checkpoint_tokens_500k_loss_2.3456'). If not specified, loads the latest checkpoint automatically.")

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

    # Chat mode specific
    parser.add_argument("--system_prompt", type=str, default=None,
                        help="System prompt for chat mode")
    parser.add_argument("--chatml", action="store_true",
                        help="Use ChatML format (for instruction-tuned models)")

    # Batch mode specific
    parser.add_argument("--prompts_file", type=str, default=None,
                        help="File containing prompts (one per line) for batch mode")

    # Device options
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to run inference on")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float32", "float16", "bfloat16"],
                        help="Data type for model weights")
    parser.add_argument("--max_length", type=int, default=2048,
                        help="Maximum sequence length")

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

        model, tokenizer = load_model_from_hf(
            repo_id=args.hf_repo_id,
            checkpoint_name=args.hf_checkpoint,
            device=args.device,
            torch_dtype=torch_dtype,
            hf_token=hf_token,
        )
    else:
        model, tokenizer = load_model_from_checkpoint(
            checkpoint_path=args.checkpoint,
            device=args.device,
            torch_dtype=torch_dtype,
        )

    # Create inference engine
    engine = InferenceEngine(
        model=model,
        tokenizer=tokenizer,
        device=args.device,
        max_length=args.max_length,
    )

    # Run inference mode
    if args.mode == "batch":
        # Load prompts from file if specified
        prompts = None
        if args.prompts_file:
            with open(args.prompts_file, 'r') as f:
                prompts = [line.strip() for line in f if line.strip()]
            print(f"Loaded {len(prompts)} prompts from {args.prompts_file}")

        batch_mode(
            engine=engine,
            prompts=prompts,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k if args.top_k > 0 else None,
            top_p=args.top_p,
        )

    elif args.mode == "chat":
        chat_mode(
            engine=engine,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k if args.top_k > 0 else None,
            top_p=args.top_p,
            system_prompt=args.system_prompt,
            use_chatml=args.chatml,
        )


if __name__ == "__main__":
    main()
