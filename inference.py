"""
Inference script for SWA-MLA model.
Supports loading from Hugging Face and two modes:
- Batch mode: Generate responses for pre-registered prompts
- Chat mode: Interactive conversation with the model
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

from swa_mla_model import SWAMLAModel, SWAMLAConfig


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


def load_model_from_hf(
    model_name_or_path: str,
    device: str = "cuda",
    torch_dtype: torch.dtype = torch.bfloat16,
) -> tuple[SWAMLAModel, AutoTokenizer]:
    """Load SWA-MLA model and tokenizer from Hugging Face.

    Args:
        model_name_or_path: HF model repo ID or local path
        device: Device to load model on
        torch_dtype: Data type for model weights

    Returns:
        Tuple of (model, tokenizer)
    """
    from huggingface_hub import hf_hub_download
    import json

    print(f"Loading model from {model_name_or_path}...")

    # Download config
    config_path = hf_hub_download(repo_id=model_name_or_path, filename="config.json")
    with open(config_path, 'r') as f:
        config_dict = json.load(f)

    # Create model config
    config = SWAMLAConfig(**config_dict)

    # Create model
    model = SWAMLAModel(config)

    # Download and load weights
    weights_path = hf_hub_download(repo_id=model_name_or_path, filename="pytorch_model.bin")
    state_dict = torch.load(weights_path, map_location="cpu")

    # Remove DDP wrapper prefix if present
    if any(key.startswith("module.") for key in state_dict.keys()):
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)
    model = model.to(device=device, dtype=torch_dtype)
    model.eval()

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

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

    # Allow TorchAO optimizer states in checkpoint
    # In PyTorch 2.6+, weights_only=True by default, but training checkpoints
    # contain optimizer states that need weights_only=False

    # Create a mock OptimStateFp8 class if TorchAO is not installed
    # This allows loading checkpoints without needing TorchAO for inference
    try:
        from torchao.optim.subclass_fp8 import OptimStateFp8
        print("TorchAO detected - using native OptimStateFp8")
    except ImportError:
        print("TorchAO not installed - creating mock OptimStateFp8 for checkpoint loading")
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
            # This is required because pickle will try to reconstruct the optimizer state
            class OptimStateFp8(torch.Tensor):
                """Mock OptimStateFp8 for loading checkpoints without TorchAO."""

                @staticmethod
                def __new__(cls, data, *args, **kwargs):
                    # Create a tensor and wrap it as this subclass
                    if isinstance(data, torch.Tensor):
                        return data.as_subclass(cls)
                    return torch.as_tensor(data).as_subclass(cls)

                def __init__(self, *args, **kwargs):
                    # Initialization is handled by __new__
                    pass

                @classmethod
                def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
                    """Required for torch.Tensor subclasses."""
                    kwargs = kwargs or {}

                    # Unwrap all OptimStateFp8 tensors to regular tensors
                    def unwrap(x):
                        if isinstance(x, OptimStateFp8):
                            return x.as_subclass(torch.Tensor)
                        return x

                    # Recursively unwrap args
                    def unwrap_args(a):
                        if isinstance(a, (list, tuple)):
                            return type(a)(unwrap_args(x) for x in a)
                        return unwrap(a)

                    args = unwrap_args(args)
                    kwargs = {k: unwrap_args(v) for k, v in kwargs.items()}

                    # Call the original function
                    result = func(*args, **kwargs)
                    return result

            subclass_fp8_module.OptimStateFp8 = OptimStateFp8

    # Load checkpoint with weights_only=False to allow optimizer states
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Extract config
    if "config" in checkpoint:
        config_dict = checkpoint["config"]

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
            'size', 'batch_size', 'max_iters', 'learning_rate', 'min_lr',
            'weight_decay', 'beta1', 'beta2', 'warmup_iters', 'grad_clip',
            'gradient_accumulation_steps', 'optimizer_type', 'enable_tf32',
            'disable_tf32', 'tokenizer_name', 'num_workers', 'output_dir',
            'log_interval', 'eval_interval', 'save_interval', 'wandb_project',
            'wandb_run_name', 'hf_repo_id', 'resume_from_hf', 'compile'
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

        # Extract model size from checkpoint if available
        # We need to infer n_layer and n_embd from the actual checkpoint
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
    else:
        raise ValueError("Checkpoint must contain 'config' key")

    # Create model
    model = SWAMLAModel(config)

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
):
    """Interactive chat mode with the model.

    Args:
        engine: Inference engine
        max_new_tokens: Maximum tokens to generate per response
        temperature: Sampling temperature
        top_k: Top-k sampling parameter
        top_p: Top-p (nucleus) sampling parameter
        system_prompt: Optional system prompt to prepend to conversation
    """
    print("\n" + "="*80)
    print("CHAT MODE - Interactive conversation with the model")
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
    if system_prompt:
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
            conversation_history += f"You: {user_input}\n"

            # Generate response
            prompt = conversation_history + "Assistant:"
            output = engine.generate(
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )

            # Extract assistant response
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
    model_group.add_argument("--hf_model", type=str, help="Hugging Face model repo ID")
    model_group.add_argument("--checkpoint", type=str, help="Path to local checkpoint file")

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
    if args.hf_model:
        model, tokenizer = load_model_from_hf(
            model_name_or_path=args.hf_model,
            device=args.device,
            torch_dtype=torch_dtype,
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
        )


if __name__ == "__main__":
    main()
