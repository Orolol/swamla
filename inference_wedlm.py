"""
WeDLM Inference script for SWA-MLA model.

Supports two generation modes:
- AR mode: Standard autoregressive generation (for WeDLM-trained models)
- Streaming mode: Streaming Parallel Decoding (WeDLM-specific, faster)

The streaming mode uses the WeDLM algorithm:
1. Fixed-size sliding window with [filled | mask] slots
2. Distance-adjusted entropy selection for parallel predictions
3. Immediate prefix commitment for KV cache validity
4. Dynamic refill to maintain constant workload

Examples:
    # Load from HuggingFace and use streaming decoding
    python inference_wedlm.py --hf_repo_id username/wedlm-model --mode streaming

    # Load from local checkpoint with AR decoding
    python inference_wedlm.py --checkpoint outputs/wedlm/checkpoint_1000.pt --mode ar

    # Interactive chat with streaming decoding
    python inference_wedlm.py --hf_repo_id username/wedlm-model --mode streaming --chat

    # Batch generation with custom prompts
    python inference_wedlm.py --checkpoint outputs/wedlm/latest.pt --prompts_file prompts.txt
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Optional, List, Dict, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

# Add models directory to path
sys.path.insert(0, str(Path(__file__).parent / 'models'))

from swa_mla_model import SWAMLAModel, SWAMLAConfig, create_swa_mla_model

# Import WeDLM components
try:
    from wedlm import WeDLMConfig, StreamingParallelDecoder, create_streaming_decoder
    WEDLM_AVAILABLE = True
except ImportError:
    WEDLM_AVAILABLE = False
    print("Warning: WeDLM module not available. Streaming decoding will be disabled.")


class WeDLMInferenceEngine:
    """Inference engine for WeDLM models with AR and streaming modes."""

    def __init__(
        self,
        model: SWAMLAModel,
        tokenizer: AutoTokenizer,
        wedlm_config: Optional[WeDLMConfig] = None,
        device: str = "cuda",
        max_length: int = 2048,
    ):
        self.model = model.to(device)
        self.model.eval()
        self.tokenizer = tokenizer
        self.device = device
        self.max_length = max_length

        # WeDLM config
        self.wedlm_config = wedlm_config or WeDLMConfig()

        # Determine mask token ID (priority: config > tokenizer > vocab_size-1)
        if self.wedlm_config.mask_token_id >= 0:
            self.mask_token_id = self.wedlm_config.mask_token_id
        elif hasattr(tokenizer, 'mask_token_id') and tokenizer.mask_token_id is not None:
            self.mask_token_id = tokenizer.mask_token_id
        else:
            # Use last token in vocabulary
            self.mask_token_id = len(tokenizer) - 1

        print(f"Using mask_token_id: {self.mask_token_id}")

        # Create streaming decoder
        self.streaming_decoder = None
        if WEDLM_AVAILABLE:
            self.wedlm_config.mask_token_id = self.mask_token_id
            self.streaming_decoder = create_streaming_decoder(
                model=self.model,
                config=self.wedlm_config,
                tokenizer=self.tokenizer,
            )

        # Disable autocast for inference
        torch.set_autocast_enabled(False)

    @torch.no_grad()
    def generate_ar(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.8,
        top_k: Optional[int] = 50,
        top_p: Optional[float] = 0.9,
        repetition_penalty: float = 1.1,
        stop_tokens: Optional[List[int]] = None,
    ) -> Tuple[str, Dict]:
        """
        Generate text using standard autoregressive decoding.

        Args:
            prompt: Input text prompt
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Nucleus sampling parameter
            repetition_penalty: Penalty for repeating tokens
            stop_tokens: List of token IDs that stop generation

        Returns:
            generated_text: Full generated text (prompt + continuation)
            metrics: Generation metrics
        """
        # Encode prompt
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)

        if input_ids.size(1) > self.max_length:
            print(f"Warning: Prompt length ({input_ids.size(1)}) exceeds max_length ({self.max_length}). Truncating.")
            input_ids = input_ids[:, -self.max_length:]

        # Track token frequencies for repetition penalty
        token_counts = {}
        tokens_generated = 0

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
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 0] = False
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = -float("inf")

            # Sample from the filtered distribution
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Update token counts for repetition penalty
            token_id = next_token.item()
            token_counts[token_id] = token_counts.get(token_id, 0) + 1
            tokens_generated += 1

            # Append to generated sequence
            generated_ids = torch.cat([generated_ids, next_token], dim=1)

            # Check for stop tokens
            if stop_tokens and token_id in stop_tokens:
                break

            # Check for EOS
            if token_id == self.tokenizer.eos_token_id:
                break

            # Check if we've hit max length
            if generated_ids.size(1) >= self.max_length:
                break

        # Decode generated text
        generated_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)

        metrics = {
            'mode': 'ar',
            'tokens_generated': tokens_generated,
            'total_forward_passes': tokens_generated,
            'speedup': 1.0,
        }

        return generated_text, metrics

    @torch.no_grad()
    def generate_streaming(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.8,
        top_k: Optional[int] = 50,
        top_p: Optional[float] = 0.9,
        verbose: bool = False,
    ) -> Tuple[str, Dict]:
        """
        Generate text using WeDLM streaming parallel decoding.

        Args:
            prompt: Input text prompt
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Nucleus sampling parameter
            verbose: Print progress during generation

        Returns:
            generated_text: Full generated text (prompt + continuation)
            metrics: Generation metrics including speedup and p_cache
        """
        if self.streaming_decoder is None:
            raise RuntimeError("Streaming decoder not available. WeDLM module may not be installed.")

        # Encode prompt
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)

        if input_ids.size(1) > self.max_length - max_new_tokens:
            print(f"Warning: Prompt too long, truncating to fit max_new_tokens.")
            input_ids = input_ids[:, -(self.max_length - max_new_tokens):]

        # Generate using streaming decoder
        generated_ids, metrics = self.streaming_decoder.generate(
            prompt_ids=input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            verbose=verbose,
        )

        # Decode generated text
        generated_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)

        metrics['mode'] = 'streaming'

        return generated_text, metrics

    def generate(
        self,
        prompt: str,
        mode: str = "streaming",
        max_new_tokens: int = 256,
        temperature: float = 0.8,
        top_k: Optional[int] = 50,
        top_p: Optional[float] = 0.9,
        repetition_penalty: float = 1.1,
        verbose: bool = False,
    ) -> Tuple[str, Dict]:
        """
        Generate text using specified mode.

        Args:
            prompt: Input text prompt
            mode: "ar" for autoregressive, "streaming" for WeDLM streaming
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Nucleus sampling parameter
            repetition_penalty: Penalty for repeating tokens (AR mode only)
            verbose: Print progress

        Returns:
            generated_text: Full generated text
            metrics: Generation metrics
        """
        if mode == "streaming":
            if self.streaming_decoder is None:
                print("Warning: Streaming decoder not available, falling back to AR mode.")
                mode = "ar"
            else:
                return self.generate_streaming(
                    prompt=prompt,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    verbose=verbose,
                )

        return self.generate_ar(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        )


def load_model_from_checkpoint(
    checkpoint_path: str,
    device: str = "cuda",
    torch_dtype: torch.dtype = torch.bfloat16,
) -> Tuple[SWAMLAModel, AutoTokenizer, Optional[WeDLMConfig]]:
    """Load WeDLM model from a local checkpoint."""
    from inference import load_model_from_checkpoint as base_load

    # Extract WeDLM config BEFORE calling base_load (which removes these params)
    wedlm_config = None
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "config" in checkpoint:
        config_dict = checkpoint["config"]
        if config_dict.get("use_wedlm", False):
            # Get mask_token_id from config, or use vocab_size-1 as fallback
            mask_token_id = config_dict.get("wedlm_mask_token_id", None)
            if mask_token_id is None:
                # Use vocab_size - 1 as default (same as train.py)
                vocab_size = config_dict.get("vocab_size", 50257)  # GPT-2 default
                mask_token_id = vocab_size - 1

            wedlm_config = WeDLMConfig(
                block_size=config_dict.get("wedlm_block_size", 32),
                min_mask_ratio=config_dict.get("wedlm_min_mask_ratio", 0.3),
                max_mask_ratio=config_dict.get("wedlm_max_mask_ratio", 1.0),
                ar_loss_weight=config_dict.get("wedlm_ar_loss_weight", 0.5),
                mask_token_id=mask_token_id,
                window_size=config_dict.get("wedlm_window_size", 32),
                entropy_threshold=config_dict.get("wedlm_entropy_threshold", 0.5),
                distance_penalty=config_dict.get("wedlm_distance_penalty", 0.1),
            )
            print(f"Loaded WeDLM config from checkpoint:")
            print(f"  Block size: {wedlm_config.block_size}")
            print(f"  Mask ratio: [{wedlm_config.min_mask_ratio}, {wedlm_config.max_mask_ratio}]")
            print(f"  Mask token ID: {wedlm_config.mask_token_id}")
            print(f"  Window size: {wedlm_config.window_size}")
            print(f"  Entropy threshold: {wedlm_config.entropy_threshold}")

    # Now load the model
    model, tokenizer = base_load(checkpoint_path, device, torch_dtype)

    return model, tokenizer, wedlm_config


def load_model_from_hf(
    repo_id: str,
    checkpoint_name: Optional[str] = None,
    device: str = "cuda",
    torch_dtype: torch.dtype = torch.bfloat16,
    hf_token: Optional[str] = None,
) -> Tuple[SWAMLAModel, AutoTokenizer, Optional[WeDLMConfig]]:
    """Load WeDLM model from HuggingFace."""
    from inference import load_model_from_hf as base_load

    model, tokenizer = base_load(repo_id, checkpoint_name, device, torch_dtype, hf_token)

    # Default WeDLM config for HF models
    wedlm_config = WeDLMConfig(
        window_size=32,
        entropy_threshold=0.5,
        distance_penalty=0.1,
    )

    return model, tokenizer, wedlm_config


# Default prompts for batch mode
DEFAULT_PROMPTS = [
    "Once upon a time, in a land far away,",
    "The future of artificial intelligence is",
    "In a world where technology has advanced beyond our wildest dreams,",
    "The most important lesson I learned was",
    "Science and magic are not as different as you might think.",
]


def batch_mode(
    engine: WeDLMInferenceEngine,
    prompts: Optional[List[str]] = None,
    mode: str = "streaming",
    max_new_tokens: int = 256,
    temperature: float = 0.8,
    top_k: Optional[int] = 50,
    top_p: Optional[float] = 0.9,
):
    """Run batch generation on prompts."""
    if prompts is None:
        prompts = DEFAULT_PROMPTS

    print("\n" + "=" * 80)
    print(f"BATCH MODE - Generating responses using {mode.upper()} decoding")
    print("=" * 80 + "\n")

    total_tokens = 0
    total_forward_passes = 0
    total_time = 0

    import time

    for i, prompt in enumerate(prompts, 1):
        print(f"\n[Prompt {i}/{len(prompts)}]")
        print(f"Input: {prompt}")
        print("-" * 80)

        start_time = time.time()
        output, metrics = engine.generate(
            prompt=prompt,
            mode=mode,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )
        elapsed = time.time() - start_time

        # Extract only the generated part
        generated = output[len(prompt):]

        print(f"Generated: {generated}")
        print(f"\nMetrics:")
        print(f"  Mode: {metrics['mode']}")
        print(f"  Tokens: {metrics['tokens_generated']}")
        print(f"  Forward passes: {metrics['total_forward_passes']}")
        print(f"  Speedup: {metrics.get('speedup', 1.0):.2f}x")
        if 'p_cache' in metrics:
            print(f"  P_cache: {metrics['p_cache']:.2%}")
        print(f"  Time: {elapsed:.2f}s")
        print(f"  Tokens/s: {metrics['tokens_generated'] / elapsed:.1f}")
        print("=" * 80)

        total_tokens += metrics['tokens_generated']
        total_forward_passes += metrics['total_forward_passes']
        total_time += elapsed

    print(f"\n{'=' * 80}")
    print("SUMMARY")
    print(f"{'=' * 80}")
    print(f"Total tokens generated: {total_tokens}")
    print(f"Total forward passes: {total_forward_passes}")
    print(f"Overall speedup: {total_tokens / total_forward_passes:.2f}x")
    print(f"Total time: {total_time:.2f}s")
    print(f"Average tokens/s: {total_tokens / total_time:.1f}")


def chat_mode(
    engine: WeDLMInferenceEngine,
    mode: str = "streaming",
    max_new_tokens: int = 256,
    temperature: float = 0.8,
    top_k: Optional[int] = 50,
    top_p: Optional[float] = 0.9,
    system_prompt: Optional[str] = None,
):
    """Interactive chat mode."""
    print("\n" + "=" * 80)
    print(f"CHAT MODE - Interactive conversation using {mode.upper()} decoding")
    print("=" * 80)
    print("\nCommands:")
    print("  /quit or /exit - Exit chat mode")
    print("  /clear - Clear conversation history")
    print("  /mode <ar|streaming> - Switch decoding mode")
    print("  /temp <value> - Change temperature")
    print("  /window <value> - Change window size (streaming mode)")
    print("  /entropy <value> - Change entropy threshold (streaming mode)")
    print("  /help - Show this help message")
    print("\n" + "=" * 80 + "\n")

    # Initialize conversation
    conversation_history = ""
    if system_prompt:
        conversation_history = system_prompt + "\n\n"
        print(f"[System prompt: {system_prompt}]\n")

    current_mode = mode

    while True:
        try:
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
                    print("[Conversation cleared]\n")
                    continue

                elif cmd == "/mode":
                    if len(cmd_parts) > 1:
                        new_mode = cmd_parts[1].lower()
                        if new_mode in ["ar", "streaming"]:
                            current_mode = new_mode
                            print(f"[Mode set to {current_mode}]\n")
                        else:
                            print("[Error: Mode must be 'ar' or 'streaming']\n")
                    else:
                        print(f"[Current mode: {current_mode}]\n")
                    continue

                elif cmd == "/temp":
                    if len(cmd_parts) > 1:
                        try:
                            temperature = float(cmd_parts[1])
                            print(f"[Temperature set to {temperature}]\n")
                        except ValueError:
                            print("[Error: Invalid temperature]\n")
                    else:
                        print(f"[Current temperature: {temperature}]\n")
                    continue

                elif cmd == "/window":
                    if len(cmd_parts) > 1 and engine.streaming_decoder:
                        try:
                            engine.wedlm_config.window_size = int(cmd_parts[1])
                            engine.streaming_decoder.window_size = int(cmd_parts[1])
                            print(f"[Window size set to {engine.wedlm_config.window_size}]\n")
                        except ValueError:
                            print("[Error: Invalid window size]\n")
                    else:
                        print(f"[Current window size: {engine.wedlm_config.window_size}]\n")
                    continue

                elif cmd == "/entropy":
                    if len(cmd_parts) > 1 and engine.streaming_decoder:
                        try:
                            engine.wedlm_config.entropy_threshold = float(cmd_parts[1])
                            engine.streaming_decoder.entropy_threshold = float(cmd_parts[1])
                            print(f"[Entropy threshold set to {engine.wedlm_config.entropy_threshold}]\n")
                        except ValueError:
                            print("[Error: Invalid entropy threshold]\n")
                    else:
                        print(f"[Current entropy threshold: {engine.wedlm_config.entropy_threshold}]\n")
                    continue

                elif cmd == "/help":
                    print("\nCommands:")
                    print("  /quit or /exit - Exit chat mode")
                    print("  /clear - Clear conversation history")
                    print("  /mode <ar|streaming> - Switch decoding mode")
                    print("  /temp <value> - Change temperature")
                    print("  /window <value> - Change window size (streaming mode)")
                    print("  /entropy <value> - Change entropy threshold (streaming mode)")
                    print("  /help - Show this help message\n")
                    continue

                else:
                    print(f"[Unknown command: {cmd}. Type /help for available commands]\n")
                    continue

            # Add user input to history
            conversation_history += f"You: {user_input}\n"
            prompt = conversation_history + "Assistant:"

            # Generate response
            import time
            start_time = time.time()

            output, metrics = engine.generate(
                prompt=prompt,
                mode=current_mode,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )

            elapsed = time.time() - start_time

            # Extract assistant response
            if "Assistant:" in output:
                response = output.split("Assistant:")[-1].strip()
            else:
                response = output[len(prompt):].strip()

            # Stop at next "You:" if model hallucinates continuation
            if "\nYou:" in response:
                response = response.split("\nYou:")[0].strip()

            print(f"Assistant: {response}")
            print(f"  [{metrics['mode']}, {metrics['tokens_generated']} tokens, "
                  f"{metrics.get('speedup', 1.0):.1f}x speedup, {elapsed:.1f}s]\n")

            # Add response to history
            conversation_history += f"Assistant: {response}\n"

        except KeyboardInterrupt:
            print("\n\n[Interrupted. Type /quit to exit or continue chatting]\n")
            continue
        except Exception as e:
            print(f"\n[Error: {e}]\n")
            continue


def main():
    parser = argparse.ArgumentParser(description="WeDLM Inference Script")

    # Model loading options
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument("--hf_repo_id", type=str,
                             help="Hugging Face model repo ID")
    model_group.add_argument("--checkpoint", type=str,
                             help="Path to local checkpoint file")

    # HuggingFace options
    parser.add_argument("--hf_checkpoint", type=str, default=None,
                        help="Specific checkpoint folder name in HF repo")

    # Inference mode
    parser.add_argument("--mode", type=str, choices=["ar", "streaming"], default="streaming",
                        help="Decoding mode: 'ar' (autoregressive) or 'streaming' (WeDLM parallel)")
    parser.add_argument("--chat", action="store_true",
                        help="Enable interactive chat mode")

    # Generation parameters
    parser.add_argument("--max_new_tokens", type=int, default=256,
                        help="Maximum number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=50,
                        help="Top-k sampling parameter (0 to disable)")
    parser.add_argument("--top_p", type=float, default=0.9,
                        help="Top-p (nucleus) sampling parameter")
    parser.add_argument("--repetition_penalty", type=float, default=1.1,
                        help="Repetition penalty (AR mode only)")

    # WeDLM streaming parameters
    parser.add_argument("--window_size", type=int, default=32,
                        help="Streaming window size")
    parser.add_argument("--entropy_threshold", type=float, default=0.5,
                        help="Entropy threshold for mask filling")
    parser.add_argument("--distance_penalty", type=float, default=0.1,
                        help="Distance penalty for entropy adjustment")

    # Chat mode options
    parser.add_argument("--system_prompt", type=str, default=None,
                        help="System prompt for chat mode")

    # Batch mode options
    parser.add_argument("--prompts_file", type=str, default=None,
                        help="File containing prompts (one per line)")

    # Device options
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to run inference on")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["float32", "float16", "bfloat16"],
                        help="Data type for model weights")
    parser.add_argument("--max_length", type=int, default=2048,
                        help="Maximum sequence length")

    args = parser.parse_args()

    # Convert dtype
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    torch_dtype = dtype_map[args.dtype]

    # Check device
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = "cpu"
        if args.dtype == "bfloat16":
            print("BFloat16 not well supported on CPU, using float32")
            torch_dtype = torch.float32

    # Load model
    wedlm_config = None
    if args.hf_repo_id:
        hf_token = os.getenv("HF_TOKEN")
        model, tokenizer, wedlm_config = load_model_from_hf(
            repo_id=args.hf_repo_id,
            checkpoint_name=args.hf_checkpoint,
            device=args.device,
            torch_dtype=torch_dtype,
            hf_token=hf_token,
        )
    else:
        model, tokenizer, wedlm_config = load_model_from_checkpoint(
            checkpoint_path=args.checkpoint,
            device=args.device,
            torch_dtype=torch_dtype,
        )

    # Override WeDLM config with CLI arguments if provided
    if wedlm_config is None:
        wedlm_config = WeDLMConfig()

    wedlm_config.window_size = args.window_size
    wedlm_config.entropy_threshold = args.entropy_threshold
    wedlm_config.distance_penalty = args.distance_penalty

    # Create inference engine
    engine = WeDLMInferenceEngine(
        model=model,
        tokenizer=tokenizer,
        wedlm_config=wedlm_config,
        device=args.device,
        max_length=args.max_length,
    )

    print(f"\nWeDLM Inference Engine initialized")
    print(f"  Device: {args.device}")
    print(f"  Dtype: {args.dtype}")
    print(f"  Mode: {args.mode}")
    print(f"  Window size: {wedlm_config.window_size}")
    print(f"  Entropy threshold: {wedlm_config.entropy_threshold}")
    print(f"  Distance penalty: {wedlm_config.distance_penalty}")

    # Run inference
    if args.chat:
        chat_mode(
            engine=engine,
            mode=args.mode,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k if args.top_k > 0 else None,
            top_p=args.top_p,
            system_prompt=args.system_prompt,
        )
    else:
        # Batch mode
        prompts = None
        if args.prompts_file:
            with open(args.prompts_file, 'r') as f:
                prompts = [line.strip() for line in f if line.strip()]
            print(f"Loaded {len(prompts)} prompts from {args.prompts_file}")

        batch_mode(
            engine=engine,
            prompts=prompts,
            mode=args.mode,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k if args.top_k > 0 else None,
            top_p=args.top_p,
        )


if __name__ == "__main__":
    main()
