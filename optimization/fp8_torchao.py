"""
TorchAO FP8 Training Integration for SWA-MLA Model.

This module provides integration with TorchAO for FP8 training, replacing the custom
FP8 implementation with PyTorch's native TorchAO library.

Key features:
- Automatic FP8 conversion using convert_to_float8_training()
- AdamWFp8 optimizer with native FP8 support
- Compatible with torch.compile and FSDP2
- Tested up to 512 GPUs with 1.5x speedups
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Tuple
import subprocess

def convert_model_to_fp8(
    model: nn.Module,
    use_compile: bool = True,
    config: Optional[Dict[str, Any]] = None
) -> nn.Module:
    """
    Convert model to FP8 training using TorchAO.

    Automatically excludes modules with non-aligned dimensions:
    - lm_head: Output projection (vocab_size may not be divisible by 16)
    - wte: Token embeddings (not a Linear layer)
    - Any Linear layer with dimensions not divisible by 16

    This is necessary because TorchAO's _scaled_mm kernel requires
    all matrix dimensions to be divisible by 16.

    Args:
        model: The model to convert
        use_compile: Whether to use torch.compile for additional speedup
        config: Optional configuration dict for FP8 training

    Returns:
        The converted model ready for FP8 training

    Example:
        >>> model = create_swa_mla_model('small', vocab_size=50257)
        >>> model = convert_model_to_fp8(model)
        >>> # Now train with model in FP8 (lm_head stays in BF16)
    """
    try:
        from torchao.float8 import convert_to_float8_training

        # Define filter function to exclude specific modules
        def module_filter_fn(mod: torch.nn.Module, fqn: str) -> bool:
            """
            Filter function to determine which modules should be converted to FP8.

            Args:
                mod: The module to check
                fqn: Fully qualified name of the module

            Returns:
                True if module should be converted to FP8, False otherwise
            """
            # Exclude lm_head (output projection with vocab_size dimension)
            if 'lm_head' in fqn:
                print(f"  Excluding from FP8: {fqn} (vocab_size alignment)")
                return False

            # Exclude embeddings (wte, wpe) - they are not Linear layers anyway
            if 'wte' in fqn or 'wpe' in fqn:
                print(f"  Excluding from FP8: {fqn} (embedding layer)")
                return False

            # For Linear layers, check dimension alignment
            if isinstance(mod, torch.nn.Linear):
                # Exclude if dimensions not divisible by 16
                if mod.in_features % 16 != 0 or mod.out_features % 16 != 0:
                    print(f"  Excluding from FP8: {fqn} (dimensions not divisible by 16: "
                          f"{mod.in_features}x{mod.out_features})")
                    return False

            # Convert everything else
            return True

        # Convert model to FP8 with filtering
        print("Converting model to FP8 (excluding lm_head and non-aligned layers)...")
        convert_to_float8_training(model, module_filter_fn=module_filter_fn)

        print("✓ FP8 conversion complete")

        # Optionally compile for additional speedup (~10-20%)
        # Note: Don't compile here, let train.py handle it with the right mode
        if use_compile:
            model = torch.compile(model)

        return model

    except ImportError as e:
        raise ImportError(
            "TorchAO is not installed. Please install it with: pip install torchao"
        ) from e


def create_fp8_optimizer(
    model: nn.Module,
    lr: float = 1e-4,
    betas: Tuple[float, float] = (0.9, 0.95),
    eps: float = 1e-8,
    weight_decay: float = 0.1,
    optimizer_type: str = 'adamw',
    fused: bool = True
) -> torch.optim.Optimizer:
    """
    Create an FP8-compatible optimizer using TorchAO.

    Args:
        model: The model to optimize
        lr: Learning rate
        betas: Adam beta parameters (beta1, beta2)
        eps: Adam epsilon for numerical stability
        weight_decay: Weight decay coefficient
        optimizer_type: Type of optimizer ('adamw', 'adamw8bit', 'adamwfp8')
        fused: Whether to use fused optimizer implementation

    Returns:
        Configured optimizer

    Note:
        - 'adamw': Standard AdamW (fused on CUDA)
        - 'adamw8bit': 8-bit quantized AdamW (2x memory reduction)
        - 'adamwfp8': FP8 AdamW (for H100/H200, best memory and speed)
    """
    # Separate parameters into decay and no_decay groups
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # No weight decay for biases, norms, embeddings
        if any(nd in name for nd in ['.bias', 'norm', 'ln_', 'wte', 'wpe']):
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    # Create parameter groups
    param_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': no_decay_params, 'weight_decay': 0.0}
    ]

    # Select optimizer based on type
    if optimizer_type == 'adamwfp8':
        try:
            from torchao.optim import AdamWFp8

            optimizer = AdamWFp8(
                param_groups,
                lr=lr,
                betas=betas,
                eps=eps,
            )
            print(f"Using TorchAO AdamWFp8 optimizer (FP8 training)")

        except ImportError:
            print("TorchAO AdamWFp8 not available, falling back to standard AdamW")
            optimizer = torch.optim.AdamW(
                param_groups,
                lr=lr,
                betas=betas,
                eps=eps,
                fused=fused and torch.cuda.is_available()
            )

    elif optimizer_type == 'adamw8bit':
        try:
            from torchao.optim import AdamW8bit

            optimizer = AdamW8bit(
                param_groups,
                lr=lr,
                betas=betas,
                eps=eps,
            )
            print(f"Using TorchAO AdamW8bit optimizer (8-bit quantized)")

        except ImportError:
            print("TorchAO AdamW8bit not available, falling back to standard AdamW")
            optimizer = torch.optim.AdamW(
                param_groups,
                lr=lr,
                betas=betas,
                eps=eps,
                fused=fused and torch.cuda.is_available()
            )

    else:  # 'adamw' or other
        optimizer = torch.optim.AdamW(
            param_groups,
            lr=lr,
            betas=betas,
            eps=eps,
            fused=fused and torch.cuda.is_available()
        )
        print(f"Using standard {'fused ' if fused and torch.cuda.is_available() else ''}AdamW optimizer")

    return optimizer


def get_fp8_memory_stats() -> Dict[str, float]:
    """
    Get memory statistics for FP8 training.

    Returns:
        Dictionary with memory usage statistics in GB
    """
    if not torch.cuda.is_available():
        return {}

    allocated = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    max_allocated = torch.cuda.max_memory_allocated() / 1e9

    return {
        'allocated_gb': allocated,
        'reserved_gb': reserved,
        'max_allocated_gb': max_allocated,
    }


def configure_fp8_training(
    model: nn.Module,
    learning_rate: float = 1e-4,
    weight_decay: float = 0.1,
    optimizer_type: str = 'adamw',
    use_compile: bool = True,
    force_fp8: bool = False
) -> Tuple[nn.Module, torch.optim.Optimizer, bool]:
    """
    Configure a model for FP8 training with TorchAO.

    This is a high-level helper that:
    1. Checks if FP8 is supported
    2. Converts the model to FP8 if supported
    3. Creates an appropriate optimizer
    4. Returns everything ready for training

    Args:
        model: Model to configure
        learning_rate: Learning rate
        weight_decay: Weight decay coefficient
        optimizer_type: Type of optimizer ('adamw', 'adamw8bit', 'adamwfp8')
        use_compile: Whether to use torch.compile
        force_fp8: Force FP8 even if GPU detection fails (use with caution)

    Returns:
        Tuple of (model, optimizer, fp8_enabled)

    Example:
        >>> model = create_swa_mla_model('small', vocab_size=50257)
        >>> model, optimizer, fp8_enabled = configure_fp8_training(model)
        >>> if fp8_enabled:
        >>>     print("Training with FP8!")
    """
    fp8_supported = is_fp8_supported()
    fp8_enabled = False

    if fp8_supported or force_fp8:
        try:
            # Convert model to FP8
            model = convert_model_to_fp8(model, use_compile=use_compile)
            fp8_enabled = True

            # Use FP8 optimizer if available
            if optimizer_type == 'adamw' and fp8_supported:
                optimizer_type = 'adamwfp8'

            print(f"✓ FP8 training enabled with TorchAO")
            print(f"  - Model converted to FP8")
            print(f"  - Optimizer: {optimizer_type}")
            print(f"  - torch.compile: {use_compile}")

        except Exception as e:
            print(f"Warning: Failed to enable FP8 training: {e}")
            print("Falling back to BF16 training")
            fp8_enabled = False
    else:
        print("FP8 training not supported on this GPU (requires H100/H200)")
        print("Using BF16 training instead")

    # Create optimizer
    optimizer = create_fp8_optimizer(
        model,
        lr=learning_rate,
        weight_decay=weight_decay,
        optimizer_type=optimizer_type
    )

    return model, optimizer, fp8_enabled


def print_fp8_training_info():
    """Print information about FP8 training support and configuration."""
    print("\n" + "="*60)
    print("TorchAO FP8 Training Information")
    print("="*60)

    # Check GPU
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"GPU: {gpu_name}")

        if hasattr(torch.cuda, 'get_device_capability'):
            major, minor = torch.cuda.get_device_capability()
            print(f"Compute Capability: {major}.{minor}")
    else:
        print("GPU: None (CUDA not available)")

    # Check FP8 support
    fp8_supported = is_fp8_supported()
    print(f"FP8 Support: {'✓ Yes' if fp8_supported else '✗ No'}")

    # Check TorchAO availability
    try:
        import torchao
        print(f"TorchAO Version: {torchao.__version__ if hasattr(torchao, '__version__') else 'installed'}")

        # Check for specific components
        try:
            from torchao.float8 import convert_to_float8_training
            print("  - convert_to_float8_training: ✓")
        except ImportError:
            print("  - convert_to_float8_training: ✗")

        try:
            from torchao.optim import AdamWFp8
            print("  - AdamWFp8: ✓")
        except ImportError:
            print("  - AdamWFp8: ✗")

    except ImportError:
        print("TorchAO: ✗ Not installed")
        print("  Install with: pip install torchao")

    print("="*60 + "\n")


if __name__ == "__main__":
    # Test/demo script
    print_fp8_training_info()

    # Example usage
    print("\nExample usage:")
    print("""
    from models.swa_mla_model import create_swa_mla_model
    from optimization.fp8_torchao import configure_fp8_training

    # Create model
    model = create_swa_mla_model('small', vocab_size=50257)
    model = model.cuda()

    # Configure for FP8 training
    model, optimizer, fp8_enabled = configure_fp8_training(
        model,
        learning_rate=1e-4,
        optimizer_type='adamw',  # Will auto-upgrade to 'adamwfp8' on H100/H200
        use_compile=True
    )

    # Train as normal
    for batch in dataloader:
        optimizer.zero_grad()
        loss = model(**batch).loss
        loss.backward()
        optimizer.step()
    """)
