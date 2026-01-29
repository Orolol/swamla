"""Exponential Moving Average (EMA) for model weights."""

import torch
import torch.nn as nn
from typing import Dict, Optional
from contextlib import contextmanager


class EMAModel:
    """
    Maintains an exponential moving average of model parameters.

    Usage:
        ema = EMAModel(model, decay=0.9999)

        # After each optimizer step:
        ema.update(model)

        # For validation:
        with ema.apply(model):
            val_loss = validate(model)
    """

    def __init__(self, model: nn.Module, decay: float = 0.9999, device: Optional[torch.device] = None):
        """
        Args:
            model: The model to track
            decay: EMA decay factor (higher = slower averaging). Typical: 0.9999
            device: Device to store EMA weights. If None, uses same device as model.
        """
        self.decay = decay
        self.device = device

        # Clone all parameters
        self.ema_params: Dict[str, torch.Tensor] = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                p = param.data.clone()
                if device is not None:
                    p = p.to(device)
                self.ema_params[name] = p

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """Update EMA parameters with current model parameters."""
        for name, param in model.named_parameters():
            if name in self.ema_params and param.requires_grad:
                # θ_ema = decay * θ_ema + (1 - decay) * θ
                # OPTIMIZATION: Check device before transfer to avoid unnecessary copies
                if param.device != self.ema_params[name].device:
                    param_data = param.data.to(self.ema_params[name].device)
                else:
                    param_data = param.data
                self.ema_params[name].lerp_(param_data, 1 - self.decay)

    @contextmanager
    def apply(self, model: nn.Module):
        """
        Context manager that temporarily replaces model weights with EMA weights.

        OPTIMIZATION: Uses pointer swapping instead of cloning to save memory.
        - Before: Creates full copy of all parameters (~4GB for 1B model)
        - After: Only swaps data pointers (minimal overhead)

        Usage:
            with ema.apply(model):
                val_loss = validate(model)
            # Original weights are restored after the block
        """
        # Store original parameter data tensors (view/reference, not clone)
        original_params: Dict[str, torch.Tensor] = {}
        for name, param in model.named_parameters():
            if name in self.ema_params and param.requires_grad:
                # Store reference to original data tensor
                original_params[name] = param.data
                # Swap to EMA weights
                param.data = self.ema_params[name].to(param.device)

        try:
            yield
        finally:
            # Restore original parameters by swapping back
            for name, param in model.named_parameters():
                if name in original_params:
                    param.data = original_params[name]

    def state_dict(self) -> Dict[str, torch.Tensor]:
        """Return EMA state for checkpointing."""
        return {
            'decay': self.decay,
            'ema_params': self.ema_params,
        }

    def load_state_dict(self, state_dict: Dict) -> None:
        """Load EMA state from checkpoint."""
        self.decay = state_dict['decay']
        self.ema_params = state_dict['ema_params']
