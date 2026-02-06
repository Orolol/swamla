"""Exponential Moving Average (EMA) for model weights."""

import logging
import torch
import torch.nn as nn
from typing import Dict, Optional
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class EMAModel:
    """
    Maintains an exponential moving average of model parameters.

    Handles torch.compile (_orig_mod.) and DDP (module.) prefixes transparently
    by normalizing parameter names before matching.

    Usage:
        ema = EMAModel(model, decay=0.9999)

        # After each optimizer step:
        ema.update(model)

        # For validation:
        with ema.apply(model):
            val_loss = validate(model)
    """

    @staticmethod
    def _normalize_name(name: str) -> str:
        """Strip torch.compile and DDP prefixes from parameter names.

        torch.compile adds '_orig_mod.' prefix, DDP adds 'module.' prefix.
        These can be nested (e.g. 'module._orig_mod.transformer...').
        """
        for prefix in ("_orig_mod.", "module."):
            while name.startswith(prefix):
                name = name[len(prefix):]
        return name

    def __init__(self, model: nn.Module, decay: float = 0.9999, device: Optional[torch.device] = None):
        """
        Args:
            model: The model to track
            decay: EMA decay factor (higher = slower averaging). Typical: 0.9999
            device: Device to store EMA weights. If None, uses same device as model.
        """
        self.decay = decay
        self.device = device

        # Clone all parameters, stored with normalized names
        self.ema_params: Dict[str, torch.Tensor] = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                norm_name = self._normalize_name(name)
                p = param.data.clone()
                if device is not None:
                    p = p.to(device)
                self.ema_params[norm_name] = p

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """Update EMA parameters with current model parameters."""
        matched = 0
        for name, param in model.named_parameters():
            norm_name = self._normalize_name(name)
            if norm_name in self.ema_params and param.requires_grad:
                # theta_ema = decay * theta_ema + (1 - decay) * theta
                if param.device != self.ema_params[norm_name].device:
                    param_data = param.data.to(self.ema_params[norm_name].device)
                else:
                    param_data = param.data
                self.ema_params[norm_name].lerp_(param_data, 1 - self.decay)
                matched += 1

        if matched == 0 and len(self.ema_params) > 0:
            model_sample = [n for n, _ in zip(model.named_parameters(), range(3))]
            model_names = [n[0] for n in model_sample]
            ema_sample = list(self.ema_params.keys())[:3]
            logger.warning(
                f"EMA update matched 0/{len(self.ema_params)} params! "
                f"Model names: {model_names}, EMA names: {ema_sample}"
            )

    @contextmanager
    def apply(self, model: nn.Module):
        """
        Context manager that temporarily replaces model weights with EMA weights.

        Uses pointer swapping instead of cloning to save memory.

        Usage:
            with ema.apply(model):
                val_loss = validate(model)
            # Original weights are restored after the block
        """
        # Store original parameter data tensors (view/reference, not clone)
        # Key by the actual model param name (not normalized) for correct restore
        original_params: Dict[str, torch.Tensor] = {}
        matched = 0
        for name, param in model.named_parameters():
            norm_name = self._normalize_name(name)
            if norm_name in self.ema_params and param.requires_grad:
                original_params[name] = param.data
                param.data = self.ema_params[norm_name].to(param.device)
                matched += 1

        if matched == 0 and len(self.ema_params) > 0:
            logger.warning(
                f"EMA apply matched 0/{len(self.ema_params)} params! "
                "EMA weights will NOT be used for validation."
            )

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
        # Normalize loaded keys to strip any compile/DDP prefixes
        raw_params = state_dict['ema_params']
        self.ema_params = {
            self._normalize_name(k): v for k, v in raw_params.items()
        }
