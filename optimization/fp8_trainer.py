"""
FP8 training utilities following DeepSeek-V3's mixed precision framework.

This module provides training utilities that implement DeepSeek's approach:
- Fine-grained quantization (tile/block-wise)
- High-precision accumulation
- Mixed precision training with selective FP8 usage
- Low-precision optimizer states and caching
"""

import torch
import torch.nn as nn
from torch.optim import Optimizer
from typing import Dict, Any, Optional, List, Tuple
import math
from contextlib import contextmanager


class FP8AdamW(torch.optim.AdamW):
    """
    AdamW optimizer with FP8 training support following DeepSeek's approach.
    
    Key features:
    - BF16 storage for first and second moments (instead of FP32)
    - Master weights and gradients stay in FP32 for stability
    - Support for parameter groups with different precision settings
    """
    
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0.01, amsgrad=False, use_low_precision_moments=True):
        self.use_low_precision_moments = use_low_precision_moments
        super().__init__(params, lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad)
    
    def _init_group(self, group, params, grads, exp_avgs, exp_avg_sqs, state_steps):
        """Initialize group with potential low-precision moments."""
        for p in group['params']:
            if p.grad is None:
                continue
                
            params.append(p)
            grads.append(p.grad)
            
            state = self.state[p]
            
            # State initialization
            if len(state) == 0:
                state['step'] = torch.zeros((), dtype=torch.int32, device=p.device)
                
                # Use BF16 for moments if requested (following DeepSeek)
                moment_dtype = torch.bfloat16 if self.use_low_precision_moments else p.dtype
                
                # First moment
                state['exp_avg'] = torch.zeros_like(
                    p, memory_format=torch.preserve_format, dtype=moment_dtype
                )
                
                # Second moment
                state['exp_avg_sq'] = torch.zeros_like(
                    p, memory_format=torch.preserve_format, dtype=moment_dtype
                )
                
                if group['amsgrad']:
                    # Maintain max of second moment in same precision
                    state['max_exp_avg_sq'] = torch.zeros_like(
                        p, memory_format=torch.preserve_format, dtype=moment_dtype
                    )
            
            exp_avgs.append(state['exp_avg'])
            exp_avg_sqs.append(state['exp_avg_sq'])
            state_steps.append(state['step'])
    
    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step with FP8 awareness."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            params = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            state_steps = []
            
            self._init_group(group, params, grads, exp_avgs, exp_avg_sqs, state_steps)
            
            beta1, beta2 = group['betas']
            
            # Update steps
            for state_step in state_steps:
                state_step += 1
            
            # Update biased first moment estimate
            for param, grad, exp_avg in zip(params, grads, exp_avgs):
                # Handle mixed precision: grad might be in different dtype
                if exp_avg.dtype != grad.dtype:
                    grad = grad.to(exp_avg.dtype)
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
            
            # Update biased second raw moment estimate
            for param, grad, exp_avg_sq in zip(params, grads, exp_avg_sqs):
                if exp_avg_sq.dtype != grad.dtype:
                    grad = grad.to(exp_avg_sq.dtype)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
            
            # Apply updates
            for i, param in enumerate(params):
                grad = grads[i]
                exp_avg = exp_avgs[i]
                exp_avg_sq = exp_avg_sqs[i]
                step = state_steps[i].item()
                
                # Bias correction
                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2 ** step
                
                # Get corrected moments in param dtype for update
                exp_avg_corrected = exp_avg.float() / bias_correction1
                exp_avg_sq_corrected = exp_avg_sq.float() / bias_correction2
                
                # Compute update
                denom = exp_avg_sq_corrected.sqrt().add_(group['eps'])
                update = exp_avg_corrected / denom
                
                # Add weight decay
                if group['weight_decay'] != 0:
                    update.add_(param.float(), alpha=group['weight_decay'])
                
                # Apply update (ensuring correct dtype)
                param.add_(update.to(param.dtype), alpha=-group['lr'])
        
        return loss


class FP8Lion(Optimizer):
    """
    Lion optimizer with FP8 training support following DeepSeek's approach.

    Key features:
    - BF16 storage for momentum (instead of FP32)
    - Master weights and gradients stay in FP32 for stability
    - Support for parameter groups with different precision settings

    Lion algorithm:
    - Simpler than AdamW: only one moment (momentum) instead of two
    - More memory efficient: ~50% less state than AdamW
    - Often converges faster with same final quality
    """

    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0,
                 use_low_precision_moments=True):
        """
        Initialize FP8Lion optimizer.

        Args:
            params: Model parameters or parameter groups
            lr: Learning rate (default: 1e-4)
            betas: Coefficients for momentum and EMA (default: (0.9, 0.99))
            weight_decay: Weight decay coefficient (default: 0.0)
            use_low_precision_moments: Store moments in BF16 instead of FP32 (default: True)
        """
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        self.use_low_precision_moments = use_low_precision_moments
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step with FP8 awareness."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group['betas']
            lr = group['lr']
            weight_decay = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    # Use BF16 for momentum if requested (following DeepSeek FP8 approach)
                    moment_dtype = torch.bfloat16 if self.use_low_precision_moments else p.dtype
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format, dtype=moment_dtype)

                exp_avg = state['exp_avg']

                # Lion update algorithm
                # 1. Interpolate between momentum and gradient
                # Handle mixed precision: grad might be in different dtype than momentum
                if exp_avg.dtype != grad.dtype:
                    grad_for_update = grad.to(exp_avg.dtype)
                else:
                    grad_for_update = grad

                # update = interpolate(exp_avg, grad, beta1)
                update = exp_avg.mul(beta1).add_(grad_for_update, alpha=1 - beta1)

                # 2. Apply weight decay (if any) and sign update
                if weight_decay != 0:
                    # Convert to param dtype for update
                    update_fp32 = update.float()
                    p_data_fp32 = p.float()

                    # Sign update with weight decay
                    p_data_fp32.mul_(1 - lr * weight_decay).add_(
                        update_fp32.sign(), alpha=-lr
                    )

                    # Copy back to param dtype
                    p.copy_(p_data_fp32.to(p.dtype))
                else:
                    # Sign update without weight decay
                    # Convert update to param dtype
                    update_for_param = update.float() if p.dtype != torch.float32 else update
                    p.add_(update_for_param.sign().to(p.dtype), alpha=-lr)

                # 3. Update momentum (EMA of gradients)
                if exp_avg.dtype != grad.dtype:
                    grad_for_momentum = grad.to(exp_avg.dtype)
                else:
                    grad_for_momentum = grad

                exp_avg.mul_(beta2).add_(grad_for_momentum, alpha=1 - beta2)

        return loss


class FP8MixedPrecisionTrainer:
    """
    Training utilities for FP8 mixed precision.
    """
    
    def __init__(self, model: nn.Module, config: Dict[str, Any]):
        self.model = model
        self.config = config
        self.use_fp8 = config.get('use_fp8', False)
        
    def create_optimizer(self, lr: float = 1e-4) -> Optimizer:
        """
        Create optimizer.
        """
        # Standard AdamW or Lion, no special FP8 optimizer needed for now 
        # (unless we implement 8-bit optimizers, but standard BF16 AdamW is fine for H100)
        
        param_groups = [
            {'params': [p for n, p in self.model.named_parameters() if p.requires_grad], 'lr': lr}
        ]
        
        # Use standard AdamW
        optimizer = torch.optim.AdamW(
            param_groups,
            lr=lr,
            betas=(0.9, 0.95),
            eps=1e-8,
            weight_decay=0.1,
            fused=torch.cuda.is_available()
        )
        
        return optimizer
    
    @contextmanager
    def fp8_training_context(self):
        """
        Context manager for training.
        With native FP8Linear layers, we just need standard BF16 autocast for the rest.
        """
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            yield
    
    def prepare_batch_fp8(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Prepare batch.
        """
        device = next(self.model.parameters()).device
        
        prepared_batch = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                value = value.to(device)
                # Keep inputs in high precision (BF16/Long), FP8 casting happens inside layers
                prepared_batch[key] = value
            else:
                prepared_batch[key] = value
        
        return prepared_batch
    
    def backward_with_fp8(self, loss: torch.Tensor, retain_graph: bool = False):
        """
        Backward pass.
        """
        loss.backward(retain_graph=retain_graph)
    
    def get_memory_stats(self) -> Dict[str, float]:
        """Get memory statistics."""
        if not torch.cuda.is_available():
            return {}
        
        return {
            'allocated_gb': torch.cuda.memory_allocated() / 1e9,
            'reserved_gb': torch.cuda.memory_reserved() / 1e9,
            'max_allocated_gb': torch.cuda.max_memory_allocated() / 1e9,
        }


def create_fp8_training_config(base_config: Dict[str, Any], 
                              gpu_arch: str = 'hopper') -> Dict[str, Any]:
    """
    Create FP8 training configuration.
    """
    config = base_config.copy()
    
    if gpu_arch == 'hopper':  # H100, H200
        config.update({
            'use_fp8': True,
        })
    else:
        config.update({
            'use_fp8': False,
        })
    
    return config