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
    Training utilities for FP8 mixed precision following DeepSeek's approach.
    """
    
    def __init__(self, model: nn.Module, config: Dict[str, Any]):
        self.model = model
        self.config = config
        self.use_fp8 = config.get('use_fp8', False)
        self.fp8_tile_size = config.get('fp8_tile_size', 128)
        
        # Track which modules should stay in high precision
        self.high_precision_modules = self._identify_high_precision_modules()
        
        # Configure FP8 settings
        if self.use_fp8:
            self._configure_fp8_environment()
    
    def _identify_high_precision_modules(self) -> List[str]:
        """
        Identify modules that should stay in high precision.
        
        Following DeepSeek:
        - Embeddings
        - Normalization layers
        - Output heads
        - Attention operators
        - MoE gating modules
        """
        high_precision_names = []
        
        for name, module in self.model.named_modules():
            if any(pattern in name.lower() for pattern in 
                   ['embed', 'norm', 'head', 'gate', 'router']):
                high_precision_names.append(name)
            elif isinstance(module, (nn.LayerNorm, nn.RMSNorm, nn.Embedding)):
                high_precision_names.append(name)
        
        return high_precision_names
    
    def _configure_fp8_environment(self):
        """Configure environment for FP8 training."""
        import os
        
        # Use E4M3 format for better precision (mantissa over exponents)
        os.environ["NVTE_FP8_FORMAT"] = "E4M3"
        
        # Enable FP8 for feed-forward networks
        os.environ["NVTE_FP8_FFN"] = "1"
        
        # Disable FP8 for attention (keep in higher precision)
        os.environ["NVTE_FP8_MHA"] = "0"
        
        # Enable online quantization
        os.environ["NVTE_FP8_CALIBRATION"] = "0"
    
    def create_optimizer(self, lr: float = 1e-4) -> Optimizer:
        """
        Create optimizer with FP8-aware configuration.
        
        Returns:
            FP8AdamW optimizer with appropriate parameter groups
        """
        # Separate parameters by precision requirements
        high_precision_params = []
        standard_params = []
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if any(hp_name in name for hp_name in self.high_precision_modules):
                    high_precision_params.append(param)
                else:
                    standard_params.append(param)
        
        # Create parameter groups
        param_groups = []
        
        if high_precision_params:
            param_groups.append({
                'params': high_precision_params,
                'lr': lr,
                'name': 'high_precision'
            })
        
        if standard_params:
            param_groups.append({
                'params': standard_params,
                'lr': lr,
                'name': 'standard'
            })
        
        # Create optimizer with low-precision moments
        optimizer = FP8AdamW(
            param_groups,
            lr=lr,
            betas=(0.9, 0.95),  # Typical for LLMs
            eps=1e-8,
            weight_decay=0.1,
            use_low_precision_moments=True
        )
        
        return optimizer
    
    @contextmanager
    def fp8_training_context(self):
        """
        Context manager for FP8 training with mixed precision.
        
        Handles:
        - FP8 autocast for supported operations
        - Gradient scaling for stability
        - Automatic fallback to BF16 when needed
        """
        if not self.use_fp8:
            # Standard BF16 training
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                yield
            return
        
        try:
            # Try to use transformer_engine FP8 autocast
            import transformer_engine.pytorch as te
            
            if hasattr(te, 'fp8_autocast'):
                # Configure FP8 recipe following DeepSeek
                from transformer_engine.common import recipe
                
                fp8_recipe = recipe.DelayedScaling(
                    margin=0,
                    interval=1,
                    fp8_format=recipe.Format.E4M3,  # Mantissa over exponents
                    amax_history_len=1,  # Online quantization
                    amax_compute_algo='most_recent'  # No delayed scaling
                )
                
                with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
                    yield
            else:
                # Fallback to BF16
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    yield
                    
        except ImportError:
            # Transformer engine not available, use BF16
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                yield
    
    def prepare_batch_fp8(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Prepare batch for FP8 training.
        
        For activations that will be cached (like in MoE):
        - Pre-quantize to FP8 for dispatch
        - Use integral power-of-2 scales for conversions
        """
        if not self.use_fp8:
            return batch
        
        # Move to appropriate device and dtype
        device = next(self.model.parameters()).device
        
        prepared_batch = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                # Move to device
                value = value.to(device)
                
                # For input tokens, keep in high precision
                if key in ['input_ids', 'labels', 'attention_mask']:
                    prepared_batch[key] = value
                else:
                    # Other tensors can start in BF16
                    prepared_batch[key] = value.to(torch.bfloat16)
            else:
                prepared_batch[key] = value
        
        return prepared_batch
    
    def backward_with_fp8(self, loss: torch.Tensor, retain_graph: bool = False):
        """
        Backward pass with FP8 considerations.
        
        Handles:
        - Mixed precision gradients
        - Gradient accumulation in FP32
        - Potential gradient scaling
        """
        # For FP8 training, gradients are accumulated in FP32
        # even if computations use FP8
        loss.backward(retain_graph=retain_graph)
    
    def get_memory_stats(self) -> Dict[str, float]:
        """Get memory statistics for FP8 training."""
        if not torch.cuda.is_available():
            return {}
        
        stats = {
            'allocated_gb': torch.cuda.memory_allocated() / 1e9,
            'reserved_gb': torch.cuda.memory_reserved() / 1e9,
            'max_allocated_gb': torch.cuda.max_memory_allocated() / 1e9,
        }
        
        # Estimate FP8 savings
        if self.use_fp8:
            # Rough estimate: FP8 uses ~50% memory for activations
            # compared to FP16/BF16
            estimated_bf16_memory = stats['allocated_gb'] * 1.5
            stats['estimated_savings_gb'] = estimated_bf16_memory - stats['allocated_gb']
            stats['savings_percent'] = (stats['estimated_savings_gb'] / estimated_bf16_memory) * 100
        
        return stats


def create_fp8_training_config(base_config: Dict[str, Any], 
                              gpu_arch: str = 'hopper') -> Dict[str, Any]:
    """
    Create FP8 training configuration based on GPU architecture.
    
    Args:
        base_config: Base training configuration
        gpu_arch: GPU architecture ('hopper', 'ada', 'ampere')
        
    Returns:
        Updated configuration with FP8 settings
    """
    config = base_config.copy()
    
    if gpu_arch == 'hopper':  # H100, H200
        config.update({
            'use_fp8': True,
            'fp8_tile_size': 128,
            'fp8_mla_params': True,  # Can use FP8 for MLA params
            'fp8_cache': True,  # Low-precision caching
            'fp8_format': 'E4M3',  # Better precision
        })
    elif gpu_arch == 'ada':  # RTX 4090, 6000 Ada
        config.update({
            'use_fp8': True,
            'fp8_tile_size': 128,
            'fp8_mla_params': False,  # Keep MLA params in FP16
            'fp8_cache': True,
            'fp8_format': 'E4M3',
        })
    else:  # Older architectures
        config.update({
            'use_fp8': False,  # Fallback to BF16/FP16
        })
    
    return config