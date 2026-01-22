# μP + Progressive Training + SWA Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add μP (Maximal Update Parametrization), Progressive Training (sequence length curriculum), and EMA weight averaging to the SWA-MLA training pipeline for faster convergence and better generalization.

**Architecture:** μP provides foundation with per-layer LR scaling and proper initialization. Progressive training starts with short sequences (512) and gradually increases to target (2048). EMA maintains running average of weights for validation and final model.

**Tech Stack:** PyTorch, existing SWA-MLA codebase, no new dependencies

**Worktree:** `/home/orolol/workspace/swamla/.worktrees/mup-progressive-swa`

---

## Task 1: Create EMA Module (Simplest Component)

**Files:**
- Create: `optimization/swa.py`

**Step 1: Create the EMA module**

```python
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
                self.ema_params[name].lerp_(param.data.to(self.ema_params[name].device), 1 - self.decay)

    @contextmanager
    def apply(self, model: nn.Module):
        """
        Context manager that temporarily replaces model weights with EMA weights.

        Usage:
            with ema.apply(model):
                val_loss = validate(model)
            # Original weights are restored after the block
        """
        # Store original parameters
        original_params: Dict[str, torch.Tensor] = {}
        for name, param in model.named_parameters():
            if name in self.ema_params and param.requires_grad:
                original_params[name] = param.data.clone()
                param.data.copy_(self.ema_params[name].to(param.device))

        try:
            yield
        finally:
            # Restore original parameters
            for name, param in model.named_parameters():
                if name in original_params:
                    param.data.copy_(original_params[name])

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
```

**Step 2: Verify the file was created correctly**

Run: `python -c "from optimization.swa import EMAModel; print('EMAModel imported successfully')"`
Expected: `EMAModel imported successfully`

**Step 3: Quick unit test**

Run:
```bash
cd /home/orolol/workspace/swamla/.worktrees/mup-progressive-swa/models && python -c "
import torch
import torch.nn as nn
import sys
sys.path.insert(0, '..')
from optimization.swa import EMAModel

# Create simple model
model = nn.Linear(10, 10)
ema = EMAModel(model, decay=0.9)

# Update a few times
for _ in range(5):
    model.weight.data.add_(0.1)
    ema.update(model)

# Test apply context
original_weight = model.weight.data.clone()
with ema.apply(model):
    assert not torch.allclose(model.weight.data, original_weight), 'EMA weights should differ'
assert torch.allclose(model.weight.data, original_weight), 'Original weights should be restored'
print('EMA test PASSED')
"
```
Expected: `EMA test PASSED`

**Step 4: Commit**

```bash
git add optimization/swa.py
git commit -m "feat: add EMA (Exponential Moving Average) module

Implements continuous weight averaging for better generalization.
- Configurable decay (default 0.9999)
- Context manager for temporary weight swapping during validation
- State dict support for checkpointing

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 2: Create Progressive Training Scheduler

**Files:**
- Create: `optimization/progressive.py`

**Step 1: Create the progressive scheduler**

```python
"""Progressive Training Scheduler for sequence length curriculum."""

from dataclasses import dataclass
from typing import List, Tuple, Optional


@dataclass
class ProgressivePhase:
    """A single phase in progressive training."""
    seq_len: int
    end_tokens: int  # End this phase after this many tokens (cumulative)
    batch_size: int


class ProgressiveScheduler:
    """
    Manages progressive training phases with sequence length curriculum.

    Automatically adjusts seq_len and batch_size based on tokens seen,
    keeping effective batch size (in tokens) approximately constant.

    Usage:
        scheduler = ProgressiveScheduler.from_schedule(
            "512:500M,1024:2B,2048:inf",
            base_batch_size=4,
            target_seq_len=2048
        )

        # In training loop:
        seq_len, batch_size = scheduler.get_current_config(total_tokens)
        if scheduler.check_phase_transition(total_tokens):
            print(f"Transitioning to seq_len={seq_len}")
    """

    def __init__(self, phases: List[ProgressivePhase]):
        """
        Args:
            phases: List of ProgressivePhase objects, ordered by end_tokens
        """
        if not phases:
            raise ValueError("At least one phase is required")

        # Sort by end_tokens
        self.phases = sorted(phases, key=lambda p: p.end_tokens)
        self.current_phase_idx = 0
        self._last_phase_idx = 0

    @classmethod
    def from_schedule(
        cls,
        schedule: str,
        base_batch_size: int,
        target_seq_len: int = 2048,
    ) -> "ProgressiveScheduler":
        """
        Create scheduler from schedule string.

        Args:
            schedule: Format "seq_len:tokens,seq_len:tokens,..."
                      e.g., "512:500M,1024:2B,2048:inf"
                      Tokens can use K, M, B suffixes or "inf"
            base_batch_size: Batch size for the target (longest) sequence length
            target_seq_len: The final/target sequence length

        Returns:
            ProgressiveScheduler instance
        """
        phases = []

        # Calculate target tokens per batch (for constant effective batch)
        target_tokens_per_batch = base_batch_size * target_seq_len

        for phase_str in schedule.split(','):
            parts = phase_str.strip().split(':')
            if len(parts) != 2:
                raise ValueError(f"Invalid phase format: {phase_str}")

            seq_len = int(parts[0])
            tokens_str = parts[1].strip().lower()

            # Parse token count
            if tokens_str == 'inf':
                end_tokens = float('inf')
            else:
                multipliers = {'k': 1_000, 'm': 1_000_000, 'b': 1_000_000_000}
                if tokens_str[-1] in multipliers:
                    end_tokens = int(float(tokens_str[:-1]) * multipliers[tokens_str[-1]])
                else:
                    end_tokens = int(tokens_str)

            # Calculate batch size to maintain constant tokens/batch
            batch_size = max(1, target_tokens_per_batch // seq_len)

            phases.append(ProgressivePhase(
                seq_len=seq_len,
                end_tokens=end_tokens,
                batch_size=batch_size,
            ))

        return cls(phases)

    def get_current_config(self, total_tokens: int) -> Tuple[int, int]:
        """
        Get current seq_len and batch_size for the given token count.

        Args:
            total_tokens: Total tokens seen so far

        Returns:
            (seq_len, batch_size) tuple
        """
        # Find current phase
        for i, phase in enumerate(self.phases):
            if total_tokens < phase.end_tokens:
                self.current_phase_idx = i
                return phase.seq_len, phase.batch_size

        # Past all phases, use the last one
        self.current_phase_idx = len(self.phases) - 1
        last_phase = self.phases[-1]
        return last_phase.seq_len, last_phase.batch_size

    def check_phase_transition(self, total_tokens: int) -> bool:
        """
        Check if we just transitioned to a new phase.

        Returns True only once per transition.
        """
        self.get_current_config(total_tokens)  # Updates current_phase_idx

        if self.current_phase_idx != self._last_phase_idx:
            self._last_phase_idx = self.current_phase_idx
            return True
        return False

    def get_phase_info(self) -> str:
        """Return human-readable phase information."""
        phase = self.phases[self.current_phase_idx]
        total_phases = len(self.phases)
        return f"Phase {self.current_phase_idx + 1}/{total_phases}: seq_len={phase.seq_len}, batch_size={phase.batch_size}"

    def state_dict(self) -> dict:
        """Return state for checkpointing."""
        return {
            'current_phase_idx': self.current_phase_idx,
            '_last_phase_idx': self._last_phase_idx,
        }

    def load_state_dict(self, state_dict: dict) -> None:
        """Load state from checkpoint."""
        self.current_phase_idx = state_dict['current_phase_idx']
        self._last_phase_idx = state_dict['_last_phase_idx']
```

**Step 2: Verify import**

Run: `python -c "from optimization.progressive import ProgressiveScheduler; print('ProgressiveScheduler imported successfully')"`
Expected: `ProgressiveScheduler imported successfully`

**Step 3: Unit test**

Run:
```bash
cd /home/orolol/workspace/swamla/.worktrees/mup-progressive-swa/models && python -c "
import sys
sys.path.insert(0, '..')
from optimization.progressive import ProgressiveScheduler

# Test schedule parsing
scheduler = ProgressiveScheduler.from_schedule(
    '512:500M,1024:2B,2048:inf',
    base_batch_size=4,
    target_seq_len=2048
)

# Phase 1: 0 tokens
seq_len, batch_size = scheduler.get_current_config(0)
assert seq_len == 512, f'Expected 512, got {seq_len}'
assert batch_size == 16, f'Expected 16, got {batch_size}'  # 4 * 2048 / 512 = 16

# Phase 2: 1B tokens
seq_len, batch_size = scheduler.get_current_config(1_000_000_000)
assert seq_len == 1024, f'Expected 1024, got {seq_len}'
assert batch_size == 8, f'Expected 8, got {batch_size}'

# Phase 3: 3B tokens
seq_len, batch_size = scheduler.get_current_config(3_000_000_000)
assert seq_len == 2048, f'Expected 2048, got {seq_len}'
assert batch_size == 4, f'Expected 4, got {batch_size}'

# Test phase transition detection
scheduler2 = ProgressiveScheduler.from_schedule('512:100,1024:inf', base_batch_size=4, target_seq_len=1024)
assert scheduler2.check_phase_transition(0) == False  # Start, no transition yet
assert scheduler2.check_phase_transition(50) == False  # Still phase 1
assert scheduler2.check_phase_transition(150) == True  # Transition to phase 2!
assert scheduler2.check_phase_transition(200) == False  # Still phase 2, no new transition

print('ProgressiveScheduler test PASSED')
"
```
Expected: `ProgressiveScheduler test PASSED`

**Step 4: Commit**

```bash
git add optimization/progressive.py
git commit -m "feat: add ProgressiveScheduler for sequence length curriculum

Implements progressive training with automatic phase transitions:
- Parse schedule from string (e.g., '512:500M,1024:2B,2048:inf')
- Auto-adjust batch size to maintain constant tokens/batch
- Phase transition detection for logging
- State dict support for checkpointing

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 3: Create μP Module - Layer Classification

**Files:**
- Create: `optimization/mup.py`

**Step 1: Create μP module with layer classification**

```python
"""Maximal Update Parametrization (μP) for width-independent hyperparameters."""

import math
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
from enum import Enum

import torch
import torch.nn as nn


class LayerType(Enum):
    """Classification of layer types for μP scaling."""
    EMBEDDING = "embedding"      # wte, wpe: 1/√d init, 1x LR, no decay
    ATTENTION = "attention"      # Q,K,V,O: 1/√d_in init, 1/width LR
    MLP_IN = "mlp_in"           # gate/up proj: 1/√d_in init, 1/width LR
    MLP_OUT = "mlp_out"         # down proj: 1/√(width*d_in) init, 1/width LR
    LM_HEAD = "lm_head"         # output: zero init, 1/width LR, no decay
    NORM = "norm"               # RMSNorm, LayerNorm: skip (no scaling)
    ENGRAM_EMBED = "engram_embed"  # Engram tables: 1/√d_mem, 5x LR, no decay
    OTHER = "other"             # Biases, etc: no scaling


@dataclass
class MuPConfig:
    """Configuration for μP scaling."""
    base_width: int = 256       # Reference width for scaling
    output_mult: float = 1.0    # Output logits multiplier (temperature)
    width: int = 1024           # Actual model width (n_embd)

    @property
    def width_mult(self) -> float:
        """Width multiplier: actual_width / base_width."""
        return self.width / self.base_width


def classify_layer(name: str, param: torch.Tensor) -> LayerType:
    """
    Classify a parameter by its role in the model.

    Args:
        name: Parameter name (e.g., 'transformer.h.0.attn.q_proj.weight')
        param: The parameter tensor

    Returns:
        LayerType classification
    """
    name_lower = name.lower()

    # Embeddings
    if any(x in name_lower for x in ['wte', 'wpe', 'token_emb', 'pos_emb']):
        return LayerType.EMBEDDING

    # LM Head (output projection)
    if 'lm_head' in name_lower or name_lower.endswith('head.weight'):
        return LayerType.LM_HEAD

    # Engram embeddings (special: 5x LR)
    if 'engram' in name_lower and 'embed' in name_lower and 'table' in name_lower:
        return LayerType.ENGRAM_EMBED

    # Normalization layers (skip)
    if any(x in name_lower for x in ['norm', 'ln_', 'layernorm', 'rmsnorm']):
        return LayerType.NORM

    # Attention projections
    if any(x in name_lower for x in ['q_proj', 'k_proj', 'v_proj', 'qkv', 'w_q', 'w_k', 'w_v',
                                      'q_a_proj', 'q_b_proj', 'kv_a_proj', 'kv_b_proj',  # MLA
                                      'o_proj', 'out_proj', 'w_o']):
        return LayerType.ATTENTION

    # MLP layers
    if any(x in name_lower for x in ['mlp', 'ffn', 'feed_forward']):
        if any(x in name_lower for x in ['down', 'out', 'fc2', 'w2']):
            return LayerType.MLP_OUT
        elif any(x in name_lower for x in ['up', 'gate', 'in', 'fc1', 'w1', 'w3']):
            return LayerType.MLP_IN
        # Default MLP to MLP_IN (more common)
        return LayerType.MLP_IN

    # MoE expert layers (treat as MLP)
    if 'expert' in name_lower:
        if 'down' in name_lower:
            return LayerType.MLP_OUT
        return LayerType.MLP_IN

    # Default: OTHER (biases, etc.)
    return LayerType.OTHER


def get_mup_lr_scale(layer_type: LayerType, config: MuPConfig, engram_lr_mult: float = 5.0) -> float:
    """
    Get the learning rate scale factor for a layer type.

    Args:
        layer_type: The layer classification
        config: μP configuration
        engram_lr_mult: LR multiplier for Engram embeddings

    Returns:
        LR scale factor (multiply base_lr by this)
    """
    width_mult = config.width_mult

    if layer_type == LayerType.EMBEDDING:
        return 1.0  # Embeddings use base LR
    elif layer_type == LayerType.LM_HEAD:
        return 1.0 / width_mult  # Scale down with width
    elif layer_type == LayerType.ATTENTION:
        return 1.0 / width_mult
    elif layer_type == LayerType.MLP_IN:
        return 1.0 / width_mult
    elif layer_type == LayerType.MLP_OUT:
        return 1.0 / width_mult
    elif layer_type == LayerType.ENGRAM_EMBED:
        return engram_lr_mult  # Keep existing 5x multiplier
    elif layer_type == LayerType.NORM:
        return 1.0  # Norms use base LR
    else:
        return 1.0  # Default: base LR


def get_mup_weight_decay(layer_type: LayerType, base_wd: float) -> float:
    """Get weight decay for a layer type."""
    if layer_type in (LayerType.EMBEDDING, LayerType.LM_HEAD, LayerType.ENGRAM_EMBED, LayerType.NORM):
        return 0.0
    return base_wd
```

**Step 2: Verify import**

Run: `python -c "from optimization.mup import classify_layer, LayerType, MuPConfig; print('μP module imported successfully')"`
Expected: `μP module imported successfully`

**Step 3: Test layer classification**

Run:
```bash
cd /home/orolol/workspace/swamla/.worktrees/mup-progressive-swa/models && python -c "
import sys
sys.path.insert(0, '..')
from optimization.mup import classify_layer, LayerType, MuPConfig, get_mup_lr_scale
import torch

# Test classification
assert classify_layer('transformer.wte.weight', torch.zeros(1)) == LayerType.EMBEDDING
assert classify_layer('transformer.h.0.attn.q_proj.weight', torch.zeros(1)) == LayerType.ATTENTION
assert classify_layer('transformer.h.0.attn.o_proj.weight', torch.zeros(1)) == LayerType.ATTENTION
assert classify_layer('transformer.h.0.ffn.gate_up_proj.weight', torch.zeros(1)) == LayerType.MLP_IN
assert classify_layer('transformer.h.0.ffn.down_proj.weight', torch.zeros(1)) == LayerType.MLP_OUT
assert classify_layer('lm_head.weight', torch.zeros(1)) == LayerType.LM_HEAD
assert classify_layer('transformer.h.0.attn_norm.weight', torch.zeros(1)) == LayerType.NORM
assert classify_layer('engram.embeddings.tables.weight', torch.zeros(1)) == LayerType.ENGRAM_EMBED

# Test LR scaling
config = MuPConfig(base_width=256, width=1024)  # 4x width
assert config.width_mult == 4.0

embed_scale = get_mup_lr_scale(LayerType.EMBEDDING, config)
attn_scale = get_mup_lr_scale(LayerType.ATTENTION, config)
engram_scale = get_mup_lr_scale(LayerType.ENGRAM_EMBED, config)

assert embed_scale == 1.0, f'Embedding scale should be 1.0, got {embed_scale}'
assert attn_scale == 0.25, f'Attention scale should be 0.25 (1/4), got {attn_scale}'
assert engram_scale == 5.0, f'Engram scale should be 5.0, got {engram_scale}'

print('μP layer classification test PASSED')
"
```
Expected: `μP layer classification test PASSED`

**Step 4: Commit**

```bash
git add optimization/mup.py
git commit -m "feat: add μP layer classification and LR scaling

Implements layer type detection for μP:
- Classify layers by role (embedding, attention, MLP, etc.)
- Compute per-layer LR scale factors based on width ratio
- Special handling for Engram embeddings (5x LR)
- Weight decay rules per layer type

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 4: Add μP Initialization

**Files:**
- Modify: `optimization/mup.py` (append to existing file)

**Step 1: Add initialization functions to mup.py**

Append the following to `optimization/mup.py`:

```python


def mup_init_weight(param: torch.Tensor, layer_type: LayerType, config: MuPConfig) -> None:
    """
    Apply μP-compliant initialization to a weight tensor in-place.

    Args:
        param: Weight tensor to initialize
        layer_type: Classification of the layer
        config: μP configuration
    """
    width = config.width
    width_mult = config.width_mult

    if layer_type == LayerType.EMBEDDING:
        # Embeddings: standard 1/√d
        std = 1.0 / math.sqrt(width)
        nn.init.normal_(param, mean=0.0, std=std)

    elif layer_type == LayerType.LM_HEAD:
        # LM Head: zero init (critical for μP!)
        nn.init.zeros_(param)

    elif layer_type == LayerType.ATTENTION:
        # Attention: 1/√d_in
        fan_in = param.shape[1] if param.dim() >= 2 else param.shape[0]
        std = 1.0 / math.sqrt(fan_in)
        nn.init.normal_(param, mean=0.0, std=std)

    elif layer_type == LayerType.MLP_IN:
        # MLP input: 1/√d_in
        fan_in = param.shape[1] if param.dim() >= 2 else param.shape[0]
        std = 1.0 / math.sqrt(fan_in)
        nn.init.normal_(param, mean=0.0, std=std)

    elif layer_type == LayerType.MLP_OUT:
        # MLP output: 1/√(width * d_in) - scaled down by width
        fan_in = param.shape[1] if param.dim() >= 2 else param.shape[0]
        std = 1.0 / math.sqrt(width_mult * fan_in)
        nn.init.normal_(param, mean=0.0, std=std)

    elif layer_type == LayerType.ENGRAM_EMBED:
        # Engram: preserve existing init (1/√d_mem handled in engram.py)
        pass

    # NORM, OTHER: keep existing init


def mup_init(model: nn.Module, config: MuPConfig) -> None:
    """
    Apply μP-compliant initialization to all model parameters.

    This should be called AFTER model creation but BEFORE optimizer creation.

    Args:
        model: The model to initialize
        config: μP configuration
    """
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        layer_type = classify_layer(name, param)

        # Only reinitialize weight tensors, not biases
        if 'bias' not in name.lower() and param.dim() >= 2:
            mup_init_weight(param, layer_type, config)


def mup_scale_output(logits: torch.Tensor, config: MuPConfig) -> torch.Tensor:
    """
    Scale output logits for μP.

    In μP, the output logits are scaled by output_mult / width_mult
    to maintain consistent gradient magnitudes.

    Args:
        logits: Output logits from lm_head
        config: μP configuration

    Returns:
        Scaled logits
    """
    scale = config.output_mult / config.width_mult
    return logits * scale
```

**Step 2: Test initialization**

Run:
```bash
cd /home/orolol/workspace/swamla/.worktrees/mup-progressive-swa/models && python -c "
import sys
sys.path.insert(0, '..')
import torch
import torch.nn as nn
from optimization.mup import mup_init, MuPConfig, LayerType, classify_layer

# Create a simple model
class SimpleModel(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.wte = nn.Embedding(1000, width)
        self.q_proj = nn.Linear(width, width)
        self.down_proj = nn.Linear(width, width)
        self.lm_head = nn.Linear(width, 1000, bias=False)

    def forward(self, x):
        return self.lm_head(self.down_proj(self.q_proj(self.wte(x))))

model = SimpleModel(width=1024)
config = MuPConfig(base_width=256, width=1024)

# Apply μP init
mup_init(model, config)

# Check LM head is zero-initialized
assert torch.allclose(model.lm_head.weight, torch.zeros_like(model.lm_head.weight)), 'LM head should be zero'

# Check other weights are not zero
assert not torch.allclose(model.q_proj.weight, torch.zeros_like(model.q_proj.weight)), 'Q proj should not be zero'

# Check std of down_proj is scaled down (1/√(4 * 1024) ≈ 0.0156)
down_std = model.down_proj.weight.std().item()
expected_std = 1.0 / (4 * 1024) ** 0.5  # width_mult=4, fan_in=1024
assert abs(down_std - expected_std) < 0.01, f'down_proj std should be ~{expected_std:.4f}, got {down_std:.4f}'

print('μP initialization test PASSED')
"
```
Expected: `μP initialization test PASSED`

**Step 3: Commit**

```bash
git add optimization/mup.py
git commit -m "feat: add μP initialization functions

Adds mup_init() to apply μP-compliant weight initialization:
- Embeddings: 1/√d standard init
- Attention/MLP_IN: 1/√d_in init
- MLP_OUT: 1/√(width × d_in) scaled init
- LM Head: zero initialization (critical!)
- Output scaling for consistent gradients

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 5: Add μP Optimizer Configuration

**Files:**
- Modify: `optimization/mup.py` (append)

**Step 1: Add optimizer configuration to mup.py**

Append to `optimization/mup.py`:

```python


def configure_mup_optimizer(
    model: nn.Module,
    config: MuPConfig,
    base_lr: float,
    weight_decay: float = 0.1,
    betas: Tuple[float, float] = (0.9, 0.95),
    engram_lr_mult: float = 5.0,
    device_type: str = 'cuda',
    optimizer_type: str = 'adamw',
) -> torch.optim.Optimizer:
    """
    Configure optimizer with μP-scaled learning rates per layer.

    Args:
        model: The model to optimize
        config: μP configuration
        base_lr: Base learning rate (will be scaled per layer)
        weight_decay: Weight decay coefficient
        betas: Adam betas
        engram_lr_mult: LR multiplier for Engram embeddings
        device_type: 'cuda' or 'cpu'
        optimizer_type: 'adamw' or 'muon'

    Returns:
        Configured optimizer
    """
    # Group parameters by their LR scale and weight decay
    param_groups: Dict[Tuple[float, float], List[torch.Tensor]] = {}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        layer_type = classify_layer(name, param)
        lr_scale = get_mup_lr_scale(layer_type, config, engram_lr_mult)
        wd = get_mup_weight_decay(layer_type, weight_decay)

        key = (lr_scale, wd)
        if key not in param_groups:
            param_groups[key] = []
        param_groups[key].append(param)

    # Build optimizer param groups
    optimizer_groups = []
    for (lr_scale, wd), params in param_groups.items():
        if params:
            optimizer_groups.append({
                'params': params,
                'lr': base_lr * lr_scale,
                'weight_decay': wd,
            })

    # Handle Muon optimizer (requires special treatment)
    if optimizer_type == 'muon' and hasattr(torch.optim, 'Muon'):
        # Muon uses different LR scale and only for 2D params
        # We need to split: Muon for 2D weights, AdamW for rest
        Muon = torch.optim.Muon

        muon_groups = []
        adamw_groups = []

        for group in optimizer_groups:
            muon_params = []
            adamw_params = []

            for param in group['params']:
                if param.ndim == 2 and 'embed' not in str(id(param)):
                    muon_params.append(param)
                else:
                    adamw_params.append(param)

            if muon_params:
                # Muon uses 200x base LR internally
                muon_groups.append({
                    'params': muon_params,
                    'lr': group['lr'] * 200,
                    'weight_decay': group['weight_decay'],
                })
            if adamw_params:
                adamw_groups.append({
                    'params': adamw_params,
                    'lr': group['lr'],
                    'weight_decay': group['weight_decay'],
                })

        optimizers = []
        if muon_groups:
            optimizers.append(Muon(muon_groups, momentum=0.95, nesterov=True, ns_steps=5))
        if adamw_groups:
            use_fused = device_type == 'cuda'
            optimizers.append(torch.optim.AdamW(adamw_groups, betas=betas, fused=use_fused))

        return optimizers  # Returns list for Muon

    # Standard AdamW
    use_fused = device_type == 'cuda'
    return torch.optim.AdamW(optimizer_groups, lr=base_lr, betas=betas, fused=use_fused)


def get_mup_param_count_by_type(model: nn.Module) -> Dict[str, int]:
    """Get parameter counts grouped by layer type (for debugging)."""
    counts: Dict[str, int] = {}

    for name, param in model.named_parameters():
        layer_type = classify_layer(name, param)
        type_name = layer_type.value
        counts[type_name] = counts.get(type_name, 0) + param.numel()

    return counts
```

**Step 2: Test optimizer configuration**

Run:
```bash
cd /home/orolol/workspace/swamla/.worktrees/mup-progressive-swa/models && python -c "
import sys
sys.path.insert(0, '..')
import torch
import torch.nn as nn
from optimization.mup import configure_mup_optimizer, MuPConfig, get_mup_param_count_by_type

class SimpleModel(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.wte = nn.Embedding(1000, width)
        self.q_proj = nn.Linear(width, width)
        self.lm_head = nn.Linear(width, 1000, bias=False)

model = SimpleModel(width=1024)
config = MuPConfig(base_width=256, width=1024)

# Configure optimizer
optimizer = configure_mup_optimizer(
    model, config,
    base_lr=1e-4,
    weight_decay=0.1,
    device_type='cpu'
)

# Check we have multiple param groups with different LRs
assert len(optimizer.param_groups) > 1, 'Should have multiple param groups'

# Find the LRs
lrs = set(g['lr'] for g in optimizer.param_groups)
print(f'LRs in optimizer: {sorted(lrs)}')

# Should have base_lr (for embeddings) and base_lr/4 (for attention, scaled by width)
assert 1e-4 in lrs, 'Should have base LR for embeddings'
assert 0.25e-4 in lrs, 'Should have scaled LR (1/4) for attention'

# Test param count helper
counts = get_mup_param_count_by_type(model)
print(f'Param counts by type: {counts}')

print('μP optimizer configuration test PASSED')
"
```
Expected: `μP optimizer configuration test PASSED`

**Step 3: Commit**

```bash
git add optimization/mup.py
git commit -m "feat: add μP optimizer configuration

Adds configure_mup_optimizer() for per-layer LR scaling:
- Groups parameters by LR scale and weight decay
- Supports both AdamW and Muon optimizers
- Handles Engram special case (5x LR)
- Adds get_mup_param_count_by_type() for debugging

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 6: Update SWAMLAConfig with μP, Progressive, and EMA Fields

**Files:**
- Modify: `models/swa_mla_model.py`

**Step 1: Add new config fields**

Find the `SWAMLAConfig` class (around line 44) and add these fields after the Engram config section (after line 124):

```python
    # μP (Maximal Update Parametrization)
    use_mup: bool = False
    mup_base_width: int = 256  # Reference width for LR scaling
    mup_output_mult: float = 1.0  # Output logits multiplier

    # Progressive Training
    use_progressive: bool = False
    progressive_schedule: str = "512:500M,1024:2B,2048:inf"  # seq_len:tokens schedule

    # EMA (Exponential Moving Average)
    use_ema: bool = False
    ema_decay: float = 0.9999  # EMA decay factor
```

**Step 2: Add mup-1b preset**

Find the `create_swa_mla_model` function presets section (around line 450) and add a new preset:

```python
        "mup-1b": dict(
            n_layer=12,
            n_embd=1024,
            n_head=16,
            use_moe=True,
            n_experts=64,
            n_shared_experts=1,
            n_activated=2,
            use_latent_moe=True,
            latent_ratio=4,
            use_engram=True,
            engram_layers=[2, 6],
            engram_d_mem=512,
            # μP settings
            use_mup=True,
            mup_base_width=256,
        ),
```

**Step 3: Verify changes**

Run:
```bash
cd /home/orolol/workspace/swamla/.worktrees/mup-progressive-swa/models && python -c "
import sys
sys.path.insert(0, '.')
from swa_mla_model import SWAMLAConfig, create_swa_mla_model

# Test new config fields
config = SWAMLAConfig(
    use_mup=True,
    mup_base_width=256,
    use_progressive=True,
    progressive_schedule='512:500M,1024:2B,2048:inf',
    use_ema=True,
    ema_decay=0.9999,
)
assert config.use_mup == True
assert config.mup_base_width == 256
assert config.use_progressive == True
assert config.use_ema == True
print('Config fields verified')

# Test mup-1b preset
model = create_swa_mla_model('mup-1b', vocab_size=50257, block_size=2048)
assert model.config.use_mup == True
assert model.config.mup_base_width == 256
assert model.config.use_engram == True
print(f'mup-1b preset created: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params')
print('SWAMLAConfig update test PASSED')
"
```
Expected: Test passes with config fields and preset verified

**Step 4: Commit**

```bash
git add models/swa_mla_model.py
git commit -m "feat: add μP, Progressive, and EMA config fields

Updates SWAMLAConfig with:
- use_mup, mup_base_width, mup_output_mult for μP
- use_progressive, progressive_schedule for curriculum
- use_ema, ema_decay for weight averaging

Adds mup-1b preset with μP enabled.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 7: Integrate μP, Progressive, and EMA into train.py

**Files:**
- Modify: `train.py`

**Step 1: Add imports at the top (after existing imports, around line 80)**

Add after the existing imports section:

```python
# μP, Progressive Training, and EMA imports
try:
    from optimization.mup import MuPConfig, mup_init, configure_mup_optimizer, mup_scale_output
    from optimization.progressive import ProgressiveScheduler
    from optimization.swa import EMAModel
    MUP_AVAILABLE = True
except ImportError:
    MUP_AVAILABLE = False
```

**Step 2: Add CLI arguments (find argparser section, typically after line 600)**

Add these arguments after the Engram arguments:

```python
    # μP arguments
    parser.add_argument('--use_mup', action='store_true', help='Enable μP (Maximal Update Parametrization)')
    parser.add_argument('--mup_base_width', type=int, default=256, help='μP base width for LR scaling')

    # Progressive training arguments
    parser.add_argument('--use_progressive', action='store_true', help='Enable progressive sequence length training')
    parser.add_argument('--progressive_schedule', type=str, default='512:500M,1024:2B,2048:inf',
                        help='Progressive schedule: seq_len:tokens,... (e.g., "512:500M,1024:2B,2048:inf")')

    # EMA arguments
    parser.add_argument('--use_ema', action='store_true', help='Enable EMA weight averaging')
    parser.add_argument('--ema_decay', type=float, default=0.9999, help='EMA decay factor')
```

**Step 3: Initialize components after model creation (find model creation section)**

After model is created and moved to device, add:

```python
    # Initialize μP if enabled
    if args.use_mup and MUP_AVAILABLE:
        mup_config = MuPConfig(
            base_width=args.mup_base_width,
            width=model.config.n_embd,
        )
        mup_init(model, mup_config)
        if rank == 0:
            print(f"μP initialized: base_width={mup_config.base_width}, width_mult={mup_config.width_mult:.1f}x")
    else:
        mup_config = None

    # Initialize Progressive Scheduler if enabled
    if args.use_progressive and MUP_AVAILABLE:
        progressive = ProgressiveScheduler.from_schedule(
            args.progressive_schedule,
            base_batch_size=args.batch_size,
            target_seq_len=args.block_size,
        )
        if rank == 0:
            print(f"Progressive training: {args.progressive_schedule}")
    else:
        progressive = None

    # Initialize EMA if enabled
    if args.use_ema and MUP_AVAILABLE:
        ema = EMAModel(model, decay=args.ema_decay)
        if rank == 0:
            print(f"EMA enabled: decay={args.ema_decay}")
    else:
        ema = None
```

**Step 4: Modify optimizer configuration**

Replace the optimizer configuration call with:

```python
    # Configure optimizer (μP-aware if enabled)
    if mup_config is not None:
        optimizer = configure_mup_optimizer(
            raw_model, mup_config,
            base_lr=args.learning_rate,
            weight_decay=args.weight_decay,
            betas=(0.9, 0.95),
            engram_lr_mult=args.engram_lr_multiplier,
            device_type=device_type,
            optimizer_type=args.optimizer_type,
        )
    else:
        optimizer = configure_optimizer(
            raw_model, args.learning_rate, args.weight_decay,
            betas=(0.9, 0.95), device_type=device_type,
            optimizer_type=args.optimizer_type,
            engram_lr_multiplier=args.engram_lr_multiplier,
        )
```

**Step 5: Update training loop for progressive and EMA**

In the training loop, add progressive check at the start of each iteration:

```python
        # Progressive training: check for phase transition
        if progressive is not None:
            new_seq_len, new_batch_size = progressive.get_current_config(total_tokens)
            if progressive.check_phase_transition(total_tokens):
                if rank == 0:
                    print(f"\n>>> Progressive transition: seq_len={new_seq_len}, batch_size={new_batch_size}")
                # Update dataloader with new seq_len if needed
                # Note: This requires dataloader recreation or dynamic adjustment
```

After optimizer.step(), add EMA update:

```python
        # Update EMA weights
        if ema is not None:
            ema.update(raw_model)
```

**Step 6: Update validation to use EMA**

Wrap validation call with EMA context:

```python
        # Validation with EMA weights if enabled
        if ema is not None:
            with ema.apply(raw_model):
                val_loss = validate(model, val_loader, device, ...)
        else:
            val_loss = validate(model, val_loader, device, ...)
```

**Step 7: Update checkpointing to save EMA and progressive state**

Add to checkpoint saving:

```python
        checkpoint = {
            'model': raw_model.state_dict(),
            'optimizer': optimizer.state_dict() if not isinstance(optimizer, list) else [o.state_dict() for o in optimizer],
            'step': step,
            'total_tokens': total_tokens,
            'config': vars(args),
        }

        # Save EMA state if enabled
        if ema is not None:
            checkpoint['ema'] = ema.state_dict()

        # Save progressive scheduler state if enabled
        if progressive is not None:
            checkpoint['progressive'] = progressive.state_dict()
```

**Step 8: Verify the integration compiles**

Run:
```bash
cd /home/orolol/workspace/swamla/.worktrees/mup-progressive-swa && python -c "
import train
print('train.py imports successfully')
"
```
Expected: `train.py imports successfully`

**Step 9: Commit**

```bash
git add train.py
git commit -m "feat: integrate μP, Progressive Training, and EMA into train.py

Training loop now supports:
- μP initialization and optimizer configuration
- Progressive sequence length curriculum with phase transitions
- EMA weight averaging with validation using EMA weights
- Checkpointing of EMA and progressive scheduler state

CLI flags: --use_mup, --use_progressive, --use_ema

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 8: Create Training Script

**Files:**
- Create: `scripts/train_mup_progressive.sh`

**Step 1: Create the training script**

```bash
#!/bin/bash

# Train with μP + Progressive Training + EMA
#
# Architecture:
# - μP: Maximal Update Parametrization for width-independent hyperparameters
# - Progressive: Sequence length curriculum (512→1024→2048)
# - EMA: Exponential moving average for better generalization
#
# Auto-detects available GPUs and launches DDP training if multiple GPUs are found

# Add Flash Attention 3 (Hopper) to PYTHONPATH if available
FA3_PATH="$HOME/.local/flash-attention-3/flash-attention/hopper"
if [ -d "$FA3_PATH" ]; then
    export PYTHONPATH="$FA3_PATH:$PYTHONPATH"
    echo "Flash Attention 3 path added: $FA3_PATH"
fi

BATCH_SIZE=${1:-4}
BLOCK_SIZE=${2:-2048}
OUTPUT_DIR=${3:-outputs/mup_progressive}
RESUME_FROM=${4:-false}
OPTIMIZER=${5:-muon}
HF_REPO_ID=${6:-}
USE_TENSORBOARD=${7:-true}

# μP configuration
MUP_BASE_WIDTH=${MUP_BASE_WIDTH:-256}

# Progressive training configuration
PROGRESSIVE_SCHEDULE=${PROGRESSIVE_SCHEDULE:-"512:500M,1024:2B,2048:inf"}

# EMA configuration
EMA_DECAY=${EMA_DECAY:-0.9999}

# Auto-detect number of GPUs
if command -v nvidia-smi &> /dev/null; then
    NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
else
    NUM_GPUS=0
fi

echo "==========================================="
echo "Training with μP + Progressive + EMA"
echo "==========================================="
echo "Batch size: $BATCH_SIZE"
echo "Block size: $BLOCK_SIZE"
echo "Output dir: $OUTPUT_DIR"
echo "Optimizer: $OPTIMIZER"
echo "Detected GPUs: $NUM_GPUS"
echo ""
echo "μP Configuration:"
echo "  Base width: $MUP_BASE_WIDTH"
echo ""
echo "Progressive Training:"
echo "  Schedule: $PROGRESSIVE_SCHEDULE"
echo ""
echo "EMA:"
echo "  Decay: $EMA_DECAY"
echo ""

# Build arguments
HF_REPO_ARG=""
if [ -n "$HF_REPO_ID" ]; then
    HF_REPO_ARG="--hf_repo_id $HF_REPO_ID"
fi

RESUME_ARG=""
if [ "$RESUME_FROM" = "true" ]; then
    RESUME_ARG="--resume_from_hf"
elif [ "$RESUME_FROM" != "false" ] && [ -n "$RESUME_FROM" ]; then
    RESUME_ARG="--resume_from $RESUME_FROM"
fi

TB_ARG=""
if [ "$USE_TENSORBOARD" = "true" ]; then
    TB_ARG="--use_tensorboard"
fi

COMMON_ARGS="--size mup-1b \
    --batch_size $BATCH_SIZE \
    --block_size $BLOCK_SIZE \
    --output_dir $OUTPUT_DIR \
    --optimizer_type $OPTIMIZER \
    --learning_rate 1e-4 \
    --weight_decay 0.1 \
    --warmup_iters 400 \
    --max_iters 1000000 \
    --grad_clip 1.0 \
    --gradient_accumulation_steps 1 \
    --gradient_checkpointing \
    --num_workers 8 \
    --use_mup \
    --mup_base_width $MUP_BASE_WIDTH \
    --use_progressive \
    --progressive_schedule $PROGRESSIVE_SCHEDULE \
    --use_ema \
    --ema_decay $EMA_DECAY \
    --compile \
    --compile_mode max-autotune \
    --log_interval 50 \
    --eval_interval 5000 \
    --save_interval 5000 \
    $HF_REPO_ARG \
    $RESUME_ARG \
    $TB_ARG"

# Launch training
if [ $NUM_GPUS -gt 1 ]; then
    echo "Launching DDP training on $NUM_GPUS GPUs..."
    echo ""
    torchrun --standalone --nproc_per_node=$NUM_GPUS train.py $COMMON_ARGS
else
    echo "Launching single GPU training..."
    echo ""
    python train.py $COMMON_ARGS
fi

exit 0
```

**Step 2: Make executable and verify**

Run:
```bash
chmod +x /home/orolol/workspace/swamla/.worktrees/mup-progressive-swa/scripts/train_mup_progressive.sh
ls -la /home/orolol/workspace/swamla/.worktrees/mup-progressive-swa/scripts/train_mup_progressive.sh
```
Expected: Script exists and is executable

**Step 3: Commit**

```bash
git add scripts/train_mup_progressive.sh
git commit -m "feat: add training script for μP + Progressive + EMA

Provides train_mup_progressive.sh with:
- mup-1b preset (1B params with MoE + Engram)
- μP with configurable base width
- Progressive schedule: 512→1024→2048
- EMA with 0.9999 decay
- Auto GPU detection for DDP

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 9: End-to-End Smoke Test

**Files:**
- None (testing only)

**Step 1: Run minimal training test**

Run:
```bash
cd /home/orolol/workspace/swamla/.worktrees/mup-progressive-swa/models && python -c "
import sys
sys.path.insert(0, '..')
import torch
from swa_mla_model import create_swa_mla_model
from optimization.mup import MuPConfig, mup_init, configure_mup_optimizer
from optimization.progressive import ProgressiveScheduler
from optimization.swa import EMAModel

# Create small model for testing
model = create_swa_mla_model('small', vocab_size=1000, block_size=512)
model = model.cuda() if torch.cuda.is_available() else model

# μP setup
mup_config = MuPConfig(base_width=256, width=model.config.n_embd)
mup_init(model, mup_config)
print(f'μP initialized: width_mult={mup_config.width_mult:.2f}x')

# Optimizer
device = 'cuda' if torch.cuda.is_available() else 'cpu'
optimizer = configure_mup_optimizer(model, mup_config, base_lr=1e-4, device_type=device)
print(f'Optimizer configured: {len(optimizer.param_groups)} param groups')

# Progressive scheduler
progressive = ProgressiveScheduler.from_schedule('256:100,512:inf', base_batch_size=2, target_seq_len=512)
seq_len, batch_size = progressive.get_current_config(0)
print(f'Progressive: starting at seq_len={seq_len}, batch_size={batch_size}')

# EMA
ema = EMAModel(model, decay=0.999)
print('EMA initialized')

# Mini training step
x = torch.randint(0, 1000, (batch_size, seq_len), device=device)
model.train()
loss, _ = model(x, targets=x)
loss.backward()
optimizer.step()
optimizer.zero_grad()
ema.update(model)

print(f'Training step completed, loss={loss.item():.4f}')

# Validation with EMA
model.eval()
with torch.no_grad():
    with ema.apply(model):
        val_loss, _ = model(x, targets=x)
print(f'Validation with EMA, loss={val_loss.item():.4f}')

print('\\n=== SMOKE TEST PASSED ===')
"
```
Expected: `=== SMOKE TEST PASSED ===`

**Step 2: Commit test success**

```bash
git commit --allow-empty -m "test: verify μP + Progressive + EMA integration

Smoke test confirms:
- μP initialization with width scaling
- μP optimizer with per-layer LR
- Progressive scheduler phase management
- EMA weight averaging and validation swap

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 10: Update optimization/__init__.py

**Files:**
- Modify: `optimization/__init__.py`

**Step 1: Update exports**

Read current file and update to include new modules:

```python
"""Optimization utilities for SWA-MLA training."""

# FP8 training (Transformer Engine)
try:
    from .fp8_te import te_checkpoint
except ImportError:
    te_checkpoint = None

# μP (Maximal Update Parametrization)
try:
    from .mup import (
        MuPConfig,
        LayerType,
        classify_layer,
        mup_init,
        configure_mup_optimizer,
        get_mup_lr_scale,
        get_mup_weight_decay,
        mup_scale_output,
    )
except ImportError:
    MuPConfig = None

# Progressive Training
try:
    from .progressive import ProgressiveScheduler, ProgressivePhase
except ImportError:
    ProgressiveScheduler = None

# EMA (Stochastic Weight Averaging)
try:
    from .swa import EMAModel
except ImportError:
    EMAModel = None

__all__ = [
    'te_checkpoint',
    'MuPConfig',
    'LayerType',
    'classify_layer',
    'mup_init',
    'configure_mup_optimizer',
    'get_mup_lr_scale',
    'get_mup_weight_decay',
    'mup_scale_output',
    'ProgressiveScheduler',
    'ProgressivePhase',
    'EMAModel',
]
```

**Step 2: Verify imports work**

Run:
```bash
cd /home/orolol/workspace/swamla/.worktrees/mup-progressive-swa && python -c "
from optimization import MuPConfig, ProgressiveScheduler, EMAModel
print('All optimization imports work')
print(f'MuPConfig: {MuPConfig}')
print(f'ProgressiveScheduler: {ProgressiveScheduler}')
print(f'EMAModel: {EMAModel}')
"
```
Expected: All imports work

**Step 3: Commit**

```bash
git add optimization/__init__.py
git commit -m "feat: export μP, Progressive, and EMA from optimization package

Updates optimization/__init__.py to export:
- MuPConfig, mup_init, configure_mup_optimizer
- ProgressiveScheduler, ProgressivePhase
- EMAModel

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Summary

**Total Tasks:** 10

**Files Created:**
- `optimization/swa.py` - EMA module
- `optimization/progressive.py` - Progressive scheduler
- `optimization/mup.py` - μP implementation
- `scripts/train_mup_progressive.sh` - Training script

**Files Modified:**
- `models/swa_mla_model.py` - Config fields + preset
- `train.py` - Full integration
- `optimization/__init__.py` - Exports

**Estimated Implementation Time:** ~530 lines of code

**Key Verification Points:**
- Each task has unit tests before commit
- Task 9 is end-to-end smoke test
- All commits are atomic and independently reviewable
