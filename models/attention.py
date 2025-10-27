"""Attention mechanisms for transformer models."""

import math
import traceback
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch.utils.checkpoint import _StopRecomputationError as _CheckpointStop
except ImportError:  # pragma: no cover - older torch versions export it elsewhere
    _CheckpointStop = None

try:
    from flash_attn import flash_attn_func as flash_attn_func_2
    FLASH_ATTENTION_AVAILABLE = True
except ImportError as e:
    print(f"Flash Attention 2 not available: {e}")
    FLASH_ATTENTION_AVAILABLE = False

try:
    import xformers.ops as xops
    XFORMERS_AVAILABLE = True
except ImportError as e:
    print(f"Xformers not available: {e}")
    XFORMERS_AVAILABLE = False

ATTENTION_BACKENDS = {
    'flash_attn_2': FLASH_ATTENTION_AVAILABLE,
    'xformers': XFORMERS_AVAILABLE,
    'sdpa': hasattr(F, 'scaled_dot_product_attention'),
    'standard': True
}

def get_best_attention_backend():
    if ATTENTION_BACKENDS['flash_attn_2']:
        return 'flash_attn_2'
    elif ATTENTION_BACKENDS['xformers']:
        return 'xformers'
    elif ATTENTION_BACKENDS['sdpa']:
        return 'sdpa'
    else:
        return 'standard'

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        
        # Grouped-Query Attention (GQA)
        self.n_head = config.n_head
        self.n_head_kv = config.n_head // config.ratio_kv
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        
        # Projections
        self.q_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.k_proj = nn.Linear(config.n_embd, self.n_head_kv * self.head_dim, bias=config.bias)
        self.v_proj = nn.Linear(config.n_embd, self.n_head_kv * self.head_dim, bias=config.bias)
        self.o_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        
        # Regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # Optional features for positional handling and masking
        self.use_rope = getattr(config, 'use_rope', True)
        self.rope_theta = getattr(config, 'rope_theta', 10000)
        self.attention_window = getattr(config, 'attention_window', getattr(config, 'sliding_window_size', None))
        if self.attention_window is not None and self.attention_window <= 0:
            self.attention_window = None

        # Attention sink: always attend to first N tokens (improves stability)
        self.attention_sink_size = getattr(config, 'attention_sink_size', 4)

        scale_base = getattr(config, 'logit_scale_base', None)
        if scale_base is not None and scale_base <= 1.0:
            scale_base = None
        self.scale_base = scale_base
        self.scale_window = max(1, getattr(config, 'logit_scale_window', 128))
        self.scale_offset = getattr(config, 'logit_scale_offset', 0)
        self.scale_min = getattr(config, 'logit_scale_min', 1.0)
        self.scale_max = getattr(config, 'logit_scale_max', None)
        self.scale_in_training = getattr(config, 'logit_scale_during_training', False)
        self._scale_log_denom = math.log(self.scale_base) if self.scale_base is not None else None

        # Attention backend setup
        if hasattr(config, 'attention_backend') and config.attention_backend is not None:
            if config.attention_backend not in ATTENTION_BACKENDS:
                raise ValueError(f"Attention backend {config.attention_backend} not available")
            if not ATTENTION_BACKENDS[config.attention_backend]:
                print(f"Warning: {config.attention_backend} not available, falling back to best available backend")
                self.attention_backend = get_best_attention_backend()
            else:
                self.attention_backend = config.attention_backend
        else:
            self.attention_backend = get_best_attention_backend()

        if self.attention_window is not None and self.attention_backend in ('flash_attn_2', 'xformers'):
            # Fallback to SDPA when explicit masking is required
            self.attention_backend = 'sdpa'

        print(f"Using attention backend: {self.attention_backend}")

        # For RoPE positioning when enabled
        from positional_encoding import RoPE
        self.rope = RoPE(self.head_dim, config.block_size, base=self.rope_theta) if self.use_rope else None

        # Préallouer le masque causal
        mask = torch.full((config.block_size, config.block_size), float('-inf'))
        mask = torch.triu(mask, diagonal=1)
        self.register_buffer('mask', mask)
        self._sliding_mask_cache = None

    def _memory_efficient_attention(self, q, k, v, mask=None, is_causal=True):
        """
        Wrapper pour l'attention memory efficient avec gestion des erreurs
        """
        try:
            # Ensure all tensors have the same dtype - use the dtype from the model weights
            # This ensures compatibility with bfloat16, float16, or float32 models
            working_dtype = q.dtype  # Use the existing dtype of the input tensors
            # No need to convert q, k, v as they should already be in the correct dtype
            # q = q.to(working_dtype)  # Commented out - already in correct dtype
            # k = k.to(working_dtype)  # Commented out - already in correct dtype
            # v = v.to(working_dtype)  # Commented out - already in correct dtype
            
            # Préparer les tenseurs avec memory pinning pour un transfert plus rapide
            q = q.contiguous()
            k = k.contiguous()
            v = v.contiguous()
            
            # Gérer l'expansion pour GQA si nécessaire
            if k.size(1) != q.size(1):  # Si les dimensions ne correspondent pas (cas GQA)
                # Expand k et v pour correspondre au nombre de têtes de q
                k = k.expand(-1, q.size(1), -1, -1)
                v = v.expand(-1, q.size(1), -1, -1)
            
            # Reshape pour xformers [B, H, T, D] -> [B, T, H, D]
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            
            # Utiliser un masque optimisé pour xformers
            if is_causal:
                attn_bias = xops.LowerTriangularMask()
            else:
                # Convertir le masque en bias d'attention pour xformers si nécessaire
                attn_bias = xops.AttentionBias.from_mask(mask) if mask is not None else None
            
            # Utiliser la mise en cache des KV pour la génération
            if hasattr(self, '_cached_k') and hasattr(self, '_cached_v'):
                k = torch.cat([self._cached_k, k], dim=1)
                v = torch.cat([self._cached_v, v], dim=1)
                if self.attention_window is not None and k.size(1) > self.attention_window:
                    k = k[:, -self.attention_window:, :, :]
                    v = v[:, -self.attention_window:, :, :]
                # Ensure cached tensors have the same dtype
                k = k.to(working_dtype)
                v = v.to(working_dtype)
                self._cached_k = k
                self._cached_v = v
            
            y = xops.memory_efficient_attention(
                q, k, v,
                attn_bias=attn_bias,
                p=self.attn_dropout.p if self.training else 0.0,
                scale=1.0 / math.sqrt(self.head_dim),
                op=None  # Let xformers choose the best operator
            )
            
            # Reshape back [B, T, H, D] -> [B, H, T, D]
            return y.transpose(1, 2)
            
        except Exception as e:
            print(f"xformers memory efficient attention failed: {e}")
            print(f"Tensor dtypes - q: {q.dtype}, k: {k.dtype}, v: {v.dtype}")
            return None

    def _flash_attention(self, q, k, v, mask=None, is_causal=True):
        """
        Wrapper pour Flash Attention 2 avec optimisations
        """
        try:
            # Convert inputs to bfloat16
            q = q.to(torch.bfloat16)
            k = k.to(torch.bfloat16)
            v = v.to(torch.bfloat16)
            
            # Handle GQA by expanding k and v heads
            if k.size(1) != q.size(1):  # If number of heads don't match (GQA case)
                # Repeat k and v heads to match q heads
                k = k.repeat_interleave(self.n_head // self.n_head_kv, dim=1)
                v = v.repeat_interleave(self.n_head // self.n_head_kv, dim=1)
            
            # Utiliser la mise en cache des KV pour la génération
            if hasattr(self, '_cached_k') and hasattr(self, '_cached_v'):
                k = torch.cat([self._cached_k, k], dim=2)  # dim=2 car Flash Attention utilise [B, H, T, D]
                v = torch.cat([self._cached_v, v], dim=2)
                if self.attention_window is not None and k.size(-2) > self.attention_window:
                    k = k[:, :, -self.attention_window:, :]
                    v = v[:, :, -self.attention_window:, :]
                self._cached_k = k
                self._cached_v = v
            
            with torch.amp.autocast(enabled=True, device_type='cuda'):
                # Appliquer Flash Attention avec les optimisations
                output = flash_attn_func_2(
                    q, k, v,
                    causal=is_causal,
                    softmax_scale=1.0 / math.sqrt(self.head_dim)
                )
            
            return output
            
        except Exception as e:
            print(f"Flash Attention 2 failed: {e}")
            print(f"Input dtypes - q: {q.dtype}, k: {k.dtype}, v: {v.dtype}")
            print(f"Input shapes - q: {q.shape}, k: {k.shape}, v: {v.shape}")
            return None

    def _sdpa_attention(self, q, k, v, mask=None, is_causal=True):
        """
        Wrapper pour Scaled Dot Product Attention avec optimisations
        """
        try:
            # Utiliser la mise en cache des KV pour la génération
            if hasattr(self, '_cached_k') and hasattr(self, '_cached_v'):
                k = torch.cat([self._cached_k, k], dim=2)
                v = torch.cat([self._cached_v, v], dim=2)
                if self.attention_window is not None and k.size(-2) > self.attention_window:
                    k = k[:, :, -self.attention_window:, :]
                    v = v[:, :, -self.attention_window:, :]
                self._cached_k = k
                self._cached_v = v
            
            # Utiliser SDPA avec optimisations
            return F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None if is_causal else mask,
                dropout_p=self.attn_dropout.p if self.training else 0.0,
                is_causal=is_causal,
                scale=1.0 / math.sqrt(self.head_dim)
            )
        except Exception as e:
            print(f"SDPA failed: {e}")
            return None

    def _standard_attention(self, q, k, v, mask=None, is_causal=True):
        """
        Implementation standard de l'attention avec optimisations mémoire
        """
        # Utiliser la mise en cache des KV pour la génération
        if hasattr(self, '_cached_k') and hasattr(self, '_cached_v'):
            k = torch.cat([self._cached_k, k], dim=2)
            v = torch.cat([self._cached_v, v], dim=2)
            if self.attention_window is not None and k.size(-2) > self.attention_window:
                k = k[:, :, -self.attention_window:, :]
                v = v[:, :, -self.attention_window:, :]
            self._cached_k = k
            self._cached_v = v
        
        # Calculer l'attention avec optimisations mémoire
        scale = 1.0 / math.sqrt(self.head_dim)
        
        # Utiliser torch.baddbmm pour une multiplication matricielle plus efficace
        att = torch.empty(q.shape[:-2] + (q.shape[-2], k.shape[-2]), 
                         dtype=q.dtype, device=q.device)
        att = torch.baddbmm(
            att, q, k.transpose(-2, -1),
            beta=0, alpha=scale
        )
        
        # Clipping pour stabilité numérique
        att = torch.clamp(att, min=-1e4, max=1e4)
        
        # Appliquer le masque causal si nécessaire
        if is_causal:
            if mask is not None:
                att = att + mask
        
        # Utiliser float32 pour le softmax pour plus de stabilité
        att = F.softmax(att.float(), dim=-1).to(q.dtype)
        att = self.attn_dropout(att)
        
        # Utiliser torch.baddbmm pour le produit final
        out = torch.empty(att.shape[:-2] + (att.shape[-2], v.shape[-1]), 
                         dtype=q.dtype, device=q.device)
        out = torch.bmm(att, v)

        return out

    def _build_sliding_mask(self, T: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        window = self.attention_window
        if window is None:
            return torch.zeros((T, T), device=device, dtype=dtype)

        compiling = False
        try:
            import torch._dynamo as _dynamo  # type: ignore
            compiling = bool(getattr(_dynamo, 'is_compiling', lambda: False)())
        except Exception:
            compiling = False

        cache = self._sliding_mask_cache if not compiling else None
        cache_key = (T, str(device), str(dtype), self.attention_sink_size)
        if cache is not None and cache.get('key') == cache_key:
            return cache['mask']

        positions = torch.arange(T, device=device)
        diff = positions.unsqueeze(1) - positions.unsqueeze(0)
        mask = torch.zeros((T, T), device=device, dtype=dtype)

        # Apply sliding window mask: block attention beyond window size
        mask = mask.masked_fill(diff > window, float('-inf'))

        # Attention sink: always allow attention to first N tokens
        if self.attention_sink_size > 0:
            # Create a mask that allows attention to sink tokens
            # For each query position, unmask the first attention_sink_size positions
            sink_mask = torch.arange(T, device=device).unsqueeze(0) < self.attention_sink_size
            sink_mask = sink_mask.expand(T, T)
            # Set mask to 0 (allow attention) for sink tokens
            mask = torch.where(sink_mask, torch.zeros_like(mask), mask)

        if not compiling:
            self._sliding_mask_cache = {'key': cache_key, 'mask': mask}
        return mask

    def _get_causal_mask(self, base_mask: Optional[torch.Tensor], T: int, device: torch.device, dtype: torch.dtype) -> Optional[torch.Tensor]:
        if self.attention_window is None:
            return None
        sliding_mask = self._build_sliding_mask(T, device, dtype)
        if base_mask is None:
            base_mask = torch.zeros((T, T), device=device, dtype=dtype)
        return base_mask + sliding_mask

    def _compute_logit_scale(self, T: int, device: torch.device, dtype: torch.dtype) -> Optional[torch.Tensor]:
        if self.scale_base is None or self._scale_log_denom is None:
            return None
        if not self.scale_in_training and self.training:
            return None
        positions = torch.arange(T, device=device, dtype=torch.float32)
        positions = positions + float(self.scale_offset)
        window = float(self.scale_window)
        window_idx = torch.div(positions, window, rounding_mode='floor')
        scale = torch.log(self.scale_base + window_idx) / self._scale_log_denom
        if self.scale_min is not None:
            scale = torch.clamp(scale, min=float(self.scale_min))
        if self.scale_max is not None:
            scale = torch.clamp(scale, max=float(self.scale_max))
        return scale.to(dtype)

    def forward(self, x, key_value=None, is_generation=False):
        B, T, C = x.size()

        # Ensure consistent dtype for all tensors - use the model's weight dtype
        # Get the dtype from the model's weight parameters (e.g., q_proj)
        working_dtype = self.q_proj.weight.dtype
        x = x.to(working_dtype)
        
        try:
            # Correction des NaN/Inf (sans branchement dépendant des données pour torch.compile)
            x = torch.nan_to_num(x, nan=0.0, posinf=1e4, neginf=-1e4)
            
            # Cross-attention
            if key_value is not None:
                q = self.q_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
                k = self.k_proj(key_value)
                v = self.v_proj(key_value)
                
                k = k.view(B, key_value.size(1), self.n_head_kv, self.head_dim).transpose(1, 2)
                v = v.view(B, key_value.size(1), self.n_head_kv, self.head_dim).transpose(1, 2)
                
                # Convert k and v to the same dtype as q
                k = k.to(working_dtype)
                v = v.to(working_dtype)
                
                # Répéter K,V pour GQA
                k = k.repeat_interleave(self.n_head // self.n_head_kv, dim=1)
                v = v.repeat_interleave(self.n_head // self.n_head_kv, dim=1)
                
                # Apply RoPE to queries and keys
                if self.rope is not None:
                    q = self.rope(q)
                    k = self.rope(k)
                
            else:
                # Self-attention
                qkv = torch.cat([
                    self.q_proj(x),
                    self.k_proj(x),
                    self.v_proj(x)
                ], dim=-1)
                
                # Ensure consistent dtype
                qkv = qkv.to(working_dtype)
                
                q, k, v = qkv.split([self.n_embd, self.n_head_kv * self.head_dim, self.n_head_kv * self.head_dim], dim=-1)
                
                q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
                k = k.view(B, T, self.n_head_kv, self.head_dim).transpose(1, 2)
                v = v.view(B, T, self.n_head_kv, self.head_dim).transpose(1, 2)
                
                # Répéter K,V pour GQA si ce n'est pas de la génération
                if not is_generation:
                    k = k.repeat_interleave(self.n_head // self.n_head_kv, dim=1)
                    v = v.repeat_interleave(self.n_head // self.n_head_kv, dim=1)
                
                # Apply RoPE to queries and keys
                if self.rope is not None:
                    q = self.rope(q)
                    k = self.rope(k)

            # Déterminer si nous sommes en mode causal et préparer le masque
            if key_value is None:
                scale = self._compute_logit_scale(T, q.device, q.dtype)
                if scale is not None:
                    q = q * scale.view(1, 1, T, 1)

            is_causal = key_value is None  # Causal seulement pour self-attention
            base_mask = None
            if is_causal:
                base_mask = self.mask[:T, :T].to(device=q.device, dtype=working_dtype)
            attn_mask = self._get_causal_mask(base_mask, T, q.device, working_dtype) if is_causal else None
            mask_for_standard = attn_mask if attn_mask is not None else base_mask
            
            y = None
            # During generation or cross-attention, use SDPA instead of Flash Attention
            if attn_mask is not None or is_generation or key_value is not None:
                y = self._sdpa_attention(q, k, v, attn_mask, is_causal)
            else:
                # Try Flash Attention first
                if self.attention_backend == 'flash_attn_2':
                    y = self._flash_attention(q, k, v, attn_mask, is_causal)
            
            if y is None and self.attention_backend == 'xformers':
                y = self._memory_efficient_attention(q, k, v, attn_mask, is_causal)
            
            if y is None and self.attention_backend == 'sdpa':
                y = self._sdpa_attention(q, k, v, attn_mask, is_causal)
            
            if y is None:
                # Standard attention
                scale = 1.0 / math.sqrt(self.head_dim)
                att = torch.matmul(q, k.transpose(-2, -1)) * scale
                
                # Clipping pour stabilité numérique
                att = torch.clamp(att, min=-1e4, max=1e4)
                
                # Appliquer le masque causal si nécessaire
                if is_causal and mask_for_standard is not None:
                    att = att + mask_for_standard
                
                # Utiliser float32 pour le softmax pour plus de stabilité
                att = F.softmax(att.float(), dim=-1).to(working_dtype)
                att = self.attn_dropout(att)
                
                y = torch.matmul(att, v)
            
            # Correction finale des NaN/Inf (compile-safe)
            y = torch.nan_to_num(y, nan=0.0, posinf=1e4, neginf=-1e4)
            
            # Reshape et projection finale
            y = y.transpose(1, 2).contiguous().view(B, T, C)
            
            # Projection finale
            output = self.resid_dropout(self.o_proj(y))
            
            return output
            
        except Exception as e:
            if _CheckpointStop is not None and isinstance(e, _CheckpointStop):
                # Allow gradient-checkpointing internals to propagate without noisy logging.
                raise
            print(f"Attention computation failed: {e}")
            print(traceback.format_exc())
            raise  # Re-raise the exception after printing the traceback 
