"""Multi-Head Latent Attention (MLA) mechanisms for transformer models."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from positional_encoding import RoPE, FoPE


# Import Flash Attention 2
try:
    from flash_attn import flash_attn_func
    FLASH_ATTN_AVAILABLE = True
except ImportError:
    FLASH_ATTN_AVAILABLE = False

# Import FlexAttention (PyTorch 2.5+)
try:
    from torch.nn.attention.flex_attention import flex_attention, create_block_mask
    FLEX_ATTENTION_AVAILABLE = True
except ImportError:
    FLEX_ATTENTION_AVAILABLE = False
    flex_attention = None
    create_block_mask = None

# Import SDPA kernel selection (PyTorch 2.5+)
try:
    from torch.nn.attention import sdpa_kernel, SDPBackend
    SDPA_KERNEL_AVAILABLE = True
except ImportError:
    SDPA_KERNEL_AVAILABLE = False
    sdpa_kernel = None
    SDPBackend = None

# Import custom Triton MLA attention kernel (H100 compatible alternative to FA2)
try:
    from triton_kernels import mla_attention_triton
    TRITON_MLA_AVAILABLE = True
except ImportError:
    TRITON_MLA_AVAILABLE = False
    mla_attention_triton = None


class MLA(nn.Module):
    """
    Multi-Head Latent Attention (MLA) Layer.

    MLA uses a low-rank projection for compressing the key-value representations,
    reducing memory usage and computational complexity while maintaining model quality.

    Attributes:
        dim (int): Dimensionality of the input features.
        n_heads (int): Number of attention heads.
        n_local_heads (int): Number of local attention heads for distributed systems.
        q_lora_rank (int): Rank for low-rank query projection.
        kv_lora_rank (int): Rank for low-rank key/value projection.
        qk_nope_head_dim (int): Dimensionality of non-positional query/key projections.
        qk_rope_head_dim (int): Dimensionality of rotary-positional query/key projections.
        qk_head_dim (int): Total dimensionality of query/key projections.
        v_head_dim (int): Dimensionality of value projections.
        softmax_scale (float): Scaling factor for softmax in attention computation.
        attention_backend (str): Backend used for attention computation.
    """
    def __init__(self, config, layer_id: Optional[int] = None):
        super().__init__()
        self.dim = config.n_embd if hasattr(config, 'n_embd') else config.dim
        self.n_heads = config.n_head if hasattr(config, 'n_head') else config.n_heads
        
        # For distributed training
        self.world_size = getattr(config, 'world_size', 1)
        self.n_local_heads = self.n_heads // self.world_size
        
        # Low-rank dimensions
        self.q_lora_rank = getattr(config, 'q_lora_rank', 0)
        self.kv_lora_rank = getattr(config, 'kv_lora_rank', 512)
        
        # Head dimensions
        self.qk_nope_head_dim = getattr(config, 'qk_nope_head_dim', 128)
        self.qk_rope_head_dim = getattr(config, 'qk_rope_head_dim', 64)
        self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.v_head_dim = getattr(config, 'v_head_dim', 128)
        
        # Optional values from config
        self.dropout = getattr(config, 'dropout', 0.0)

        bias = config.bias if hasattr(config, 'bias') else False

        # Linear projections
        if self.q_lora_rank == 0:
            self.wq = nn.Linear(self.dim, self.n_heads * self.qk_head_dim, bias=bias)
        else:
            self.wq_a = nn.Linear(self.dim, self.q_lora_rank, bias=bias)
            self.q_norm = nn.LayerNorm(self.q_lora_rank)
            self.wq_b = nn.Linear(self.q_lora_rank, self.n_heads * self.qk_head_dim, bias=bias)

        # Low-rank projection for keys and values
        self.wkv_a = nn.Linear(self.dim, self.kv_lora_rank + self.qk_rope_head_dim, bias=bias)
        self.kv_norm = nn.LayerNorm(self.kv_lora_rank)
        self.wkv_b = nn.Linear(self.kv_lora_rank, self.n_heads * (self.qk_nope_head_dim + self.v_head_dim), bias=bias)

        # Output projection
        self.wo = nn.Linear(self.n_heads * self.v_head_dim, self.dim, bias=bias)
        
        # Dropout layers
        self.attn_dropout = nn.Dropout(self.dropout)
        self.resid_dropout = nn.Dropout(self.dropout)
        
        # Attention scaling factor
        self.softmax_scale = self.qk_head_dim ** -0.5

        # YaRN attention temperature scaling
        # When YaRN is enabled, multiply softmax_scale by attn_factor
        # YaRN paper: sqrt(1/t) = 0.1 * ln(s) + 1, so softmax_scale *= attn_factor
        yarn_enabled = getattr(config, 'yarn_enabled', False)
        yarn_attn_factor = getattr(config, 'yarn_attn_factor', None)
        if yarn_enabled and yarn_attn_factor is not None and yarn_attn_factor != 1.0:
            self.softmax_scale = self.softmax_scale * yarn_attn_factor

        # For extended sequences (legacy mscale logic, only if YaRN not enabled)
        if not yarn_enabled:
            rope_factor = getattr(config, 'rope_factor', 1.0)
            if rope_factor > 1.0 and hasattr(config, 'original_seq_len') and hasattr(config, 'max_seq_len'):
                if config.max_seq_len > config.original_seq_len:
                    mscale = getattr(config, 'mscale', 1.0)
                    mscale = 0.1 * mscale * math.log(rope_factor) + 1.0
                    self.softmax_scale = self.softmax_scale * mscale * mscale

        
        # Set up caching for inference
        self.max_batch_size = getattr(config, 'max_batch_size', 8)
        self.max_seq_len = getattr(config, 'max_seq_len', 4096)
        self.attn_impl = getattr(config, 'attn_impl', "absorb")

        # Flash Attention 2 support
        self.use_flash_attention = getattr(config, 'use_flash_attention', False) and FLASH_ATTN_AVAILABLE
        print(f"MLA: FLASH_ATTN_AVAILABLE = {FLASH_ATTN_AVAILABLE}")
        print(f"MLA: use_flash_attention = {self.use_flash_attention}")
        if self.use_flash_attention:
            print(f"MLA: Using Flash Attention 2")

        # Custom Triton MLA kernel (H100 compatible alternative to FA2)
        # Use this when FA2 causes CUDA graph issues with torch.compile
        self.use_triton_mla = getattr(config, 'use_triton_mla', False) and TRITON_MLA_AVAILABLE
        if self.use_triton_mla:
            print(f"MLA: Using custom Triton attention kernel")

        # FlexAttention support (for WeDLM with custom masks)
        # NOTE: FlexAttention with score_mod doesn't work well with torch.compile
        # when using dynamic tensor closures. Disabled by default, use SDPA instead.
        self.use_flex_attention = getattr(config, 'use_flex_attention', False) and FLEX_ATTENTION_AVAILABLE
        self._flex_attention_compiled = None  # Will be compiled on first use

        self.use_cudnn_sdpa = getattr(config, 'use_cudnn_sdpa', True) and SDPA_KERNEL_AVAILABLE
        self.force_cudnn_sdpa = getattr(config, 'force_cudnn_sdpa', False)
        self._warned_cudnn_fallback = False
        cudnn_compatible_heads = getattr(config, 'cudnn_compatible_heads', False)
        cudnn_head_dim_limit = 256 if cudnn_compatible_heads else 128
        if self.use_cudnn_sdpa and self.qk_head_dim > cudnn_head_dim_limit:
            if self.force_cudnn_sdpa:
                print(
                    f"MLA: force_cudnn_sdpa=True with head_dim={self.qk_head_dim} "
                    f"(recommended ≤ {cudnn_head_dim_limit})"
                )
            else:
                self.use_cudnn_sdpa = False

        # Blackwell (CC >= 10.0): disable cuDNN SDPA to avoid graph breaks.
        # PyTorch 2.10 auto-selects the best SDPA backend (FlashAttention2 or
        # math fallback) without requiring @torch.compiler.disable, so the
        # native SDPA path is fully compilable with zero graph breaks.
        if self.use_cudnn_sdpa and torch.cuda.is_available():
            cc = torch.cuda.get_device_capability()
            if cc[0] >= 10 and not self.force_cudnn_sdpa:
                self.use_cudnn_sdpa = False
                print(f"MLA: Blackwell detected (CC {cc[0]}.{cc[1]}), using native SDPA (no graph break)")
            elif cc[0] >= 10 and self.force_cudnn_sdpa:
                print(f"MLA: Blackwell detected (CC {cc[0]}.{cc[1]}), force_cudnn_sdpa=True (experimental)")

        if self.use_cudnn_sdpa:
            print(f"MLA: Using cuDNN SDPA backend (head_dim={self.qk_head_dim})")

        if not self.use_cudnn_sdpa and not self.use_flash_attention and not self.use_triton_mla:
            print(f"MLA: Using PyTorch native SDPA backend (head_dim={self.qk_head_dim})")   
        
        # Initialize position embeddings (RoPE or FoPE)
        # FoPE (Fourier Position Embedding) replaces RoPE for better length generalization
        fope_enabled = getattr(config, 'fope_enabled', False)
        if fope_enabled:
            fope_n_harmonics = getattr(config, 'fope_n_harmonics', 4)
            fope_floor_ratio = getattr(config, 'fope_floor_ratio', 0.1)
            fope_coef_init_std = getattr(config, 'fope_coef_init_std', 0.3)
            self.rope = FoPE(
                dim=self.qk_rope_head_dim,
                max_seq_len=self.max_seq_len,
                base=getattr(config, 'rope_theta', 10000.0),
                n_harmonics=fope_n_harmonics,
                floor_ratio=fope_floor_ratio,
                coef_init_std=fope_coef_init_std,
            )
            self.use_fope = True
        else:
            self.rope = RoPE(self.qk_rope_head_dim, self.max_seq_len)
            self.use_fope = False
        
        # Only create caches for inference, not for training
        # This prevents memory leaks during training
        self.inference_mode = False
        
        # Initialize cache attributes to None to prevent attribute errors
        self.k_cache = None
        self.v_cache = None
        self.kv_cache = None
        self.pe_cache = None

        # Value Embeddings (VE) - applied at alternating layers
        # VE adds token-based bias to V: v = v + gate * ve
        self.layer_id = layer_id
        use_value_embeds = getattr(config, 'use_value_embeds', False)
        # Apply VE at alternating MLA layers (every other layer)
        self.has_value_embeds = (
            use_value_embeds
            and layer_id is not None
            and layer_id % 2 == 0
        )
        if self.has_value_embeds:
            vocab_size = getattr(config, 'vocab_size', 50304)
            kv_dim = self.n_heads * self.v_head_dim
            self.value_embeds = nn.Embedding(vocab_size, kv_dim)
            # Gate projection: x[:, :, :gate_dim] -> scalar for gating
            ve_gate_dim = getattr(config, 've_gate_dim', 32)
            self.ve_gate = nn.Linear(ve_gate_dim, 1, bias=False)
            self.ve_gate_dim = ve_gate_dim
            # Zero-init for identity at start (like Engram)
            nn.init.zeros_(self.ve_gate.weight)
        
    def set_inference_mode(self, mode=True):
        """
        Set the module to inference mode (with caching) or training mode (no caching).
        
        Args:
            mode (bool): True for inference mode, False for training mode
        """
        if mode == self.inference_mode:
            return  # No change needed
            
        self.inference_mode = mode
        
        # Create caches for inference mode if they don't exist
        if mode:
            if self.attn_impl == "naive":
                if not hasattr(self, "k_cache") or self.k_cache is None:
                    self.register_buffer("k_cache", torch.zeros(
                        self.max_batch_size, self.max_seq_len, self.n_local_heads, self.qk_head_dim
                    ), persistent=False)
                if not hasattr(self, "v_cache") or self.v_cache is None:
                    self.register_buffer("v_cache", torch.zeros(
                        self.max_batch_size, self.max_seq_len, self.n_local_heads, self.v_head_dim
                    ), persistent=False)
            else:
                if not hasattr(self, "kv_cache") or self.kv_cache is None:
                    self.register_buffer("kv_cache", torch.zeros(
                        self.max_batch_size, self.max_seq_len, self.kv_lora_rank
                    ), persistent=False)
                if not hasattr(self, "pe_cache") or self.pe_cache is None:
                    self.register_buffer("pe_cache", torch.zeros(
                        self.max_batch_size, self.max_seq_len, self.qk_rope_head_dim
                    ), persistent=False)
        else:
            # Remove caches in training mode to free memory
            if hasattr(self, "k_cache"):
                delattr(self, "k_cache")
            if hasattr(self, "v_cache"):
                delattr(self, "v_cache")
            if hasattr(self, "kv_cache"):
                delattr(self, "kv_cache")
            if hasattr(self, "pe_cache"):
                delattr(self, "pe_cache")

    def _apply_value_embeds(
        self,
        v: torch.Tensor,
        x: torch.Tensor,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply Value Embeddings (VE) to the value tensor.

        VE adds token-based bias to V with a learned gate:
            ve = value_embeds(token_ids)  # (B, T, kv_dim)
            gate = 2 * sigmoid(ve_gate(x[:, :, :gate_dim]))  # range (0, 2)
            v = v + gate * ve

        Args:
            v: Value tensor (B, T, H, D_v)
            x: Input hidden states (B, T, D) for gate computation
            input_ids: Token IDs (B, T) for embedding lookup

        Returns:
            v with VE applied: (B, T, H, D_v)
        """
        bsz, seqlen = input_ids.shape

        # Get value embeddings from token IDs
        ve = self.value_embeds(input_ids)  # (B, T, n_heads * v_head_dim)
        ve = ve.view(bsz, seqlen, self.n_heads, self.v_head_dim)  # (B, T, H, D_v)

        # Compute gate from first gate_dim dimensions of x
        # gate output is (B, T, 1), expand to (B, T, 1, 1) for broadcast
        gate = 2 * torch.sigmoid(self.ve_gate(x[:, :, :self.ve_gate_dim]))  # (B, T, 1)
        gate = gate.unsqueeze(-1)  # (B, T, 1, 1)

        # Apply gated VE to V
        return v + gate * ve




    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
    ):
        """
        Forward pass for the Multi-Head Latent Attention (MLA) Layer.

        NOTE: This method is decorated with @torch._dynamo.disable to prevent
        torch.compile from capturing FA2's internal operations into CUDA graphs.

        On H100 with FA2 + torch.compile max-autotune, CUDA graph replay can
        fail with stride assertion errors because FA2's internal transpose
        creates tensors with non-standard strides that differ between recording
        and replay. By disabling dynamo for the entire MLA forward, FA2 runs
        normally while the rest of the model (DeltaNet, MLP, norms) benefits
        from torch.compile optimization.

        Blackwell (B200) doesn't seem to have this issue.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, dim).
            start_pos (int): Starting position in the sequence for caching.
            freqs_cis (Optional[torch.Tensor]): Precomputed complex exponential values for rotary embeddings.
            mask (Optional[torch.Tensor]): Mask tensor to exclude certain positions from attention.
            position_ids (Optional[torch.Tensor]): Explicit position IDs for RoPE [B, T].
                If provided, allows physical position to differ from logical position
                (used for WeDLM topological reordering).

        Returns:
            torch.Tensor: Output tensor with the same shape as the input.
        """
        # IMPORTANT: Always use training mode during training to prevent cache memory leaks
        if self.training and self.inference_mode:
            self.set_inference_mode(False)
        
        bsz, seqlen, _ = x.size()
        end_pos = start_pos + seqlen
        
        # Apply query projections (either direct or low-rank)
        if self.q_lora_rank == 0:
            q = self.wq(x)
        else:
            q = self.wq_b(self.q_norm(self.wq_a(x)))
        
        # Reshape and split queries
        q = q.view(bsz, seqlen, self.n_heads, self.qk_head_dim)
        q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        
        # Apply rotary positional encoding to q_pe
        # RoPE expects [B, H, T, D] but q_pe is [B, T, H, D], so transpose
        if position_ids is not None:
            # WeDLM mode: use explicit position IDs for RoPE
            # This enables topological reordering where physical != logical position
            q_pe = self._apply_rope_with_position_ids(q_pe, position_ids)
        elif freqs_cis is not None:
            q_pe = q_pe.transpose(1, 2).contiguous()  # [B, H, T, D]
            q_pe = self.rope(q_pe, start_pos)
            q_pe = q_pe.transpose(1, 2).contiguous()  # [B, T, H, D]
        else:
            # For backward compatibility with models that don't use freqs_cis
            q_pe = q_pe.transpose(1, 2).contiguous()  # [B, H, T, D]
            q_pe = self.rope(q_pe)
            q_pe = q_pe.transpose(1, 2).contiguous()  # [B, T, H, D]
        
        # Process keys and values through low-rank projections
        kv = self.wkv_a(x)
        kv, k_pe = torch.split(kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        
        # Apply rotary positional encoding to k_pe
        if position_ids is not None:
            # WeDLM mode: use explicit position IDs for RoPE
            k_pe = self._apply_rope_with_position_ids_1d(k_pe, position_ids)
        elif freqs_cis is not None:
            # We need to reshape k_pe to match rope's expected input format
            k_pe = k_pe.view(bsz, seqlen, 1, -1)  # [B, T, 1, D]
            k_pe = k_pe.transpose(1, 2)  # [B, 1, T, D]
            k_pe = self.rope(k_pe.contiguous(), start_pos).squeeze(1)  # [B, T, D]
        else:
            # We need to reshape k_pe to match rope's expected input format
            k_pe = k_pe.view(bsz, seqlen, 1, -1)  # [B, T, 1, D]
            k_pe = k_pe.transpose(1, 2)  # [B, 1, T, D]
            k_pe = self.rope(k_pe.contiguous()).squeeze(1)  # [B, T, D]
        
        # Select attention implementation approach
        attn_impl = getattr(self, 'attn_impl', "absorb")
        
        # Check if we're in training or inference mode
        is_inference = self.inference_mode
        
        if attn_impl == "naive":
            # Standard approach: compute full attention matrices
            q = torch.cat([q_nope, q_pe], dim=-1)
            kv = self.wkv_b(self.kv_norm(kv))
            kv = kv.view(bsz, seqlen, self.n_heads, self.qk_nope_head_dim + self.v_head_dim)
            k_nope, v = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
            k = torch.cat([k_nope, k_pe.expand(-1, -1, self.n_heads, -1)], dim=-1)

            # Apply Value Embeddings if enabled for this layer
            if self.has_value_embeds and input_ids is not None:
                v = self._apply_value_embeds(v, x, input_ids)

            if is_inference and hasattr(self, 'k_cache') and hasattr(self, 'v_cache'):
                # Only update caches in inference mode
                self.k_cache[:bsz, start_pos:end_pos] = k
                self.v_cache[:bsz, start_pos:end_pos] = v
                # Use the cached values for attention
                k_to_use = self.k_cache[:bsz, :end_pos]
                v_to_use = self.v_cache[:bsz, :end_pos]
            else:
                # In training mode, just use the current batch values (no caching)
                # This dramatically reduces memory usage
                k_to_use = k
                v_to_use = v
            
            if self.use_triton_mla and mask is None and mla_attention_triton is not None:
                # Use custom Triton MLA kernel (H100 compatible, avoids FA2 CUDA graph issues)
                # Expects B, T, H, D and handles V with different head dim
                x = mla_attention_triton(
                    q.contiguous(), k_to_use.contiguous(), v_to_use.contiguous(),
                    scale=self.softmax_scale, causal=seqlen > 1
                )
            elif self.use_flash_attention and mask is None:
                # Use Flash Attention (expects B, T, H, D)
                # q, k_to_use, v_to_use are already (B, T, H, D)
                # IMPORTANT: Make tensors contiguous for torch.compile compatibility
                x = self._flash_attention(q.contiguous(), k_to_use.contiguous(), v_to_use.contiguous(), causal=seqlen > 1)
            elif self.use_flex_attention and mask is not None and FLEX_ATTENTION_AVAILABLE:
                # Use FlexAttention for custom masks (WeDLM dual-stream)
                # FlexAttention expects [B, H, S, D]
                q_flex = q.transpose(1, 2).contiguous()
                k_flex = k_to_use.transpose(1, 2).contiguous()
                v_flex = v_to_use.transpose(1, 2).contiguous()

                attn_output = self._flex_attention_with_mask(q_flex, k_flex, v_flex, mask)
                x = attn_output.transpose(1, 2).contiguous()
            else:
                # Reshape for SDPA: [B, H, S, D]
                # IMPORTANT: Make tensors contiguous for torch.compile compatibility
                q_sdpa = q.transpose(1, 2).contiguous()  # [B, H, S, D]
                k_sdpa = k_to_use.transpose(1, 2).contiguous()  # [B, H, T, D]
                v_sdpa = v_to_use.transpose(1, 2).contiguous()  # [B, H, T, D]

                # Use SDPA for attention computation
                # Expand mask if provided from [S, T] to [B, H, S, T]
                attn_mask = None
                if mask is not None:
                    attn_mask = mask.unsqueeze(0).unsqueeze(0).expand(bsz, self.n_heads, -1, -1)
                    # Ensure mask has the same dtype as query tensor for SDPA compatibility
                    attn_mask = attn_mask.to(q_sdpa.dtype)

                if self.use_cudnn_sdpa:
                    # cuDNN path: requires @torch.compiler.disable for head_dim workaround
                    attn_output = self._sdpa_attention(
                        q_sdpa, k_sdpa, v_sdpa,
                        attn_mask=attn_mask,
                        is_causal=mask is None and seqlen > 1,
                    )
                else:
                    # Standard SDPA: fully compilable, no graph break
                    attn_output = F.scaled_dot_product_attention(
                        q_sdpa, k_sdpa, v_sdpa,
                        attn_mask=attn_mask,
                        dropout_p=self.dropout if self.training else 0.0,
                        is_causal=mask is None and seqlen > 1,
                        scale=self.softmax_scale,
                    )

                # Transpose back to [B, S, H, D]
                x = attn_output.transpose(1, 2).contiguous()
        else:
            # Optimized approach: use low-rank decomposition
            # Uses self.wkv_b() as a proper Linear call (not .weight access)
            # to preserve FP8 quantization path when using Transformer Engine.

            # Prepare normalized kv and pe tensors
            kv_norm_tensor = self.kv_norm(kv)

            if is_inference and hasattr(self, 'kv_cache') and hasattr(self, 'pe_cache'):
                # Only update caches in inference mode
                self.kv_cache[:bsz, start_pos:end_pos] = kv_norm_tensor
                self.pe_cache[:bsz, start_pos:end_pos] = k_pe

                # Use the cached values for attention
                kv_to_use = self.kv_cache[:bsz, :end_pos]
                pe_to_use = self.pe_cache[:bsz, :end_pos]
            else:
                # In training mode, just use the current batch values (no caching)
                # This dramatically reduces memory usage
                kv_to_use = kv_norm_tensor
                pe_to_use = k_pe

            # Project K_nope and V from low-rank space via Linear call
            # (goes through te.Linear FP8 path when available, and reads kv_to_use once)
            kv_projected = self.wkv_b(kv_to_use)  # [B, T, n_heads * (D_nope + D_v)]
            kv_projected = kv_projected.view(bsz, -1, self.n_heads, self.qk_nope_head_dim + self.v_head_dim)
            k_nope_full = kv_projected[..., :self.qk_nope_head_dim]
            v = kv_projected[..., self.qk_nope_head_dim:]

            # Apply Value Embeddings if enabled for this layer
            if self.has_value_embeds and input_ids is not None:
                v = self._apply_value_embeds(v, x, input_ids)

            # Reshape queries - keep q_nope and q_pe in their original dimensions
            q_full = torch.cat([q_nope, q_pe], dim=-1)  # Combine q components [B, S, H, D]
            # Properly expand pe_to_use from [B, T, D] to [B, T, H, D] where each head gets the same RoPE
            # This ensures consistent rotary positional encoding across all attention heads
            pe_expanded = pe_to_use.unsqueeze(2).expand(-1, -1, self.n_heads, -1)
            k_full = torch.cat([k_nope_full, pe_expanded], dim=-1)

            if self.use_triton_mla and mask is None and mla_attention_triton is not None:
                # Use custom Triton MLA kernel (H100 compatible, avoids FA2 CUDA graph issues)
                x = mla_attention_triton(
                    q_full.contiguous(), k_full.contiguous(), v.contiguous(),
                    scale=self.softmax_scale, causal=seqlen > 1
                )
            elif self.use_flash_attention and mask is None:
                # Use Flash Attention (expects B, T, H, D)
                # q_full, k_full are (B, T, H, qk_head_dim), v is (B, T, H, v_head_dim)
                # IMPORTANT: Make tensors contiguous for torch.compile compatibility
                x = self._flash_attention(q_full.contiguous(), k_full.contiguous(), v.contiguous(), causal=seqlen > 1)
            elif self.use_flex_attention and mask is not None and FLEX_ATTENTION_AVAILABLE:
                # Use FlexAttention for custom masks (WeDLM dual-stream)
                # FlexAttention expects [B, H, S, D]
                q_flex = q_full.transpose(1, 2).contiguous()
                k_flex = k_full.transpose(1, 2).contiguous()
                v_flex = v.transpose(1, 2).contiguous()

                attn_output = self._flex_attention_with_mask(q_flex, k_flex, v_flex, mask)
                x = attn_output.transpose(1, 2).contiguous()
            else:
                # Reshape for SDPA: [B, H, S, D]
                # IMPORTANT: Make tensors contiguous for torch.compile compatibility
                q_sdpa = q_full.transpose(1, 2).contiguous()  # [B, H, S, D]
                k_sdpa = k_full.transpose(1, 2).contiguous()  # [B, H, T, D]
                v_sdpa = v.transpose(1, 2).contiguous()  # [B, H, T, D]

                # Use SDPA for attention computation
                # Expand mask if provided from [S, T] to [B, H, S, T]
                attn_mask = None
                if mask is not None:
                    attn_mask = mask.unsqueeze(0).unsqueeze(0).expand(bsz, self.n_heads, -1, -1)
                    # Ensure mask has the same dtype as query tensor for SDPA compatibility
                    attn_mask = attn_mask.to(q_sdpa.dtype)

                if self.use_cudnn_sdpa:
                    # cuDNN path: requires @torch.compiler.disable for head_dim workaround
                    attn_output = self._sdpa_attention(
                        q_sdpa, k_sdpa, v_sdpa,
                        attn_mask=attn_mask,
                        is_causal=mask is None and seqlen > 1,
                    )
                else:
                    # Standard SDPA: fully compilable, no graph break
                    attn_output = F.scaled_dot_product_attention(
                        q_sdpa, k_sdpa, v_sdpa,
                        attn_mask=attn_mask,
                        dropout_p=self.dropout if self.training else 0.0,
                        is_causal=mask is None and seqlen > 1,
                        scale=self.softmax_scale,
                    )

                # Transpose back to [B, S, H, D]
                x = attn_output.transpose(1, 2).contiguous()

        # Reshape and project to output dimension
        x = x.reshape(bsz, seqlen, -1)
        x = self.wo(x)
        
        x = self.resid_dropout(x)
        
        return x
    
    def _flash_attention(self, q, k, v, causal=True):
        """
        Run Flash Attention 2 with proper dtype and dimension handling.

        Args:
            q: (B, T, H, D_qk) queries
            k: (B, T, H, D_qk) keys
            v: (B, T, H, D_v) values - may have different head dim
            causal: whether to use causal masking

        Returns:
            output: (B, T, H, D_v)
        """
        # Ensure inputs are contiguous
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()

        # Flash Attention requires bf16 or fp16
        orig_dtype = q.dtype
        if q.dtype not in (torch.float16, torch.bfloat16):
            q = q.to(torch.bfloat16)
            k = k.to(torch.bfloat16)
            v = v.to(torch.bfloat16)

        # Flash Attention requires Q, K, V to have the same head dimension
        # If V has different dimension, pad it
        d_qk = q.shape[-1]
        d_v = v.shape[-1]

        if d_v != d_qk:
            # Pad V to match Q/K dimension
            pad_size = d_qk - d_v
            # F.pad can create non-contiguous tensors, ensure contiguous after padding
            v_padded = F.pad(v, (0, pad_size), value=0.0).contiguous()
        else:
            v_padded = v

        # Run Flash Attention 2
        # flash_attn_func expects (B, T, H, D) with contiguous memory layout
        attn_output = flash_attn_func(
            q, k, v_padded,
            dropout_p=self.dropout if self.training else 0.0,
            softmax_scale=self.softmax_scale,
            causal=causal,
        )

        # Remove padding from output if we padded V
        if d_v != d_qk:
            # Slicing creates a view, make contiguous
            attn_output = attn_output[..., :d_v].contiguous()

        # Convert back to original dtype
        if attn_output.dtype != orig_dtype:
            attn_output = attn_output.to(orig_dtype).contiguous()

        return attn_output

    # Disable torch.compile tracing for this method because backend selection
    # and fallback handling depend on runtime kernel availability.
    @torch.compiler.disable
    def _sdpa_attention(self, q, k, v, attn_mask=None, is_causal=False):
        """
        Run SDPA with cuDNN backend when available, handling V dimension padding.

        cuDNN attention requires Q/K/V to have the same head dimension.
        MLA has Q/K head_dim=192 but V head_dim=128, so we pad V.

        Args:
            q: [B, H, S, D_qk] queries
            k: [B, H, T, D_qk] keys
            v: [B, H, T, D_v] values (may have different head dim)
            attn_mask: optional attention mask [B, H, S, T]
            is_causal: whether to use causal masking
        Returns:
            output: [B, H, S, D_v]
        """
        d_qk = q.shape[-1]
        d_v = v.shape[-1]
        used_cudnn = False
        padded_v_for_cudnn = self.use_cudnn_sdpa and d_v != d_qk
        v_for_cudnn = F.pad(v, (0, d_qk - d_v), value=0.0).contiguous() if padded_v_for_cudnn else v

        if self.use_cudnn_sdpa:
            try:
                with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
                    attn_output = F.scaled_dot_product_attention(
                        q, k, v_for_cudnn,
                        attn_mask=attn_mask,
                        dropout_p=self.dropout if self.training else 0.0,
                        is_causal=is_causal,
                        scale=self.softmax_scale
                    )
                used_cudnn = True
            except RuntimeError as e:
                err = str(e)
                cudnn_unavailable = (
                    "No available kernel" in err
                    or "CUDNN_ATTENTION is not available" in err
                    or "head_dim should be no more than 128" in err
                    or "No execution plans support the graph" in err
                )
                if not cudnn_unavailable:
                    raise

                if not self._warned_cudnn_fallback:
                    print(
                        "MLA: cuDNN SDPA kernel unavailable at runtime, "
                        "falling back to native SDPA for stability."
                    )
                    self._warned_cudnn_fallback = True
                # Disable cuDNN path for subsequent steps to avoid repeated failures.
                self.use_cudnn_sdpa = False

                attn_output = F.scaled_dot_product_attention(
                    q, k, v,
                    attn_mask=attn_mask,
                    dropout_p=self.dropout if self.training else 0.0,
                    is_causal=is_causal,
                    scale=self.softmax_scale
                )
        else:
            attn_output = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=is_causal,
                scale=self.softmax_scale
            )

        # Remove padding only when cuDNN path actually ran with padded V.
        if used_cudnn and padded_v_for_cudnn:
            attn_output = attn_output[..., :d_v].contiguous()

        return attn_output

    def clear_cache(self):
        """Explicitly clear all caches - useful for memory management during training"""
        self.k_cache = None
        self.v_cache = None
        self.kv_cache = None
        self.pe_cache = None
        # Force garbage collection of any remaining references
        if hasattr(self, '_cache_tensors'):
            del self._cache_tensors

    def _apply_rope_with_position_ids(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply RoPE using explicit position IDs for WeDLM topological reordering.

        Args:
            x: [B, T, H, D] tensor (queries or keys with head dimension)
            position_ids: [B, T] logical position indices

        Returns:
            x_rotated: [B, T, H, D] with RoPE applied using logical positions
        """
        B, T, H, D = x.shape
        device = x.device

        # Get cos/sin from cached values using position indices
        # self.rope has cos_cached and sin_cached of shape [1, 1, max_seq_len, D]
        cos_cached = self.rope.cos_cached.squeeze(0).squeeze(0)  # [max_seq_len, D]
        sin_cached = self.rope.sin_cached.squeeze(0).squeeze(0)  # [max_seq_len, D]

        # Gather cos/sin for the specified positions
        # position_ids: [B, T] -> indices into [max_seq_len, D]
        flat_pos = position_ids.view(-1)  # [B*T]

        # Only use D//2 as that's what we need for rotation
        half_D = D // 2
        cos_gathered = cos_cached[flat_pos, :half_D].view(B, T, half_D)  # [B, T, D//2]
        sin_gathered = sin_cached[flat_pos, :half_D].view(B, T, half_D)  # [B, T, D//2]

        # Expand for heads: [B, T, 1, D//2] -> broadcast to [B, T, H, D//2]
        cos_gathered = cos_gathered.unsqueeze(2)  # [B, T, 1, D//2]
        sin_gathered = sin_gathered.unsqueeze(2)  # [B, T, 1, D//2]

        # Split x into two halves for rotation
        x1 = x[..., :half_D]  # [B, T, H, D//2]
        x2 = x[..., half_D:]  # [B, T, H, D//2]

        # Apply rotation
        x_rotated = torch.cat([
            x1 * cos_gathered - x2 * sin_gathered,
            x2 * cos_gathered + x1 * sin_gathered,
        ], dim=-1)

        return x_rotated.contiguous()

    def _apply_rope_with_position_ids_1d(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply RoPE using explicit position IDs for 1D case (k_pe without head dim).

        Args:
            x: [B, T, D] tensor (k_pe before head expansion)
            position_ids: [B, T] logical position indices

        Returns:
            x_rotated: [B, T, D] with RoPE applied using logical positions
        """
        B, T, D = x.shape

        # Get cos/sin from cached values
        cos_cached = self.rope.cos_cached.squeeze(0).squeeze(0)  # [max_seq_len, D_rope]
        sin_cached = self.rope.sin_cached.squeeze(0).squeeze(0)  # [max_seq_len, D_rope]

        # Gather cos/sin for the specified positions
        flat_pos = position_ids.view(-1)  # [B*T]

        half_D = D // 2
        cos_gathered = cos_cached[flat_pos, :half_D].view(B, T, half_D)  # [B, T, D//2]
        sin_gathered = sin_cached[flat_pos, :half_D].view(B, T, half_D)  # [B, T, D//2]

        # Split x into two halves for rotation
        x1 = x[..., :half_D]  # [B, T, D//2]
        x2 = x[..., half_D:]  # [B, T, D//2]

        # Apply rotation
        x_rotated = torch.cat([
            x1 * cos_gathered - x2 * sin_gathered,
            x2 * cos_gathered + x1 * sin_gathered,
        ], dim=-1)

        return x_rotated.contiguous()

    def _flex_attention_with_mask(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute attention using FlexAttention with a mask.

        FlexAttention compiles the mask into efficient CUDA kernels,
        providing near Flash Attention performance for arbitrary masks.

        Args:
            q: [B, H, S, D] queries
            k: [B, H, T, D] keys
            v: [B, H, T, D_v] values
            mask: [S, T] mask - either:
                  - boolean (True = attend, False = don't attend)
                  - float (0 = attend, -inf = don't attend)

        Returns:
            output: [B, H, S, D_v]
        """
        if not FLEX_ATTENTION_AVAILABLE:
            raise RuntimeError("FlexAttention not available")

        B, H, S, D = q.shape
        _, _, T, D_v = v.shape

        # Ensure all tensors have the same dtype (FlexAttention requirement)
        target_dtype = q.dtype
        if k.dtype != target_dtype:
            k = k.to(target_dtype)
        if v.dtype != target_dtype:
            v = v.to(target_dtype)

        # Handle V dimension mismatch by padding if necessary
        need_v_padding = D_v != D
        if need_v_padding:
            pad_size = D - D_v
            v = F.pad(v, (0, pad_size), value=0.0)

        # Convert mask to attention bias format [1, 1, S, T]
        # Handle both boolean and float masks
        if mask.dtype == torch.bool:
            # Boolean mask: True -> 0.0, False -> -inf
            attn_bias = torch.where(
                mask,
                torch.zeros((), device=q.device, dtype=q.dtype),
                torch.tensor(float('-inf'), device=q.device, dtype=q.dtype),
            ).unsqueeze(0).unsqueeze(0)  # [1, 1, S, T]
        else:
            # Float mask: already in correct format (0 or -inf)
            # Just ensure correct dtype and add batch/head dimensions
            attn_bias = mask.to(q.dtype).unsqueeze(0).unsqueeze(0)  # [1, 1, S, T]

        # Use score_mod to apply the bias
        # Capture attn_bias in closure - flex_attention will compile this
        _bias = attn_bias

        def score_mod(score, b, h, q_idx, kv_idx):
            return score + _bias[0, 0, q_idx, kv_idx]

        # Compile flex_attention for this mask pattern
        attn_output = flex_attention(
            q, k, v,
            score_mod=score_mod,
            scale=self.softmax_scale,
        )

        # Remove padding from output if we padded V
        if need_v_padding:
            attn_output = attn_output[..., :D_v]

        return attn_output.contiguous()
