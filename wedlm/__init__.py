"""
WeDLM: Causal Diffusion Language Model

Implementation of WeDLM innovations for SWA-MLA:
- Topological Reordering: Reorder [observed | masked] while preserving logical positions
- Dual-Stream Masking: Clean memory stream + masked prediction stream for training
- Streaming Parallel Decoding: Fixed-size sliding window with immediate prefix commitment

Reference: "WeDLM: Reconciling Diffusion Language Models with Standard Causal Attention for Fast Inference"
"""

from .config import WeDLMConfig
from .topological_reorder import topological_reorder, inverse_reorder
from .dual_stream_masking import DualStreamMasker
from .loss import WeDLMLoss
from .deltanet_adapter import adapt_deltanet_for_wedlm, WeDLMGatedDeltaNet
from .streaming_decoder import StreamingParallelDecoder, create_streaming_decoder

__all__ = [
    'WeDLMConfig',
    'topological_reorder',
    'inverse_reorder',
    'DualStreamMasker',
    'WeDLMLoss',
    'adapt_deltanet_for_wedlm',
    'WeDLMGatedDeltaNet',
    'StreamingParallelDecoder',
    'create_streaming_decoder',
]
