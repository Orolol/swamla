"""
WeDLM Configuration
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class WeDLMConfig:
    """Configuration for WeDLM training and inference."""

    # Training parameters
    use_wedlm_training: bool = False
    block_size: int = 32                    # Size of prediction blocks for dual-stream masking
    min_mask_ratio: float = 0.3             # Minimum masking ratio per block (0.3 for smoother training)
    max_mask_ratio: float = 1.0             # Maximum masking ratio per block
    ar_loss_weight: float = 0.5             # Weight for auxiliary AR loss
    mask_token_id: int = -1                 # Token ID for [MASK], will be set from tokenizer

    # Inference parameters (Streaming Parallel Decoding)
    window_size: int = 32                   # W: size of sliding window
    entropy_threshold: float = 0.5          # tau: confidence threshold for mask filling
    distance_penalty: float = 0.1           # lambda: penalty for distance from leftmost mask
    min_commit_size: int = 1                # Minimum tokens to commit per step
    refinement_iters: int = 3               # Number of forward passes per window before committing
                                            # Allows predictions to condition on each other

    # Architecture adaptation
    adapt_deltanet: bool = True             # Adapt GatedDeltaNet blocks for WeDLM
    deltanet_pos_emb: bool = True           # Add position embeddings to DeltaNet

    def __post_init__(self):
        assert 0.0 <= self.min_mask_ratio <= 1.0, "min_mask_ratio must be in [0, 1]"
        assert 0.0 <= self.max_mask_ratio <= 1.0, "max_mask_ratio must be in [0, 1]"
        assert self.min_mask_ratio <= self.max_mask_ratio, "min_mask_ratio must be <= max_mask_ratio"
        assert self.block_size > 0, "block_size must be positive"
        assert self.window_size > 0, "window_size must be positive"
        assert self.entropy_threshold > 0, "entropy_threshold must be positive"
        assert self.distance_penalty >= 0, "distance_penalty must be non-negative"
