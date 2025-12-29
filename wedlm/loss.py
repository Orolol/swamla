"""
WeDLM Loss Functions

Implements the WeDLM training objective with:
- Per-block weighted cross-entropy loss (1/gamma weighting)
- Optional auxiliary AR loss to preserve autoregressive capability

Reference: WeDLM paper Section 4.1, Equation 7 and 10
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional


class WeDLMLoss(nn.Module):
    """
    WeDLM loss with per-block weighting.

    Loss formula:
        L = sum_k (1/gamma_k) * sum_{j in M_k} -log P(x_j | context)

    Where:
        - k indexes prediction blocks
        - gamma_k is the masking ratio for block k
        - M_k is the set of masked positions in block k
        - 1/gamma_k compensates for varying numbers of masked tokens

    Optionally includes auxiliary AR loss:
        L_total = L_wedlm + alpha * L_ar

    Where L_ar is standard next-token prediction on non-masked positions.
    """

    def __init__(
        self,
        ar_loss_weight: float = 0.5,
        label_smoothing: float = 0.0,
        ignore_index: int = -100,
    ):
        """
        Args:
            ar_loss_weight: Weight for auxiliary AR loss (0 to disable)
            label_smoothing: Label smoothing for cross-entropy
            ignore_index: Index to ignore in loss computation (padding)
        """
        super().__init__()
        self.ar_loss_weight = ar_loss_weight
        self.label_smoothing = label_smoothing
        self.ignore_index = ignore_index

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        target_mask: torch.Tensor,
        mask_ratios: torch.Tensor,
        block_size: int,
        return_components: bool = True,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute WeDLM loss with optional AR auxiliary loss.

        Args:
            logits: [B, L, V] model output logits for prediction stream
            targets: [B, L] target token IDs (original unmasked tokens)
            target_mask: [B, L] boolean mask (True = masked position to predict)
            mask_ratios: [B, K] masking ratios per block
            block_size: size of each prediction block
            return_components: whether to return loss components dict

        Returns:
            total_loss: scalar loss tensor
            loss_dict: dictionary with loss components (if return_components=True)
        """
        B, L, V = logits.shape
        K = mask_ratios.shape[1]
        device = logits.device

        # Compute per-position cross-entropy loss
        # Shape: [B, L]
        # Use reshape instead of view for non-contiguous tensors
        ce_loss = F.cross_entropy(
            logits.reshape(-1, V),
            targets.reshape(-1),
            reduction='none',
            label_smoothing=self.label_smoothing,
            ignore_index=self.ignore_index,
        ).reshape(B, L)

        # ===== WeDLM Loss with per-block weighting =====
        wedlm_loss = torch.tensor(0.0, device=device)
        num_valid_blocks = 0

        for k in range(K):
            start = k * block_size
            end = min((k + 1) * block_size, L)

            # Get mask and loss for this block
            block_mask = target_mask[:, start:end]  # [B, block_len]
            block_loss = ce_loss[:, start:end]       # [B, block_len]

            # Sum loss over masked positions in this block
            # [B]
            masked_loss_sum = (block_loss * block_mask.float()).sum(dim=1)
            num_masked = block_mask.sum(dim=1).float()  # [B]

            # Skip blocks with no masked tokens
            valid_batch = num_masked > 0
            if not valid_batch.any():
                continue

            num_valid_blocks += 1

            # Per-block weighting: 1/gamma_k
            # The 1/gamma weighting compensates for blocks with fewer masked tokens
            # However, with small gamma (e.g., 0.1), the weight becomes very large (10x)
            # causing gradient variance issues and loss stagnation.
            #
            # Solution: Use sqrt(1/gamma) instead of 1/gamma for smoother weighting
            # This still gives higher weight to low-mask-ratio blocks but less aggressively
            # - gamma=0.1 -> sqrt(1/0.1) = 3.16x (vs 10x with 1/gamma)
            # - gamma=0.5 -> sqrt(1/0.5) = 1.41x (vs 2x with 1/gamma)
            # - gamma=1.0 -> sqrt(1/1.0) = 1.0x (vs 1x with 1/gamma)
            gamma_k = mask_ratios[:, k].clamp(min=0.1)  # [B], increased min from 0.01
            weight = torch.sqrt(1.0 / gamma_k)  # sqrt for smoother weighting

            # Compute weighted loss for this block
            # Average over masked positions, weight by sqrt(1/gamma), average over batch
            block_wedlm_loss = (
                weight * masked_loss_sum / num_masked.clamp(min=1)
            )
            # Only count valid batches (those with masked tokens in this block)
            block_wedlm_loss = block_wedlm_loss[valid_batch].mean()

            wedlm_loss = wedlm_loss + block_wedlm_loss

        # Normalize by number of blocks
        if num_valid_blocks > 0:
            wedlm_loss = wedlm_loss / num_valid_blocks

        # ===== Auxiliary AR Loss =====
        ar_loss = torch.tensor(0.0, device=device)

        if self.ar_loss_weight > 0:
            # AR loss on non-masked positions (next-token prediction)
            ar_mask = ~target_mask  # [B, L] - positions that were NOT masked

            if ar_mask.any():
                ar_loss_sum = (ce_loss * ar_mask.float()).sum()
                num_ar_positions = ar_mask.sum().float()
                ar_loss = ar_loss_sum / num_ar_positions.clamp(min=1)

        # ===== Total Loss =====
        total_loss = wedlm_loss + self.ar_loss_weight * ar_loss

        if return_components:
            loss_dict = {
                'wedlm_loss': wedlm_loss.detach(),
                'ar_loss': ar_loss.detach(),
                'total_loss': total_loss.detach(),
                'num_masked': target_mask.sum().float().detach(),
                'num_observed': (~target_mask).sum().float().detach(),
            }
            return total_loss, loss_dict

        return total_loss, {}


class SimpleMaskedLMLoss(nn.Module):
    """
    Simple masked language modeling loss (without block weighting).

    For comparison and simpler training scenarios.
    """

    def __init__(
        self,
        ar_loss_weight: float = 0.0,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.ar_loss_weight = ar_loss_weight
        self.label_smoothing = label_smoothing

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        target_mask: torch.Tensor,
        mask_ratio: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute simple MLM loss.

        Args:
            logits: [B, L, V]
            targets: [B, L]
            target_mask: [B, L] boolean
            mask_ratio: optional global mask ratio for weighting

        Returns:
            loss, loss_dict
        """
        B, L, V = logits.shape
        device = logits.device

        ce_loss = F.cross_entropy(
            logits.reshape(-1, V),
            targets.reshape(-1),
            reduction='none',
            label_smoothing=self.label_smoothing,
        ).reshape(B, L)

        # MLM loss on masked positions
        mlm_loss = (ce_loss * target_mask.float()).sum() / target_mask.sum().float().clamp(min=1)

        # Weight by 1/gamma if provided
        if mask_ratio is not None:
            mlm_loss = mlm_loss / mask_ratio.clamp(min=0.01)

        # Optional AR loss
        ar_loss = torch.tensor(0.0, device=device)
        if self.ar_loss_weight > 0:
            ar_mask = ~target_mask
            if ar_mask.any():
                ar_loss = (ce_loss * ar_mask.float()).sum() / ar_mask.sum().float().clamp(min=1)

        total_loss = mlm_loss + self.ar_loss_weight * ar_loss

        return total_loss, {
            'mlm_loss': mlm_loss.detach(),
            'ar_loss': ar_loss.detach(),
        }


def compute_accuracy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """
    Compute prediction accuracy on masked positions.

    Args:
        logits: [B, L, V]
        targets: [B, L]
        mask: [B, L] boolean mask of positions to evaluate

    Returns:
        accuracy: scalar tensor
    """
    predictions = logits.argmax(dim=-1)  # [B, L]
    correct = (predictions == targets) & mask
    accuracy = correct.sum().float() / mask.sum().float().clamp(min=1)
    return accuracy
