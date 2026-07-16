
"""Dice loss for binary segmentation."""

import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    """Soft Dice loss operating directly on logits.

    Dice loss is included (rather than relying on BCE alone) because BCE
    treats every pixel independently and is dominated by the background
    class (clothing typically covers a minority of pixels). Dice directly
    optimizes for mask overlap, which is more aligned with the eventual
    evaluation metric (Dice/IoU) and more robust to class imbalance.
    """

    def __init__(self, smooth: float = 1.0) -> None:
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        probs = probs.view(probs.size(0), -1)
        targets = targets.view(targets.size(0), -1)

        intersection = (probs * targets).sum(dim=1)
        union = probs.sum(dim=1) + targets.sum(dim=1)

        dice_score = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice_score.mean()
