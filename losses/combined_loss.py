
"""Combined Dice + BCE loss."""

import torch
import torch.nn as nn

from losses.dice_loss import DiceLoss
from losses.bce_loss import BCELoss


class DiceBCELoss(nn.Module):
    """Weighted sum of Dice and BCE losses.

    BCE provides stable per-pixel gradient signal early in training (when
    Dice gradients can be unstable for near-empty predicted masks), while
    Dice pushes directly toward better mask overlap. Combining both is a
    well-established practice for imbalanced binary segmentation tasks.
    """

    def __init__(self, dice_weight: float = 0.5, bce_weight: float = 0.5) -> None:
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.dice = DiceLoss()
        self.bce = BCELoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        dice_loss = self.dice(logits, targets)
        bce_loss = self.bce(logits, targets)
        return self.dice_weight * dice_loss + self.bce_weight * bce_loss
