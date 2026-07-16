
"""Dice score metric (evaluation, not loss)."""

import torch


@torch.no_grad()
def dice_score(
    logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5, smooth: float = 1.0
) -> float:
    """Computes hard Dice score using thresholded binary predictions —
    distinct from DiceLoss, which uses soft (probability) values for
    differentiability. Metrics should reflect actual thresholded output
    quality, which is what matters at inference time.
    """
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()

    preds = preds.view(preds.size(0), -1)
    targets = targets.view(targets.size(0), -1)

    intersection = (preds * targets).sum(dim=1)
    union = preds.sum(dim=1) + targets.sum(dim=1)

    dice = (2.0 * intersection + smooth) / (union + smooth)
    return dice.mean().item()
