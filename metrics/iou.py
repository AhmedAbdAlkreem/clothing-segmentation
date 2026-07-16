
"""Intersection-over-Union (Jaccard) metric."""

import torch


@torch.no_grad()
def iou_score(
    logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5, smooth: float = 1.0
) -> float:
    """Computes mean IoU over the batch using thresholded predictions."""
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()

    preds = preds.view(preds.size(0), -1)
    targets = targets.view(targets.size(0), -1)

    intersection = (preds * targets).sum(dim=1)
    union = preds.sum(dim=1) + targets.sum(dim=1) - intersection

    iou = (intersection + smooth) / (union + smooth)
    return iou.mean().item()
