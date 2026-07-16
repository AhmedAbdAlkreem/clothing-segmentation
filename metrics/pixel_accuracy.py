
"""Pixel-wise accuracy metric — supplementary, not primary, given class imbalance."""

import torch


@torch.no_grad()
def pixel_accuracy(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> float:
    """Computes fraction of correctly classified pixels.

    Included mainly for completeness in reporting — this metric is
    misleadingly high on imbalanced masks (a model predicting all-background
    can still score >80% pixel accuracy), so Dice/IoU remain the primary
    metrics for model selection.
    """
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()
    correct = (preds == targets).float().sum()
    total = torch.numel(targets)
    return (correct / total).item()
