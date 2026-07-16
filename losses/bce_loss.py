
"""Binary cross-entropy loss wrapper."""

import torch
import torch.nn as nn


class BCELoss(nn.Module):
    """Thin wrapper around BCEWithLogitsLoss.

    Using the "WithLogits" variant (rather than sigmoid + BCELoss) avoids
    numerical instability from applying sigmoid manually before computing
    log-loss — BCEWithLogitsLoss uses the log-sum-exp trick internally.
    """

    def __init__(self) -> None:
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.bce(logits, targets)
