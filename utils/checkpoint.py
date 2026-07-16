

"""Model checkpoint save/load utilities."""

import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim

logger = logging.getLogger(__name__)


def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    epoch: int,
    metric_value: float,
    checkpoint_path: Path,
) -> None:
    """Saves a full training state, not just model weights — allows
    resuming training later (not just inference), which matters for
    Kaggle's session time limits.
    """
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metric_value": metric_value,
        },
        checkpoint_path,
    )
    logger.info(f"Checkpoint saved at epoch {epoch} (metric={metric_value:.4f}) -> {checkpoint_path}")


def load_checkpoint(
    checkpoint_path: Path,
    model: nn.Module,
    optimizer: Optional[optim.Optimizer] = None,
    device: Optional[torch.device] = None,
) -> dict:
    """Loads a checkpoint into model (and optimizer, if provided).

    Returns the raw checkpoint dict so callers can inspect epoch/metric.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    logger.info(f"Loaded checkpoint from {checkpoint_path} (epoch {checkpoint.get('epoch')})")
    return checkpoint


class BestModelSaver:
    """Tracks the best metric value seen so far and only saves when improved.

    Encapsulating this as a class (rather than a loose `best_so_far` variable
    in train.py) keeps the "did this improve?" logic testable and reusable
    across train.py/evaluate.py.
    """

    def __init__(self, checkpoint_path: Path, mode: str = "max") -> None:
        assert mode in ("max", "min")
        self.checkpoint_path = checkpoint_path
        self.mode = mode
        self.best_value = float("-inf") if mode == "max" else float("inf")

    def step(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        epoch: int,
        metric_value: float,
    ) -> bool:
        """Returns True if this call resulted in a new best checkpoint."""
        is_better = (
            metric_value > self.best_value
            if self.mode == "max"
            else metric_value < self.best_value
        )
        if is_better:
            self.best_value = metric_value
            save_checkpoint(model, optimizer, epoch, metric_value, self.checkpoint_path)
            return True
        return False
