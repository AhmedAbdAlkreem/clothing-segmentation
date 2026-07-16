
"""Validation loop — computes loss and evaluation metrics without gradient tracking."""

import logging
from typing import Dict

import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader

from metrics.dice import dice_score
from metrics.iou import iou_score
from metrics.pixel_accuracy import pixel_accuracy

logger = logging.getLogger(__name__)


class Validator:
    """Runs evaluation over a dataloader and aggregates metrics.

    Metrics are computed per-batch and averaged rather than accumulated
    over the full concatenated tensor, trading a small amount of precision
    for constant memory usage — necessary since validation sets can be
    large relative to available GPU/CPU memory.
    """

    def __init__(self, model: nn.Module, criterion: nn.Module, device: torch.device, use_amp: bool = True) -> None:
        self.model = model
        self.criterion = criterion
        self.device = device
        self.use_amp = use_amp and device.type == "cuda"

    @torch.no_grad()
    def validate(self, dataloader: DataLoader, epoch: int) -> Dict[str, float]:
        self.model.eval()

        running_loss, running_dice, running_iou, running_acc = 0.0, 0.0, 0.0, 0.0

        for batch in dataloader:
            images = batch["image"].to(self.device, non_blocking=True)
            masks = batch["mask"].to(self.device, non_blocking=True)

            with autocast(enabled=self.use_amp):
                logits = self.model(images)
                loss = self.criterion(logits, masks)

            running_loss += loss.item()
            running_dice += dice_score(logits, masks)
            running_iou += iou_score(logits, masks)
            running_acc += pixel_accuracy(logits, masks)

        n = len(dataloader)
        metrics = {
            "val_loss": running_loss / n,
            "dice_score": running_dice / n,
            "iou_score": running_iou / n,
            "pixel_accuracy": running_acc / n,
        }

        logger.info(
            f"Epoch {epoch} validation | "
            f"Loss: {metrics['val_loss']:.4f} | "
            f"Dice: {metrics['dice_score']:.4f} | "
            f"IoU: {metrics['iou_score']:.4f} | "
            f"PixelAcc: {metrics['pixel_accuracy']:.4f}"
        )
        return metrics
