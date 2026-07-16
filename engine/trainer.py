
"""Training loop encapsulated as a class for state management (AMP scaler,
optimizer, best-model tracking) across epochs."""

import logging
from typing import Dict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


class Trainer:
    """Handles a single epoch of training, including mixed-precision
    forward/backward passes and gradient clipping.

    Encapsulated as a class (rather than a free function) because it holds
    state across epochs — specifically the GradScaler, which must persist
    between epochs for correct AMP loss-scaling behavior.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        criterion: nn.Module,
        device: torch.device,
        use_amp: bool = True,
        grad_clip_norm: float = 1.0,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.use_amp = use_amp and device.type == "cuda"
        self.grad_clip_norm = grad_clip_norm
        self.scaler = GradScaler(enabled=self.use_amp)

    def train_one_epoch(self, dataloader: DataLoader, epoch: int) -> float:
        self.model.train()
        running_loss = 0.0

        for step, batch in enumerate(dataloader):
            images = batch["image"].to(self.device, non_blocking=True)
            masks = batch["mask"].to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=self.use_amp):
                logits = self.model(images)
                loss = self.criterion(logits, masks)

            self.scaler.scale(loss).backward()

            # Gradient clipping must happen after unscaling but before step,
            # otherwise it clips the scaled (not true) gradient magnitudes.
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)

            self.scaler.step(self.optimizer)
            self.scaler.update()

            running_loss += loss.item()

            if step % 20 == 0:
                logger.info(
                    f"Epoch {epoch} | Step {step}/{len(dataloader)} | Loss: {loss.item():.4f}"
                )

        avg_loss = running_loss / len(dataloader)
        logger.info(f"Epoch {epoch} completed. Avg train loss: {avg_loss:.4f}")
        return avg_loss
