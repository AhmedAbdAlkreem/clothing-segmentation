
"""Inference on single images — used by both inference.py (CLI) and
scripts/visualize_samples.py (notebook-driven sanity checks)."""

import logging
from pathlib import Path
from typing import Tuple

import albumentations as A
import cv2
import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class InferenceEngine:
    """Wraps a trained model for single-image inference, handling resizing
    back to original resolution so output masks always match the input
    image's native dimensions — required for real-world usage (e.g., a
    virtual fitting room overlay) where users upload arbitrary-sized photos.
    """

    def __init__(
        self,
        model: nn.Module,
        transforms: A.Compose,
        device: torch.device,
        threshold: float = 0.5,
    ) -> None:
        self.model = model.eval()
        self.transforms = transforms
        self.device = device
        self.threshold = threshold

    @torch.no_grad()
    def predict(self, image_path: Path) -> Tuple[np.ndarray, np.ndarray]:
        """Returns (original_rgb_image, binary_mask_at_original_resolution)."""
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_size = image.shape[:2]  # (H, W)

        augmented = self.transforms(image=image)
        input_tensor = augmented["image"].unsqueeze(0).to(self.device)

        logits = self.model(input_tensor)
        probs = torch.sigmoid(logits)
        pred_mask = (probs > self.threshold).float().squeeze().cpu().numpy()

        mask_resized = cv2.resize(
            pred_mask.astype(np.uint8),
            (original_size[1], original_size[0]),
            interpolation=cv2.INTER_NEAREST,
        )
        return image, mask_resized
