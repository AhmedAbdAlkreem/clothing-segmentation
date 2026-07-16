
"""Visualization helpers for validation/inference sanity checks."""

from pathlib import Path
from typing import Optional

import numpy as np
import torch
import matplotlib.pyplot as plt


def denormalize(
    img_tensor: torch.Tensor,
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
) -> np.ndarray:
    """Reverses ImageNet normalization for display purposes."""
    img = img_tensor.cpu().numpy().transpose(1, 2, 0)
    img = (img * std) + mean
    return np.clip(img, 0, 1)


def visualize_prediction(
    image: torch.Tensor,
    gt_mask: Optional[torch.Tensor],
    pred_mask: torch.Tensor,
    save_path: Optional[Path] = None,
) -> None:
    """Plots original | ground truth | prediction side by side.

    gt_mask is Optional because inference on unlabeled/custom images has
    no ground truth to compare against.
    """
    num_cols = 3 if gt_mask is not None else 2
    fig, axes = plt.subplots(1, num_cols, figsize=(5 * num_cols, 5))

    img_np = denormalize(image)
    axes[0].imshow(img_np)
    axes[0].set_title("Original")
    axes[0].axis("off")

    col = 1
    if gt_mask is not None:
        axes[col].imshow(gt_mask.squeeze().cpu().numpy(), cmap="gray")
        axes[col].set_title("Ground Truth")
        axes[col].axis("off")
        col += 1

    axes[col].imshow(pred_mask.squeeze().cpu().numpy(), cmap="gray")
    axes[col].set_title("Prediction")
    axes[col].axis("off")

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=100)
    plt.show()
    plt.close(fig)


def overlay_mask(image: torch.Tensor, mask: torch.Tensor, alpha: float = 0.45) -> np.ndarray:
    """Overlays a red-tinted mask onto the original image for intuitive display."""
    img_np = (denormalize(image) * 255).astype(np.uint8)
    mask_np = mask.squeeze().cpu().numpy().astype(np.uint8)

    color_mask = np.zeros_like(img_np)
    color_mask[..., 0] = 255  # red channel
    color_mask = color_mask * mask_np[..., None]

    overlay = (img_np * (1 - alpha) + color_mask * alpha).astype(np.uint8)
    return overlay
