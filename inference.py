
"""CLI entry point for running inference on a single custom image."""

import argparse
import logging
from pathlib import Path

from config import CONFIG
from data_pipeline.transforms import get_val_transforms
from models.segformer import build_model
from engine.inference_engine import InferenceEngine
from utils.checkpoint import load_checkpoint
from utils.visualization import overlay_mask
from utils.logger import get_logger

import matplotlib.pyplot as plt

logger = get_logger("inference", CONFIG.log_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run clothing segmentation inference.")
    parser.add_argument("--image_path", type=str, required=True, help="Path to input image.")
    parser.add_argument(
        "--checkpoint_path", type=str,
        default=str(CONFIG.checkpoint_dir / "best_model.pth"),
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--output_path", type=str, default=None, help="Optional path to save overlay.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    model = build_model(
        pretrained_model_name=CONFIG.pretrained_model_name,
        num_labels=CONFIG.num_labels,
        device=CONFIG.device,
    )
    load_checkpoint(Path(args.checkpoint_path), model, device=CONFIG.device)

    engine = InferenceEngine(
        model, get_val_transforms(CONFIG.image_size), CONFIG.device, threshold=args.threshold
    )

    image, mask = engine.predict(Path(args.image_path))

    import torch
    image_tensor = torch.tensor(image / 255.0).permute(2, 0, 1)
    mask_tensor = torch.tensor(mask)
    overlay = overlay_mask(image_tensor, mask_tensor)

    plt.figure(figsize=(6, 6))
    plt.imshow(overlay)
    plt.axis("off")
    if args.output_path:
        plt.savefig(args.output_path, dpi=100)
        logger.info(f"Overlay saved to {args.output_path}")
    plt.show()


if __name__ == "__main__":
    main()
