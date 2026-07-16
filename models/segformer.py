"""
SegFormer-based binary clothing segmentation model.

Wraps HuggingFace SegformerForSemanticSegmentation and adapts it for
a single-channel binary segmentation head (clothing vs background).
"""

import logging
from typing import Optional

import torch
import torch.nn as nn
from transformers import SegformerForSemanticSegmentation, SegformerConfig

logger = logging.getLogger(__name__)


class ClothingSegformer(nn.Module):
    """Binary clothing segmentation model built on SegFormer-B0."""

    def __init__(
        self,
        pretrained_model_name: str = "nvidia/segformer-b0-finetuned-ade-512-512",
        num_labels: int = 1,
        freeze_encoder: bool = False,
    ) -> None:
        super().__init__()
        self.num_labels = num_labels

        config = SegformerConfig.from_pretrained(pretrained_model_name)
        config.num_labels = num_labels

        self.model = SegformerForSemanticSegmentation.from_pretrained(
            pretrained_model_name,
            config=config,
            ignore_mismatched_sizes=True,
        )

        if freeze_encoder:
            self._freeze_encoder()
            logger.info("Encoder frozen, only decode head will be trained.")

    def _freeze_encoder(self) -> None:
        for param in self.model.segformer.encoder.parameters():
            param.requires_grad = False

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        outputs = self.model(pixel_values=pixel_values)
        logits = outputs.logits

        upsampled_logits = nn.functional.interpolate(
            logits,
            size=pixel_values.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        return upsampled_logits

    @torch.no_grad()
    def predict_mask(
        self, pixel_values: torch.Tensor, threshold: float = 0.5
    ) -> torch.Tensor:
        self.eval()
        logits = self.forward(pixel_values)
        probs = torch.sigmoid(logits)
        return (probs > threshold).float()


def build_model(
    pretrained_model_name: str = "nvidia/segformer-b0-finetuned-ade-512-512",
    num_labels: int = 1,
    freeze_encoder: bool = False,
    device: Optional[torch.device] = None,
) -> ClothingSegformer:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ClothingSegformer(
        pretrained_model_name=pretrained_model_name,
        num_labels=num_labels,
        freeze_encoder=freeze_encoder,
    )
    model.to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model built on {device}. Trainable parameters: {num_params:,}")

    return model
