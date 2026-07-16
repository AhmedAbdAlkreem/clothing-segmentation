
"""PyTorch Dataset for binary clothing segmentation."""

import logging
from pathlib import Path
from typing import Optional

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from data_pipeline.rle import rle_decode

logger = logging.getLogger(__name__)


class ClothingSegmentationDataset(Dataset):
    """Loads images and builds binary (clothing vs. background) masks by
    OR-combining all instance-level RLE masks belonging to each image.

    Grouping by ImageId happens once in __init__ (via groupby), not inside
    __getitem__, since repeated DataFrame filtering per sample is a common
    performance bug in notebook-derived pipelines.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        image_dir: Path,
        transforms: Optional[A.Compose] = None,
    ) -> None:
        self.image_dir = Path(image_dir)
        self.transforms = transforms

        self.grouped = df.groupby("ImageId")
        self.image_ids = list(self.grouped.groups.keys())

        logger.info(f"Dataset initialized with {len(self.image_ids)} images.")

    def __len__(self) -> int:
        return len(self.image_ids)

    def _build_binary_mask(self, image_id: str) -> np.ndarray:
        rows = self.grouped.get_group(image_id)
        height, width = int(rows.iloc[0]["Height"]), int(rows.iloc[0]["Width"])
        mask = np.zeros((height, width), dtype=np.uint8)

        for _, row in rows.iterrows():
            instance_mask = rle_decode(row["EncodedPixels"], (height, width))
            mask = np.logical_or(mask, instance_mask).astype(np.uint8)

        return mask

    def __getitem__(self, idx: int) -> dict:
        image_id = self.image_ids[idx]
        image_path = self.image_dir / f"{image_id}.jpg"

        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = self._build_binary_mask(image_id)

        if self.transforms is not None:
            augmented = self.transforms(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        # Ensure mask is (1, H, W) float tensor matching model output shape,
        # for direct use with BCEWithLogitsLoss / Dice loss without reshaping
        # scattered across trainer/validator code.
        if not isinstance(mask, torch.Tensor):
            mask = torch.tensor(mask)
        mask = mask.float().unsqueeze(0)

        return {"image": image, "mask": mask, "image_id": image_id}
