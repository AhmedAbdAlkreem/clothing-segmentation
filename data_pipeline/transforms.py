
"""Albumentations-based augmentation pipelines."""

import albumentations as A
from albumentations.pytorch import ToTensorV2

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def get_train_transforms(image_size: int = 512) -> A.Compose:
    """Training augmentations.

    Augmentation choices are geometry- and lighting-focused, not heavy
    distortions — clothing segmentation masks are sensitive to shape, so
    aggressive elastic/grid distortions risk corrupting mask-image
    alignment for limited benefit at this project's scale.
    """
    return A.Compose(
        [
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.HueSaturationValue(p=0.2),
            A.ShiftScaleRotate(
                shift_limit=0.05, scale_limit=0.1, rotate_limit=10, p=0.4
            ),
            A.GaussNoise(p=0.15),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )


def get_val_transforms(image_size: int = 512) -> A.Compose:
    """Validation/inference transforms — deterministic, no augmentation,
    so evaluation results are reproducible and comparable across runs.
    """
    return A.Compose(
        [
            A.Resize(image_size, image_size),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )
