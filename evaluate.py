
"""Standalone evaluation script — loads a saved checkpoint and reports
metrics on the validation (or held-out) set without retraining."""

import logging

from torch.utils.data import DataLoader

from config import CONFIG
from data_pipeline.preprocessing import build_category_inclusion_map, load_and_group_annotations
from data_pipeline.split import split_dataframe
from data_pipeline.transforms import get_val_transforms
from data_pipeline.dataset import ClothingSegmentationDataset
from models.segformer import build_model
from losses.combined_loss import DiceBCELoss
from engine.validator import Validator
from utils.checkpoint import load_checkpoint
from utils.logger import get_logger

logger = get_logger("evaluate", CONFIG.log_dir)


def main() -> None:
    inclusion_map = build_category_inclusion_map(CONFIG.data_root / "label_descriptions.json")
    df = load_and_group_annotations(
        CONFIG.data_root / "train.csv", inclusion_map,
        subset_size=CONFIG.subset_size, seed=CONFIG.seed,
    )
    _, val_df = split_dataframe(df, val_split=CONFIG.val_split, seed=CONFIG.seed)

    val_dataset = ClothingSegmentationDataset(
        val_df, CONFIG.data_root / "train", transforms=get_val_transforms(CONFIG.image_size)
    )
    val_loader = DataLoader(
        val_dataset, batch_size=CONFIG.batch_size, shuffle=False, num_workers=CONFIG.num_workers
    )

    model = build_model(
        pretrained_model_name=CONFIG.pretrained_model_name,
        num_labels=CONFIG.num_labels,
        device=CONFIG.device,
    )
    checkpoint_path = CONFIG.checkpoint_dir / "best_model.pth"
    load_checkpoint(checkpoint_path, model, device=CONFIG.device)

    criterion = DiceBCELoss(dice_weight=CONFIG.dice_weight, bce_weight=CONFIG.bce_weight)
    validator = Validator(model, criterion, CONFIG.device, use_amp=CONFIG.use_amp)

    metrics = validator.validate(val_loader, epoch=0)
    logger.info(f"Final evaluation metrics: {metrics}")


if __name__ == "__main__":
    main()
