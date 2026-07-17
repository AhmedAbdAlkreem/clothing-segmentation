
"""Entry point for training the clothing segmentation model.

Kept intentionally thin — orchestration only. All actual logic lives in
datasets/, models/, losses/, engine/, so this file reads like a summary
of the training pipeline rather than containing implementation details.
"""

import logging

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from config import CONFIG
from data_pipeline.preprocessing import build_category_inclusion_map, load_and_group_annotations
from data_pipeline.split import split_dataframe
from data_pipeline.transforms import get_train_transforms, get_val_transforms
from data_pipeline.dataset import ClothingSegmentationDataset
from models.segformer import build_model
from losses.combined_loss import DiceBCELoss
from engine.trainer import Trainer
from engine.validator import Validator
from utils.checkpoint import BestModelSaver
from utils.logger import get_logger
from utils.seed import set_seed

logger = get_logger("train", CONFIG.log_dir)


def main() -> None:
    set_seed(CONFIG.seed)
    logger.info(f"Using device: {CONFIG.device}")

    # --- Data ---
    inclusion_map = build_category_inclusion_map(CONFIG.data_root / "label_descriptions.json")
    df = load_and_group_annotations(
        CONFIG.data_root / "train.csv",
        inclusion_map,
        subset_size=CONFIG.subset_size,
        seed=CONFIG.seed,
    )
    train_df, val_df = split_dataframe(df, val_split=CONFIG.val_split, seed=CONFIG.seed)

    train_dataset = ClothingSegmentationDataset(
        train_df, CONFIG.data_root / "train", transforms=get_train_transforms(CONFIG.image_size)
    )
    val_dataset = ClothingSegmentationDataset(
        val_df, CONFIG.data_root / "train", transforms=get_val_transforms(CONFIG.image_size)
    )

    train_loader = DataLoader(
        train_dataset, batch_size=CONFIG.batch_size, shuffle=True,
        num_workers=CONFIG.num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=CONFIG.batch_size, shuffle=False,
        num_workers=CONFIG.num_workers, pin_memory=True,
    )

    # --- Model, loss, optimizer ---
    model = build_model(
        pretrained_model_name=CONFIG.pretrained_model_name,
        num_labels=CONFIG.num_labels,
        freeze_encoder=CONFIG.freeze_encoder,
        device=CONFIG.device,
    )
    criterion = DiceBCELoss(dice_weight=CONFIG.dice_weight, bce_weight=CONFIG.bce_weight)
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG.lr, weight_decay=CONFIG.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG.epochs)

    trainer = Trainer(
        model, optimizer, criterion, CONFIG.device,
        use_amp=CONFIG.use_amp, grad_clip_norm=CONFIG.grad_clip_norm,
    )
    validator = Validator(model, criterion, CONFIG.device, use_amp=CONFIG.use_amp)
    best_model_saver = BestModelSaver(
        CONFIG.checkpoint_dir / "best_model.pth", mode="max"
    )

    # --- Training loop ---
    for epoch in range(1, CONFIG.epochs + 1):
        trainer.train_one_epoch(train_loader, epoch)
        metrics = validator.validate(val_loader, epoch)
        scheduler.step()

        improved = best_model_saver.step(
            model, optimizer, epoch, metric_value=metrics["dice_score"]
        )
        if improved:
            logger.info(f"New best Dice score: {metrics['dice_score']:.4f}")

    logger.info(f"Training complete. Best Dice score: {best_model_saver.best_value:.4f}")


if __name__ == "__main__":
    main()
