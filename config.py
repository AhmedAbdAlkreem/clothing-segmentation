"""Central configuration for the clothing segmentation project.

Using a single dataclass (rather than scattering constants across files
or using argparse-only config) makes the config importable, type-checked,
and serializable — useful for logging exact run configs to compare
experiments later.
"""

from dataclasses import dataclass, field
from pathlib import Path
import torch


@dataclass
class Config:
    # --- Paths ---
    data_root: Path = Path("/kaggle/input/competitions/imaterialist-fashion-2020-fgvc7")
    project_root: Path = Path("/kaggle/working/clothing_segmentation")
    checkpoint_dir: Path = field(init=False)
    log_dir: Path = field(init=False)

    # --- Dataset ---
    image_size: int = 512
    subset_size: int = 3000          # number of unique images used (None = full dataset)
    val_split: float = 0.15
    seed: int = 42

    # --- Model ---
    pretrained_model_name: str = "nvidia/segformer-b0-finetuned-ade-512-512"
    num_labels: int = 1              # binary segmentation
    freeze_encoder: bool = False

    # --- Training ---
    batch_size: int = 8
    num_workers: int = 2
    epochs: int = 20
    lr: float = 6e-5
    weight_decay: float = 1e-4
    use_amp: bool = True             # mixed precision
    grad_clip_norm: float = 1.0

    # --- Loss weighting ---
    dice_weight: float = 0.5
    bce_weight: float = 0.5

    # --- Device ---
    device: torch.device = field(
        default_factory=lambda: torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

    def __post_init__(self) -> None:
        self.checkpoint_dir = self.project_root / "checkpoints"
        self.log_dir = self.project_root / "logs"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)


CONFIG = Config()
