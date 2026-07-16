"""Central configuration for the clothing segmentation project."""
# ... (paste the full config.py content here)

"""Reproducibility utilities."""

import os
import random
import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Fixes seeds across all RNG sources used in the pipeline.

    Full determinism with cuDNN has a performance cost, so we only enable
    `torch.backends.cudnn.deterministic` — not `benchmark=False` — as a
    reasonable reproducibility/speed tradeoff for a Kaggle GPU budget.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
