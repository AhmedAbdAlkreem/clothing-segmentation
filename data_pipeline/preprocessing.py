

"""Dataset-level preprocessing: category mapping and image-instance grouping."""

import json
import logging
from pathlib import Path
from typing import Dict, Set

import pandas as pd

logger = logging.getLogger(__name__)

# Supercategories representing garment PARTS, not standalone garments.
# Instances under these are excluded from the binary foreground mask to
# avoid double-counting pixels already covered by their parent garment.
PART_SUPERCATEGORIES: Set[str] = {
    "sleeve", "collar", "lapel", "epaulette", "pocket", "neckline",
    "buckle", "zipper", "applique", "bead", "bow", "flower", "fringe",
    "ribbon", "rivet", "ruffle", "sequin", "tassel",
}


def build_category_inclusion_map(label_descriptions_path: Path) -> Dict[int, bool]:
    """Builds a {class_id: include_in_binary_mask} mapping from the
    dataset's label_descriptions.json, based on each category's
    supercategory field.

    Building this programmatically (rather than hardcoding a list of
    class IDs) means the logic is self-documenting and survives if the
    category ID ordering changes.
    """
    with open(label_descriptions_path, "r") as f:
        label_data = json.load(f)

    inclusion_map = {}
    for cat in label_data["categories"]:
        supercategory = cat.get("supercategory", "").lower()
        inclusion_map[cat["id"]] = supercategory not in PART_SUPERCATEGORIES

    num_excluded = sum(1 for v in inclusion_map.values() if not v)
    logger.info(
        f"Category inclusion map built: {len(inclusion_map)} total, "
        f"{num_excluded} excluded as garment-parts."
    )
    return inclusion_map


def load_and_group_annotations(
    csv_path: Path,
    category_inclusion_map: Dict[int, bool],
    subset_size: int = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Loads train.csv, filters out garment-part instances, and optionally
    subsamples to a fixed number of unique images.

    Filtering happens once here (not per-__getitem__ call) so Dataset
    objects only ever iterate over rows that are already relevant —
    keeps __getitem__ fast and simple.
    """
    df = pd.read_csv(csv_path)
    df["include"] = df["ClassId"].map(category_inclusion_map)
    df = df[df["include"]].drop(columns=["include"]).reset_index(drop=True)

    if subset_size is not None:
        unique_ids = df["ImageId"].unique()
        if subset_size < len(unique_ids):
            sampled_ids = pd.Series(unique_ids).sample(
                n=subset_size, random_state=seed
            ).tolist()
            df = df[df["ImageId"].isin(sampled_ids)].reset_index(drop=True)

    logger.info(
        f"Loaded annotations: {df['ImageId'].nunique()} unique images, "
        f"{len(df)} instance rows after filtering."
    )
    return df
