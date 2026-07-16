

"""Train/validation split utilities."""

from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split


def split_dataframe(
    df: pd.DataFrame, val_split: float = 0.15, seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Splits annotation rows into train/val sets by unique ImageId — never
    by row — to guarantee no instance from the same image leaks across
    the split boundary.
    """
    unique_ids = df["ImageId"].unique()
    train_ids, val_ids = train_test_split(
        unique_ids, test_size=val_split, random_state=seed
    )
    train_df = df[df["ImageId"].isin(train_ids)].reset_index(drop=True)
    val_df = df[df["ImageId"].isin(val_ids)].reset_index(drop=True)
    return train_df, val_df
