"""Stratified train/validation/test splitting.

Partitions de-duplicated records into three disjoint subsets whose union is
the input, preserving the per-class proportions in each subset.

Requirements: 3.4, 3.5
"""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split


def stratified_split(
    df: pd.DataFrame,
    ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
    seed: int = 42,
    label_col: str = "class_label",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split ``df`` into (train, val, test) preserving class proportions.

    The split is done in two stratified stages: first hold out (val + test),
    then divide that holdout into val and test. Every input row lands in
    exactly one subset (disjoint and covering).

    Parameters
    ----------
    df:
        De-duplicated records with a ``label_col`` column.
    ratios:
        (train, val, test) fractions; must sum to 1.
    seed:
        Random seed for reproducibility.

    Requirements: 3.4, 3.5
    """
    train_r, val_r, test_r = ratios
    if abs((train_r + val_r + test_r) - 1.0) > 1e-9:
        raise ValueError(f"ratios must sum to 1, got {ratios}")

    df = df.reset_index(drop=True)

    # Stage 1: train vs. holdout (val + test).
    holdout_r = val_r + test_r
    train_df, holdout_df = train_test_split(
        df,
        test_size=holdout_r,
        random_state=seed,
        stratify=df[label_col],
    )

    # Stage 2: split holdout into val and test.
    # test proportion *within* the holdout.
    test_within = test_r / holdout_r if holdout_r > 0 else 0.0
    val_df, test_df = train_test_split(
        holdout_df,
        test_size=test_within,
        random_state=seed,
        stratify=holdout_df[label_col],
    )

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )
