"""Reproducible feature-group partitions for the UCI benchmark."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from phishguard.data import PhishingDataset, validate_dataset

SEED = 42


@dataclass(frozen=True)
class SplitIndices:
    """Zero-based row positions in the unchanged input dataset."""

    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray


def feature_groups(features: pd.DataFrame) -> np.ndarray:
    """Assign equal IDs to rows with identical values across all 30 features."""
    return pd.MultiIndex.from_frame(features).factorize(sort=False)[0]


def split_dataset(dataset: PhishingDataset, *, seed: int = SEED) -> SplitIndices:
    """Target 70/15/15 by assigning whole feature groups.

    GroupShuffleSplit samples groups, so row fractions are approximate. Targets
    are validated but never used to construct groups or choose partitions.
    The held-out test is chosen first and is not used for later tuning.
    """
    features, target = dataset.features, dataset.target
    validate_dataset(features, target.to_frame(), expected_rows=len(features))
    groups = feature_groups(features)
    positions = np.arange(len(features))
    first = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=seed)
    remaining_local, test_local = next(first.split(features, groups=groups))
    remaining, test = positions[remaining_local], positions[test_local]
    second = GroupShuffleSplit(n_splits=1, test_size=0.15 / 0.85, random_state=seed + 1)
    train_local, validation_local = next(second.split(remaining, groups=groups[remaining]))
    return SplitIndices(
        train=np.sort(remaining[train_local]),
        validation=np.sort(remaining[validation_local]),
        test=np.sort(test),
    )


def split_summary(dataset: PhishingDataset, split: SplitIndices) -> dict:
    """Report row counts and raw label proportions without remapping classes."""
    target = dataset.target
    total = len(target)
    result = {}
    for name in ("train", "validation", "test"):
        indices = getattr(split, name)
        counts = target.iloc[indices].value_counts().sort_index()
        result[name] = {
            "rows": len(indices),
            "row_fraction": len(indices) / total,
            "target_counts": {str(label): int(counts.get(label, 0)) for label in (-1, 1)},
            "target_proportions": {str(label): float(counts.get(label, 0) / len(indices)) for label in (-1, 1)},
            "feature_groups": int(len(np.unique(feature_groups(dataset.features.iloc[indices])))),
        }
    return result
