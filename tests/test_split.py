"""Offline checks for group isolation and complete, reproducible assignment."""

import socket

import numpy as np
import pandas as pd
import pytest

from phishguard.data import FEATURE_NAMES, FEATURE_VALUES, PhishingDataset
from phishguard.split import feature_groups, split_dataset, split_summary


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    monkeypatch.setattr(socket, "socket", lambda *a, **k: (_ for _ in ()).throw(AssertionError("network access")))


@pytest.fixture
def dataset():
    # 120 distinct categorical patterns, each repeated 1–4 times. One group
    # has contradictory target labels. This is a synthetic test fixture.
    rows, labels = [], []
    for group in range(120):
        row = {name: min(values) for name, values in FEATURE_VALUES.items()}
        for i, name in enumerate(FEATURE_NAMES[:6]):
            allowed = sorted(FEATURE_VALUES[name])
            row[name] = allowed[(group // (2 ** i)) % len(allowed)]
        # Six mostly binary columns cannot encode 120 groups; use ternary columns.
        for i, name in enumerate(("url_length", "having_sub_domain", "sslfinal_state")):
            row[name] = sorted(FEATURE_VALUES[name])[(group // (3 ** i)) % 3]
        row["url_length"] = sorted(FEATURE_VALUES["url_length"])[group % 3]
        for repeat in range(1 + group % 4):
            rows.append(row.copy())
            labels.append(1 if (group + (repeat == 1 and group == 7)) % 2 else -1)
    features = pd.DataFrame(rows, columns=FEATURE_NAMES)
    target = pd.Series(labels, name="result")
    assert target.nunique() == 2
    assert features.duplicated().any()
    assert target.groupby(feature_groups(features)).nunique().max() == 2
    return PhishingDataset(features, target, {})


def test_complete_disjoint_grouped_and_reproducible(dataset):
    split = split_dataset(dataset)
    again = split_dataset(dataset)
    partitions = [split.train, split.validation, split.test]
    for name in ("train", "validation", "test"):
        np.testing.assert_array_equal(getattr(split, name), getattr(again, name))
    all_rows = np.concatenate(partitions)
    np.testing.assert_array_equal(np.sort(all_rows), np.arange(len(dataset.target)))
    assert len(np.unique(all_rows)) == len(all_rows)
    groups = feature_groups(dataset.features)
    group_sets = [set(groups[indices]) for indices in partitions]
    assert all(group_sets[i].isdisjoint(group_sets[j]) for i, j in ((0, 1), (0, 2), (1, 2)))
    assert all(len(indices) > 0 for indices in partitions)
    assert sum(split_summary(dataset, split)[name]["rows"] for name in ("train", "validation", "test")) == len(dataset.target)


def test_target_is_not_used_for_grouping_or_assignment(dataset):
    original = split_dataset(dataset)
    relabeled = PhishingDataset(dataset.features.copy(), dataset.target.iloc[::-1].reset_index(drop=True), {})
    changed = split_dataset(relabeled)
    for name in ("train", "validation", "test"):
        np.testing.assert_array_equal(getattr(original, name), getattr(changed, name))


def test_seed_and_input_unchanged(dataset):
    features_before = dataset.features.copy(deep=True)
    target_before = dataset.target.copy(deep=True)
    first = split_dataset(dataset, seed=42)
    other = split_dataset(dataset, seed=43)
    assert not np.array_equal(first.test, other.test)
    pd.testing.assert_frame_equal(features_before, dataset.features)
    pd.testing.assert_series_equal(target_before, dataset.target)


def test_raw_class_distribution_is_reported(dataset):
    split = split_dataset(dataset)
    report = split_summary(dataset, split)
    for name in ("train", "validation", "test"):
        counts = report[name]["target_counts"]
        assert sum(counts.values()) == report[name]["rows"]
        assert set(counts) == {"-1", "1"}
        assert abs(sum(report[name]["target_proportions"].values()) - 1) < 1e-12
    assert sum(report[name]["target_counts"]["-1"] for name in report) == int((dataset.target == -1).sum())
    assert sum(report[name]["target_counts"]["1"] for name in report) == int((dataset.target == 1).sum())
