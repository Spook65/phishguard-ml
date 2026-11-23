"""Offline numerical checks for empirical information gain and model diagnostics."""

import socket

import numpy as np
import pandas as pd
import pytest

from phishguard.analysis import analyze_training_validation, entropy, information_gain, rank_information_gain
from phishguard.data import FEATURE_NAMES, FEATURE_VALUES, PhishingDataset
from phishguard.split import SplitIndices


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    monkeypatch.setattr(socket, "socket", lambda *a, **k: (_ for _ in ()).throw(AssertionError("network access")))


def test_entropy_known_cases():
    assert entropy([-1, -1, 1, 1]) == pytest.approx(1.0)
    assert entropy([1, 1, 1]) == pytest.approx(0.0)
    assert entropy([-1, -1, -1, 1]) == pytest.approx(0.8112781244591328)
    with pytest.raises(ValueError):
        entropy([])


def test_information_gain_perfect_and_partial():
    labels = pd.Series([-1, -1, 1, 1], index=[10, 11, 12, 13])
    perfect = pd.Series([-1, -1, 1, 1], index=labels.index)
    assert information_gain(perfect, labels)["information_gain_bits"] == pytest.approx(1.0)
    partial = pd.Series([-1, 0, 0, 1], index=labels.index)
    result = information_gain(partial, labels)
    assert result["entropy_before_bits"] == pytest.approx(1.0)
    assert result["conditional_entropy_bits"] == pytest.approx(0.5)
    assert result["information_gain_bits"] == pytest.approx(0.5)
    assert [part["rows"] for part in result["categories"]] == [1, 2, 1]
    assert information_gain(pd.Series([1, 1, 1, 1], index=labels.index), labels)["information_gain_bits"] == pytest.approx(0)
    with pytest.raises(ValueError, match="matching row indices"):
        information_gain(perfect.reset_index(drop=True), labels)


def test_feature_name_alignment_and_determinism():
    features = pd.DataFrame({name: [min(values)] * 4 for name, values in FEATURE_VALUES.items()})
    features["url_length"] = [-1, -1, 1, 1]
    labels = pd.Series([-1, -1, 1, 1], name="result")
    first = rank_information_gain(features, labels)
    assert first == rank_information_gain(features, labels)
    assert [row["feature"] for row in first] == ["url_length"] + sorted(set(FEATURE_NAMES) - {"url_length"})
    assert first[0]["information_gain_bits"] == pytest.approx(1)
    with pytest.raises(ValueError, match="feature names or order"):
        rank_information_gain(features[features.columns[::-1]], labels)


def test_diagnostics_never_select_test_rows():
    rows, labels = [], []
    for i in range(12):
        row = {name: min(values) for name, values in FEATURE_VALUES.items()}
        row["url_length"] = (-1, 0, 1)[i % 3]
        row["sslfinal_state"] = (-1, 0, 1)[i // 3 % 3]
        rows.append(row)
        labels.append(-1 if i % 2 else 1)
    dataset = PhishingDataset(pd.DataFrame(rows, columns=FEATURE_NAMES), pd.Series(labels, name="result"), {})
    split = SplitIndices(np.arange(9), np.arange(9, 12), np.array([1000]))
    report, tree = analyze_training_validation(dataset, split)
    repeated, _ = analyze_training_validation(dataset, split)
    assert report == repeated
    assert report["train_rows"] == 9
    assert report["validation_rows"] == 3
    assert len(report["information_gain"]) == len(FEATURE_NAMES)
    assert set(report["tree_importances"]) == {"depth_3", "depth_5", "unrestricted"}
    assert all({row["feature"] for row in ranking} == set(FEATURE_NAMES)
               for ranking in report["tree_importances"].values())
    assert len(report["diagnostics"]) == 6
    assert tree.n_features_in_ == 30
