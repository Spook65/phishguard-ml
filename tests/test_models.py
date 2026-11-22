"""Offline model and metric semantics tests on synthetic observations."""

import socket

import numpy as np
import pandas as pd
import pytest

from phishguard.data import FEATURE_NAMES, FEATURE_VALUES, PhishingDataset
from phishguard.models import LABEL_ORDER, run_validation, training_label_check, validation_metrics
from phishguard.split import SplitIndices, split_dataset


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    monkeypatch.setattr(socket, "socket", lambda *a, **k: (_ for _ in ()).throw(AssertionError("network access")))


@pytest.fixture
def example():
    rows, labels = [], []
    for group in range(60):
        row = {name: min(values) for name, values in FEATURE_VALUES.items()}
        for i, name in enumerate(("url_length", "having_sub_domain", "sslfinal_state", "web_traffic")):
            row[name] = sorted(FEATURE_VALUES[name])[(group // 3 ** i) % 3]
        for _ in range(2):
            rows.append(row.copy())
            labels.append(-1 if group % 2 else 1)
    dataset = PhishingDataset(pd.DataFrame(rows, columns=FEATURE_NAMES), pd.Series(labels, name="result"), {})
    return dataset, split_dataset(dataset)


def test_phishing_positive_and_confusion_order():
    # Actual -1: two hits and one miss; actual +1: one false alarm and two correct.
    metrics = validation_metrics([-1, -1, -1, 1, 1, 1], [-1, -1, 1, -1, 1, 1], 0.25)
    assert LABEL_ORDER == (-1, 1)
    assert metrics["confusion_matrix"] == [[2, 1], [1, 2]]  # [[TP, FN], [FP, TN]]
    assert metrics["phishing_precision"] == pytest.approx(2 / 3)
    assert metrics["phishing_recall"] == pytest.approx(2 / 3)
    assert metrics["phishing_f1"] == pytest.approx(2 / 3)
    assert metrics["accuracy"] == pytest.approx(2 / 3)
    assert metrics["prediction_seconds"] == 0.25
    all_legitimate = validation_metrics([-1, 1], [1, 1], 0)
    assert all_legitimate["phishing_precision"] == 0
    assert all_legitimate["phishing_recall"] == 0


def test_training_dimensions_baseline_and_configuration(example):
    dataset, split = example
    report = run_validation(dataset, split)
    assert report["train_rows"] == len(split.train)
    assert report["validation_rows"] == len(split.validation)
    assert len(dataset.features.columns) == 30
    assert [r["model"] for r in report["results"]] == [
        "majority", "tree_depth_3", "tree_depth_5", "tree_depth_None",
        "knn_k_3", "knn_k_5", "knn_k_9",
    ]
    majority = report["training_majority_label"]
    baseline = report["results"][0]
    assert majority == int(dataset.target.iloc[split.train].value_counts().idxmax())
    expected_predictions = np.full(len(split.validation), majority)
    assert baseline["confusion_matrix"] == validation_metrics(dataset.target.iloc[split.validation], expected_predictions, 0)["confusion_matrix"]
    assert all(len(r["confusion_matrix"]) == 2 for r in report["results"])


def test_reproducible_metrics_and_no_test_access(example):
    dataset, split = example
    # An invalid test position proves the runner never selects test rows.
    guarded = SplitIndices(split.train, split.validation, np.array([len(dataset.target) + 100]))
    first = run_validation(dataset, guarded)
    second = run_validation(dataset, guarded)
    for a, b in zip(first["results"], second["results"]):
        a = {k: v for k, v in a.items() if k != "prediction_seconds"}
        b = {k: v for k, v in b.items() if k != "prediction_seconds"}
        assert a == b
    assert first["best_model"] == second["best_model"]


def test_training_check_uses_supplied_rows_only(example):
    dataset, split = example
    x = dataset.features.iloc[split.train]
    y = dataset.target.iloc[split.train]
    expected = training_label_check(x, y)
    assert run_validation(dataset, split)["label_check"] == expected
    assert sum(v["rows"] for v in expected.values()) <= len(split.train)


def test_wrong_feature_count_is_rejected(example):
    dataset, split = example
    wrong = PhishingDataset(dataset.features.drop(columns="iframe"), dataset.target, {})
    with pytest.raises(ValueError, match="30 feature columns"):
        run_validation(wrong, split)
