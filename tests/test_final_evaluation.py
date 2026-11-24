"""Synthetic, offline tests for the locked evaluation gate and scoring logic."""

import socket

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from phishguard import final_evaluation as evaluation
from phishguard.data import FEATURE_NAMES, FEATURE_VALUES, PhishingDataset
from phishguard.final_protocol import make_final_models
from phishguard.split import SplitIndices
from scripts import final_evaluation as cli


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    monkeypatch.setattr(socket, "socket", lambda *a, **k: (_ for _ in ()).throw(AssertionError("network access")))


@pytest.fixture
def small_dataset(monkeypatch):
    rows = []
    for value in (-1, 0, 1):
        row = {name: min(allowed) for name, allowed in FEATURE_VALUES.items()}
        row["url_length"] = value
        rows.extend((row.copy(), row.copy()))
    data = PhishingDataset(
        pd.DataFrame(rows, columns=FEATURE_NAMES),
        pd.Series([-1, 1, -1, 1, -1, 1], name="result"),
        {"uci_id": 327, "name": "Phishing Websites", "num_instances": 6, "num_features": 30},
    )
    split = SplitIndices(np.array([0, 1]), np.array([2, 3]), np.array([4, 5]))
    monkeypatch.setattr(evaluation, "EXPECTED_ROWS", 6)
    monkeypatch.setattr(evaluation, "EXPECTED_SPLIT_ROWS", (2, 2, 2))
    monkeypatch.setattr(evaluation, "EXPECTED_GROUPS", (1, 1, 1))
    return data, split


def test_preflight_passes_without_inspecting_test_label_values(small_dataset):
    data, split = small_dataset
    data.target.iloc[4:] = 99  # Invalid target values are deferred until after the gate.
    checks = evaluation.preflight(data, split, make_final_models())
    assert checks["all_rows_assigned_once"] is True
    assert checks["feature_groups_isolated"] is True
    assert checks["row_counts"] == {"train": 2, "validation": 2, "test": 2}
    assert checks["positive_label"] == -1
    assert checks["confusion_label_order"] == [-1, 1]


@pytest.mark.parametrize("bad_split, message", [
    (SplitIndices(np.array([0, 1]), np.array([2, 3]), np.array([4, 4])), "missing or assigned"),
    (SplitIndices(np.array([0, 2]), np.array([1, 3]), np.array([4, 5])), "cross partitions"),
    (SplitIndices(np.array([0, 1, 2]), np.array([3]), np.array([4, 5])), "row counts"),
])
def test_partition_failures_stop_before_scoring(small_dataset, bad_split, message):
    data, _ = small_dataset
    with pytest.raises(ValueError, match=message):
        evaluation.preflight(data, bad_split, make_final_models())


def test_source_and_model_drift_fail_preflight(small_dataset):
    data, split = small_dataset
    data.metadata["uci_id"] = 999
    with pytest.raises(ValueError, match="source"):
        evaluation.preflight(data, split, make_final_models())
    data.metadata["uci_id"] = 327
    models = list(make_final_models())
    models[3][1].set_params(n_neighbors=5)
    with pytest.raises(ValueError, match="KNN configuration"):
        evaluation.preflight(data, split, models)


def test_models_fit_only_supplied_training_rows_and_score_four_results(monkeypatch):
    rows = []
    for i in range(16):
        row = {name: min(allowed) for name, allowed in FEATURE_VALUES.items()}
        row["url_length"] = (-1, 0, 1)[i % 3]
        row["sslfinal_state"] = (-1, 0, 1)[i // 3 % 3]
        rows.append(row)
    features = pd.DataFrame(rows, columns=FEATURE_NAMES)
    labels = pd.Series([(-1 if i % 2 else 1) for i in range(16)], name="result")
    calls = []
    for cls in (DummyClassifier, DecisionTreeClassifier, KNeighborsClassifier):
        original = cls.fit
        def spy(self, x, y, *args, _original=original, **kwargs):
            calls.append((len(x), len(y), tuple(x.index)))
            return _original(self, x, y, *args, **kwargs)
        monkeypatch.setattr(cls, "fit", spy)
    results = evaluation.evaluate_models(features.iloc[:12], labels.iloc[:12],
                                         features.iloc[12:], labels.iloc[12:], make_final_models())
    assert len(results) == 4
    assert calls == [(12, 12, tuple(range(12)))] * 4
    assert all(sum(map(sum, row["confusion_matrix"])) == 4 for row in results)
    assert all(row["prediction_seconds"] >= 0 for row in results)
    comparisons = evaluation.comparison(results)
    assert [row["model"] for row in comparisons] == [row["model"] for row in results]
    assert evaluation.validation_reference()["knn_k_9"]["phishing_f1"] == pytest.approx(0.9452662721893491)


def test_cli_does_not_score_when_preflight_fails(monkeypatch, small_dataset, tmp_path):
    data, split = small_dataset
    monkeypatch.setattr(cli, "OUTPUT", tmp_path / "final.json")
    monkeypatch.setattr(cli, "load_dataset", lambda **kwargs: data)
    monkeypatch.setattr(cli, "split_dataset", lambda *args, **kwargs: split)
    def fail(*args):
        raise ValueError("preflight blocked")
    monkeypatch.setattr(cli, "preflight", fail)
    monkeypatch.setattr(cli, "evaluate_models", lambda *args: pytest.fail("test scoring was reached"))
    with pytest.raises(ValueError, match="preflight blocked"):
        cli.main([])
    assert not cli.OUTPUT.exists()


def test_cli_refuses_to_overwrite_results_before_loading(monkeypatch, tmp_path):
    output = tmp_path / "final.json"
    output.write_text("existing results")
    monkeypatch.setattr(cli, "OUTPUT", output)
    monkeypatch.setattr(cli, "load_dataset", lambda **kwargs: pytest.fail("dataset was loaded"))
    with pytest.raises(FileExistsError, match="refusing to rerun"):
        cli.main([])
    assert output.read_text() == "existing results"
