"""Small synthetic fixtures exercise intake logic, never benchmark results."""

import json
import socket
from types import SimpleNamespace

import pandas as pd
import pytest

from phishguard import data
from scripts import inspect_dataset as cli


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def blocked(*args, **kwargs):
        raise AssertionError("Tests must not access the internet")
    monkeypatch.setattr(socket, "socket", blocked)
    monkeypatch.setattr(data, "fetch_ucirepo", blocked)


@pytest.fixture
def frames():
    # A,A,B,B,C: A repeats consistently, B has contradictory labels.
    features = pd.DataFrame({name: [min(values)] * 5 for name, values in data.FEATURE_VALUES.items()})
    features["url_length"] = [-1, -1, 0, 0, 1]
    targets = pd.DataFrame({"result": [-1, -1, -1, 1, 1]})
    return features, targets


def test_valid_schema_preserves_values(frames):
    features, targets = frames
    original = features.copy(deep=True)
    data.validate_dataset(features, targets, expected_rows=5)
    pd.testing.assert_frame_equal(features, original)
    assert data.LABEL_MAPPING == {-1: "phishing", 1: "legitimate"}


@pytest.mark.parametrize("case, message", [
    ("rows", "rows and 30 features"),
    ("columns", "feature names or order"),
    ("order", "feature names or order"),
    ("target_name", "target column"),
    ("target_count", "target column"),
    ("alignment", "indices"),
    ("duplicate_index", "indices"),
    ("missing_feature", "Missing values"),
    ("missing_target", "Missing values"),
    ("feature_category", "Invalid integer categories"),
    ("redirect_category", "Invalid integer categories"),
    ("string_feature", "Invalid integer categories"),
    ("fractional_feature", "Invalid integer categories"),
    ("target_category", "target labels"),
    ("string_target", "target labels"),
    ("single_class", "target labels"),
])
def test_rejects_bad_input(frames, case, message):
    features, targets = frames
    if case == "rows":
        features = features.iloc[:-1]
    elif case == "columns":
        features = features.rename(columns={"url_length": "wrong"})
    elif case == "order":
        features = features[features.columns[::-1]]
    elif case == "target_name":
        targets = targets.rename(columns={"result": "wrong"})
    elif case == "target_count":
        targets["extra"] = 1
    elif case == "alignment":
        targets.index = targets.index[::-1]
    elif case == "duplicate_index":
        targets.index = features.index = [0] * 5
    elif case == "missing_feature":
        features.loc[0, "url_length"] = float("nan")
    elif case == "missing_target":
        targets.loc[0, "result"] = float("nan")
    elif case == "feature_category":
        features.loc[0, "having_ip_address"] = 0
    elif case == "redirect_category":
        features.loc[0, "redirect"] = -1
    elif case == "string_feature":
        features["url_length"] = features["url_length"].astype(str)
    elif case == "fractional_feature":
        features["url_length"] = features["url_length"].astype(float) + 0.5
    elif case == "target_category":
        targets.loc[0, "result"] = 0
    elif case == "string_target":
        targets["result"] = targets["result"].astype(str)
    elif case == "single_class":
        targets["result"] = 1
    with pytest.raises(ValueError, match=message):
        data.validate_dataset(features, targets, expected_rows=5)


def test_duplicate_definitions_and_conflict_positions(frames):
    features, targets = frames
    features.index = targets.index = [10, 20, 30, 40, 50]
    dataset = data.PhishingDataset(features, targets["result"], {"uci_id": 327})
    before = features.copy(deep=True)
    report = data.inspect_dataset(dataset)
    assert report["duplicates"] == {
        "unique_feature_vectors": 3,
        "repeated_feature_groups": 2,
        "rows_in_repeated_groups": 4,
        "extra_rows_beyond_first": 2,
        "largest_group": 2,
        "conflicting_feature_groups": 1,
        "rows_in_conflicting_groups": 2,
        "extra_identical_labeled_rows": 1,
        "conflicts": [{"row_positions": [2, 3], "label_counts": {"-1": 1, "1": 1}}],
    }
    assert report["target"]["counts"] == {"-1": 3, "1": 2}
    assert report["features"]["url_length"]["value_counts"] == {"-1": 2, "0": 2, "1": 1}
    assert report["missing_total"] == 0
    assert len(report["features"]) == 30
    json.dumps(report)
    pd.testing.assert_frame_equal(features, before)


def test_no_duplicates(frames):
    features, targets = frames
    features = features.iloc[[0, 2, 4]].reset_index(drop=True)
    target = targets.iloc[[0, 2, 4]].reset_index(drop=True)["result"]
    report = data.inspect_dataset(data.PhishingDataset(features, target, {}))
    assert report["duplicates"]["extra_rows_beyond_first"] == 0
    assert report["duplicates"]["conflicts"] == []


def test_loader_requests_official_id_and_preserves_raw_labels(monkeypatch, frames):
    features, targets = frames
    features = pd.concat([features] * 2211, ignore_index=True)
    targets = pd.concat([targets] * 2211, ignore_index=True)
    calls = []
    def fake_fetch(*, id):
        calls.append(id)
        return SimpleNamespace(metadata={"uci_id": 327}, data=SimpleNamespace(features=features, targets=targets))
    monkeypatch.setattr(data, "fetch_ucirepo", fake_fetch)
    dataset = data.load_dataset()
    assert calls == [327]
    assert dataset.features.shape == (11055, 30)
    pd.testing.assert_series_equal(dataset.target, targets["result"])
    dataset.features.iloc[0, 0] = 1
    assert features.iloc[0, 0] == -1


def test_loader_rejects_wrong_source(monkeypatch):
    monkeypatch.setattr(data, "fetch_ucirepo", lambda **kwargs: SimpleNamespace(metadata={"uci_id": 1}))
    with pytest.raises(ValueError, match="dataset ID"):
        data.load_dataset()


def test_network_failure_is_not_replaced(monkeypatch):
    def unavailable(**kwargs):
        raise ConnectionError("UCI unavailable")
    monkeypatch.setattr(data, "fetch_ucirepo", unavailable)
    with pytest.raises(ConnectionError, match="UCI unavailable"):
        data.load_dataset()


def test_cli_prints_json(monkeypatch, frames, capsys):
    features, targets = frames
    monkeypatch.setattr(cli, "load_dataset", lambda: data.PhishingDataset(features, targets["result"], {}))
    assert cli.main() == 0
    assert json.loads(capsys.readouterr().out)["rows"] == 5


def test_cli_failure_is_nonzero(monkeypatch, capsys):
    def unavailable():
        raise ConnectionError("UCI unavailable")
    monkeypatch.setattr(cli, "load_dataset", unavailable)
    assert cli.main() == 1
    output = capsys.readouterr()
    assert output.out == ""
    assert "UCI unavailable" in output.err
