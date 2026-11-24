"""Offline checks that the pre-test model choices stay frozen."""

import socket

import pytest
from sklearn.dummy import DummyClassifier
from sklearn.exceptions import NotFittedError
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.validation import check_is_fitted

from phishguard.final_protocol import (
    CONFUSION_LABEL_ORDER, DATASET_ID, FINAL_MODEL_NAMES, LEADING_MODEL,
    POSITIVE_LABEL, SPLIT_SEED, make_final_models,
)


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def blocked(*args, **kwargs):
        raise AssertionError("Protocol tests must not use the network")
    monkeypatch.setattr(socket, "socket", blocked)


def test_frozen_protocol_metadata():
    assert DATASET_ID == 327
    assert SPLIT_SEED == 42
    assert POSITIVE_LABEL == -1
    assert CONFUSION_LABEL_ORDER == (-1, 1)
    assert FINAL_MODEL_NAMES == ("majority", "tree_depth_5", "tree_unrestricted", "knn_k_9")
    assert LEADING_MODEL == "knn_k_9"


def test_exact_model_configuration_and_fresh_unfitted_instances():
    models = make_final_models()
    assert tuple(name for name, _ in models) == FINAL_MODEL_NAMES
    assert [type(model) for _, model in models] == [
        DummyClassifier, DecisionTreeClassifier, DecisionTreeClassifier, KNeighborsClassifier,
    ]
    assert models[0][1].get_params()["strategy"] == "most_frequent"
    for depth, (_, tree) in ((5, models[1]), (None, models[2])):
        params = tree.get_params()
        assert params["criterion"] == "entropy"
        assert params["max_depth"] == depth
        assert params["random_state"] == 42
    knn = models[3][1].get_params()
    assert knn["n_neighbors"] == 9
    assert knn["metric"] == "hamming"
    assert knn["algorithm"] == "brute"
    assert knn["weights"] == "uniform"
    assert all(model is not new_model for (_, model), (_, new_model) in zip(models, make_final_models()))
    for _, model in models:
        with pytest.raises(NotFittedError):
            check_is_fitted(model)
