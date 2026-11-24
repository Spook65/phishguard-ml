"""Preflight gate and one planned evaluation of frozen models."""

from time import perf_counter

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.exceptions import NotFittedError
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.validation import check_is_fitted

from phishguard.data import EXPECTED_ROWS, FEATURE_NAMES, PhishingDataset, validate_dataset
from phishguard.final_protocol import (
    CONFUSION_LABEL_ORDER, DATASET_ID, FINAL_MODEL_NAMES, LEADING_MODEL,
    POSITIVE_LABEL, SPLIT_SEED,
)
from phishguard.models import validation_metrics
from phishguard.split import SplitIndices, feature_groups

EXPECTED_SPLIT_ROWS = (7676, 1629, 1750)
EXPECTED_GROUPS = (4049, 868, 868)
# Exact validation confusion counts recorded in Checkpoint 2: [[TP,FN],[FP,TN]].
VALIDATION_MATRICES = {
    "majority": ((0, 687), (0, 942)),
    "tree_depth_5": ((598, 89), (11, 931)),
    "tree_unrestricted": ((644, 43), (62, 880)),
    "knn_k_9": ((639, 48), (26, 916)),
}


def verify_partition(features, split: SplitIndices) -> dict:
    """Check complete, unique assignments and exact feature-group isolation."""
    partitions = (split.train, split.validation, split.test)
    lengths = tuple(len(indices) for indices in partitions)
    if lengths != EXPECTED_SPLIT_ROWS:
        raise ValueError(f"Unexpected split row counts: {lengths}")
    for indices in partitions:
        if np.asarray(indices).dtype.kind not in "iu":
            raise ValueError("Split positions must be integers")
    assigned = np.concatenate(partitions)
    if not np.array_equal(np.sort(assigned), np.arange(len(features))):
        raise ValueError("Rows are missing or assigned more than once")
    groups = feature_groups(features)
    group_sets = tuple(set(groups[indices]) for indices in partitions)
    if any(group_sets[i] & group_sets[j] for i, j in ((0, 1), (0, 2), (1, 2))):
        raise ValueError("Identical feature vectors cross partitions")
    counts = tuple(len(group_set) for group_set in group_sets)
    if counts != EXPECTED_GROUPS:
        raise ValueError(f"Unexpected feature-group counts: {counts}")
    return {"row_counts": dict(zip(("train", "validation", "test"), lengths)),
            "feature_group_counts": dict(zip(("train", "validation", "test"), counts)),
            "all_rows_assigned_once": True, "feature_groups_isolated": True}


def verify_models(models) -> None:
    """Reject any estimator or setting outside the Checkpoint 3B lock."""
    if tuple(name for name, _ in models) != FINAL_MODEL_NAMES:
        raise ValueError("Frozen model list or order changed")
    expected_types = (DummyClassifier, DecisionTreeClassifier, DecisionTreeClassifier, KNeighborsClassifier)
    if tuple(type(model) for _, model in models) != expected_types:
        raise ValueError("Frozen estimator types changed")
    baseline = models[0][1].get_params()
    if baseline["strategy"] != "most_frequent":
        raise ValueError("Majority baseline changed")
    for position, depth in ((1, 5), (2, None)):
        params = models[position][1].get_params()
        if any((params["criterion"] != "entropy", params["max_depth"] != depth,
                params["random_state"] != 42)):
            raise ValueError("Entropy tree configuration changed")
    knn = models[3][1].get_params()
    if any((knn["n_neighbors"] != 9, knn["metric"] != "hamming",
            knn["algorithm"] != "brute", knn["weights"] != "uniform")):
        raise ValueError("KNN configuration changed")
    for _, model in models:
        try:
            check_is_fitted(model)
        except NotFittedError:
            continue
        raise ValueError("Final models must be unfitted at preflight")


def preflight(dataset: PhishingDataset, split: SplitIndices, models) -> dict:
    """Run every frozen check before the caller selects test labels."""
    if dataset.metadata.get("uci_id") != DATASET_ID or dataset.metadata.get("name") != "Phishing Websites":
        raise ValueError("Unexpected UCI dataset source")
    if dataset.metadata.get("num_instances") != EXPECTED_ROWS or dataset.metadata.get("num_features") != 30:
        raise ValueError("Unexpected UCI source dimensions")
    # The final test labels are deliberately not read during preflight.
    validate_dataset(dataset.features, dataset.target.to_frame(), expected_rows=EXPECTED_ROWS,
                     check_target_values=False)
    if tuple(dataset.features.columns) != FEATURE_NAMES:
        raise ValueError("Feature order changed")
    if SPLIT_SEED != 42 or POSITIVE_LABEL != -1 or CONFUSION_LABEL_ORDER != (-1, 1):
        raise ValueError("Frozen split or metric convention changed")
    partition_checks = verify_partition(dataset.features, split)
    verify_models(models)
    return {"source_id": DATASET_ID, "source_name": dataset.metadata["name"],
            "rows": EXPECTED_ROWS, "features": len(FEATURE_NAMES), "seed": SPLIT_SEED,
            "positive_label": POSITIVE_LABEL, "confusion_label_order": list(CONFUSION_LABEL_ORDER),
            "model_names": list(FINAL_MODEL_NAMES), "fit_partition": "train",
            **partition_checks}


def validation_reference() -> dict:
    """Derive exact validation metrics from recorded Checkpoint 2 matrices."""
    reference = {}
    for name, ((tp, fn), (fp, tn)) in VALIDATION_MATRICES.items():
        total = tp + fn + fp + tn
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
        reference[name] = {"accuracy": (tp + tn) / total,
                           "phishing_precision": precision, "phishing_recall": recall,
                           "phishing_f1": f1}
    return reference


def evaluate_models(x_train, y_train, x_test, y_test, models) -> list[dict]:
    """Fit from supplied training rows and score supplied test rows once each.

    The production caller must complete preflight before invoking this function.
    """
    verify_models(models)
    if tuple(x_train.columns) != FEATURE_NAMES or tuple(x_test.columns) != FEATURE_NAMES:
        raise ValueError("Expected the original 30 feature columns")
    if len(x_train) != len(y_train) or len(x_test) != len(y_test):
        raise ValueError("Feature/label row counts differ")
    results = []
    for name, model in models:
        model.fit(x_train, y_train)
        start = perf_counter()
        predictions = model.predict(x_test)
        elapsed = perf_counter() - start
        results.append({"model": name, **validation_metrics(y_test, predictions, elapsed)})
    return results


def comparison(results: list[dict]) -> list[dict]:
    """Signed test-minus-validation differences; no model reselection."""
    reference = validation_reference()
    return [{"model": row["model"], "validation": reference[row["model"]],
             "test_minus_validation": {metric: row[metric] - reference[row["model"]][metric]
                                       for metric in reference[row["model"]]}}
            for row in results]
