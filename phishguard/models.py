"""Checkpoint 2 training and validation; the held-out test is never read here."""

from time import perf_counter

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from phishguard.data import PhishingDataset
from phishguard.split import SEED, SplitIndices

PHISHING_LABEL = -1  # Documented working convention; see Checkpoint 1B.
LABEL_ORDER = (-1, 1)
TREE_DEPTHS = (3, 5, None)
KNN_NEIGHBORS = (3, 5, 9)


def training_label_check(features: pd.DataFrame, target: pd.Series) -> dict:
    """Check a documented SSL feature rule against training labels only.

    This is supporting evidence about the working mapping, not proof of it.
    """
    result = {}
    for value in (-1, 1):
        matching = target.loc[features["sslfinal_state"] == value]
        result[str(value)] = {
            "rows": len(matching),
            "raw_minus_one": int((matching == -1).sum()),
            "raw_plus_one": int((matching == 1).sum()),
            "minus_one_rate": float((matching == -1).mean()),
        }
    return result


def validation_metrics(y_true, y_pred, prediction_seconds: float) -> dict:
    """Rows: actual [-1, 1]; columns: predicted [-1, 1]."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "phishing_precision": float(precision_score(y_true, y_pred, pos_label=PHISHING_LABEL, zero_division=0)),
        "phishing_recall": float(recall_score(y_true, y_pred, pos_label=PHISHING_LABEL, zero_division=0)),
        "phishing_f1": float(f1_score(y_true, y_pred, pos_label=PHISHING_LABEL, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=LABEL_ORDER).tolist(),
        "prediction_seconds": float(prediction_seconds),
    }


def run_validation(dataset: PhishingDataset, split: SplitIndices) -> dict:
    """Fit on training rows, score on validation rows, select by phishing F1.

    Test positions are deliberately unused. Tie-break by accuracy, then the
    listed configuration order; prediction time is descriptive, not a tuner.
    """
    x_train = dataset.features.iloc[split.train]
    y_train = dataset.target.iloc[split.train]
    x_validation = dataset.features.iloc[split.validation]
    y_validation = dataset.target.iloc[split.validation]
    if x_train.shape[1] != 30 or x_validation.shape[1] != 30:
        raise ValueError("Expected 30 feature columns")
    if len(x_train) != len(y_train) or len(x_validation) != len(y_validation):
        raise ValueError("Feature/target row mismatch")

    models = [("majority", DummyClassifier(strategy="most_frequent"))]
    models += [(f"tree_depth_{depth}", DecisionTreeClassifier(criterion="entropy", max_depth=depth, random_state=SEED))
               for depth in TREE_DEPTHS]
    models += [(f"knn_k_{neighbors}", KNeighborsClassifier(n_neighbors=neighbors, metric="hamming", algorithm="brute"))
               for neighbors in KNN_NEIGHBORS]
    results = []
    for name, model in models:
        model.fit(x_train, y_train)
        start = perf_counter()
        predictions = model.predict(x_validation)
        seconds = perf_counter() - start
        results.append({"model": name, **validation_metrics(y_validation, predictions, seconds)})
    candidates = [row for row in results if row["model"] != "majority"]
    best = max(candidates, key=lambda row: (row["phishing_f1"], row["accuracy"]))
    return {
        "label_check": training_label_check(x_train, y_train),
        "train_rows": len(x_train),
        "validation_rows": len(x_validation),
        "training_majority_label": int(y_train.value_counts().idxmax()),
        "results": results,
        "best_model": best["model"],
    }
