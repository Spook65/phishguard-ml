"""Training-only feature analysis and train/validation diagnostics."""

from math import log2

import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from phishguard.data import FEATURE_NAMES, PhishingDataset
from phishguard.models import PHISHING_LABEL, TREE_DEPTHS, validation_metrics
from phishguard.split import SEED, SplitIndices


def entropy(labels) -> float:
    """Empirical Shannon entropy in bits; zero for a pure sample."""
    labels = pd.Series(labels)
    if labels.empty:
        raise ValueError("Entropy requires at least one label")
    if labels.isna().any():
        raise ValueError("Labels must not be missing")
    probabilities = labels.value_counts(normalize=True)
    return float(-sum(p * log2(p) for p in probabilities))


def information_gain(feature: pd.Series, labels: pd.Series) -> dict:
    """H(Y) - sum_v P(X=v) H(Y | X=v), using observed categories."""
    if len(feature) != len(labels) or not feature.index.equals(labels.index):
        raise ValueError("Feature and labels must have matching row indices")
    if feature.empty or feature.isna().any():
        raise ValueError("Feature must be nonempty without missing values")
    grouped = labels.groupby(feature, sort=True)
    components = []
    for value, subset in grouped:
        components.append({
            "value": int(value),
            "rows": len(subset),
            "entropy_bits": entropy(subset),
            "weighted_bits": len(subset) / len(labels) * entropy(subset),
            "label_counts": {str(label): int(count) for label, count in subset.value_counts().sort_index().items()},
        })
    before = entropy(labels)
    after = sum(item["weighted_bits"] for item in components)
    return {"entropy_before_bits": before, "conditional_entropy_bits": after,
            "information_gain_bits": before - after, "categories": components}


def rank_information_gain(features: pd.DataFrame, labels: pd.Series) -> list[dict]:
    """Return named results in descending gain, breaking ties by feature name."""
    if tuple(features.columns) != FEATURE_NAMES:
        raise ValueError("Unexpected feature names or order")
    ranked = [{"feature": name, **information_gain(features[name], labels)} for name in FEATURE_NAMES]
    return sorted(ranked, key=lambda row: (-row["information_gain_bits"], row["feature"]))


def analyze_training_validation(dataset: PhishingDataset, split: SplitIndices) -> tuple[dict, DecisionTreeClassifier]:
    """Read only train and validation positions; return depth-5 tree for plotting."""
    x_train, y_train = dataset.features.iloc[split.train], dataset.target.iloc[split.train]
    x_val, y_val = dataset.features.iloc[split.validation], dataset.target.iloc[split.validation]
    if tuple(x_train.columns) != FEATURE_NAMES or tuple(x_val.columns) != FEATURE_NAMES:
        raise ValueError("Unexpected feature names or order")
    gain = rank_information_gain(x_train, y_train)
    trees = {}
    tree_importances = {}
    for depth in TREE_DEPTHS:
        tree = DecisionTreeClassifier(criterion="entropy", max_depth=depth, random_state=SEED)
        tree.fit(x_train, y_train)
        key = "unrestricted" if depth is None else f"depth_{depth}"
        trees[key] = tree
        tree_importances[key] = [
            {"feature": name, "importance": float(importance)}
            for name, importance in sorted(zip(FEATURE_NAMES, tree.feature_importances_),
                                           key=lambda pair: (-pair[1], pair[0]))
        ]
    knn = KNeighborsClassifier(n_neighbors=9, metric="hamming", algorithm="brute")
    knn.fit(x_train, y_train)
    diagnostics = []
    for name, model in (("tree_depth_5", trees["depth_5"]),
                        ("tree_unrestricted", trees["unrestricted"]), ("knn_k_9", knn)):
        for partition, x, y in (("train", x_train, y_train), ("validation", x_val, y_val)):
            metrics = validation_metrics(y, model.predict(x), 0.0)
            diagnostics.append({"model": name, "partition": partition,
                                "accuracy": metrics["accuracy"], "phishing_f1": metrics["phishing_f1"]})
    return ({"train_rows": len(x_train), "validation_rows": len(x_val),
             "working_phishing_label": PHISHING_LABEL, "label_entropy_bits": entropy(y_train),
             "information_gain": gain, "tree_importances": tree_importances,
             "diagnostics": diagnostics,
             "tree_structure": {key: {"depth": int(tree.get_depth()), "leaves": int(tree.get_n_leaves())}
                                for key, tree in trees.items()}}, trees["depth_5"])
