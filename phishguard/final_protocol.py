"""Frozen model choices for the later, one-time held-out evaluation.

This module constructs unfitted estimators. It does not load data, split rows,
predict, score, or select a model from test results.
"""

from sklearn.dummy import DummyClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from phishguard.models import LABEL_ORDER, PHISHING_LABEL
from phishguard.split import SEED

DATASET_ID = 327
SPLIT_SEED = SEED
POSITIVE_LABEL = PHISHING_LABEL
CONFUSION_LABEL_ORDER = LABEL_ORDER
LEADING_MODEL = "knn_k_9"
FINAL_MODEL_NAMES = (
    "majority",
    "tree_depth_5",
    "tree_unrestricted",
    "knn_k_9",
)


def make_final_models():
    """Return new, unfitted estimators in the documented comparison order."""
    return (
        ("majority", DummyClassifier(strategy="most_frequent")),
        ("tree_depth_5", DecisionTreeClassifier(criterion="entropy", max_depth=5, random_state=SEED)),
        ("tree_unrestricted", DecisionTreeClassifier(criterion="entropy", max_depth=None, random_state=SEED)),
        ("knn_k_9", KNeighborsClassifier(n_neighbors=9, metric="hamming", algorithm="brute")),
    )
