"""Official UCI 327 intake; no cleaning, splitting, or model fitting."""

from dataclasses import dataclass

import pandas as pd
from pandas.api.types import is_integer_dtype
from ucimlrepo import fetch_ucirepo

DATASET_ID = 327
EXPECTED_ROWS = 11_055
FEATURE_NAMES = (
    "having_ip_address", "url_length", "shortining_service", "having_at_symbol",
    "double_slash_redirecting", "prefix_suffix", "having_sub_domain",
    "sslfinal_state", "domain_registration_length", "favicon", "port",
    "https_token", "request_url", "url_of_anchor", "links_in_tags", "sfh",
    "submitting_to_email", "abnormal_url", "redirect", "on_mouseover",
    "rightclick", "popupwindow", "iframe", "age_of_domain", "dnsrecord",
    "web_traffic", "page_rank", "google_index", "links_pointing_to_page",
    "statistical_report",
)
TERNARY_FEATURES = {
    "url_length", "having_sub_domain", "sslfinal_state", "url_of_anchor",
    "links_in_tags", "sfh", "web_traffic", "links_pointing_to_page",
}
FEATURE_VALUES = {
    name: ({0, 1} if name == "redirect" else
           {-1, 0, 1} if name in TERNARY_FEATURES else {-1, 1})
    for name in FEATURE_NAMES
}
# README convention, NOT inferred from counts or feature/target correlations.
# UCI's current metadata/ARFF only explicitly specify the numeric target domain.
LABEL_MAPPING = {-1: "phishing", 1: "legitimate"}
MAPPING_STATUS = "Project README convention; semantic mapping needs primary-source confirmation"


@dataclass
class PhishingDataset:
    features: pd.DataFrame
    target: pd.Series
    metadata: dict


def validate_dataset(features, targets, *, expected_rows=EXPECTED_ROWS, check_target_values=True):
    """Fail on schema drift, missing values, misalignment, or invalid categories.

    expected_rows is adjustable for small offline test fixtures. Production
    loading always uses the official row count. Original labels are preserved.
    When check_target_values is false, only target shape, name, index, and
    integer dtype are checked; target contents are deferred until preflight
    has passed. The default retains full validation for all existing callers.
    """
    if not isinstance(features, pd.DataFrame) or not isinstance(targets, pd.DataFrame):
        raise ValueError("Features and targets must be pandas DataFrames")
    if features.shape != (expected_rows, len(FEATURE_NAMES)):
        raise ValueError(f"Expected {expected_rows} rows and 30 features; got {features.shape}")
    if tuple(features.columns) != FEATURE_NAMES:
        raise ValueError("Unexpected feature names or order")
    if targets.shape != (expected_rows, 1) or list(targets.columns) != ["result"]:
        raise ValueError("Expected one target column named result with matching row count")
    if not features.index.equals(targets.index) or not features.index.is_unique:
        raise ValueError("Feature and target indices must match and be unique")
    if features.isna().any().any() or (check_target_values and targets.isna().any().any()):
        raise ValueError("Missing values found")
    for name, allowed in FEATURE_VALUES.items():
        if not is_integer_dtype(features[name]) or not set(features[name]).issubset(allowed):
            raise ValueError(f"Invalid integer categories in {name}; allowed {sorted(allowed)}")
    target = targets["result"]
    if not is_integer_dtype(target) or (check_target_values and set(target) != set(LABEL_MAPPING)):
        raise ValueError("Expected both integer target labels -1 and 1")


def load_dataset(*, check_target_values=True):
    """Fetch only precomputed UCI data via ucimlrepo; never fetch website URLs.

    Network errors propagate: there is no fallback data or silent substitution.
    The final evaluator may defer target-value checks until after preflight.
    """
    dataset = fetch_ucirepo(id=DATASET_ID)
    if dataset.metadata.get("uci_id") != DATASET_ID:
        raise ValueError("Unexpected UCI dataset ID")
    features, targets = dataset.data.features, dataset.data.targets
    validate_dataset(features, targets, check_target_values=check_target_values)
    return PhishingDataset(features.copy(), targets["result"].copy(), dict(dataset.metadata))


def inspect_dataset(dataset):
    """Return JSON-compatible statistics without altering observations.

    Duplicate groups use all 30 features, excluding the target. Row positions
    are zero based and preserve UCI order, independent of DataFrame index.
    """
    features, target = dataset.features, dataset.target
    validate_dataset(features, target.to_frame(), expected_rows=len(features))
    combined = features.reset_index(drop=True).copy()
    combined["result"] = target.to_numpy()
    grouped = combined.groupby(list(FEATURE_NAMES), sort=False, dropna=False)
    sizes = grouped.size()
    label_counts = grouped["result"].nunique()
    conflicts = []
    for _, group in grouped:
        if group["result"].nunique() > 1:
            conflicts.append({
                "row_positions": group.index.tolist(),
                "label_counts": {str(k): int(v) for k, v in group["result"].value_counts().sort_index().items()},
            })
    return {
        "source": {key: dataset.metadata.get(key) for key in
                   ("uci_id", "name", "repository_url", "data_url", "dataset_doi", "last_updated")},
        "rows": len(features),
        "feature_count": features.shape[1],
        "features": {name: {
            "dtype": str(features[name].dtype),
            "value_counts": {str(k): int(v) for k, v in features[name].value_counts().sort_index().items()},
            "missing": int(features[name].isna().sum()),
        } for name in FEATURE_NAMES},
        "target": {
            "name": target.name,
            "dtype": str(target.dtype),
            "label_mapping": LABEL_MAPPING,
            "mapping_status": MAPPING_STATUS,
            "counts": {str(k): int(v) for k, v in target.value_counts().sort_index().items()},
            "missing": int(target.isna().sum()),
        },
        "missing_total": int(features.isna().sum().sum() + target.isna().sum()),
        "duplicates": {
            "unique_feature_vectors": len(sizes),
            "repeated_feature_groups": int((sizes > 1).sum()),
            "rows_in_repeated_groups": int(sizes[sizes > 1].sum()),
            "extra_rows_beyond_first": int((sizes - 1).sum()),
            "largest_group": int(sizes.max()),
            "conflicting_feature_groups": int((label_counts > 1).sum()),
            "rows_in_conflicting_groups": sum(len(c["row_positions"]) for c in conflicts),
            "extra_identical_labeled_rows": int(combined.duplicated().sum()),
            "conflicts": conflicts,
        },
    }
