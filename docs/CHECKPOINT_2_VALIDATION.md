# Checkpoint 2: initial model validation

The official UCI Phishing Websites dataset (ID 327) was loaded through `ucimlrepo`, checked by the existing loader, and split with the unchanged feature-group protocol and seed 42. Models used 7,676 training rows; the results below use only 1,629 validation rows. The 1,750 held-out test rows were not evaluated or used for selection. No rows or labels were changed.

## Working label interpretation

This checkpoint follows the approved working convention **`-1 = phishing`, `1 = legitimate`** and explicitly uses `pos_label=-1` for precision, recall, and F1. This is a research assumption, not a verified UCI class dictionary. [Checkpoint 1B](CHECKPOINT_1B_EVALUATION_PROTOCOL.md) records the source evidence and the conflicting published definition, which claims the reverse mapping. The raw target values remain unchanged.

As a limited, **training-only** consistency check, the authors' SSL final state rule associates `-1` with phishing and `1` with legitimacy. In training rows with `sslfinal_state=-1`, 2,048/2,408 (85.05%) have raw target `-1`. With `sslfinal_state=1`, 533/4,405 (12.10%) have raw target `-1`. The feature also has a `0` category, omitted from this two-state comparison. This pattern supports the working mapping but does not prove the target semantics: features are imperfect, and a feature's coding is not a target dictionary. These counts were not used for model selection.

## Models and protocol

- **Baseline:** `DummyClassifier(strategy="most_frequent")` fitted only on training rows. It learned raw label `1` (4,248 of 7,676 training rows).
- **Tree:** scikit-learn `DecisionTreeClassifier(criterion="entropy", random_state=42)` with `max_depth` 3, 5, and `None`. This is scikit-learn's CART-style tree using entropy, **not exact ID3**.
- **KNN:** `KNeighborsClassifier(metric="hamming", algorithm="brute")` with 3, 5, and 9 neighbors. Hamming distance counts the fraction of the 30 features whose category values differ. It treats `-1`, `0`, and `1` as discrete categories: `-1` versus `0` is one mismatch, as is `-1` versus `1`. This avoids interpreting category codes as numeric distances. All features have equal weight; no scaling or encoding was added. Ties use scikit-learn's default uniform vote behavior.

Each model was fitted on training data only and predicted validation rows once. The selected setting maximizes validation phishing F1; ties break by accuracy, then listed order. Prediction runtime measures only the `predict` call for all 1,629 validation rows using `time.perf_counter`, excludes fitting and metric calculation, and is hardware/load dependent.

The confusion matrix uses **rows = actual `[-1, 1]`**, **columns = predicted `[-1, 1]`**. With `-1` as phishing, `[[TP, FN], [FP, TN]]` applies.

| Model | Accuracy | Phishing precision | Phishing recall | Phishing F1 | Confusion matrix | Predict (s) |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| Majority (`1`) | 0.5783 | 0.0000 | 0.0000 | 0.0000 | `[[0,687],[0,942]]` | 0.00008 |
| Tree depth 3 | 0.9202 | 0.9877 | 0.8210 | 0.8967 | `[[564,123],[7,935]]` | 0.00075 |
| Tree depth 5 | 0.9386 | 0.9819 | 0.8705 | 0.9228 | `[[598,89],[11,931]]` | 0.00082 |
| Tree unrestricted | 0.9355 | 0.9122 | 0.9374 | 0.9246 | `[[644,43],[62,880]]` | 0.00140 |
| KNN k=3 | 0.9411 | 0.9352 | 0.9243 | 0.9297 | `[[635,52],[44,898]]` | 0.23416 |
| KNN k=5 | 0.9411 | 0.9391 | 0.9199 | 0.9294 | `[[632,55],[41,901]]` | 0.36944 |
| **KNN k=9** | **0.9546** | **0.9609** | **0.9301** | **0.9453** | `[[639,48],[26,916]]` | 0.26266 |

KNN k=9 has the highest validation phishing F1 and accuracy among these settings. The unrestricted tree has the highest phishing recall (0.9374), but also 62 false alarms versus 26 for KNN k=9. Tree depth 3 has the highest precision (0.9877), with lower recall. Timing differences between tree and KNN are substantial in this run; timing within the three KNN settings should not be interpreted as a stable ordering from a single measurement.

## Reproduce and verify

From the repository root with access to UCI's dataset API:

```bash
.venv/bin/python -m scripts.validate_models
.venv/bin/python -m pytest -q
git diff --check
```

The complete machine-readable output from this run is in ignored local `results/generated/checkpoint2_validation.json`. Offline tests cover data dimensions, repeatability of predictions and metrics, baseline learning, phishing-positive metric and confusion-matrix semantics, and a guard showing test positions are never read by the validation runner. They use synthetic fixtures with network sockets blocked.

## Limits and next checkpoint

The 64 conflicting feature groups remain intact in one partition each; 44 groups/252 rows are in training and 8 groups/36 rows are in validation. Identical feature vectors can therefore have opposing labels, limiting deterministic classification. Exact feature-vector overlap is blocked across partitions, but raw site identities and collection dates are unavailable. Validation was used to choose among the specified settings, so its best score is a selection result, not an unbiased final performance estimate. Before reporting final phishing-class test metrics, settle or explicitly retain the target-mapping assumption. Then lock one chosen configuration and evaluate the held-out test exactly once in a later checkpoint.
