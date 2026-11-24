# Checkpoint 3B: frozen held-out evaluation protocol

**Status: locked before final test evaluation.** No held-out predictions or label analysis were run in this checkpoint. The next checkpoint may execute the one-time evaluation described below; its results must not change these choices.

## Dataset and partition lock

- Source: official UCI *Phishing Websites* dataset **ID 327**, loaded by the existing `ucimlrepo` loader with its schema checks. Preserve all 11,055 original observations, all 30 precomputed categorical features, and the raw `result` labels. No deduplication, relabeling, conflict resolution, preprocessing, or site fetching.
- Use the existing `split_dataset` implementation unchanged: exact matching of all 30 feature values defines groups, the target does not define groups, seed **42**, and an approximate 70% / 15% / 15% train/validation/test row split. Existing observed partition sizes are 7,676 / 1,629 / 1,750. The held-out partition must remain separate from fitting and selection.
- Working class interpretation: **`-1 = phishing`, `1 = legitimate`**. This is a declared research convention, not an explicit UCI target dictionary. [Checkpoint 1B](CHECKPOINT_1B_EVALUATION_PROTOCOL.md) records both supporting evidence and the published study that states the reverse mapping. Keep raw labels and disclose this uncertainty alongside final class-specific metrics.

## Model lock and selection rationale

Use the four unfitted estimators from `phishguard.final_protocol.make_final_models()` in this exact order. No new hyperparameters or model families are to be selected after test observation.

| Comparison | Frozen configuration | Validation phishing F1 used for decision |
| --- | --- | ---: |
| Majority baseline | `DummyClassifier(strategy="most_frequent")`; learn the majority from training rows only | 0.0000 |
| Decision Tree depth 5 | `DecisionTreeClassifier(criterion="entropy", max_depth=5, random_state=42)` | 0.9228 |
| Decision Tree unrestricted | `DecisionTreeClassifier(criterion="entropy", max_depth=None, random_state=42)` | 0.9246 |
| **KNN k=9** | `KNeighborsClassifier(n_neighbors=9, metric="hamming", algorithm="brute", weights="uniform")` | **0.9453** |

KNN k=9 is the **leading model**, selected by the highest validation phishing F1 in [Checkpoint 2](CHECKPOINT_2_VALIDATION.md). The unrestricted tree is retained because it had the highest validation F1 within the tree family; [Checkpoint 3A](CHECKPOINT_3A_FEATURE_ANALYSIS.md) also found a sizable train/validation gap for it. The depth-5 tree is retained for readable feature analysis and an overfitting comparison. The baseline provides a training-majority reference. Scikit-learn's entropy tree uses CART-style binary thresholds; it is **not exact ID3**. Hamming distance counts category mismatches across the 30 features without treating `-1`, `0`, and `1` as ordinal distances.

## One-time final evaluation procedure, for the next checkpoint

1. Load UCI ID 327 and run the established schema and source checks. Recreate the existing seed-42 split. Before scoring, verify the expected row counts, complete and unique row assignment, and that no exact 30-feature group crosses partitions. If source order, schema, or split assignments drift from the locked protocol, stop and investigate before using test results.
2. Construct exactly the four frozen models from `make_final_models()`. Fit each using **training rows only**. Do not combine validation and training or use test rows for fitting, preprocessing, feature selection, threshold choice, or another parameter comparison.
3. Predict each model's held-out test rows once. Compute accuracy and phishing precision, recall, and F1 with **`pos_label=-1`**. Compute `confusion_matrix(..., labels=[-1, 1])`; rows are actual and columns predicted, so the cells are `[[TP, FN], [FP, TN]]` under the working class interpretation. Report each model's full matrix and test metrics.
4. Time only each `predict` call and describe the runtime as dependent on hardware and load. Compare the test metrics with the already recorded validation metrics, including absolute differences. Report all four models irrespective of their test ranking.
5. Preserve **KNN k=9** as the validation-selected leading model even if another model scores higher on test. Do not tune, rerun to pick a better seed, or reselect the headline model based on test outcomes. Explain the historical dataset, duplicate/conflicting labels, absent site identities and timestamps, class-name disagreement, and limits on generalization.

## Code review and safeguards

The existing `run_validation` and `analyze_training_validation` functions select only `split.train` and `split.validation`; their offline tests pass an invalid test position to catch accidental access. The split groups by features alone, although its schema validation examines the target column. The older `inspect_dataset` and `split_summary` functions can read full-dataset labels, so they must **not** be called during model selection or this pre-test lock. Checkpoint 1B previously recorded test class counts; those were not used to choose the models above. No final test runner exists yet, so this checkpoint adds only an unfitted model factory and configuration tests; it does not contain a route that predicts or scores test rows.

The final evaluation runner still needs explicit preflight checks for source/schema and split integrity, must use only the frozen factory, and should fail before any test scoring if those checks do not pass. If later code changes the loader, split, feature order, label convention, or model parameters, review and re-lock the protocol **before** opening the test partition. The seed alone does not protect against source row-order or library-version changes.

## Verification

`tests/test_final_protocol.py` checks the official dataset ID, fixed seed, label order, phishing-positive label, complete model list, exact key parameters, and that factory calls return fresh, unfitted estimators. The full suite remains offline. Run `.venv/bin/python -m pytest -q` and `git diff --check` before review. No test predictions or held-out labels are generated by these checks.
