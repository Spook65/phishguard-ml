# Checkpoint 3C: one-time held-out test evaluation

The frozen [Checkpoint 3B protocol](CHECKPOINT_3B_TEST_PROTOCOL.md) was evaluated once on the official UCI Phishing Websites dataset (ID 327) on November 23, 2026. KNN k=9 remains the leading model **because it was selected before test evaluation by validation phishing F1**. All four predetermined configurations are reported below; no test-driven tuning, reselection, new features, or relabeling occurred. The complete machine-readable output is in [final_evaluation.json](../results/checkpoint3c/final_evaluation.json).

## Pre-scoring verification

The offline suite passed before the held-out run. A separate `--preflight-only` invocation then passed without scoring or selecting test labels. The final invocation repeated the gate before selecting test labels or predicting:

| Check | Observed / result |
| --- | --- |
| UCI source | ID 327, name `Phishing Websites` |
| Source/schema | 11,055 rows, 30 ordered categorical features, one integer `result` column |
| Group-safe seed | 42; existing split implementation unchanged |
| Row assignments | 7,676 train + 1,629 validation + 1,750 test = 11,055; each row exactly once |
| Feature groups | 4,049 train, 868 validation, 868 test; no identical 30-feature vector crosses partitions |
| Model list | Majority; entropy tree depth 5; entropy tree unrestricted; KNN k=9 with Hamming distance, in frozen order and unfitted at preflight |
| Fitting scope | Original training rows only; no validation or test rows used for fitting |
| Metric convention | `pos_label=-1`; confusion labels `[-1, 1]` |

The existing loader and splitter were given a deferred target-value check during preflight, so test target values were not inspected to establish the source, split, groups, or model settings. After those checks passed, the normal full target validation succeeded, then the planned held-out evaluation ran. The final script refuses to overwrite its results file, guarding against accidental reruns.

## Final held-out metrics

The working interpretation remains **`-1 = phishing`, `1 = legitimate`**. For every confusion matrix, rows are actual `[-1, 1]` and columns predicted `[-1, 1]`, so `[[TP, FN], [FP, TN]]` applies when phishing is positive. The test partition contains 1,750 rows. Times cover one `predict` call over that partition, exclude fitting and scoring, and are descriptive.

| Frozen model | Accuracy | Phishing precision | Phishing recall | Phishing F1 | Confusion matrix | Predict (s) |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| Majority baseline | 0.5526 | 0.0000 | 0.0000 | 0.0000 | `[[0,783],[0,967]]` | 0.000293 |
| Entropy tree, depth 5 | 0.9211 | 0.9562 | 0.8633 | 0.9074 | `[[676,107],[31,936]]` | 0.000534 |
| Entropy tree, unrestricted | 0.9200 | 0.9004 | 0.9234 | 0.9117 | `[[723,60],[80,887]]` | 0.000650 |
| **KNN, k=9, Hamming** | **0.9251** | 0.9347 | 0.8953 | **0.9145** | `[[701,82],[49,918]]` | 0.301622 |

Under the working label mapping, KNN k=9 had 701 true phishing predictions, missed 82 phishing rows, and falsely flagged 49 legitimate rows. The unrestricted tree recovered more phishing rows (723 true positives, 60 misses) but also produced more false alarms (80). The depth-5 tree made 31 false alarms but missed 107 phishing rows. These are observations on this historical held-out partition, not operational security guarantees.

## Validation versus test

The validation reference is derived from the exact matrices recorded in [Checkpoint 2](CHECKPOINT_2_VALIDATION.md). Differences below are **test minus validation**, in percentage points; negative values mean lower test performance.

| Frozen model | Validation accuracy | Test accuracy | Difference | Validation phishing F1 | Test phishing F1 | Difference |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Majority baseline | 0.5783 | 0.5526 | −2.57 | 0.0000 | 0.0000 | 0.00 |
| Entropy tree, depth 5 | 0.9386 | 0.9211 | −1.75 | 0.9228 | 0.9074 | −1.55 |
| Entropy tree, unrestricted | 0.9355 | 0.9200 | −1.55 | 0.9246 | 0.9117 | −1.29 |
| **KNN, k=9** | **0.9546** | **0.9251** | **−2.94** | **0.9453** | **0.9145** | **−3.07** |

All three trained classifiers scored lower on test than validation. KNN had the largest decline, yet still narrowly led on test F1 (0.9145 versus 0.9117 for the unrestricted tree). The test ranking does not reopen model selection. The unrestricted tree's [large train/validation gap](CHECKPOINT_3A_FEATURE_ANALYSIS.md) remains evidence of overfitting; its final test score neither erases that diagnosis nor proves stability across other data. The baseline's accuracy change partly reflects the differing class proportions across partitions.

## Limits and reproducibility

The UCI data is historical and contains repeated feature vectors and 64 groups whose identical 30-feature values have conflicting labels. The split prevents exact feature-vector overlap across partitions, but raw website identities and collection dates are unavailable, so related sites or time effects cannot be ruled out. The project follows `-1 = phishing`, yet [published accounts disagree on which raw target value names the phishing class](CHECKPOINT_1B_EVALUATION_PROTOCOL.md), and UCI's current metadata does not explicitly resolve that disagreement. Precision, recall, F1, and the TP/FN/FP/TN interpretation depend on this declared mapping. The numeric predictions and matrices are preserved without relabeling.

To reproduce in a fresh checkout with official UCI access, run `.venv/bin/python -m scripts.final_evaluation --preflight-only`, then `.venv/bin/python -m scripts.final_evaluation` **once** when the output path does not already exist. The CLI refuses to replace `results/checkpoint3c/final_evaluation.json`. To inspect this completed run, read the checked-in JSON instead of rerunning held-out scoring. Offline verification: `.venv/bin/python -m pytest -q` (**48 passed**) and `git diff --check` (**passed**). A read-only check of the saved JSON confirmed four frozen model names and 1,750 observations per confusion matrix without rescoring.
