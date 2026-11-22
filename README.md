# PhishGuard ML

A reproducible COMP 151 machine-learning comparison for classifying websites as **phishing** or **legitimate** using an entropy-based decision tree and K-nearest neighbors.

> Status: **Checkpoint 1 — dataset intake implemented.** No models have been trained and no accuracy claims are made yet. See [the intake report](docs/CHECKPOINT_1_DATASET_INTAKE.md) for findings and review items.

## Scope (two weeks)
- Official UCI Phishing Websites dataset (ID 327; 11,055 records, 30 features).
- Decision Tree (`criterion="entropy"`) vs KNN with a categorical-aware distance metric.
- Majority-class baseline, basic parameter comparisons, held-out evaluation.
- Accuracy, phishing-class precision/recall/F1, confusion matrix, prediction time.
- Reproducible splits; check duplicate feature vectors to reduce leakage.

**Out of scope:** live URL scanning, fetching phishing links, browser extensions, and integration with SecureAgent Lab.

## Local setup
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pytest -q
```

Python 3.11+ recommended. Tests cover dataset validation and duplicate/conflict inspection using synthetic fixtures, with network access blocked. Training code comes in a later checkpoint.

Run dataset intake from the repository root (requires access to official UCI servers):

```bash
python -m scripts.inspect_dataset
# Optional: save the JSON report locally (ignored by git).
mkdir -p results/generated
python -m scripts.inspect_dataset > results/generated/dataset_inspection.json
```

The reusable loader is `phishguard.data.load_dataset()`. It uses `fetch_ucirepo(id=327)`, validates the original schema and category domains, and preserves raw target values. It fails on unexpected data instead of cleaning it silently. Each invocation fetches UCI again; the loader has no disk cache.

## Dataset
Official source: https://archive.ics.uci.edu/dataset/327/phishing+websites

UCI dataset credit: Mohammad, R. & McCluskey, L. (2012). *Phishing Websites* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C51W2X. Licensed CC BY 4.0.

The dataset provides **precomputed numeric features**, not raw website files. The project convention is `-1` for phishing and `1` for legitimate. Intake verified the numeric target domain and counts; the downloaded UCI metadata and ARFF do not explicitly name the target classes. Primary-source confirmation of the semantic mapping remains a review item before evaluation. Avoid downloading or visiting live phishing URLs.

## Research caveats
This is a course benchmark using a historical dataset donated in 2015, **not** an operational security scanner. Avoid claims of modern threat detection. Keep the test set unseen until final evaluation; tune settings on validation data only.

## Next checkpoint
Review [Checkpoint 1](docs/CHECKPOINT_1_DATASET_INTAKE.md), especially target semantics, duplicate grouping, and contradictory labels, before authorizing splitting or model training.
