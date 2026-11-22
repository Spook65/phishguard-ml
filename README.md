# PhishGuard ML

A reproducible COMP 151 machine-learning comparison for classifying websites as **phishing** or **legitimate** using an entropy-based decision tree and K-nearest neighbors.

> Status: **Project scaffold only.** No models have been trained and no accuracy claims are made yet.

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

Python 3.11+ recommended. `pytest` initially validates only the project scaffold; training code comes in a later checkpoint.

## Dataset
Official source: https://archive.ics.uci.edu/dataset/327/phishing+websites

UCI dataset credit: Mohammad, R. & McCluskey, L. (2012). *Phishing Websites* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C51W2X. Licensed CC BY 4.0.

The dataset provides **precomputed numeric features**, not raw website files. Labels must be checked after loading (`-1` for phishing, `1` for legitimate according to dataset conventions). Avoid downloading or visiting live phishing URLs.

## Research caveats
This is a course benchmark using a historical dataset donated in 2015, **not** an operational security scanner. Avoid claims of modern threat detection. Keep the test set unseen until final evaluation; tune settings on validation data only.

## Next checkpoint
Follow `docs/CODEX_START_HERE.md`. Implement data loading + schema verification only, then inspect the output before model training.
