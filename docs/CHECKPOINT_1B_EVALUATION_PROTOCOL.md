# Checkpoint 1B: evaluation protocol

No classifier has been fitted. This checkpoint fixes row assignments and records the evidence needed before class-specific evaluation.

## Label evidence

**Verified from the downloaded official UCI 327 data:** the target column is `result`, its only values are `-1` and `1`, and their counts are 4,898 and 6,157 respectively. The [UCI dataset page](https://archive.ics.uci.edu/dataset/327/phishing+websites) names the target and describes the 30 features but does not define which target number means phishing. The official `Training Dataset.arff` header says `@attribute Result { -1,1 }` without semantic names. The authors' [feature description](https://eprints.hud.ac.uk/id/eprint/24330/6/MohammadPhishing14July2015.pdf) describes phishing and legitimate **feature** states, but does not explicitly map the `Result` target.

**Current project convention, still an assumption:** `-1 = phishing`, `1 = legitimate`, as recorded in README.md and Checkpoint 1. Stevanović, [*Embedding and Weighting of Website Features for Phishing Detection* (2025)](https://casopisi.junis.ni.ac.rs/index.php/FUMathInf/article/download/11279/pdf), section 3.1, independently describes the same 11,055-row UCI dataset as 4,898 phishing and 6,157 legitimate sites. That supports the class counts under the project convention, but is not an explicit class dictionary from the data creators.

**Conflicting source:** Hutchinson et al., [*Detecting Phishing Websites with Random Forest* (2018)](https://eudl.eu/pdf/10.1007/978-3-030-00557-3_46), section 3.1, says that the 11,055-row UCI data has `1 = phishing`, `-1 = legitimate` and gives counts of 6,157 phishing and 4,898 legitimate. This is a research team's description of its use of UCI data, not UCI's own target dictionary; it may reflect a reversal or an error, and the paper does not settle the source encoding. Some other studies use relabeled targets, which also cannot establish the original coding.

**Decision needed before phishing-class metrics:** obtain an explicit class dictionary from the dataset creators or UCI, or consciously adopt a documented convention with the uncertainty stated in the report. Keep raw numeric labels and never silently invert them. Until this is settled, the split table below reports numeric classes only.

## Fixed split

`phishguard.split.split_dataset` uses seed **42**. It groups observations by an exact match of **all 30 feature values**, excluding the target. `GroupShuffleSplit` first holds out test groups at 15% of groups, then assigns validation groups from the remainder at `0.15 / 0.85`; all other groups are training. Targets are not used to define or select groups. It returns sorted, zero-based row positions in original UCI order. Use `dataset.features.iloc[indices]` and `dataset.target.iloc[indices]` to select each partition; the original observations and raw labels remain intact.

The percentages are **targets, not exact row quotas**. The algorithm samples groups, whose sizes vary (up to 25 rows). It does not stratify labels, because conflicting groups can contain both classes and exact balance is not guaranteed. The held-out test positions are fixed before any tuning. Later parameter selection should use training and validation only, then evaluate the selected model once on test.

Actual results on the official UCI `Training Dataset.arff` in the local unmodified UCI archive (cross-checked against the `ucimlrepo` intake report):

| Partition | Rows | Share of rows | `-1` count | `-1` within partition | `1` count | `1` within partition | Feature groups |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Train | 7,676 | 69.43% | 3,428 | 44.66% | 4,248 | 55.34% | 4,049 |
| Validation | 1,629 | 14.74% | 687 | 42.17% | 942 | 57.83% | 868 |
| Test | 1,750 | 15.83% | 783 | 44.74% | 967 | 55.26% | 868 |
| **Total** | **11,055** | **100%** | **4,898** | **44.31% overall** | **6,157** | **55.69% overall** | **5,785** |

The 5,785 groups occur in exactly one partition each. No observation was removed, duplicated in assignments, merged, or relabeled. All 64 groups with conflicting labels (357 rows) from [Checkpoint 1](CHECKPOINT_1_DATASET_INTAKE.md) remain intact inside single partitions: train has 44 groups/252 rows, validation has 8 groups/36 rows, and test has 12 groups/69 rows. They make a perfect mapping from feature vector to target impossible for those vectors; they may reflect coarse features, different underlying sites, or annotation errors. The data does not let us distinguish those explanations.

To reproduce with the official loader:

```python
from phishguard.data import load_dataset
from phishguard.split import split_dataset, split_summary

dataset = load_dataset()  # requires UCI network access
indices = split_dataset(dataset)  # seed 42 by default
print(split_summary(dataset, indices))
```

## Checks and limits

Run `.venv/bin/python -m pytest -q` and `git diff --check`. Offline tests verify complete and unique row assignment, exact feature-group isolation, fixed-seed repeatability, independence from target assignment, unchanged input data, and raw class totals/proportions. They use synthetic fixtures; no internet access is needed. An actual-data run checked split counts and feature-group coverage against the official archive and Checkpoint 1 intake statistics.

The fixed seed makes this protocol repeatable under the repository's supported package versions, but group sampling and row counts are not guaranteed to be identical after major library algorithm changes. Keep the seed and library versions in the final report. Group isolation prevents exact feature-vector overlap only; raw domain identities and collection times are unavailable, so related sites may still cross partitions. No accuracy or security claim follows from this split.
