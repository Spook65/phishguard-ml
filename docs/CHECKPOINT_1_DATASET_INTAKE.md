# Checkpoint 1: dataset intake

Inspected on November 21, 2026. No models trained, splits created, or observations removed.

## Implementation and reproduction

- `phishguard/data.py`: reusable UCI loader, strict schema/category checks, and non-mutating inspection.
- `scripts/inspect_dataset.py`: prints JSON; returns nonzero on intake failure. Run from the repository root.
- `tests/test_data.py`: small, explicitly synthetic fixtures and mocked fetching; sockets and real UCI fetching blocked during tests.
- `README.md`: intake command, status, and mapping caveat.
- This report: measured findings, all feature domains/counts, and decisions for review.

```bash
.venv/bin/python -m scripts.inspect_dataset
.venv/bin/python -m pytest -q
git diff --check
```

The live intake script succeeded. Tests: **25 passed in 0.69s**, including the existing smoke test. No dependencies installed or changed. Environment: Python 3.11.18, ucimlrepo 0.0.7, pandas 2.3.3, pytest 9.2.0. An independent parse of the official Training Dataset.arff also matched shape, numeric label counts, feature duplicate count, and identical labeled duplicate count.

The initial sandbox fetch failed DNS resolution. Retrying with approved network access succeeded. Tests require no network. Subsequent intake runs need network access and fail rather than substitute data.

## Provenance and label semantics

Data loaded using `ucimlrepo.fetch_ucirepo(id=327)` from the [official UCI dataset](https://archive.ics.uci.edu/dataset/327/phishing+websites). Source data URL: https://archive.ics.uci.edu/static/public/327/data.csv. UCI metadata last_updated: Tue Mar 05 2024.

Credit: Mohammad, R. & McCluskey, L. (2012). Phishing Websites [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C51W2X. License: CC BY 4.0.

The official [UCI archive](https://archive.ics.uci.edu/static/public/327/phishing+websites.zip) contains Training Dataset.arff, .old.arff, and Phishing Websites Features.docx. Only the current training ARFF was used for the cross-check; .old.arff is a different older encoding and must not be mixed into this benchmark. Documentation was parsed as text; none of its example URLs or embedded content was executed or retrieved.

Verified numeric target: one integer column, `result`, with exactly {-1, 1}; no zero or missing labels. **Class names remain a review item:** the existing project README assigns -1 = phishing and 1 = legitimate. The current UCI API metadata, ARFF header, feature document, and authors' [feature description](https://eprints.hud.ac.uk/id/eprint/24330/6/MohammadPhishing14July2015.pdf) do not explicitly link numeric target values to class names. The code labels this mapping as a project convention needing primary-source confirmation; it does not infer semantics from counts, feature correlations, or feature-level rules. Raw labels are never remapped.

| Raw target | Project convention (pending confirmation) | Count | Share |
| --- | --- | ---: | ---: |
| -1 | phishing | 4,898 | 44.31% |
| 1 | legitimate | 6,157 | 55.69% |

## Dataset statistics

| Measurement | Observed |
| --- | ---: |
| Rows | 11,055 |
| Features (target excluded) | 30 |
| Missing feature or target values | 0 |
| Distinct feature vectors | 5,785 |
| Feature groups occurring more than once | 2,614 |
| Rows in repeated feature groups (all occurrences) | 7,884 |
| Extra feature rows beyond first occurrence | 5,270 |
| Largest feature group | 25 |
| Extra identical rows including target | 5,206 |
| Feature groups containing both labels | 64 |
| Rows in conflicting groups (all occurrences) | 357 |

Feature duplicates compare all 30 features, excluding the target. Identical labeled duplicates compare all 31 columns. Conflicting groups contain both numeric target labels. All 64 conflict groups and their original zero-based row positions and label counts are listed in the local JSON report.

Example: zero-based positions 111, 252, 1480, 1944, 2487, 3518, 4348, 4371, 4943 share one feature vector, but have four -1 labels and five +1 labels.

## Every feature inspected

Original UCI API names and ordering are preserved, including spelling. All features and target have dtype `int64`; each column has zero missing values. Counts below are observations from the fetched data. Allowed numeric domains were checked against Training Dataset.arff; numeric codes are categorical, and their semantic interpretation is feature-specific.

| Feature | -1 count | 0 count | 1 count |
| --- | ---: | ---: | ---: |
| `having_ip_address` | 3793 | — | 7262 |
| `url_length` | 8960 | 135 | 1960 |
| `shortining_service` | 1444 | — | 9611 |
| `having_at_symbol` | 1655 | — | 9400 |
| `double_slash_redirecting` | 1429 | — | 9626 |
| `prefix_suffix` | 9590 | — | 1465 |
| `having_sub_domain` | 3363 | 3622 | 4070 |
| `sslfinal_state` | 3557 | 1167 | 6331 |
| `domain_registration_length` | 7389 | — | 3666 |
| `favicon` | 2053 | — | 9002 |
| `port` | 1502 | — | 9553 |
| `https_token` | 1796 | — | 9259 |
| `request_url` | 4495 | — | 6560 |
| `url_of_anchor` | 3282 | 5337 | 2436 |
| `links_in_tags` | 3956 | 4449 | 2650 |
| `sfh` | 8440 | 761 | 1854 |
| `submitting_to_email` | 2014 | — | 9041 |
| `abnormal_url` | 1629 | — | 9426 |
| `redirect` | — | 9776 | 1279 |
| `on_mouseover` | 1315 | — | 9740 |
| `rightclick` | 476 | — | 10579 |
| `popupwindow` | 2137 | — | 8918 |
| `iframe` | 1012 | — | 10043 |
| `age_of_domain` | 5189 | — | 5866 |
| `dnsrecord` | 3443 | — | 7612 |
| `web_traffic` | 2655 | 2569 | 5831 |
| `page_rank` | 8201 | — | 2854 |
| `google_index` | 1539 | — | 9516 |
| `links_pointing_to_page` | 548 | 6156 | 4351 |
| `statistical_report` | 1550 | — | 9505 |

## Local evidence

`results/generated/dataset_inspection.json` contains the full generated report, including per-column counts and conflicting group row positions. `data/raw/uci327-source.zip` contains the unmodified official documentation/data archive. Both paths are ignored by git. The CLI prints JSON and does not itself save or cache raw observations.

- Inspection JSON SHA-256: `2966fa267f4cb8fc8ea450eaf1bcc8e38bfc6a757ab049c4bc28942cc802cc79`
- UCI archive SHA-256: `514f582cb44cba17251baf1d59f64cddff16efa2b644d091ce8d2a1e8d492940`

## Risks and decisions before Checkpoint 2

1. Confirm the numeric-to-class-name mapping before computing any phishing-class metrics. This is the unresolved semantic verification item from intake.
2. Decide how to keep identical feature patterns in one partition. Random row splitting can place the same pattern into training and test data and inflate evaluation results. No split policy has been implemented.
3. Decide how to handle contradictory labels. Coarse categorical features can collapse different websites to the same vector; conflicting labels alone do not prove annotation errors. No rows were deleted, merged, voted on, or relabeled.
4. Repeated observations can overweight common patterns. Report whether the eventual experiment preserves or deduplicates them; any policy needs to be fixed before tuning or evaluating.
5. These are historical precomputed observations, with no raw website identity or timestamps supplied in this intake. Feature grouping cannot establish domain independence, temporal generalization, or modern security performance.
6. Some source feature prose describes categories absent in the current ARFF (for example request_url and redirect). Validate per-column numeric domains rather than assume every feature has all three values or apply a universal semantic decoder.

Stop here for review. Training, tuning, held-out evaluation, and classifier performance metrics belong to later checkpoints.
