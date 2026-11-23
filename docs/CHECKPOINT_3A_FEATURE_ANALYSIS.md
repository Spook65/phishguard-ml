# Checkpoint 3A: feature analysis and overfitting diagnostics

All calculations use the established UCI 327 dataset, feature-group split, and seed 42. Information gain, tree importance, and model fitting use **training rows only** (7,676). Diagnostics compare those rows with the existing **validation** partition (1,629). The held-out test partition was not accessed. Raw target values and the working `-1 = phishing` convention are unchanged; the conflicting published target definitions remain documented in [Checkpoint 1B](CHECKPOINT_1B_EVALUATION_PROTOCOL.md).

## Information gain

For the training labels, `H(Y) = -p(-1) log₂ p(-1) - p(1) log₂ p(1) = 0.991752` bits, with 3,428 raw `-1` and 4,248 raw `1` labels. For each observed feature value `v`, the code computes the label entropy in that subset, weights it by its fraction of training rows, and subtracts the sum from `H(Y)`. Values are discrete categories. The full 30-feature results, value counts, and conditional entropies are in [analysis.json](../results/checkpoint3a/analysis.json).

| Rank | Feature | Information gain (bits) |
| ---: | --- | ---: |
| 1 | `url_of_anchor` | 0.483167 |
| 2 | `sslfinal_state` | 0.480436 |
| 3 | `prefix_suffix` | 0.119782 |
| 4 | `web_traffic` | 0.113389 |
| 5 | `having_sub_domain` | 0.101709 |
| 6 | `links_in_tags` | 0.048098 |
| 7 | `request_url` | 0.041324 |
| 8 | `sfh` | 0.037567 |
| 9 | `domain_registration_length` | 0.033053 |
| 10 | `page_rank` | 0.009621 |

For `url_of_anchor`, the observed training categories are `-1` (2,328 rows, entropy 0.091209), `0` (3,734 rows, entropy 0.851213), and `1` (1,614 rows, entropy 0.317927). Their weighted contributions are 0.027662, 0.414074, and 0.066849 bits. Therefore `H(Y | url_of_anchor) = 0.508585` bits and `IG = 0.991752 - 0.508585 = 0.483167` bits. See the [top-10 chart](../results/checkpoint3a/information_gain_top10.png).

## Entropy-tree importance

The same entropy-based scikit-learn Decision Tree configurations from Checkpoint 2 were refitted to the unchanged training partition with `random_state=42`. Their `feature_importances_` values are normalized total reductions in weighted node impurity. This is scikit-learn's CART-style binary-threshold tree, **not exact ID3**. The [depth-5 importance chart](../results/checkpoint3a/tree_depth5_importance.png) shows its nonzero features. The [tree diagram](../results/checkpoint3a/tree_first_three_levels.png) shows only the root and next two levels of the depth-5 tree; deeper nodes are deliberately hidden. The diagram's `value=[-1 count, 1 count]` follows scikit-learn class order.

| Rank | Depth-5 feature | Importance | Unrestricted feature | Importance |
| ---: | --- | ---: | --- | ---: |
| 1 | `sslfinal_state` | 0.644120 | `sslfinal_state` | 0.488015 |
| 2 | `url_of_anchor` | 0.227321 | `url_of_anchor` | 0.174918 |
| 3 | `web_traffic` | 0.050573 | `web_traffic` | 0.046035 |
| 4 | `links_in_tags` | 0.038713 | `links_in_tags` | 0.043458 |
| 5 | `prefix_suffix` | 0.024990 | `having_sub_domain` | 0.031410 |

The depth-3 tree also ranks `sslfinal_state` first (0.714576) and `url_of_anchor` second (0.252186); full rankings for all three trees are in the JSON output. The two leading features agree broadly across methods, but `url_of_anchor` narrowly leads individual-feature information gain while `sslfinal_state` leads tree importance. Individual-feature gain evaluates a **multiway grouping of one feature at the root**; the tree uses binary thresholds such as `sslfinal_state <= 0.5` and measures gains after earlier splits. Correlated features can share or displace importance, and the tree's normalized weights sum over all used nodes. Neither ranking proves a causal role or a feature's future usefulness.

## Training and validation comparison

| Model | Train accuracy | Validation accuracy | Train phishing F1 | Validation phishing F1 |
| --- | ---: | ---: | ---: | ---: |
| Tree depth 5 | 0.9179 | 0.9386 | 0.9032 | 0.9228 |
| Tree unrestricted | 0.9892 | 0.9355 | 0.9879 | 0.9246 |
| KNN k=9, Hamming | 0.9660 | 0.9546 | 0.9615 | 0.9453 |

The unrestricted tree has a **5.36 percentage-point accuracy gap** and **6.33-point phishing-F1 gap** between training and validation. Its depth is 19 with 392 leaves, compared with depth 5 and 18 leaves for the constrained tree. This train/validation comparison suggests substantial overfitting by the unrestricted tree. Training scores still fall short of 1.0 in part because some identical feature vectors have conflicting labels; the data cannot be fit perfectly by a deterministic function of these 30 values. The depth-5 tree's validation score exceeds its training score, so this run does not show the same overfitting pattern; differing group composition and difficulty can produce that direction. KNN k=9 has smaller gaps (1.14 accuracy points, 1.62 F1 points) and remains the best of these three on validation F1. None of these gaps alone proves future generalization.

## Reproduction and limits

Run from the repository root:

```bash
.venv/bin/python -m scripts.analyze_features
.venv/bin/python -m pytest -q
git diff --check
```

The script fetches official UCI data through the existing loader and writes JSON and three PNGs under `results/checkpoint3a/`. Tests are offline and cover known entropy and information-gain values, feature-name alignment, deterministic analysis, and a guard that diagnostics never select test positions. The dataset contains repeated feature vectors and conflicting labels; the original group-safe split prevents exact cross-partition feature duplicates. The published label-mapping conflict remains unresolved. Before final test evaluation, review these findings, lock the selected model and configuration using validation evidence only, and state the target-label convention explicitly in the final report.
