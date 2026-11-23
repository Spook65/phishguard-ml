"""Run from repository root: python -m scripts.analyze_features."""

import json
import os
from pathlib import Path

# Keep Matplotlib's font/config cache inside the repository.
_cache = Path("results/generated/matplotlib-cache").resolve()
_cache.mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(_cache)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

from phishguard.analysis import analyze_training_validation
from phishguard.data import FEATURE_NAMES, load_dataset
from phishguard.split import split_dataset

OUTPUT = Path("results/checkpoint3a")


def bar_chart(rows, value_key, title, xlabel, path):
    top = [row for row in rows if row[value_key] > 0][:10][::-1]
    fig, ax = plt.subplots(figsize=(9, 5.8))
    ax.barh([row["feature"] for row in top], [row[value_key] for row in top], color="#2876a8")
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.25)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    dataset = load_dataset()
    report, tree = analyze_training_validation(dataset, split_dataset(dataset))
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "analysis.json").write_text(json.dumps(report, indent=2) + "\n")
    bar_chart(report["information_gain"], "information_gain_bits", "Top 10 single-feature information gains",
              "Information gain (bits)", OUTPUT / "information_gain_top10.png")
    bar_chart(report["tree_importances"]["depth_5"], "importance", "Depth-5 tree feature importance",
              "Normalized impurity reduction", OUTPUT / "tree_depth5_importance.png")
    fig, ax = plt.subplots(figsize=(15, 7))
    plot_tree(tree, feature_names=list(FEATURE_NAMES), class_names=["-1", "1"],
              max_depth=2, filled=True, rounded=True, fontsize=8, ax=ax)
    ax.set_title("Depth-5 entropy tree: first three levels (deeper nodes hidden)")
    fig.tight_layout()
    fig.savefig(OUTPUT / "tree_first_three_levels.png", dpi=160)
    plt.close(fig)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
