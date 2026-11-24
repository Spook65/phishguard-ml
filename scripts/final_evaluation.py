"""Run once from repository root: python -m scripts.final_evaluation."""

import argparse
import json
from pathlib import Path

from phishguard.data import load_dataset, validate_dataset
from phishguard.final_evaluation import comparison, evaluate_models, preflight
from phishguard.final_protocol import LEADING_MODEL, make_final_models
from phishguard.split import split_dataset

OUTPUT = Path("results/checkpoint3c/final_evaluation.json")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Frozen UCI 327 final evaluation")
    parser.add_argument("--preflight-only", action="store_true", help="verify without test scoring")
    args = parser.parse_args(argv)
    if not args.preflight_only and OUTPUT.exists():
        raise FileExistsError(f"Final results already exist: {OUTPUT}; refusing to rerun or overwrite")

    dataset = load_dataset(check_target_values=False)
    split = split_dataset(dataset, check_target_values=False)
    models = make_final_models()
    checks = preflight(dataset, split, models)
    print("Preflight passed: source, schema, split, groups, models, and metric labels.", flush=True)
    if args.preflight_only:
        return 0

    # No held-out label selection or model scoring occurs before preflight.
    validate_dataset(dataset.features, dataset.target.to_frame())
    x_train, y_train = dataset.features.iloc[split.train], dataset.target.iloc[split.train]
    x_test, y_test = dataset.features.iloc[split.test], dataset.target.iloc[split.test]
    results = evaluate_models(x_train, y_train, x_test, y_test, models)
    report = {"preflight": checks, "leading_model_selected_on_validation": LEADING_MODEL,
              "results": results, "validation_comparison": comparison(results)}
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("x") as file:
        json.dump(report, file, indent=2)
        file.write("\n")
    print(f"Saved one-time final results to {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
