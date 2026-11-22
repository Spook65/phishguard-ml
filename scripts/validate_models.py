"""Run from repository root: python -m scripts.validate_models."""

import json

from phishguard.data import load_dataset
from phishguard.models import run_validation
from phishguard.split import split_dataset


def main():
    dataset = load_dataset()
    print(json.dumps(run_validation(dataset, split_dataset(dataset)), indent=2))


if __name__ == "__main__":
    main()
