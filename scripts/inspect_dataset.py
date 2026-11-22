"""Run from the repository root: python -m scripts.inspect_dataset."""

import json
import sys

from phishguard.data import inspect_dataset, load_dataset


def main():
    try:
        report = inspect_dataset(load_dataset())
    except (ConnectionError, OSError, ValueError) as exc:
        print(f"Dataset intake failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
