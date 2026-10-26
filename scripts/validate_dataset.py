"""Validate data/processed and print class counts per split and per modality.

    uv run python scripts/validate_dataset.py [--root data/processed]

Exits with status 1 if any error is found.
"""

import argparse
import sys
from pathlib import Path

from aerial_animal_rescue.classes import ClassRegistry
from aerial_animal_rescue.config import PROCESSED_DIR
from aerial_animal_rescue.validate import summary_tables, validate_dataset


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--root", type=Path, default=PROCESSED_DIR)
    p.add_argument("--max-errors", type=int, default=50, help="how many errors to print")
    p.add_argument("--skip-gps", action="store_true", help="skip the (slower) GPS check")
    args = p.parse_args()

    rep = validate_dataset(args.root, list(ClassRegistry.from_yaml().names), not args.skip_gps)
    print(summary_tables(rep))
    for w in rep.warnings:
        print("warning:", w)
    if rep.ok:
        print("\nOK: no problems found")
        return
    print(f"\n{len(rep.errors)} errors:")
    for e in rep.errors[: args.max_errors]:
        print("  " + e)
    if len(rep.errors) > args.max_errors:
        print(f"  ... and {len(rep.errors) - args.max_errors} more")
    sys.exit(1)


if __name__ == "__main__":
    main()
