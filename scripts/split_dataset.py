"""Merge pools, split by flight/sequence (no leakage, real-only test), write data/processed.

uv run python scripts/split_dataset.py           # all data/interim/* pools + data/synthetic
uv run python scripts/split_dataset.py --pools data/interim/hit_uav data/interim/own --overwrite
"""

import argparse
from pathlib import Path

from aerial_animal_rescue.classes import ClassRegistry
from aerial_animal_rescue.config import INTERIM_DIR, PROCESSED_DIR, SYNTHETIC_DIR
from aerial_animal_rescue.split import build_processed


def default_pools() -> list[Path]:
    pools = sorted(p for p in INTERIM_DIR.iterdir() if (p / "metadata.csv").exists())
    if (SYNTHETIC_DIR / "metadata.csv").exists():
        pools.append(SYNTHETIC_DIR)
    return pools


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--pools", nargs="*", type=Path, help="pool folders (with metadata.csv)")
    p.add_argument("--out", type=Path, default=PROCESSED_DIR)
    p.add_argument(
        "--ratios", nargs=3, type=float, default=(0.7, 0.15, 0.15), metavar=("TRAIN", "VAL", "TEST")
    )
    p.add_argument("--group-by", choices=["sequence", "source"], default="sequence")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--overwrite", action="store_true", help="replace an existing processed set")
    p.add_argument(
        "--train-empty-frac",
        type=float,
        default=0.3,
        help="max share of empty images in train (val/test keep all); -1 keeps all. "
        "Empty frames teach the model not to fire on background, i.e. fewer false positives",
    )
    args = p.parse_args()

    pools = args.pools or default_pools()
    if not pools:
        raise SystemExit("no pools found: run convert_external / ingest_raw / composite first")
    print("pools:", ", ".join(str(x) for x in pools))
    build_processed(
        pools,
        args.out,
        list(ClassRegistry.from_yaml().names),
        tuple(args.ratios),
        args.seed,
        args.group_by,
        args.overwrite,
        None if args.train_empty_frac < 0 else args.train_empty_frac,
    )


if __name__ == "__main__":
    main()
