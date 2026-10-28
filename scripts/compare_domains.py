"""Compare pools (sim and real) on the same image statistics, side by side.

    uv run python scripts/compare_domains.py data/interim/sim_phase0 data/interim/birdsai
    uv run python scripts/compare_domains.py data/interim/birdsai --by sequence_id --sample 2000

Per group: frame size, share of empty images, sensor noise (median neighbour difference),
brightness spread (1st/99th percentile); per class: box size (median side, share <16 px
and 16-32 px) and how much objects stand out from their surroundings (peak minus ring
mean, and as a multiple of the ring's spread). Grey levels 0-255. See sim/PLAN.md.
"""

import argparse
from pathlib import Path

import pandas as pd

from aerial_animal_rescue.classes import ClassRegistry
from aerial_animal_rescue.domain_stats import measure_pool, summarize


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("pools", nargs="+", type=Path, help="pool folders with metadata.csv")
    p.add_argument("--by", default="source", choices=["source", "sequence_id"])
    p.add_argument("--sample", type=int, help="random images per pool (default: all)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, help="also save the table as CSV")
    args = p.parse_args()

    names = list(ClassRegistry.from_yaml().names)
    tables = []
    for pool in args.pools:
        images, boxes = measure_pool(pool, names, args.sample, args.seed)
        tables.append(summarize(images, boxes, args.by))
    table = pd.concat(tables, ignore_index=True)
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(table.round(3).to_string(index=False))
    if args.out:
        table.to_csv(args.out, index=False)
        print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
