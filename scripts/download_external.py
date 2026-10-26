"""Download external datasets listed in configs/sources.yaml.

Manual sources print instructions instead of failing; anything over 1 GB (or of
unknown size) asks first.

    uv run python scripts/download_external.py --list
    uv run python scripts/download_external.py hit_uav birdsai
"""

import argparse
from pathlib import Path

from aerial_animal_rescue.config import EXTERNAL_DIR
from aerial_animal_rescue.sources import fetch_source, load_sources


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("names", nargs="*", help="source names (default: all)")
    p.add_argument("--list", action="store_true", help="show the registry and exit")
    p.add_argument("--out", type=Path, default=EXTERNAL_DIR)
    args = p.parse_args()

    sources = load_sources()
    if args.list:
        for s in sources.values():
            size = f"{s.size_gb} GB" if s.size_gb is not None else "? GB"
            print(
                f"{s.name:<16} {s.download:<9} {size:>8}  {s.license:<20} "
                f"redistribution={s.redistribution}  role={s.role}"
            )
        return
    unknown = set(args.names) - set(sources)
    if unknown:
        raise SystemExit(f"unknown sources {sorted(unknown)}; known: {sorted(sources)}")
    for name in args.names or sources:
        fetch_source(sources[name], args.out)


if __name__ == "__main__":
    main()
