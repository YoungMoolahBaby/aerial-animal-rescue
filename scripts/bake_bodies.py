"""Fetch the CC0 models listed in sim/bodies.csv and bake them for the storm sim.

    uv run python scripts/bake_bodies.py

Models live under data/external/ (`file` column; a few MB in all). Ones with a `url` are
downloaded if missing; the rest (people: Quaternius' Universal Base Characters and
Universal Animation Library, free "Standard" zips on itch.io, see `page`) are unpacked
there by hand. Baked bodies go to sim/assets/bodies/<kind>/<name>.body, which
`datagen --bodies assets/bodies` reads. See aerial_animal_rescue.bodies for the poses and
the file format.
"""

import argparse
import csv
import urllib.request
from pathlib import Path

import numpy as np

from aerial_animal_rescue.bodies import HEAD, LIMB, TORSO, bake, write_body
from aerial_animal_rescue.config import EXTERNAL_DIR

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--list", type=Path, default=ROOT / "sim" / "bodies.csv")
    p.add_argument("--models", type=Path, default=EXTERNAL_DIR)
    p.add_argument("--out", type=Path, default=ROOT / "sim" / "assets" / "bodies")
    args = p.parse_args()

    with args.list.open(newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        model = args.models / row["file"]
        if not model.exists():
            if not row["url"]:
                raise SystemExit(f"{model} is missing: get it from {row['page']}")
            print(f"downloading {row['name']} from {row['url']}")
            model.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(row["url"], model)
        anim = args.models / row["animations"] if row["animations"] else None
        if anim is not None and not anim.exists():
            raise SystemExit(f"{anim} is missing: see the page of {row['name']}, {row['page']}")
        cold = tuple(m for m in row["cold_materials"].split(";") if m)
        poses = bake(model, float(row["size_m"]), cold, anim, person=row["kind"] == "person")
        out = args.out / row["kind"] / f"{row['name']}.body"
        out.parent.mkdir(parents=True, exist_ok=True)
        write_body(out, poses)
        rest = np.vstack([v for v, _ in poses[0][1]])
        size = rest.max(0) - rest.min(0)
        tris = [len(poses[0][1][k][1]) for k in (TORSO, LIMB, HEAD)]
        print(
            f"{row['kind']:<7} {row['name']:<9} {len(poses):>2} poses  "
            f"{size[0]:.2f} wide x {size[1]:.2f} high x {size[2]:.2f} long m  "
            f"triangles torso/limb/head {tris[0]}/{tris[1]}/{tris[2]}"
        )


if __name__ == "__main__":
    main()
