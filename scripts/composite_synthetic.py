"""Make synthetic training images by pasting animal cutouts onto disaster backgrounds.

uv run python scripts/composite_synthetic.py --n 500
    --backgrounds data/synthetic/backgrounds/floodnet
    --background-license CDLA-Permissive-1.0 --cutout-license CC-BY-4.0
"""

import argparse
from pathlib import Path

from aerial_animal_rescue.classes import ClassRegistry
from aerial_animal_rescue.config import SYNTHETIC_DIR
from aerial_animal_rescue.metadata import SCENE_TYPES
from aerial_animal_rescue.synthetic import CompositeConfig, generate


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--n", type=int, default=100, help="number of images")
    p.add_argument("--backgrounds", type=Path, default=SYNTHETIC_DIR / "backgrounds")
    p.add_argument("--cutouts", type=Path, default=SYNTHETIC_DIR / "cutouts")
    p.add_argument("--out", type=Path, default=SYNTHETIC_DIR)
    p.add_argument("--background-license", required=True)
    p.add_argument("--cutout-license", required=True)
    p.add_argument("--scene-type", default="flood", choices=sorted(SCENE_TYPES))
    p.add_argument("--min-size", type=int, default=12, help="min object size in px")
    p.add_argument("--max-size", type=int, default=80, help="max object size in px")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    cfg = CompositeConfig(
        n_images=args.n, object_size_px=(args.min_size, args.max_size), seed=args.seed
    )
    records = generate(
        args.backgrounds,
        args.cutouts,
        args.out,
        ClassRegistry.from_yaml(),
        args.background_license,
        args.cutout_license,
        cfg,
        scene_type=args.scene_type,
    )
    print(f"wrote {len(records)} composites to {args.out}")


if __name__ == "__main__":
    main()
