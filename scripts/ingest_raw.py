"""Import our own drone footage: extract frames, strip GPS, write metadata.

Put each flight in its own folder under data/raw/, then e.g.:

    uv run python scripts/ingest_raw.py data/raw/2026-11-02_river --modality thermal
        --interval 2 --scene-type flood --weather rain --altitude 40

Output goes to data/interim/own/ (unlabeled until annotated).
"""

import argparse
from pathlib import Path

from aerial_animal_rescue.config import INTERIM_DIR
from aerial_animal_rescue.ingest import ingest
from aerial_animal_rescue.metadata import MODALITIES, SCENE_TYPES, WEATHER


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("input_dir", type=Path)
    p.add_argument("--modality", required=True, choices=sorted(MODALITIES))
    p.add_argument("--interval", type=float, default=2.0, help="seconds between video frames")
    p.add_argument("--scene-type", default="other", choices=sorted(SCENE_TYPES))
    p.add_argument("--weather", default="unknown", choices=sorted(WEATHER))
    p.add_argument("--altitude", type=float, default=None, help="flight altitude in metres")
    p.add_argument("--flight-id", default=None, help="one sequence id for everything ingested")
    p.add_argument("--license", default="CC-BY-4.0")
    p.add_argument("--out", type=Path, default=INTERIM_DIR / "own")
    args = p.parse_args()

    records = ingest(
        args.input_dir,
        args.out,
        modality=args.modality,
        interval_s=args.interval,
        source_license=args.license,
        sequence_id=args.flight_id,
        scene_type=args.scene_type,
        weather=args.weather,
        altitude_m=args.altitude,
    )
    print(f"ingested {len(records)} images into {args.out} (GPS stripped)")
    print(
        "next: annotate them (YOLO export into labels/), review for faces and consent, "
        "then set privacy_reviewed=True in metadata.csv"
    )


if __name__ == "__main__":
    main()
