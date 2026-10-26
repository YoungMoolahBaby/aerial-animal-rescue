"""Evaluate trained weights: mAP overall, per class, per modality, synthetic vs real, per size.

    uv run python scripts/evaluate.py runs/train/baseline/weights/best.pt --split test

The test split is real-only by design, so its synthetic-vs-real table has only
a "real" row. Use --split train to see how the model does on synthetic data.
"""

import argparse
from pathlib import Path

from aerial_animal_rescue.classes import ClassRegistry
from aerial_animal_rescue.config import PROCESSED_DIR, RUNS_DIR
from aerial_animal_rescue.evaluate import format_report, run_evaluation, save_report


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("weights")
    p.add_argument("--data", type=Path, default=PROCESSED_DIR)
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--device", default=None)
    p.add_argument("--out", type=Path, help="JSON report path (default runs/eval/<split>.json)")
    args = p.parse_args()

    report = run_evaluation(
        args.weights,
        args.data,
        list(ClassRegistry.from_yaml().names),
        split=args.split,
        imgsz=args.imgsz,
        device=args.device,
    )
    print(format_report(report))
    out = args.out or RUNS_DIR / "eval" / f"{args.split}.json"
    save_report(report, out)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
