"""Fine-tune a YOLO baseline on data/processed; runs are saved under runs/train/.

uv run python scripts/train_baseline.py --model yolo11s.pt --epochs 100 --imgsz 1280 --device 0
"""

import argparse
from pathlib import Path

from aerial_animal_rescue.classes import ClassRegistry
from aerial_animal_rescue.config import PROCESSED_DIR
from aerial_animal_rescue.train import train


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--data", type=Path, default=PROCESSED_DIR)
    p.add_argument("--model", default="yolo11n.pt", help="pretrained weights or a model .yaml")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--device", default=None, help="e.g. 0 for the first GPU, or cpu")
    p.add_argument(
        "--workers",
        type=int,
        default=4,
        help="data loader processes; Ultralytics' default of 8 ran out of Windows page file "
        "during validation on a 32 GB machine",
    )
    p.add_argument("--name", default="baseline")
    args = p.parse_args()

    train(
        args.data,
        list(ClassRegistry.from_yaml().names),
        model=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        name=args.name,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
