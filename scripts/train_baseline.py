"""Fine-tune a YOLO baseline on data/processed; runs are saved under runs/train/.

uv run python scripts/train_baseline.py --model yolo11s.pt --imgsz 1280 --device 0
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
    p.add_argument(
        "--epochs",
        type=int,
        default=60,
        help="the first 100-epoch baseline peaked on val around epoch 50, then overfit",
    )
    p.add_argument(
        "--patience",
        type=int,
        default=20,
        help="stop after this many epochs without a new best on val (val swings a lot "
        "from epoch to epoch, so not much lower)",
    )
    p.add_argument(
        "--scale",
        type=float,
        default=0.75,
        help="random zoom range +/-; Ultralytics' default is 0.5. Wider zoom-out shows the "
        "model more tiny animals, which are most of what a search drone sees",
    )
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
        patience=args.patience,
        scale=args.scale,
    )


if __name__ == "__main__":
    main()
