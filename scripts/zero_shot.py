"""Class-agnostic check of trained weights on pools or splits: found rates per size, alarms.

    uv run python scripts/zero_shot.py runs/train/hituav/weights/best.pt \
        data/interim/sim_p1_w10 data/processed --split val --modality thermal --sample 800

Each folder is a pool or a processed set (anything with metadata.csv); `--split` and
`--modality` filter rows that have those columns set. Prints, per folder, the share of
objects with a box of any class on them (IoU 0.5 and a loose 0.1) per size, and false
alarms per empty image. `--alarms` saves the most confident false alarms as crops, to
see what the model takes for a body. See aerial_animal_rescue.zero_shot and sim/PLAN.md.
Never point it at the test split while iterating.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from aerial_animal_rescue.classes import ClassRegistry
from aerial_animal_rescue.evaluate import load_ground_truth, predict
from aerial_animal_rescue.metadata import read_metadata
from aerial_animal_rescue.zero_shot import summarize


def alarm_sheet(images: list[Path], gts, preds, conf: float, out: Path, n: int = 20) -> None:
    alarms = [
        (float(p[5]), path, p[1:5])
        for path, gt, pred in zip(images, gts, preds, strict=True)
        if len(gt) == 0
        for p in pred
        if p[5] >= conf
    ]
    alarms.sort(key=lambda a: -a[0])
    tiles = []
    for score, path, b in alarms[:n]:
        im = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        cx, cy = int((b[0] + b[2]) / 2), int((b[1] + b[3]) / 2)
        pad = cv2.copyMakeBorder(im, 32, 32, 32, 32, cv2.BORDER_REFLECT)
        crop = cv2.resize(
            pad[cy : cy + 64, cx : cx + 64], (128, 128), interpolation=cv2.INTER_NEAREST
        )
        crop = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
        cv2.putText(crop, f"{score:.2f}", (2, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        tiles.append(crop)
    if not tiles:
        return
    while len(tiles) % 10:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[i : i + 10]) for i in range(0, len(tiles), 10)]
    cv2.imwrite(str(out), np.vstack(rows))
    print(f"saved {out}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("weights")
    p.add_argument("folders", nargs="+", type=Path, help="pools or processed sets")
    p.add_argument("--split", help="keep rows of this split (if the column is set)")
    p.add_argument("--modality", help="keep rows of this modality")
    p.add_argument("--sample", type=int, help="random images per folder (default: all)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--device", default=None)
    p.add_argument("--alarms", type=Path, help="folder for false-alarm crop sheets")
    p.add_argument("--by-class", action="store_true", help="also print found rates per class")
    args = p.parse_args()
    if args.split == "test":
        raise SystemExit("the test set is held back until iterating is done")

    for folder in args.folders:
        m = read_metadata(folder / "metadata.csv")
        if args.split and m["split"].notna().any():
            m = m[m["split"] == args.split]
        if args.modality:
            m = m[m["modality"] == args.modality]
        if args.sample and len(m) > args.sample:
            m = m.sample(args.sample, random_state=args.seed)
        images = [folder / f for f in m["file_path"]]
        if not images:
            print(f"\n== {folder}: no images match --split/--modality, skipped")
            continue
        gts = [load_ground_truth(i) for i in images]
        preds = predict(args.weights, images, args.imgsz, args.conf, args.device)
        res = summarize(gts, preds, args.conf)
        table = pd.DataFrame(res["by_size"]).T
        print(f"\n== {folder}: {len(images)} images")
        print(table.round(3).to_string())
        if args.by_class:
            names = ClassRegistry.from_yaml().names
            found_by = {
                names[c]: {b: f"{v['found_iou50']:.2f} ({v['boxes']})" for b, v in sizes.items()}
                for c, sizes in res["by_class"].items()
            }
            print("found at IoU 0.5 per class (boxes):")
            print(pd.DataFrame(found_by).T.fillna("-").to_string())
        alarms = res["alarms_per_empty_image"]
        print(
            f"alarms per empty image: {alarms:.2f} (of {res['empty_images']})"
            if alarms is not None
            else "no empty images"
        )
        if args.alarms:
            args.alarms.mkdir(parents=True, exist_ok=True)
            out = args.alarms / f"{folder.name}_alarms.png"
            alarm_sheet(images, gts, preds, args.conf, out)


if __name__ == "__main__":
    main()
