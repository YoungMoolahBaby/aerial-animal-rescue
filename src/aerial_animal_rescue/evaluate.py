"""Evaluate a trained model with breakdowns by class, modality, synthetic/real and object size.

One prediction pass over the split, then every breakdown is computed from the
cached predictions with metrics.evaluate().
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from aerial_animal_rescue.metadata import read_metadata
from aerial_animal_rescue.metrics import ImageResult, evaluate, operating_point
from aerial_animal_rescue.yolo import label_path_for, read_labels, yolo_to_xyxy

# Object size buckets by box area in pixels^2. Finer than COCO at the small end,
# because aerial animals are mostly tiny.
SIZE_BUCKETS = {
    "tiny (<16px)": (0, 16**2),
    "small (16-32px)": (16**2, 32**2),
    "medium (32-96px)": (32**2, 96**2),
    "large (>96px)": (96**2, float("inf")),
}


def load_ground_truth(image_path: Path) -> np.ndarray:
    with Image.open(image_path) as im:
        w, h = im.size
    boxes = read_labels(label_path_for(image_path))
    return np.array([(b[0], *yolo_to_xyxy(b, w, h)) for b in boxes]).reshape(-1, 5)


def predict(weights: str, images: list[Path], imgsz: int, conf: float, device=None) -> list:
    from ultralytics import YOLO

    model = YOLO(weights)
    preds = []
    for path in tqdm(images, desc="predict"):
        r = model.predict(str(path), imgsz=imgsz, conf=conf, device=device, verbose=False)[0]
        b = r.boxes
        arr = np.column_stack(
            [b.cls.cpu().numpy(), b.xyxy.cpu().numpy(), b.conf.cpu().numpy()]
        ).reshape(-1, 6)
        preds.append(arr)
    return preds


def run_evaluation(
    weights: str,
    data_root: Path,
    class_names: list[str],
    split: str = "test",
    imgsz: int = 1280,
    conf: float = 0.001,
    device=None,
    operating_confs: tuple[float, ...] = (0.25, 0.5),
) -> dict:
    data_root = Path(data_root)
    meta = read_metadata(data_root / "metadata.csv")
    meta = meta[meta["split"] == split].reset_index(drop=True)
    if meta.empty:
        raise ValueError(f"no images in split {split!r}")
    paths = [data_root / p for p in meta["file_path"]]
    preds = predict(weights, paths, imgsz, conf, device)
    results = [ImageResult(load_ground_truth(p), pr) for p, pr in zip(paths, preds, strict=True)]
    n = len(class_names)

    def subset(mask: pd.Series) -> list[ImageResult]:
        return [r for r, keep in zip(results, mask, strict=True) if keep]

    report = {"weights": weights, "split": split, "images": len(results)}
    overall = evaluate(results, n)
    report["overall"] = {k: overall[k] for k in ("map50", "map50_95", "n_gt")}
    report["per_class"] = {
        class_names[c]: v for c, v in overall["per_class"].items() if v["n_gt"] > 0
    }
    report["by_modality"] = {
        m: _summary(evaluate(subset(meta["modality"] == m), n))
        for m in sorted(meta["modality"].dropna().unique())
    }
    report["by_synthetic"] = {
        ("synthetic" if s else "real"): _summary(evaluate(subset(meta["synthetic"] == s), n))
        for s in sorted(meta["synthetic"].unique())
    }
    report["by_size"] = {
        name: _summary(evaluate(results, n, area_range=rng)) for name, rng in SIZE_BUCKETS.items()
    }
    # at one detection threshold, as deployed: precision, recall, alarms on empty ground
    groups = {"all": pd.Series(True, index=meta.index)} | {
        m: meta["modality"] == m for m in sorted(meta["modality"].dropna().unique())
    }
    report["operating_points"] = {
        g: [operating_point(subset(mask), n, c) for c in operating_confs]
        for g, mask in groups.items()
    }
    report["by_sequence"] = by_sequence(results, meta, class_names, operating_confs[0])
    return report


def by_sequence(
    results: list[ImageResult], meta: pd.DataFrame, class_names: list[str], conf: float
) -> dict:
    """mAP, recall and false alarms per flight (source/sequence_id), largest first.

    A split holds only a handful of flights per source, so one flight can carry most of a
    class's boxes (e.g. one video holding half of val's people). `share` says how much of
    the class in this split the flight accounts for.
    """
    n = len(class_names)
    totals = evaluate(results, n)["per_class"]
    keys = meta["source"].astype(str) + "/" + meta["sequence_id"].astype(str)
    out = {}
    for key in keys.unique():
        rs = [r for r, k in zip(results, keys, strict=True) if k == key]
        res, op = evaluate(rs, n), operating_point(rs, n, conf)
        out[key] = {
            "images": len(rs),
            **_summary(res),
            "conf": conf,
            "recall": op["recall"],
            "false_alarms_per_empty_image": op["false_alarms_per_empty_image"],
            "empty_images": op["empty_images"],
            "per_class": {
                class_names[c]: {
                    "ap50": v["ap50"],
                    "n_gt": v["n_gt"],
                    "share": v["n_gt"] / totals[c]["n_gt"],
                }
                for c, v in res["per_class"].items()
                if v["n_gt"] > 0
            },
        }
    return dict(sorted(out.items(), key=lambda kv: (-kv[1]["n_gt"], kv[0])))


def _summary(res: dict) -> dict:
    return {k: res[k] for k in ("map50", "map50_95", "n_gt")}


def format_report(report: dict) -> str:
    def row(name, v):
        return (
            f"  {name:<22} mAP50={v['map50']:.3f}  mAP50-95={v['map50_95']:.3f}  boxes={v['n_gt']}"
        )

    lines = [f"split={report['split']} images={report['images']}", "overall:"]
    lines.append(row("all", report["overall"]))
    for section in ("per_class", "by_modality", "by_synthetic", "by_size"):
        lines.append(f"{section}:")
        lines += [
            row(k, {"map50": v["ap50"], "map50_95": v["ap50_95"], "n_gt": v["n_gt"]})
            if "ap50" in v
            else row(k, v)
            for k, v in report[section].items()
        ]
    lines.append("operating_points (IoU 0.5; alarms = detections on images with no objects):")
    for group, ops in report.get("operating_points", {}).items():
        lines += [
            f"  {group:<8} conf>={op['conf']:<5} precision={op['precision']:.3f}  "
            f"recall={op['recall']:.3f}  "
            f"alarms/empty image={op['false_alarms_per_empty_image']:.3f}  "
            f"empty images with an alarm={op['empty_images_with_false_alarm']:.1%} "
            f"(of {op['empty_images']})"
            for op in ops
        ]
    seqs = report.get("by_sequence", {})
    if seqs:
        conf = next(iter(seqs.values()))["conf"]
        lines.append(f"by_sequence (recall and alarms at conf>={conf}; % = share of the class):")
    for key, v in seqs.items():
        classes = ", ".join(
            f"{c} AP50={pc['ap50']:.3f} ({pc['n_gt']}, {pc['share']:.0%})"
            for c, pc in v["per_class"].items()
        )
        alarms = (
            f"alarms/empty image={v['false_alarms_per_empty_image']:.2f} (of {v['empty_images']})"
            if v["empty_images"]
            else "no empty images"
        )
        lines.append(
            f"  {key:<34} images={v['images']:<5} boxes={v['n_gt']:<5} "
            f"recall={v['recall']:.3f}  {alarms}  {classes}"
        )
    return "\n".join(lines)


def save_report(report: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")
