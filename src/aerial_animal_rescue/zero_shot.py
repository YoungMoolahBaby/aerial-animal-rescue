"""Class-agnostic check of a model on data it wasn't trained on (sim/PLAN.md).

Does the model put a box of any class on each object, and how often does it fire on
images with nothing in them? Class-agnostic, because sim has classes (dog, horse,
sheep) that real data lacks; per size bucket, because sim and real differ most for
tiny objects. Run the same model on sim and on real val and compare: similar rates mean
the sim looks real to the model, many false alarms on sim clutter mean the clutter
looks like bodies.
"""

import numpy as np

from aerial_animal_rescue.evaluate import SIZE_BUCKETS
from aerial_animal_rescue.metrics import box_area, iou_matrix


def found(gt: np.ndarray, pred: np.ndarray, thr: float) -> np.ndarray:
    """For each ground-truth box (N, 4 xyxy): does any prediction (M, 4) reach IoU `thr`?"""
    if len(gt) == 0:
        return np.zeros(0, dtype=bool)
    if len(pred) == 0:
        return np.zeros(len(gt), dtype=bool)
    return iou_matrix(gt, pred).max(axis=1) >= thr


def size_bucket(area: float) -> str:
    return next(name for name, (lo, hi) in SIZE_BUCKETS.items() if lo <= area < hi)


def summarize(gts: list[np.ndarray], preds: list[np.ndarray], conf: float = 0.25) -> dict:
    """Found rates per size (IoU 0.5 and a loose 0.1) and alarms on empty images.

    `gts[i]` is (N, 5) class+xyxy, `preds[i]` is (M, 6) class+xyxy+score; predictions
    below `conf` are dropped.
    """
    rows = {name: [0, 0, 0] for name in SIZE_BUCKETS}  # boxes, found@0.5, found@0.1
    alarms, empty = 0, 0
    for gt, pred in zip(gts, preds, strict=True):
        p = pred[pred[:, 5] >= conf, 1:5] if len(pred) else np.zeros((0, 4))
        g = gt[:, 1:5]
        if len(g) == 0:
            empty += 1
            alarms += len(p)
            continue
        hit50, hit10 = found(g, p, 0.5), found(g, p, 0.1)
        for area, a, b in zip(box_area(g), hit50, hit10, strict=True):
            r = rows[size_bucket(float(area))]
            r[0] += 1
            r[1] += int(a)
            r[2] += int(b)
    by_size = {
        name: {"boxes": n, "found_iou50": f50 / n, "found_iou10": f10 / n}
        for name, (n, f50, f10) in rows.items()
        if n
    }
    return {
        "conf": conf,
        "by_size": by_size,
        "empty_images": empty,
        "alarms_per_empty_image": alarms / empty if empty else None,
    }
