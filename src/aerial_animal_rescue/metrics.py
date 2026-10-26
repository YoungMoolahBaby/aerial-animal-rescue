"""Small, dependency-free COCO-style average precision.

We compute AP ourselves (instead of relying only on Ultralytics' validator)
so one prediction pass can be sliced any way: per class, modality,
synthetic/real, object size. Matching follows COCO: greedy by score, and for
a size range, ground truth outside the range is "ignored" (matching it is
neither a hit nor a false positive) and unmatched predictions outside the
range are ignored too. AP uses 101-point interpolated precision.
"""

from dataclasses import dataclass

import numpy as np

IOU_THRESHOLDS = np.linspace(0.5, 0.95, 10)


@dataclass
class ImageResult:
    gt: np.ndarray  # (N, 5): class, x1, y1, x2, y2 (pixels)
    pred: np.ndarray  # (M, 6): class, x1, y1, x2, y2, score


def box_area(b: np.ndarray) -> np.ndarray:
    return (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU between every box in `a` (M, 4) and `b` (N, 4), both xyxy."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    return inter / (box_area(a)[:, None] + box_area(b)[None, :] - inter + 1e-12)


def _match(gt, pred, gt_ignore, pred_out_of_range, thr):
    """Greedy COCO matching for one image & class. Returns (tp, ignore) per prediction."""
    ious = iou_matrix(pred[:, 1:5], gt[:, 1:5])
    matched = np.zeros(len(gt), dtype=bool)
    tp = np.zeros(len(pred), dtype=bool)
    ignore = np.zeros(len(pred), dtype=bool)
    for i in range(len(pred)):  # pred is sorted by score, high first
        best, best_iou = -1, thr
        for j in np.argsort(gt_ignore, kind="stable"):  # non-ignored GT first
            if matched[j]:
                continue
            if best >= 0 and not gt_ignore[best] and gt_ignore[j]:
                break  # already have a real match; don't switch to an ignored GT
            if ious[i, j] >= best_iou:
                best, best_iou = j, ious[i, j]
        if best >= 0:
            matched[best] = True
            tp[i] = not gt_ignore[best]
            ignore[i] = gt_ignore[best]
        else:
            ignore[i] = pred_out_of_range[i]
    return tp, ignore


def average_precision(tp: np.ndarray, scores: np.ndarray, n_gt: int) -> float:
    """101-point interpolated AP from per-detection hit flags."""
    if n_gt == 0:
        return float("nan")
    if len(tp) == 0:
        return 0.0
    order = np.argsort(-scores, kind="stable")
    tp = tp[order].astype(float)
    tps, fps = np.cumsum(tp), np.cumsum(1 - tp)
    recall = tps / n_gt
    precision = tps / np.maximum(tps + fps, 1e-12)
    precision = np.maximum.accumulate(precision[::-1])[::-1]  # make monotonic
    points = np.linspace(0, 1, 101)
    idx = np.searchsorted(recall, points, side="left")
    return float(np.mean([precision[k] if k < len(precision) else 0.0 for k in idx]))


def evaluate(
    results: list[ImageResult],
    num_classes: int,
    area_range: tuple[float, float] | None = None,
) -> dict:
    """AP per class at each IoU threshold; returns mAP50, mAP50-95 and per-class values.

    `area_range` (in pixels^2, e.g. (0, 16**2)) restricts evaluation to objects of that size.
    Classes with no ground truth in `results` are left out of the means.
    """
    per_class = {}
    for c in range(num_classes):
        aps, n_gt = [], 0
        hits = [[] for _ in IOU_THRESHOLDS]
        for r in results:
            gt = r.gt[r.gt[:, 0] == c] if len(r.gt) else np.zeros((0, 5))
            pred = r.pred[r.pred[:, 0] == c] if len(r.pred) else np.zeros((0, 6))
            pred = pred[np.argsort(-pred[:, 5], kind="stable")]
            if area_range is None:
                gt_ignore = np.zeros(len(gt), dtype=bool)
                pred_oor = np.zeros(len(pred), dtype=bool)
            else:
                lo, hi = area_range
                ga, pa = box_area(gt[:, 1:5]), box_area(pred[:, 1:5])
                gt_ignore = (ga < lo) | (ga >= hi)
                pred_oor = (pa < lo) | (pa >= hi)
            n_gt += int((~gt_ignore).sum())
            for t, thr in enumerate(IOU_THRESHOLDS):
                tp, ign = _match(gt, pred, gt_ignore, pred_oor, thr)
                hits[t].append((tp[~ign], pred[~ign, 5]))
        for t in range(len(IOU_THRESHOLDS)):
            tp = np.concatenate([h[0] for h in hits[t]]) if hits[t] else np.zeros(0, bool)
            sc = np.concatenate([h[1] for h in hits[t]]) if hits[t] else np.zeros(0)
            aps.append(average_precision(tp, sc, n_gt))
        per_class[c] = {"n_gt": n_gt, "ap50": aps[0], "ap50_95": float(np.mean(aps))}
    valid = [v for v in per_class.values() if v["n_gt"] > 0]
    return {
        "map50": float(np.mean([v["ap50"] for v in valid])) if valid else float("nan"),
        "map50_95": float(np.mean([v["ap50_95"] for v in valid])) if valid else float("nan"),
        "n_gt": sum(v["n_gt"] for v in valid),
        "per_class": per_class,
    }
