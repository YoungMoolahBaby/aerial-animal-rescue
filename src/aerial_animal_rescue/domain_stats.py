"""Image statistics that compare sim and real data the same way (sim/PLAN.md, phase 0).

Per image: sensor noise (median difference between horizontal neighbours), brightness
spread (1st and 99th percentile) and whether it is empty. Per box: size in pixels and how
much the object stands out from its surroundings (brightest pixel in the box minus the
mean of a ring around it, also as a multiple of the ring's spread). All on the grey image,
0-255. These are the numbers the tiny-person look used, so a sim pool can be tuned until
its spread covers the real flights.
"""

import random
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from aerial_animal_rescue.metadata import read_metadata
from aerial_animal_rescue.yolo import label_path_for, read_labels, yolo_to_xyxy


def frame_noise(gray: np.ndarray) -> float:
    """Median absolute difference between horizontal neighbours."""
    return float(np.median(np.abs(np.diff(gray.astype(float), axis=1))))


def object_contrast(
    gray: np.ndarray, box: tuple[float, float, float, float]
) -> tuple[float, float]:
    """(peak minus ring mean, that difference / ring std) for an (x1, y1, x2, y2) box.

    The ring is the box grown by its longer side on every edge, minus the box.
    """
    a = gray.astype(float)
    x1, y1, x2, y2 = (int(round(v)) for v in box)
    x2, y2 = max(x2, x1 + 1), max(y2, y1 + 1)
    inner = a[y1:y2, x1:x2]
    pad = max(x2 - x1, y2 - y1)
    outer = a[max(y1 - pad, 0) : y2 + pad, max(x1 - pad, 0) : x2 + pad]
    n_ring = outer.size - inner.size
    if inner.size == 0 or n_ring <= 0:
        return float("nan"), float("nan")
    ring_mean = (outer.sum() - inner.sum()) / n_ring
    ring_var = ((outer**2).sum() - (inner**2).sum()) / n_ring - ring_mean**2
    diff = inner.max() - ring_mean
    return float(diff), float(diff / (np.sqrt(max(ring_var, 0.0)) + 1e-6))


def measure_pool(
    pool: Path, class_names: list[str], sample: int | None = None, seed: int = 0
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(one row per image, one row per box) for a pool folder with metadata.csv."""
    pool = Path(pool)
    meta = read_metadata(pool / "metadata.csv")
    rows = list(meta.itertuples())
    if sample is not None and sample < len(rows):
        rows = random.Random(seed).sample(rows, sample)
    images, boxes = [], []
    for r in rows:
        path = pool / r.file_path
        labels_path = label_path_for(path)
        if not labels_path.exists():
            continue
        with Image.open(path) as im:
            gray = np.asarray(im.convert("L"))
        h, w = gray.shape
        labels = read_labels(labels_path)
        images.append(
            {
                "source": r.source,
                "sequence_id": r.sequence_id,
                "frame": f"{w}x{h}",
                "empty": not labels,
                "noise": frame_noise(gray),
                "p1": float(np.percentile(gray, 1)),
                "p99": float(np.percentile(gray, 99)),
            }
        )
        for b in labels:
            x1, y1, x2, y2 = yolo_to_xyxy(b, w, h)
            contrast, z = object_contrast(gray, (x1, y1, x2, y2))
            boxes.append(
                {
                    "source": r.source,
                    "sequence_id": r.sequence_id,
                    "class": class_names[int(b[0])],
                    "side": float(np.sqrt(max(x2 - x1, 0) * max(y2 - y1, 0))),
                    "contrast": contrast,
                    "contrast_z": z,
                }
            )
    return pd.DataFrame(images), pd.DataFrame(boxes)


def summarize(images: pd.DataFrame, boxes: pd.DataFrame, by: str = "source") -> pd.DataFrame:
    """One row per group (and class): the statistics table of sim/PLAN.md."""
    img = images.groupby(by).agg(
        images=("empty", "size"),
        empty=("empty", "mean"),
        frame=("frame", lambda s: s.mode().iat[0]),  # most common size
        noise=("noise", "median"),
        p1=("p1", "median"),
        p99=("p99", "median"),
    )
    if boxes.empty:
        return img.reset_index()
    box = boxes.groupby([by, "class"]).agg(
        boxes=("side", "size"),
        side=("side", "median"),
        lt16=("side", lambda s: (s < 16).mean()),
        s16_32=("side", lambda s: ((s >= 16) & (s < 32)).mean()),
        contrast=("contrast", "median"),
        contrast_z=("contrast_z", "median"),
    )
    return img.join(box.reset_index(level="class"), how="left").reset_index()
