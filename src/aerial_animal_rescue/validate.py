"""Dataset validator for a split YOLO dataset (data/processed/ layout).

Checks: every image has a label file, labels parse, class ids are valid,
boxes are inside the image, no duplicate images (by content hash) or
duplicate boxes, metadata covers every image, no sequence leaks across
splits, no synthetic images in test, no GPS left in any image.
"""

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from aerial_animal_rescue.config import IMAGE_EXTS
from aerial_animal_rescue.metadata import ImageRecord, read_metadata, validate_record
from aerial_animal_rescue.privacy import has_gps
from aerial_animal_rescue.split import group_keys
from aerial_animal_rescue.yolo import label_path_for, read_labels

EPS = 1e-6  # tolerance for float rounding in label files


@dataclass
class Report:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    boxes: pd.DataFrame | None = None  # one row per box: split, modality, synthetic, class
    images: pd.DataFrame | None = None  # one row per image: split, modality, synthetic

    @property
    def ok(self) -> bool:
        return not self.errors


def check_boxes(boxes, n_classes: int, where: str) -> list[str]:
    errs = []
    seen = set()
    for i, (c, cx, cy, w, h) in enumerate(boxes, 1):
        if not 0 <= c < n_classes:
            errs.append(f"{where} box {i}: class id {c} not in 0..{n_classes - 1}")
        if w <= 0 or h <= 0:
            errs.append(f"{where} box {i}: non-positive size")
        if cx - w / 2 < -EPS or cy - h / 2 < -EPS or cx + w / 2 > 1 + EPS or cy + h / 2 > 1 + EPS:
            errs.append(f"{where} box {i}: out of image bounds")
        key = (c, round(cx, 4), round(cy, 4), round(w, 4), round(h, 4))
        if key in seen:
            errs.append(f"{where} box {i}: duplicate box")
        seen.add(key)
    return errs


def validate_dataset(root: str | Path, class_names: list[str], check_gps: bool = True) -> Report:
    root = Path(root)
    rep = Report()
    images = sorted(p for p in (root / "images").rglob("*") if p.suffix.lower() in IMAGE_EXTS)
    if not images:
        rep.errors.append(f"no images under {root / 'images'}")
        return rep

    meta_path = root / "metadata.csv"
    meta = read_metadata(meta_path) if meta_path.exists() else None
    if meta is None:
        rep.errors.append(f"missing {meta_path}")
    meta_by_path = (
        {}
        if meta is None
        else {fp: r for fp, r in zip(meta["file_path"], meta.itertuples(), strict=True)}
    )

    hashes: dict[str, Path] = {}
    box_rows, image_rows = [], []
    for img in images:
        rel = img.relative_to(root).as_posix()
        split = img.relative_to(root / "images").parts[0]
        label = label_path_for(img)
        if not label.exists():
            rep.errors.append(f"{rel}: missing label file")
            boxes = []
        else:
            try:
                boxes = read_labels(label)
            except ValueError as e:
                rep.errors.append(str(e))
                boxes = []
            rep.errors += check_boxes(boxes, len(class_names), rel)

        digest = hashlib.md5(img.read_bytes()).hexdigest()
        if digest in hashes:
            rep.errors.append(f"{rel}: duplicate of {hashes[digest].relative_to(root).as_posix()}")
        hashes.setdefault(digest, img)

        if check_gps and has_gps(img):
            rep.errors.append(f"{rel}: still contains GPS metadata")

        m = meta_by_path.get(rel)
        if meta is not None and m is None:
            rep.errors.append(f"{rel}: no metadata row")
        if m is not None and m.split != split:
            rep.errors.append(f"{rel}: metadata split {m.split!r} but stored under {split!r}")
        modality = m.modality if m is not None else "unknown"
        synthetic = bool(m.synthetic) if m is not None else False
        image_rows.append((split, modality, synthetic))
        for b in boxes:
            name = class_names[b[0]] if 0 <= b[0] < len(class_names) else f"invalid:{b[0]}"
            box_rows.append((split, modality, synthetic, name))

    if meta is not None:
        for r in meta.itertuples(index=False):
            rec = ImageRecord(**{k: (None if pd.isna(v) else v) for k, v in r._asdict().items()})
            rep.errors += validate_record(rec)
        on_disk = {p.relative_to(root).as_posix() for p in images}
        for fp in set(meta["file_path"]) - on_disk:
            rep.errors.append(f"metadata row without image: {fp}")
        if meta["image_id"].duplicated().any():
            rep.errors.append("duplicate image_id values in metadata")
        keys = group_keys(meta)
        leaks = meta.assign(_g=keys).groupby("_g")["split"].nunique()
        for g in leaks[leaks > 1].index:
            rep.errors.append(f"group {g!r} appears in more than one split (leakage)")
        if (meta["synthetic"] & (meta["split"] == "test")).any():
            rep.errors.append("synthetic images found in the test split")

    rep.boxes = pd.DataFrame(box_rows, columns=["split", "modality", "synthetic", "class"])
    rep.images = pd.DataFrame(image_rows, columns=["split", "modality", "synthetic"])
    return rep


def summary_tables(rep: Report) -> str:
    """Image counts, and box counts per class by split and by modality."""
    parts = []
    if rep.images is not None and not rep.images.empty:
        table = pd.crosstab(
            [rep.images["split"], rep.images["synthetic"]], rep.images["modality"], margins=True
        )
        parts.append("images per split/synthetic x modality:\n" + table.to_string())
    if rep.boxes is not None and not rep.boxes.empty:
        for col in ("split", "modality"):
            table = pd.crosstab(rep.boxes["class"], rep.boxes[col], margins=True)
            parts.append(f"boxes per class x {col}:\n" + table.to_string())
    return "\n\n".join(parts) or "(no data)"
