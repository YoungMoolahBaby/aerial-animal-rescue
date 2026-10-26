"""Per-image metadata: one row per image, stored as CSV (or Parquet by file suffix).

Each data "pool" (data/interim/<source>/, data/synthetic/, data/processed/)
has a `metadata.csv` next to its `images/` and `labels/` folders. `file_path`
is relative to that pool root.

Precise GPS coordinates are deliberately NOT part of the schema (privacy).
"""

from dataclasses import asdict, dataclass, fields
from pathlib import Path

import pandas as pd

MODALITIES = {"rgb", "thermal"}
WEATHER = {"clear", "rain", "fog", "unknown"}
SCENE_TYPES = {"flood", "debris_flow", "pasture", "urban", "other"}
SPLITS = {"train", "val", "test"}


@dataclass
class ImageRecord:
    image_id: str  # unique across the whole dataset, also the file stem in data/processed
    file_path: str  # relative to the pool root, e.g. "images/abc.jpg"
    source: str  # key in configs/sources.yaml, "own", or "synthetic"
    source_license: str  # SPDX-style id, e.g. "CC-BY-4.0"; composites list every input
    modality: str = "rgb"  # rgb | thermal
    synthetic: bool = False
    altitude_m: float | None = None
    weather: str = "unknown"  # clear | rain | fog | unknown
    scene_type: str = "other"  # flood | debris_flow | pasture | urban | other
    sequence_id: str | None = None  # drone flight or video sequence; the split grouping key
    capture_date: str | None = None  # YYYY-MM-DD
    split: str | None = None  # train | val | test, set by split_dataset
    # Extension beyond the core schema: our own footage must be checked by a
    # human (no identifiable faces, consent for property/animals) before upload.
    privacy_reviewed: bool = False


COLUMNS = [f.name for f in fields(ImageRecord)]


def validate_record(r: ImageRecord) -> list[str]:
    """Return a list of problems with one record (empty if fine)."""
    problems = []
    if not r.image_id:
        problems.append("empty image_id")
    if not r.source_license or r.source_license.strip().upper() in {"TODO", "UNKNOWN"}:
        problems.append(f"{r.image_id}: source_license not set")
    if r.modality not in MODALITIES:
        problems.append(f"{r.image_id}: modality {r.modality!r} not in {sorted(MODALITIES)}")
    if r.weather not in WEATHER:
        problems.append(f"{r.image_id}: weather {r.weather!r} not in {sorted(WEATHER)}")
    if r.scene_type not in SCENE_TYPES:
        problems.append(f"{r.image_id}: scene_type {r.scene_type!r} not in {sorted(SCENE_TYPES)}")
    if r.split is not None and r.split not in SPLITS:
        problems.append(f"{r.image_id}: split {r.split!r} not in {sorted(SPLITS)}")
    return problems


def to_frame(records: list[ImageRecord]) -> pd.DataFrame:
    return pd.DataFrame([asdict(r) for r in records], columns=COLUMNS)


def write_metadata(df: pd.DataFrame | list[ImageRecord], path: str | Path) -> None:
    if isinstance(df, list):
        df = to_frame(df)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".parquet":
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False)


def read_metadata(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path, dtype={"image_id": str, "sequence_id": str, "capture_date": str})
    for col in COLUMNS:
        if col not in df.columns:
            df[col] = None
    df["synthetic"] = df["synthetic"].fillna(False).astype(bool)
    df["privacy_reviewed"] = df["privacy_reviewed"].fillna(False).astype(bool)
    return df[COLUMNS]
