"""Shared paths and YAML loading.

All default paths are relative to the current working directory, so run the
scripts from the repository root.
"""

from pathlib import Path
from typing import Any

import yaml

CONFIGS_DIR = Path("configs")
CLASSES_YAML = CONFIGS_DIR / "classes.yaml"
SOURCES_YAML = CONFIGS_DIR / "sources.yaml"

DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"  # our own original drone footage (never modified)
EXTERNAL_DIR = DATA_DIR / "external"  # third-party downloads, one folder per source
INTERIM_DIR = DATA_DIR / "interim"  # converted/ingested pools, not yet split
SYNTHETIC_DIR = DATA_DIR / "synthetic"  # composites (+ cutouts/ and backgrounds/ inputs)
PROCESSED_DIR = DATA_DIR / "processed"  # final, split, YOLO-format dataset

RUNS_DIR = Path("runs")

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}


def load_yaml(path: str | Path) -> Any:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)
