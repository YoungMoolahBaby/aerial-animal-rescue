"""Shared helpers: tiny fake images, labels and pools."""

import zlib
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from aerial_animal_rescue.classes import ClassRegistry
from aerial_animal_rescue.metadata import ImageRecord, write_metadata
from aerial_animal_rescue.yolo import write_labels

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def registry() -> ClassRegistry:
    return ClassRegistry.from_yaml(REPO / "configs" / "classes.yaml")


def make_image(path: Path, size=(64, 48), seed: int = 0) -> Path:
    """Write a small random RGB image (different seeds -> different bytes)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.random.default_rng(seed).integers(0, 255, (size[1], size[0], 3), dtype=np.uint8)
    Image.fromarray(arr).save(path)
    return path


def make_pool(root: Path, rows: list[dict]) -> Path:
    """A pool with images/, labels/ (one centered box each) and metadata.csv.

    Each row needs image_id; other ImageRecord fields are optional.
    """
    records = []
    for i, row in enumerate(rows):
        image_id = row["image_id"]
        make_image(root / "images" / f"{image_id}.png", seed=zlib.crc32(image_id.encode()) + i)
        write_labels(root / "labels" / f"{image_id}.txt", [(1, 0.5, 0.5, 0.2, 0.2)])
        defaults = {"source": "test", "source_license": "CC-BY-4.0"}
        records.append(ImageRecord(file_path=f"images/{image_id}.png", **{**defaults, **row}))
    write_metadata(records, root / "metadata.csv")
    return root
