"""Common conversion step: source boxes -> our classes -> YOLO pool on disk."""

import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image
from tqdm import tqdm

from aerial_animal_rescue.classes import ClassMapper
from aerial_animal_rescue.metadata import ImageRecord, write_metadata
from aerial_animal_rescue.privacy import copy_without_gps
from aerial_animal_rescue.yolo import write_labels, xyxy_to_yolo

# (source class name, x1, y1, x2, y2) in pixels
PixelBox = tuple[str, float, float, float, float]


@dataclass
class SourceImage:
    path: Path
    boxes: list[PixelBox] = field(default_factory=list)
    sequence_id: str | None = None
    width: int | None = None  # read from the file when not given
    height: int | None = None
    altitude_m: float | None = None  # per image, when the source records it


def drop_excluded(items: Iterable[SourceImage], names: Iterable[str]) -> list[SourceImage]:
    """Remove images whose file name (without extension) is in `names`.

    Raises ValueError if a name matches nothing, so a typo can't silently keep an image.
    """
    names = set(names)
    items = list(items)
    found = {i.path.stem for i in items} & names
    if found != names:
        raise ValueError(f"exclude list names images that don't exist: {sorted(names - found)}")
    return [i for i in items if i.path.stem not in names]


def drop_images_with_classes(
    items: Iterable[SourceImage], names: Iterable[str]
) -> tuple[list[SourceImage], int]:
    """Remove images holding a box of any source class in `names`; returns (kept, dropped).

    For classes like "DontCare" that may hide our objects: dropping only the box would
    leave them unlabelled, which teaches the model they are background.
    """
    names, items = set(names), list(items)
    kept = [i for i in items if not any(b[0] in names for b in i.boxes)]
    return kept, len(items) - len(kept)


def safe_id(text: str) -> str:
    """Make a filesystem- and URL-safe id fragment."""
    return re.sub(r"[^A-Za-z0-9_-]+", "-", text).strip("-")


def write_pool(
    images: Iterable[SourceImage],
    mapper: ClassMapper,
    out_dir: str | Path,
    source: str,
    source_license: str,
    modality: str,
    image_root: Path | None = None,
    **record_defaults,
) -> list[ImageRecord]:
    """Write a converted pool: out_dir/{images,labels}/, metadata.csv, conversion_log.txt.

    image_id = "<source>_<path relative to image_root>", so ids stay unique and stable.
    Images keep their labels even when every box was dropped: an empty label
    file marks a verified negative, which is still useful training data.
    """
    out_dir = Path(out_dir)
    records, skipped_boxes = [], 0
    for item in tqdm(list(images), desc=f"convert {source}"):
        rel = item.path.relative_to(image_root) if image_root else Path(item.path.name)
        image_id = safe_id(f"{source}_{rel.with_suffix('').as_posix()}")
        if item.width is None or item.height is None:
            with Image.open(item.path) as im:
                item.width, item.height = im.size
        boxes = []
        for name, x1, y1, x2, y2 in item.boxes:
            cls = mapper.map(name)
            if cls is None:
                continue
            yb = xyxy_to_yolo(cls, x1, y1, x2, y2, item.width, item.height)
            if yb is None:
                skipped_boxes += 1
            else:
                boxes.append(yb)
        dst = out_dir / "images" / f"{image_id}{item.path.suffix.lower()}"
        copy_without_gps(item.path, dst)
        write_labels(out_dir / "labels" / f"{image_id}.txt", boxes)
        records.append(
            ImageRecord(
                image_id=image_id,
                file_path=dst.relative_to(out_dir).as_posix(),
                source=source,
                source_license=source_license,
                modality=modality,
                sequence_id=item.sequence_id,
                **record_defaults,
                **({"altitude_m": item.altitude_m} if item.altitude_m is not None else {}),
            )
        )
    write_metadata(records, out_dir / "metadata.csv")
    log = f"source: {source}\nimages: {len(records)}\n{mapper.report()}\n"
    log += f"degenerate/out-of-image boxes skipped: {skipped_boxes}\n"
    (out_dir / "conversion_log.txt").write_text(log, encoding="utf-8")
    print(log)
    return records
