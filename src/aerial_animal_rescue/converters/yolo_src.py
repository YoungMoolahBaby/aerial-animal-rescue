"""Reader for datasets already in YOLO format but with their own class list."""

from collections import Counter
from collections.abc import Iterator
from pathlib import Path

from PIL import Image

from aerial_animal_rescue.config import IMAGE_EXTS
from aerial_animal_rescue.converters.base import SourceImage
from aerial_animal_rescue.yolo import read_labels, yolo_to_xyxy


def _labels_by_stem(annotations: Path) -> dict[str, Path]:
    """Label files by file name, leaving out names that occur more than once."""
    labels = list(annotations.rglob("*.txt"))
    counts = Counter(p.stem for p in labels)
    return {p.stem: p for p in labels if counts[p.stem] == 1}


def read_yolo(
    annotations: Path,
    images: Path,
    class_names: list[str] | None = None,
    sequence_from_folder: bool = False,
    **_,
) -> Iterator[SourceImage]:
    """`annotations`: folder of .txt labels; `class_names`: the source's names by id.

    A label is looked up at the image's relative path first, then by file name
    anywhere under `annotations` (CVAT exports nest labels in obj_train_data/,
    and folder names do not always match the image folders).

    Without `class_names`, the numeric id (as a string) is used as the name, so
    class_map can map e.g. "0": cattle. With `sequence_from_folder`, the image's
    folder (relative to `images`) is its sequence id, e.g. one folder per flight.
    """
    annotations, images = Path(annotations), Path(images)
    by_stem = None
    for img_path in sorted(p for p in images.rglob("*") if p.suffix.lower() in IMAGE_EXTS):
        rel = img_path.relative_to(images)
        label = annotations / rel.with_suffix(".txt")
        if not label.exists():
            if by_stem is None:
                by_stem = _labels_by_stem(annotations)
            label = by_stem.get(img_path.stem)
        if label is None:
            print(f"warning: no label for {img_path}, skipped")
            continue
        with Image.open(img_path) as im:
            w, h = im.size
        boxes = []
        for box in read_labels(label):
            name = class_names[box[0]] if class_names else str(box[0])
            boxes.append((name, *yolo_to_xyxy(box, w, h)))
        sequence_id = rel.parent.as_posix() if sequence_from_folder and rel.parent.parts else None
        yield SourceImage(path=img_path, boxes=boxes, width=w, height=h, sequence_id=sequence_id)
