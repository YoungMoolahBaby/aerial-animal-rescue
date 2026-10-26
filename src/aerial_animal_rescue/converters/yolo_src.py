"""Reader for datasets already in YOLO format but with their own class list."""

from collections.abc import Iterator
from pathlib import Path

from PIL import Image

from aerial_animal_rescue.config import IMAGE_EXTS
from aerial_animal_rescue.converters.base import SourceImage
from aerial_animal_rescue.yolo import read_labels, yolo_to_xyxy


def read_yolo(
    annotations: Path, images: Path, class_names: list[str] | None = None, **_
) -> Iterator[SourceImage]:
    """`annotations`: folder of .txt labels; `class_names`: the source's names by id.

    Without `class_names`, the numeric id (as a string) is used as the name, so
    class_map can map e.g. "0": cattle.
    """
    annotations, images = Path(annotations), Path(images)
    for img_path in sorted(p for p in images.rglob("*") if p.suffix.lower() in IMAGE_EXTS):
        label = annotations / img_path.relative_to(images).with_suffix(".txt")
        if not label.exists():
            print(f"warning: no label for {img_path}, skipped")
            continue
        with Image.open(img_path) as im:
            w, h = im.size
        boxes = []
        for box in read_labels(label):
            name = class_names[box[0]] if class_names else str(box[0])
            boxes.append((name, *yolo_to_xyxy(box, w, h)))
        yield SourceImage(path=img_path, boxes=boxes, width=w, height=h)
