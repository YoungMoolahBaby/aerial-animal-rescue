"""Reader for COCO-style JSON (`images`, `annotations`, `categories`; bbox = x, y, w, h)."""

import json
from collections import defaultdict
from collections.abc import Iterator
from pathlib import Path

from aerial_animal_rescue.converters.base import SourceImage


def read_coco(annotations: Path, images: Path, **_) -> Iterator[SourceImage]:
    """`annotations` is the JSON file, `images` the folder `file_name` is relative to."""
    data = json.loads(Path(annotations).read_text(encoding="utf-8"))
    # HIT-UAV's JSON says "annotation" instead of "annotations"; accept both.
    anns = data.get("annotations", data.get("annotation", []))
    cat_names = {c["id"]: c["name"] for c in data["categories"]}
    by_image = defaultdict(list)
    for a in anns:
        x, y, w, h = a["bbox"]
        by_image[a["image_id"]].append((cat_names[a["category_id"]], x, y, x + w, y + h))
    for img in data["images"]:
        yield SourceImage(
            path=Path(images) / img["file_name"],
            boxes=by_image.get(img["id"], []),
            width=img.get("width"),
            height=img.get("height"),
        )
