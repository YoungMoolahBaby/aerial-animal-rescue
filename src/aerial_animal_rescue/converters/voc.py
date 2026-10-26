"""Reader for PASCAL VOC XML (one .xml per image)."""

import xml.etree.ElementTree as ET
from collections.abc import Iterator
from pathlib import Path

from aerial_animal_rescue.config import IMAGE_EXTS
from aerial_animal_rescue.converters.base import SourceImage


def _find_image(images: Path, stem: str, filename: str | None) -> Path | None:
    if filename and (images / filename).exists():
        return images / filename
    for ext in IMAGE_EXTS:
        for cand in (images / f"{stem}{ext}", images / f"{stem}{ext.upper()}"):
            if cand.exists():
                return cand
    return None


def read_voc(annotations: Path, images: Path, **_) -> Iterator[SourceImage]:
    """`annotations` is a folder of .xml files; `images` holds the matching images."""
    annotations, images = Path(annotations), Path(images)
    for xml_path in sorted(annotations.rglob("*.xml")):
        root = ET.parse(xml_path).getroot()
        img_path = _find_image(images, xml_path.stem, root.findtext("filename"))
        if img_path is None:
            print(f"warning: no image for {xml_path}, skipped")
            continue
        boxes = []
        for obj in root.iter("object"):
            bb = obj.find("bndbox")
            if bb is None:
                continue
            coords = [float(bb.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax")]
            boxes.append((obj.findtext("name", "").strip(), *coords))
        size = root.find("size")
        w = int(float(size.findtext("width", "0"))) if size is not None else 0
        h = int(float(size.findtext("height", "0"))) if size is not None else 0
        yield SourceImage(path=img_path, boxes=boxes, width=w or None, height=h or None)
