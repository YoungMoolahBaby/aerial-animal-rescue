"""YOLO label format helpers.

A YOLO label file has one line per object: `class_id cx cy w h`, with the box
center and size normalized to [0, 1] by the image width and height. The label
for `<root>/images/<sub>/x.jpg` lives at `<root>/labels/<sub>/x.txt`. An empty
label file means "checked, no objects" (a valid negative example).
"""

from pathlib import Path

YoloBox = tuple[int, float, float, float, float]  # class_id, cx, cy, w, h (normalized)


def xyxy_to_yolo(
    class_id: int, x1: float, y1: float, x2: float, y2: float, img_w: int, img_h: int
) -> YoloBox | None:
    """Convert a pixel box (corners) to a normalized YOLO box.

    The box is clipped to the image. Returns None when nothing is left
    (degenerate or fully outside), so callers can skip it.
    """
    x1, x2 = sorted((max(0.0, min(x1, img_w)), max(0.0, min(x2, img_w))))
    y1, y2 = sorted((max(0.0, min(y1, img_h)), max(0.0, min(y2, img_h))))
    w, h = x2 - x1, y2 - y1
    if w <= 0 or h <= 0:
        return None
    return (class_id, (x1 + w / 2) / img_w, (y1 + h / 2) / img_h, w / img_w, h / img_h)


def yolo_to_xyxy(box: YoloBox, img_w: int, img_h: int) -> tuple[float, float, float, float]:
    """Convert a normalized YOLO box back to pixel corners."""
    _, cx, cy, w, h = box
    return ((cx - w / 2) * img_w, (cy - h / 2) * img_h, (cx + w / 2) * img_w, (cy + h / 2) * img_h)


def format_line(box: YoloBox) -> str:
    c, cx, cy, w, h = box
    return f"{c} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


def write_labels(path: str | Path, boxes: list[YoloBox]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(format_line(b) + "\n" for b in boxes), encoding="utf-8")


def read_labels(path: str | Path) -> list[YoloBox]:
    """Parse a label file. Raises ValueError on malformed lines."""
    boxes = []
    for n, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"{path}:{n}: expected 5 values, got {len(parts)}")
        cls = float(parts[0])
        if not cls.is_integer():
            raise ValueError(f"{path}:{n}: class id {parts[0]!r} is not an integer")
        boxes.append((int(cls), *(float(p) for p in parts[1:])))
    return boxes


def label_path_for(image_path: str | Path) -> Path:
    """`.../images/<sub>/x.jpg` -> `.../labels/<sub>/x.txt` (last `images` dir swapped)."""
    parts = list(Path(image_path).parts)
    idx = len(parts) - 1 - parts[::-1].index("images")
    parts[idx] = "labels"
    return Path(*parts).with_suffix(".txt")
