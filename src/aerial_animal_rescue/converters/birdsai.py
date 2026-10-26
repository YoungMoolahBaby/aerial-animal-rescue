"""Reader for BIRDSAI (LILA "Conservation Drones"): thermal video frames + MOT-style CSV.

Each video has a CSV with rows `frame, object_id, x, y, w, h, class, species,
occlusion, noise` (class 0 = animal, 1 = human). We turn class + species into
a name that configs/sources.yaml maps onto our classes.

Layout (checked on the v01 real train/test zips, 2026-10-26):
  <split>/annotations/<video>.csv and <split>/images/<video>/<video>_<number>.jpg

The CSV frame is 0-based from the video's FIRST image, not the number in the
file name: almost every video starts at 0, but 0000000011_0000000000 starts at
87 (confirmed by box brightness peaking at that offset). Frames outside the
first..last annotated frame are skipped, since we can't tell "empty" from
"not annotated" there.
"""

import csv
import re
from collections import defaultdict
from collections.abc import Iterator
from pathlib import Path

from aerial_animal_rescue.config import IMAGE_EXTS
from aerial_animal_rescue.converters.base import SourceImage

SPECIES = {
    -1: "unknown_animal",
    0: "human",
    1: "elephant",
    2: "lion",
    3: "giraffe",
    4: "dog",
    5: "crocodile",
    6: "hippo",
    7: "zebra",
    8: "rhino",
}


def class_name(cls: int, species: int) -> str:
    return "human" if cls == 1 else SPECIES.get(species, "unknown_animal")


def _frame_number(path: Path) -> int | None:
    m = re.search(r"(\d+)$", path.stem)
    return int(m.group(1)) if m else None


def read_birdsai(
    annotations: Path, images: Path, frame_step: int = 1, **_
) -> Iterator[SourceImage]:
    """Yield every `frame_step`-th frame of each video's annotated span (sequence id = video)."""
    annotations, images = Path(annotations), Path(images)
    for csv_path in sorted(annotations.glob("*.csv")):
        video = csv_path.stem
        boxes = defaultdict(list)
        with open(csv_path, newline="", encoding="utf-8") as f:
            for row in csv.reader(f):
                if not row or not row[0].strip().lstrip("-").isdigit():
                    continue  # header or blank line
                frame, _, x, y, w, h, cls, species = row[:8]
                x, y, w, h = float(x), float(y), float(w), float(h)
                boxes[int(frame)].append((class_name(int(cls), int(species)), x, y, x + w, y + h))
        frames = {}
        for p in (images / video).glob("*"):
            n = _frame_number(p)
            if p.suffix.lower() in IMAGE_EXTS and n is not None:
                frames[n] = p
        if not frames or not boxes:
            print(f"warning: no frames or no boxes for {video}, skipped")
            continue
        first = min(frames)
        span = range(min(boxes), max(boxes) + 1)
        kept = [n for n in sorted(frames) if n - first in span]
        for n in kept[::frame_step]:
            yield SourceImage(path=frames[n], boxes=boxes.get(n - first, []), sequence_id=video)
