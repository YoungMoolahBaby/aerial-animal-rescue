"""Ingest our own drone footage: videos -> frames every N seconds, stills -> copies.

Every output image is GPS-free (see privacy.py). Output is an unlabeled pool
(images/ + metadata.csv); frames enter the final dataset only once a label
file exists for them (annotate with e.g. CVAT or Label Studio, export YOLO).

Sequence ids: one per video file; for stills, the folder they sit in (one
folder per flight is the convention for data/raw/).
"""

from datetime import date
from pathlib import Path

import cv2
import pandas as pd
from PIL import Image
from tqdm import tqdm

from aerial_animal_rescue.config import IMAGE_EXTS, VIDEO_EXTS
from aerial_animal_rescue.converters.base import safe_id
from aerial_animal_rescue.metadata import ImageRecord, read_metadata, to_frame, write_metadata
from aerial_animal_rescue.privacy import copy_without_gps

EXIF_IFD = 0x8769
DATETIME_ORIGINAL = 0x9003


def capture_date_of(path: Path) -> str:
    """EXIF DateTimeOriginal for stills, else the file's modification date (YYYY-MM-DD)."""
    if path.suffix.lower() in IMAGE_EXTS:
        try:
            with Image.open(path) as im:
                dt = im.getexif().get_ifd(EXIF_IFD).get(DATETIME_ORIGINAL)
            if dt:
                return dt[:10].replace(":", "-")
        except Exception:
            pass
    return date.fromtimestamp(path.stat().st_mtime).isoformat()


def extract_frames(video: Path, out_dir: Path, stem: str, interval_s: float) -> list[Path]:
    """Save one JPEG every `interval_s` seconds. OpenCV writes no EXIF/GPS."""
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise OSError(f"cannot open video {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, round(fps * interval_s))
    out_dir.mkdir(parents=True, exist_ok=True)
    written, idx = [], 0
    while True:
        ok = cap.grab()
        if not ok:
            break
        if idx % step == 0:
            ok, frame = cap.retrieve()
            if ok:
                dst = out_dir / f"{stem}_f{idx:06d}.jpg"
                ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                dst.write_bytes(buf.tobytes())  # imencode + write: safe for non-ASCII paths
                written.append(dst)
        idx += 1
    cap.release()
    return written


def ingest(
    input_dir: Path,
    out_dir: Path,
    modality: str,
    interval_s: float = 2.0,
    source: str = "own",
    source_license: str = "CC-BY-4.0",
    sequence_id: str | None = None,
    **record_defaults,
) -> list[ImageRecord]:
    """Ingest everything under `input_dir`, appending to `out_dir/metadata.csv`."""
    input_dir, out_dir = Path(input_dir), Path(out_dir)
    images_dir = out_dir / "images"
    files = sorted(p for p in input_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTS | VIDEO_EXTS)
    records = []
    for path in tqdm(files, desc="ingest"):
        rel = path.relative_to(input_dir)
        if path.suffix.lower() in VIDEO_EXTS:
            seq = sequence_id or safe_id(rel.with_suffix("").as_posix())
            outputs = extract_frames(path, images_dir, safe_id(f"{source}_{seq}"), interval_s)
        else:
            seq = sequence_id or safe_id(rel.parent.as_posix() or input_dir.name)
            stem = safe_id(f"{source}_{rel.with_suffix('').as_posix()}")
            dst = images_dir / f"{stem}{path.suffix.lower()}"
            copy_without_gps(path, dst)
            outputs = [dst]
        when = capture_date_of(path)
        for out in outputs:
            records.append(
                ImageRecord(
                    image_id=out.stem,
                    file_path=out.relative_to(out_dir).as_posix(),
                    source=source,
                    source_license=source_license,
                    modality=modality,
                    sequence_id=seq,
                    capture_date=when,
                    **record_defaults,
                )
            )
    meta_path = out_dir / "metadata.csv"
    df = to_frame(records)
    if meta_path.exists():  # append, replacing rows of re-ingested images
        old = read_metadata(meta_path)
        df = pd.concat([old[~old["image_id"].isin(df["image_id"])], df])
    write_metadata(df.reset_index(drop=True), meta_path)
    return records
