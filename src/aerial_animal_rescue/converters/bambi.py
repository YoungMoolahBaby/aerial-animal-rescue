"""Reader for BAMBI (bambi.eco): nadir thermal + RGB drone video of forest wildlife.

One zip per flight (Zenodo "The BAMBI Dataset: Multimodal Nadir UAV-Recordings of
Forest Wildlife", v2, CC BY 4.0), holding among others:
  <id>_matched_processed.mp4  2048x1024: thermal left half, RGB right half (undistorted)
  <id>_gt.txt                 MOT rows `frame, track, left, top, w, h, conf, class_id,
                              visibility, species, gender, age, is_propagated`
                              in thermal pixels; only the human key frames are stored
  <id>_metadata.json          start time, drone, and a `split` that we do NOT use
Layout and columns checked on the sample record and the bambi-eco/Dataset docs
(2026-11-04). `frame` counts decoded video frames from 0 (their frame_extraction.py).

Split: from `splits_file`, a copy of flight_metadata/splits.json of bambi-eco/Dataset
(commit 252ab46), which clusters flights by named location and GPS (extend_splits.py).
The `split` inside each flight's metadata.json disagrees for 134 of 386 flights and
puts the same locations into train and val, so it is ignored. `exclude_flights` leaves
out train flights at the two named locations splits.json also uses for test.

Which frames: key frames of any track, at least `min_gap_s` apart, at most
`max_per_flight` per flight (evenly spread). Each track is linearly interpolated
between its key frames, as their mot_interpolation.py does, so a frame also gets
the boxes of animals whose own key frames fall elsewhere. Outside a track's first..
last key frame the animal has no box, which is why only key frames are used.
Key frames come about once a second per track (gap p95 1.6 s on 29 flights); frames
that need a gap longer than `max_interp_s` are skipped (1% of frames), since a box
interpolated that far may have slid off its animal.

Frames are decoded once into `cache` (thermal half only, JPEG) and reused, so a
re-conversion doesn't touch the videos again. RGB is not used yet: its boxes are
machine-transferred and unreviewed for most flights.
"""

import json
import zipfile
from bisect import bisect_left
from collections import defaultdict
from collections.abc import Iterator
from pathlib import Path

from aerial_animal_rescue.converters.base import SourceImage

THERMAL_SIZE = 1024  # left half of the 2048x1024 video


def read_tracks(gt_text: str) -> dict[int, list[tuple[int, float, float, float, float, str]]]:
    """Key frames per track: track -> sorted [(frame, left, top, w, h, species)]."""
    tracks = defaultdict(list)
    for line in gt_text.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 10 or not parts[0].isdigit():
            continue
        frame, track = int(parts[0]), int(parts[1])
        left, top, w, h = (float(v) for v in parts[2:6])
        tracks[track].append((frame, left, top, w, h, parts[9]))
    return {t: sorted(v) for t, v in tracks.items()}


def boxes_at(tracks: dict, frame: int) -> list[tuple[str, float, float, float, float]]:
    """(species, x1, y1, x2, y2) of every track alive at `frame` (key or interpolated)."""
    out = []
    for keys in tracks.values():
        frames = [k[0] for k in keys]
        i = bisect_left(frames, frame)
        if i < len(keys) and frames[i] == frame:
            _, left, top, w, h, species = keys[i]
        elif 0 < i < len(keys):
            a, b = keys[i - 1], keys[i]
            t = (frame - a[0]) / (b[0] - a[0])
            left, top, w, h = (round(a[j] + (b[j] - a[j]) * t) for j in range(1, 5))
            species = a[5]
        else:
            continue
        out.append((species, left, top, left + w, top + h))
    return out


def longest_gap(tracks: dict, frame: int) -> int:
    """Longest key-frame gap (frames) any track is interpolated across at `frame`."""
    longest = 0
    for keys in tracks.values():
        frames = [k[0] for k in keys]
        i = bisect_left(frames, frame)
        if 0 < i < len(keys) and frames[i] != frame:
            longest = max(longest, frames[i] - frames[i - 1])
    return longest


def pick_frames(key_frames: list[int], min_gap: int, max_n: int) -> list[int]:
    """Key frames at least `min_gap` apart (greedy from the first), then at most `max_n`."""
    picked = []
    for f in sorted(set(key_frames)):
        if not picked or f - picked[-1] >= min_gap:
            picked.append(f)
    if len(picked) > max_n:
        step = len(picked) / max_n
        picked = [picked[int(i * step)] for i in range(max_n)]
    return picked


def decode_frames(
    zf: zipfile.ZipFile,
    member: str,
    frames: list[int],
    out: dict[int, Path],
    tmp_dir: Path,
    quality: int = 95,
) -> None:
    """Write the thermal half of `frames` (decoded in order, no seeking) to `out` paths."""
    import cv2

    tmp_dir.mkdir(parents=True, exist_ok=True)
    video = Path(zf.extract(member, tmp_dir))
    try:
        cap = cv2.VideoCapture(str(video))
        wanted, idx, last = set(frames), 0, max(frames)
        while idx <= last:
            if idx in wanted:
                ok, img = cap.read()
                if not ok:
                    break
                if img.shape[:2] != (THERMAL_SIZE, 2 * THERMAL_SIZE):
                    raise ValueError(f"{member}: frame size {img.shape[:2]}, want 1024x2048")
                out[idx].parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(
                    str(out[idx]), img[:, :THERMAL_SIZE], [cv2.IMWRITE_JPEG_QUALITY, quality]
                )
            elif not cap.grab():
                break
            idx += 1
        cap.release()
        missing = [f for f in frames if not out[f].exists()]
        if missing:
            raise ValueError(f"{member}: video ended before frames {missing[:5]}")
    finally:
        video.unlink(missing_ok=True)


def read_bambi(
    annotations: Path,
    images: Path,
    min_gap_s: float = 2.0,
    fps: float = 30.0,
    max_per_flight: int = 60,
    max_interp_s: float = 3.0,
    flights: list[int] | None = None,
    splits_file: str = "splits.json",
    exclude_flights: list[int] = (),
    **_,
) -> Iterator[SourceImage]:
    """`annotations`: folder of flight_<id>.zip + `splits_file`; `images`: frame cache.

    Sequence id = flight id; split = BAMBI's splits.json (use with keep_source_split).
    """
    zips, cache = Path(annotations), Path(images)
    split_of = {
        str(f): s for s, fs in json.loads((zips / splits_file).read_text()).items() for f in fs
    }
    paths = sorted(zips.glob("flight_*.zip"), key=lambda p: int(p.stem.split("_")[1]))
    for zp in paths:
        fid = zp.stem.split("_")[1]
        if (flights is not None and int(fid) not in flights) or int(fid) in exclude_flights:
            continue
        if fid not in split_of:
            raise ValueError(f"flight {fid} is not in {splits_file}")
        with zipfile.ZipFile(zp) as zf:
            names = {Path(n).name: n for n in zf.namelist()}
            meta = json.loads(zf.read(names[f"{fid}_metadata.json"]))
            tracks = read_tracks(zf.read(names[f"{fid}_gt.txt"]).decode("utf-8"))
            if not tracks:
                continue
            limit = max_interp_s * fps
            keys = [k[0] for v in tracks.values() for k in v]
            keys = [f for f in keys if longest_gap(tracks, f) <= limit]
            frames = pick_frames(keys, round(min_gap_s * fps), max_per_flight)
            out = {f: cache / fid / f"{fid}_{f:08d}.jpg" for f in frames}
            todo = [f for f in frames if not out[f].exists()]
            if todo:
                decode_frames(zf, names[f"{fid}_matched_processed.mp4"], todo, out, cache / "_tmp")
        date = (meta.get("flight_info") or {}).get("start_time", "")[:10] or None
        for f in frames:
            yield SourceImage(
                path=out[f],
                boxes=boxes_at(tracks, f),
                sequence_id=f"flight{fid}",
                width=THERMAL_SIZE,
                height=THERMAL_SIZE,
                capture_date=date,
                split=split_of[fid],
            )
