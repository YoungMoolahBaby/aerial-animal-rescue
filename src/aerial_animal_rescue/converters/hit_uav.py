"""Reader for HIT-UAV: PASCAL VOC boxes plus video and altitude from the file names.

File names are <time>_<altitude m>_<camera angle deg>_<weather>_<serial>, e.g.
0_100_30_0_03280 (time is day/night; the README doesn't say which digit is which).
The serial is unique across the dataset and follows the source videos: within one
time/altitude/angle setting it steps by at most 50 inside a video and jumps by
hundreds between videos (2-3 videos per setting, 264 in total). Each such run is
one sequence. Neighbouring settings often film the same place from another
angle, so splitting by video would still leak; sources.yaml keeps the whole
source in train (`train_only`).
"""

from collections import defaultdict
from collections.abc import Iterator
from pathlib import Path

from aerial_animal_rescue.converters.base import SourceImage
from aerial_animal_rescue.converters.voc import read_voc

RUN_GAP = 100  # serial jump that starts a new video (observed: <=50 within, >200 between)


def parse_name(stem: str) -> tuple[str, int, int, int]:
    """(time, altitude m, camera angle deg, serial) from a HIT-UAV file stem."""
    time, altitude, angle, _weather, serial = stem.split("_")
    return time, int(altitude), int(angle), int(serial)


def read_hit_uav(annotations: Path, images: Path, **_) -> Iterator[SourceImage]:
    items = list(read_voc(annotations, images))
    by_setting = defaultdict(list)
    for item in items:
        time, altitude, angle, serial = parse_name(item.path.stem)
        item.altitude_m = float(altitude)
        by_setting[(time, altitude, angle)].append((serial, item))
    for (time, altitude, angle), group in by_setting.items():
        group.sort(key=lambda g: g[0])
        run = 0
        for k, (serial, item) in enumerate(group):
            if k and serial - group[k - 1][0] > RUN_GAP:
                run += 1
            item.sequence_id = f"t{time}_{altitude}m_{angle}deg_{run}"
    yield from items
