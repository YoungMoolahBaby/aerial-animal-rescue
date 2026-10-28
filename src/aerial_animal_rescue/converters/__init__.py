"""Converters: third-party annotations -> our YOLO pools.

Each reader yields `SourceImage`s (an image path plus pixel boxes labeled with
the SOURCE's class names). `base.write_pool` then maps the names onto our
classes, strips GPS, and writes images/, labels/, metadata.csv and
conversion_log.txt. Adding a dataset = writing one small reader.
"""

from aerial_animal_rescue.converters.base import SourceImage, write_pool
from aerial_animal_rescue.converters.birdsai import read_birdsai
from aerial_animal_rescue.converters.coco import read_coco
from aerial_animal_rescue.converters.hit_uav import read_hit_uav
from aerial_animal_rescue.converters.voc import read_voc
from aerial_animal_rescue.converters.yolo_src import read_yolo

READERS = {
    "birdsai_csv": read_birdsai,
    "coco": read_coco,
    "hit_uav": read_hit_uav,
    "voc": read_voc,
    "yolo": read_yolo,
}

__all__ = ["READERS", "SourceImage", "write_pool"]
