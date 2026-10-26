"""Convert a downloaded (and unzipped) external dataset into a YOLO pool.

Reads the converter settings and class_map from configs/sources.yaml. Paths in
sources.yaml are relative to data/external/<source>/; --images/--annotations
override them.

    uv run python scripts/convert_external.py hit_uav
    -> data/interim/hit_uav/{images,labels,metadata.csv,conversion_log.txt}
"""

import argparse
from pathlib import Path

from aerial_animal_rescue.classes import ClassMapper, ClassRegistry
from aerial_animal_rescue.config import EXTERNAL_DIR, INTERIM_DIR
from aerial_animal_rescue.converters import READERS, write_pool
from aerial_animal_rescue.metadata import SCENE_TYPES
from aerial_animal_rescue.sources import load_sources


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("source")
    p.add_argument("--images", type=Path, help="override converter.images")
    p.add_argument("--annotations", type=Path, help="override converter.annotations")
    p.add_argument("--out", type=Path, help="default: data/interim/<source>")
    p.add_argument("--scene-type", default="other", choices=sorted(SCENE_TYPES))
    args = p.parse_args()

    src = load_sources()[args.source]
    options = dict(src.converter)
    fmt = options.pop("format", None)
    if fmt not in READERS:
        raise SystemExit(f"{args.source}: no converter for format {fmt!r} (see sources.yaml)")
    base = EXTERNAL_DIR / src.name
    images = args.images or base / str(options.pop("images", "TODO"))
    annotations = args.annotations or base / str(options.pop("annotations", "TODO"))
    options.pop("images", None)
    options.pop("annotations", None)
    for path in (images, annotations):
        if "TODO" in path.as_posix() or not path.exists():
            raise SystemExit(f"{path} not found: set the paths in sources.yaml or pass flags")

    write_pool(
        READERS[fmt](annotations=annotations, images=images, **options),
        ClassMapper(ClassRegistry.from_yaml(), src.class_map),
        args.out or INTERIM_DIR / src.name,
        source=src.name,
        source_license=src.license,
        modality=src.modality,
        image_root=images,
        scene_type=args.scene_type,
    )


if __name__ == "__main__":
    main()
