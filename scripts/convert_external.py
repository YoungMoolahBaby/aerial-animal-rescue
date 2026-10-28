"""Convert a downloaded (and unzipped) external dataset into a YOLO pool.

Reads the converter settings and class_map from configs/sources.yaml. Paths in
sources.yaml are relative to data/external/<source>/ and may be lists (paired
in order, e.g. one per split); --images/--annotations override them.

    uv run python scripts/convert_external.py hit_uav
    -> data/interim/hit_uav/{images,labels,metadata.csv,conversion_log.txt}
"""

import argparse
from itertools import chain
from pathlib import Path

from aerial_animal_rescue.classes import ClassMapper, ClassRegistry
from aerial_animal_rescue.config import EXTERNAL_DIR, INTERIM_DIR
from aerial_animal_rescue.converters import READERS, write_pool
from aerial_animal_rescue.converters.base import drop_excluded, drop_images_with_classes
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
    images = [args.images] if args.images else as_paths(base, options.pop("images", "TODO"))
    annotations = (
        [args.annotations]
        if args.annotations
        else as_paths(base, options.pop("annotations", "TODO"))
    )
    options.pop("images", None)
    options.pop("annotations", None)
    if len(images) != len(annotations):
        raise SystemExit("images and annotations must list the same number of folders")
    for path in images + annotations:
        if "TODO" in path.as_posix() or not path.exists():
            raise SystemExit(f"{path} not found: set the paths in sources.yaml or pass flags")

    reader = READERS[fmt]
    items = chain.from_iterable(
        reader(annotations=a, images=i, **options) for i, a in zip(images, annotations, strict=True)
    )
    exclude = src.raw.get("exclude") or []
    if exclude:
        items = drop_excluded(items, exclude)
        print(f"{src.name}: excluded {len(exclude)} images listed in sources.yaml")
    exclude_classes = src.raw.get("exclude_classes") or []
    if exclude_classes:
        items, dropped = drop_images_with_classes(items, exclude_classes)
        print(f"{src.name}: excluded {dropped} images with a box of {exclude_classes}")
    write_pool(
        items,
        ClassMapper(ClassRegistry.from_yaml(), src.class_map),
        args.out or INTERIM_DIR / src.name,
        source=src.name,
        source_license=src.license,
        modality=src.modality,
        # several folders -> ids relative to the source dir, so they stay unique
        image_root=images[0] if len(images) == 1 else base,
        scene_type=args.scene_type,
    )


def as_paths(base: Path, value: str | list[str]) -> list[Path]:
    return [base / str(v) for v in (value if isinstance(value, list) else [value])]


if __name__ == "__main__":
    main()
