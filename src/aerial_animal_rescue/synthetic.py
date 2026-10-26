"""Synthetic composites: paste animal cutouts onto disaster backgrounds.

Inputs (under data/synthetic/ by default):
  cutouts/<class_name>/*.png   RGBA cutouts, transparent background, ideally
                               seen from above (nadir) like drone footage
  backgrounds/**/*.jpg         e.g. FloodNet flood scenes
Outputs: images/, labels/, metadata.csv with synthetic=True.

Every composite from one background shares a sequence id, so the splitter
keeps them together. Composites never go into the test set.
"""

import math
import random
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageEnhance
from tqdm import tqdm

from aerial_animal_rescue.classes import ClassRegistry
from aerial_animal_rescue.config import IMAGE_EXTS
from aerial_animal_rescue.converters.base import safe_id
from aerial_animal_rescue.metadata import ImageRecord, write_metadata
from aerial_animal_rescue.yolo import YoloBox, write_labels, xyxy_to_yolo

ALPHA_THRESHOLD = 32  # alpha above this counts as "object" for the tight box


@dataclass
class CompositeConfig:
    n_images: int = 100
    objects_per_image: tuple[int, int] = (1, 6)
    object_size_px: tuple[int, int] = (12, 80)  # longest side after scaling ~ flight altitude
    max_background_side: int = 2048  # downscale large backgrounds first (speed)
    max_rotation_deg: float = 180.0
    brightness: tuple[float, float] = (0.6, 1.3)
    blur_prob: float = 0.3
    blur_sigma: tuple[float, float] = (0.5, 1.5)
    rain_prob: float = 0.25
    noise_prob: float = 0.5
    noise_sigma: tuple[float, float] = (2.0, 10.0)
    max_overlap_iou: float = 0.1
    seed: int = 0


def _iou(a: tuple, b: tuple) -> float:
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union else 0.0


def transform_cutout(cut: Image.Image, cfg: CompositeConfig, rng: random.Random) -> Image.Image:
    """Random scale (longest side), rotation and brightness; returns RGBA."""
    cut = cut.convert("RGBA")
    target = rng.randint(*cfg.object_size_px)
    scale = target / max(cut.size)
    cut = cut.resize(
        (max(1, round(cut.width * scale)), max(1, round(cut.height * scale))),
        Image.Resampling.LANCZOS,
    )
    cut = cut.rotate(
        rng.uniform(-cfg.max_rotation_deg, cfg.max_rotation_deg),
        resample=Image.Resampling.BICUBIC,
        expand=True,
    )
    rgb, alpha = cut.convert("RGB"), cut.getchannel("A")
    rgb = ImageEnhance.Brightness(rgb).enhance(rng.uniform(*cfg.brightness))
    rgb.putalpha(alpha)
    return rgb


def add_rain(img: np.ndarray, rng: random.Random) -> np.ndarray:
    """Draw thin, slanted, semi-transparent streaks."""
    h, w = img.shape[:2]
    layer = np.zeros_like(img)
    angle = math.radians(rng.uniform(-20, 20))
    length = max(8, h // 40)
    for _ in range(int(w * h / 1500)):
        x, y = rng.randrange(w), rng.randrange(h)
        dx, dy = int(length * math.sin(angle)), int(length * math.cos(angle))
        cv2.line(layer, (x, y), (x + dx, y + dy), (200, 200, 200), 1)
    layer = cv2.blur(layer, (3, 3))
    return cv2.addWeighted(img, 1.0, layer, 0.35, 0)


def degrade(img: np.ndarray, cfg: CompositeConfig, rng: random.Random) -> np.ndarray:
    """Whole-image effects: blur, rain, sensor noise."""
    if rng.random() < cfg.blur_prob:
        img = cv2.GaussianBlur(img, (0, 0), rng.uniform(*cfg.blur_sigma))
    if rng.random() < cfg.rain_prob:
        img = add_rain(img, rng)
    if rng.random() < cfg.noise_prob:
        np_rng = np.random.default_rng(rng.randrange(2**32))
        noise = np_rng.normal(0, rng.uniform(*cfg.noise_sigma), img.shape)
        img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return img


def composite_one(
    background: Image.Image,
    cutouts: list[tuple[int, Path]],
    cfg: CompositeConfig,
    rng: random.Random,
) -> tuple[Image.Image, list[YoloBox]]:
    """Paste a random number of cutouts on `background`; returns the image and YOLO boxes."""
    canvas = background.convert("RGB").copy()
    placed, boxes = [], []
    for _ in range(rng.randint(*cfg.objects_per_image)):
        cls, path = rng.choice(cutouts)
        with Image.open(path) as im:
            cut = transform_cutout(im, cfg, rng)
        mask = cut.getchannel("A").point(lambda a: 255 if a > ALPHA_THRESHOLD else 0)
        tight = mask.getbbox()
        if tight is None or cut.width >= canvas.width or cut.height >= canvas.height:
            continue
        for _attempt in range(10):
            x = rng.randrange(canvas.width - cut.width)
            y = rng.randrange(canvas.height - cut.height)
            box = (x + tight[0], y + tight[1], x + tight[2], y + tight[3])
            if all(_iou(box, p) <= cfg.max_overlap_iou for p in placed):
                break
        else:
            continue  # no free spot found
        canvas.paste(cut, (x, y), cut)
        placed.append(box)
        yb = xyxy_to_yolo(cls, *box, canvas.width, canvas.height)
        if yb:
            boxes.append(yb)
    return canvas, boxes


def load_cutouts(cutouts_dir: Path, registry: ClassRegistry) -> list[tuple[int, Path]]:
    """Find cutouts/<class_name>/*.png; folders not named after a class are skipped."""
    found = []
    for class_dir in sorted(p for p in Path(cutouts_dir).iterdir() if p.is_dir()):
        if class_dir.name not in registry.names:
            print(f"warning: {class_dir} is not a class name in classes.yaml, skipped")
            continue
        cls = registry.id_of(class_dir.name)
        found += [(cls, p) for p in sorted(class_dir.glob("*.png"))]
    return found


def generate(
    backgrounds_dir: Path,
    cutouts_dir: Path,
    out_dir: Path,
    registry: ClassRegistry,
    background_license: str,
    cutout_license: str,
    cfg: CompositeConfig,
    scene_type: str = "flood",
) -> list[ImageRecord]:
    rng = random.Random(cfg.seed)
    backgrounds = sorted(
        p for p in Path(backgrounds_dir).rglob("*") if p.suffix.lower() in IMAGE_EXTS
    )
    cutouts = load_cutouts(Path(cutouts_dir), registry)
    if not backgrounds or not cutouts:
        raise FileNotFoundError(
            f"need images in {backgrounds_dir} and PNG cutouts in {cutouts_dir}/<class_name>/"
        )
    out_dir = Path(out_dir)
    records = []
    for i in tqdm(range(cfg.n_images), desc="composite"):
        bg_path = rng.choice(backgrounds)
        with Image.open(bg_path) as bg:
            bg.thumbnail((cfg.max_background_side, cfg.max_background_side))
            img, boxes = composite_one(bg, cutouts, cfg, rng)
        arr = degrade(np.asarray(img), cfg, rng)
        image_id = f"synthetic_{i:06d}_{safe_id(bg_path.stem)}"
        dst = out_dir / "images" / f"{image_id}.jpg"
        dst.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(arr).save(dst, quality=92)  # fresh image: no EXIF at all
        write_labels(out_dir / "labels" / f"{image_id}.txt", boxes)
        records.append(
            ImageRecord(
                image_id=image_id,
                file_path=dst.relative_to(out_dir).as_posix(),
                source="synthetic",
                source_license=f"background={background_license}; cutouts={cutout_license}",
                modality="rgb",
                synthetic=True,
                scene_type=scene_type,
                sequence_id=f"synthetic-bg-{safe_id(bg_path.stem)}",
            )
        )
    write_metadata(records, out_dir / "metadata.csv")
    return records
