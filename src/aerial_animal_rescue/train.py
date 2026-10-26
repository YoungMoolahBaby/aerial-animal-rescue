"""Fine-tune an Ultralytics YOLO model on data/processed."""

from pathlib import Path

from aerial_animal_rescue.config import RUNS_DIR
from aerial_animal_rescue.split import write_data_yaml


def train(
    data_root: Path,
    class_names: list[str],
    model: str = "yolo11n.pt",
    epochs: int = 100,
    imgsz: int = 1280,
    batch: int = 8,
    device: str | None = None,
    name: str = "baseline",
    **overrides,
):
    """Train and return the Ultralytics results object.

    imgsz defaults to 1280: aerial animals are small, and 640 throws most of them away.
    The data.yaml is regenerated each time so it always matches classes.yaml.
    """
    from ultralytics import YOLO  # heavy import: keep it out of module import time

    data_yaml = write_data_yaml(Path(data_root), class_names)
    yolo = YOLO(model)  # a .pt name downloads pretrained COCO weights on first use
    return yolo.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=str(RUNS_DIR.resolve() / "train"),
        name=name,
        **overrides,
    )
