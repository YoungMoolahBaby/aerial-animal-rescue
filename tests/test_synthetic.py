import numpy as np
from PIL import Image

from aerial_animal_rescue.metadata import read_metadata
from aerial_animal_rescue.synthetic import CompositeConfig, generate
from aerial_animal_rescue.yolo import read_labels
from tests.conftest import make_image


def test_generate_composites(tmp_path, registry):
    make_image(tmp_path / "bg" / "flood1.jpg", size=(320, 240))
    cut = np.zeros((40, 30, 4), dtype=np.uint8)
    cut[5:35, 5:25] = (180, 120, 60, 255)  # opaque body, transparent margin
    (tmp_path / "cutouts" / "dog").mkdir(parents=True)
    Image.fromarray(cut, "RGBA").save(tmp_path / "cutouts" / "dog" / "d1.png")

    cfg = CompositeConfig(n_images=5, objects_per_image=(1, 3), object_size_px=(20, 40))
    out = tmp_path / "out"
    records = generate(
        tmp_path / "bg", tmp_path / "cutouts", out, registry, "CC-BY-4.0", "CC-BY-4.0", cfg
    )

    assert len(records) == 5
    meta = read_metadata(out / "metadata.csv")
    assert meta["synthetic"].all()
    assert set(meta["sequence_id"]) == {"synthetic-bg-flood1"}
    n_boxes = 0
    for r in records:
        for c, cx, cy, w, h in read_labels(out / "labels" / f"{r.image_id}.txt"):
            n_boxes += 1
            assert c == registry.id_of("dog")
            assert cx - w / 2 >= 0 and cx + w / 2 <= 1 and cy - h / 2 >= 0 and cy + h / 2 <= 1
    assert n_boxes >= 5
