import numpy as np
import pytest

from aerial_animal_rescue.domain_stats import frame_noise, measure_pool, object_contrast, summarize
from tests.conftest import make_pool


def test_object_contrast_and_noise():
    gray = np.full((40, 40), 60, np.uint8)
    gray[18:22, 18:22] = 160  # a 4 px warm blob, 100 above flat ground
    diff, z = object_contrast(gray, (18, 18, 22, 22))
    assert diff == pytest.approx(100.0)
    assert z > 1000  # flat ring: stands out by any measure
    assert frame_noise(gray) == 0.0
    stripes = np.tile(np.array([0, 10], np.uint8), (8, 4))
    assert frame_noise(stripes) == 10.0


def test_measure_and_summarize_a_pool(tmp_path, registry):
    pool = make_pool(
        tmp_path / "pool", [{"image_id": f"a{i}", "sequence_id": "s"} for i in range(3)]
    )
    images, boxes = measure_pool(pool, list(registry.names))
    assert len(images) == 3 and len(boxes) == 3
    assert set(boxes["class"]) == {registry.names[1]}
    table = summarize(images, boxes)
    row = table.iloc[0]
    assert (row["source"], row["images"], row["frame"], row["empty"]) == ("test", 3, "64x48", 0.0)
    assert row["side"] == pytest.approx(np.sqrt(12.8 * 9.6), rel=0.01)  # 20% of 64x48
    assert len(measure_pool(pool, list(registry.names), sample=2)[0]) == 2
