import json

import pytest

from aerial_animal_rescue.classes import ClassMapper
from aerial_animal_rescue.converters import read_coco, read_voc, write_pool
from aerial_animal_rescue.converters.birdsai import read_birdsai
from aerial_animal_rescue.metadata import read_metadata
from aerial_animal_rescue.yolo import read_labels
from tests.conftest import make_image


def test_coco_to_yolo_pool(tmp_path, registry):
    src = tmp_path / "src"
    make_image(src / "a.png", size=(100, 50))
    coco = {
        "images": [{"id": 1, "file_name": "a.png", "width": 100, "height": 50}],
        # HIT-UAV style key "annotation" must work too
        "annotation": [
            {"image_id": 1, "category_id": 7, "bbox": [10, 10, 20, 10]},
            {"image_id": 1, "category_id": 8, "bbox": [50, 20, 10, 10]},
        ],
        "categories": [{"id": 7, "name": "Cow"}, {"id": 8, "name": "Tractor"}],
    }
    (src / "ann.json").write_text(json.dumps(coco))
    mapper = ClassMapper(registry, {"Cow": "cattle"})
    out = tmp_path / "pool"
    write_pool(
        read_coco(src / "ann.json", src), mapper, out, "demo", "CC-BY-4.0", "rgb", image_root=src
    )

    [box] = read_labels(out / "labels" / "demo_a.txt")
    assert box == pytest.approx((4, 0.2, 0.3, 0.2, 0.2))  # cattle; Tractor dropped
    assert mapper.dropped == {"Tractor": 1}
    meta = read_metadata(out / "metadata.csv")
    assert list(meta["image_id"]) == ["demo_a"]
    assert meta.loc[0, "source_license"] == "CC-BY-4.0"
    assert "Tractor: 1" in (out / "conversion_log.txt").read_text()


def test_voc_reader(tmp_path):
    make_image(tmp_path / "img" / "x.jpg", size=(64, 48))
    (tmp_path / "ann").mkdir()
    (tmp_path / "ann" / "x.xml").write_text(
        "<annotation><filename>x.jpg</filename><size><width>64</width><height>48</height></size>"
        "<object><name>Person</name><bndbox><xmin>1</xmin><ymin>2</ymin><xmax>11</xmax>"
        "<ymax>22</ymax></bndbox></object></annotation>"
    )
    [item] = read_voc(tmp_path / "ann", tmp_path / "img")
    assert item.boxes == [("Person", 1.0, 2.0, 11.0, 22.0)]
    assert (item.width, item.height) == (64, 48)


def test_birdsai_reader(tmp_path):
    for n in range(4):
        make_image(tmp_path / "images" / "vid1" / f"vid1_{n:05d}.jpg", seed=n)
    (tmp_path / "ann").mkdir()
    (tmp_path / "ann" / "vid1.csv").write_text(
        "0,1,5,5,10,10,0,4,0,0\n"  # frame 0: dog
        "2,2,1,1,4,4,1,0,0,0\n"  # frame 2: human
    )
    items = list(read_birdsai(tmp_path / "ann", tmp_path / "images", frame_step=2))
    assert [i.path.name for i in items] == ["vid1_00000.jpg", "vid1_00002.jpg"]
    assert items[0].boxes == [("dog", 5.0, 5.0, 15.0, 15.0)]
    assert items[1].boxes[0][0] == "human"
    assert {i.sequence_id for i in items} == {"vid1"}
