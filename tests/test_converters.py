import json

import pytest

from aerial_animal_rescue.classes import ClassMapper
from aerial_animal_rescue.converters import read_coco, read_voc, read_yolo, write_pool
from aerial_animal_rescue.converters.base import (
    SourceImage,
    drop_excluded,
    drop_images_with_classes,
)
from aerial_animal_rescue.converters.birdsai import read_birdsai
from aerial_animal_rescue.converters.hit_uav import read_hit_uav
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


def test_birdsai_frames_count_from_first_image(tmp_path):
    # images start at 87 (as in video 0000000011_0000000000); CSV frames are 0-based
    for n in range(87, 93):
        make_image(tmp_path / "images" / "vid2" / f"vid2_{n:010d}.jpg", seed=n)
    (tmp_path / "ann").mkdir()
    (tmp_path / "ann" / "vid2.csv").write_text(
        "1,1,5,5,10,10,1,0,0,0\n"  # frame 1 -> image 88
        "3,1,6,6,10,10,1,0,0,0\n"  # frame 3 -> image 90
    )
    items = list(read_birdsai(tmp_path / "ann", tmp_path / "images"))
    # only the annotated span 1..3 is kept; frame 2 is a negative inside it
    assert [i.path.name for i in items] == [f"vid2_{n:010d}.jpg" for n in (88, 89, 90)]
    assert [len(i.boxes) for i in items] == [1, 0, 1]


def test_birdsai_clips_of_one_video_share_a_sequence(tmp_path):
    # video 0000000067 comes as many <video>_<clip> folders; they are one flight
    (tmp_path / "ann").mkdir()
    for clip in ("0000000067_0000000005", "0000000067_0000000058", "0000000010_0000000000"):
        make_image(tmp_path / "images" / clip / f"{clip}_0000000000.jpg")
        (tmp_path / "ann" / f"{clip}.csv").write_text("0,1,5,5,10,10,0,4,0,0\n")
    items = list(read_birdsai(tmp_path / "ann", tmp_path / "images"))
    assert sorted(i.sequence_id for i in items) == ["0000000010", "0000000067", "0000000067"]


def test_yolo_reader_finds_cvat_labels_by_name(tmp_path):
    # ICAERUS layout: labels under obj_train_data/, in a differently named flight folder
    make_image(tmp_path / "img" / "flight_042" / "a.jpg", size=(100, 50))
    make_image(tmp_path / "img" / "flight_042" / "b.jpg", size=(100, 50))
    make_image(tmp_path / "img" / "flight_043" / "c.jpg", size=(100, 50))
    lab = tmp_path / "lab" / "flight_031" / "obj_train_data"
    lab.mkdir(parents=True)
    (lab / "a.txt").write_text("1 0.5 0.5 0.2 0.2\n")  # calf
    (lab / "b.txt").write_text("")  # empty pasture
    items = list(
        read_yolo(
            tmp_path / "lab",
            tmp_path / "img",
            class_names=["cow", "calf"],
            sequence_from_folder=True,
        )
    )
    assert [i.path.name for i in items] == ["a.jpg", "b.jpg"]  # c.jpg has no label
    assert items[0].boxes == [("calf", 40.0, 20.0, 60.0, 30.0)]
    assert items[1].boxes == []
    assert {i.sequence_id for i in items} == {"flight_042"}


def test_drop_excluded(tmp_path):
    items = [SourceImage(path=tmp_path / f"{n}.jpg") for n in ("a", "b", "c")]
    assert [i.path.stem for i in drop_excluded(items, ["b"])] == ["a", "c"]
    with pytest.raises(ValueError, match="typo"):
        drop_excluded(items, ["b", "typo"])


def test_hit_uav_reader_sequences_and_altitude(tmp_path):
    # two videos at 60 m / 30 deg (serials jump by > RUN_GAP), one at 100 m / 30 deg
    stems = ["1_60_30_0_00010", "1_60_30_0_00012", "1_60_30_0_04000", "0_100_30_0_00011"]
    for stem in stems:
        make_image(tmp_path / "img" / f"{stem}.jpg", size=(64, 48))
        (tmp_path / "ann").mkdir(exist_ok=True)
        (tmp_path / "ann" / f"{stem}.xml").write_text(
            f"<annotation><filename>{stem}.jpg</filename></annotation>"
        )
    items = {i.path.stem: i for i in read_hit_uav(tmp_path / "ann", tmp_path / "img")}
    seq = {s: i.sequence_id for s, i in items.items()}
    assert seq["1_60_30_0_00010"] == seq["1_60_30_0_00012"] == "t1_60m_30deg_0"
    assert seq["1_60_30_0_04000"] == "t1_60m_30deg_1"
    assert seq["0_100_30_0_00011"] == "t0_100m_30deg_0"
    assert items["0_100_30_0_00011"].altitude_m == 100.0


def test_drop_images_with_classes(tmp_path):
    items = [
        SourceImage(path=tmp_path / "a.jpg", boxes=[("Person", 0, 0, 1, 1)]),
        SourceImage(
            path=tmp_path / "b.jpg", boxes=[("Person", 0, 0, 1, 1), ("DontCare", 0, 0, 1, 1)]
        ),
        SourceImage(path=tmp_path / "c.jpg"),
    ]
    kept, dropped = drop_images_with_classes(items, ["DontCare"])
    assert [i.path.stem for i in kept] == ["a", "c"] and dropped == 1


def test_write_pool_records_per_image_altitude(tmp_path, registry):
    make_image(tmp_path / "src" / "a.png", size=(10, 10))
    item = SourceImage(path=tmp_path / "src" / "a.png", altitude_m=80.0)
    write_pool([item], ClassMapper(registry, {}), tmp_path / "pool", "demo", "CC0-1.0", "thermal")
    assert read_metadata(tmp_path / "pool" / "metadata.csv").loc[0, "altitude_m"] == 80.0
