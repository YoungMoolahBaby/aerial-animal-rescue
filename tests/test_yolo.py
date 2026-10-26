import pytest

from aerial_animal_rescue.yolo import (
    label_path_for,
    read_labels,
    write_labels,
    xyxy_to_yolo,
    yolo_to_xyxy,
)


def test_xyxy_to_yolo_basic():
    assert xyxy_to_yolo(3, 10, 20, 30, 60, 100, 200) == pytest.approx((3, 0.2, 0.2, 0.2, 0.2))


def test_box_is_clipped_to_image():
    c, cx, cy, w, h = xyxy_to_yolo(0, -10, -10, 50, 50, 100, 100)
    assert (cx, cy, w, h) == pytest.approx((0.25, 0.25, 0.5, 0.5))


def test_degenerate_or_outside_box_is_dropped():
    assert xyxy_to_yolo(0, 10, 10, 10, 50, 100, 100) is None  # zero width
    assert xyxy_to_yolo(0, 150, 150, 200, 200, 100, 100) is None  # fully outside


def test_swapped_corners_are_fixed():
    assert xyxy_to_yolo(1, 30, 60, 10, 20, 100, 200) == xyxy_to_yolo(1, 10, 20, 30, 60, 100, 200)


def test_roundtrip(tmp_path):
    box = xyxy_to_yolo(2, 5, 6, 25, 46, 64, 48)
    path = tmp_path / "labels" / "a.txt"
    write_labels(path, [box])
    [back] = read_labels(path)
    assert back == pytest.approx(box, abs=1e-6)
    assert yolo_to_xyxy(back, 64, 48) == pytest.approx((5, 6, 25, 46), abs=1e-3)


def test_malformed_label_raises(tmp_path):
    path = tmp_path / "bad.txt"
    path.write_text("1 0.5 0.5 0.1\n")
    with pytest.raises(ValueError):
        read_labels(path)


def test_label_path_for(tmp_path):
    img = tmp_path / "images" / "train" / "x.jpg"
    assert label_path_for(img) == tmp_path / "labels" / "train" / "x.txt"
