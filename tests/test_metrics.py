import math

import numpy as np
import pytest

from aerial_animal_rescue.metrics import ImageResult, evaluate, operating_point


def result(gt, pred):
    return ImageResult(np.array(gt, float).reshape(-1, 5), np.array(pred, float).reshape(-1, 6))


def test_perfect_predictions():
    r = result(
        [[1, 0, 0, 10, 10], [2, 20, 20, 60, 60]], [[1, 0, 0, 10, 10, 0.9], [2, 20, 20, 60, 60, 0.8]]
    )
    res = evaluate([r], num_classes=3)
    assert res["map50"] == pytest.approx(1.0)
    assert res["map50_95"] == pytest.approx(1.0)
    assert math.isnan(res["per_class"][0]["ap50"])  # no ground truth for class 0


def test_no_predictions_is_zero():
    assert evaluate([result([[0, 0, 0, 10, 10]], [])], 1)["map50"] == 0.0


def test_wrong_class_is_a_miss():
    assert evaluate([result([[0, 0, 0, 10, 10]], [[1, 0, 0, 10, 10, 0.9]])], 2)["map50"] == 0.0


def test_size_range_ignores_other_sizes():
    # one small (10x10) object found, one large (100x100) object missed
    r = result([[0, 0, 0, 10, 10], [0, 200, 200, 300, 300]], [[0, 0, 0, 10, 10, 0.9]])
    assert evaluate([r], 1)["map50"] == pytest.approx(0.5, abs=0.02)
    assert evaluate([r], 1, area_range=(0, 32**2))["map50"] == pytest.approx(1.0)
    assert evaluate([r], 1, area_range=(96**2, 1e9))["map50"] == 0.0


def test_operating_point_counts_false_alarms_on_empty_images():
    found = result([[0, 0, 0, 10, 10]], [[0, 0, 0, 10, 10, 0.9], [0, 50, 50, 60, 60, 0.3]])
    missed = result([[0, 0, 0, 10, 10]], [])
    empty_two_alarms = result([], [[0, 0, 0, 10, 10, 0.8], [0, 30, 30, 40, 40, 0.6]])
    empty_quiet = result([], [[0, 0, 0, 10, 10, 0.1]])  # below the threshold
    rs = [found, missed, empty_two_alarms, empty_quiet]
    op = operating_point(rs, 1, conf=0.25)
    assert op["recall"] == pytest.approx(0.5)  # 1 of 2 objects
    assert op["precision"] == pytest.approx(1 / 4)  # 1 hit, 3 false alarms
    assert op["false_alarms_per_empty_image"] == pytest.approx(1.0)  # 2 alarms / 2 empty
    assert op["empty_images_with_false_alarm"] == pytest.approx(0.5)
    strict = operating_point(rs, 1, conf=0.7)
    assert strict["precision"] == pytest.approx(1 / 2)  # the 0.9 hit and the 0.8 alarm
    assert strict["false_alarms_per_empty_image"] == pytest.approx(0.5)
