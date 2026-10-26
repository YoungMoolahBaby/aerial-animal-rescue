import math

import numpy as np
import pytest

from aerial_animal_rescue.metrics import ImageResult, evaluate


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
