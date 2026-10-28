import numpy as np

from aerial_animal_rescue.zero_shot import found, size_bucket, summarize


def test_found_is_class_agnostic_and_thresholded():
    gt = np.array([[0, 0, 10, 10], [50, 50, 60, 60]], dtype=float)
    pred = np.array([[1, 1, 11, 11]], dtype=float)  # overlaps the first box only
    assert found(gt, pred, 0.5).tolist() == [True, False]
    assert found(gt, np.zeros((0, 4)), 0.5).tolist() == [False, False]


def test_size_buckets_by_area():
    assert size_bucket(10 * 10) == "tiny (<16px)"
    assert size_bucket(20 * 20) == "small (16-32px)"
    assert size_bucket(100 * 100) == "large (>96px)"


def test_summarize_counts_found_and_alarms():
    gts = [
        np.array([[0, 0, 0, 10, 10], [6, 100, 100, 140, 140]], dtype=float),
        np.zeros((0, 5)),
        np.zeros((0, 5)),
    ]
    preds = [
        # class ignored: a "cattle" box on a person still counts as found
        np.array([[4, 0, 0, 10, 10, 0.9], [0, 100, 100, 140, 140, 0.1]], dtype=float),
        np.array([[0, 5, 5, 9, 9, 0.8], [0, 20, 20, 30, 30, 0.3]], dtype=float),
        np.zeros((0, 6)),
    ]
    res = summarize(gts, preds, conf=0.25)
    assert res["by_size"]["tiny (<16px)"] == {"boxes": 1, "found_iou50": 1.0, "found_iou10": 1.0}
    assert res["by_size"]["medium (32-96px)"]["found_iou50"] == 0.0  # below conf
    assert res["empty_images"] == 2
    assert res["alarms_per_empty_image"] == 1.0
    assert res["by_class"][0] == {"tiny (<16px)": {"boxes": 1, "found_iou50": 1.0}}
    assert res["by_class"][6]["medium (32-96px)"]["found_iou50"] == 0.0
