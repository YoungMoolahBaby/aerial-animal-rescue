import numpy as np
import pandas as pd
import pytest

from aerial_animal_rescue.evaluate import by_sequence, format_report
from aerial_animal_rescue.metrics import ImageResult


def result(gt, pred):
    return ImageResult(np.array(gt, float).reshape(-1, 5), np.array(pred, float).reshape(-1, 6))


def test_by_sequence_splits_results_per_flight():
    box = [0, 0, 0, 10, 10]
    rs = [
        result([box], [[*box, 0.9]]),  # flight a: 1 person, found
        result([box, [0, 20, 20, 30, 30]], []),  # flight b: 2 people, missed
        result([box], []),  # flight b
        result([], [[0, 0, 0, 10, 10, 0.8]]),  # flight b: empty frame with an alarm
    ]
    meta = pd.DataFrame({"source": ["s"] * 4, "sequence_id": ["a", "b", "b", "b"]})
    seqs = by_sequence(rs, meta, ["person"], conf=0.25)
    assert list(seqs) == ["s/b", "s/a"]  # most boxes first
    b, a = seqs["s/b"], seqs["s/a"]
    assert (b["images"], b["n_gt"], b["recall"]) == (3, 3, 0.0)
    assert b["per_class"]["person"]["share"] == pytest.approx(0.75)
    assert b["false_alarms_per_empty_image"] == pytest.approx(1.0)
    assert a["recall"] == 1.0 and a["map50"] == pytest.approx(1.0)
    assert a["empty_images"] == 0

    text = format_report(
        {
            "split": "val",
            "images": 4,
            "overall": a,
            "per_class": {},
            "by_modality": {},
            "by_synthetic": {},
            "by_size": {},
            "by_sequence": seqs,
        }
    )
    assert "s/b" in text and "75%" in text and "no empty images" in text
