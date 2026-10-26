import pandas as pd

from aerial_animal_rescue.metadata import read_metadata
from aerial_animal_rescue.split import assign_splits, build_processed, group_keys
from tests.conftest import make_pool


def frame(n_flights=20, frames_per_flight=5, n_synthetic=10) -> pd.DataFrame:
    rows = []
    for f in range(n_flights):
        for k in range(frames_per_flight):
            rows.append(
                {
                    "image_id": f"f{f}_{k}",
                    "source": "own",
                    "sequence_id": f"flight{f}",
                    "synthetic": False,
                }
            )
    for s in range(n_synthetic):
        rows.append(
            {
                "image_id": f"syn{s}",
                "source": "synthetic",
                "sequence_id": f"bg{s % 3}",
                "synthetic": True,
            }
        )
    return pd.DataFrame(rows)


def test_no_group_spans_two_splits():
    df = frame()
    df["split"] = assign_splits(df, seed=1)
    per_group = df.assign(g=group_keys(df)).groupby("g")["split"].nunique()
    assert (per_group == 1).all()


def test_all_three_splits_used_and_roughly_proportional():
    df = frame(n_flights=40)
    split = assign_splits(df, ratios=(0.7, 0.15, 0.15), seed=0)
    counts = split[~df["synthetic"]].value_counts()
    assert set(counts.index) == {"train", "val", "test"}
    assert 120 <= counts["train"] <= 160  # 70% of 200 = 140, give or take a flight


def test_test_set_is_real_only():
    df = frame()
    split = assign_splits(df, seed=3)
    assert (split[df["synthetic"]] == "train").all()


def test_deterministic():
    df = frame()
    assert assign_splits(df, seed=7).equals(assign_splits(df, seed=7))


def test_missing_sequence_falls_back_to_single_images():
    df = pd.DataFrame(
        {
            "image_id": ["a", "b"],
            "source": ["x", "x"],
            "sequence_id": [None, None],
            "synthetic": [False, False],
        }
    )
    assert group_keys(df).nunique() == 2
    assert group_keys(df, "source").nunique() == 1


def test_build_processed(tmp_path, registry):
    rows = [
        {"image_id": f"own_{f}_{k}", "sequence_id": f"fl{f}"} for f in range(6) for k in range(3)
    ]
    pool = make_pool(tmp_path / "pool", rows)
    (pool / "labels" / "own_0_0.txt").unlink()  # not annotated yet -> skipped
    out = tmp_path / "processed"
    df = build_processed([pool], out, list(registry.names), seed=0)
    assert len(df) == 17
    meta = read_metadata(out / "metadata.csv")
    for row in meta.itertuples():
        assert (out / row.file_path).exists()
        assert row.file_path.startswith(f"images/{row.split}/")
    assert (out / "data.yaml").read_text().count(":") >= 7
