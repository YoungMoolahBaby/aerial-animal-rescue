import pandas as pd

from aerial_animal_rescue.metadata import read_metadata
from aerial_animal_rescue.split import (
    assign_splits,
    build_processed,
    group_keys,
    sample_train_empty,
)
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


def test_sample_train_empty_only_thins_train():
    df = frame(n_flights=20, frames_per_flight=5, n_synthetic=0)
    df["split"] = assign_splits(df, seed=0)
    empty = pd.Series([k % 5 != 0 for k in range(len(df))], index=df.index)  # 80% empty
    keep = sample_train_empty(df, empty, frac=0.1, seed=0)
    train = df["split"] == "train"
    kept_empty = int((keep & train & empty).sum())
    n_pos = int((train & ~empty).sum())
    assert kept_empty == int(0.1 * n_pos / 0.9)  # empty share of train <= 10%
    assert keep[~train].all() and keep[~empty].all()  # val/test and positives untouched
    # spread: the kept backgrounds come from distinct flights
    assert df[keep & train & empty]["sequence_id"].nunique() == kept_empty
    assert keep.equals(sample_train_empty(df, empty, frac=0.1, seed=0))


def test_build_processed_samples_empty_train_images(tmp_path, registry):
    rows = [
        {"image_id": f"own_{f}_{k}", "sequence_id": f"fl{f}"} for f in range(10) for k in range(4)
    ]
    pool = make_pool(tmp_path / "pool", rows)
    for f in range(10):  # frames 1..3 of every flight are empty pasture
        for k in (1, 2, 3):
            (pool / "labels" / f"own_{f}_{k}.txt").write_text("")
    df = build_processed([pool], tmp_path / "out", list(registry.names), train_empty_frac=0.0)
    assert set(df[df["split"] == "train"]["image_id"].str[-1]) == {"0"}
    assert (df[df["split"] != "train"]["image_id"].str[-1] != "0").any()  # val/test keep them


def test_balances_every_weight_column_not_just_images():
    # 20 flights of 5 images each; flights 0-7 hold all the tiny animals, flights 8-19
    # the large ones. Balancing images alone can put every tiny-animal flight in train.
    df = frame(n_flights=20, frames_per_flight=5, n_synthetic=0)
    flight = df["sequence_id"].str[6:].astype(int)
    weights = pd.DataFrame(
        {
            "images": 1.0,
            "tiny": (flight < 8) * 10.0,
            "large": (flight >= 8) * 10.0,
        }
    )
    split = assign_splits(df, seed=0, weights=weights)
    tiny_splits = set(split[flight < 8])
    assert {"val", "test"} <= tiny_splits  # tiny animals reach val and test too
    for col in ("tiny", "large"):
        share = weights[col].groupby(split).sum() / weights[col].sum()
        assert 0.5 <= share["train"] <= 0.8, (col, share.to_dict())


def test_single_column_weights_match_series():
    df = frame(n_flights=30, n_synthetic=0)
    w = pd.Series(range(len(df)), index=df.index, dtype=float)
    assert assign_splits(df, seed=2, weights=w).equals(
        assign_splits(df, seed=2, weights=w.to_frame("images"))
    )


def test_balance_columns_count_boxes_by_class_and_size(tmp_path):
    from PIL import Image

    from aerial_animal_rescue.split import balance_columns

    (tmp_path / "images").mkdir()
    (tmp_path / "labels").mkdir()
    for name, labels in (("a", "0 0.5 0.5 0.1 0.1\n6 0.5 0.5 0.5 0.5\n"), ("b", "")):
        Image.new("L", (100, 100)).save(tmp_path / "images" / f"{name}.jpg")
        (tmp_path / "labels" / f"{name}.txt").write_text(labels)
    cols = balance_columns(pd.Series([tmp_path / "images" / f"{n}.jpg" for n in "ab"]))
    assert cols["with_boxes"].tolist() == [1, 0]
    assert cols["empty"].tolist() == [0, 1]
    assert cols["0/tiny (<16px)"].tolist() == [1, 0]  # 10x10 px person
    assert cols["6/medium (32-96px)"].tolist() == [1, 0]  # 50x50 px animal


def test_each_source_is_split_on_its_own():
    big = frame(n_flights=60, frames_per_flight=10, n_synthetic=0)
    small = frame(n_flights=20, frames_per_flight=5, n_synthetic=0).assign(source="small")
    df = pd.concat([big, small], ignore_index=True)
    split = assign_splits(df, seed=0)
    for source in ("own", "small"):
        counts = split[df["source"] == source].value_counts(normalize=True)
        assert set(counts.index) == {"train", "val", "test"}
        assert 0.6 <= counts["train"] <= 0.8, (source, counts.to_dict())


def test_train_only_source_leaves_other_splits_unchanged():
    base = frame(n_flights=20, frames_per_flight=5, n_synthetic=0)
    extra = frame(n_flights=10, frames_per_flight=5, n_synthetic=0).assign(source="extra")
    extra["image_id"] = "x" + extra["image_id"]
    df = pd.concat([base, extra], ignore_index=True)
    split = assign_splits(df, seed=0, train_only=["extra"])
    assert (split[df["source"] == "extra"] == "train").all()
    assert split[df["source"] == "own"].tolist() == assign_splits(base, seed=0).tolist()


def test_train_only_source_is_kept_whole_and_not_counted_for_empties():
    base = frame(n_flights=20, frames_per_flight=5, n_synthetic=0)
    extra = frame(n_flights=10, frames_per_flight=5, n_synthetic=0).assign(source="extra")
    extra["image_id"] = "x" + extra["image_id"]
    df = pd.concat([base, extra], ignore_index=True)
    df["split"] = assign_splits(df, seed=0, train_only=["extra"])
    empty = pd.Series([k % 5 != 0 for k in range(len(df))], index=df.index)
    keep = sample_train_empty(df, empty, frac=0.1, seed=0, train_only=["extra"])
    assert keep[df["source"] == "extra"].all()
    alone = base.assign(split=assign_splits(base, seed=0))
    own = (df["source"] == "own").to_numpy()
    expected = sample_train_empty(alone, empty[own].reset_index(drop=True), frac=0.1, seed=0)
    assert keep[own].tolist() == expected.tolist()


def test_synthetic_images_are_kept_whole_and_not_counted_for_empties():
    base = frame(n_flights=20, frames_per_flight=5, n_synthetic=0)
    df = frame(n_flights=20, frames_per_flight=5, n_synthetic=30)
    df["split"] = assign_splits(df, seed=0)
    empty = pd.Series([k % 5 != 0 for k in range(len(df))], index=df.index)
    keep = sample_train_empty(df, empty, frac=0.1, seed=0)
    syn = df["synthetic"].to_numpy()
    assert keep[syn].all()
    alone = base.assign(split=assign_splits(base, seed=0))
    expected = sample_train_empty(alone, empty[~syn].reset_index(drop=True), frac=0.1, seed=0)
    assert keep[~syn].tolist() == expected.tolist()
