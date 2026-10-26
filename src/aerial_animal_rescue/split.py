"""Grouped train/val/test splitting and assembly of data/processed/.

Leakage rule: all frames of one flight / video sequence land in the same
split, because neighbouring frames are near-duplicates. Images without a
sequence id are grouped by their source (`group_by="source"`) or treated as
singletons (`group_by="sequence"`, the default, prints a warning count).

Test rule: the test set is real images only. Synthetic images always go to
train, so validation and test numbers measure real-world performance.

Empty images (no boxes) can be thinned out of train only: val and test keep
all of them, since false alarms on empty frames are part of what we measure.
"""

import random
import shutil
from itertools import zip_longest
from pathlib import Path

import pandas as pd

from aerial_animal_rescue.metadata import read_metadata, write_metadata
from aerial_animal_rescue.yolo import label_path_for

SPLIT_NAMES = ("train", "val", "test")


def group_keys(df: pd.DataFrame, group_by: str = "sequence", warn: bool = True) -> pd.Series:
    """Grouping key per row: `source/sequence_id`, or the fallback described above."""
    if group_by == "source":
        return df["source"].astype(str)
    if group_by != "sequence":
        raise ValueError(f"group_by must be 'sequence' or 'source', not {group_by!r}")
    has_seq = df["sequence_id"].notna() & (df["sequence_id"].astype(str) != "")
    keys = df["source"].astype(str) + "/" + df["sequence_id"].astype(str)
    singles = df["source"].astype(str) + "/image/" + df["image_id"].astype(str)
    missing = int((~has_seq).sum())
    if missing and warn:
        print(f"warning: {missing} images have no sequence_id and are split one by one")
    return keys.where(has_seq, singles)


def assign_splits(
    df: pd.DataFrame,
    ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
    seed: int = 0,
    group_by: str = "sequence",
    weights: pd.Series | None = None,
) -> pd.Series:
    """Return a split name per row. Whole groups go to one split.

    Each real group is given to the split currently furthest below its target
    share, largest groups first (ties shuffled by `seed`), which balances
    uneven groups well. A group's size is the sum of `weights` over its rows
    (default 1 per image; build_processed counts an empty image as 1/100, so a
    long flight over empty pasture doesn't count as big). With
    `group_by="sequence"` this runs per source, so every source gets roughly
    `ratios` instead of a big source swamping a small one.
    """
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError(f"ratios must sum to 1, got {ratios}")
    keys = group_keys(df, group_by)
    synthetic = df["synthetic"].astype(bool)
    result = pd.Series("train", index=df.index, dtype=object)

    real = ~synthetic
    weights = pd.Series(1, index=df.index) if weights is None else weights
    strata = df["source"].astype(str) if group_by == "sequence" else pd.Series("", index=df.index)
    group_split = {}
    for stratum in sorted(strata[real].unique()):
        rows = real & (strata == stratum)
        sizes = weights[rows].groupby(keys[rows]).sum().sort_index()
        order = list(sizes.index)
        random.Random(f"{seed}/{stratum}").shuffle(order)
        order.sort(key=lambda g: -sizes[g])  # stable: equal sizes keep the shuffled order
        total = float(sizes.sum())
        counts = dict.fromkeys(SPLIT_NAMES, 0)
        for g in order:
            split = max(SPLIT_NAMES, key=lambda s: ratios[SPLIT_NAMES.index(s)] * total - counts[s])
            group_split[g] = split
            counts[split] += float(sizes[g])
    result[real] = keys[real].map(group_split)

    mixed = set(keys[synthetic]) & set(group_split)
    if mixed:  # a group with both synthetic and real images: keep it out of val/test
        result[keys.isin(mixed)] = "train"
    return result


def sample_train_empty(
    df: pd.DataFrame,
    empty: pd.Series,
    frac: float,
    seed: int = 0,
    group_by: str = "sequence",
) -> pd.Series:
    """Rows to keep so empty images are at most `frac` of train; other splits untouched.

    Empty train images are taken round-robin across groups (flights), each group
    shuffled by `seed`, so the kept backgrounds come from as many flights as possible.
    """
    if not 0 <= frac < 1:
        raise ValueError(f"frac must be in [0, 1), got {frac}")
    train = df["split"] == "train"
    candidates = train & empty
    n_keep = int(frac * int((train & ~empty).sum()) / (1 - frac))
    keys = group_keys(df[candidates], group_by, warn=False)
    rng = random.Random(seed)
    groups = [list(idx) for _, idx in sorted(keys.groupby(keys).groups.items())]
    rng.shuffle(groups)
    for g in groups:
        rng.shuffle(g)
    order = [i for rnd in zip_longest(*groups) for i in rnd if i is not None]
    keep = ~candidates
    keep.loc[order[:n_keep]] = True
    return keep


def build_processed(
    pools: list[Path],
    out_dir: Path,
    class_names: list[str],
    ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
    seed: int = 0,
    group_by: str = "sequence",
    overwrite: bool = False,
    train_empty_frac: float | None = None,
) -> pd.DataFrame:
    """Merge pools (each with metadata.csv, images/, labels/), split, copy into `out_dir`.

    Images without a label file are not annotated yet and are skipped. With
    `train_empty_frac`, empty images are sampled down to that share of train.
    """
    out_dir = Path(out_dir)
    for sub in ("images", "labels"):
        if (out_dir / sub).exists():
            if not overwrite:
                raise FileExistsError(f"{out_dir / sub} exists; pass overwrite=True to rebuild")
            shutil.rmtree(out_dir / sub)

    frames = []
    for pool in pools:
        pool = Path(pool)
        df = read_metadata(pool / "metadata.csv")
        df["src_path"] = [pool / p for p in df["file_path"]]
        labeled = df["src_path"].map(lambda p: label_path_for(p).exists())
        if (~labeled).any():
            print(f"{pool}: skipping {int((~labeled).sum())} unlabeled images")
        frames.append(df[labeled])
    df = pd.concat(frames, ignore_index=True)
    dupes = df["image_id"][df["image_id"].duplicated()]
    if not dupes.empty:
        raise ValueError(f"duplicate image_ids across pools, e.g. {dupes.iloc[0]!r}")

    empty = df["src_path"].map(lambda p: not label_path_for(p).read_text().strip())
    df["split"] = assign_splits(
        df, ratios, seed, group_by, weights=empty.map({True: 0.01, False: 1.0})
    )
    if train_empty_frac is not None:
        keep = sample_train_empty(df, empty, train_empty_frac, seed, group_by)
        print(
            f"train: kept {int((keep & empty & (df['split'] == 'train')).sum())} of "
            f"{int((empty & (df['split'] == 'train')).sum())} empty images"
        )
        df = df[keep].reset_index(drop=True)
    new_paths = []
    for row in df.itertuples():
        rel = Path("images") / row.split / f"{row.image_id}{Path(row.src_path).suffix}"
        (out_dir / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(row.src_path, out_dir / rel)
        label_dst = label_path_for(out_dir / rel)
        label_dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(label_path_for(row.src_path), label_dst)
        new_paths.append(rel.as_posix())
    df["file_path"] = new_paths
    df = df.drop(columns="src_path")
    write_metadata(df, out_dir / "metadata.csv")
    write_data_yaml(out_dir, class_names)
    print(df.groupby(["split", "synthetic"]).size().rename("images").to_string())
    return df


def write_data_yaml(root: Path, class_names: list[str]) -> Path:
    """Ultralytics dataset file (absolute `path` so it works from any cwd)."""
    lines = [
        f"path: {Path(root).resolve().as_posix()}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "names:",
        *[f"  {i}: {n}" for i, n in enumerate(class_names)],
    ]
    path = Path(root) / "data.yaml"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
