"""Publish data/processed to a Hugging Face dataset repo, behind license and privacy gates.

Gates (any failure blocks the upload):
  - every image has a real license (not empty / TODO / unknown)
  - no image comes from a source whose `redistribution` is `not_allowed` or `TODO`
  - every image of our own footage has `privacy_reviewed = True`
Non-commercial licenses are allowed but reported, so the dataset card can say so.

The repo id comes from HF_DATASET_REPO and the token from HF_TOKEN; neither is
ever written to disk by this code.
"""

import os
from pathlib import Path

import pandas as pd

from aerial_animal_rescue.metadata import read_metadata
from aerial_animal_rescue.sources import load_sources

OWN_SOURCES = {"own"}
SKIP_SOURCES = {"synthetic"} | OWN_SOURCES  # not in sources.yaml


def check_publishable(meta: pd.DataFrame, sources_yaml: Path | None = None) -> list[str]:
    """Return blocking problems (empty list = OK to publish)."""
    sources = load_sources(sources_yaml) if sources_yaml else load_sources()
    problems = []
    lic = meta["source_license"].fillna("").astype(str)
    bad_lic = lic.str.strip().str.upper().isin({"", "TODO", "UNKNOWN"}) | lic.str.contains("TODO")
    if bad_lic.any():
        problems.append(f"{int(bad_lic.sum())} images without a confirmed license")
    for src in sorted(set(meta["source"]) - SKIP_SOURCES):
        s = sources.get(src)
        if s is None:
            problems.append(f"source {src!r} is not in sources.yaml")
        elif s.redistribution not in {"allowed", "non_commercial"}:
            problems.append(f"source {src!r} has redistribution={s.redistribution!r}")
    unreviewed = meta["source"].isin(OWN_SOURCES) & ~meta["privacy_reviewed"].astype(bool)
    if unreviewed.any():
        problems.append(
            f"{int(unreviewed.sum())} own images not privacy-reviewed "
            "(check for faces / consent, then set privacy_reviewed=True)"
        )
    return problems


def non_commercial_sources(meta: pd.DataFrame) -> list[str]:
    nc = meta["source_license"].astype(str).str.contains("NC", case=True)
    return sorted(set(meta.loc[nc, "source"]))


def upload(data_root: Path, card: Path, private: bool = True, dry_run: bool = False) -> None:
    data_root = Path(data_root)
    meta = read_metadata(data_root / "metadata.csv")
    problems = check_publishable(meta)
    if problems:
        raise SystemExit("upload blocked:\n  " + "\n  ".join(problems))
    nc = non_commercial_sources(meta)
    if nc:
        print(f"note: non-commercial licensed sources included: {nc} - say so in the card")

    repo_id = os.environ.get("HF_DATASET_REPO")
    token = os.environ.get("HF_TOKEN")
    if not repo_id:
        raise SystemExit("set HF_DATASET_REPO, e.g. <your-hf-user>/aerial-animal-rescue")
    print(f"{len(meta)} images -> {repo_id} ({'private' if private else 'PUBLIC'})")
    if dry_run:
        print("dry run: nothing uploaded")
        return
    if not token:
        raise SystemExit("set HF_TOKEN (a write token from huggingface.co/settings/tokens)")

    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(repo_id, repo_type="dataset", private=private, exist_ok=True)
    api.upload_file(
        path_or_fileobj=str(card),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
    )
    license_file = Path("DATA_LICENSE.md")
    if license_file.exists():  # the card's license_link points here
        api.upload_file(
            path_or_fileobj=str(license_file),
            path_in_repo="LICENSE.md",
            repo_id=repo_id,
            repo_type="dataset",
        )
    api.upload_large_folder(
        folder_path=str(data_root),
        repo_id=repo_id,
        repo_type="dataset",
        allow_patterns=["images/**", "labels/**", "metadata.csv"],  # not data.yaml: local path
    )
    print(f"done: https://huggingface.co/datasets/{repo_id}")
