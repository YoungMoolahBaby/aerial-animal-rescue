"""Upload data/processed + metadata + DATASET_CARD.md to a Hugging Face dataset repo.

Needs env vars HF_DATASET_REPO (e.g. your-user/aerial-animal-rescue) and HF_TOKEN.
Blocks the upload if any image lacks a confirmed license, comes from a source
that may not be redistributed, or is our own footage not yet privacy-reviewed.

    uv run python scripts/upload_to_hf.py --dry-run
    uv run python scripts/upload_to_hf.py            # private repo
    uv run python scripts/upload_to_hf.py --public
"""

import argparse
from pathlib import Path

from aerial_animal_rescue.config import PROCESSED_DIR
from aerial_animal_rescue.hf_upload import upload


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--data", type=Path, default=PROCESSED_DIR)
    p.add_argument("--card", type=Path, default=Path("DATASET_CARD.md"))
    p.add_argument("--public", action="store_true", help="create the repo as public")
    p.add_argument("--dry-run", action="store_true", help="run the checks, upload nothing")
    args = p.parse_args()
    upload(args.data, args.card, private=not args.public, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
