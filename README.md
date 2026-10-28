# aerial-animal-rescue

An open dataset and baseline detector for **finding animals (pets and livestock) in flood and
disaster scenes from drone imagery**, in both RGB and thermal.

> Status: early setup. Tooling is in place; data collection has not started.

## Why

When floods and debris flows hit, people are evacuated, but animals are often left behind:
dogs on roofs, horses stranded in paddocks, cattle cut off by water. Drones are already
flown over these scenes, but there is **no public aerial dataset for detecting animals in
disaster conditions**. Existing aerial datasets cover wildlife on savannahs, livestock on
clear pastures, or flood damage without any animals. This project wants to close that gap
with:

1. **A dataset**: RGB and thermal drone images of animals in flood, debris-flow and pasture
   scenes. Our own footage, compatible third-party datasets, and synthetic composites, all
   labeled in one schema. It will be hosted on Hugging Face.
2. **A baseline model**: a fine-tuned YOLO detector, with honest evaluation broken out by
   class, modality, real vs synthetic, and object size.

Classes: `person`, `dog`, `cat`, `horse`, `cattle`, `small_livestock` (goats and sheep), and
`other_animal`. See [configs/classes.yaml](configs/classes.yaml).

## Quickstart

Requires [uv](https://docs.astral.sh/uv/) and Python 3.11+.

```bash
git clone https://github.com/YoungMoolahBaby/aerial-animal-rescue.git
cd aerial-animal-rescue
uv sync                     # creates .venv with all dependencies
uv run pytest               # tests run on tiny fake data, no downloads
```

### The pipeline

Every step is a thin script around the `aerial_animal_rescue` package. Run them from the repo
root.

```bash
# 1. External data: look at the registry, then fetch (asks before anything > 1 GB)
uv run python scripts/download_external.py --list
uv run python scripts/download_external.py hit_uav
#    unzip into data/external/<source>/, set the paths in configs/sources.yaml, then:
uv run python scripts/convert_external.py hit_uav          # -> data/interim/hit_uav/

# 2. Your own footage (one folder per flight under data/raw/); GPS is stripped on ingest
uv run python scripts/ingest_raw.py data/raw/<flight> --modality rgb --scene-type flood --interval 2
#    annotate the frames (e.g. CVAT / Label Studio, YOLO export) into data/interim/own/labels/

# 3. Synthetic composites (cutouts in data/synthetic/cutouts/<class>/*.png)
uv run python scripts/composite_synthetic.py --n 500 --backgrounds data/synthetic/backgrounds \
    --background-license CDLA-Permissive-1.0 --cutout-license CC-BY-4.0

# 4. Split (grouped by flight/sequence, test set real-only) and validate
uv run python scripts/split_dataset.py
uv run python scripts/validate_dataset.py

# 5. Baseline
uv run python scripts/train_baseline.py --model yolo11s.pt --imgsz 1280 --device 0
uv run python scripts/evaluate.py runs/train/baseline/weights/best.pt --split test

# 6. Publish (license + privacy gates run first; private repo unless --public)
HF_DATASET_REPO=<hf-user>/aerial-animal-rescue HF_TOKEN=... uv run python scripts/upload_to_hf.py --dry-run
```

**GPU training:** on Windows, `uv sync` installs the CUDA 13.0 build of PyTorch from the
PyTorch index (see `[tool.uv.sources]` in `pyproject.toml`); Linux gets CUDA builds from PyPI.
For reference, on an RTX 4070 Ti (12 GB) with the command above (yolo11s, 1280 px, batch 8),
one epoch over the 2026-10-26 build (6,034 train images) took about 2 min 50 s of training
plus 20 s of validation, using 9 GB of VRAM, so 100 epochs take roughly 5-7 hours.
`--workers` defaults to 4: with Ultralytics' 8, validation ran out of Windows page file on
a 32 GB machine.

### Data layout

```
data/
  raw/          your original footage, never modified, never published
  external/     third-party downloads, one folder per source
  interim/      converted or ingested "pools": images/, labels/, metadata.csv
  synthetic/    cutouts/ + backgrounds/ (inputs); images/, labels/, metadata.csv (outputs)
  processed/    the final dataset: images/{train,val,test}, labels/..., metadata.csv, data.yaml
```

Every image has one metadata row: `image_id, file_path, source, source_license, modality,
synthetic, altitude_m, weather, scene_type, sequence_id, capture_date, split,
privacy_reviewed`. **No GPS coordinates are ever stored.** GPS EXIF and XMP are stripped when
an image enters the pipeline, and the validator checks for leftovers.

## Roadmap

- [x] Repo, schema, pipeline tooling, tests, CI
- [ ] Confirm third-party licenses and layouts (see TODOs in `configs/sources.yaml`)
- [x] Convert HIT-UAV, BIRDSAI and ICAERUS into the shared schema
- [ ] Collect our first flood/pasture flights (RGB and thermal), with permission from owners
- [ ] Annotation workflow (CVAT or Label Studio) and labeling guide
- [ ] Top-down animal cutouts (SAM-assisted) for compositing; thermal composites
- [ ] Baseline v0 and a first public release on Hugging Face
- [ ] Partner with rescue organisations for real-world footage and feedback

## Contributing

Footage, labels, code and reviews are all welcome. Read [CONTRIBUTING.md](CONTRIBUTING.md)
first, especially the sections on **consent and privacy** and **licenses**.

## License

- Code: [Apache-2.0](LICENSE)
- Data: our own data is CC BY 4.0. Imported third-party data keeps its original license,
  tracked per image. See [DATA_LICENSE.md](DATA_LICENSE.md).
