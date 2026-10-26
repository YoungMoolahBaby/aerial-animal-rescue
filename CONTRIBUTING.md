# Contributing

Thanks for helping. This project is about getting animals out of disasters, so contributions
of every kind count: footage, annotations, code, reviews, and connections to rescue groups.

## Ground rules for data

### Consent and privacy (read this before you fly)

- **Ask permission** before you film anyone's animals or property, and follow your local
  drone rules. Note the consent in your flight log. Do not upload footage you could not
  defend to the owner.
- **No identifiable people.** People can appear in the data (the `person` class exists so
  the model does not mistake them for animals), but faces, license plates, house numbers and
  similar details must not be identifiable in published images. Blur them, or drop the frame.
- **No location data.** Ingest strips GPS EXIF and XMP automatically. Don't add coordinates
  anywhere else either (file names, notes, metadata).
- Our own footage is published only after a person has reviewed it and set
  `privacy_reviewed=True` in `metadata.csv`. The upload script enforces this.
- In an active disaster, rescue work comes first. Never fly where you could interfere with
  emergency aircraft or responders.

### Licenses

- Footage you contribute is released under **CC BY 4.0** (see [DATA_LICENSE.md](DATA_LICENSE.md)).
  Only contribute footage you own or have the right to license.
- To add a third-party dataset, add an entry to `configs/sources.yaml` with its landing page,
  license and `redistribution` status. **Check the license yourself on the source's own
  page**, not on a mirror. Some datasets are non-commercial only, and some forbid
  redistribution. When in doubt, mark it `TODO`: the upload script refuses `TODO`.

## Development

```bash
uv sync
uv run ruff check . && uv run ruff format --check .
uv run pytest
```

- Keep modules small and documented. Put logic in `src/aerial_animal_rescue/`; files in
  `scripts/` should be thin argument-parsing wrappers around it.
- Add a test with tiny fake data for every new converter or data rule.
- Never commit data, weights or secrets. `.gitignore` covers `data/`, `runs/`, `*.pt` and
  `.env`.

## Adding a dataset converter

1. Add the source to `configs/sources.yaml`, including a `class_map` onto our classes.
2. If its format is not already supported (`coco`, `voc`, `yolo`, `birdsai_csv`), write a
   reader in `src/aerial_animal_rescue/converters/` that yields `SourceImage`s, and register
   it in `converters/__init__.py`.
3. Run `scripts/convert_external.py <name>` and read `conversion_log.txt`. It lists every
   dropped class.

## Pull requests

Open an issue first for big changes. Keep pull requests focused, make sure CI passes, and
describe how you tested.

By participating you agree to the [Code of Conduct](CODE_OF_CONDUCT.md).
