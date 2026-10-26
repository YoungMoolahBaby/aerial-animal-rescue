---
# Hugging Face dataset card metadata. This file is uploaded as the dataset repo's README.md.
license: other
license_name: mixed-per-image
license_link: LICENSE.md
pretty_name: Aerial Animal Rescue
task_categories:
  - object-detection
tags:
  - aerial
  - drone
  - uav
  - thermal
  - infrared
  - disaster-response
  - flood
  - animals
  - livestock
  - pets
size_categories:
  - n<1K  # TODO: update on each release
---

# Aerial Animal Rescue

Drone images (RGB and thermal) of pets and livestock in flood, debris-flow and pasture
scenes, with bounding boxes, for training detectors that help find animals left behind in
disasters.

- **Code and baseline:** https://github.com/YoungMoolahBaby/aerial-animal-rescue
- **Version:** TODO (e.g. v0.1, date)
- **Contact:** [GitHub issues](https://github.com/YoungMoolahBaby/aerial-animal-rescue/issues)

## Dataset description

TODO: one paragraph on what is in this release, how many images, what share is real vs
synthetic, and what share is RGB vs thermal.

## Classes

| id | name | notes |
|---:|------|-------|
| 0 | person | humans; kept so the model separates people from animals |
| 1 | dog | |
| 2 | cat | |
| 3 | horse | horses, ponies, donkeys, mules |
| 4 | cattle | cows, bulls, calves, buffalo |
| 5 | small_livestock | goats and sheep |
| 6 | other_animal | pigs, poultry, wildlife, unknown species |

## Structure

```
images/{train,val,test}/<image_id>.<ext>
labels/{train,val,test}/<image_id>.txt   # YOLO: class cx cy w h (normalized); empty = no objects
metadata.csv                             # one row per image
```

`metadata.csv` columns: `image_id, file_path, source, source_license, modality (rgb|thermal),
synthetic, altitude_m, weather (clear|rain|fog|unknown), scene_type
(flood|debris_flow|pasture|urban|other), sequence_id, capture_date, split, privacy_reviewed`.

Splits are **grouped by flight or video sequence**, so near-duplicate frames never cross
splits. Each source is split on its own (about 70/15/15 of its images with boxes), so
every source appears in val and test. **The test set contains real images only.** Synthetic images are in train only.

Empty images (no boxes) are sampled down to at most 10% of train, spread across flights.
Val and test keep every empty image, so their numbers include false alarms on empty
ground, which is most of what a search drone sees.

TODO: table of image and box counts per split, class and modality (the output of
`scripts/validate_dataset.py`).

## Sources and licenses

Licenses differ per image: **always check `source_license`**. Some sources may be
non-commercial.

| source | what it adds | license | citation |
|--------|--------------|---------|----------|
| own | our drone footage | CC-BY-4.0 | this dataset |
| synthetic | composites (FloodNet backgrounds + cutouts) | see per-image | this dataset + inputs |
| birdsai | thermal night video frames from a fixed-wing drone over protected areas in Africa: people, elephants, giraffes, lions, dogs, unknown animals ([LILA BC](https://lila.science/datasets/conservationdrones)) | [CDLA-Permissive-1.0](https://cdla.dev/permissive-1-0/) | Bondi et al., "BIRDSAI: A Dataset for Detection and Tracking in Aerial Thermal Infrared Videos", WACV 2020 |

Fill this from `configs/sources.yaml` for every source in this release.

## Collection methods

- **Own footage:** TODO (drones and cameras, altitudes, regions at country level only,
  dates, consent process). Frames are extracted from video at a fixed interval.
- **Third-party data:** converted from the original annotations. Source classes are mapped
  onto ours, and classes without a match are dropped (see the conversion logs).
  - **BIRDSAI:** the 48 real videos (train and test zips; the simulated set is not used).
    Every 5th frame is kept, since neighbouring frames are near-duplicates, and only frames
    between a video's first and last annotated frame are used, since outside that range
    "no boxes" may mean "not annotated". Humans become `person`. Every animal species,
    including the source's "dog", becomes `other_animal`: the footage is from African
    reserves, so these are likely wild dogs or ranger dogs, not the domestic dogs our `dog`
    class means.
- **Synthetic:** segmented animal cutouts pasted onto disaster backgrounds, with random
  scale, rotation, brightness, blur, rain and noise. Labels come from the cutout masks.
- **Annotation:** TODO (tool, guidelines, who annotated, how it was quality-checked).

## Known limitations

- Small dataset; class imbalance (TODO: numbers). Rare classes (cats, small livestock in
  floods) are under-represented.
- Imported datasets come from different regions, sensors and altitudes. Thermal data mostly
  comes from non-disaster scenes: BIRDSAI is night-time footage of African bushland, not
  floods, and contributes no thermal examples of pets or livestock.
- Third-party labels are used as published. BIRDSAI misses some animals (for example only
  part of a herd boxed), so a few positives are labelled as background.
- Synthetic composites have no true perspective, shadows or water interaction. Results on
  synthetic data overstate real performance, so report results on the real-only test set.
- Animals partly submerged, under debris, or in dense vegetation are hard and often missing.
- Bounding boxes only; no segmentation, tracking IDs, or species below our class level.

## Ethics and privacy

- **No location data:** GPS EXIF and XMP are stripped from every image, and no coordinates
  are stored.
- **People:** people can appear (class `person`). Our own images are reviewed before
  publishing so that no one is identifiable (faces, plates, house numbers). In third-party
  data such as BIRDSAI, people are small heat signatures in low-resolution thermal video
  shot from altitude, with no identifiable detail.
- **Consent:** our footage was taken with the permission of the property and animal owners,
  in line with local drone regulations.
- **Intended use:** research and tools for animal search and rescue, and disaster response.
  This data must not be used to surveil people or private property.
- **Takedown:** to request removal of an image, open an issue on the GitHub repository.

## Citation

```bibtex
@misc{aerial_animal_rescue,
  title  = {Aerial Animal Rescue: a drone dataset for detecting animals in disaster scenes},
  author = {{aerial-animal-rescue contributors}},
  year   = {TODO},
  url    = {https://github.com/YoungMoolahBaby/aerial-animal-rescue}
}
```

Please also cite every third-party source you use (see `configs/sources.yaml`).
