# Storm sim: plan

A Rust/Bevy simulator inside this repo, built on the published `cortenforge` crates, for
two jobs:

1. **Synthetic training data.** Render storm and disaster scenes (thermal first, then RGB)
   with exact bounding boxes, in this repo's dataset format, to fill gaps the real data
   can't: faint tiny people, animals in floods, pets and livestock, bad weather.
2. **Vehicles in weather.** Fly drones (later boats and ground robots) through wind, rain
   and floods, with our detector in the loop, to test search-and-rescue behaviour before
   it meets the real world.

Both use one world: the same terrain, weather, animals and sensors.

Status (2026-10-28): phases 0 and 1 done. The phase 1 pilot is a GO: adding 4,000 sim images
improved real val (see the experiment result under phase 1). Phase 2 next.

Build and run (from `sim/`; the first release build of Bevy + CortenForge takes ~18 min):

    cargo run --release --bin sim_view                 # Space: new shot, I: id look
    cargo run --release --bin datagen -- --out ../runs/sim/test --frames 50 --seed 0
    cargo test --release

Animal models (phase 2): `uv run python scripts/bake_bodies.py` (from the repo root)
fetches the CC0 models in `sim/bodies.csv` and bakes them into `sim/assets/bodies/`;
`--bodies assets/bodies` makes datagen and sim_view use them.

`SIM_DEBUG=1` makes datagen also save each shot's raw render and id frame under
`<pool>/../debug/` (to check labels against pixels).

Pools for training go to `data/interim/sim_<name>/` (picked up by `split_dataset.py`);
test pools go under `runs/sim/` so they never end up in a real build.

## Ground rules

- **Real data decides.** Sim images are `synthetic=True`: train only, never val or test
  (the split already enforces this). A phase counts as done when it moves numbers on the
  **real** val set (`scripts/evaluate.py`: per-flight table, size buckets, false alarms),
  not when the renders look good.
- **Match the sensor, not just the world.** At 60-130 m our objects are 5-30 px. What
  matters is object size in pixels, optics blur, sensor noise, compression and, for
  thermal, temperature contrast and the camera's auto-gain. Fine geometry barely shows.
- **Randomise widely** (terrain, temperatures, lighting, noise, gain, altitude, angle)
  rather than chasing one photorealistic look; then check the spread covers the real data.
- **Licenses per image.** Every render records the licenses of the assets in it (3D
  models, textures), like composites do today. Only CC0 / CC-BY assets.
- **Use CortenForge for what it's good at:** physics, terrain, vehicles, weather forces.
  Bevy renders. Our own code does capture, labels and sensor models (the old
  capture/vision/data-contract crates from CortenForge-Archived are yanked and not in 0.9).
- **Log CortenForge friction as we go** in `sim/FINDINGS.md` (bugs, doc gaps, awkward
  APIs, what works, and suggestions), same format as cortenforge-play's log. Use the
  published crates only, as an outside user would.

## What we can measure against (already in the repo)

Per real flight (see `runs/eval/*_val.json` and the tiny-object diagnostics):

| statistic | real range | used for |
|---|---|---|
| thermal frame size | 635-701 x 321-512 px (BIRDSAI), 640x512 (HIT-UAV) | render size |
| tiny person brightness above surroundings | 50-125 grey levels (0-255) | thermal contrast |
| pixel-to-pixel noise (median neighbour diff) | 1-5 grey levels | sensor noise |
| object size | person 5-30 px, animals mostly <32 px in hard flights | altitude/FOV range |
| empty frames | 30-36% | scene sampling |

Phase 0 turns these into a script so sim and real are compared the same way every time.

## Layout

```
sim/
  PLAN.md            this file
  FINDINGS.md        CortenForge friction log and suggestions, kept up to date while building
  Cargo.toml         one crate, binaries below (own workspace; Python tooling ignores it)
  src/
    world/           terrain (heightfield), vegetation, water, debris; seeded
    actors/          people, animals: mesh + temperature + pose; placement rules
    weather/         wind field, rain, fog; shared by renderer and physics
    sensor/          camera models: thermal (temperature -> grey, AGC, blur, noise, FPN)
                     and RGB (exposure, blur, noise, JPEG)
    capture/         id-mask pass -> boxes; writes images/, labels/, metadata.csv
    vehicle/         (phase 3) cortenforge sim-core models, controllers
  src/bin/
    sim_view.rs      interactive viewer, sensor overlays, label overlay
    datagen.rs       headless: N scenes -> data/interim/sim_<name>/ (our pool format)
assets/sim/          3D models + LICENSES.csv (one row per asset: source, license, author)
scripts/compare_domains.py   sim vs real statistics (phase 0)
```

## Phases

### Phase 0: skeleton and measuring stick (small) - done 2026-10-28

Result: 50-frame pool (`runs/sim/phase0`) passes `split_dataset.py` (all train, as
synthetic), `validate_dataset.py` and the publish gate (`sim` is skipped like
`synthetic`, per-image licenses still checked). `compare_domains.py` against 1,500
images each of BIRDSAI / HIT-UAV (`runs/domain_real.csv`):

| | sim phase 0 | BIRDSAI | HIT-UAV |
|---|---|---|---|
| person size, median | 7 px (all <16) | 22 px | 15 px |
| brightness p1-p99 | 89-90 | 72-191 | 52-195 |
| person minus surroundings | 166 (381x ring spread) | 130 (5.6x) | 109 (5.4x) |
| noise (median neighbour diff) | 1.0 (Bevy's deband dither) | 2.0 | 1.0 |

So phase 1 must: randomise the lens (HIT-UAV used 25 mm, about 25x20 deg; 37 deg makes
people half the real size), turn off Bevy's deband dither and model noise ourselves, and
give the ground real temperature variation so objects stand out about 3-6x the
background's spread instead of 380x. Also: a blank-frame bug (Bevy draws nothing while
it compiles shaders) is handled by a warm-up plus a retake on uniform captures.

Planned:

- `sim/` crate builds with `cortenforge` 0.9 + Bevy 0.19 (the pair already working in
  cortenforge-play), opens a window with a heightfield and a camera at altitude.
- `datagen` writes a valid pool: one test scene, one box, passes `validate_dataset.py`
  and `split_dataset.py` puts it in train.
- `scripts/compare_domains.py`: the statistics table above for any pool (contrast of
  objects vs surroundings, noise, object sizes, histogram), sim next to real.
- Decide how sim pools are registered (a `sim_*` entry in `sources.yaml` with
  `train_only` and the asset licenses, or a new `own`-style source).

### Phase 1: thermal pilot (the go/no-go)

Progress (2026-10-28). Built:

- **Temperature render**: every material is an unlit grey = temperature through a
  per-shot window (`thermal.rs`), so the 8-bit frame keeps its precision. Ground by time
  of day (sun-facing slopes, vegetation cover, soil patches, a fine tiling texture),
  bodies a few degrees off the ground under them (occasionally darker by day), plus
  unlabelled clutter: textured bushes and trees, rocks, houses (cold roofs at night),
  cars, roads and flood water.
- **Sensor model** (`sensor.rs`): optics blur, raw noise, column/row stripes, percentile
  auto-gain, output noise, JPEG 70-95.
- **Labels from an id render**: each shot is drawn a second time with target k in red
  10k; boxes are the visible pixels (>= 6), so bodies behind hills, trees, roofs or
  under water get no box and partly hidden ones get a box on what shows. Checked
  pixel-exact against the raw render (`SIM_DEBUG`).
- **Bodies from primitives**: people standing, sitting or lying; cattle, horse, deer,
  sheep, dog, boar as capsule-and-cylinder quadrupeds, standing or lying. No third-party
  assets, so no asset licenses (CC0 models can come later if shape turns out to matter).
- **Shots**: lens 12-37 deg vertical (log-uniform), 60-130 m, pitch 30-90 deg with the
  horizon kept out; ~30% empty, else a group of people, a herd of one kind, or a mix.
- **Label margin**: real person boxes are looser than the bodies (median short side
  12-17 px in HIT-UAV/BIRDSAI vs 7 px for tight id-mask boxes at the same lens and
  altitude ranges), so sim boxes get the blur width plus 2 px per side
  (`LABEL_MARGIN` in datagen). This is a stopgap. Part of the gap is our bodies being
  too slim (no arms, legs together, no warm halo), and that part shrinks as bodies get
  more realistic. Part is labellers' margin, and that stays, because real val is scored
  against those boxes. Each time the bodies change, re-measure tight sim boxes against
  real boxes at matched size and lower the margin by whatever the body gap closed.

`compare_domains.py` on a 120-frame pool (one world) vs `runs/domain_real.csv`:

| | sim phase 1 | BIRDSAI | HIT-UAV |
|---|---|---|---|
| person size, median | 13 px | 22 px | 15 px |
| person minus surroundings | 90 (5.6x ring spread) | 130 (5.6x) | 109 (5.4x) |
| animals, x ring spread | 3.3-5.9 | 3.6 | - |
| noise (median neighbour diff) | 1.0 | 2.0 | 1.0 |
| brightness p1-p99 | 46-215 | 72-191 | 52-195 |
| empty frames | 44% | 27% | 41% |

Those medians hid a gap. A crop gallery at matched sizes (8-20 px) against the val
flights the go/no-go is judged on showed that real BIRDSAI is much harder than the sim
was: flight 361 has noise 5 and contrast 50 grey levels (3.7x the ring spread), and its
people are faint, soft blobs; flights 58/60/61 show motion smear. Sim people were crisp,
bright and clean. So the sensor model now spans clean (HIT-UAV) to hard (BIRDSAI):
- blur 0.5-2 px, log-uniform;
- motion smear 1-4 px in half the frames;
- output noise 0.8-8 levels plus blotchy correlated noise;
- JPEG kept at 65-96 (lower quality smoothed the noise away);
- squeezed brightness ranges;
- body contrast log-uniform, so faint bodies are common.

Per-image noise now overlaps BIRDSAI's: 0-2 levels in 72% of sim images vs 63%, 4-5 in
12% vs 25%. Caveat: tuned on image statistics (never detection results) of val flights.
The test set stays the independent check.

| 8 pilot worlds | sim | BIRDSAI | HIT-UAV |
|---|---|---|---|
| person size, median | 15 px | 22 px | 15 px |
| person minus surroundings | 68 (5.6x) | 130 (5.6x); 361: 50 (3.7x) | 109 (5.4x) |
| animals, x ring spread | 3.6-5.2 | 3.6; 58/60/61: 4.5-8.0 | - |
| brightness p1-p99 | 42-186 | 72-191 | 52-195 |
| empty frames | 32% | 27% | 41% |

Pilot pool: `data/interim/sim_p1_w10` ... `sim_p1_w17` (8 worlds x 500 frames, 4,000
images). Built with the real data on D: (`split_dataset.py --out`); the real images and
their splits are identical to `data/processed`, and all sim images are in train.

Before the full run, two cheap checks:
1. **Zero-shot**: the real-trained model on the sim images, per size, against its real
   val recall.
2. **Sim-only**: a short training on sim alone, scored on real thermal val.

Zero-shot result (2026-10-28, `hituav` best.pt, class-agnostic, IoU >= 0.5, conf
0.25). The first two columns come from a quick version with size by longer side, on 800
and 400 images. The last two come from `scripts/zero_shot.py` (area buckets as in
`evaluate.py`, 800 images each) on the final 8-world pilot set:

| found | sim, smooth clutter | sim, lumpy (1 world) | sim pilot (8 worlds) | real thermal val |
|---|---|---|---|---|
| tiny | 16% | 15% | 16% | 7% |
| small | 30% | 24% | 35% | 28% |
| medium | 28% | 19% | 20% | 87% |
| alarms per empty frame | 0.87 | 0.43 | 0.58 | 0.12 |

    uv run python scripts/zero_shot.py <best.pt> D:/aar/sim_p1_only --split train --sample 800 --alarms runs/eval/zero_shot_alarms
    uv run python scripts/zero_shot.py <best.pt> data/processed --split val --modality thermal --sample 800

- Small sim bodies aren't foreign to the model: it finds them more easily than real
  ones.
- Medium sim bodies (cattle, horses at 32-96 px) are rarely found: thermal livestock
  shapes it has never seen. That is new knowledge rather than a defect, but if the pilot
  is a go, better body meshes are the first thing to try there.
- Alarms on empty frames went to smooth uniform discs (bushes, rocks) and plain
  rectangles. Training on those as negatives would teach "a smooth warm blob is
  background", which is what real animals look like at this size.
- So clutter was made unlike bodies:
  - bushes and trees are clumps of noise-displaced, finely subdivided blobs with
    mottled temperatures;
  - rocks are lumpy and cooler;
  - cars have cold glass on a warm body;
  - roofs come in two halves.

  Alarms halved. What's left is mostly cars (fair hard negatives: real cars are
  unlabelled in HIT-UAV) and a few body-sized bushes.
- Sim-only check (2026-10-28): yolo11s trained 15 epochs on the 4,000 pilot images
  alone (`runs/sim_only_p1.yaml`), scored on real val.
  - Its best epoch (3) finds tiny and small real thermal objects about as often as the
    real-trained model: tiny 8.3% vs 6.7%, small 24.6% vs 27.6% (class-agnostic).
  - On flight 362 people it scores AP50 0.658, vs 0.623 for `hituav`, with no real image
    in training. Flight 361 stays at about zero (0.011).
  - Medium objects don't transfer (3.8% vs 87%): the big real wildlife looks nothing
    like the sim herds. Alarms on empty real frames run at 0.68 vs 0.12.
  - Later epochs overfit to sim (epoch 15: tiny 1.5%), so mixing with real data matters.
  - Watch in the mixed run: sim teaches "thermal four-legged blob = cattle", which may
    compete with `other_animal` on the BIRDSAI animal flights.
- Realism round 2 (2026-10-28), aimed at what the sim-only model fired on in empty real
  frames, and at the medium-size gap:
  - **Wildlife**: elephant, giraffe and lion shapes as `other_animal` (the BIRDSAI
    animal flights).
  - **Poses**: every body instance gets its own pose at spawn (animals mid-stride, grazers
    with heads down, people with arms).
  - **Body temperatures**: torso, cooler limbs, warmer head.
  - **Video artifacts**: interlacing (odd rows shifted 1-3 px, in 40% of frames), and
    video-style compression (JPEG round trip at quality 40-80, then link noise, in half
    the frames). Quality below 40 wiped out the noise real recordings keep.
  - **Pixel-scale clutter**: sparse warm and cold specks 0.1-0.4 m across in the ground
    texture. The texture now centres on byte 230, with a matching gain, so specks can
    also be warmer than the ground.
  - Same 8 worlds, regenerated as `sim_p2_w10..17`. v1 is reproducible from commit
    1e18d03.
  - Result (sim-only, 6 epochs, `runs/train/sim_only_p2`; real val):

    | | v1 best | v2 best (ep 2) | v2 last (ep 6) | `hituav` (real) |
    |---|---|---|---|---|
    | mAP50 | 0.038 | 0.083 | 0.090 | 0.596 |
    | tiny / small found | 8.3% / 24.6% | 0.9% / 23.5% | 0.0% / 19.5% | 6.7% / 27.6% |
    | medium / large found | 3.8% / 0% | 12.2% / 3.4% | 31.0% / 44.8% | 86.7% / 100% |
    | alarms per empty frame | 0.68 | 0.23 | 0.06 | 0.12 |
    | flight 362 people AP50 | 0.658 | 0.819 | | 0.623 |
    | flight 58 / 352 animals AP50 | 0.085 / 0.000 | 0.397 / 0.194 | | 0.154 / 0.860 |

    The wildlife, pose and clutter changes worked. Sim-only now beats the real-trained
    model on flights 362 and 58, and fires less than it does on empty frames. Tiny
    objects regressed to about zero; 361, 60 and 61, all tiny, stay at zero. Suspect:
    the 1-8 px ground specks look like tiny bodies, so they teach "a tiny warm dot is
    background".
- Round 3 (v3, 2026-10-28, `sim_p3_w10..17`): specks halved, 75% cold, warm ones wider
  and fainter. Sim-only, best epoch 2 (last epoch 6 in brackets):
  - tiny found 3.1% (0.6%), small 42.7% (26.5%), medium 8.8% (11.6%), large 17% (45%);
  - alarms per empty frame 1.03 (0.05);
  - flight 362 people AP50 0.774 (recall 82%); flights 58/352 animals 0.282/0.142;
    361/60/61 still 0.

  The speck trade-off is real: warm specks as negatives cut alarms but suppress tiny
  real objects. Other v2/v3 differences are within the noise of these short runs
  (mAP swings 2-4x between epochs). In a mixed run, real empty frames supply real warm
  specks as negatives, so v3 goes into the mixed run.

**Experiment result (2026-10-28): GO.** `sim_p3` = the `hituav` settings plus the 4,000
v3 sim images; real data and splits identical. It early-stopped at epoch 54 with the
best model at epoch 34. `runs/eval/sim_p3_val.json` vs `runs/eval/hituav_val.json`:

| real val | hituav | sim_p3 |
|---|---|---|
| overall mAP50 / 50-95 | 0.596 / 0.316 | 0.604 / 0.330 |
| thermal mAP50 / 50-95 | 0.442 / 0.159 | 0.475 / 0.201 |
| person / other_animal / cattle (RGB) AP50 | 0.214 / 0.671 / 0.904 | 0.251 / 0.700 / 0.860 |
| tiny thermal found (class-agnostic) | 6.7% | 11.9% |
| flights 58 / 60 / 61 animals AP50 | 0.154 / 0.367 / 0.001 | 0.279 / 0.592 / 0.000 |
| flights 361 / 362 people AP50 | 0.000 / 0.623 | 0.000 / 0.713 |
| thermal alarms per empty frame, conf 0.25 / 0.5 | 0.169 / 0.042 | 0.131 / 0.044 |

- Go criteria: 58 and 60 improved a lot and thermal alarms didn't rise.
- 361 and 61, the faintest tiny targets, are still at zero.
- RGB cattle dropped 0.04, mostly on two ICAERUS flights. Sim cattle are thermal-only,
  so cross-talk is possible, but it's within the single-seed spread of flights with a
  few hundred boxes. RGB alarms fell from 0.62 to 0.10 per empty frame.
- One training run per side: check with a second seed before relying on differences
  under ~0.05.
- Detail is about outlines and temperature variation, not polygon count. At 4-10 cm a
  pixel, grass and leaves are sub-pixel, so they stay as texture.

Scope: thermal only, 640x512, white-hot; oblique and nadir views from 60-130 m; bush,
grass and flooded terrain; day and night; people plus 3-4 animal shapes (deer, cattle,
dog, boar).

- Lens randomised over the real cameras (about 20-37 deg vertical field of view), so
  object sizes match the real spread (phase 0 showed 7 px people vs 15-22 px real).
- Temperature per material (ground by time of day, water, vegetation, warm bodies with
  clothing/fur variation); render temperature, then the sensor model: optics blur,
  auto-gain (percentile stretch, the main driver of contrast), noise, fixed-pattern
  stripes, mild JPEG.
- Labels from an object-id mask, not projected bounding boxes: tight boxes, correct under
  partial occlusion; drop objects with too few visible pixels. Label only what the camera
  sees: a fully hidden person gets no box (a box on bare hillside teaches false alarms),
  a partly hidden one gets a box around the visible part. Phase 0 only checks the line
  to the object's centre, which wrongly drops a person whose head shows over a ridge.
- 3,000-5,000 images, randomised across the real ranges; `compare_domains.py` shows the
  spread covers the real flights.
- **Experiment:** retrain with the split2 settings, with and without the sim pool, on the
  same real data. Go if real val improves on at least one of: flight 361 tiny people
  (0% now), animal flights 58/60/61 (AP50 0.00-0.34), without raising false alarms on
  empty thermal frames. No-go: stop and find out why before adding more.

### Phase 2: wider coverage (after a go)

- **Better body meshes**, first. The real model finds only ~20% of medium (32-96 px)
  sim cattle and horses, against 87% of real medium objects (zero-shot, phase 1).
  - What to try: legs apart and walking poses, heads down grazing, people with arms and
    a warm head; then CC0 animal and human models if primitives aren't enough
    (Quaternius, Kenney; verify each license and record it per image).
  - Real scans (searched 2026-10-28): the models built from real animal scans (SMAL,
    hSMAL, VAREN from MPI; PFERD horse data) are licensed for non-commercial research
    only and can't be shared on, so they can't go into the published CC-BY dataset
    without MPI's written permission. No openly licensed scans of live cattle or horses
    turned up. Sketchfab/Objaverse have CC-BY cow and horse models, mostly made by
    artists; they're usable with per-image attribution, after checking each one.
  - How to judge: medium-size found rate in `scripts/zero_shot.py`, and the body-size
    gap behind `LABEL_MARGIN` (lower the margin by what closes).
  - Progress (2026-10-28): animals first.
    - Quaternius' Ultimate Animated Animal pack (CC0). Google Drive was over quota, so the
      12 models came from poly.pizza (Quaternius' own uploads, CC0); each source is in
      `sim/bodies.csv`.
    - In use: Cow and Bull (cattle); Horse x2 and Donkey (horse, as `classes.yaml` has
      it); Deer and Stag (deer); Husky and Shiba Inu (dog). Wolf, Fox and Alpaca are
      left out for now.
    - `scripts/bake_bodies.py` poses each model at 15 moments of its Idle, Idle_Headlow,
      Eating, Walk and Gallop clips and splits it into the sim's temperature parts by
      bone. Head and ears are the head. Lower legs, tail, horns, hooves and antlers are
      limbs. The torso includes hips, thighs and shoulders. Each model is scaled to a
      real nose-to-rump length.
    - `datagen --bodies assets/bodies` builds those kinds from the models (a random
      model and pose per pool body) and records each shot's bodies in `sim_params.csv`.
      For a seed the shots are identical with or without `--bodies`, so the comparison
      against v3 is like for like.
    - People: the free Universal Base Characters pack (CC0) has only the "Superhero"
      bodies, in a T-pose. Drawn at real pixel scales with the arms down:
      - At 6-13 px/m (130 m on a wide lens, the faint flight-361 range) they are the same
        blob as our capsule person.
      - At 25-40 px/m (60 m on a narrow lens, like HIT-UAV) they show legs, arms and a
        warm head; the broad male shoulders barely show.
      - Our capsule person has no warm head part at all.
      - Poses would need the free Universal Animation Library. Next after the animals.
- RGB camera model; pasture, flood and debris scenes; cattle, horses, sheep/goats, dogs,
  cats (the classes with no real data yet).
- Weather in the image: rain streaks, fog/haze, wet surfaces; flood water reflecting the
  sky (cold and mirror-like in thermal).
- Each addition is kept only if real val holds or improves.

### Phase 3: vehicles in weather

- Drone model in cortenforge sim-core (MJCF/URDF loaders), flown through the world's wind
  field (sim-core `wind`, fluid drag per geom, gusts via `xfrc_applied`) and rain.
- The camera rides on the vehicle; our detector runs in the loop (ONNX export of the
  YOLO weights).
- Search missions scored like the real job: area covered, time to first find, people and
  animals found, false alarms sent out, battery/wind margins. Then boats and ground robots
  for flooded streets.

## Open questions

- Assets: CC0 animal and human models with enough variety (candidates to verify:
  Quaternius, Kenney, Poly Haven for terrain textures), if the primitive bodies turn out
  to be too simple.
- Answered in phase 1: temperature renders as unlit grey through a per-shot window, and
  the sensor model runs on the CPU after capture; datagen keeps a visible window (a few
  shots per second, both captures included).
- What, if anything, to take from the CortenForge research on navigating noisy media for
  phase 3 controllers.
