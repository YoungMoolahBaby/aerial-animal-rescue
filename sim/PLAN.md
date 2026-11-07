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
  - Result v4 (2026-10-28): same 4,000 shots as v3 (seeds 10-17), only the bodies
    differ.

    | | v3 primitives | v4 models |
    |---|---|---|
    | zero-shot (hituav) medium found, all / horse / cattle | 21% / 21% / 8% | 17% / 5% / 4% |
    | sim-only 6 epochs, real val mAP50 (best epoch) | 0.087 | 0.055 |
    | thermal mAP50 | 0.133 | 0.086 |
    | flight 352 / 58 animals AP50 | 0.142 / 0.282 | 0.071 / 0.372 |
    | flight 362 people AP50 | 0.774 | 0.723 |
    | thermal alarms per empty frame, conf 0.25 | 0.93 | 0.47 |

    - No gain on anything real val can show; the sim-only proxy swings 2-4x per epoch,
      so this is "not better", not "worse".
    - Why the measure can't credit them: real val has no thermal cattle, horses or dogs.
      Its thermal animals are BIRDSAI wildlife (elephants, giraffes, ...; `other_animal`),
      still primitive in the sim, and real cattle are RGB only. A real model finding a
      body measures resemblance to elephants and people, not realism.
    - Kept as an option (`--bodies`), off by default; v3 stays the training pool. The
      models matter again once there is real thermal farm-animal data to judge them.
    - Next, on what real val does measure: people (flights 361/362) with the Superhero
      bodies and the animation library, and wildlife models (elephant, giraffe, rhino,
      lion) for the BIRDSAI flights; plus the faint-target work for 361/61.
  - v5, people (2026-10-28):
    - Bodies: the free Superhero man and woman (Universal Base Characters, CC0).
    - Clips: the Universal Animation Library (Standard, CC0, 43 clips on the same rig),
      copied across by bone name (rotations, plus the pelvis height).
    - Poses: 15 per body from idle, talking, walk, jog, crouch, crouch-walk, kneel, push
      and lying (Death01). Sitting is left out because it sits on a chair. Swimming is
      left out because people stand on the ground in the sim.
    - Parts: head and neck are warm; arms, hands, calves and feet are limbs.
    - Every model body now carries all poses of its kind's models, and each shot shows
      one of them, picked by a hash of where the body stands. Before, each body had one
      fixed pose per world. The pick needs no random draws.
    - Model people skip the primitive's squash-to-sit and tip-to-lie, since their posture
      is in the mesh. They draw the same random numbers, so shots stay identical to v3.
    - `--body-kinds person` limits the models to people, so v5 against v3 changes only
      the people.
  - Result v5: same 4,000 shots as v3, only the people differ.

    | | v3 capsule people | v5 model people |
    |---|---|---|
    | zero-shot (hituav) person found, tiny / small / medium | 15% / 34% / 18% | 13% / 30% / 18% |
    | sim-only 6 epochs, real val person AP50 | 0.162 | 0.046 |
    | flight 362 / 361 people AP50 | 0.774 / 0.007 | 0.506 / 0.003 |
    | overall / thermal mAP50 | 0.087 / 0.133 | 0.054 / 0.089 |
    | thermal alarms per empty frame, conf 0.25 | 0.93 | 0.78 |

    - Worse on the one thing that changed, people, and the zero-shot agrees. Raw contrast
      is the same (mean 47 vs 48 grey levels above the ground on 9 matched people), so
      it's the shape.
    - Likely reason: at 10-30 px a real thermal person (BIRDSAI at night, HIT-UAV) is a
      compact bright blob. Thin limbs at limb temperature and sprawled poses (lying,
      jogging, arms out) spread the body into a shape real people don't show at that size.
    - Not kept: v3's capsule people stay. `--bodies` and `--body-kinds` stay as options.
- **Faint targets** (2026-10-28), aimed at flights 361 (tiny people) and 61 (tiny animals),
  both still at 0.
  - What makes them hard (`compare_domains.py` plus a band-pass clutter check; contrast =
    peak minus surroundings, grey levels):

    | | size | contrast | noise | target / background p99 at body scale |
    |---|---|---|---|---|
    | 361 people | 10 px | 50 | 5 | 1.69 (24.5 vs 14.9) |
    | 362 people (found) | 26 px | 56 | 6 | 1.94 |
    | 61 animals | 12 px | 59 | 2 | 0.55 (25.9 vs 45.0) |
    | 58 animals (found) | 25 px | 114 | 2 | 2.93 |
    | sim v3 | 15-25 px | 70-80 | 1 | 2.3 (37 vs 16) |

    - 68% of 361's people are tiny (< 14 px), faint (< 60) and in a noisy frame (>= 4)
      all at once; in v3 about 1% of targets are.
    - Real thermal val is noisier than anything we train on: median frame noise 4 and
      51% of frames >= 4. BIRDSAI overall is 2 and 23%, HIT-UAV 1 and 0%, sim v3 1
      and 10%.
    - 361's people are weak blobs (24.5 at body scale vs the sim's 37) in ordinary
      mottle.
    - 61 is a daytime riverbed: tiny animals on sun-warmed sand, in frames whose
      strongest blobs are sunlit treetops and cold shadows. The sim's canopies have no
      sun side or shadow.
  - v6, sensor:
    - Haze, fog or drizzle in 35% of frames (`haze` 0.3-0.8) takes contrast away
      before the sensor; the auto-gain stretches it back, raw noise with it. Recorded as
      `weather: fog`.
    - Raw noise widened to 0.5-4 (log) and output noise to 1.5-9 (log).
    - Both drawn so shots stay paired with v3; the weather has its own stream.
    - Result: frame noise median 2, p75 5, p90 8, 36% of frames >= 4.
  - Result v6 (sim-only, 6 epochs, real val; v3 in brackets):
    - overall mAP50 0.076 (0.087), person 0.159 (0.162);
    - tiny 0.008 (0.006), small 0.151 (0.138), medium 0.096 (0.181);
    - flights 361 / 61: 0.013 / 0.000 (0.007 / 0.000);
    - thermal alarms per empty frame 0.34 (0.93) at conf 0.25, 0.004 (0.095) at 0.5.
    - Noise alone doesn't reach 361 or 61, but alarms fall by about two thirds.
      Everything else is within the proxy's noise, except medium (flight 352).
    - Kept as the new base: closer to the real noise, and far fewer alarms.
  - Next: weak tiny bodies (1.5-2x the ground's mottle at body scale, like 361), and,
    for 61, sunlit canopy (hot tops, cold shadows) with daytime bodies near the ground
    temperature.
  - v7, faint bodies:
    - In 40% of shots with bodies, each body stands out from the ground under it by
      16-60 grey levels of the render (log), set from the shot's temperature span. A
      fixed lead in degrees is bright in a cool field and invisible next to a hot roof.
      Limbs and head keep their proportions to the torso.
    - Drawn from a stream seeded by the aim point, so shots stay paired with v6.
      `faint` column in `sim_params.csv`.
    - Tuned on 200 frames to the 361 statistic: tiny/small targets at 1.59x the
      background's p99 at body scale (361: 1.57; v6: 2.2); faint shots alone 1.38.
  - Result v7 (sim-only, 6 epochs, real val; v6 in brackets):
    - flight 361 people 0.007 (0.013), 61 animals 0.000 (0.000);
    - person 0.096 (0.159), flight 362 0.587 (0.740);
    - other_animal 0.156 (0.067), flight 352 0.277 (0.094);
    - overall / thermal mAP50 0.084 / 0.127 (0.076 / 0.122);
    - thermal alarms per empty frame 0.93 (0.34) at conf 0.25.
    - Matching 361's statistics didn't move 361. The rest is a trade within the
      proxy's noise, and alarms are back up.
    - Not kept: `--faint-share` (default 0) keeps it as an option; datagen's default
      output is v6 again, byte for byte.
  - Where this leaves the faint flights:
    - Four rounds (v4-v7) haven't moved 361 or 61 in the sim-only check, and the
      real-only model scores 0 on 361 too.
    - Next questions:
      - Does v6 carry over in a mixed real+sim run (the only trustworthy judge)?
      - Is 361 a detector-resolution limit rather than a data gap? Test with tiled,
        zoomed-in inference on 361 with the existing models.
    - 61 (sunlit canopy) is parked until those answer.
  - Result: mixed run `sim_p6` (hituav settings + 4,000 v6 images, `D:/aar/processed_sim_p6`,
    real images and splits identical to `sim_p3`; 60 epochs). Real val, `runs/eval/sim_p6_val.json`:

    | val | hituav (real only) | sim_p3 (v3) | sim_p6 (v6) |
    |---|---|---|---|
    | overall mAP50 | 0.596 | 0.604 | 0.608 |
    | thermal mAP50 | 0.442 | 0.475 | 0.464 |
    | person / other_animal | 0.214 / 0.671 | 0.251 / 0.700 | 0.233 / 0.694 |
    | cattle (RGB, no sim) | 0.904 | 0.860 | 0.898 |
    | small / medium | 0.374 / 0.730 | 0.328 / 0.702 | 0.408 / 0.660 |
    | thermal alarms per empty frame, conf 0.25 | 0.169 | 0.131 | 0.149 |
    | flights 58 / 60 / 361 / 362 / 61 | .154 / .367 / 0 / .623 / 0 | .279 / .592 / 0 / .713 / 0 | .300 / .564 / .001 / .759 / 0 |

    - v6 ties v3: the differences (0.01-0.04) are within run-to-run noise. Cattle,
      which the sim doesn't touch, moved 0.04 between the two runs.
    - The sim-only alarm cut doesn't carry over: with real data in the mix the alarm
      rate is set by the real images.
    - v6 trades medium (flight 352, -0.04) for small (+0.08), in line with its wider
      noise. Both still beat real-only on thermal.
    - 361 and 61 stay at 0.
  - Result: zoom test on 361 (is it a resolution limit?). Each val frame of flights 361, 362
    and 61 cut into n x n tiles (25% overlap; boxes kept when >= 50% inside, clipped),
    then `model.val` at imgsz 1280. The models already see these ~640 px frames at 2x, so
    2x2 is ~4x and 3x3 ~6x. AP50 (recall):

    | flight | tiles | hituav | sim_p3 | sim_p6 |
    |---|---|---|---|---|
    | 361 people | 1x1 | 0.000 (0.006) | 0.000 (0.009) | 0.006 (0.020) |
    | | 2x2 | 0.002 (0.013) | 0.011 (0.036) | 0.001 (0.008) |
    | | 3x3 | 0.003 (0.025) | 0.001 (0.012) | 0.000 (0.002) |
    | 61 animals | 1x1 | 0.000 (0.003) | 0.000 (0.000) | 0.000 (0.000) |
    | | 2x2 | 0.048 (0.112) | 0.004 (0.059) | 0.005 (0.029) |
    | | 3x3 | 0.016 (0.117) | 0.004 (0.098) | 0.013 (0.077) |
    | 362 people (control) | 1x1 | 0.651 (0.599) | 0.737 (0.602) | 0.802 (0.665) |
    | | 2x2 | 0.662 (0.579) | 0.699 (0.606) | 0.722 (0.594) |
    | | 3x3 | 0.576 (0.498) | 0.654 (0.645) | 0.690 (0.598) |

    - Zooming doesn't find 361's people: at 3-6x the pixels no model gets past 0.04
      recall. So 361 is not a resolution limit. The models don't take these faint
      blobs in noise for people at any size, and neither real training data nor v7's
      faint sim bodies taught them to.
    - 61 gains a little recall with zoom (0.11 at 2x2) but almost no AP: partly size,
      mostly the sunlit-canopy look.
    - The control (362) holds up under zoom, slowly losing as people outgrow the
      sizes trained on, so the test itself is sound.
    - (Single-flight `model.val` numbers differ a little from `evaluate.py`'s
      per-flight AP, e.g. 362 sim_p6 0.802 vs 0.759: different scope and thresholds.)
- 361 and 61 parked (2026-10-29): the sim can't reach them without real faint-people
  thermal data. v6 stays the thermal base. Next is what only the sim can add.
- **RGB camera, pasture first** (2026-10-31). Real val can score RGB only on cattle: 197
  ICAERUS frames from 4 flights, 758 cows. There are no real horses, sheep/goats, dogs or
  cats anywhere yet. So the RGB look is judged on cattle, and the other classes ride on it.
  - What the real RGB data is (train flights only; val stays the judge):
    - DJI stills (4000x3000, 5280x3956) looking straight down at about 90 m.
    - At training size (1280) a cow is 13x21 px (median).
    - Ground median sRGB 95-143, from green pasture to straw.
    - Cows are 55% white (near clipping), 20% black, 25% brown.
    - Some flights have dark lens corners.
    - Val flight 032 (121 frames, 102 empty) is a harvested field with round bales: the
      RGB false alarms come from there.
  - `datagen --modality rgb` (`rgb.rs`, `scene.rs`):
    - 1280x960 frames, 45-60 deg lens, 60-150 m (log). 80% straight down, the rest
      tilted 70-90 deg. No flood yet.
    - Lit, matt materials. A sun (15-70 deg up) casts shadows; the sky fills them. 25%
      of shots are overcast.
    - MSAA on the picture only, so the id render stays exact. Boxes are padded by blur +
      1 px.
    - Ground: pasture 55%, dry grass 25%, stubble 20%. Per vertex, a second cover and
      soil patches. A colour texture adds per-texel grain, lush clumps, dry tufts, dung
      specks, and mowing or harvest stripes.
    - Clutter: canopies with a leafy texture, stones, houses, cars. On 70% of stubble
      shots, also 1-40 bales: round lying, round on end, or square; straw, white,
      black or green wrap.
    - Pool (31 bodies, the id render's limit): 5 people, 12 cattle, 5 horses, 7
      sheep/goats, 2 dogs.
    - Coats by kind:
      - cattle white / Holstein patches / black / brown / cream;
      - horses bay / chestnut / black / grey / palomino;
      - wool from clean to muddy, some black faces;
      - people in work clothes, hi-vis and dark, with skin or hair on the head.
    - Camera model: white balance error, auto-exposure to a median of 95-145, vignette,
      dark lens corners on 30% of frames, blur 0.2-0.6 px, light haze on 10%, contrast,
      saturation, noise, JPEG 75-96.
    - Thermal output is byte-identical to before, checked on a 24-frame pool.
  - How it's judged:
    - Sim-only: yolo11s, 6 epochs on 4,000 sim RGB frames (seeds 10-17), validated on
      the 197 real RGB val frames (`runs/val_rgb.txt`). Main number: cattle AP50 with no
      real RGB image in training. The real-only model gets 0.904 with real RGB cows in
      training.
    - Also the alarms on flight 032's empty frames. That rate swings across runs: the
      real-only model gets 0.84 per empty frame at conf 0.25, and two mixed runs without
      RGB sim got 0.13 and 0.03.
    - Zero-shot: the real-trained model on sim RGB cattle, found rate by size.
  - Not built yet:
    - field structure: straight edges between fields, hedges, fences, narrow tracks with
      wheel ruts (roads are still the thermal ribbons);
    - farmyards;
    - flood and debris in RGB;
    - cats.
  - Result RGB v1 (2026-10-31). Pools: `D:/aar/interim/sim_rgb1_w10..17`. Sim-only set:
    `D:/aar/sim_rgb1_only`. Run: `runs/train/sim_only_rgb1`. Eval:
    `runs/eval/sim_only_rgb1_val.json`.
    - Zero-shot, `hituav` on world 10:
      - it finds sim cattle about as often as real ones: 89% tiny, 95% small, 80%
        medium, against 93% of real val's (mostly medium) cows;
      - it finds sim horses at 78-88% and sim sheep at 43-86%, as some class;
      - alarms per empty frame: 2.76, against 0.62 on real RGB val. The top ones are
        the plain box cars and square and round bales.
    - Sim-only, 6 epochs, scored on the 197 real RGB val frames. The best epoch is 3
      (0.588 in training's own val); epochs 5-6 fall to 0.42 (overfitting to sim).

      | real RGB val | `hituav` (real RGB cows in training) | sim-only RGB v1 |
      |---|---|---|
      | cattle AP50 | 0.904 | 0.600 |
      | flight 070, dense herd, 344 cows | 0.974 | 0.888 |
      | flight 061 | 0.854 | 0.609 |
      | flight 032, bale field | 0.899 | 0.512 |
      | flight 056 | 0.834 | 0.201 |
      | alarms per empty frame, conf 0.25 / 0.5 | 0.62 / 0.26 | 9.9 / 0.58 |

    - Sim cows transfer: 0.60 AP50 and 0.89 on the dense-herd flight, with no real RGB
      image in training. The thermal sim-only proxies reached about 0.09 overall.
    - What fails is the background. On empty real frames the sim-only model fires on
      bright straight or elongated ground: wheel ruts and tramlines, track edges, field
      boundaries, straw swaths, light bare patches, and bales seen up close. Nothing like
      them is in the sim, so it never learned they're background. The field structure
      listed under "not built yet" is the next step, judged the same way (alarms on
      032 and 056, cattle AP50).
    - The real judge is still a mixed run (real + sim RGB). With real RGB cows in
      training, the alarm rate there is likely set by the real images, as in thermal.
  - RGB v2, field structure (2026-10-31, `fields.rs`):
    - Each shot paints the ground under the camera: 2048 px, about one texel per picture
      pixel, on a copy of the terrain 8 cm up. The ground shading and tiling texture
      couldn't draw sharp straight lines.
    - Fields with straight edges on one grid angle: pasture, dry grass, stubble, a green
      crop, bare soil. Edges are plain, fenced (uncut strip, faint wire, posts) or hedged
      (hedge geometry with shadows).
    - Farm tracks with bright wheel ruts and a grass centre, gravel tracks, the odd paved
      road. They replace the thermal ribbon roads in RGB.
    - Mowing stripes, furrows, tramlines, straw swaths with the bales in the rows,
      vehicle ruts across grass.
    - Vehicles: car, pickup, van, tractor, often on a track; up to 6 instead of 10 plain
      box cars.
    - Pixel-scale detail: median neighbour difference 9.0, real train flights 9.0, v1
      5.5. The first v2 render was 3.0 (no grain), fixed before the pools.
    - Pools `D:/aar/interim/sim_rgb2_w10..17` (seeds as v1), set `D:/aar/sim_rgb2_only`,
      run `runs/train/sim_only_rgb2`, eval `runs/eval/sim_only_rgb2_val.json`. The
      training was stopped by Claude Code for low memory after its 6th epoch;
      results.csv and best.pt were complete.
  - Result RGB v2:
    - Zero-shot, `hituav` on world 10: cattle found 89-96% (v1 89-95%), alarms per
      empty frame 2.37 (v1 2.76). Its top alarms are now objects (bushes, stones, bales,
      vehicles), not ground lines.
    - Sim-only on real RGB val (per-epoch val mAP50, v1 in brackets): 0.63 (0.11), 0.70
      (0.58), 0.69 (0.59), 0.66 (0.59), 0.70 (0.42), 0.68 (0.42). It no longer
      collapses after epoch 3.

      | real RGB val | `hituav` | sim-only v1 | sim-only v2 |
      |---|---|---|---|
      | cattle AP50 | 0.904 | 0.600 | 0.691 |
      | flight 070, dense herd | 0.974 | 0.888 | 0.968 |
      | flight 061 | 0.854 | 0.609 | 0.634 |
      | flight 032, bale field | 0.899 | 0.512 | 0.420 |
      | flight 056 | 0.834 | 0.201 | 0.281 |
      | alarms per empty frame, conf 0.25 / 0.5 | 0.62 / 0.26 | 9.9 / 0.58 | 27.3 / 4.1 |

    - Alarms on the 141 empty real frames, conf 0.25: sheep/goat 976 -> 3,359, cattle
      415 -> 372; under 16 px 1,048 -> 3,510. One empty 032 frame gets 122 boxes.
    - So field structure helps cattle (070 now nearly matches the real-trained model) and
      didn't add cattle alarms. The jump is tiny "sheep": the model fires on small pale
      specks in grass (dead-grass tufts, stones, light bare spots, dung). Sim sheep are
      5-15 px pale blobs, and the painted ground has grain and clumps but no discrete
      specks of that size. v1's tiling texture had bright "dry tuft" specks; the painted
      ground dropped them.
    - Real val has no sheep, so every sheep box there is a false alarm. The real images
      that would teach "pale speck = background" are the ones the sim must supply.
    - Next (v3): sparse discrete specks in the painted ground, 0.2-1.5 m: pale dead
      grass, stones, light bare spots, dung pats, molehills, thistles. Judged on sheep
      alarms and cattle AP50.
  - RGB v3, specks (2026-10-31, `fields.rs` `specks`):
    - One chance of a speck per 1.5 m cell, at a share set per parcel (pasture and dry
      grass most, crops least) and thinned in patches 15 m across. Kinds: pale dead-grass
      tufts, light bare spots, stones, dung, molehills, dark weeds. Each is 0.05-0.5 m,
      a little stretched, with a wobbly rim, broken up by the grain, and only partly
      covering the grass.
    - The first try (larger, solid-filled specks) looked like cows to the real model:
      zero-shot `hituav` on world 10 gave 8.23 alarms per empty frame (v2 2.37) and
      found only 72% of tiny cattle. Smaller, fainter, ragged specks brought that to 1.53
      alarms and 91%; that version made the pools.
    - Pools `D:/aar/interim/sim_rgb3b_w10..17` (seeds as v1), set `D:/aar/sim_rgb3b_only`,
      run `runs/train/sim_only_rgb3`, eval `runs/eval/sim_only_rgb3_val.json`.
  - Result RGB v3:
    - Per-epoch val mAP50 (v2 in brackets): 0.70 (0.63), 0.68 (0.70), 0.58 (0.69),
      0.70 (0.66), 0.67 (0.70), 0.72 (0.68).

      | real RGB val | `hituav` | sim-only v1 | sim-only v2 | sim-only v3 |
      |---|---|---|---|---|
      | cattle AP50 | 0.904 | 0.600 | 0.691 | 0.739 |
      | flight 070, dense herd | 0.974 | 0.888 | 0.968 | 0.966 |
      | flight 061 | 0.854 | 0.609 | 0.634 | 0.585 |
      | flight 032, bale field | 0.899 | 0.512 | 0.420 | 0.535 |
      | flight 056 | 0.834 | 0.201 | 0.281 | 0.474 |
      | alarms per empty frame, conf 0.25 / 0.5 | 0.62 / 0.26 | 9.9 / 0.58 | 27.3 / 4.1 | 5.6 / 0.65 |
      | precision at conf 0.5 | | | 0.36 | 0.72 |

    - Alarms on the 141 empty real frames, conf 0.25 (v2 in brackets): sheep/goat 245
      (3,359), cattle 208 (372), person 193 (65), horse 78 (7), dog 71 (46); under
      16 px 602 (3,510). The empty 032 frames went from 34.8 alarms each to 5.9.
    - Specks work: sheep alarms fall 14x and cattle AP50 is the best sim-only yet, with
      the two hard flights (032, 056) up most. Alarms at conf 0.5 are near v1's and
      only 2.5x the real-trained model's. 061 dips; it has 17 frames, so that may be noise.
    - What's left (top 30 alarm crops): crisp near-white spots in grass, brighter and
      sharper than v3's specks (paper, plastic, white stones, bleached dung); bleached
      fallen branches and dead trees (the worst frame, 27 boxes, is a dead tree lying by
      a stream); a white trough, a board, a bench, a manhole on asphalt, fence posts.
      The person and horse alarms are these elongated pale things.
    - Next candidates: a few crisp white specks; bleached dead wood (fallen trees,
      branches); farm objects (troughs, feeders, gates, benches). Or a mixed real + sim
      RGB run, which is the real judge.
  - Mixed run `sim_p6_rgb3` (2026-10-31): `sim_p6`'s set plus the 4,000 RGB v3 images
    (`D:/aar/processed_sim_p6_rgb3`: real train 10,520 + 8,000 sim; val and test
    file-for-file the same as `sim_p6`), same settings, 60 epochs. Against `sim_p6` only
    the RGB sim differs. Judged on real RGB cattle AP50 and RGB alarms, with thermal
    expected to hold within noise (~0.04 per class).
  - Result `sim_p6_rgb3` (2026-11-01, 60 epochs, ~7.8 h; `runs/eval/sim_p6_rgb3_val.json`):

    | real val | hituav (real only) | sim_p6 | sim_p6_rgb3 |
    |---|---|---|---|
    | overall mAP50 / 50-95 | 0.596 / 0.316 | 0.608 / 0.327 | 0.601 / 0.344 |
    | cattle (RGB) | 0.904 | 0.898 | 0.936 |
    | RGB flights 070 / 061 / 056 / 032 | .974 / .854 / .834 / .899 | .943 / .872 / .832 / .921 | .982 / .938 / .860 / .937 |
    | RGB recall / alarms per empty frame, conf 0.25 | 0.912 / 0.62 | 0.872 / 0.035 | 0.921 / 0.000 |
    | RGB recall, conf 0.5 | 0.854 | 0.768 | 0.846 |
    | thermal mAP50 | 0.442 | 0.464 | 0.434 |
    | person / other_animal | 0.214 / 0.671 | 0.233 / 0.694 | 0.180 / 0.687 |
    | thermal flights 362 / 58 / 60 | .623 / .154 / .367 | .759 / .300 / .564 | .575 / .211 / .595 |
    | thermal alarms per empty frame, conf 0.25 | 0.169 | 0.149 | 0.164 |

    - RGB sim helps real RGB: cattle 0.936 is the best of any run, every one of the four
      flights is up on both other models, recall is back above the real-only model's,
      and no empty RGB frame gets an alarm. The single gain (+0.03-0.04) is at the noise
      level, but four flights all moving the same way makes it more than noise.
    - Thermal slips: mAP50 0.464 -> 0.434, person 0.233 -> 0.180, almost all of it flight
      362 (0.759 -> 0.575) and 58. 362 has moved this much between runs before (hituav
      0.623, sim_p3 0.713, sim_p6 0.759), so it may be noise; it may also be dilution, as
      the thermal share of train fell from 93% to 73% (13,534 of 18,520). Thermal alarms hold.
    - Overall mAP50-95 0.344 is the best yet (larger and better-placed RGB boxes).
    - Next: the same run with seed 1 (`sim_p6_rgb3_s1`, `train_baseline.py --seed 1`).
      362 back near 0.75 = the slip was noise and RGB v3 joins the base; near 0.58 =
      dilution, and the next try is fewer RGB sim images or more thermal sim.
  - Result seed 1 (2026-11-01, `runs/eval/sim_p6_rgb3_s1_val.json`; early-stopped at
    epoch 50, best 30; seed 0's best was 45 of 60):

    | real val | sim_p6 | sim_p6_rgb3 seed 0 | seed 1 |
    |---|---|---|---|
    | overall mAP50 / 50-95 | 0.608 / 0.327 | 0.601 / 0.344 | 0.592 / 0.336 |
    | cattle (RGB) | 0.898 | 0.936 | 0.894 |
    | RGB flights 070 / 061 / 056 / 032 | .943 / .872 / .832 / .921 | .982 / .938 / .860 / .937 | .951 / .864 / .795 / .927 |
    | RGB recall / alarms, conf 0.25 | 0.872 / 0.035 | 0.921 / 0.000 | 0.876 / 0.028 |
    | thermal mAP50 | 0.464 | 0.434 | 0.441 |
    | person / other_animal | 0.233 / 0.694 | 0.180 / 0.687 | 0.207 / 0.675 |
    | thermal flights 362 / 58 / 60 | .759 / .300 / .564 | .575 / .211 / .595 | .702 / .109 / .372 |
    | thermal alarms, conf 0.25 | 0.149 | 0.164 | 0.191 |

    - Both of seed 0's moves were the seed. Flight 362 came back to 0.70, so the thermal
      slip wasn't dilution. But the RGB gain went too: cattle 0.894, the same as sim_p6,
      and three of the four flights fell back.
    - Seed noise on this val set is larger than the ~0.04 assumed: cattle 0.04, flight
      58 0.10, flight 60 0.22, between two runs on identical data.
    - So in a mixed run, RGB v3 sim neither helps nor hurts measurably. The real RGB
      cows (986 train frames) already put cattle near 0.90; there is little left for sim
      to add on these four flights. sim_p6 is itself one seed, so its 362 (0.759) and
      58 (0.300) may also be lucky.
    - Lesson: one mixed run per change can't judge a sim change. Judge on at least two
      seeds, or where the sim is meant to matter: classes or scenes with little or no
      real data.
  - Scarce-data test (2026-11-01): does RGB sim help when real RGB is scarce?
    - Real RGB train cut to two flights, 039 and 042: 92 of 986 frames, 390 of 3,316
      cows. Neither was flown on the same day as a val flight (055, 060 and 071 were, and
      are left out). All thermal train (real + 4,000 v6 sim) stays.
    - `scarce_rgb` (no RGB sim) against `scarce_rgb_sim` (+ the 4,000 RGB v3 images),
      seeds 0 and 1 each, same settings as sim_p6_rgb3. Train lists on `D:/aar/scarce/`,
      images and val from `D:/aar/processed_sim_p6_rgb3`; configs `runs/scarce_rgb*.yaml`.
    - Judged on real RGB cattle AP50 and RGB alarms, against the full-data runs (~0.89-0.94).
      The gap the sim closes should be well above the seed noise seen above.
  - Result scarce-data test (2026-11-02, `runs/eval/scarce_rgb*_val.json`; best epochs:
    no sim 18 of 38 and 33 of 53, with sim 60 of 60 and 41 of 60):

    | real RGB val | full data (sim_p6, sim_p6_rgb3 s0/s1) | scarce, no sim (s0 / s1) | scarce + RGB sim (s0 / s1) |
    |---|---|---|---|
    | cattle AP50 | 0.898 / 0.936 / 0.894 | 0.505 / 0.598 | 0.764 / 0.774 |
    | cattle AP50-95 | 0.611 / 0.680 / 0.636 | 0.241 / 0.302 | 0.489 / 0.480 |
    | flight 070, dense herd | .943 / .982 / .951 | .429 / .560 | .914 / .845 |
    | flight 061 | .872 / .938 / .864 | .302 / .345 | .486 / .478 |
    | flight 056 | .832 / .860 / .795 | .664 / .696 | .606 / .744 |
    | flight 032, bale field | .921 / .937 / .927 | .866 / .914 | .904 / .907 |
    | RGB recall, conf 0.25 | 0.872 / 0.921 / 0.876 | 0.515 / 0.628 | 0.720 / 0.776 |
    | RGB alarms per empty frame, conf 0.25 | 0.035 / 0.000 / 0.028 | 0.177 / 0.560 | 0.035 / 0.050 |
    | RGB precision, conf 0.5 | 0.968 / 0.980 / 0.963 | 0.715 / 0.746 | 0.899 / 0.852 |
    | thermal mAP50 | 0.464 / 0.434 / 0.441 | 0.516 / 0.441 | 0.440 / 0.462 |

    - The sim carries over when real data is scarce. With 92 real RGB frames, cattle
      AP50 averages 0.55; adding the RGB sim makes it 0.77 on both seeds, against 0.91
      with all 986. That closes about 60% of the gap (AP50-95 too: 0.27 -> 0.48, full
      0.64). Both sim seeds beat both no-sim seeds by 0.17 or more, four times the cattle
      seed noise.
    - Recall goes from 0.57 to 0.75 and empty-frame alarms fall to the full-data level.
    - Most of the gain is the dense herd (070: 0.49 -> 0.88) and 061 (0.32 -> 0.48).
      032 (bale field) is already near full with 92 frames; 056 is mixed (one seed down,
      one up). 061 stays far below full data (0.48 vs 0.87); the next look is what it
      misses.
    - Sim-only v3 (no real RGB, 6 epochs, no thermal) got 0.739; scarce + sim gets 0.77.
      Most of the cattle skill here comes from the sim, and 92 real frames add a little.
    - Thermal is unchanged within noise (scarce_rgb_s0's 0.516 / person 0.309 is the
      best thermal of any run, by chance: same data as s1 at 0.441).
    - So: where real RGB exists (cattle) the sim adds nothing measurable; where it's
      scarce, the sim gives most of what the real data would. That is the case for the
      classes and scenes with no real data at all (horses, sheep, dogs, cats, floods).
    - Caveat: 039 and 042 are both from Jalogny, the farm of val flights 032 and 056.
      That's why 032 is near full with 92 frames. 061 and 070 are from "Other_farms",
      whose train flights (060, 062, 071) were all left out.
  - Why 061 lags (2026-11-02, scratch `flight_misses.py` / `grass_split.py`; cows found
    at conf 0.25, IoU 0.5; "yard" = little green around the box):

    | flight 061, 106 cows | on grass (59) | in the yard (47) |
    |---|---|---|
    | scarce, no sim (s0 / s1) | 42 / 48 | 0 / 0 |
    | scarce + RGB sim (s0 / s1) | 53 / 54 | 0 / 1 |
    | sim-only v3 | 56 | 14 |
    | full data (sim_p6 / sim_p6_rgb3 s1) | 52 / 59 | 27 / 24 |

    - 44% of 061's cows stand in a farmyard: pens with straw and mud bedding, metal rails
      and gates, often half under a shed roof, packed together. The scarce models find
      none of them, with or without sim; full data finds about half (from 060, the same
      farm on the same day). On grass, the sim takes the scarce model from 76% to 91%
      (full data 88-100%).
    - The sim has no farmyard: only open fields. Sim-only still finds 14 yard cows, so
      a cow shape partly carries over; the 92 real pasture frames seem to teach the
      mixed model that cows stand on grass.
    - Next candidate (RGB v4): a farmyard scene. Pens and yards with straw, mud and
      concrete, rails, gates and feed barriers, sheds whose roofs hide part of the
      herd, crowded cows lying and standing. Also the place animals are trapped in a
      flood. Look tuned on the train flights only; judged on 061's yard cows in the
      scarce setting.
    - Yard cows in the train flights (same green test): 060 152, 062 61, Derval 012 62
      and 014 47, Jalogny 058 44 and 059 43; Mauron and 071 none. 039/042 have 9 and 15,
      so the scarce runs saw almost no yard cows. The green test is rough: brown dry
      grass also counts as yard.
  - RGB v4, farmyards (2026-11-02, `yard.rs`):
    - The train-flight yards (060, 062, Derval) have long sheds with light sheet roofs,
      some open-fronted; railed pens in front on straw, mud or concrete; a trampled
      muddy gateway into the field with ring feeders; a silage clamp under black
      plastic held down by tyres; tyre stacks, a manure heap, wrapped bales. Some
      "yard" cows are just bunched in tree shade (Derval); not built.
    - 35% of RGB shots get a yard somewhere in the frame, its sheds along the field
      grid: 1-3 sheds (half open-fronted), a railed pen in front of most (bays, a gate),
      hard standing, mud at gates and sometimes a wide trampled gateway, feeders, a
      clamp and heap off one end, wrapped bales in a row. Sheds, rails (3 m panels),
      feeders and stacks are 3D and cast shadows; the ground is painted. Roofs and walls
      carry a weathered sheet texture; one yard's rails share one paint.
    - In 80% of yard shots the herd moves into the yard, most of it into one area (a
      pen, the front 4 m of an open shed where the roof hides part of it, or the mud),
      centres at least 1.4 m apart. Hedges, bushes, houses and bales stay out of it.
    - Thermal unchanged (md5 of the 24-frame seed-3 pool). Pools
      `D:/aar/interim/sim_rgb4_w10..17` (seeds as v1).
    - Judged like v3: zero-shot and sim-only first (cheap), then the scarce setting with
      the v4 sim on seeds 0 and 1 against scarce + v3 (cattle 0.764 / 0.774; 061 yard
      cows 0 / 1 of 47).
    - Pools: 35% of shots have a yard (162-187 per world), 30% of bodies stand in one.
  - Result RGB v4, cheap checks:
    - Zero-shot `hituav` on world 10 (v3 in brackets): cattle found tiny 74% (72%),
      small 78% (91%), medium 62% (81%); alarms per empty frame 1.65 (1.53). Yard cows
      are harder for the real model too (it finds about half of 061's). Its top alarms
      on v4 are clutter look-alikes: bushes, vehicles, wrapped bales, troughs, tanks.
    - Sim-only, 6 epochs, real RGB val (v3 in brackets; RGB frames only):

      | | sim-only v3 | sim-only v4 |
      |---|---|---|
      | RGB mAP50 | 0.739 | 0.729 |
      | flights 070 / 061 / 056 / 032 | .966 / .585 / .474 / .535 | .948 / .622 / .304 / .864 |
      | alarms per empty frame, conf 0.25 / 0.5 | 5.6 / 0.65 | 0.53 / 0.035 |
      | precision, conf 0.5 | 0.72 | 0.89 |
      | 061 cows found, grass / yard | 56 / 14 | 51 / 11 |

    - Alarms fall tenfold and the bale field (032) goes from 0.54 to 0.86: the yards'
      clutter teaches background. Overall cattle ties (0.73).
    - Yard cows don't improve (11 vs 14 of 47). Caveats: 061's 47 yard boxes are 17
      frames of essentially one pen, so this is one scene; and a 6-epoch sim-only run
      is noisy. Side by side, the real pen holds cows packed into small stalls, often
      lying flank to flank; sim yard cows are spread over a bigger pen. Not tuned
      further against that one val scene.
    - (`evaluate.py`'s per-class cattle AP also counts cattle boxes on thermal frames;
      from here on the RGB-only `by_modality` number is the cattle figure. For v1-v3
      they were within 0.01; for v4 0.667 vs 0.729.)
  - Scarce test with v4 (2026-11-03): the same 92 real RGB frames + all thermal, with the
    4,000 v4 RGB sim images in place of v3; seeds 0 and 1, 60 epochs. RGB-only numbers on
    val:

      | | no sim (s0 / s1) | + v3 sim | + v4 sim |
      |---|---|---|---|
      | cattle AP50 (RGB) | .505 / .599 | .766 / .774 | .832 / .839 |
      | flight 070 | .429 / .560 | .914 / .845 | .924 / .950 |
      | flight 061 | .302 / .345 | .486 / .478 | .651 / .657 |
      | flight 056 | .664 / .696 | .606 / .744 | .731 / .701 |
      | flight 032 | .866 / .914 | .904 / .907 | .903 / .891 |
      | recall / alarms per empty frame, conf 0.25 | .52-.63 / .18-.56 | .72-.78 / .035-.050 | .81-.82 / .050-.156 |
      | recall / alarms, conf 0.5 | .45-.56 / .04-.16 | .61-.65 / .014-.021 | .68-.72 / .007-.014 |
      | 061 cows found, grass / yard (of 59 / 47) | | 53-54 / 0-1 | 58 / 7 |
      | thermal mAP50 | .516 / .441 | .440 / .462 | .442 / .443 |

    - v4 beats v3 on both seeds: cattle +0.065 (seed noise 0.04, and the two seeds sit
      within 0.01 of each other), 061 +0.17, recall +0.05-0.10 at both thresholds, with
      alarms at the same low level. Thermal unchanged. Full-data real is about 0.91.
    - 061 yard cows go from 0-1 to 7 of 47 on both seeds; most of 061's gain is grass cows
      found more surely (58/59) and ranked better. Packed-stall cows remain the gap.
    - v4 is the RGB sim from here on.
- RGB v5: packed yard herds (2026-11-03). Measured on the train flights' yards (060, 062,
  Derval 012/014, Jalogny 058/059; every third frame; a cow is "yard" if the ring round its
  box isn't green, so cows in tree shade count too): cows stand close and line up. In the
  pictures: rosettes of 6-12 heads-in round ring feeders, rows side by side in pens and
  along open-shed fronts, packed lines in tree shade. v4 spread them out with random
  headings.
  - `yard.rs`: feeders have a kind (ring, trough, tank; at most 6/4/2, each in fixed mesh
    slots so the shot knows where the rings are). `Yard::herd_slots` gives a packed herd's
    places: 70% of the time when the home area has a ring feeder, 7-10 standing places
    round it, heads in (the rest bunch behind); else rows facing a pen edge (side gap
    1.0-1.6 m, row gap 2.5-3.3 m, +-0.25 rad), looser on mud (1.3-2.0 m, +-0.6 rad, 30%
    turned round). Slots keep off feeders and bay rails.
  - `scene.rs`: 85% of yard herds pack (`YARD_PACKED`); people never take a slot;
    scattered bodies keep off feeders. Yard herds are cattle 70% of the time and bigger
    (the larger of two draws of the herd size). Nothing but the yard's own clutter stands
    on its mud (a bush grew in one rosette). The RGB ground is drawn before the targets
    now; thermal draws nothing there, and the 24-frame thermal check is byte-identical.
  - Packing, nearest neighbour in body lengths and group size per cow (linked within 1.5
    lengths); sim measured on frames whose bodies are all in the yard (the green test
    misreads sim pens next to bright grass):

      | | real train yards | v4 | v5 trial (400 frames) |
      |---|---|---|---|
      | NN median | 0.84 | 1.41 | 0.71 |
      | NN < 0.8 / < 1.0 / < 1.5 | 44 / 70 / 79% | 16 / 27 / 54% | 57 / 67 / 81% |
      | group: median, alone | 6, 23% | 2, 49% | 4, 21% |
      | group >= 5 / >= 10 | 51 / 38% | 15 / 0% | 47 / 14% |

    Groups of 10+ stay short of real (some real ones are tree-shade lines, not yards).
  - Scarce test with v5 (2026-11-03; same 92 real RGB frames + all thermal, 4,000 v5 sim
    in place of v4, seeds 0 and 1):

      | | + v4 sim (s0 / s1) | + v5 sim (s0 / s1) |
      |---|---|---|
      | cattle AP50 (RGB) | .832 / .839 | .792 / .812 |
      | flight 070 | .924 / .950 | .904 / .972 |
      | flight 061 | .651 / .657 | .573 / .663 |
      | flight 056 | .731 / .701 | .659 / .486 |
      | flight 032 | .903 / .891 | .925 / .933 |
      | recall / alarms per empty frame, conf 0.5 | .68-.72 / .007-.014 | .60-.67 / 0-.007 |
      | 061 cows found, grass / yard (of 59 / 47) | 58 / 7 | 52-56 / 2-6 |
      | thermal mAP50 | .442 / .443 | .490 / .474 |
      | best epoch (of run) | 23 (43) / 22 (42) | 11 (31) / 58 (60) |

    - Not better: cattle -0.03 to -0.04 on both seeds (seed noise 0.04), and the yard cows
      it was built for go down (2-6 vs 7 of 47). Fewer alarms, lower recall; 032 (bales)
      up a little, 056 down on seed 1.
    - Caveat: the checkpoint is picked on all-class fitness, and thermal moved (0.44 ->
      0.48 with no thermal change), so which epoch was kept differs a lot (11 vs 58); the
      RGB number carries that noise too.
    - Kept in code (packing is closer to the real yards), but v4 stays the RGB pool for
      training. Packing wasn't what kept 061's pen cows hidden; what's left there is
      the look (lying under roof shade, cut by rails), with one val scene to judge it.
- Real thermal animals from BAMBI (2026-11-04). Until now real val had no thermal farm
  or forest animals, so the thermal animal sim was never measured on real animals.
  BAMBI (CC BY 4.0, nadir thermal video over Austrian forests, meadows and alpine
  slopes; source `bambi` in configs/sources.yaml) has deer, boar, ibex, chamois and
  people, with a published split that keeps places apart. Its val (37 flights, 648
  frames, 2,485 animal and 189 person boxes, mostly 16-96 px) is the new measure.
  Zero-shot on it, before any BAMBI training (runs/eval/bambi_zs/, not in git):

      | model | animal AP50 | person AP50 | found IoU .5 small / medium (conf .1) | found IoU .1 | alarms / empty |
      |---|---|---|---|---|---|
      | hituav (real only) | .032 | .014 | .15 / .19 | .36 / .38 | 0 |
      | split2 (real only, older split) | .050 | .011 | | | |
      | sim_p3 (real + thermal sim v3) | .048 | .014 | | | |
      | sim_p6 (real + thermal sim v6) | .052 | .012 | .17 / .23 | .41 / .40 | 0 |
      | sim_p6_rgb3 / _s1 | .054 / .025 | .007 / .005 | | | |
      | sim_only_p6 (sim only) | .071 | .000 | .20 / .37 | .51 / .57 | 12 (of 13) |

    - Every model fails on BAMBI: the real thermal training data (BIRDSAI African nights,
      HIT-UAV towns) is far from European forest. The thermal sim in mixed runs adds
      nothing measurable (.03-.05, seed noise as large).
    - Sim-only finds the most animals (medium .37 vs .19-.23), so the sim's scenes are
      closer to BAMBI than the real data is, but it fires 12 times per empty frame, and
      mixed with real data the gain is gone. Many hits are boxes of the wrong size
      (found at IoU .1 is twice IoU .5).
    - Val above includes flight 221, found later to be a second processing of flight 57
      (same video); it is now left out (val 36 flights, 632 frames, 2,403 animal boxes).
      The 16 doubled frames don't change the picture.
    - Full build (D:/aar/processed_bambi): real sources + BAMBI 1,582 train / 632 val /
      1,587 test frames from 55 / 36 / 65 flights.
    - Trained `bambi` (real + all BAMBI train, 60 epochs, best epoch 43; reaped at epoch
      41 and resumed). On BAMBI val (632 frames, runs/eval/bambi_val2/), same frames for all:

          | model | animal AP50 | person AP50 | P / R at conf .25 |
          |---|---|---|---|
          | hituav | .019 | .014 | .08 / .03 |
          | split2 | .049 | .011 | .10 / .04 |
          | sim_p6 | .044 | .014 | .12 / .04 |
          | sim_only_p6 | .072 | .000 | .02 / .04 |
          | bambi | .498 | .211 | .74 / .42 |

      By size: small (16-32 px) .28, medium .39. Per flight median .40; the 6 flights near
      0 hold 39 boxes in all. On the old real val it holds: .588 vs split2 .573 (person
      .218 vs .145, cattle .875 vs .889, other_animal .670 vs .686), alarms per empty
      image at conf .25 .12 vs .23. Single seed.
    - So 55 flights of in-domain data take animals from .05 to .50: the gap was the data,
      not the model. Next: the scarce test, a few BAMBI train flights with and without
      the thermal sim, to see if the sim helps where in-domain data is short.
    - Scarce test set up (2026-11-05): BAMBI train cut to flights 20 (Mageregg, red
      deer + people), 79 (Purkersdorf, boar + people), 135 (Mairist enclosure, boar), 227
      (Zollfeld, roe deer), 252 (Oberwollanig enclosure, red deer): 131 frames, 469 boxes,
      one flight per main place/species, picked from data stats only. All other real
      train kept. `scarce_bambi` (10,651 images) vs `scarce_bambi_sim` (+ the 4,000 v6
      sim images of processed_sim_p6), seeds 0 and 1, same settings as `bambi`. Lists on
      D:/aar/scarce/, configs runs/scarce_bambi*.yaml. Judged on BAMBI val animal and
      person AP50 (full BAMBI .498 / .211, no BAMBI ~.05 / .01).
    - Result (2026-11-06, runs/eval/bambi_val2/ and runs/eval/full_val/, 60 epochs each):

          | BAMBI val | animal AP50 | person AP50 | small / medium | flight median |
          |---|---|---|---|---|
          | no BAMBI (split2) | .049 | .011 | .02 / .06 | |
          | scarce, no sim (s0 / s1) | .184 / .137 | .030 / .057 | .02-.04 / .15-.16 | .08 / .04 |
          | scarce + sim v6 (s0 / s1) | .247 / .195 | .053 / .054 | .05-.06 / .16-.22 | .14 / .09 |
          | full BAMBI (bambi) | .498 | .211 | .28 / .39 | .39 |

      Full val (all sources): scarce .525 / .530, scarce + sim .555 / .540, full .588;
      alarms per empty image .11-.13 for all.
    - The thermal sim helps where in-domain data is scarce: animals +.06 on both seeds
      (paired), small animals doubled, better on 14 of 25 val flights with >= 20 boxes
      and worse on 3. But it closes only ~18% of the gap to full BAMBI (.16 -> .22 of
      .50), against ~60% for the RGB sim with cattle. People don't move (.03-.06).
    - So 131 real frames from 5 flights are worth far more than 4,000 sim images, and
      the sim's European-forest animals are still far from real ones. Candidates: deer
      and boar shapes and sizes seen from straight above, canopy occlusion (BAMBI keeps
      half-hidden animals labelled), forest-floor thermal texture. BAMBI val is the
      cheap measure for each change (sim-only zero-shot, then this scarce setting).
- Forest sim (2026-11-06; `datagen --modality forest`, `src/forest.rs`, commit b97b6b4).
  BAMBI train frames next to v6 showed what was missing: v6 ground was flat with smooth
  blobs, its animals small and grey, half its views oblique, and no black bands. BAMBI
  is straight down over busy ground (grass, leaf litter, snow, fields, bare branch
  networks, evergreen crowns, shadows, tracks), with near-white animals in herds, stored
  1024x1024 with black bands bowed by undistortion (rows 74-102 and 912-940, the same
  on every flight). Forest shots now:
  - straight down (87-90 deg), short side 15-60 m, a herd of 1-12 deer or boar (deer
    0.9-1.5x the roe-sized primitive), sometimes people and a dog, 12% empty;
  - ground painted per shot as temperatures (2048 texels over the footprint): one or
    two covers with an edge (sometimes scrub along it), bare trees as forking twig
    networks (mostly warm), lumpy evergreen crowns, their shadows by day, tracks with
    ruts, logs, stumps and rocks (12% as warm as an animal, like BAMBI's "No-animal");
  - bodies 3-25 C above the ground under them, the window topped 1-8 C above the
    ground's 99.9th percentile, so most come out white;
  - a cleaner camera (`SensorParams::forest`), then the BAMBI warp (`bambi_frame`).
  Image statistics against BAMBI train (500 frames each): noise 1.0 / 1.0, p1-p99
  0-180 / 0-174, animal side 38 / 42 px, animal contrast 160 / 157 grey levels
  (v6: 25 px, 75).
  - Round 1 (sim-only, 4,000 images = 8 worlds x 500, 6 epochs, BAMBI val): animal AP50
    .095 (v6 .072), precision / recall at conf .25 .25 / .16 (v6 .02 / .04). On BAMBI
    train frames the model found most visible animals, but its boxes were much tighter
    than BAMBI's hand-drawn ones: labels 1.48x wider and 1.35x taller than its boxes,
    5-7 px per side of 1024 at every size. A tight box fails IoU .5 twice (a miss and an
    alarm).
  - Round 2: forest label margin 1 -> 5 render px per side (same seeds: images byte for
    byte the same, only the boxes grow).

        | BAMBI val | animal AP50 | AP50-95 | person | small / medium | P / R at .25 |
        |---|---|---|---|---|---|
        | sim only v6 | .072 | .019 | .000 | .00 / .05 | .02 / .04 |
        | sim only forest round 1 | .095 | .019 | .007 | .01 / .09 | .25 / .16 |
        | sim only forest round 2 | .318 | .129 | .059 | .08 / .23 | .37 / .34 |
        | sim only forest round 3 | .450 | .180 | .028 | .15 / .26 | .70 / .32 |
        | ... same images, training seed 1 | .438 | .168 | .068 | .17 / .25 | .80 / .24 |
        | sim only forest round 4 (tussocks + faint shots) | .401 | .159 | .012 | .11 / .24 | .68 / .26 |
        | sim only forest round 4b (tussocks only) | .459 | .184 | .013 | .14 / .27 | .69 / .35 |
        | sim only forest round 5 (crowns overhead) | .488 | .198 | .046 | .17 / .29 | .66 / .41 |
        | sim only forest round 6 (BAMBI-sized animals) | .481 | .193 | .063 | .19 / .28 | .55 / .48 |
        | sim only forest round 7 (man-made things) | .508 | .209 | .010 | .16 / .29 | .60 / .48 |
        | sim only forest round 8 (camera detail enhancement) | .505 | .195 | .028 | .20 / .28 | .61 / .48 |
        | sim only forest round 9 (animals' shadows) | .540 | .213 | .037 | .20 / .29 | .61 / .52 |
        | sim only forest round 10 (big herds) | .529 | .206 | .048 | .20 / .31 | .63 / .49 |
        | sim only forest round 11 (BAMBI's herd sizes; dropped) | .506 | .202 | .046 | .20 / .28 | .67 / .44 |
        | sim only forest round 12 (herds anywhere, edge animals boxed whole) | .540 | .221 | .022 | .19 / .30 | .64 / .48 |
        | sim only forest round 13 (sunlit relief on the ground; dropped) | .515 | .205 | .007 | .17 / .28 | .66 / .44 |
        | real + 5 BAMBI flights (scarce s0) | .184 | .061 | .030 | .02 / .16 | .42 / .18 |
        | ... + v6 sim (scarce_sim s0) | .247 | .090 | .053 | .05 / .22 | .52 / .20 |
        | real + all BAMBI (bambi) | .498 | .195 | .211 | .28 / .39 | .74 / .42 |

    - Sim alone, no real image, now beats real data plus 5 BAMBI flights. Most of the
      jump is the label margin, which only pays off because the model now finds the
      animals (round 1's recall .16 at IoU .5 was mostly right animals, wrong box).
    - The margin is set from BAMBI train (box sizes, not detection scores); val was only
      scored. Alarms on BAMBI val's 13 empty frames: 1.0 per frame (too few frames to
      judge).
  - Round 3, three changes at once, from round 2's misses on BAMBI train frames:
    - a leaf canopy: an alpha-masked plane 8-20 m up (at most 0.6 x altitude), painted as
      clumps of foliage with gaps, over 60% of litter shots, 30% of snow, 15% of meadow and
      5% of field; 30-92% cover, mostly a few degrees warmer than the ground. Animals
      under it are hidden in the id render too, so only the visible part is labelled.
    - mottled bodies: the body takes the ground texture times a gain of 0.05-0.3.
    - closer shots: short side of the footprint 10-60 m (round 2: 15-60 m).
  - Animal AP50 .318 -> .450 and precision at conf .25 .37 -> .70 at about the same
    recall. Alarms on the 13 empty val frames fell 1.0 -> 0.08 per frame. Person dropped
    .059 -> .028 (189 boxes, 73 of them in one flight). Still missed on BAMBI train frames:
    half of a packed herd (many lying, round), small animals under bright sunlit foliage,
    and grey animals that are barely warmer than the ground.
  - Seed noise: round 3's images trained with seed 1 instead of 0 score .438 animal
    (.450), but person .068 (.028): a 6-epoch run moves animal AP50 by about .01 and
    person by several hundredths, so person is not judged on these runs.
  - Round 4, two changes against round 3's misses; the sim's ground was much smoother
    than BAMBI's up close:
    - tussocks, leaf heaps and stubble: clumps 8-30 cm across (6-20 cm stubble, 10-40 cm
      litter), up to 0.2-5 °C warmer than the cover between them (30% of covers colder:
      frost), on hashed noise that never repeats; none on snow.
    - faint shots: in 20%, every body only 1-5 °C warmer than the ground.
    - Animal AP50 fell .450 -> .401, recall at .25 .32 -> .26. Round 4b, tussocks only:
      .459, recall at .25 .35 and at .5 .24 (round 3: .13). The faint shots cost about .05
      and were dropped; the tussocks hold AP and make the model surer, so they stay.
      (Round 4b's renders were made with the faint share set to 0, which still drew one
      random number per shot, so the kept code doesn't redraw them byte for byte.)
  - Recall of round 4b on 600 BAMBI train frames (2,108 animal boxes, conf .25, IoU .5),
    by what is around each box: crowding made no difference (0 neighbours .40, 3-5 .44),
    but busy surroundings did: with the ring of one box-side around it at a grey-level
    sd of 35 or more (a third of the boxes) recall was .29, against about .5. The sim drew
    its trees flat into the ground, while in BAMBI the drone looks down through the
    crowns, so twigs cross the animals. A few flights have label quirks that cap recall
    on their own (a whole herd in one box; boxes in the black bands or on empty ground).
  - Round 5, trees overhead: conifer crowns and bare branches move from the ground into
    the canopy layer (now 2,048 texels across, as fine as the ground), over the leaves if
    any, 8-20 m up (at most 0.6 x altitude). Twigs and crowns hide what is under them in
    the id render, so a body fully under a crown has no label. The picture blends the
    layer by its alpha, so twigs thinner than a pixel show faintly (the id look cuts at
    half); a clear texel beside a mark takes its temperature, so filtering leaves no halo.
    Up close the crowns grow larger than the ground, and toward the frame's edges they
    slide outward, as BAMBI's trees do.
  - Animal AP50 .459 -> .488 (3x the seed noise), recall at .25 .35 -> .41, AP50-95 .198:
    sim alone is now within .01 of real data plus all of BAMBI train (.498). On the 600
    train frames recall .415 -> .449; busy surroundings .29 -> .36, the brightest animals
    .37 -> .50. Weakest now: animals that are less than 20 grey levels brighter than
    their surroundings (a third of the boxes, recall .37) and small ones (<24 px: .12).
  - Round 6, BAMBI-sized animals: the sim's boxes were too big (side at 1024 median 58 px,
    41% over 64 px; BAMBI train median 42, 10th-90th percentile 26-57, 3% over 64). The
    footprint's short side goes from 10-60 m to 20-90 m (log-uniform), which gives 31-63
    px, median 40.
  - Animal AP50 .488 -> .481 (within the seed noise), but the trade moved: recall at .25
    .41 -> .48, precision .66 -> .55; small boxes .17 -> .19, medium .29 -> .28. On the
    600 train frames recall .449 -> .500 (24-40 px .41 -> .44, 40-64 px .51 -> .59, low
    contrast .37 -> .44; <24 px and busy surroundings unchanged), and false alarms (a
    detection overlapping no label at IoU .3) .38 -> .83 per frame. Kept as the base: the
    sizes are BAMBI's and the misses it fixes are real; the alarms are the next target.
  - What the alarms are, on train frames: man-made things with hard edges (a row of
    animal-sized hot objects along a hedge, seen in every frame of one flight; roofs and
    high-seats; things on paved paths), bright branch forks, and label quirks (a whole herd
    in one box, so each animal found is an alarm). The sim had nothing hard-edged.
  - Round 7, man-made things in 35% of the shots: hard-edged, flat-coloured rectangles and
    discs, antialiased over a texel. A row of 2-8 bales or boxes (half the time beside a
    track), 1-4 benches, feeders or boxes, a high-seat (a two-slope roof on stilts), a car
    (warm body, cold windscreen) or a roof 5-16 m long; 70% warmer than the ground by
    1.5-14 °C (log-uniform), the rest 1-5 °C colder. By day each throws a shadow away
    from the sun: joined to it, or off to the side for a roof on stilts.
  - Animal AP50 .481 -> .508 (3x the seed noise), AP50-95 .209, precision at .25 .55 ->
    .60 at the same recall: sim alone now passes real data plus all of BAMBI train
    (.498). On the 600 train frames false alarms fell .83 -> .62 per frame (round 5:
    .38) and recall .500 -> .463; busy surroundings are still the weakest (.33).
    Person .010: one flight holds 73 of the 189 person boxes, and person moves by
    several hundredths between seeds, so it is not judged here.
  - What the busy-surround misses are (round 6 model, 764 boxes with ring sd >= 35): not
    faint (contrast median 59, hits 55) but on brighter ground (ring median 85 vs 68),
    three in four with no detection near. Mostly flights 249, 250, 252 and 143: crisp
    bodies drawn as bright rims round greyer middles, on rough ground of clumps as white
    as the animals, the look of a thermal core's detail enhancement (DDE). The sim's
    ground never came near an animal's brightness, so its animals were always the
    brightest thing in view.
  - Round 8, detail enhancement in half the forest shots (`sensor::enhance`), after the
    optics and before the noise: the image's blur over 1.5-6 px (the base) is histogram
    equalized within its own range, blended 30-90% of the way, and the rest (the detail)
    added back boosted 1.5-5x. Equalizing squeezes the empty range between the ground
    and a small warm body, so its middle comes out little brighter than the brightest
    ground and its rim white. (A plain unsharp mask was tried first: the render already
    clips bodies white, so boosting only pushed them further over.)
  - Animal AP50 .508 -> .505, AP50-95 .209 -> .195 (both within or near the seed noise:
    round 3 moved .012 in AP50-95 between seeds); small .16 -> .20, medium .29 -> .28;
    no alarm on the 13 empty val frames (round 7: 0.15). On the 600 train frames it did
    what it was built for: flights 250 .10 -> .26, 252 .10 -> .27, 143 .39 -> .51, but
    flight 17 .67 -> .44; overall .463 -> .472, alarms .62 -> .59 per frame. Kept: val
    holds, and the camera is more like BAMBI's.
  - Found while adding body shadows: ground marks (crown, slab and debris shadows) blend
    toward the shot's base temperature plus their offset, keeping 0.4 of the ground's
    texture, so on a cover colder than the base (snow, 2-6 °C below) a shadow comes out
    warmer than the snow around it. Left as is for now (changing it would confound the
    rounds); the body shadows keep the whole local texture and so always cool.
  - Round 9, animals' shadows: by day (sun in 49% of shots) every deer and boar throws a
    shadow on the ground, painted after the bodies are placed (`ForestLook::
    body_shadows`): a capsule the body's length and width, lifted off its feet by the
    height of its middle and stretched by its depth, away from the sun (at most 4 m per
    metre of height, as the crowns' shadows), 0.8x as cold as a crown's shadow, keeping
    the ground's texture. BAMBI's sunlit deer (flight 249) all have dark shadows beside
    them; the sim's had none. People throw none (rare in BAMBI; a thin line).
  - Animal AP50 .505 -> .540 (3.5x the seed noise), AP50-95 .213, recall at .25 .48 ->
    .52 and at .5 .38 -> .43 at the same precision.
  - Next: big herds. BAMBI train has 71% of its animal frames with 1-3 animals, 24% with
    4-12 and 5% with 13 or more, but by boxes 30% / 45% / 25%: a quarter of the animals
    stand in herds of 13 or more (up to 59, flight 17). The sim's pool held 12 deer and
    12 boar, its herds uniform 1-12 (frames 23% / 77% / 0%, boxes 6% / 94% / 0%).
  - Round 10, big herds: the forest pool holds 28 deer and 28 boar (the forest's id
    render steps red by 4, not 8, to fit them; `Modality::id_step`), and one herd in ten
    is 13-28 strong, spread out as the root of its size. Labelled frames with animals
    now split 29% / 61% / 10% by herd size, boxes 8% / 64% / 28% (BAMBI train 30% / 45% /
    25%).
  - Animal AP50 .540 -> .529, AP50-95 .213 -> .206 (both within the seed noise); recall
    at .25 .52 -> .49, precision .61 -> .63; medium .29 -> .31. On the 600 train frames
    recall .516 -> .477 (40-64 px .62 -> .55, under 24 px .11 -> .05), but flight 17,
    the 59-deer herd, .49 -> .74, and false alarms .85 -> .49 per frame. Kept: val
    holds, and the herds are BAMBI's.
  - The single animals are still missing: half of the 600 train frames' animal boxes
    have no other box centre within 1.5 sides of theirs (recall .44, against .50-.54
    with neighbours), and 71% of BAMBI train's frames with animals hold 1-3 of them, the
    sim's 29%.
  - Round 11, herd sizes like BAMBI train's: one herd in twenty 13-28, else exp(u^2 ln 13)
    (half alone, three in four 1-3); 3.0 boxes per frame instead of 5.3. Animal AP50 .529
    -> .506 (twice the seed noise), recall at .25 .49 -> .44. On the 600 train frames the
    single animals didn't gain (.44 -> .43) and everything else fell (.477 -> .422; the
    rimmed flights 250 and 252 to .05). Dropped: fewer animals per shot teach less, and
    the herd sizes stay as in round 10.
  - Not the margin: by the bright blob inside each box, BAMBI's labellers leave the same
    margin as the sim at every blob size (about 9-11 px per side of 1024 for 12-30 px
    blobs), only less evenly (middle half 7-15 px against 8-10). Its boxes under 24 px
    (8%) are tight boxes from a few flights (140, 142), which no model can tell from the
    image; the model's boxes look squeezed toward 40 px only when binned by the label's
    size (regression to the mean: binned by its own size, it tracks the labels).
  - At the frame's edge: BAMBI boxes an animal cut by the edge whole, on into the black
    round the picture (some boxes stay on after the animal has left), and its margins run
    into the black too; the sim clipped both at the picture. 8% of BAMBI train's animal
    boxes are a quarter or more black (sim 0.9%), and round 10 finds few of the boxes
    that touch the black at all: recall .34 under 10% black, .36 at 10-25%, .13 at
    25-50%, 0 past half (437 of 2,108 boxes; .54 for the rest). The sim also kept its
    herds near the middle: 2-3% of its boxes are centred in each edge tenth of the frame,
    BAMBI's 8%.
  - Round 12, on round 10: the herd's centre is drawn anywhere in the frame (along the
    frame's own axes, not the world's), and an animal cut by the frame's edge is boxed
    whole: each side of its visible box on the edge moves out to its whole body's
    (`scene::body_extent`: torso ends and head as spheres, projected through the
    camera), and margins run on into the black, clipped only at the 1024 canvas. Boxes
    a quarter or more black 0.6% -> 3.4% (BAMBI 8%, some of which are boxes left on
    after the animal has gone); boxes per frame 5.5 -> 4.8.
  - Animal AP50 .529 -> .540, AP50-95 .206 -> .221 (the best yet), precision at .25 .63
    -> .64, recall .49 -> .48. On the 600 train frames the edge boxes didn't clearly
    gain: the model's boxes now reach into the black where the label is 10-25% black
    (median 12% black, round 10 5%), but recall under 10% black fell .34 -> .25, most of
    those boxes in the rimmed flights 250, 252 and 143, which swing from round to round
    (252: .38, .23, .05, .17 in rounds 9-12). Kept: val gains on both measures, and the
    herds and labels are BAMBI's.
  - Round 13, on round 12, sunlit relief: BAMBI's sunny meadows in the rimmed flights
    look embossed (each tussock and hummock bright on its sunward side, dark on the
    other, nearly as white as the animals), the sim's evenly warm. In sunny shots the
    ground's slopes facing the sun were warmed and those facing away cooled (0.5-6 C
    times the sun's height, on the tussocks and on new 0.3-1 m hummocks; not on snow).
    Animal AP50 .540 -> .515, AP50-95 .221 -> .205, recall at .25 .48 -> .44; on the
    train frames recall .46 -> .42, and the rimmed flights it was aimed at fell too (250
    .11 -> .05, 252 .17 -> .08, 143 .39 -> .32). Dropped.
- RGB flood and debris scenes; cats.
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
