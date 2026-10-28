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

Status (2026-10-28): phase 0 done; phase 1 renderer done, 4,000-image pilot pool built,
training experiment next.

Build and run (from `sim/`; the first release build of Bevy + CortenForge takes ~18 min):

    cargo run --release --bin sim_view                 # Space: new shot, I: id look
    cargo run --release --bin datagen -- --out ../runs/sim/test --frames 50 --seed 0
    cargo test --release

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

Pilot pool: `data/interim/sim_p1_w10` ... `sim_p1_w17` (8 worlds x 500 frames).

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
