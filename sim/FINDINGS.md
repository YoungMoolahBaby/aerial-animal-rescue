# CortenForge 0.9.0: findings from the storm sim

Feedback gathered while building the storm sim (`sim/`) on the published crates, used as
an outside user would: crates.io sources and docs only. Same format as the friction log
in cortenforge-play (`docs/FINDINGS.md` there), so the two can be read side by side; this
one covers what a perception and vehicles-in-weather workload runs into (terrain, wind
and fluid forces, MJCF/URDF vehicles, rendering and capture next to Bevy).

How to read it:
- **Dates** are when a finding was logged. Newest at the bottom of each section.
- **The friction log** is grouped by crate. Each entry starts with its type and date:
  **bug** (wrong results or a crash), **docs** (missing or misleading docs), **API**
  (works, but awkward or missing a piece), **perf**, **setup**, or **works** (worth keeping
  as is). Paths are under the crate's `src/` with line numbers, from the 0.9.0 sources in
  `~/.cargo/registry/src/*/cortenforge-*-0.9.0/`. Bugs and doc gaps are the upstream fix
  list.
- **Suggestions** are features this workload wanted that no crate has, with what we built
  instead.

## Build notes

### Phase 0 (cf-geometry + Bevy 0.19.1, 2026-10-28): works, slow first build

- Used: `cf_geometry::HeightFieldData` for the terrain (generation, sampling, normals,
  vertex grid) and for line-of-sight checks; Bevy renders and captures. Compiled first
  try with no warnings next to Bevy 0.19.1, same as in cortenforge-play.
- First release build: 17 min 43 s on 6 of 12 threads (Ryzen 5 5600), most of it the
  sim family that phase 0 doesn't use (see the `setup` entry below). Changing the facade
  features afterwards (`default-features = false, features = ["sim"]`) rebuilt part of
  the tree again for more than 10 min.
- 8 unit tests (terrain, line of sight, capture) pass in 0.01 s.

### Phase 1 (thermal pilot, 2026-10-28): CortenForge unchanged, incremental builds fast

- Still only `cf_geometry::HeightFieldData` (now 641 x 641 vertices, 1.6 km); the rest of
  phase 1 is Bevy (temperature materials, id render) and our own Bevy-free modules
  (`thermal.rs`, `sensor.rs`, `capture.rs`). Incremental release builds of our crate take
  seconds; 18 unit tests pass in 1.6 s.
- Phase 0's hand-written terrain line of sight was dropped: the id render now decides
  what is visible. cf-geometry already had what it reimplemented (see the `cf-geometry`
  docs entry), which phase 3 (drone sees target?) should use.

## Friction log

### `cortenforge` (facade) and docs overall

- **API** (2026-10-27): the 0.6 capture / vision / data-contract / dataset crates that the
  colonoscopy sim (CortenForge-Archived, Deep-Poo-Archived) used for screenshot +
  bounding-box datagen are yanked and have no 0.9 counterpart, so sim-to-dataset capture
  has to be rebuilt by each app. Found while planning; see Suggestions.
- **setup** (2026-10-27): no way to depend on a small slice. The facade's features are
  `sim`, `mesh` and `fabrication` (cortenforge Cargo.toml `[features]`), and the `sim`
  umbrella (cortenforge-sim 0.9.0) has no features at all: its 11 crates (sim-core,
  -coupling, -mjcf, -ml-chassis, -opt, -rl, -soft, -therm-env, -thermostat, -types, -urdf)
  always build, with faer, gemm and npyz behind them. Phase 0 here only uses
  `cf_geometry::HeightFieldData` and still compiles soft bodies, the thermostat and RL.
  Per-crate features on the umbrella (e.g. `sim = ["core", "mjcf"]`) would cut the first
  build a lot. Workaround: `default-features = false, features = ["sim"]` drops mesh and
  fabrication at least.

### `cf-geometry`

- **works** (2026-10-27): `HeightFieldData::from_fn` / `sample_clamped` / `normal_clamped`
  / `vertex_position` (heightfield.rs:119, 247, 285, 366) were all the terrain needed;
  reachable as `cortenforge::cf_geometry::HeightFieldData`. Z-up with x/y on the ground;
  a Y-up renderer (Bevy) has to swap axes, which flips handedness: worth one line in the
  type's docs, since the swap mirrors the mesh and turns triangle winding inside out.
- **docs** (2026-10-28): `query::ray_cast` handles `Shape::HeightField` (ray marching with
  binary-search refinement, query/ray_cast.rs:17, 479), but nothing on `HeightFieldData`
  (heightfield.rs) points to it, so phase 0 wrote its own sampled line-of-sight check. A
  "see also: `ray_cast` with `Shape::height_field`" line on the type would have found it.
- **API** (2026-10-28): no height field -> triangle mesh export. Every renderer needs one
  (positions, indices, normals, maybe UVs); here it is `Terrain::vertices` plus
  `scene::terrain_mesh`, a loop over `vertex_position` and `normal_clamped`. An
  `HeightFieldData::to_indexed_mesh()` (cf-geometry already has `IndexedMesh` with
  `compute_vertex_normals`, mesh.rs:28, 291) would cover it.

### `sim-core`

### `sim-mjcf` / `sim-urdf`

### Platform

## Suggestions

- **Synthetic-data capture** (2026-10-27): a small, renderer-agnostic crate for
  "simulated scene -> labelled dataset": per-object ids, visible-pixel masks -> tight
  boxes with an occlusion threshold, run manifests, and writers for common formats
  (YOLO txt, COCO json). The archived 0.6 crates covered part of this; a perception
  workload like this one needs it first. Built here meanwhile as `sim/src/capture.rs`,
  with no Bevy dependency so it can move into a crate as is: `PixelBox` (projected points
  -> clipped box, YOLO line), `boxes_from_id_mask` (tight boxes + visible-pixel counts),
  `PoolWriter` (images/, labels/, metadata.csv; refuses to write into an existing pool).
  A renderer-side lesson for such a crate: the first frames after startup can be blank
  while shaders compile, so a capture API should reject uniform frames.
  Phase 1 (2026-10-28) moved to the id-mask path and added two pieces such a crate would
  want: `id_mask_from_rgb` (object k drawn in red k * step, so small colour errors still
  decode) and the lessons for the renderer side: the id frame must come out exact
  (no MSAA, no tonemapping, no deband dither: Bevy's `Camera3d` turns dither on by
  default), its background must not be zero so "blank capture" and "nothing to label"
  differ, and both looks must be drawn during warm-up so their pipelines are compiled
  before the first capture. A sensor-model stage (blur, noise, fixed pattern, auto-gain,
  `sensor.rs`) sits naturally between render and writer.
