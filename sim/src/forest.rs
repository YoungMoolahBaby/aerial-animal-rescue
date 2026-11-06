//! The ground of one forest shot (`Modality::Forest`), laid out in metres and painted as
//! temperatures into one image over the camera's footprint. No Bevy here.
//!
//! Built to look like BAMBI (nadir thermal video over Austrian forests, meadows and alpine
//! slopes; `configs/sources.yaml`). A model trained on the older thermal sim found almost
//! none of BAMBI's deer and boar (AP50 .07, `sim/PLAN.md`). Next to BAMBI train frames the
//! sim's ground was flat with smooth blobs, where the real ground is busy everywhere:
//! - a cover with fine texture: meadow grass, forest-floor leaf litter with sticks, snow,
//!   or a ploughed or stubble field; often two covers with an edge (sometimes a hedge);
//! - bare deciduous trees seen from above, a network of branches radiating from the trunk,
//!   mostly warmer than the ground (white in the frames), sometimes colder;
//! - conifer crowns: round, spiky, warm or cold, and by day their cold shadows;
//! - paths and forest tracks with wheel ruts;
//! - logs, stumps and rocks, some as warm as an animal (BAMBI's "No-animal" boxes are
//!   rocks and stumps that look like animals), so the model learns they aren't;
//! - man-made things with hard edges in about a third of the shots: bales or boxes in a
//!   row, often along a track, benches and feeders, a high-seat, a car, a roof (most of a
//!   round-6 model's false alarms on BAMBI train frames were things like these).
//!
//! Temperatures are plausible, not physical. What has to match is how animals stand out:
//! BAMBI's animals are near white (95th percentile grey 222 at the median) on a darker,
//! textured ground (frame median 79), so bodies are many degrees warmer than any of this.

use std::f64::consts::{PI, TAU};
use std::sync::Arc;

use crate::fields::{NoiseTable, band};
use crate::rng::{Rng, lattice};

/// Texels across the painted ground (as `fields::OVERLAY_TEX`).
pub const FOREST_TEX: usize = 2048;
/// Texels across the canopy layer (as fine as the ground: it carries the bare twigs).
pub const CANOPY_TEX: usize = 2048;

/// Everything over the animals, `height_m` above the ground: a leaf canopy of clumps and
/// gaps over part of the shots, and the trees' crowns, conifers and bare branches. Animals
/// under it show only through the gaps (BAMBI's forest flights are full of faint animals
/// seen between leaves, the most common miss of round 2, and of animals crossed by twigs,
/// since the drone looks down through the crowns).
#[derive(Clone, Debug)]
pub struct Canopy {
    pub height_m: f64,
    /// Share of the square covered by leaves (0: bare crowns only).
    pub cover: f64,
    /// Temperatures and opacity per texel, `CANOPY_TEX` squared. A clear texel next to a
    /// leaf, crown or twig has its temperature, so filtering at the edges blends no halo.
    pub temps: Arc<Vec<f32>>,
    pub alpha: Arc<Vec<u8>>,
}

/// What the ground is covered with.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Cover {
    /// Grass: fine streaky texture, tussocks.
    Meadow,
    /// Forest floor: mottled leaf litter, moss, sticks.
    Litter,
    /// Snow: cold and smooth, with trails.
    Snow,
    /// Ploughed or stubble field in rows.
    Field,
}

impl Cover {
    pub fn name(self) -> &'static str {
        match self {
            Cover::Meadow => "meadow",
            Cover::Litter => "litter",
            Cover::Snow => "snow",
            Cover::Field => "field",
        }
    }

    fn random(rng: &mut Rng) -> Self {
        let u = rng.unit();
        if u < 0.35 {
            Cover::Meadow
        } else if u < 0.7 {
            Cover::Litter
        } else if u < 0.85 {
            Cover::Snow
        } else {
            Cover::Field
        }
    }
}

/// One cover's look: its temperature offset from the shot's base and texture strengths.
#[derive(Clone, Debug)]
struct Surface {
    cover: Cover,
    offset_c: f64,
    patch_c: f64,
    grain_c: f64,
    /// Direction of streaks (meadow) or rows (field), radians.
    dir: f64,
    /// Row spacing of a field, metres.
    rows_m: f64,
    /// Tussocks, leaf heaps or stubble: clumps `tuft_m` across, `tuft_c` warmer (or, if
    /// negative, colder) than the cover between them. BAMBI's meadows and litter are rough
    /// up close, with bright tips nearly as white as an animal.
    tuft_c: f64,
    tuft_m: f64,
}

impl Surface {
    fn random(rng: &mut Rng, cover: Cover) -> Self {
        let (offset_c, patch, grain) = match cover {
            Cover::Meadow => (0.0, (0.4, 2.5), (0.3, 1.8)),
            Cover::Litter => (rng.range(-1.5, 1.5), (0.6, 3.0), (0.6, 2.5)),
            Cover::Snow => (-rng.range(2.0, 6.0), (0.1, 0.6), (0.05, 0.4)),
            Cover::Field => (rng.range(-2.0, 2.0), (0.3, 1.5), (0.5, 2.5)),
        };
        // (strength range °C, clump size range m); snow buries them
        let (tuft_c, tuft_m) = match cover {
            Cover::Meadow => ((0.3_f64, 5.0_f64), (0.08, 0.3)),
            Cover::Litter => ((0.3, 4.0), (0.1, 0.4)),
            Cover::Field => ((0.2, 3.0), (0.06, 0.2)),
            Cover::Snow => ((0.0, 0.0), (0.1, 0.1)),
        };
        // mostly warm tips (sunlit, or holding the day's heat), some cold (night frost)
        let sign = if rng.chance(0.7) { 1.0 } else { -1.0 };
        let tuft_c = if tuft_c.1 > 0.0 {
            sign * rng.range(tuft_c.0.ln(), tuft_c.1.ln()).exp()
        } else {
            0.0
        };
        Self {
            cover,
            offset_c,
            patch_c: rng.range(patch.0, patch.1),
            grain_c: rng.range(grain.0, grain.1),
            dir: rng.range(0.0, PI),
            rows_m: rng.range(0.5, 1.5),
            tuft_c,
            tuft_m: rng.range(tuft_m.0, tuft_m.1),
        }
    }
}

/// A line between two covers: points with `side() > 0` are on the second cover.
#[derive(Clone, Debug)]
struct Edge {
    point: [f64; 2],
    normal: [f64; 2],
    wobble_m: f64,
    /// Hedge or scrub along it: width (m) and temperature offset.
    hedge: Option<(f64, f64)>,
}

impl Edge {
    fn side(&self, table: &NoiseTable, x: f64, z: f64) -> f64 {
        let (dx, dz) = (x - self.point[0], z - self.point[1]);
        let along = -dx * self.normal[1] + dz * self.normal[0];
        dx * self.normal[0] + dz * self.normal[1] + self.wobble_m * table.at(along / 25.0, 7.3)
    }
}

/// A line segment drawn into the field: from, to, width (m), temperature offset.
#[derive(Clone, Copy, Debug)]
struct Stroke {
    a: [f64; 2],
    b: [f64; 2],
    width: f64,
    offset_c: f64,
}

/// A soft ellipse drawn into the field (crowns, shadows, stumps, rocks, logs).
#[derive(Clone, Copy, Debug)]
struct Blob {
    centre: [f64; 2],
    /// Half lengths along and across `angle`, metres.
    radii: [f64; 2],
    angle: f64,
    offset_c: f64,
    /// Conifer crown: spiky edge and radial texture.
    spiky: bool,
}

/// A hard-edged, flat-coloured rectangle (or disc) drawn into the field: a man-made thing
/// seen from above, or its shadow.
#[derive(Clone, Copy, Debug)]
struct Slab {
    centre: [f64; 2],
    /// Half lengths along and across `angle`, metres (a disc: radius `half[0]`).
    half: [f64; 2],
    angle: f64,
    round: bool,
    offset_c: f64,
    /// A roof's two slopes: the half on the positive side across `angle` is this much
    /// warmer than the other.
    ridge_c: f64,
    /// The shape is swept along this (metres): a shadow cast from the thing's top.
    sweep: [f64; 2],
    /// Edge width, metres (0: one texel).
    soft_m: f64,
    /// Share of the ground's texture kept through it.
    keep: f64,
}

/// The layout of one forest shot, in metres; `paint` turns it into temperatures.
#[derive(Clone, Debug)]
pub struct ForestLook {
    pub centre: [f64; 2],
    pub size: f64,
    pub seed: u64,
    /// Ground temperature the offsets are relative to.
    pub base_c: f64,
    pub covers: [Cover; 2],
    surfaces: [Surface; 2],
    edge: Option<Edge>,
    /// On the ground, drawn in order: shadows, tracks, debris, man-made things (their
    /// shadows first). Conifer crowns and bare branches are drawn over the leaves into the
    /// canopy layer.
    shadows: Vec<Blob>,
    tracks: Vec<Stroke>,
    debris: Vec<Blob>,
    slabs: Vec<Slab>,
    pub n_trees: usize,
    pub n_debris: usize,
    pub n_man_made: usize,
    /// The painted field, `FOREST_TEX` squared, row by row (z rows, x columns).
    pub field: Arc<Vec<f32>>,
    pub canopy: Option<Canopy>,
}

impl ForestLook {
    /// A random layout over the square `size` metres across around `centre`, then painted.
    /// `sun`: unit vector towards the sun (y up) by day, `None` at night or overcast.
    pub fn random(
        rng: &mut Rng,
        centre: [f64; 2],
        size: f64,
        sun: Option<[f64; 3]>,
        table: &NoiseTable,
    ) -> Self {
        let base_c = rng.range(-2.0, 18.0);
        let first = Cover::random(rng);
        let second = if rng.chance(0.45) {
            Cover::random(rng)
        } else {
            first
        };
        let surfaces = [Surface::random(rng, first), Surface::random(rng, second)];
        let edge = (second != first || rng.chance(0.2)).then(|| {
            let a = rng.range(0.0, TAU);
            Edge {
                point: [
                    centre[0] + rng.range(-0.3, 0.3) * size,
                    centre[1] + rng.range(-0.3, 0.3) * size,
                ],
                normal: [a.cos(), a.sin()],
                wobble_m: rng.range(0.5, 6.0),
                hedge: rng
                    .chance(0.35)
                    .then(|| (rng.range(1.5, 5.0), rng.range(1.0, 5.0))),
            }
        });
        let (x0, z0) = (centre[0] - size / 2.0, centre[1] - size / 2.0);
        let area_ha = size * size / 10_000.0;
        let cover_at = |x: f64, z: f64| match &edge {
            Some(e) if e.side(table, x, z) > 0.0 => second,
            _ => first,
        };

        // trees per hectare, by the cover they stand on
        let density = |c: Cover, rng: &mut Rng| match c {
            Cover::Litter => rng.range(80.0, 350.0),
            Cover::Meadow => rng.range(0.0, 25.0),
            Cover::Snow => rng.range(0.0, 200.0),
            Cover::Field => rng.range(0.0, 5.0),
        };
        let dens = [density(first, rng), density(second, rng)];
        let conifer_share = rng.range(0.0, 0.7);
        let bright_branches = rng.chance(0.75);
        let branch_c = if bright_branches {
            rng.range(1.5, 7.0)
        } else {
            -rng.range(1.0, 4.0)
        };
        let crown_c = if rng.chance(0.7) {
            rng.range(1.0, 5.0)
        } else {
            -rng.range(1.0, 4.0)
        };
        let shadow_c = rng.range(0.5, 3.0);
        let mut shadows = Vec::new();
        let mut crowns = Vec::new();
        let mut branches = Vec::new();
        let mut n_trees = 0;
        let tries = (dens[0].max(dens[1]) * area_ha).round() as usize;
        for _ in 0..tries {
            let (x, z) = (x0 + rng.unit() * size, z0 + rng.unit() * size);
            let c = cover_at(x, z);
            let d = if c == first { dens[0] } else { dens[1] };
            if rng.unit() * dens[0].max(dens[1]) > d {
                continue;
            }
            n_trees += 1;
            let conifer = rng.chance(conifer_share);
            let radius = if conifer {
                rng.range(1.2, 3.5)
            } else {
                rng.range(2.5, 8.0)
            };
            let height = radius * rng.range(2.5, 4.5);
            if let Some(s) = sun {
                // a crown's shadow lies away from the sun, stretched by its low angle
                let elev = s[1].asin().max(0.2);
                let reach = (height / elev.tan()).min(4.0 * radius);
                let (hx, hz) = (s[0], s[2]);
                let len = (hx * hx + hz * hz).sqrt().max(1e-6);
                let (ux, uz) = (-hx / len, -hz / len);
                shadows.push(Blob {
                    centre: [x + ux * reach * 0.6, z + uz * reach * 0.6],
                    radii: [radius + reach * 0.4, radius * 0.9],
                    angle: uz.atan2(ux),
                    offset_c: -shadow_c * if conifer { 1.0 } else { 0.35 },
                    spiky: false,
                });
            }
            let vary = rng.range(0.7, 1.3);
            if conifer {
                crowns.push(Blob {
                    centre: [x, z],
                    radii: [radius, radius * rng.range(0.85, 1.0)],
                    angle: rng.range(0.0, TAU),
                    offset_c: crown_c * vary,
                    spiky: true,
                });
            } else {
                bare_tree(rng, [x, z], radius, branch_c * vary, &mut branches);
            }
        }

        let mut tracks = Vec::new();
        if rng.chance(0.35) {
            let a = rng.range(0.0, PI);
            let (dx, dz) = (a.cos(), a.sin());
            let off = rng.range(-0.35, 0.35) * size;
            let (cx, cz) = (centre[0] - dz * off, centre[1] + dx * off);
            let width = rng.range(1.5, 4.0);
            let offset_c = if rng.chance(0.6) {
                rng.range(0.5, 3.0)
            } else {
                -rng.range(0.5, 2.5)
            };
            // gently curving: a few straight pieces
            let pieces = 6;
            let bend = rng.range(-0.15, 0.15);
            let (mut px, mut pz, mut ang) = (cx - dx * size, cz - dz * size, a);
            for _ in 0..pieces {
                let step = 2.0 * size / pieces as f64;
                let (nx, nz) = (px + ang.cos() * step, pz + ang.sin() * step);
                tracks.push(Stroke {
                    a: [px, pz],
                    b: [nx, nz],
                    width,
                    offset_c,
                });
                // two ruts, darker or brighter than the track
                let rut = -offset_c.signum() * rng.range(0.5, 2.0);
                for s in [-0.3, 0.3] {
                    let (ox, oz) = (-ang.sin() * width * s, ang.cos() * width * s);
                    tracks.push(Stroke {
                        a: [px + ox, pz + oz],
                        b: [nx + ox, nz + oz],
                        width: 0.25,
                        offset_c: offset_c + rut,
                    });
                }
                px = nx;
                pz = nz;
                ang += bend;
            }
        }

        let mut debris = Vec::new();
        let n_debris = rng.index(26);
        for _ in 0..n_debris {
            let (x, z) = (x0 + rng.unit() * size, z0 + rng.unit() * size);
            // some as warm as an animal; most a little warm, a few cold
            let offset_c = if rng.chance(0.12) {
                rng.range(4.0, 14.0)
            } else if rng.chance(0.8) {
                rng.range(0.8, 4.0)
            } else {
                -rng.range(0.5, 3.0)
            };
            let u = rng.unit();
            let radii = if u < 0.4 {
                // log
                [rng.range(0.6, 2.5), rng.range(0.12, 0.3)]
            } else if u < 0.7 {
                // stump
                let r = rng.range(0.15, 0.45);
                [r, r * rng.range(0.8, 1.0)]
            } else {
                // rock or mound, about an animal's size
                let r = rng.range(0.3, 0.9);
                [r, r * rng.range(0.4, 0.9)]
            };
            debris.push(Blob {
                centre: [x, z],
                radii,
                angle: rng.range(0.0, TAU),
                offset_c,
                spiky: false,
            });
        }

        // man-made things in about a third of the shots, by day with their shadows
        let things = if rng.chance(0.35) {
            man_made(rng, (centre, size), &tracks)
        } else {
            Vec::new()
        };
        let n_man_made = things.iter().filter(|(_, h, _)| *h > 0.0).count();
        let mut slabs = Vec::new();
        if let Some(s) = sun {
            let elev = s[1].asin().max(0.2);
            let len = s[0].hypot(s[2]).max(1e-6);
            let away = [-s[0] / len, -s[2] / len];
            for (t, height, stilts) in &things {
                if *height <= 0.0 {
                    continue;
                }
                let reach = (height / elev.tan()).min(4.0 * height);
                let shift = [away[0] * reach, away[1] * reach];
                slabs.push(Slab {
                    // a roof on stilts throws its own shadow off to the side; a thing
                    // standing on the ground, one joined to it
                    centre: if *stilts {
                        [t.centre[0] + shift[0], t.centre[1] + shift[1]]
                    } else {
                        t.centre
                    },
                    offset_c: -shadow_c,
                    ridge_c: 0.0,
                    sweep: if *stilts { [0.0, 0.0] } else { shift },
                    soft_m: 0.15,
                    keep: 0.4,
                    ..*t
                });
            }
        }
        slabs.extend(things.iter().map(|(t, _, _)| *t));

        // leaf canopy: mostly over forest floor
        let canopy_p = match first {
            Cover::Litter => 0.6,
            Cover::Snow => 0.3,
            Cover::Meadow => 0.15,
            Cover::Field => 0.05,
        };
        let leaves = rng.chance(canopy_p).then(|| {
            let cover = rng.range(0.3, 0.92);
            let offset_c = if rng.chance(0.7) {
                rng.range(1.0, 6.0)
            } else {
                -rng.range(1.0, 4.0)
            };
            let mottle_c = rng.range(0.5, 2.5);
            let clump_m = rng.range(1.0, 3.0);
            (cover, offset_c, mottle_c, clump_m, rng.next_u64())
        });
        let height_m = rng.range(8.0, 20.0);
        let overhead = leaves.is_some() || !crowns.is_empty() || !branches.is_empty();
        let canopy = overhead.then(|| {
            let nn = CANOPY_TEX * CANOPY_TEX;
            let (cover, mut temps, alpha) = match leaves {
                Some((cover, offset_c, mottle_c, clump_m, seed)) => {
                    let (t, a) = paint_canopy(
                        centre, size, base_c, cover, offset_c, mottle_c, clump_m, seed, table,
                    );
                    (cover, t, a)
                }
                None => (0.0, vec![base_c as f32; nn], vec![0; nn]),
            };
            let mut alpha: Vec<f32> = alpha.iter().map(|&a| a as f32 / 255.0).collect();
            let layout = (centre, size, base_c);
            paint_crowns(&mut temps, &mut alpha, layout, &crowns, &branches, table);
            let alpha = alpha
                .iter()
                .map(|&a| (a.clamp(0.0, 1.0) * 255.0).round() as u8)
                .collect();
            Canopy {
                height_m,
                cover,
                temps: Arc::new(temps),
                alpha: Arc::new(alpha),
            }
        });

        let seed = rng.next_u64();
        let mut look = Self {
            centre,
            size,
            seed,
            base_c,
            covers: [first, second],
            surfaces,
            edge,
            shadows,
            tracks,
            debris,
            slabs,
            n_trees,
            n_debris,
            n_man_made,
            field: Arc::new(Vec::new()),
            canopy,
        };
        look.field = Arc::new(paint(&look, table, FOREST_TEX));
        look
    }

    /// Ground temperature at (x, z), from the painted field (nearest texel).
    pub fn ground_at(&self, x: f64, z: f64) -> f64 {
        let n = FOREST_TEX;
        let ts = self.size / n as f64;
        let i = ((x - (self.centre[0] - self.size / 2.0)) / ts).floor();
        let j = ((z - (self.centre[1] - self.size / 2.0)) / ts).floor();
        if i < 0.0 || j < 0.0 || i >= n as f64 || j >= n as f64 || self.field.is_empty() {
            return self.base_c;
        }
        self.field[j as usize * n + i as usize] as f64
    }

    /// Percentile of the painted temperatures (`pct` in 0..100).
    pub fn percentile(&self, pct: f64) -> f64 {
        let mut v: Vec<f32> = self.field.iter().step_by(17).copied().collect();
        v.sort_by(f32::total_cmp);
        let k = ((pct / 100.0) * (v.len() - 1) as f64).round() as usize;
        v[k] as f64
    }

    /// Percentile of the canopy's leaf temperatures, if it has one.
    pub fn canopy_percentile(&self, pct: f64) -> Option<f64> {
        let c = self.canopy.as_ref()?;
        let mut v: Vec<f32> = c
            .temps
            .iter()
            .zip(c.alpha.iter())
            .step_by(7)
            .filter(|(_, a)| **a >= 128)
            .map(|(t, _)| *t)
            .collect();
        if v.is_empty() {
            return None;
        }
        v.sort_by(f32::total_cmp);
        Some(v[((pct / 100.0) * (v.len() - 1) as f64).round() as usize] as f64)
    }

    /// The canopy as RGBA: grey through `grey`, alpha its leaves; all clear without one.
    pub fn canopy_rgba(&self, grey: impl Fn(f64) -> f64) -> Vec<u8> {
        let n = CANOPY_TEX * CANOPY_TEX;
        let Some(c) = &self.canopy else {
            return vec![0; n * 4];
        };
        let mut out = Vec::with_capacity(n * 4);
        for (&t, &a) in c.temps.iter().zip(c.alpha.iter()) {
            let g = (grey(t as f64).clamp(0.0, 1.0) * 255.0).round() as u8;
            out.extend_from_slice(&[g, g, g, a]);
        }
        out
    }

    /// The canopy's leaves in white for the id look (alpha as in `canopy_rgba`).
    pub fn canopy_mask(&self) -> Vec<u8> {
        let n = CANOPY_TEX * CANOPY_TEX;
        let Some(c) = &self.canopy else {
            return vec![0; n * 4];
        };
        c.alpha.iter().flat_map(|&a| [255, 255, 255, a]).collect()
    }

    /// The field as sRGB grey bytes (RGBA) through `grey` (temperature -> 0..1).
    pub fn rgba(&self, grey: impl Fn(f64) -> f64) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.field.len() * 4);
        for &t in self.field.iter() {
            let g = (grey(t as f64).clamp(0.0, 1.0) * 255.0).round() as u8;
            out.extend_from_slice(&[g, g, g, 255]);
        }
        out
    }
}

/// Leaf clumps over the square: noise at the clump scale plus finer lumps, cut at the
/// level that leaves `cover` of it opaque, with small holes between leaves; leaves a
/// little warmer or colder than the ground and mottled.
#[allow(clippy::too_many_arguments)]
fn paint_canopy(
    centre: [f64; 2],
    size: f64,
    base_c: f64,
    cover: f64,
    offset_c: f64,
    mottle_c: f64,
    clump_m: f64,
    seed: u64,
    table: &NoiseTable,
) -> (Vec<f32>, Vec<u8>) {
    let n = CANOPY_TEX;
    let ts = size / n as f64;
    let (x0, z0) = (centre[0] - size / 2.0, centre[1] - size / 2.0);
    let o = (seed % 83) as f64 * 2.7;
    let mut level = Vec::with_capacity(n * n);
    let mut temps = Vec::with_capacity(n * n);
    for j in 0..n {
        for i in 0..n {
            let (x, z) = (x0 + (i as f64 + 0.5) * ts, z0 + (j as f64 + 0.5) * ts);
            let l = 0.65 * table.at(x / clump_m + o, z / clump_m)
                + 0.35 * table.at(x / (clump_m * 0.3) + o, z / (clump_m * 0.3) + 5.0);
            level.push(l);
            let mottle = table.at(x / 0.4 + o, z / 0.4 + 9.0);
            temps.push((base_c + offset_c + mottle_c * mottle) as f32);
        }
    }
    let mut sorted: Vec<f64> = level.iter().step_by(13).copied().collect();
    sorted.sort_by(f64::total_cmp);
    let cut = sorted[(((1.0 - cover) * (sorted.len() - 1) as f64).round() as usize).min(sorted.len() - 1)];
    // opacity as a smooth ramp round the cut, so the renderer's bilinear read and the
    // 0.5 alpha cut give smooth leaf edges at any distance; irregular gaps between leaves
    // a couple of decimetres across
    let alpha = level
        .iter()
        .enumerate()
        .map(|(k, &l)| {
            let (x, z) = (x0 + (k % n) as f64 * ts, z0 + (k / n) as f64 * ts);
            let gap = table.at(x / 0.25 + o, z / 0.25 + 3.0);
            let v = (l - cut) / 0.04 - (gap - 0.35).max(0.0) / 0.05;
            ((0.5 + 0.5 * v.clamp(-1.0, 1.0)) * 255.0).round() as u8
        })
        .collect();
    (temps, alpha)
}

/// A bare deciduous tree from above: 4-8 limbs from the trunk, each forking a few times,
/// thinner and shorter each time, a little crooked.
fn bare_tree(rng: &mut Rng, at: [f64; 2], radius: f64, offset_c: f64, out: &mut Vec<Stroke>) {
    // BAMBI's bare crowns are fine, tangled networks rather than stars: thin limbs that
    // start off the trunk and fork four times into twigs a few centimetres wide
    let limbs = 3 + rng.index(5);
    let start = rng.range(0.0, TAU);
    for k in 0..limbs {
        let a = start + TAU * k as f64 / limbs as f64 + rng.range(-0.5, 0.5);
        let len = radius * rng.range(0.35, 0.55);
        let width = rng.range(0.06, 0.16);
        let from = [
            at[0] + a.cos() * rng.range(0.0, 0.6),
            at[1] + a.sin() * rng.range(0.0, 0.6),
        ];
        let own = offset_c * rng.range(0.6, 1.15);
        branch(rng, from, a, len, width, own, 0, out);
    }
}

#[allow(clippy::too_many_arguments)]
fn branch(
    rng: &mut Rng,
    from: [f64; 2],
    angle: f64,
    len: f64,
    width: f64,
    offset_c: f64,
    depth: u32,
    out: &mut Vec<Stroke>,
) {
    // three crooked pieces
    let (mut p, mut a) = (from, angle);
    for _ in 0..3 {
        a += rng.range(-0.4, 0.4);
        let q = [p[0] + a.cos() * len / 3.0, p[1] + a.sin() * len / 3.0];
        out.push(Stroke {
            a: p,
            b: q,
            width,
            // thin twigs fade into the ground's temperature
            offset_c: offset_c * (0.4 + 0.6 * (width / 0.12).min(1.0)),
        });
        p = q;
    }
    if depth >= 4 {
        return;
    }
    let forks = 2 + rng.chance(0.35) as usize;
    for f in 0..forks {
        let spread = rng.range(0.25, 0.9) * if f % 2 == 0 { 1.0 } else { -1.0 };
        let shorter = rng.range(0.55, 0.8);
        branch(
            rng,
            p,
            a + spread,
            len * shorter,
            (width * 0.65).max(0.015),
            offset_c,
            depth + 1,
            out,
        );
    }
}

/// Man-made things for one shot over the square (`centre`, `size`), each with its height
/// (0 for a part of another thing, such as a car's windscreen: no shadow of its own) and
/// whether it stands on stilts. BAMBI's train frames show bales or boxes in a row along a
/// hedge or a track, roofs, high-seats and things on paths, and a model trained without
/// them takes them for animals: they are about an animal's size and often as warm.
fn man_made(rng: &mut Rng, square: ([f64; 2], f64), tracks: &[Stroke]) -> Vec<(Slab, f64, bool)> {
    let (centre, size) = square;
    // mostly sun-warmed or holding the day's heat, some cold (wet, or metal and glass
    // under a clear night sky)
    let temp = |rng: &mut Rng, hot: f64| {
        if rng.chance(0.7) {
            rng.range(1.5_f64.ln(), hot.ln()).exp()
        } else {
            -rng.range(1.0, 5.0)
        }
    };
    let slab = |centre: [f64; 2], half: [f64; 2], angle: f64, offset_c: f64| Slab {
        centre,
        half,
        angle,
        round: false,
        offset_c,
        ridge_c: 0.0,
        sweep: [0.0, 0.0],
        soft_m: 0.0,
        keep: 0.1,
    };
    let mut out = Vec::new();
    let u = rng.unit();
    if u < 0.4 {
        // a row of bales or boxes, half the time beside a track
        let n = 2 + rng.index(7);
        let round = rng.chance(0.4);
        let half = if round {
            let r = rng.range(0.55, 0.9);
            [r, r]
        } else {
            [rng.range(0.5, 1.1), rng.range(0.35, 0.7)]
        };
        let pieces = tracks.len() / 3;
        let (start, dir) = if pieces > 0 && rng.chance(0.5) {
            // the tracks are pieces of three strokes: the track, then its two ruts
            let s = &tracks[3 * rng.index(pieces)];
            let (dx, dz) = (s.b[0] - s.a[0], s.b[1] - s.a[1]);
            let len = dx.hypot(dz).max(1e-6);
            let side = if rng.chance(0.5) { 1.0 } else { -1.0 };
            let off = side * (s.width / 2.0 + half[1] + rng.range(0.5, 3.0));
            let t = rng.range(0.0, 0.5);
            (
                [
                    s.a[0] + dx * t - dz / len * off,
                    s.a[1] + dz * t + dx / len * off,
                ],
                dz.atan2(dx),
            )
        } else {
            (
                [
                    centre[0] + rng.range(-0.4, 0.4) * size,
                    centre[1] + rng.range(-0.4, 0.4) * size,
                ],
                rng.range(0.0, TAU),
            )
        };
        let across = rng.chance(0.5);
        let step = 2.0 * if across { half[1] } else { half[0] } + rng.range(0.3, 5.0);
        let row_c = temp(rng, 14.0);
        let height = rng.range(0.8, 1.5);
        for k in 0..n {
            let (along, aside) = (k as f64 * step, rng.range(-0.3, 0.3));
            let at = [
                start[0] + dir.cos() * along - dir.sin() * aside,
                start[1] + dir.sin() * along + dir.cos() * aside,
            ];
            let angle = dir + if across { PI / 2.0 } else { 0.0 } + rng.range(-0.15, 0.15);
            let mut s = slab(at, half, angle, row_c * rng.range(0.8, 1.2));
            s.round = round;
            out.push((s, height, false));
        }
    } else if u < 0.65 {
        // a few small things: feeders, benches, boxes, tarps
        for _ in 0..1 + rng.index(4) {
            let at = [
                centre[0] + rng.range(-0.4, 0.4) * size,
                centre[1] + rng.range(-0.4, 0.4) * size,
            ];
            let round = rng.chance(0.15);
            let half = if round {
                let r = rng.range(0.3, 0.7);
                [r, r]
            } else {
                [rng.range(0.3, 1.2), rng.range(0.2, 0.6)]
            };
            let mut s = slab(at, half, rng.range(0.0, TAU), temp(rng, 14.0));
            s.round = round;
            out.push((s, rng.range(0.3, 1.2), false));
        }
    } else if u < 0.8 {
        // a high-seat: a small roof on stilts
        let h = rng.range(0.7, 1.3);
        let at = [
            centre[0] + rng.range(-0.4, 0.4) * size,
            centre[1] + rng.range(-0.4, 0.4) * size,
        ];
        let mut s = slab(at, [h, h * rng.range(0.8, 1.0)], rng.range(0.0, TAU), temp(rng, 10.0));
        s.ridge_c = rng.range(1.0, 4.0) * if rng.chance(0.5) { 1.0 } else { -1.0 };
        out.push((s, rng.range(3.5, 7.0), true));
    } else if u < 0.9 {
        // a car: a warm body (engine, sun), a cold windscreen (glass shows the sky)
        let at = [
            centre[0] + rng.range(-0.4, 0.4) * size,
            centre[1] + rng.range(-0.4, 0.4) * size,
        ];
        let angle = rng.range(0.0, TAU);
        let len = rng.range(2.0, 2.4);
        let body_c = temp(rng, 12.0);
        out.push((slab(at, [len, rng.range(0.85, 0.95)], angle, body_c), 1.5, false));
        let glass = [
            at[0] + angle.cos() * len * 0.3,
            at[1] + angle.sin() * len * 0.3,
        ];
        let glass_c = body_c - rng.range(2.0, 8.0);
        out.push((slab(glass, [0.35, 0.75], angle, glass_c), 0.0, false));
    } else {
        // a roof: two slopes, one sunlit
        let at = [
            centre[0] + rng.range(-0.4, 0.4) * size,
            centre[1] + rng.range(-0.4, 0.4) * size,
        ];
        let half = [rng.range(3.0, 8.0), rng.range(2.5, 6.0)];
        let mut s = slab(at, half, rng.range(0.0, TAU), rng.range(-4.0, 8.0));
        s.ridge_c = rng.range(1.0, 5.0) * if rng.chance(0.5) { 1.0 } else { -1.0 };
        out.push((s, rng.range(4.0, 9.0), false));
    }
    out
}

/// Paint the layout's temperatures into an `n` x `n` field over its square.
pub fn paint(look: &ForestLook, table: &NoiseTable, n: usize) -> Vec<f32> {
    let ts = look.size / n as f64;
    let (x0, z0) = (
        look.centre[0] - look.size / 2.0,
        look.centre[1] - look.size / 2.0,
    );
    let mut field = vec![0f32; n * n];
    let threads = std::thread::available_parallelism().map_or(4, |t| t.get());
    let rows_per = n.div_ceil(threads);
    std::thread::scope(|scope| {
        for (chunk, rows) in field.chunks_mut(rows_per * n).enumerate() {
            scope.spawn(move || {
                for (r, row) in rows.chunks_mut(n).enumerate() {
                    let j = chunk * rows_per + r;
                    for (i, v) in row.iter_mut().enumerate() {
                        let (x, z) = (x0 + (i as f64 + 0.5) * ts, z0 + (j as f64 + 0.5) * ts);
                        *v = ground(look, table, x, z, ts) as f32;
                    }
                }
            });
        }
    });
    let at = |p: [f64; 2]| ((p[0] - x0) / ts, (p[1] - z0) / ts);
    let base = look.base_c;
    // blend toward the mark, keeping a little of the ground's texture through it
    let mark = |field: &mut [f32], k: usize, cov: f64, target: f64, keep: f64| {
        if cov > 0.0 {
            let v = field[k] as f64;
            let t = target + (v - base) * keep;
            field[k] = (v * (1.0 - cov) + t * cov) as f32;
        }
    };
    for b in &look.shadows {
        blob(n, ts, at, b, table, look.seed, |k, cov, texture| {
            mark(&mut field, k, cov, base + b.offset_c * texture, 0.4)
        });
    }
    for s in &look.tracks {
        stroke(n, ts, at, s, |k, cov| {
            mark(&mut field, k, cov, base + s.offset_c, 0.3)
        });
    }
    for b in &look.debris {
        blob(n, ts, at, b, table, look.seed, |k, cov, texture| {
            mark(&mut field, k, cov, base + b.offset_c * texture, 0.4)
        });
    }
    for s in &look.slabs {
        slab(n, ts, at, s, |k, cov, side| {
            mark(&mut field, k, cov, base + s.offset_c + side * s.ridge_c / 2.0, s.keep)
        });
    }
    field
}

/// Conifer crowns and bare branches over the canopy layer (`temps`, `alpha` 0..1), laid
/// over whatever is there (leaves, or nothing). `layout`: the square's centre, size and
/// base temperature.
fn paint_crowns(
    temps: &mut [f32],
    alpha: &mut [f32],
    layout: ([f64; 2], f64, f64),
    crowns: &[Blob],
    branches: &[Stroke],
    table: &NoiseTable,
) {
    let (centre, size, base_c) = layout;
    let n = CANOPY_TEX;
    let ts = size / n as f64;
    let (x0, z0) = (centre[0] - size / 2.0, centre[1] - size / 2.0);
    let at = |p: [f64; 2]| ((p[0] - x0) / ts, (p[1] - z0) / ts);
    for b in crowns {
        blob(n, ts, at, b, table, 0, |k, cov, texture| {
            over(&mut temps[k], &mut alpha[k], base_c + b.offset_c * texture, cov)
        });
    }
    for s in branches {
        stroke(n, ts, at, s, |k, cov| {
            over(&mut temps[k], &mut alpha[k], base_c + s.offset_c, cov)
        });
    }
}

/// Lay `target` with coverage `cov` over a texel whose opacity so far is `a`. A clear
/// texel beside the mark (`cov` 0) takes its temperature, so the renderer's bilinear read
/// at the mark's edge doesn't mix in the temperature of nothing.
fn over(t: &mut f32, a: &mut f32, target: f64, cov: f64) {
    if cov <= 0.0 {
        if *a <= 0.0 {
            *t = target as f32;
        }
        return;
    }
    let (t0, a0) = (*t as f64, *a as f64);
    let a1 = a0 + cov * (1.0 - a0);
    *t = ((t0 * a0 * (1.0 - cov) + target * cov) / a1) as f32;
    *a = a1 as f32;
}

/// The covers' temperature at (x, z): base, offset, patches, grain, rows or streaks.
fn ground(look: &ForestLook, table: &NoiseTable, x: f64, z: f64, ts: f64) -> f64 {
    let side = look.edge.as_ref().map_or(-1.0, |e| e.side(table, x, z));
    // blend the two covers over a metre or so
    let w = (0.5 + side / 1.5).clamp(0.0, 1.0);
    let mut t = 0.0;
    for (k, s) in look.surfaces.iter().enumerate() {
        let weight = if k == 0 { 1.0 - w } else { w };
        if weight > 0.0 {
            t += weight * surface(s, table, look.seed + k as u64, x, z, ts);
        }
    }
    if let Some(e) = &look.edge
        && let Some((width, off)) = e.hedge
    {
        // scrub along the edge: a warm, lumpy band
        let near = band(side.abs(), width / 2.0, 1.0) as f64;
        if near > 0.0 {
            let lump = 0.5 + 0.5 * table.at(x / 1.2 + 40.0, z / 1.2);
            t += near * off * lump;
        }
    }
    look.base_c + t
}

fn surface(s: &Surface, table: &NoiseTable, seed: u64, x: f64, z: f64, ts: f64) -> f64 {
    let o = (seed % 97) as f64 * 3.1;
    let patches = table.at(x / 6.0 + o, z / 6.0) + 0.5 * table.at(x / 2.0 + o, z / 2.0 + 11.0);
    let (c, sn) = (s.dir.cos(), s.dir.sin());
    let (u, v) = (x * c + z * sn, -x * sn + z * c);
    // per-texel grain, hashed so it doesn't repeat
    let cell = 0.06_f64.max(ts);
    let speck =
        2.0 * lattice(
            seed ^ 0x6A1,
            (x / cell).floor() as i64,
            (z / cell).floor() as i64,
        ) - 1.0;
    let grain = match s.cover {
        // grass: streaks along the wind or the mowing, plus speckle
        Cover::Meadow => 0.6 * table.at(u / 1.5 + o, v / 0.12) + 0.4 * speck,
        // leaf litter: blotches a few decimetres across, sharp-edged
        Cover::Litter => {
            let b = table.at(x / 0.35 + o, z / 0.35);
            0.7 * (b * 2.0).clamp(-1.0, 1.0) + 0.3 * speck
        }
        Cover::Snow => 0.5 * table.at(x / 0.8 + o, z / 0.8) + 0.2 * speck,
        // rows (ploughed furrows or stubble lines) with a little speckle
        // (faint: at 6-12 cm a pixel, strong rows turn into stripes BAMBI doesn't show)
        Cover::Field => {
            0.3 * (TAU * v / s.rows_m).sin() + 0.4 * table.at(x / 0.5 + o, z / 0.5) + 0.3 * speck
        }
    };
    s.offset_c + s.patch_c * patches + s.grain_c * grain + s.tuft_c * tufts(s, seed, x, z)
}

/// Clumps of a surface, 0 between them and up to 1 on their tips; most of the ground is
/// between. On hashed cells (`value_noise`), so the clumps never repeat.
fn tufts(s: &Surface, seed: u64, x: f64, z: f64) -> f64 {
    if s.tuft_c == 0.0 {
        return 0.0;
    }
    let m = s.tuft_m;
    let t = 0.65 * value_noise(seed ^ 0x7F7, x / m, z / m)
        + 0.35 * value_noise(seed ^ 0x81B, x / (0.43 * m) + 0.5, z / (0.43 * m));
    ((t - 0.55) / 0.3).clamp(0.0, 1.0).powf(1.3)
}

/// Smooth value noise on hashed lattice cells, 0..1; one unit of (x, z) is one cell.
fn value_noise(seed: u64, x: f64, z: f64) -> f64 {
    let (x0, z0) = (x.floor(), z.floor());
    let (fx, fz) = (x - x0, z - z0);
    let (sx, sz) = (fx * fx * (3.0 - 2.0 * fx), fz * fz * (3.0 - 2.0 * fz));
    let (i, j) = (x0 as i64, z0 as i64);
    let l = |a: i64, b: i64| lattice(seed, a, b);
    let a = l(i, j) + (l(i + 1, j) - l(i, j)) * sx;
    let b = l(i, j + 1) + (l(i + 1, j + 1) - l(i, j + 1)) * sx;
    a + (b - a) * sz
}

/// A segment's coverage of the texels near it, antialiased over one texel: calls
/// `put(texel, coverage)` out to a texel and a half past its edge (coverage 0 there).
fn stroke(
    n: usize,
    ts: f64,
    at: impl Fn([f64; 2]) -> (f64, f64),
    s: &Stroke,
    mut put: impl FnMut(usize, f64),
) {
    let (ax, ay) = at(s.a);
    let (bx, by) = at(s.b);
    let half = s.width / 2.0 / ts; // texels
    let pad = half + 1.5;
    let (i0, i1) = (ax.min(bx) - pad, ax.max(bx) + pad);
    let (j0, j1) = (ay.min(by) - pad, ay.max(by) + pad);
    if i1 < 0.0 || j1 < 0.0 || i0 >= n as f64 || j0 >= n as f64 {
        return;
    }
    let (dx, dy) = (bx - ax, by - ay);
    let len2 = (dx * dx + dy * dy).max(1e-9);
    // a twig thinner than a texel still darkens or brightens it a little
    let (half_drawn, strength) = if half < 0.5 {
        (0.5, half * 2.0)
    } else {
        (half, 1.0)
    };
    for j in (j0.max(0.0) as usize)..=(j1.min(n as f64 - 1.0) as usize) {
        for i in (i0.max(0.0) as usize)..=(i1.min(n as f64 - 1.0) as usize) {
            let (px, py) = (i as f64 + 0.5 - ax, j as f64 + 0.5 - ay);
            let t = ((px * dx + py * dy) / len2).clamp(0.0, 1.0);
            let (qx, qy) = (px - t * dx, py - t * dy);
            let d = (qx * qx + qy * qy).sqrt();
            if d <= half_drawn + 1.5 {
                put(j * n + i, band(d, half_drawn, 1.0) as f64 * strength);
            }
        }
    }
}

/// A soft ellipse's coverage (spiky for conifer crowns): calls `put(texel, coverage,
/// texture)` over its square, `texture` scaling its offset (a crown's clumps).
fn blob(
    n: usize,
    ts: f64,
    at: impl Fn([f64; 2]) -> (f64, f64),
    b: &Blob,
    table: &NoiseTable,
    seed: u64,
    mut put: impl FnMut(usize, f64, f64),
) {
    let (cx, cy) = at(b.centre);
    let r = b.radii[0].max(b.radii[1]) / ts * 1.3 + 2.0;
    let (c, s) = (b.angle.cos(), b.angle.sin());
    let o = (seed % 89) as f64 + cx * 0.01;
    for j in ((cy - r).max(0.0) as usize)..=((cy + r).min(n as f64 - 1.0).max(0.0) as usize) {
        for i in ((cx - r).max(0.0) as usize)..=((cx + r).min(n as f64 - 1.0).max(0.0) as usize) {
            let (px, py) = ((i as f64 + 0.5 - cx) * ts, (j as f64 + 0.5 - cy) * ts);
            let (u, v) = (px * c + py * s, -px * s + py * c);
            let (eu, ev) = (u / b.radii[0], v / b.radii[1]);
            let mut d = (eu * eu + ev * ev).sqrt();
            let mut texture = 1.0;
            let (wx, wz) = (b.centre[0] + px, b.centre[1] + py);
            if b.spiky {
                // a lumpy outline and clumps of foliage with gaps (BAMBI's evergreen crowns
                // are bright, fuzzy clumps, not discs)
                let ang = ev.atan2(eu);
                d *=
                    1.0 + 0.35 * table.at(ang * 1.2 + o, 3.0) + 0.12 * table.at(ang * 4.0 + o, 9.0);
                let clumps = table.at(wx / 0.5 + o, wz / 0.5);
                texture = (0.7 + 0.6 * clumps).clamp(0.1, 1.3);
            } else {
                // stumps, rocks and logs: lumpy too
                d *= 1.0 + 0.15 * table.at(wx / 0.3 + o, wz / 0.3);
            }
            // soft edge over about two texels or a share of the radius (shadows: wide)
            let share = if b.offset_c < 0.0 && !b.spiky {
                0.35
            } else {
                0.12
            };
            let soft = (2.0 * ts / b.radii[0].min(b.radii[1])).max(share);
            put(j * n + i, ((1.0 - d) / soft).clamp(0.0, 1.0), texture);
        }
    }
}

/// A slab's coverage, antialiased over a texel (or `soft_m`): calls `put(texel, coverage,
/// side)` over the texels it touches, `side` +1 or -1 across its long axis (a roof's two
/// slopes). A swept slab covers every place along its sweep (a shadow).
fn slab(
    n: usize,
    ts: f64,
    at: impl Fn([f64; 2]) -> (f64, f64),
    s: &Slab,
    mut put: impl FnMut(usize, f64, f64),
) {
    let (cx, cy) = at(s.centre);
    let r = (s.half[0].max(s.half[1]) * 1.42 + s.soft_m) / ts + 2.0;
    let (wx, wy) = (s.sweep[0] / ts, s.sweep[1] / ts);
    let (i0, i1) = (cx + wx.min(0.0) - r, cx + wx.max(0.0) + r);
    let (j0, j1) = (cy + wy.min(0.0) - r, cy + wy.max(0.0) + r);
    if i1 < 0.0 || j1 < 0.0 || i0 >= n as f64 || j0 >= n as f64 {
        return;
    }
    let (c, sn) = (s.angle.cos(), s.angle.sin());
    // steps along the sweep a quarter of the thing's width, so its sides come out straight
    let steps = (s.sweep[0].hypot(s.sweep[1]) / (0.25 * s.half[0].min(s.half[1])).max(ts))
        .ceil()
        .min(256.0) as usize;
    let edge = ts.max(s.soft_m);
    let dist = |px: f64, py: f64| {
        let (u, v) = (px * c + py * sn, -px * sn + py * c);
        if s.round {
            u.hypot(v) - s.half[0]
        } else {
            (u.abs() - s.half[0]).max(v.abs() - s.half[1])
        }
    };
    for j in (j0.max(0.0) as usize)..=(j1.min(n as f64 - 1.0) as usize) {
        for i in (i0.max(0.0) as usize)..=(i1.min(n as f64 - 1.0) as usize) {
            let (px, py) = ((i as f64 + 0.5 - cx) * ts, (j as f64 + 0.5 - cy) * ts);
            let mut d = dist(px, py);
            for k in 1..=steps {
                let t = k as f64 / steps as f64;
                d = d.min(dist(px - t * s.sweep[0], py - t * s.sweep[1]));
            }
            let cov = (0.5 - d / edge).clamp(0.0, 1.0);
            if cov > 0.0 {
                let side = if -px * sn + py * c >= 0.0 { 1.0 } else { -1.0 };
                put(j * n + i, cov, side);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tufts_are_clumps_that_dont_repeat() {
        let mut rng = Rng::new(3);
        let mut s = Surface::random(&mut rng, Cover::Meadow);
        s.tuft_c = 2.0;
        s.tuft_m = 0.2;
        let v: Vec<f64> = (0..4000)
            .map(|k| tufts(&s, 9, k as f64 * 0.037, (k % 61) as f64 * 0.05))
            .collect();
        assert!(v.iter().all(|t| (0.0..=1.0).contains(t)));
        let on = v.iter().filter(|&&t| t > 0.0).count() as f64 / v.len() as f64;
        assert!((0.15..0.5).contains(&on), "share on clumps {on}");
        // the noise table would repeat every 16 clumps (3.2 m); these don't
        let same = (0..200)
            .filter(|&k| {
                let (x, z) = (k as f64 * 0.11, 1.3);
                (tufts(&s, 9, x, z) - tufts(&s, 9, x + 3.2, z)).abs() < 1e-9
            })
            .count();
        assert!(same < 100, "{same} of 200 repeat");
    }

    #[test]
    fn paints_every_texel_and_is_reproducible() {
        let table = NoiseTable::new(3);
        let mut a = Rng::new(9);
        let look = ForestLook::random(&mut a, [10.0, -5.0], 60.0, Some([0.5, 0.6, 0.62]), &table);
        assert_eq!(look.field.len(), FOREST_TEX * FOREST_TEX);
        assert!(look.field.iter().all(|t| t.is_finite()));
        let mut b = Rng::new(9);
        let again = ForestLook::random(&mut b, [10.0, -5.0], 60.0, Some([0.5, 0.6, 0.62]), &table);
        assert_eq!(look.field, again.field);
        // textured: not a flat colour
        assert!(look.percentile(99.0) - look.percentile(1.0) > 0.5);
    }

    #[test]
    fn ground_at_reads_the_field_and_falls_back_outside() {
        let table = NoiseTable::new(3);
        let look = ForestLook::random(&mut Rng::new(2), [0.0, 0.0], 40.0, None, &table);
        let inside = look.ground_at(1.0, 1.0);
        let n = FOREST_TEX;
        let ts = 40.0 / n as f64;
        let (i, j) = (((1.0 + 20.0) / ts) as usize, ((1.0 + 20.0) / ts) as usize);
        assert_eq!(inside, look.field[j * n + i] as f64);
        assert_eq!(look.ground_at(500.0, 0.0), look.base_c);
    }

    #[test]
    fn canopy_covers_about_its_share() {
        let table = NoiseTable::new(5);
        let (temps, alpha) =
            paint_canopy([0.0, 0.0], 50.0, 5.0, 0.6, 3.0, 1.0, 2.0, 11, &table);
        assert_eq!((temps.len(), alpha.len()), (CANOPY_TEX * CANOPY_TEX, CANOPY_TEX * CANOPY_TEX));
        let share = alpha.iter().filter(|&&a| a >= 128).count() as f64 / alpha.len() as f64;
        // the cut leaves 60%, the gaps between leaves take some of that
        assert!((0.4..0.62).contains(&share), "{share}");
    }

    #[test]
    fn over_lays_marks_on_the_layer() {
        // a clear texel beside a mark takes its temperature but stays clear
        let (mut t, mut a) = (5.0_f32, 0.0_f32);
        over(&mut t, &mut a, 9.0, 0.0);
        assert_eq!((t, a), (9.0, 0.0));
        // half covered: the mark's temperature, half opaque
        let (mut t, mut a) = (5.0_f32, 0.0_f32);
        over(&mut t, &mut a, 9.0, 0.5);
        assert_eq!((t, a), (9.0, 0.5));
        // over an opaque leaf: mixed by coverage, still opaque
        let (mut t, mut a) = (5.0_f32, 1.0_f32);
        over(&mut t, &mut a, 9.0, 0.25);
        assert!((t - 6.0).abs() < 1e-6 && a == 1.0);
        // an opaque texel beside a mark keeps its own temperature
        let (mut t, mut a) = (5.0_f32, 1.0_f32);
        over(&mut t, &mut a, 9.0, 0.0);
        assert_eq!((t, a), (5.0, 1.0));
    }

    #[test]
    fn bare_crowns_go_overhead() {
        let table = NoiseTable::new(4);
        // a shot with trees and no leaves: the layer holds only the crowns and twigs
        let look = (0..200)
            .map(|k| ForestLook::random(&mut Rng::new(k), [0.0, 0.0], 30.0, None, &table))
            .find(|l| l.n_trees > 2 && l.canopy.as_ref().is_some_and(|c| c.cover == 0.0))
            .expect("a bare-tree shot");
        let c = look.canopy.as_ref().unwrap();
        assert_eq!(c.alpha.len(), CANOPY_TEX * CANOPY_TEX);
        let share = c.alpha.iter().filter(|&&a| a >= 128).count() as f64 / c.alpha.len() as f64;
        assert!((0.001..0.6).contains(&share), "{share}");
        // shots without trees or leaves have nothing overhead
        let open = (0..200)
            .map(|k| ForestLook::random(&mut Rng::new(k), [0.0, 0.0], 30.0, None, &table))
            .find(|l| l.n_trees == 0)
            .expect("an open shot");
        assert!(open.canopy.as_ref().is_none_or(|c| c.cover > 0.0));
    }

    #[test]
    fn slabs_are_hard_edged_and_shadows_sweep() {
        let (n, ts) = (240, 0.05);
        let at = |p: [f64; 2]| (p[0] / ts, p[1] / ts);
        let s = Slab {
            centre: [5.0, 5.0],
            half: [1.0, 0.5],
            angle: 0.4,
            round: false,
            offset_c: 3.0,
            ridge_c: 2.0,
            sweep: [0.0, 0.0],
            soft_m: 0.0,
            keep: 0.1,
        };
        let (mut area, mut edge, mut sides) = (0.0, 0usize, [0usize; 2]);
        slab(n, ts, at, &s, |_, cov, side| {
            area += cov * ts * ts;
            edge += (cov < 1.0) as usize;
            sides[(side > 0.0) as usize] += 1;
        });
        assert!((area - 2.0).abs() < 0.02, "{area}");
        // the edge is about a texel wide all round (6 m of it)
        assert!((edge as f64) < 1.6 * 6.0 / ts, "{edge}");
        // a roof's two slopes are the same size
        assert!((sides[0] as f64 / sides[1] as f64 - 1.0).abs() < 0.1, "{sides:?}");
        // a disc swept 4 m covers a capsule
        let shadow = Slab {
            round: true,
            half: [1.0, 1.0],
            sweep: [4.0, 0.0],
            soft_m: 0.15,
            ..s
        };
        let mut area = 0.0;
        slab(n, ts, at, &shadow, |_, cov, _| area += cov * ts * ts);
        let capsule = PI + 2.0 * 4.0;
        assert!((area - capsule).abs() < 0.03 * capsule, "{area} vs {capsule}");
    }

    #[test]
    fn man_made_things_look_like_animals_or_are_bigger() {
        let mut rng = Rng::new(5);
        let track = Stroke {
            a: [-20.0, 0.0],
            b: [20.0, 0.0],
            width: 3.0,
            offset_c: 1.0,
        };
        let (mut all, mut warm, mut cold) = (0, 0, 0);
        for _ in 0..400 {
            let things = man_made(&mut rng, ([0.0, 0.0], 60.0), &[track; 3]);
            assert!(!things.is_empty());
            for (s, h, _) in &things {
                assert!((0.3..=8.0).contains(&s.half[0]) && (0.2..=6.0).contains(&s.half[1]));
                assert!((0.0..=9.0).contains(h));
                all += 1;
                warm += (s.offset_c >= 4.0) as usize;
                cold += (s.offset_c < 0.0) as usize;
            }
        }
        let (warm, cold) = (warm as f64 / all as f64, cold as f64 / all as f64);
        assert!((0.25..0.6).contains(&warm), "warm as an animal {warm}");
        assert!((0.15..0.45).contains(&cold), "cold {cold}");
    }

    #[test]
    fn man_made_things_are_painted() {
        let table = NoiseTable::new(6);
        // the last one drawn inside the square (rows run on past its edge; nothing drawn
        // later covers a thing's centre)
        let (look, s) = (0..60)
            .map(|k| ForestLook::random(&mut Rng::new(k), [0.0, 0.0], 40.0, None, &table))
            .find_map(|l| {
                let inside = |s: &&Slab| s.centre[0].abs() < 18.0 && s.centre[1].abs() < 18.0;
                let s = *l.slabs.iter().rev().find(inside)?;
                Some((l, s))
            })
            .expect("a shot with man-made things");
        let got = look.ground_at(s.centre[0], s.centre[1]);
        let want = look.base_c + s.offset_c;
        assert!((got - want).abs() <= s.ridge_c.abs() / 2.0 + 1.0, "{got} vs {want}");
    }

    #[test]
    fn bare_tree_branches_thin_out() {
        let mut out = Vec::new();
        bare_tree(&mut Rng::new(1), [0.0, 0.0], 6.0, 3.0, &mut out);
        assert!(out.len() > 50);
        let max_w = out.iter().map(|s| s.width).fold(0.0, f64::max);
        let min_w = out.iter().map(|s| s.width).fold(f64::MAX, f64::min);
        assert!(max_w > 0.05 && min_w <= 0.02);
        // nothing reaches far beyond the crown
        assert!(out.iter().all(|s| s.b[0].hypot(s.b[1]) < 6.0 * 2.0));
    }
}
