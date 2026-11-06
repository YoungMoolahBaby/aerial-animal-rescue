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
//!   rocks and stumps that look like animals), so the model learns they aren't.
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
/// Texels across the canopy layer (coarser: leaves are drawn as clumps and gaps).
pub const CANOPY_TEX: usize = 1024;

/// A closed leaf canopy over part of the shot, `height_m` above the ground: clumps of
/// foliage with gaps. Animals under it show only through the gaps (BAMBI's forest flights
/// are full of faint animals seen between leaves, the most common miss of round 2).
#[derive(Clone, Debug)]
pub struct Canopy {
    pub height_m: f64,
    /// Share of the square covered by leaves.
    pub cover: f64,
    /// Temperatures and opacity (0 or 255) per texel, `CANOPY_TEX` squared.
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
    /// Drawn in order: shadows, tracks, debris, conifer crowns, branches.
    shadows: Vec<Blob>,
    tracks: Vec<Stroke>,
    debris: Vec<Blob>,
    crowns: Vec<Blob>,
    branches: Vec<Stroke>,
    pub n_trees: usize,
    pub n_debris: usize,
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

        // leaf canopy: mostly over forest floor
        let canopy_p = match first {
            Cover::Litter => 0.6,
            Cover::Snow => 0.3,
            Cover::Meadow => 0.15,
            Cover::Field => 0.05,
        };
        let canopy = rng.chance(canopy_p).then(|| {
            let cover = rng.range(0.3, 0.92);
            let offset_c = if rng.chance(0.7) {
                rng.range(1.0, 6.0)
            } else {
                -rng.range(1.0, 4.0)
            };
            let mottle_c = rng.range(0.5, 2.5);
            let clump_m = rng.range(1.0, 3.0);
            let height_m = rng.range(8.0, 20.0);
            let seed = rng.next_u64();
            let (temps, alpha) = paint_canopy(
                centre, size, base_c, cover, offset_c, mottle_c, clump_m, seed, table,
            );
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
            crowns,
            branches,
            n_trees,
            n_debris,
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
    for b in &look.shadows {
        blob(&mut field, n, ts, at, b, look.base_c, table, look.seed);
    }
    for s in &look.tracks {
        stroke(&mut field, n, ts, at, s, look.base_c);
    }
    for b in &look.debris {
        blob(&mut field, n, ts, at, b, look.base_c, table, look.seed);
    }
    for b in &look.crowns {
        blob(&mut field, n, ts, at, b, look.base_c, table, look.seed);
    }
    for s in &look.branches {
        stroke(&mut field, n, ts, at, s, look.base_c);
    }
    field
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

/// Blend toward `base + offset` along a segment, antialiased over one texel.
fn stroke(
    field: &mut [f32],
    n: usize,
    ts: f64,
    at: impl Fn([f64; 2]) -> (f64, f64),
    s: &Stroke,
    base_c: f64,
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
            let cov = band(d, half_drawn, 1.0) as f64 * strength;
            if cov > 0.0 {
                let v = &mut field[j * n + i];
                let target = base_c + s.offset_c + (*v as f64 - base_c) * 0.3;
                *v = (*v as f64 * (1.0 - cov) + target * cov) as f32;
            }
        }
    }
}

/// Blend toward `base + offset` inside a soft ellipse (spiky for conifer crowns).
#[allow(clippy::too_many_arguments)]
fn blob(
    field: &mut [f32],
    n: usize,
    ts: f64,
    at: impl Fn([f64; 2]) -> (f64, f64),
    b: &Blob,
    base_c: f64,
    table: &NoiseTable,
    seed: u64,
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
            let cov = ((1.0 - d) / soft).clamp(0.0, 1.0);
            if cov > 0.0 {
                let v = &mut field[j * n + i];
                let target = base_c + b.offset_c * texture + (*v as f64 - base_c) * 0.4;
                *v = (*v as f64 * (1.0 - cov) + target * cov) as f32;
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
