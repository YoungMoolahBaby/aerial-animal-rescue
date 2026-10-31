//! The farm ground of one RGB shot, laid out in metres and painted into one image that
//! covers the camera's footprint. No Bevy here.
//!
//! A sim-only model trained on RGB v1 fired ~10 times per empty real frame. It fired on
//! the bright straight things real fields are full of and v1 had none of: wheel ruts and
//! tramlines, track edges, field boundaries, straw swaths. So the ground is now:
//! - fields with straight edges between different covers (pasture, dry grass, stubble,
//!   a green crop, bare soil). An edge is plain, fenced (a strip of uncut grass and a
//!   faint wire with posts) or hedged (`scene.rs` puts hedge geometry on it);
//! - farm tracks: two bright wheel ruts with grass or a worn strip between them, a verge
//!   on each side; sometimes gravel, sometimes a paved road;
//! - inside a field, along its rows: mowing stripes, tramlines (pairs of wheel tracks
//!   every 12-36 m), straw swaths on stubble, and curved vehicle ruts across grass.
//!
//! The image is about one texel per picture pixel (`OVERLAY_TEX` across the footprint), so
//! lines a few decimetres wide come out sharp like in a 5 K photo scaled to 1280.

use std::f64::consts::{PI, TAU};

use crate::rgb::{Rgb, albedo, mix, pick, vary};
use crate::rng::{Rng, lattice};

/// Texels across the painted ground.
pub const OVERLAY_TEX: usize = 2048;
/// Distance between the two wheels of a tractor or car, metres.
const GAUGE_M: f64 = 1.8;

/// What a field is covered with.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Field {
    /// Grazed grass.
    Pasture,
    /// Dry summer grass.
    Dry,
    /// Harvested cereal: straw stubble, often with swaths and bales (ICAERUS flight 032).
    Stubble,
    /// A young green crop in drilled rows, with tramlines.
    Crop,
    /// Ploughed or bare soil.
    Bare,
}

impl Field {
    pub fn name(self) -> &'static str {
        match self {
            Field::Pasture => "pasture",
            Field::Dry => "dry",
            Field::Stubble => "stubble",
            Field::Crop => "crop",
            Field::Bare => "bare",
        }
    }

    fn random(rng: &mut Rng) -> Self {
        pick(
            rng,
            &[
                (0.45, Field::Pasture),
                (0.2, Field::Dry),
                (0.15, Field::Stubble),
                (0.12, Field::Crop),
                (0.08, Field::Bare),
            ],
        )
    }
}

/// One field: its cover and what's drawn along its rows.
#[derive(Clone, Debug)]
pub struct Parcel {
    pub field: Field,
    /// Main cover, patches of a second cover, and bare soil (gateways, molehills).
    pub main: Rgb,
    pub second: Rgb,
    pub soil: Rgb,
    /// Unit vector along its rows (drilling, mowing, harvesting).
    pub dir: [f64; 2],
    /// Mowing stripes: brightness change (0 = none) and width, metres.
    pub mow: f64,
    pub mow_m: f64,
    /// Straw swaths: spacing and width, metres (stubble only).
    pub swaths: Option<(f64, f64)>,
    pub straw: Rgb,
    /// Tramlines: spacing (0 = none) and offset, metres, and the colour of their ruts.
    pub tram_m: f64,
    pub tram_off: f64,
    pub rut: Rgb,
    /// Fine texture strength and grain (every few centimetres its own brightness: real
    /// photos of grass differ by ~9 grey levels from one pixel to the next at 1280).
    pub texture: f64,
    pub grain: f64,
    /// Where it reads the noise table, so neighbouring fields don't share a pattern.
    pub offset: [f64; 2],
    /// Share of `SPECK_CELL_M` cells holding a speck (dead grass, stones, dung, ...).
    pub specks: f64,
}

impl Parcel {
    fn random(rng: &mut Rng, field: Field, dir: [f64; 2]) -> Self {
        let soil = vary(rng, albedo(125, 100, 75), 0.3);
        let (main, second) = match field {
            Field::Pasture => (
                vary(rng, albedo(110, 135, 60), 0.25),
                if rng.chance(0.5) {
                    vary(rng, albedo(150, 145, 95), 0.2) // paler, drier patches
                } else {
                    vary(rng, albedo(75, 105, 45), 0.2) // lush dark patches
                },
            ),
            Field::Dry => (
                vary(rng, albedo(165, 150, 105), 0.2),
                vary(rng, albedo(115, 125, 70), 0.2),
            ),
            Field::Stubble => (
                vary(rng, albedo(200, 175, 120), 0.15),
                vary(rng, albedo(150, 130, 90), 0.2),
            ),
            Field::Crop => (
                vary(rng, albedo(95, 140, 55), 0.2),
                vary(rng, albedo(120, 150, 70), 0.15),
            ),
            Field::Bare => (
                vary(rng, albedo(140, 112, 82), 0.3),
                vary(rng, albedo(110, 90, 70), 0.2),
            ),
        };
        let (mow, mow_m) = match field {
            Field::Pasture if rng.chance(0.25) => (rng.range(0.04, 0.1), rng.range(3.0, 8.0)),
            Field::Dry if rng.chance(0.15) => (rng.range(0.03, 0.08), rng.range(3.0, 8.0)),
            Field::Stubble => (rng.range(0.03, 0.08), rng.range(2.0, 5.0)),
            Field::Bare => (rng.range(0.03, 0.1), rng.range(0.6, 2.0)), // furrows
            _ => (0.0, 1.0),
        };
        let swaths = (field == Field::Stubble && rng.chance(0.6))
            .then(|| (rng.range(5.0, 9.0), rng.range(1.2, 2.2)));
        let tram = match field {
            Field::Crop => 0.8,
            Field::Stubble => 0.5,
            Field::Bare => 0.2,
            Field::Dry => 0.1,
            Field::Pasture => 0.05,
        };
        let tram_m = if rng.chance(tram) {
            pick(
                rng,
                &[
                    (0.3, 12.0),
                    (0.2, 18.0),
                    (0.3, 24.0),
                    (0.1, 28.0),
                    (0.1, 36.0),
                ],
            )
        } else {
            0.0
        };
        Self {
            field,
            main,
            second,
            soil,
            dir,
            mow,
            mow_m,
            swaths,
            straw: vary(rng, albedo(225, 205, 150), 0.1),
            tram_m,
            tram_off: rng.range(0.0, 36.0),
            rut: vary(rng, albedo(165, 145, 115), 0.2),
            texture: rng.range(0.1, 0.35),
            grain: rng.range(0.3, 0.6),
            offset: [rng.range(0.0, 1000.0), rng.range(0.0, 1000.0)],
            specks: match field {
                Field::Pasture => rng.range(0.1, 0.45),
                Field::Dry => rng.range(0.1, 0.4),
                Field::Stubble | Field::Bare => rng.range(0.05, 0.2),
                Field::Crop => rng.range(0.02, 0.08),
            },
        }
    }
}

/// How a field edge looks.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Edge {
    Plain,
    Fence,
    Hedge,
}

/// A straight field edge: the points p with `normal . p = offset`.
#[derive(Clone, Debug)]
pub struct Boundary {
    pub normal: [f64; 2],
    pub offset: f64,
    pub edge: Edge,
    /// Uncut grass on both sides (fences and hedges): width each side, metres.
    pub strip_m: f64,
    pub strip: Rgb,
}

impl Boundary {
    /// Unit vector along the edge.
    pub fn along(&self) -> [f64; 2] {
        [-self.normal[1], self.normal[0]]
    }

    /// Signed distance of (x, z) from the edge.
    pub fn distance(&self, x: f64, z: f64) -> f64 {
        self.normal[0] * x + self.normal[1] * z - self.offset
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Surface {
    Dirt,
    Gravel,
    Paved,
}

/// A farm track or road along a polyline.
#[derive(Clone, Debug)]
pub struct Track {
    pub points: Vec<[f64; 2]>,
    /// Half its width, metres.
    pub half: f64,
    pub surface: Surface,
    pub colour: Rgb,
    /// Grass between the wheel ruts (dirt and gravel tracks).
    pub centre_grass: bool,
}

/// A vehicle's wheel ruts across grass: an arc.
#[derive(Clone, Debug)]
pub struct Ruts {
    pub centre: [f64; 2],
    pub radius: f64,
    pub from: f64,
    pub span: f64,
    /// Brightness change on the ruts (flattened grass is lighter or darker).
    pub tone: f64,
}

/// The ground of one shot: fields, edges, tracks, ruts, within a square around `centre`.
#[derive(Clone, Debug)]
pub struct Layout {
    pub centre: [f64; 2],
    pub size: f64,
    pub boundaries: Vec<Boundary>,
    /// Indexed by which side of each boundary a point is (bit k = boundary k).
    pub parcels: Vec<Parcel>,
    pub tracks: Vec<Track>,
    pub ruts: Vec<Ruts>,
    pub seed: u64,
}

fn unit(angle: f64) -> [f64; 2] {
    [angle.cos(), angle.sin()]
}

impl Layout {
    /// A farm around `centre`, laid out for a square `size` metres across. Fields follow
    /// one grid angle (give or take a few degrees), as real field systems do.
    pub fn random(rng: &mut Rng, centre: [f64; 2], size: f64) -> Self {
        let grid = rng.range(0.0, PI);
        let n_edges = pick(rng, &[(0.15, 0usize), (0.3, 1), (0.35, 2), (0.2, 3)]);
        let boundaries: Vec<Boundary> = (0..n_edges)
            .map(|_| {
                let angle =
                    grid + if rng.chance(0.5) { PI / 2.0 } else { 0.0 } + rng.range(-0.07, 0.07);
                let normal = unit(angle);
                let edge = pick(
                    rng,
                    &[(0.35, Edge::Plain), (0.35, Edge::Fence), (0.3, Edge::Hedge)],
                );
                let strip = if rng.chance(0.6) {
                    vary(rng, albedo(80, 105, 50), 0.2)
                } else {
                    vary(rng, albedo(170, 160, 110), 0.15)
                };
                Boundary {
                    normal,
                    offset: normal[0] * centre[0]
                        + normal[1] * centre[1]
                        + rng.range(-0.45, 0.45) * size,
                    edge,
                    strip_m: if edge == Edge::Plain {
                        0.0
                    } else {
                        rng.range(0.4, 1.8)
                    },
                    strip,
                }
            })
            .collect();
        let parcels = (0..1usize << n_edges)
            .map(|_| {
                let rows = grid + if rng.chance(0.5) { PI / 2.0 } else { 0.0 };
                let field = Field::random(rng);
                Parcel::random(rng, field, unit(rows))
            })
            .collect();
        let n_tracks = pick(rng, &[(0.35, 0usize), (0.45, 1), (0.2, 2)]);
        let tracks = (0..n_tracks)
            .map(|_| Track::random(rng, centre, size, grid, &boundaries))
            .collect();
        let ruts = (0..rng.index(4))
            .map(|_| Ruts {
                centre: [
                    centre[0] + rng.range(-0.5, 0.5) * size,
                    centre[1] + rng.range(-0.5, 0.5) * size,
                ],
                radius: rng.range(15.0, 80.0),
                from: rng.range(0.0, TAU),
                span: rng.range(0.3, 1.5),
                tone: rng.range(0.06, 0.15) * if rng.chance(0.5) { 1.0 } else { -1.0 },
            })
            .collect();
        Self {
            centre,
            size,
            boundaries,
            parcels,
            tracks,
            ruts,
            seed: rng.next_u64(),
        }
    }

    pub fn parcel_at(&self, x: f64, z: f64) -> &Parcel {
        let bits = self.boundaries.iter().enumerate().fold(0, |acc, (k, b)| {
            acc | ((b.distance(x, z) > 0.0) as usize) << k
        });
        &self.parcels[bits]
    }

    /// Where hedge segments go: (x, z, unit vector along the hedge), every 3.5-5.5 m
    /// along each hedged edge inside the square, with a gap now and then. At most `max`.
    pub fn hedge_segments(&self, rng: &mut Rng, max: usize) -> Vec<(f64, f64, [f64; 2])> {
        let half = self.size / 2.0 + 5.0;
        let mut out = Vec::new();
        for b in self.boundaries.iter().filter(|b| b.edge == Edge::Hedge) {
            let a = b.along();
            let base = [b.normal[0] * b.offset, b.normal[1] * b.offset];
            let mid = a[0] * self.centre[0] + a[1] * self.centre[1];
            let mut s = mid - self.size;
            while s < mid + self.size && out.len() < max {
                let (x, z) = (base[0] + a[0] * s, base[1] + a[1] * s);
                let inside = (x - self.centre[0]).abs() < half && (z - self.centre[1]).abs() < half;
                if inside && !rng.chance(0.08) {
                    let j = rng.range(-0.4, 0.4);
                    out.push((x + b.normal[0] * j, z + b.normal[1] * j, a));
                }
                s += rng.range(3.5, 5.5);
            }
        }
        out
    }

    /// A spot on a track and the track's direction there (for parked vehicles), if any
    /// track crosses the square.
    pub fn track_spot(&self, rng: &mut Rng) -> Option<(f64, f64, [f64; 2])> {
        if self.tracks.is_empty() {
            return None;
        }
        let t = &self.tracks[rng.index(self.tracks.len())];
        let half = self.size / 2.0;
        let inside: Vec<usize> = (0..t.points.len() - 1)
            .filter(|&k| {
                let p = t.points[k];
                (p[0] - self.centre[0]).abs() < half && (p[1] - self.centre[1]).abs() < half
            })
            .collect();
        let &k = inside.get(rng.index(inside.len().max(1)))?;
        let (p, q) = (t.points[k], t.points[k + 1]);
        let len = ((q[0] - p[0]).powi(2) + (q[1] - p[1]).powi(2))
            .sqrt()
            .max(1e-9);
        let dir = [(q[0] - p[0]) / len, (q[1] - p[1]) / len];
        let f = rng.unit();
        let side = rng.range(-0.5, 0.5) * t.half;
        Some((
            p[0] + (q[0] - p[0]) * f - dir[1] * side,
            p[1] + (q[1] - p[1]) * f + dir[0] * side,
            dir,
        ))
    }
}

impl Track {
    fn random(rng: &mut Rng, centre: [f64; 2], size: f64, grid: f64, edges: &[Boundary]) -> Self {
        let surface = pick(
            rng,
            &[
                (0.5, Surface::Dirt),
                (0.35, Surface::Gravel),
                (0.15, Surface::Paved),
            ],
        );
        let (half, colour) = match surface {
            Surface::Dirt => (rng.range(1.4, 2.0), vary(rng, albedo(150, 125, 95), 0.2)),
            Surface::Gravel => (rng.range(1.4, 2.2), vary(rng, albedo(185, 178, 160), 0.12)),
            Surface::Paved => (rng.range(2.5, 3.5), vary(rng, albedo(95, 95, 95), 0.2)),
        };
        // along a field edge (just inside it), or across the fields
        let (start, heading, drift) = if !edges.is_empty() && rng.chance(0.4) {
            let b = &edges[rng.index(edges.len())];
            let off = (half + rng.range(1.0, 4.0)) * if rng.chance(0.5) { 1.0 } else { -1.0 };
            let a = b.along();
            let s = a[0] * centre[0] + a[1] * centre[1];
            let start = [
                b.normal[0] * (b.offset + off) + a[0] * s,
                b.normal[1] * (b.offset + off) + a[1] * s,
            ];
            (start, a[1].atan2(a[0]), 0.0)
        } else {
            let start = [
                centre[0] + rng.range(-0.4, 0.4) * size,
                centre[1] + rng.range(-0.4, 0.4) * size,
            ];
            let heading = if rng.chance(0.4) {
                grid + if rng.chance(0.5) { PI / 2.0 } else { 0.0 } + rng.range(-0.17, 0.17)
            } else {
                rng.range(0.0, TAU)
            };
            (start, heading, 0.03)
        };
        let step = 8.0;
        let n = (size / step).ceil() as usize + 4;
        let mut walk = |dir: f64| {
            let (mut p, mut h) = (start, heading + dir);
            let mut pts = Vec::with_capacity(n);
            for _ in 0..n {
                h += rng.range(-drift, drift);
                p = [p[0] + step * h.cos(), p[1] + step * h.sin()];
                pts.push(p);
            }
            pts
        };
        let back = walk(PI);
        let ahead = walk(0.0);
        let mut points: Vec<[f64; 2]> = back.into_iter().rev().collect();
        points.push(start);
        points.extend(ahead);
        Self {
            points,
            half,
            surface,
            colour,
            centre_grass: surface != Surface::Paved && rng.chance(0.6),
        }
    }
}

/// A tiling value-noise table (fbm, about -0.6..0.6) to read smooth noise from cheaply.
pub struct NoiseTable {
    n: usize,
    v: Vec<f32>,
}

/// Cells of the table per feature: `NoiseTable::at(x / f, ..)` has features `f` across.
const TABLE_FEATURE: f64 = 16.0;

impl NoiseTable {
    pub fn new(seed: u64) -> Self {
        let n = 256;
        let v = (0..n * n)
            .map(|k| {
                let (u, w) = ((k % n) as f64 / n as f64, (k / n) as f64 / n as f64);
                crate::thermal::tiling_fbm(seed, u, w, n as f64 / TABLE_FEATURE) as f32
            })
            .collect();
        Self { n, v }
    }

    /// Bilinear read; one unit of (x, y) is one feature.
    fn at(&self, x: f64, y: f64) -> f64 {
        let (x, y) = (x * TABLE_FEATURE, y * TABLE_FEATURE);
        let (x0, y0) = (x.floor(), y.floor());
        let (fx, fy) = ((x - x0) as f32, (y - y0) as f32);
        let n = self.n as i64;
        let idx = |i: i64, j: i64| self.v[(j.rem_euclid(n) * n + i.rem_euclid(n)) as usize];
        let (i, j) = (x0 as i64, y0 as i64);
        let a = idx(i, j) + (idx(i + 1, j) - idx(i, j)) * fx;
        let b = idx(i, j + 1) + (idx(i + 1, j + 1) - idx(i, j + 1)) * fx;
        (a + (b - a) * fy) as f64
    }
}

/// Coverage of a line `half` wide at distance `d`, antialiased over one texel `ts`.
fn band(d: f64, half: f64, ts: f64) -> f32 {
    ((half + ts / 2.0 - d) / ts).clamp(0.0, 1.0) as f32
}

fn scale(c: Rgb, f: f64) -> Rgb {
    c.map(|v| (v * f as f32).clamp(0.0, 1.0))
}

/// Distance from (x, z) to each track, per texel (`f32::MAX` far from it): drawn segment by
/// segment over the texels near it, so it costs what the tracks cover, not the image.
fn track_distances(layout: &Layout, x0: f64, z0: f64, ts: f64, n: usize) -> Vec<Vec<f32>> {
    layout
        .tracks
        .iter()
        .map(|t| {
            let mut d = vec![f32::MAX; n * n];
            let reach = t.half + 1.5;
            for s in t.points.windows(2) {
                let (p, q) = (s[0], s[1]);
                let lo = |a: f64, b: f64, o: f64| {
                    ((a.min(b) - reach - o) / ts).floor().max(0.0) as usize
                };
                let hi = |a: f64, b: f64, o: f64| {
                    (((a.max(b) + reach - o) / ts).ceil().max(0.0) as usize).min(n)
                };
                let (i0, i1) = (lo(p[0], q[0], x0), hi(p[0], q[0], x0));
                let (j0, j1) = (lo(p[1], q[1], z0), hi(p[1], q[1], z0));
                let (dx, dz) = (q[0] - p[0], q[1] - p[1]);
                let len2 = (dx * dx + dz * dz).max(1e-9);
                for j in j0..j1 {
                    for i in i0..i1 {
                        let (x, z) = (x0 + (i as f64 + 0.5) * ts, z0 + (j as f64 + 0.5) * ts);
                        let f = (((x - p[0]) * dx + (z - p[1]) * dz) / len2).clamp(0.0, 1.0);
                        let (ex, ez) = (x - p[0] - f * dx, z - p[1] - f * dz);
                        let e = (ex * ex + ez * ez).sqrt() as f32;
                        let k = j * n + i;
                        d[k] = d[k].min(e);
                    }
                }
            }
            d
        })
        .collect()
}

/// Paint the layout into an `n` x `n` sRGB RGBA image: texel (i, j) is at
/// x = centre.x - size/2 + (i + 0.5) * size/n, z likewise with j.
pub fn paint(layout: &Layout, table: &NoiseTable, n: usize) -> Vec<u8> {
    let ts = layout.size / n as f64;
    let (x0, z0) = (
        layout.centre[0] - layout.size / 2.0,
        layout.centre[1] - layout.size / 2.0,
    );
    let tracks = track_distances(layout, x0, z0, ts, n);
    let mut out = vec![0u8; n * n * 4];
    let threads = std::thread::available_parallelism().map_or(4, |t| t.get());
    let rows_per = n.div_ceil(threads);
    std::thread::scope(|scope| {
        for (chunk, rows) in out.chunks_mut(rows_per * n * 4).enumerate() {
            let tracks = &tracks;
            scope.spawn(move || {
                for (r, row) in rows.chunks_mut(n * 4).enumerate() {
                    let j = chunk * rows_per + r;
                    for i in 0..n {
                        let (x, z) = (x0 + (i as f64 + 0.5) * ts, z0 + (j as f64 + 0.5) * ts);
                        let c = texel(layout, table, tracks, j * n + i, x, z, ts);
                        let px = &mut row[4 * i..4 * i + 4];
                        for k in 0..3 {
                            let s = crate::rgb::linear_to_srgb(c[k].clamp(0.0, 1.0));
                            px[k] = (s * 255.0).round() as u8;
                        }
                        px[3] = 255;
                    }
                }
            });
        }
    });
    out
}

/// Albedo of the ground at (x, z).
fn texel(
    layout: &Layout,
    table: &NoiseTable,
    tracks: &[Vec<f32>],
    k: usize,
    x: f64,
    z: f64,
    ts: f64,
) -> Rgb {
    let p = layout.parcel_at(x, z);
    let (ox, oz) = (p.offset[0], p.offset[1]);
    // cover: second cover in patches tens of metres across, soil in a few small ones
    let patch = table.at(x / 30.0 + ox, z / 30.0 + oz);
    let mut c = mix(
        p.main,
        p.second,
        ((patch - 0.1) * 2.0).clamp(0.0, 0.5) as f32,
    );
    let soil = table.at(x / 8.0 + oz, z / 8.0 + ox);
    c = mix(c, p.soil, ((soil - 0.5) * 4.0).clamp(0.0, 0.6) as f32);
    // fine texture (tussocks, then clumps a hand across), grain, a hue drift between
    // greener and yellower
    let fine = table.at(x / 0.6 + ox * 7.0, z / 0.6);
    let clumps = table.at(x / 0.25 + oz * 5.0, z / 0.25);
    let hue = 0.15 * table.at(x / 3.0, z / 3.0 + oz * 3.0);
    let grain =
        2.0 * lattice(
            layout.seed,
            (x / 0.07).floor() as i64,
            (z / 0.07).floor() as i64,
        ) - 1.0;
    let mut b = (1.0 - p.texture * (0.5 + 0.5 * fine))
        * (1.0 + 0.8 * p.texture * clumps)
        * (1.0 + p.grain * grain);
    // rows: t across them, s along them
    let t = x * p.dir[1] - z * p.dir[0];
    let s = x * p.dir[0] + z * p.dir[1];
    if p.mow > 0.0 {
        b *= 1.0
            + p.mow
                * if (t / p.mow_m).floor() as i64 % 2 == 0 {
                    1.0
                } else {
                    -1.0
                };
    }
    c = [
        (c[0] as f64 * b * (1.0 + hue)) as f32,
        (c[1] as f64 * b) as f32,
        (c[2] as f64 * b * (1.0 + 1.5 * hue)) as f32,
    ];
    if let Some((spacing, width)) = p.swaths {
        let u = t.rem_euclid(spacing);
        let edge = width / 2.0 * (1.0 + 0.25 * fine);
        // swaths thin out and break where the baler has been
        let left = table.at(s / 12.0 + ox, t / 6.0) > -0.25;
        if left {
            let cov = band((u - spacing / 2.0).abs(), edge, ts.max(0.25));
            c = mix(c, scale(p.straw, 1.0 + 0.15 * grain), 0.9 * cov);
        }
    }
    if p.tram_m > 0.0 {
        let u = (t - p.tram_off).rem_euclid(p.tram_m);
        let d = u.min(p.tram_m - u).min((u - GAUGE_M).abs());
        let cov = band(d, 0.22, ts);
        c = mix(c, scale(p.rut, 1.0 + 0.1 * grain), 0.85 * cov);
    }
    c = specks(layout, p, table, c, x, z, ts, grain);
    for r in &layout.ruts {
        let (dx, dz) = (x - r.centre[0], z - r.centre[1]);
        let a = (dz.atan2(dx) - r.from).rem_euclid(TAU);
        if a < r.span {
            let d = (dx * dx + dz * dz).sqrt() - r.radius;
            let w = (d - GAUGE_M / 2.0).abs().min((d + GAUGE_M / 2.0).abs());
            c = scale(c, 1.0 + r.tone * band(w, 0.18, ts) as f64);
        }
    }
    for e in &layout.boundaries {
        let d = e.distance(x, z).abs();
        if e.strip_m > 0.0 && d < e.strip_m + ts {
            let cov = band(d, e.strip_m, ts);
            c = mix(c, scale(e.strip, b), 0.8 * cov);
        }
        if e.edge == Edge::Fence {
            // a wire (a faint dark line at this scale) and a post every 3.5 m
            let along = (x * -e.normal[1] + z * e.normal[0]).rem_euclid(3.5);
            let post = band(along.min(3.5 - along), 0.07, ts) * band(d, 0.07, ts);
            let wire = 0.35 * band(d, 0.02, ts);
            c = scale(c, 1.0 - 0.7 * post.max(wire) as f64);
        }
    }
    for (t, d) in layout.tracks.iter().zip(tracks) {
        let d = d[k] as f64;
        if d > t.half + 1.0 {
            continue;
        }
        let tex = 1.0 + 0.12 * grain + 0.08 * fine;
        match t.surface {
            Surface::Paved => {
                c = mix(c, scale(t.colour, tex * 0.9), band(d, t.half, ts));
                // a white edge line on some roads
                if t.half > 3.0 {
                    let line = band((d - (t.half - 0.3)).abs(), 0.06, ts);
                    c = mix(c, albedo(220, 220, 215), line);
                }
            }
            Surface::Dirt | Surface::Gravel => {
                // worn track between verges; bright ruts; grass or wear in the middle
                let worn = band(d, t.half, ts) * 0.5;
                c = mix(c, scale(t.colour, tex * 0.9), worn);
                let rut = band((d - GAUGE_M / 2.0).abs(), 0.28, ts);
                c = mix(c, scale(t.colour, tex), rut);
                if !t.centre_grass {
                    c = mix(c, scale(t.colour, tex * 0.85), band(d, 0.45, ts));
                }
            }
        }
    }
    c
}

/// Specks sit in cells this size, at most one per cell.
const SPECK_CELL_M: f64 = 1.5;

/// What a speck is: colour, radius range (m), and how much it covers its disc at most.
/// Kept under ~1 m and patchy: the first v3 render had specks up to 3 m long, smooth and
/// evenly filled, and the real-trained model took them for cows (8.2 alarms per empty
/// frame, cattle found 72% instead of 91%).
fn speck_kind(h: &mut Rng) -> (Rgb, (f64, f64), f32) {
    pick(
        h,
        &[
            // dead grass and seed heads, light bare spots: the pale ones a model can take
            // for a sheep at 10 px (RGB v2's alarms)
            (0.35, (albedo(205, 195, 155), (0.1, 0.35), 0.75)),
            (0.15, (albedo(180, 165, 135), (0.2, 0.5), 0.55)),
            (0.12, (albedo(195, 192, 185), (0.05, 0.2), 0.9)), // stones
            (0.15, (albedo(70, 60, 42), (0.1, 0.25), 0.8)),    // dung pats
            (0.08, (albedo(110, 85, 60), (0.2, 0.4), 0.8)),    // molehills
            (0.15, (albedo(50, 70, 32), (0.15, 0.4), 0.6)),    // thistles, docks
        ],
    )
}

/// The texel's colour with any speck over it: specks in `SPECK_CELL_M` cells, one cell in
/// `parcel.specks` (more in some patches tens of metres across than in others), each its
/// own kind, size, elongation and turn, with a ragged edge.
#[allow(clippy::too_many_arguments)]
fn specks(
    layout: &Layout,
    p: &Parcel,
    table: &NoiseTable,
    mut c: Rgb,
    x: f64,
    z: f64,
    ts: f64,
    grain: f64,
) -> Rgb {
    if p.specks <= 0.0 {
        return c;
    }
    let patchy = (0.3 + 1.4 * (0.5 + 0.5 * table.at(x / 15.0 + 3.7, z / 15.0))).max(0.0);
    let share = p.specks * patchy;
    let (ci, cj) = (
        (x / SPECK_CELL_M).floor() as i64,
        (z / SPECK_CELL_M).floor() as i64,
    );
    for dj in -1..=1 {
        for di in -1..=1 {
            let (i, j) = (ci + di, cj + dj);
            let key = layout.seed
                ^ 0x5BEC_u64
                ^ (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
                ^ (j as u64).wrapping_mul(0xC2B2_AE3D_27D4_EB4F);
            let mut h = Rng::new(key);
            if h.unit() >= share {
                continue;
            }
            let (cx, cz) = (
                (i as f64 + h.unit()) * SPECK_CELL_M,
                (j as f64 + h.unit()) * SPECK_CELL_M,
            );
            let (colour, (r0, r1), amount) = speck_kind(&mut h);
            let r = h.range(r0, r1);
            let (sa, ca) = h.range(0.0, PI).sin_cos();
            let long = h.range(1.0, 1.5);
            let amount = amount * h.range(0.5, 1.0) as f32;
            let (p2, p3) = (h.range(0.0, TAU), h.range(0.0, TAU));
            let (dx, dz) = (x - cx, z - cz);
            if dx.abs() > 1.0 || dz.abs() > 1.0 {
                continue;
            }
            // in the speck's own axes, stretched along one; an irregular rim
            let (u, v) = (dx * ca + dz * sa, -dx * sa + dz * ca);
            let d = ((u / long).powi(2) + v * v).sqrt();
            let a = v.atan2(u);
            let rim = r * (1.0 + 0.25 * (2.0 * a + p2).sin() + 0.2 * (3.0 * a + p3).sin());
            let ragged = rim * (1.0 + 0.3 * grain);
            // patchy inside: tufts, not paint
            let fill = (0.55 + 0.45 * table.at(x / 0.15 + 7.1, z / 0.15)).clamp(0.2, 1.0) as f32;
            let cov = band(d, ragged, ts) * amount * fill;
            if cov > 0.0 {
                c = mix(c, scale(colour, 1.0 + 0.12 * grain), cov);
            }
        }
    }
    c
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parcels_cover_every_side_combination() {
        let mut rng = Rng::new(1);
        for _ in 0..50 {
            let l = Layout::random(&mut rng, [10.0, -20.0], 200.0);
            assert_eq!(l.parcels.len(), 1 << l.boundaries.len());
            for &(x, z) in &[(0.0, 0.0), (100.0, 50.0), (-90.0, 80.0)] {
                let _ = l.parcel_at(x, z);
            }
        }
    }

    #[test]
    fn hedges_follow_their_edge_inside_the_square() {
        let mut rng = Rng::new(2);
        let mut seen = 0;
        for _ in 0..40 {
            let l = Layout::random(&mut rng, [0.0, 0.0], 150.0);
            for (x, z, a) in l.hedge_segments(&mut rng, 200) {
                seen += 1;
                assert!(x.abs() < 81.0 && z.abs() < 81.0);
                let on_edge = l.boundaries.iter().any(|b| {
                    b.edge == Edge::Hedge
                        && b.distance(x, z).abs() < 0.5
                        && (b.along()[0] * a[0] + b.along()[1] * a[1]).abs() > 0.999
                });
                assert!(on_edge);
            }
        }
        assert!(seen > 50, "{seen} hedge segments");
    }

    #[test]
    fn tracks_paint_bright_ruts_a_gauge_apart() {
        let mut rng = Rng::new(3);
        let mut l = Layout::random(&mut rng, [0.0, 0.0], 64.0);
        l.boundaries.clear();
        l.parcels.truncate(1);
        l.ruts.clear();
        l.parcels[0].tram_m = 0.0;
        l.parcels[0].swaths = None;
        l.parcels[0].specks = 0.0;
        l.parcels[0].main = albedo(60, 90, 40);
        l.parcels[0].second = l.parcels[0].main;
        l.tracks = vec![Track {
            points: vec![[-100.0, 0.0], [100.0, 0.0]],
            half: 1.8,
            surface: Surface::Gravel,
            colour: albedo(190, 180, 160),
            centre_grass: true,
        }];
        let n = 640; // 10 cm texels
        let img = paint(&l, &NoiseTable::new(4), n);
        // average green down the column of texels across the track
        let lum = |z: f64| {
            let j = ((z + 32.0) / 0.1) as usize;
            (0..n).map(|i| img[4 * (j * n + i) + 1] as f64).sum::<f64>() / n as f64
        };
        let (rut, centre, field) = (lum(GAUGE_M / 2.0), lum(0.0), lum(10.0));
        assert!(
            rut > centre + 30.0 && rut > field + 30.0,
            "{rut} {centre} {field}"
        );
    }

    #[test]
    fn specks_dot_the_grass_at_their_density() {
        let mut rng = Rng::new(6);
        let mut l = Layout::random(&mut rng, [0.0, 0.0], 64.0);
        l.boundaries.clear();
        l.parcels.truncate(1);
        l.tracks.clear();
        l.ruts.clear();
        let p = &mut l.parcels[0];
        (p.field, p.tram_m, p.swaths, p.mow) = (Field::Pasture, 0.0, None, 0.0);
        (p.main, p.second, p.texture, p.grain) =
            (albedo(90, 120, 50), albedo(90, 120, 50), 0.0, 0.0);
        let table = NoiseTable::new(7);
        let changed = |l: &Layout| {
            let img = paint(l, &table, 320);
            let base = img[1];
            img.chunks(4)
                .filter(|px| (px[1] as i32 - base as i32).abs() > 12)
                .count() as f64
                / (320.0 * 320.0)
        };
        l.parcels[0].specks = 0.0;
        let none = changed(&l);
        l.parcels[0].specks = 0.3;
        let some = changed(&l);
        // a speck covers ~0.05-0.6 m2, faintly in places, of each 2.25 m2 cell holding one
        assert!(
            none < 0.005 && (0.005..0.3).contains(&some),
            "{none} {some}"
        );
    }

    #[test]
    fn noise_table_tiles_and_stays_small() {
        let t = NoiseTable::new(5);
        assert!((t.at(0.3, 0.7) - t.at(0.3 + 16.0, 0.7 - 16.0)).abs() < 1e-6);
        for k in 0..200 {
            assert!(t.at(k as f64 * 0.37, k as f64 * 0.11).abs() < 1.2);
        }
    }
}
