//! A farmyard in the RGB ground: sheds, railed pens, mud, a silage clamp. No Bevy here;
//! `fields.rs` paints its ground and `scene.rs` builds its sheds, rails and feeders.
//!
//! Why: on real val flight 061, 47 of 106 cows stand in a yard (pens with straw and mud
//! bedding, metal rails, half under a shed roof, packed together). Models trained on
//! scarce real data plus RGB v3 sim found none of them; the sim had only open fields
//! (sim/PLAN.md, RGB v4). The look follows the yards of the train flights (060, 062,
//! Derval 012/014): long sheds with light corrugated roofs, some open-fronted; pens in
//! front of them; a trampled muddy gateway into the field with ring feeders; a silage
//! clamp under black plastic held down by tyres; a manure heap; wrapped bales.
//!
//! Herds in a yard pack close and line up as they do there (RGB v5, `Yard::herd_slots`):
//! heads in round a ring feeder, side by side in rows facing a rail, a looser bunch on the
//! mud.
//!
//! The yard has its own frame: u along the shed row (the field grid's angle), v away from
//! the sheds towards the pens. Sheds sit at v < 0, pens at v > 0.

use std::f64::consts::{PI, TAU};

use crate::fields::{NoiseTable, band, scale};
use crate::rgb::{Rgb, albedo, mix, pick, vary};
use crate::rng::Rng;

/// A rectangle in yard coordinates: centre and half sizes along u and v.
#[derive(Clone, Copy, Debug)]
pub struct Rect {
    pub c: [f64; 2],
    pub half: [f64; 2],
}

impl Rect {
    fn new(u0: f64, u1: f64, v0: f64, v1: f64) -> Self {
        Self {
            c: [(u0 + u1) / 2.0, (v0 + v1) / 2.0],
            half: [(u1 - u0).abs() / 2.0, (v1 - v0).abs() / 2.0],
        }
    }

    /// Signed distance to the edge: negative inside.
    fn distance(&self, u: f64, v: f64) -> f64 {
        ((u - self.c[0]).abs() - self.half[0]).max((v - self.c[1]).abs() - self.half[1])
    }

    fn contains(&self, u: f64, v: f64, margin: f64) -> bool {
        self.distance(u, v) < margin
    }

    /// Coverage, antialiased over a texel `ts`.
    fn cover(&self, u: f64, v: f64, ts: f64) -> f32 {
        ((0.5 - self.distance(u, v) / ts) as f32).clamp(0.0, 1.0)
    }

    /// A random point at least `inset` metres inside.
    fn spot(&self, rng: &mut Rng, inset: f64) -> [f64; 2] {
        let hu = (self.half[0] - inset).max(0.0);
        let hv = (self.half[1] - inset).max(0.0);
        [
            self.c[0] + rng.range(-hu, hu.max(1e-9)),
            self.c[1] + rng.range(-hv, hv.max(1e-9)),
        ]
    }
}

/// What a pen's floor is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Floor {
    Straw,
    Mud,
    Concrete,
}

#[derive(Clone, Debug)]
pub struct Pen {
    pub rect: Rect,
    pub floor: Floor,
    /// Bay dividers: u positions of rails across the pen.
    pub bays: Vec<f64>,
    /// The gate: a gap in the outer rail, (u centre, half width).
    pub gate: (f64, f64),
}

/// What feeds or waters the cows: a ring feeder round a bale (1 m radius), a 3 m trough,
/// a round water tank.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Feed {
    Ring,
    Trough,
    Tank,
}

/// How many of each feeder a yard can have: the scene builds this many meshes of each, in
/// this order.
pub const FEEDS: [(Feed, usize); 3] = [(Feed::Ring, 6), (Feed::Trough, 4), (Feed::Tank, 2)];

/// Hoop radius of a ring feeder, metres.
pub const RING_R: f64 = 1.0;

/// Where one cow of a packed herd goes: yard-frame spot, the yard direction its head
/// faces, and whether it must stand (heads in a ring feeder).
#[derive(Clone, Copy, Debug)]
pub struct Slot {
    pub p: [f64; 2],
    pub facing: f64,
    pub stand: bool,
}

#[derive(Clone, Debug)]
pub struct Shed {
    pub rect: Rect,
    pub height: f64,
    /// Open-fronted (roof on posts, open towards +v): cows lie inside on straw.
    pub open: bool,
}

/// Trampled mud: a blob stretched along `dir` (radians, yard frame), ragged at its edge.
#[derive(Clone, Copy, Debug)]
pub struct Mud {
    pub c: [f64; 2],
    pub r: f64,
    pub stretch: f64,
    pub dir: f64,
}

impl Mud {
    fn distance(&self, u: f64, v: f64) -> f64 {
        let (du, dv) = (u - self.c[0], v - self.c[1]);
        let (s, c) = self.dir.sin_cos();
        let (a, b) = (du * c + dv * s, -du * s + dv * c);
        ((a / self.stretch).powi(2) + b * b).sqrt()
    }
}

/// A part of the yard cows stand in: a rectangle (kept `inset` metres off its edge) or mud.
#[derive(Clone, Copy, Debug)]
pub enum Area {
    Rect(Rect, f64),
    Mud(Mud),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Surface {
    Concrete,
    Gravel,
    Dirt,
}

/// A farmyard; see the module doc.
#[derive(Clone, Debug)]
pub struct Yard {
    pub origin: [f64; 2],
    /// Unit vector of u in the world (x, z); v is (-axis.z, axis.x).
    pub axis: [f64; 2],
    pub sheds: Vec<Shed>,
    pub pens: Vec<Pen>,
    /// Hard standing around the buildings.
    pub apron: Option<(Rect, Surface)>,
    /// A concrete feed passage beyond the pens, with silage along the barrier (v of the
    /// barrier side is `rect.c.v - half.v`).
    pub alley: Option<Rect>,
    pub mud: Vec<Mud>,
    pub clamp: Option<Rect>,
    /// Manure heap: stretched along u.
    pub heap: Option<Mud>,
    /// Ring feeders, troughs and water tanks (yard frame, turn, which).
    pub feeders: Vec<([f64; 2], f64, Feed)>,
    /// Wrapped bales, tyre stacks, tanks lined up by the buildings (yard frame, turn).
    pub stacks: Vec<([f64; 2], f64)>,
    colours: Colours,
    /// Everything lies within this distance of `origin`.
    reach: f64,
}

#[derive(Clone, Copy, Debug)]
struct Colours {
    straw: Rgb,
    straw_wet: Rgb,
    mud: Rgb,
    mud_wet: Rgb,
    concrete: Rgb,
    dung: Rgb,
    apron: Rgb,
    plastic: Rgb,
    heap: Rgb,
    silage: Rgb,
}

impl Yard {
    /// A yard at `origin` with its sheds along `axis`.
    pub fn random(rng: &mut Rng, origin: [f64; 2], axis: [f64; 2]) -> Self {
        let mut sheds = Vec::new();
        let mut u = 0.0;
        let depth = rng.range(10.0, 22.0);
        for _ in 0..1 + rng.index(3) {
            let len = rng.range(15.0, 45.0);
            // sheds in a row share their back line, some deeper or shallower
            let d = depth * rng.range(0.8, 1.15);
            sheds.push(Shed {
                rect: Rect::new(u, u + len, -d, 0.0),
                height: rng.range(4.0, 8.0),
                open: rng.chance(0.5),
            });
            u += len + rng.range(2.0, 10.0);
        }
        let row = u - 2.0;
        for s in &mut sheds {
            s.rect.c[0] -= row / 2.0;
        }
        let pens: Vec<Pen> = sheds
            .iter()
            .filter_map(|s| {
                if !rng.chance(0.8) {
                    return None;
                }
                let u0 = s.rect.c[0] - s.rect.half[0] + rng.range(0.0, 3.0);
                let u1 = s.rect.c[0] + s.rect.half[0] - rng.range(0.0, 3.0);
                let v1 = rng.range(5.0, 16.0);
                let floor = pick(
                    rng,
                    &[
                        (0.4, Floor::Straw),
                        (0.4, Floor::Mud),
                        (0.2, Floor::Concrete),
                    ],
                );
                let n = rng.index(4);
                let bays = (1..=n)
                    .map(|k| u0 + (u1 - u0) * k as f64 / (n + 1) as f64)
                    .collect();
                let gate = (
                    rng.range(u0 + 2.0, (u1 - 2.0).max(u0 + 2.1)),
                    rng.range(1.5, 2.5),
                );
                Some(Pen {
                    rect: Rect::new(u0, u1, 0.0, v1),
                    floor,
                    bays,
                    gate,
                })
            })
            .collect();
        let front = pens
            .iter()
            .map(|p| p.rect.c[1] + p.rect.half[1])
            .fold(0.0, f64::max);
        let alley = (!pens.is_empty() && rng.chance(0.35)).then(|| {
            let w = rng.range(3.0, 5.0);
            Rect::new(-row / 2.0, row / 2.0, front, front + w)
        });
        let outer = alley.map_or(front, |a| a.c[1] + a.half[1]);
        // trampled mud at the gates and beyond, sometimes a wide gateway into the field
        let mut mud = Vec::new();
        for p in &pens {
            if rng.chance(0.6) {
                mud.push(Mud {
                    c: [p.gate.0, p.rect.c[1] + p.rect.half[1] + rng.range(1.0, 4.0)],
                    r: rng.range(3.0, 8.0),
                    stretch: rng.range(1.0, 2.0),
                    dir: rng.range(0.0, PI),
                });
            }
        }
        if rng.chance(0.5) {
            mud.push(Mud {
                c: [
                    rng.range(-row / 2.0, row / 2.0),
                    outer + rng.range(5.0, 15.0),
                ],
                r: rng.range(8.0, 20.0),
                stretch: rng.range(1.0, 2.5),
                dir: rng.range(0.0, PI),
            });
        }
        let apron = rng.chance(0.75).then(|| {
            let m = rng.range(3.0, 12.0);
            let back = -depth * 1.2 - rng.range(2.0, 10.0);
            let surface = pick(
                rng,
                &[
                    (0.4, Surface::Concrete),
                    (0.35, Surface::Gravel),
                    (0.25, Surface::Dirt),
                ],
            );
            (
                Rect::new(-row / 2.0 - m, row / 2.0 + m, back, outer.max(2.0)),
                surface,
            )
        });
        // clamp and heap off one end of the row
        let side = if rng.chance(0.5) { 1.0 } else { -1.0 };
        let mut end = row / 2.0 + rng.range(4.0, 12.0);
        let clamp = rng.chance(0.55).then(|| {
            let (w, l) = (rng.range(8.0, 15.0), rng.range(18.0, 40.0));
            let v0 = rng.range(-depth, 0.0);
            let r = Rect::new(side * end, side * (end + w), v0 - l / 2.0, v0 + l / 2.0);
            end += w + rng.range(3.0, 8.0);
            r
        });
        let heap = rng.chance(0.4).then(|| Mud {
            c: [side * (end + 3.0), rng.range(-depth, 5.0)],
            r: rng.range(2.0, 4.0),
            stretch: rng.range(2.0, 4.0),
            dir: PI / 2.0 + rng.range(-0.2, 0.2),
        });
        // ring feeders mostly in the mud, troughs and tanks in the pens; no more of each
        // than the scene has meshes for (`FEEDS`)
        let mut feeders = Vec::new();
        for m in &mud {
            for _ in 0..rng.index(3) {
                let (a, d) = (rng.range(0.0, TAU), rng.range(0.0, 0.6) * m.r);
                let feed = pick(rng, &[(0.8, Feed::Ring), (0.2, Feed::Trough)]);
                feeders.push((
                    [m.c[0] + d * a.cos(), m.c[1] + d * a.sin()],
                    rng.range(0.0, TAU),
                    feed,
                ));
            }
        }
        for p in &pens {
            if rng.chance(0.4) {
                let feed = pick(
                    rng,
                    &[(0.4, Feed::Ring), (0.45, Feed::Trough), (0.15, Feed::Tank)],
                );
                feeders.push((p.rect.spot(rng, 1.5), rng.range(0.0, TAU), feed));
            }
        }
        for (feed, max) in FEEDS {
            let mut k = 0;
            feeders.retain(|f| {
                k += (f.2 == feed) as usize;
                f.2 != feed || k <= max
            });
        }
        // wrapped bales in a row along the clamp or a shed's back
        let mut stacks = Vec::new();
        if rng.chance(0.6) {
            let (start, along): ([f64; 2], [f64; 2]) = match &clamp {
                Some(c) if rng.chance(0.6) => (
                    [c.c[0] + side * (c.half[0] + 2.0), c.c[1] - c.half[1]],
                    [0.0, 1.0],
                ),
                _ => ([-row / 2.0, -depth * 1.2 - 2.0], [1.0, 0.0]),
            };
            let turn = along[1].atan2(along[0]);
            for k in 0..3 + rng.index(14) {
                let s = k as f64 * rng.range(1.4, 1.7);
                stacks.push(([start[0] + along[0] * s, start[1] + along[1] * s], turn));
            }
        }
        let colours = Colours {
            straw: vary(rng, albedo(190, 165, 110), 0.12),
            straw_wet: vary(rng, albedo(120, 98, 65), 0.15),
            mud: vary(rng, albedo(85, 70, 52), 0.15),
            mud_wet: vary(rng, albedo(55, 48, 40), 0.15),
            concrete: vary(rng, albedo(150, 148, 140), 0.12),
            dung: vary(rng, albedo(95, 82, 60), 0.15),
            apron: vary(rng, albedo(140, 135, 122), 0.15),
            plastic: if rng.chance(0.8) {
                vary(rng, albedo(34, 35, 37), 0.15)
            } else {
                vary(rng, albedo(60, 85, 62), 0.15) // green net
            },
            heap: vary(rng, albedo(88, 70, 50), 0.15),
            silage: vary(rng, albedo(105, 98, 58), 0.15),
        };
        let mut yard = Self {
            origin,
            axis,
            sheds,
            pens,
            apron,
            alley,
            mud,
            clamp,
            heap,
            feeders,
            stacks,
            colours,
            reach: 0.0,
        };
        yard.reach = yard.extent();
        yard
    }

    fn extent(&self) -> f64 {
        let corner = |r: &Rect| (r.c[0].abs() + r.half[0]).hypot(r.c[1].abs() + r.half[1]);
        let rects = self
            .sheds
            .iter()
            .map(|s| &s.rect)
            .chain(self.pens.iter().map(|p| &p.rect))
            .chain(self.apron.iter().map(|a| &a.0))
            .chain(&self.alley)
            .chain(&self.clamp);
        let blobs = self.mud.iter().chain(&self.heap);
        rects
            .map(corner)
            .chain(blobs.map(|m| m.c[0].hypot(m.c[1]) + 1.5 * m.r * m.stretch))
            .chain(self.stacks.iter().map(|(p, _)| p[0].hypot(p[1]) + 2.0))
            .fold(0.0, f64::max)
            + 2.0
    }

    /// Yard coordinates (u, v) of world (x, z).
    pub fn local(&self, x: f64, z: f64) -> (f64, f64) {
        let (dx, dz) = (x - self.origin[0], z - self.origin[1]);
        let a = self.axis;
        (dx * a[0] + dz * a[1], -dx * a[1] + dz * a[0])
    }

    /// World (x, z) of yard coordinates.
    pub fn world(&self, p: [f64; 2]) -> (f64, f64) {
        let a = self.axis;
        (
            self.origin[0] + p[0] * a[0] - p[1] * a[1],
            self.origin[1] + p[0] * a[1] + p[1] * a[0],
        )
    }

    /// Turn about y (Bevy) that puts a body's local x along yard direction `turn`.
    pub fn yaw(&self, turn: f64) -> f64 {
        let (s, c) = turn.sin_cos();
        let d = [
            c * self.axis[0] - s * self.axis[1],
            c * self.axis[1] + s * self.axis[0],
        ];
        (-d[1]).atan2(d[0])
    }

    /// Is (x, z) in the yard's buildings, pens, clamp or heap (so nothing else goes there)?
    pub fn blocks(&self, x: f64, z: f64) -> bool {
        let (u, v) = self.local(x, z);
        if u.hypot(v) > self.reach {
            return false;
        }
        self.sheds.iter().any(|s| s.rect.contains(u, v, 1.5))
            || self.pens.iter().any(|p| p.rect.contains(u, v, 1.0))
            || self.alley.is_some_and(|a| a.contains(u, v, 0.5))
            || self.clamp.is_some_and(|c| c.contains(u, v, 1.0))
            || self.heap.is_some_and(|h| h.distance(u, v) < h.r + 1.0)
            || self
                .stacks
                .iter()
                .any(|(p, _)| (p[0] - u).hypot(p[1] - v) < 1.5)
    }

    /// Is (x, z) anywhere in the yard (hard standing included)?
    pub fn covers(&self, x: f64, z: f64) -> bool {
        let (u, v) = self.local(x, z);
        self.blocks(x, z) || self.apron.is_some_and(|(a, _)| a.contains(u, v, 1.0))
    }

    /// Could a body at (x, z) be one of the yard's: in it, or on its mud?
    pub fn holds(&self, x: f64, z: f64) -> bool {
        let (u, v) = self.local(x, z);
        self.covers(x, z) || self.mud.iter().any(|m| m.distance(u, v) < m.r)
    }

    /// Where cows stand in the yard, with a weight each: pens, the front of open sheds
    /// (half under the roof, as on flight 061; deeper in they can't be seen) and the mud.
    pub fn herd_areas(&self) -> Vec<(f64, Area)> {
        let mut areas: Vec<(f64, Area)> = Vec::new();
        for p in &self.pens {
            areas.push((0.55 / self.pens.len() as f64, Area::Rect(p.rect, 0.8)));
        }
        let open: Vec<&Shed> = self.sheds.iter().filter(|s| s.open).collect();
        for s in &open {
            let front = s.rect.c[1] + s.rect.half[1];
            let strip = Rect::new(
                s.rect.c[0] - s.rect.half[0],
                s.rect.c[0] + s.rect.half[0],
                front - 4.0,
                front,
            );
            areas.push((0.15 / open.len() as f64, Area::Rect(strip, 0.8)));
        }
        for m in &self.mud {
            areas.push((0.3 / self.mud.len() as f64, Area::Mud(*m)));
        }
        areas
    }

    /// A random spot (world x, z) in `area`.
    pub fn spot_in(&self, rng: &mut Rng, area: &Area) -> (f64, f64) {
        let p = match area {
            Area::Rect(r, inset) => r.spot(rng, *inset),
            Area::Mud(m) => {
                let (a, d) = (rng.range(0.0, TAU), rng.unit().sqrt() * 0.7 * m.r);
                [m.c[0] + d * a.cos(), m.c[1] + d * a.sin()]
            }
        };
        self.world(p)
    }

    /// Up to `n` slots for a packed herd in `area`, in the order to fill them. Real yard
    /// cows stand close and line up (train flights 060, 062, Derval 012/014: nearest
    /// neighbour 0.84 body lengths, median group of 6): heads in round a ring feeder,
    /// rows facing a pen's rail or feed barrier, a looser bunch on the mud. Fewer slots
    /// than `n` when the area is full.
    pub fn herd_slots(&self, rng: &mut Rng, area: &Area, n: usize) -> Vec<Slot> {
        let rings: Vec<[f64; 2]> = self
            .feeders
            .iter()
            .filter(|f| f.2 == Feed::Ring && self.area_has(area, f.0))
            .map(|f| f.0)
            .collect();
        if !rings.is_empty() && rng.chance(0.7) {
            let c = rings[rng.index(rings.len())];
            let mut slots = self.rosette(rng, c, n);
            if slots.len() < n {
                // the rest bunch up behind them
                let more = self.rows(rng, &Area::Rect(Rect { c, half: [6.0, 6.0] }, 0.0), n);
                slots.extend(more.into_iter().filter(|s| {
                    (s.p[0] - c[0]).hypot(s.p[1] - c[1]) > RING_R + 2.6
                        && self.area_has(area, s.p)
                }));
                slots.truncate(n);
            }
            return slots;
        }
        self.rows(rng, area, n)
    }

    fn area_has(&self, area: &Area, p: [f64; 2]) -> bool {
        match area {
            Area::Rect(r, _) => r.contains(p[0], p[1], 0.0),
            Area::Mud(m) => m.distance(p[0], p[1]) < m.r,
        }
    }

    /// Standing cows round a ring feeder, heads in, from one side round: 7-10 places.
    fn rosette(&self, rng: &mut Rng, c: [f64; 2], n: usize) -> Vec<Slot> {
        let places = 7 + rng.index(4);
        let start = rng.range(0.0, TAU);
        // the body's centre: nose just inside the hoop
        let r = RING_R + rng.range(1.0, 1.25);
        (0..places.min(n))
            .map(|k| {
                let a = start + TAU * k as f64 / places as f64 + rng.range(-0.08, 0.08);
                let r = r + rng.range(-0.15, 0.15);
                Slot {
                    p: [c[0] + r * a.cos(), c[1] + r * a.sin()],
                    facing: a + PI + rng.range(-0.2, 0.2),
                    stand: true,
                }
            })
            .collect()
    }

    /// Rows of cows facing one way, filled from the edge they face (a rail, a barrier):
    /// in a pen or shed front side by side 1.0-1.6 m apart, on mud looser and less lined
    /// up. Clear of feeders and bay rails.
    fn rows(&self, rng: &mut Rng, area: &Area, n: usize) -> Vec<Slot> {
        let (rect, loose) = match area {
            Area::Rect(r, _) => (*r, false),
            Area::Mud(m) => {
                let h = 0.6 * m.r;
                (
                    Rect {
                        c: m.c,
                        half: [h, h],
                    },
                    true,
                )
            }
        };
        // the way the heads face: a yard axis, mostly towards the sheds or away
        let facing = pick(rng, &[(0.7, PI / 2.0), (0.3, 0.0)])
            + if rng.chance(0.5) { PI } else { 0.0 };
        let (sf, cf) = facing.sin_cos();
        // along = how far forward (towards the edge faced), side = across
        let along_u = cf.abs() > 0.5;
        let (ha, hs) = if along_u {
            (rect.half[0], rect.half[1])
        } else {
            (rect.half[1], rect.half[0])
        };
        let (a_max, s_max) = (ha - 1.3, hs - 0.6);
        if a_max < 0.0 || s_max < 0.0 {
            return Vec::new();
        }
        let (side_gap, row_gap, spin) = if loose {
            (rng.range(1.3, 2.0), rng.range(2.6, 3.6), 0.6)
        } else {
            (rng.range(1.0, 1.6), rng.range(2.5, 3.3), 0.25)
        };
        let fwd = if along_u { cf.signum() } else { sf.signum() };
        let shift = rng.range(0.0, side_gap);
        let mut slots = Vec::new();
        let mut a = a_max;
        while a >= -a_max && slots.len() < n {
            let mut s = -s_max + shift;
            // rows nearer the front fill first; within a row, from a random side
            let mut row = Vec::new();
            while s <= s_max {
                if !rng.chance(0.12) {
                    let (aa, ss) = (
                        a + rng.range(-0.3, 0.3),
                        s + rng.range(-0.12, 0.12) * side_gap,
                    );
                    let (du, dv) = if along_u {
                        (fwd * aa, ss)
                    } else {
                        (ss, fwd * aa)
                    };
                    let p = [rect.c[0] + du, rect.c[1] + dv];
                    let flip = rng.chance(if loose { 0.3 } else { 0.1 });
                    row.push(Slot {
                        p,
                        facing: facing + rng.range(-spin, spin) + if flip { PI } else { 0.0 },
                        stand: false,
                    });
                }
                s += side_gap;
            }
            if rng.chance(0.5) {
                row.reverse();
            }
            slots.extend(row.into_iter().filter(|sl| self.clear(sl, along_u)));
            a -= row_gap;
        }
        slots.truncate(n);
        slots
    }

    /// Is a body at yard-frame `p` clear of the feeders?
    pub fn off_feeders(&self, p: [f64; 2]) -> bool {
        self.feeders.iter().all(|f| {
            let d = (f.0[0] - p[0]).hypot(f.0[1] - p[1]);
            d > match f.2 {
                Feed::Ring => RING_R + 1.2,
                Feed::Trough => 2.2,
                Feed::Tank => 1.8,
            }
        })
    }

    /// Is a slot off the feeders and the pens' bay rails?
    fn clear(&self, sl: &Slot, along_u: bool) -> bool {
        let [u, v] = sl.p;
        let feeders_ok = self.off_feeders(sl.p);
        let half_u = if along_u { 1.3 } else { 0.5 };
        let bays_ok = self.pens.iter().all(|p| {
            !p.rect.contains(u, v, 0.0) || p.bays.iter().all(|b| (b - u).abs() > half_u)
        });
        feeders_ok && bays_ok
    }

    /// Rail panels along the pens' edges and bay dividers, with a gap for the gate:
    /// (yard-frame start, end) of each straight run.
    pub fn rail_runs(&self) -> Vec<([f64; 2], [f64; 2])> {
        let mut runs = Vec::new();
        for p in &self.pens {
            let (u0, u1) = (p.rect.c[0] - p.rect.half[0], p.rect.c[0] + p.rect.half[0]);
            let (v0, v1) = (p.rect.c[1] - p.rect.half[1], p.rect.c[1] + p.rect.half[1]);
            let (g, gw) = p.gate;
            if g - gw > u0 {
                runs.push(([u0, v1], [g - gw, v1]));
            }
            if g + gw < u1 {
                runs.push(([g + gw, v1], [u1, v1]));
            }
            runs.push(([u0, v0], [u0, v1]));
            runs.push(([u1, v0], [u1, v1]));
            for &b in &p.bays {
                runs.push(([b, v0], [b, v1]));
            }
        }
        runs
    }

    /// The yard's ground over colour `c` at world (x, z).
    #[allow(clippy::too_many_arguments)]
    pub fn paint(
        &self,
        table: &NoiseTable,
        mut c: Rgb,
        x: f64,
        z: f64,
        ts: f64,
        grain: f64,
    ) -> Rgb {
        let (u, v) = self.local(x, z);
        if u.hypot(v) > self.reach {
            return c;
        }
        let k = &self.colours;
        let n = |s: f64, o: f64| table.at(x / s + o, z / s - o);
        // dirt and dung on hard standing: broad soft patches (the first try's blotches, 2 m
        // across and up to 70% dung, looked like camouflage)
        let stains =
            (0.3 * (n(8.0, 1.3) + 0.3) + 0.25 * (n(1.5, 2.6) - 0.2)).clamp(0.0, 0.35) as f32;
        if let Some((a, surface)) = &self.apron {
            let cov = a.cover(u, v, ts.max(0.3));
            if cov > 0.0 {
                let base = match surface {
                    Surface::Concrete => k.concrete,
                    Surface::Gravel => mix(k.apron, albedo(170, 165, 150), 0.5),
                    Surface::Dirt => mix(k.mud, k.apron, 0.5),
                };
                let g = mix(base, k.dung, stains);
                c = mix(c, scale(g, 1.0 + 0.08 * grain + 0.04 * n(0.4, 2.2)), cov);
            }
        }
        for m in &self.mud {
            let rim = m.r * (1.0 + 0.35 * n(6.0, 3.1 + m.c[0]));
            let d = m.distance(u, v);
            if d < rim + 1.0 {
                // solid inside; over its last 1.5 m it breaks up into the grass
                let edge = ((rim - d) / 1.5).clamp(0.0, 1.0);
                let broken = (edge + 0.8 * n(0.8, 4.4)).clamp(0.0, 1.0);
                let cov = (edge * broken) as f32;
                let wet = ((n(4.0, 5.5) - 0.15) * 3.0).clamp(0.0, 0.8) as f32;
                let g = mix(k.mud, k.mud_wet, wet);
                c = mix(c, scale(g, 1.0 + 0.15 * grain + 0.06 * n(0.3, 6.6)), cov);
            }
        }
        for p in &self.pens {
            let cov = p.rect.cover(u, v, ts);
            if cov > 0.0 {
                c = mix(c, self.floor(table, p.floor, x, z, grain), cov);
            }
        }
        for s in self.sheds.iter().filter(|s| s.open) {
            let cov = s.rect.cover(u, v, ts);
            if cov > 0.0 {
                c = mix(c, self.floor(table, Floor::Straw, x, z, grain), cov);
            }
        }
        if let Some(a) = &self.alley {
            let cov = a.cover(u, v, ts);
            if cov > 0.0 {
                let g = mix(k.concrete, k.dung, stains);
                let mut g = scale(g, 1.0 + 0.08 * grain);
                // silage pushed up against the barrier, ragged
                let from = v - (a.c[1] - a.half[1]);
                let reach = 0.5 + 0.4 * n(1.0, 7.7);
                g = mix(
                    g,
                    scale(k.silage, 1.0 + 0.25 * grain),
                    band(from, reach, ts),
                );
                c = mix(c, g, cov);
            }
        }
        if let Some(cl) = &self.clamp {
            let cov = cl.cover(u, v, ts);
            if cov > 0.0 {
                let sheen = 1.0 + 0.25 * n(1.5, 8.8) + 0.05 * grain;
                let mut g = scale(k.plastic, sheen);
                // tyres on the sheet, about one per 1.2 m, jittered
                let cell = 1.2;
                let (i, j) = ((u / cell).floor(), (v / cell).floor());
                let jit = |s: f64| crate::rng::lattice(0x7E5 ^ s.to_bits(), i as i64, j as i64);
                let (cu, cv) = (
                    (i + 0.3 + 0.4 * jit(1.0)) * cell,
                    (j + 0.3 + 0.4 * jit(2.0)) * cell,
                );
                let d = (u - cu).hypot(v - cv);
                if jit(3.0) < 0.75 {
                    let ring = band((d - 0.32).abs(), 0.09, ts);
                    g = mix(g, albedo(20, 20, 21), 0.9 * ring);
                }
                c = mix(c, g, cov);
            }
        }
        if let Some(h) = &self.heap {
            let rim = h.r * (1.0 + 0.3 * n(2.0, 9.9));
            let d = h.distance(u, v);
            if d < rim + 0.5 {
                let cov = ((rim - d) / 0.6).clamp(0.0, 1.0) as f32;
                let lumpy = 1.0 + 0.35 * n(0.5, 1.9) + 0.2 * grain;
                let g = mix(k.heap, k.straw, (n(0.7, 2.8) * 1.5).clamp(0.0, 0.4) as f32);
                c = mix(c, scale(g, lumpy), cov);
            }
        }
        c
    }

    fn floor(&self, table: &NoiseTable, floor: Floor, x: f64, z: f64, grain: f64) -> Rgb {
        let k = &self.colours;
        let n = |s: f64, o: f64| table.at(x / s + o, z / s - o);
        match floor {
            Floor::Straw => {
                let wet = ((n(3.0, 0.7) + 0.2) * 1.5).clamp(0.0, 0.75) as f32;
                let g = mix(k.straw, k.straw_wet, wet);
                scale(g, 1.0 + 0.25 * grain + 0.15 * n(0.2, 1.1))
            }
            Floor::Mud => {
                let wet = ((n(4.0, 2.4) - 0.1) * 3.0).clamp(0.0, 0.9) as f32;
                let g = mix(k.mud, k.mud_wet, wet);
                // a little straw trodden in
                let g = mix(
                    g,
                    k.straw,
                    ((n(1.0, 3.3) - 0.2) * 2.0).clamp(0.0, 0.35) as f32,
                );
                scale(g, 1.0 + 0.3 * grain)
            }
            Floor::Concrete => {
                let smear = (n(2.0, 4.1) * 2.0 + 0.2).clamp(0.0, 0.7) as f32;
                scale(mix(k.concrete, k.dung, smear), 1.0 + 0.08 * grain)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packed_herds_face_the_feeder_or_line_up_without_overlapping() {
        let mut rng = Rng::new(9);
        let (mut rosettes, mut rows) = (0, 0);
        for _ in 0..200 {
            let a = rng.range(0.0, TAU);
            let y = Yard::random(&mut rng, [0.0, 0.0], [a.cos(), a.sin()]);
            for (feed, max) in FEEDS {
                assert!(y.feeders.iter().filter(|f| f.2 == feed).count() <= max);
            }
            for (_, area) in y.herd_areas() {
                let slots = y.herd_slots(&mut rng, &area, 12);
                assert!(slots.len() <= 12);
                for (i, s) in slots.iter().enumerate() {
                    assert!(y.area_has(&area, s.p) || s.stand, "slot off its area");
                    for t in &slots[..i] {
                        // side by side at least a body's width apart
                        let d = (s.p[0] - t.p[0]).hypot(s.p[1] - t.p[1]);
                        assert!(d > 0.75, "slots {d:.2} m apart");
                    }
                }
                if slots.first().is_some_and(|s| s.stand) {
                    rosettes += 1;
                    // the ring they all stand round (two rings can be close)
                    let ring = y
                        .feeders
                        .iter()
                        .filter(|f| f.2 == Feed::Ring)
                        .map(|f| f.0)
                        .find(|c| {
                            slots.iter().filter(|s| s.stand).all(|s| {
                                let d = (c[0] - s.p[0]).hypot(c[1] - s.p[1]);
                                (RING_R + 0.8..RING_R + 1.45).contains(&d)
                            })
                        })
                        .expect("a ring the cows stand round");
                    for s in slots.iter().filter(|s| s.stand) {
                        let to = (ring[1] - s.p[1]).atan2(ring[0] - s.p[0]);
                        let off = (s.facing - to + PI).rem_euclid(TAU) - PI;
                        assert!(off.abs() < 0.25, "head {off:.2} rad off the feeder");
                    }
                } else if slots.len() > 3 {
                    rows += 1;
                }
            }
        }
        assert!(rosettes > 10 && rows > 50, "{rosettes} rosettes, {rows} rows");
    }

    #[test]
    fn yards_hold_cows_rails_and_stay_within_reach() {
        let mut rng = Rng::new(4);
        let (mut spots, mut runs) = (0, 0);
        for _ in 0..50 {
            let a = rng.range(0.0, TAU);
            let y = Yard::random(&mut rng, [100.0, -40.0], [a.cos(), a.sin()]);
            assert!(!y.sheds.is_empty());
            for s in &y.sheds {
                // sheds behind the line v = 0, pens in front
                assert!(s.rect.c[1] + s.rect.half[1] <= 1e-9);
            }
            for p in &y.pens {
                assert!(p.rect.c[1] - p.rect.half[1] >= -1e-9);
            }
            runs += y.rail_runs().len();
            let areas = y.herd_areas();
            let total: f64 = areas.iter().map(|a| a.0).sum();
            assert!(areas.is_empty() || total <= 1.0 + 1e-9);
            for (_, area) in &areas {
                for _ in 0..5 {
                    let (x, z) = y.spot_in(&mut rng, area);
                    spots += 1;
                    let (u, v) = y.local(x, z);
                    assert!(u.hypot(v) < y.reach);
                    assert!(y.holds(x, z));
                    // not inside a closed shed
                    assert!(
                        !y.sheds
                            .iter()
                            .any(|s| !s.open && s.rect.contains(u, v, -0.1)),
                        "cow in a closed shed"
                    );
                }
            }
            // local and world agree
            let (x, z) = y.world([3.0, -7.0]);
            let (u, v) = y.local(x, z);
            assert!((u - 3.0).abs() < 1e-9 && (v + 7.0).abs() < 1e-9);
        }
        assert!(spots > 500, "{spots} cow spots");
        assert!(runs > 100, "{runs} rail runs");
    }

    #[test]
    fn yard_ground_paints_mud_straw_and_tyres() {
        let mut rng = Rng::new(9);
        let table = NoiseTable::new(3);
        let grass = albedo(90, 120, 60);
        let mut changed = 0;
        for _ in 0..10 {
            let y = Yard::random(&mut rng, [0.0, 0.0], [1.0, 0.0]);
            for p in &y.pens {
                let (x, z) = y.world(p.rect.c);
                let c = y.paint(&table, grass, x, z, 0.05, 0.0);
                // a pen floor is never the grass
                assert!(c != grass);
                changed += 1;
            }
            // far away: untouched
            assert_eq!(y.paint(&table, grass, 5000.0, 5000.0, 0.05, 0.0), grass);
        }
        assert!(changed > 5);
    }
}
