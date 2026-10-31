//! RGB camera: daylight and the colours of the world for one shot, the colour ground
//! texture, and the camera model that turns the render into a drone photo. No Bevy here.
//!
//! The real RGB data (ICAERUS cows) is DJI stills looking straight down on pasture:
//! green and dry grass, harvested fields with round bales, tree canopies with hard
//! shadows; cows white (55%), black (20%) or brown. The colours below are albedos (sRGB
//! bytes as you'd pick them from a photo, linearised); measured on train flights only
//! (median ground, cow centres), never on val.
//!
//! The renderer lights the scene with a sun (shadows on) and a sky; the camera model then
//! does what a drone camera does: white balance, auto-exposure, vignetting, a little blur
//! and noise, contrast and saturation, and JPEG when the frame is saved.

pub use crate::fields::Field;
use crate::fields::Layout;
use crate::rng::Rng;
use crate::scene::Kind;
use crate::sensor::{blur, percentile};
use crate::thermal::{TEXTURE_BASE, srgb_to_linear};

/// Linear RGB.
pub type Rgb = [f32; 3];

/// Linear albedo of an sRGB colour picked from a photo.
pub fn albedo(r: u8, g: u8, b: u8) -> Rgb {
    [r, g, b].map(|v| srgb_to_linear(v as f64 / 255.0) as f32)
}

/// `c` a shade lighter or darker (up to +-`spread`, log), its hue nudged by a third of
/// that per channel.
pub(crate) fn vary(rng: &mut Rng, c: Rgb, spread: f64) -> Rgb {
    let all = rng.range(-spread, spread).exp() as f32;
    let hue = [(); 3].map(|_| rng.range(-spread / 3.0, spread / 3.0).exp() as f32);
    [0, 1, 2].map(|k| (c[k] * all * hue[k]).min(0.95))
}

pub(crate) fn mix(a: Rgb, b: Rgb, t: f32) -> Rgb {
    [0, 1, 2].map(|k| a[k] + (b[k] - a[k]) * t)
}

/// Sun and sky for one shot.
#[derive(Clone, Debug)]
pub struct Daylight {
    /// Unit vector towards the sun, Bevy coordinates (y up).
    pub sun_dir: [f64; 3],
    /// Direct sun as a share of full sunlight (overcast: 0.05-0.3), and the sky's light
    /// as a share of what a white surface gets from full sun.
    pub sun: f32,
    pub sky: f32,
    pub sun_tint: Rgb,
    pub sky_tint: Rgb,
}

impl Daylight {
    pub fn random(rng: &mut Rng) -> Self {
        // DJI flights are flown in daylight, mostly with some sun: low sun gives long
        // shadows (ICAERUS flights at 08:38 and 18:33), high sun short ones.
        let elevation = rng.range(15.0_f64, 70.0).to_radians();
        let azimuth = rng.range(0.0, std::f64::consts::TAU);
        let overcast = rng.chance(0.25);
        let (sun, sky) = if overcast {
            (rng.range(0.05, 0.3), rng.range(0.5, 0.9))
        } else {
            (1.0, rng.range(0.18, 0.35))
        };
        // low sun is warmer
        let warm = (1.0 - elevation / 70f64.to_radians()) as f32 * rng.range(0.0, 0.15) as f32;
        Self {
            sun_dir: [
                elevation.cos() * azimuth.cos(),
                elevation.sin(),
                elevation.cos() * azimuth.sin(),
            ],
            sun: sun as f32,
            sky: sky as f32,
            sun_tint: [1.0, 1.0 - 0.4 * warm, 1.0 - warm],
            sky_tint: [0.85, 0.93, 1.0],
        }
    }
}

/// The ground of one shot: the farm layout under the camera (`fields.rs`, painted into
/// the ground image), and the colours of its central field for the terrain beyond it.
#[derive(Clone, Debug)]
pub struct Ground {
    /// The field the camera looks at (its centre).
    pub field: Field,
    /// Main cover, patches of a second cover, and bare soil (tracks, gateways, molehills).
    pub main: Rgb,
    pub second: Rgb,
    pub soil: Rgb,
    /// Mowing or harvest stripes: how much they change the brightness (0 = none) and how
    /// many fit in one texture tile (so the texture still tiles).
    pub stripes: f64,
    pub stripes_per_tile: usize,
    /// Fine texture strength, the share of dark clumps and light tufts, grain (blades and
    /// stalks: every texel its own brightness, so a frame is busy at the pixel scale like
    /// a real photo of grass), and the seed.
    pub texture: f64,
    pub tufts: f64,
    pub grain: f64,
    pub seed: u64,
    pub layout: Layout,
}

impl Ground {
    /// The ground for a camera footprint: a square `size` metres across around `centre`.
    pub fn random(rng: &mut Rng, centre: [f64; 2], size: f64) -> Self {
        let layout = Layout::random(rng, centre, size);
        let p = layout.parcel_at(centre[0], centre[1]).clone();
        Self {
            field: p.field,
            main: p.main,
            second: p.second,
            soil: p.soil,
            stripes: p.mow,
            stripes_per_tile: 4 + rng.index(9),
            texture: p.texture,
            tufts: rng.range(0.2, 1.0),
            grain: p.grain,
            seed: rng.next_u64(),
            layout,
        }
    }

    /// Albedo of the ground at a terrain vertex: second cover in patches tens of metres
    /// across, soil in a few smaller ones.
    pub fn at(&self, x: f64, z: f64) -> Rgb {
        use crate::world::fbm;
        let patch = fbm(self.seed, x / 30.0, z / 30.0, 3);
        let soil = fbm(self.seed ^ 0x501, x / 8.0, z / 8.0, 3);
        let c = mix(
            self.main,
            self.second,
            ((patch - 0.1) * 2.0).clamp(0.0, 1.0) as f32,
        );
        mix(c, self.soil, ((soil - 0.45) * 4.0).clamp(0.0, 0.9) as f32)
    }
}

/// Colour detail of the ground as sRGB RGBA bytes (`size` x `size`, tiling) around
/// `TEXTURE_BASE`, which the material's `texture_gain` undoes: grain, grass clumps darker
/// and greener (few on stubble), dry tufts lighter and yellower, dark specks (dung,
/// holes), and stripes along one texture axis that wander a little.
pub fn ground_texture(g: &Ground, size: usize) -> Vec<u8> {
    use std::f64::consts::TAU;
    let mut val: Vec<[f64; 3]> = Vec::with_capacity(size * size);
    for j in 0..size {
        for i in 0..size {
            let (u, v) = (i as f64 / size as f64, j as f64 / size as f64);
            let n = crate::thermal::tiling_fbm(g.seed, u, v, 48.0);
            let h = crate::thermal::tiling_fbm(g.seed ^ 0x4E, u, v, 12.0);
            let grain = 2.0 * crate::rng::lattice(g.seed ^ 0x6A, i as i64, j as i64) - 1.0;
            let b = (1.0 - g.texture * (0.5 + 0.5 * n)) * (1.0 + g.grain * grain);
            // hue: towards green where h < 0, towards straw where h > 0
            let t = 0.15 * h;
            let stripe = if g.stripes > 0.0 {
                let wander = 0.03 * (v * 2.0 * TAU).sin();
                let s = ((u * g.stripes_per_tile as f64 + wander) * TAU).sin();
                1.0 - g.stripes * (0.5 + 0.5 * s.signum() * s.abs().sqrt())
            } else {
                1.0
            };
            val.push([
                b * (1.0 + t) * stripe,
                b * stripe,
                b * (1.0 + 1.5 * t) * stripe,
            ]);
        }
    }
    let lush = match g.field {
        Field::Pasture => 0.5,
        Field::Dry | Field::Crop => 0.3,
        Field::Stubble | Field::Bare => 0.08,
    };
    let mut rng = Rng::new(g.seed ^ 0x7AF7);
    for _ in 0..size * size / 400 {
        let (cx, cy) = (rng.range(0.0, size as f64), rng.range(0.0, size as f64));
        let u = rng.unit();
        let (r, amp): (f64, [f64; 3]) = if u < lush {
            // clump of lush grass: darker, greener
            let a = g.tufts * rng.range(0.15, 0.35);
            (rng.range(2.0, 6.0), [-a * 1.2, -a * 0.7, -a * 1.3])
        } else if u < 0.87 {
            // dry tuft or seed heads: lighter, yellower
            let a = g.tufts * rng.range(0.1, 0.3);
            (rng.range(1.5, 4.0), [a, a * 0.9, a * 0.4])
        } else {
            // dung, a hole, a dark weed
            let a = rng.range(0.2, 0.5);
            (rng.range(1.0, 2.5), [-a; 3])
        };
        let reach = r.ceil() as i64 + 1;
        for dy in -reach..=reach {
            for dx in -reach..=reach {
                let w = (-((dx * dx + dy * dy) as f64) / (r * r)).exp();
                let (x, y) = (
                    (cx as i64 + dx).rem_euclid(size as i64) as usize,
                    (cy as i64 + dy).rem_euclid(size as i64) as usize,
                );
                for k in 0..3 {
                    val[y * size + x][k] *= 1.0 + amp[k] * w;
                }
            }
        }
    }
    val.iter()
        .flat_map(|c| {
            let b = |v: f64| (TEXTURE_BASE * v).round().clamp(0.0, 255.0) as u8;
            [b(c[0]), b(c[1]), b(c[2]), 255]
        })
        .collect()
}

/// Colours of one body: torso, limbs, head (as `scene.rs` splits them), and whether the
/// torso carries black patches (Holstein cattle, piebald horses and dogs).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Coat {
    pub parts: [Rgb; 3],
    pub patches: bool,
}

/// Pick from `(weight, value)` pairs.
pub(crate) fn pick<T: Copy>(rng: &mut Rng, items: &[(f64, T)]) -> T {
    let total: f64 = items.iter().map(|i| i.0).sum();
    let mut u = rng.range(0.0, total);
    for &(w, v) in items {
        if u < w {
            return v;
        }
        u -= w;
    }
    items[items.len() - 1].1
}

impl Coat {
    pub fn random(rng: &mut Rng, kind: Kind) -> Self {
        let a = |r, g, b| albedo(r, g, b);
        let white = a(235, 232, 225);
        let black = a(40, 38, 36);
        // one colour, shaded a little per part
        let same = |rng: &mut Rng, c: Rgb, spread: f64| -> [Rgb; 3] {
            let c = vary(rng, c, spread);
            [c, vary(rng, c, 0.1), vary(rng, c, 0.1)]
        };
        let (parts, patches) = match kind {
            Kind::Cattle => {
                // train flights: 55% light, 20% dark, 25% mid (brown)
                let (c, patches) = pick(
                    rng,
                    &[
                        (0.3, (white, false)),
                        (0.25, (white, true)),
                        (0.2, (black, false)),
                        (0.15, (a(140, 90, 55), false)),
                        (0.1, (a(215, 200, 170), false)),
                    ],
                );
                (same(rng, c, 0.12), patches)
            }
            Kind::Horse => {
                let (body, legs) = pick(
                    rng,
                    &[
                        (0.35, (a(110, 65, 40), black)),         // bay
                        (0.2, (a(150, 80, 45), a(150, 80, 45))), // chestnut
                        (0.15, (black, black)),
                        (0.2, (a(200, 200, 195), a(170, 170, 165))), // grey
                        (0.1, (a(210, 180, 120), a(210, 180, 120))), // palomino
                    ],
                );
                let body = vary(rng, body, 0.12);
                ([body, vary(rng, legs, 0.1), body], rng.chance(0.08))
            }
            Kind::Sheep => {
                // wool, from clean to muddy; some dark breeds; goats white, brown or black
                let wool = pick(
                    rng,
                    &[
                        (0.6, a(215, 205, 185)),
                        (0.2, a(170, 160, 140)),
                        (0.1, a(60, 55, 50)),
                        (0.1, a(120, 85, 55)),
                    ],
                );
                let face = if rng.chance(0.3) { black } else { wool };
                let wool = vary(rng, wool, 0.1);
                ([wool, vary(rng, face, 0.1), vary(rng, face, 0.1)], false)
            }
            Kind::Dog => {
                let c = pick(
                    rng,
                    &[
                        (0.3, black),
                        (0.25, a(120, 80, 50)),
                        (0.2, a(200, 160, 100)),
                        (0.15, white),
                        (0.1, a(130, 130, 130)),
                    ],
                );
                (same(rng, c, 0.1), rng.chance(0.2))
            }
            Kind::Person => {
                // clothes (the capsule is the whole body), arms in sleeves or bare, hair
                let clothes = pick(
                    rng,
                    &[
                        (0.2, a(40, 50, 80)),
                        (0.15, a(30, 30, 30)),
                        (0.12, a(120, 120, 120)),
                        (0.12, a(150, 135, 100)),
                        (0.1, a(60, 80, 50)),
                        (0.08, a(170, 40, 35)),
                        (0.08, a(220, 220, 215)),
                        (0.08, a(220, 230, 60)),
                        (0.07, a(230, 110, 30)),
                    ],
                );
                let clothes = vary(rng, clothes, 0.15);
                let skin = pick(
                    rng,
                    &[
                        (0.5, a(225, 185, 150)),
                        (0.3, a(170, 120, 90)),
                        (0.2, a(100, 70, 50)),
                    ],
                );
                let arms = if rng.chance(0.6) { clothes } else { skin };
                let hair = pick(
                    rng,
                    &[
                        (0.35, a(40, 30, 25)),
                        (0.25, a(100, 70, 45)),
                        (0.15, a(200, 170, 110)),
                        (0.1, a(150, 150, 150)),
                        (0.15, clothes), // a hat or hood
                    ],
                );
                ([clothes, arms, hair], false)
            }
            // wildlife isn't in the RGB pool; brown if it ever is
            _ => (same(rng, a(130, 100, 70), 0.2), false),
        };
        Self { parts, patches }
    }
}

/// Colour of an unlabelled thing (`scene.rs` clutter): canopies, stones, roofs and walls,
/// car paint and glass, bales.
pub fn canopy(rng: &mut Rng) -> Rgb {
    let c = pick(
        rng,
        &[
            (0.6, albedo(70, 95, 50)),
            (0.25, albedo(90, 110, 60)),
            (0.15, albedo(110, 110, 80)),
        ],
    );
    vary(rng, c, 0.25)
}

pub fn stone(rng: &mut Rng) -> Rgb {
    vary(rng, albedo(150, 145, 135), 0.3)
}

pub fn building(rng: &mut Rng) -> Rgb {
    let c = pick(
        rng,
        &[
            (0.3, albedo(150, 70, 55)),   // tiles
            (0.3, albedo(110, 110, 115)), // slate, sheet metal
            (0.2, albedo(200, 195, 185)), // render, light metal
            (0.2, albedo(90, 75, 60)),    // weathered wood
        ],
    );
    vary(rng, c, 0.2)
}

pub fn car(rng: &mut Rng) -> Rgb {
    let c = pick(
        rng,
        &[
            (0.25, albedo(220, 220, 220)),
            (0.2, albedo(30, 30, 32)),
            (0.2, albedo(140, 140, 145)),
            (0.1, albedo(150, 30, 30)),
            (0.1, albedo(40, 60, 120)),
            (0.15, albedo(90, 100, 70)),
        ],
    );
    vary(rng, c, 0.1)
}

pub fn glass(rng: &mut Rng) -> Rgb {
    vary(rng, albedo(45, 50, 60), 0.2)
}

/// Round and square bales: straw, lighter and smoother than the stubble around them; some
/// wrapped in white, green or black film (silage).
pub fn bale(rng: &mut Rng) -> Rgb {
    let c = pick(
        rng,
        &[
            (0.7, albedo(215, 195, 140)),
            (0.12, albedo(235, 235, 230)),
            (0.1, albedo(40, 40, 40)),
            (0.08, albedo(80, 110, 70)),
        ],
    );
    vary(rng, c, 0.12)
}

pub fn road(rng: &mut Rng) -> Rgb {
    let c = pick(
        rng,
        &[
            (0.5, albedo(150, 135, 110)),
            (0.3, albedo(110, 110, 105)),
            (0.2, albedo(170, 165, 150)),
        ],
    );
    vary(rng, c, 0.15)
}

/// Camera settings for one RGB frame.
#[derive(Clone, Debug)]
pub struct RgbParams {
    /// White balance error per channel (green = 1).
    pub wb: Rgb,
    /// Auto-exposure puts the frame's median brightness here (sRGB, 0-255).
    pub median: f32,
    /// Darkening at the corners (0 = none; real lenses up to about a third).
    pub vignette: f32,
    /// Where the lens's image circle ends, as a share of the half-diagonal (0 = it
    /// covers the frame): the corners beyond go dark, as in several real flights.
    pub circle: f32,
    pub blur_sigma: f32,
    /// Light haze: share of the picture replaced by a bluish grey.
    pub haze: f32,
    /// Contrast and saturation around neutral (1 = unchanged).
    pub contrast: f32,
    pub saturation: f32,
    /// Noise in output grey levels.
    pub noise: f32,
    pub jpeg_quality: u8,
}

impl RgbParams {
    pub fn random(rng: &mut Rng) -> Self {
        let wb = [
            rng.range(-0.06, 0.06).exp() as f32,
            1.0,
            rng.range(-0.08, 0.08).exp() as f32,
        ];
        Self {
            wb,
            // train flights: ground median 95-143
            median: rng.range(95.0, 145.0) as f32,
            vignette: if rng.chance(0.5) {
                rng.range(0.05, 0.35) as f32
            } else {
                0.0
            },
            circle: if rng.chance(0.3) {
                rng.range(0.85, 0.97) as f32
            } else {
                0.0
            },
            // 5 K stills scaled to 1280 are sharp
            blur_sigma: rng.range(0.15f64.ln(), 0.45f64.ln()).exp() as f32,
            haze: if rng.chance(0.1) {
                rng.range(0.02, 0.08) as f32
            } else {
                0.0
            },
            contrast: rng.range(1.0, 1.35) as f32,
            saturation: rng.range(0.75, 1.1) as f32,
            noise: rng.range(0.5f64.ln(), 3.0f64.ln()).exp() as f32,
            jpeg_quality: rng.range(75.0, 96.0) as u8,
        }
    }
}

pub(crate) fn linear_to_srgb(v: f32) -> f32 {
    if v <= 0.003_130_8 {
        v * 12.92
    } else {
        1.055 * v.powf(1.0 / 2.4) - 0.055
    }
}

/// Apply the camera to a `width` x `height` RGB render (sRGB bytes, 3 per pixel).
pub fn apply(raw: &[u8], width: usize, height: usize, p: &RgbParams, rng: &mut Rng) -> Vec<u8> {
    let n = width * height;
    assert_eq!(raw.len(), 3 * n);
    let haze = [0.55f32, 0.6, 0.68];
    let (cx, cy) = (width as f32 / 2.0, height as f32 / 2.0);
    let r2max = cx * cx + cy * cy;
    // per channel, linear: white balance, haze, vignette, optics blur
    let mut chans: Vec<Vec<f32>> = (0..3)
        .map(|k| {
            let ch: Vec<f32> = (0..n)
                .map(|i| {
                    let v = srgb_to_linear(raw[3 * i + k] as f64 / 255.0) as f32 * p.wb[k];
                    let v = v * (1.0 - p.haze) + haze[k] * p.haze;
                    let (x, y) = ((i % width) as f32 - cx, (i / width) as f32 - cy);
                    let r2 = (x * x + y * y) / r2max;
                    let cut = if p.circle > 0.0 {
                        let t = ((r2.sqrt() - p.circle) / (1.0 - p.circle)).clamp(0.0, 1.0);
                        1.0 - 0.95 * t * t * (3.0 - 2.0 * t)
                    } else {
                        1.0
                    };
                    v * (1.0 - p.vignette * r2) * cut
                })
                .collect();
            blur(&ch, width, height, p.blur_sigma)
        })
        .collect();
    // auto-exposure: the frame's median brightness to `p.median`
    let lum: Vec<f32> = (0..n)
        .step_by(7)
        .map(|i| 0.2126 * chans[0][i] + 0.7152 * chans[1][i] + 0.0722 * chans[2][i])
        .collect();
    let target = srgb_to_linear(p.median as f64 / 255.0) as f32;
    let gain = target / percentile(&lum, 50.0).max(1e-4);
    for ch in &mut chans {
        for v in ch.iter_mut() {
            *v = linear_to_srgb((*v * gain).clamp(0.0, 1.0));
        }
    }
    let mut out = Vec::with_capacity(3 * n);
    for ((&r, &g), &b) in chans[0].iter().zip(&chans[1]).zip(&chans[2]) {
        let c = [r, g, b];
        let grey = (c[0] + c[1] + c[2]) / 3.0;
        for v in c {
            let v = grey + (v - grey) * p.saturation;
            let v = 0.5 + (v - 0.5) * p.contrast;
            let v = v * 255.0 + p.noise * rng.normal() as f32;
            out.push(v.round().clamp(0.0, 255.0) as u8);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exposure_sets_the_median_and_keeps_size() {
        let (w, h) = (64, 48);
        let raw: Vec<u8> = (0..w * h).flat_map(|i| [(i % 200) as u8 + 20; 3]).collect();
        let mut rng = Rng::new(1);
        let p = RgbParams {
            wb: [1.0; 3],
            median: 120.0,
            vignette: 0.0,
            circle: 0.0,
            blur_sigma: 0.0,
            haze: 0.0,
            contrast: 1.0,
            saturation: 1.0,
            noise: 0.0,
            jpeg_quality: 90,
        };
        let out = apply(&raw, w, h, &p, &mut rng);
        assert_eq!(out.len(), raw.len());
        let g: Vec<f32> = out.iter().step_by(3).map(|&v| v as f32).collect();
        let med = percentile(&g, 50.0);
        assert!((med - 120.0).abs() < 4.0, "median {med}");
    }

    #[test]
    fn vignette_darkens_corners_only() {
        let (w, h) = (80, 60);
        let raw = vec![128u8; 3 * w * h];
        let mut rng = Rng::new(2);
        let mut p = RgbParams::random(&mut rng);
        (
            p.vignette,
            p.circle,
            p.haze,
            p.noise,
            p.wb,
            p.saturation,
            p.contrast,
        ) = (0.3, 0.0, 0.0, 0.0, [1.0; 3], 1.0, 1.0);
        let out = apply(&raw, w, h, &p, &mut rng);
        let at = |x: usize, y: usize| out[3 * (y * w + x)] as i32;
        assert!(at(0, 0) + 10 < at(w / 2, h / 2));
        assert_eq!(at(w / 2, h / 2), at(w / 2 + 1, h / 2));
    }

    #[test]
    fn ground_texture_tiles_and_varies() {
        let mut rng = Rng::new(3);
        let mut g = Ground::random(&mut rng, [0.0, 0.0], 200.0);
        g.stripes = 0.2;
        let size = 64;
        let t = ground_texture(&g, size);
        assert_eq!(t.len(), size * size * 4);
        let row = |y: usize| &t[y * size * 4..(y + 1) * size * 4];
        // stripes run along v: columns differ, and a wrap-around stays smooth
        let col = |x: usize| t[4 * x + 1] as i32;
        assert!((0..size).map(col).max() > (0..size).map(col).min());
        assert!((col(0) - col(size - 1)).abs() < 60);
        assert!(row(0).iter().skip(3).step_by(4).all(|&a| a == 255));
    }

    #[test]
    fn coats_follow_the_kind() {
        let mut rng = Rng::new(4);
        let mut light = 0;
        for _ in 0..400 {
            let c = Coat::random(&mut rng, Kind::Cattle);
            light += (c.parts[0][1] > 0.4) as usize;
            for part in c.parts {
                assert!(part.iter().all(|v| (0.0..=0.95).contains(v)));
            }
        }
        // white, patched and cream cattle: about 65% light
        assert!((200..320).contains(&light), "{light} light of 400");
        let p = Coat::random(&mut rng, Kind::Person);
        assert!(!p.patches);
    }
}
