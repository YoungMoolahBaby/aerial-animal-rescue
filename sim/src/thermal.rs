//! Temperatures of the world for one shot, and how they map to grey levels. No Bevy here.
//!
//! The renderer draws temperature as an unlit grey: every material's colour is
//! `grey(temperature)` in a window chosen per shot, so the 8-bit render keeps as much
//! precision as possible before the sensor model (`sensor.rs`) stretches it like a
//! camera's auto-gain. The numbers are plausible, not physical: what has to match the real
//! data is how objects stand out from the ground around them (`scripts/compare_domains.py`).

use crate::rng::Rng;
use crate::world::fbm;

/// Weather and time of day for one shot, in degrees Celsius.
#[derive(Clone, Debug)]
pub struct Conditions {
    pub night: bool,
    pub air_c: f64,
    /// Extra warmth of bare ground facing the sun (0 at night).
    pub sun_c: f64,
    /// Unit vector towards the sun, Bevy coordinates (y up).
    pub sun_dir: [f64; 3],
    /// At night, how much colder open ground gets than ground under vegetation.
    pub cooling_c: f64,
    /// Soil and grass patches, metres to tens of metres across.
    pub patch_c: f64,
    /// Fine ground texture (grass clumps, stones) as a fraction of the grey level.
    pub texture: f64,
    /// Flood water: surface height (m) and temperature.
    pub water: Option<(f64, f64)>,
}

impl Conditions {
    pub fn random(rng: &mut Rng, water_level: Option<f64>) -> Self {
        let night = rng.chance(0.5);
        let air_c = rng.range(0.0, 30.0);
        let elevation = rng.range(15.0_f64, 70.0).to_radians();
        let azimuth = rng.range(0.0, std::f64::consts::TAU);
        let sun_dir = [
            elevation.cos() * azimuth.cos(),
            elevation.sin(),
            elevation.cos() * azimuth.sin(),
        ];
        let water = water_level.map(|level| {
            // Water lags the air: cooler than sunny ground by day, often warmer at night.
            let t = if night {
                air_c + rng.range(-1.0, 4.0)
            } else {
                air_c + rng.range(-5.0, 1.0)
            };
            (level, t)
        });
        Self {
            night,
            air_c,
            sun_c: if night { 0.0 } else { rng.range(2.0, 15.0) },
            sun_dir,
            cooling_c: if night { rng.range(1.0, 6.0) } else { 0.0 },
            patch_c: rng.range(0.3, 3.0),
            texture: rng.range(0.02, 0.2),
            water,
        }
    }

    /// Ground temperature at Bevy (x, z) with unit normal `n`.
    pub fn ground_c(&self, seed: u64, x: f64, z: f64, n: [f64; 3]) -> f64 {
        let cover = vegetation(seed, x, z);
        let patches = fbm(seed.wrapping_add(200), x / 9.0, z / 9.0, 3);
        if self.night {
            self.air_c - self.cooling_c * (1.0 - cover) + self.patch_c * patches
        } else {
            let facing =
                (n[0] * self.sun_dir[0] + n[1] * self.sun_dir[1] + n[2] * self.sun_dir[2]).max(0.0);
            self.air_c + self.sun_c * facing * (1.0 - 0.6 * cover) - 2.0 * cover
                + self.patch_c * patches
        }
    }

    /// Vegetation (bushes, trees): a little warmer than open ground at night, cooler than
    /// sunny ground by day.
    pub fn vegetation_c(&self, rng: &mut Rng) -> f64 {
        if self.night {
            self.air_c + rng.range(-1.0, 2.0)
        } else {
            self.air_c + rng.range(-1.0, 0.5) * self.sun_c.min(5.0)
        }
    }

    /// Rocks and other warm clutter: hot in the sun, holding heat at night. Not labelled;
    /// they are there so the model learns that a warm blob isn't always a body.
    pub fn clutter_c(&self, rng: &mut Rng) -> f64 {
        if self.night {
            self.air_c + rng.range(-3.0, 3.0)
        } else {
            self.air_c + self.sun_c * rng.range(0.3, 1.4)
        }
    }

    /// Roofs: cold at night (radiating to the sky) unless heated, hot in the sun.
    pub fn building_c(&self, rng: &mut Rng) -> f64 {
        if self.night {
            if rng.chance(0.2) {
                self.air_c + rng.range(1.0, 6.0)
            } else {
                self.air_c - rng.range(2.0, 8.0)
            }
        } else {
            self.air_c + self.sun_c * rng.range(0.3, 1.5)
        }
    }

    /// Vehicles: metal in the sun, or a warm engine at night.
    pub fn vehicle_c(&self, rng: &mut Rng) -> f64 {
        if self.night {
            self.air_c + rng.range(-4.0, 10.0)
        } else {
            self.air_c + self.sun_c * rng.range(0.5, 2.0)
        }
    }

    /// Car glass reflects the cold sky: cooler than the body, day and night.
    pub fn glass_c(&self, rng: &mut Rng) -> f64 {
        if self.night {
            self.air_c - rng.range(2.0, 8.0)
        } else {
            self.air_c + rng.range(-3.0, 3.0)
        }
    }

    /// Roads and tracks hold heat: warmer than soil at night, hotter in the sun.
    pub fn road_c(&self, rng: &mut Rng) -> f64 {
        if self.night {
            self.air_c + rng.range(-1.0, 4.0)
        } else {
            self.air_c + 1.0 + self.sun_c * rng.range(0.6, 1.3)
        }
    }

    /// A body's surface temperature, from the ground under it: people and animals
    /// stand out from what's around them, less in the heat of the day, and clothing or wet
    /// fur sometimes makes them darker than sun-baked ground.
    pub fn body_c(&self, rng: &mut Rng, ground_c: f64, person: bool) -> f64 {
        // log-uniform: faint bodies (BIRDSAI flight 361: 50 grey levels, 3.7x the ground's
        // spread) are as common as bright ones
        let (lo, hi) = match (self.night, person) {
            (true, true) => (2.0_f64, 15.0_f64),
            (true, false) => (1.5, 12.0),
            (false, true) => (1.5, 10.0),
            (false, false) => (1.0, 8.0),
        };
        let delta = rng.range(lo.ln(), hi.ln()).exp();
        let darker = !self.night && rng.chance(0.1);
        ground_c + if darker { -delta } else { delta }
    }
}

/// Share of vegetation cover in [0, 1], in patches tens of metres across.
pub fn vegetation(seed: u64, x: f64, z: f64) -> f64 {
    let v = fbm(seed.wrapping_add(100), x / 45.0, z / 45.0, 3);
    ((v + 0.2) * 2.0).clamp(0.0, 1.0)
}

/// Temperature window for one shot: `grey` maps `lo..hi` to `0..1`.
#[derive(Clone, Copy, Debug)]
pub struct TempWindow {
    pub lo: f64,
    pub hi: f64,
}

impl TempWindow {
    /// A window around every temperature in the shot, with a margin.
    pub fn around(temps: impl IntoIterator<Item = f64>) -> Self {
        let (mut lo, mut hi) = (f64::MAX, f64::MIN);
        for t in temps {
            lo = lo.min(t);
            hi = hi.max(t);
        }
        Self {
            lo: lo - 1.0,
            hi: hi.max(lo + 1.0) + 1.0,
        }
    }

    pub fn grey(&self, t: f64) -> f64 {
        ((t - self.lo) / (self.hi - self.lo)).clamp(0.0, 1.0)
    }

    /// Linear colour value that comes out of an sRGB render target as `grey(t)`: unlit
    /// materials take linear colour and the window encodes it to sRGB.
    pub fn linear(&self, t: f64) -> f32 {
        srgb_to_linear(self.grey(t)) as f32
    }
}

pub fn srgb_to_linear(v: f64) -> f64 {
    if v <= 0.04045 {
        v / 12.92
    } else {
        ((v + 0.055) / 1.055).powf(2.4)
    }
}

/// Fine ground texture as sRGB bytes (`size` x `size`, tiling): 255 is "no change", darker
/// texels pull the ground's grey down by up to `amount`.
pub fn ground_texture(seed: u64, size: usize, amount: f64) -> Vec<u8> {
    let cells = 24.0; // features about 1/24 of the tile, a few pixels each in the picture
    let mut out = Vec::with_capacity(size * size);
    for j in 0..size {
        for i in 0..size {
            let (u, v) = (i as f64 / size as f64, j as f64 / size as f64);
            let n = tiling_fbm(seed, u, v, cells);
            out.push((255.0 * (1.0 - amount * (0.5 + 0.5 * n))).round() as u8);
        }
    }
    out
}

/// Value-noise fbm over the unit square that wraps at its edges.
fn tiling_fbm(seed: u64, u: f64, v: f64, cells: f64) -> f64 {
    let (mut sum, mut amp, mut norm, mut period) = (0.0, 1.0, 0.0, cells);
    for o in 0..4u64 {
        sum += amp
            * (2.0 * tiling_noise(seed.wrapping_add(o), u * period, v * period, period as i64)
                - 1.0);
        norm += amp;
        amp *= 0.5;
        period *= 2.0;
    }
    sum / norm
}

fn tiling_noise(seed: u64, x: f64, y: f64, period: i64) -> f64 {
    use crate::rng::lattice;
    let (x0, y0) = (x.floor(), y.floor());
    let s = |t: f64| t * t * (3.0 - 2.0 * t);
    let (fx, fy) = (s(x - x0), s(y - y0));
    let (i, j) = (x0 as i64, y0 as i64);
    let l = |a: i64, b: i64| lattice(seed, a.rem_euclid(period), b.rem_euclid(period));
    let a = l(i, j) + (l(i + 1, j) - l(i, j)) * fx;
    let b = l(i, j + 1) + (l(i + 1, j + 1) - l(i, j + 1)) * fx;
    a + (b - a) * fy
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn window_maps_ends_and_survives_srgb() {
        let w = TempWindow::around([10.0, 20.0]);
        assert_eq!((w.lo, w.hi), (9.0, 21.0));
        assert_eq!(w.grey(9.0), 0.0);
        assert_eq!(w.grey(30.0), 1.0);
        // linear -> sRGB encode gives the grey back
        let g = w.grey(15.0);
        let lin = w.linear(15.0) as f64;
        let enc = if lin <= 0.0031308 {
            lin * 12.92
        } else {
            1.055 * lin.powf(1.0 / 2.4) - 0.055
        };
        assert!((enc - g).abs() < 1e-5);
    }

    #[test]
    fn bodies_stand_out_at_night() {
        let mut rng = Rng::new(4);
        let c = Conditions {
            night: true,
            ..Conditions::random(&mut rng, None)
        };
        for _ in 0..100 {
            assert!(c.body_c(&mut rng, 5.0, true) >= 7.0);
        }
    }

    #[test]
    fn texture_tiles_and_stays_in_range() {
        let t = ground_texture(1, 64, 0.2);
        assert_eq!(t.len(), 64 * 64);
        assert!(t.iter().all(|&v| v >= 200));
        // wraps: the noise at u = 0 and u = 1 is the same
        assert!((tiling_fbm(1, 0.0, 0.3, 8.0) - tiling_fbm(1, 1.0, 0.3, 8.0)).abs() < 1e-9);
    }
}
