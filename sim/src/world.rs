//! The world: terrain and where things stand on it. No Bevy here.
//!
//! Terrain is a CortenForge `HeightFieldData` (Z-up, x/y on the ground). Bevy is Y-up, so
//! the public helpers take Bevy-style ground coordinates (x, z) and return heights (y).

use cortenforge::cf_geometry::HeightFieldData;

use crate::rng::{Rng, lattice};

pub struct Terrain {
    pub field: HeightFieldData,
}

impl Terrain {
    /// Rolling ground of `size_m` x `size_m`, centred on the origin, sampled every `cell_m`.
    /// `relief_m` is the height range of the hills.
    pub fn generate(seed: u64, size_m: f64, cell_m: f64, relief_m: f64) -> Self {
        let n = (size_m / cell_m).round() as usize + 1;
        let field = HeightFieldData::from_fn(n, n, cell_m, |x, y| {
            relief_m * fbm(seed, x / 120.0, y / 120.0, 4)
        });
        Self { field }
    }

    fn half(&self) -> f64 {
        self.field.extent_x() / 2.0
    }

    /// Ground height at Bevy coordinates (x, z); clamped at the edges.
    pub fn height(&self, x: f64, z: f64) -> f64 {
        self.field.sample_clamped(x + self.half(), z + self.half())
    }

    /// Grid vertices in Bevy coordinates (x, height, z), row by row, and the grid size.
    pub fn vertices(&self) -> (Vec<[f64; 3]>, usize, usize) {
        let (w, d) = (self.field.width(), self.field.depth());
        let half = self.half();
        let mut out = Vec::with_capacity(w * d);
        for j in 0..d {
            for i in 0..w {
                let p = self
                    .field
                    .vertex_position(i, j)
                    .expect("grid index in range");
                out.push([p.x - half, p.z, p.y - half]);
            }
        }
        (out, w, d)
    }

    /// Unit normal at Bevy coordinates (x, z), as (x, y, z) with y up.
    pub fn normal(&self, x: f64, z: f64) -> [f64; 3] {
        let n = self.field.normal_clamped(x + self.half(), z + self.half());
        [n.x, n.z, n.y]
    }

    /// A random spot on the ground within `radius` of the origin.
    pub fn random_spot(&self, rng: &mut Rng, radius: f64) -> [f64; 3] {
        let (x, z) = (rng.range(-radius, radius), rng.range(-radius, radius));
        [x, self.height(x, z), z]
    }
}

/// Fractal value noise in about [-1, 1].
pub fn fbm(seed: u64, x: f64, y: f64, octaves: u32) -> f64 {
    let (mut sum, mut amp, mut freq, mut norm) = (0.0, 1.0, 1.0, 0.0);
    for o in 0..octaves {
        sum += amp * (2.0 * value_noise(seed.wrapping_add(o as u64), x * freq, y * freq) - 1.0);
        norm += amp;
        amp *= 0.5;
        freq *= 2.0;
    }
    sum / norm
}

fn value_noise(seed: u64, x: f64, y: f64) -> f64 {
    let (x0, y0) = (x.floor(), y.floor());
    let (fx, fy) = (smooth(x - x0), smooth(y - y0));
    let (i, j) = (x0 as i64, y0 as i64);
    let a = lattice(seed, i, j) + (lattice(seed, i + 1, j) - lattice(seed, i, j)) * fx;
    let b = lattice(seed, i, j + 1) + (lattice(seed, i + 1, j + 1) - lattice(seed, i, j + 1)) * fx;
    a + (b - a) * fy
}

fn smooth(t: f64) -> f64 {
    t * t * (3.0 - 2.0 * t)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn terrain_is_reproducible_and_centred() {
        let a = Terrain::generate(3, 400.0, 2.0, 15.0);
        let b = Terrain::generate(3, 400.0, 2.0, 15.0);
        assert_eq!(a.height(10.0, -20.0), b.height(10.0, -20.0));
        let (v, w, d) = a.vertices();
        assert_eq!((w, d), (201, 201));
        assert_eq!(v[0][0], -200.0);
        assert_eq!(v[w * d - 1][2], 200.0);
        // a vertex height matches the sampled height at its spot
        let p = v[w * 50 + 70];
        assert!((a.height(p[0], p[2]) - p[1]).abs() < 1e-9);
        assert!(v.iter().all(|p| p[1].abs() <= 15.0));
    }

    #[test]
    fn normals_point_up() {
        let t = Terrain::generate(1, 200.0, 2.0, 10.0);
        let n = t.normal(5.0, 5.0);
        assert!(n[1] > 0.5);
    }
}
