//! Small seeded RNG (SplitMix64), so every scene is reproducible from its seed.

#[derive(Clone, Debug)]
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(seed)
    }

    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in [0, 1).
    pub fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Uniform in [lo, hi).
    pub fn range(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.unit()
    }

    /// True with probability `p`.
    pub fn chance(&mut self, p: f64) -> bool {
        self.unit() < p
    }

    /// Uniform index in 0..n.
    pub fn index(&mut self, n: usize) -> usize {
        (self.unit() * n as f64) as usize
    }

    /// Standard normal (Box-Muller).
    pub fn normal(&mut self) -> f64 {
        let u = 1.0 - self.unit(); // (0, 1], so ln is finite
        (-2.0 * u.ln()).sqrt() * (std::f64::consts::TAU * self.unit()).cos()
    }
}

/// Hash of integer lattice coordinates to [0, 1), for value noise.
pub fn lattice(seed: u64, x: i64, y: i64) -> f64 {
    let key = seed
        ^ (x as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
        ^ (y as u64).wrapping_mul(0xC2B2_AE3D_27D4_EB4F);
    Rng::new(key).unit()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_seed_same_sequence_and_unit_range() {
        let (mut a, mut b) = (Rng::new(7), Rng::new(7));
        for _ in 0..1000 {
            let u = a.unit();
            assert_eq!(u, b.unit());
            assert!((0.0..1.0).contains(&u));
        }
    }

    #[test]
    fn normal_has_unit_spread() {
        let mut r = Rng::new(1);
        let xs: Vec<f64> = (0..20_000).map(|_| r.normal()).collect();
        let mean = xs.iter().sum::<f64>() / xs.len() as f64;
        let var = xs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / xs.len() as f64;
        assert!(mean.abs() < 0.03 && (var - 1.0).abs() < 0.05);
    }
}
