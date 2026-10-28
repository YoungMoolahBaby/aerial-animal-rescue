//! Thermal camera model: turns the renderer's temperature image into what an uncooled
//! 640x512 core puts out. No Bevy here.
//!
//! Order follows the camera: optics blur, then noise and fixed-pattern stripes in raw
//! counts, then auto-gain (a percentile stretch, the main reason the same scene looks
//! different from frame to frame), then output noise and 8-bit output. Compression happens when the image
//! is saved (JPEG quality is part of the parameters).

use crate::rng::Rng;

#[derive(Clone, Debug)]
pub struct SensorParams {
    /// Gaussian optics blur, pixels.
    pub blur_sigma: f32,
    /// Temporal noise in raw grey levels (0-255 scale, before auto-gain).
    pub noise: f32,
    /// Noise in output grey levels, after auto-gain. Real footage shows 1-2 levels of
    /// pixel-to-pixel noise whatever the scene, so most of the noise is set here; raw
    /// noise alone would vanish whenever a hot roof widens the gain range.
    pub out_noise: f32,
    /// Spread of per-column and per-row offsets (fixed-pattern noise).
    pub fpn_col: f32,
    pub fpn_row: f32,
    /// Auto-gain: the input percentiles that map to the output range.
    pub agc_lo_pct: f32,
    pub agc_hi_pct: f32,
    pub out_lo: f32,
    pub out_hi: f32,
    pub jpeg_quality: u8,
}

impl SensorParams {
    pub fn random(rng: &mut Rng) -> Self {
        Self {
            blur_sigma: rng.range(0.5, 1.3) as f32,
            noise: rng.range(0.2, 0.8) as f32,
            out_noise: rng.range(0.8, 2.5) as f32,
            fpn_col: rng.range(0.0, 0.6) as f32,
            fpn_row: rng.range(0.0, 0.8) as f32,
            agc_lo_pct: rng.range(0.1, 2.0) as f32,
            agc_hi_pct: rng.range(98.0, 99.95) as f32,
            out_lo: rng.range(10.0, 80.0) as f32,
            out_hi: rng.range(180.0, 250.0) as f32,
            jpeg_quality: rng.range(70.0, 96.0) as u8,
        }
    }
}

/// Apply the camera to a `width` x `height` grey render.
pub fn apply(raw: &[u8], width: usize, height: usize, p: &SensorParams, rng: &mut Rng) -> Vec<u8> {
    assert_eq!(raw.len(), width * height);
    let img: Vec<f32> = raw.iter().map(|&v| v as f32).collect();
    let mut img = blur(&img, width, height, p.blur_sigma);
    let cols: Vec<f32> = (0..width)
        .map(|_| p.fpn_col * rng.normal() as f32)
        .collect();
    let rows: Vec<f32> = (0..height)
        .map(|_| p.fpn_row * rng.normal() as f32)
        .collect();
    for (k, v) in img.iter_mut().enumerate() {
        *v += p.noise * rng.normal() as f32 + cols[k % width] + rows[k / width];
    }
    let (lo, hi) = (
        percentile(&img, p.agc_lo_pct),
        percentile(&img, p.agc_hi_pct),
    );
    let gain = (p.out_hi - p.out_lo) / (hi - lo).max(1.0);
    img.iter()
        .map(|&v| {
            let out = p.out_lo + (v - lo) * gain + p.out_noise * rng.normal() as f32;
            out.round().clamp(0.0, 255.0) as u8
        })
        .collect()
}

/// Separable Gaussian blur with clamped edges.
pub fn blur(img: &[f32], width: usize, height: usize, sigma: f32) -> Vec<f32> {
    if sigma <= 0.0 {
        return img.to_vec();
    }
    let r = (3.0 * sigma).ceil() as isize;
    let kernel: Vec<f32> = (-r..=r)
        .map(|i| (-(i * i) as f32 / (2.0 * sigma * sigma)).exp())
        .collect();
    let sum: f32 = kernel.iter().sum();
    let kernel: Vec<f32> = kernel.iter().map(|k| k / sum).collect();
    let pass = |src: &[f32], horizontal: bool| -> Vec<f32> {
        let mut out = vec![0.0; src.len()];
        for y in 0..height {
            for x in 0..width {
                let mut acc = 0.0;
                for (t, k) in kernel.iter().enumerate() {
                    let d = t as isize - r;
                    let (sx, sy) = if horizontal {
                        ((x as isize + d).clamp(0, width as isize - 1) as usize, y)
                    } else {
                        (x, (y as isize + d).clamp(0, height as isize - 1) as usize)
                    };
                    acc += k * src[sy * width + sx];
                }
                out[y * width + x] = acc;
            }
        }
        out
    };
    pass(&pass(img, true), false)
}

/// The `pct` percentile (0-100) of `values`.
pub fn percentile(values: &[f32], pct: f32) -> f32 {
    let mut v = values.to_vec();
    let k = ((pct / 100.0) * (v.len() - 1) as f32).round() as usize;
    let (_, x, _) = v.select_nth_unstable_by(k, |a, b| a.total_cmp(b));
    *x
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blur_keeps_the_mean_and_spreads_a_point() {
        let mut img = vec![0.0; 25];
        img[12] = 25.0;
        let b = blur(&img, 5, 5, 1.0);
        assert!((b.iter().sum::<f32>() - 25.0).abs() < 1.5); // edges clamp, so ~25
        assert!(b[12] < 25.0 && b[11] > 0.0 && b[7] > 0.0);
    }

    #[test]
    fn auto_gain_stretches_to_the_output_range() {
        let raw: Vec<u8> = (0..100 * 100).map(|k| (100 + k % 20) as u8).collect();
        let p = SensorParams {
            blur_sigma: 0.0,
            noise: 0.0,
            out_noise: 0.0,
            fpn_col: 0.0,
            fpn_row: 0.0,
            agc_lo_pct: 0.0,
            agc_hi_pct: 100.0,
            out_lo: 20.0,
            out_hi: 220.0,
            jpeg_quality: 90,
        };
        let out = apply(&raw, 100, 100, &p, &mut Rng::new(0));
        assert_eq!(*out.iter().min().unwrap(), 20);
        assert_eq!(*out.iter().max().unwrap(), 220);
    }

    #[test]
    fn percentile_ends() {
        let v: Vec<f32> = (0..=100).map(|x| x as f32).collect();
        assert_eq!(percentile(&v, 0.0), 0.0);
        assert_eq!(percentile(&v, 50.0), 50.0);
        assert_eq!(percentile(&v, 100.0), 100.0);
    }
}
