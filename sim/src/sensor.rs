//! Thermal camera model: turns the renderer's temperature image into what an uncooled
//! 640x512 core puts out. No Bevy here.
//!
//! Order follows the camera: optics blur, then noise and fixed-pattern stripes in raw
//! counts, then auto-gain (a percentile stretch, the main reason the same scene looks
//! different from frame to frame), then output noise and 8-bit output. Compression happens
//! when the image is saved (JPEG quality is part of the parameters), and half the frames
//! first go through video-style compression; 40% are interlaced.
//!
//! The ranges span clean frames (HIT-UAV: noise ~1 grey level, crisp edges) to the hard
//! BIRDSAI flights (noise 5-6 levels, in blotches a few pixels across, that survive the
//! compression; soft faint bodies; a squeezed brightness range), drawn log-uniform so
//! both ends are common.

use crate::rng::Rng;

#[derive(Clone, Debug)]
pub struct SensorParams {
    /// Gaussian optics blur, pixels.
    pub blur_sigma: f32,
    /// Motion smear while the aircraft moves (BIRDSAI is filmed from a fixed-wing plane):
    /// length in pixels (0 = none) and direction in radians.
    pub motion_px: f32,
    pub motion_dir: f32,
    /// Interlaced video: odd rows come from the second field, shifted sideways by this
    /// many pixels as the aircraft moves (0 = progressive). Gives the combed edges of
    /// the BIRDSAI frames.
    pub interlace_px: f32,
    /// Video compression before the frame is saved: JPEG round trip at this quality
    /// (blocky, smears fine noise into blotches), or none.
    pub video_quality: Option<u8>,
    /// Noise added after compression (the video link and digitiser), grey levels.
    pub link_noise: f32,
    /// Temporal noise in raw grey levels (0-255 scale, before auto-gain).
    pub noise: f32,
    /// Noise in output grey levels, after auto-gain. Real footage shows 1-6 levels of
    /// pixel-to-pixel noise whatever the scene, so most of the noise is set here; raw
    /// noise alone would vanish whenever a hot roof widens the gain range.
    pub out_noise: f32,
    /// Blotchy noise in output grey levels: white noise blurred over `blotch_px`, like the
    /// mottled texture of compressed thermal video.
    pub blotch: f32,
    pub blotch_px: f32,
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
        let out_lo = rng.range(10.0, 80.0) as f32;
        Self {
            blur_sigma: log_range(rng, 0.5, 2.0),
            motion_px: if rng.chance(0.5) {
                rng.range(1.0, 4.0) as f32
            } else {
                0.0
            },
            motion_dir: rng.range(0.0, std::f64::consts::PI) as f32,
            interlace_px: if rng.chance(0.4) {
                rng.range(1.0, 3.0) as f32
            } else {
                0.0
            },
            // gentler than it looks: quality < 40 wiped out the noise real recordings keep
            video_quality: rng.chance(0.5).then(|| rng.range(40.0, 80.0) as u8),
            link_noise: rng.range(0.3, 2.0) as f32,
            noise: rng.range(0.2, 0.8) as f32,
            out_noise: log_range(rng, 0.8, 8.0),
            blotch: log_range(rng, 0.3, 6.0),
            blotch_px: rng.range(1.0, 3.0) as f32,
            fpn_col: rng.range(0.0, 0.6) as f32,
            fpn_row: rng.range(0.0, 0.8) as f32,
            agc_lo_pct: rng.range(0.1, 2.0) as f32,
            agc_hi_pct: rng.range(98.0, 99.95) as f32,
            out_lo,
            // a squeezed range sometimes (BIRDSAI flight 361: p1-p99 37-147)
            out_hi: (out_lo + rng.range(80.0, 200.0) as f32).min(250.0),
            // low JPEG quality smooths noise away; real hard flights keep theirs
            jpeg_quality: rng.range(65.0, 96.0) as u8,
        }
    }
}

/// Log-uniform in [lo, hi).
fn log_range(rng: &mut Rng, lo: f64, hi: f64) -> f32 {
    rng.range(lo.ln(), hi.ln()).exp() as f32
}

/// Apply the camera to a `width` x `height` grey render.
pub fn apply(raw: &[u8], width: usize, height: usize, p: &SensorParams, rng: &mut Rng) -> Vec<u8> {
    assert_eq!(raw.len(), width * height);
    let img: Vec<f32> = raw.iter().map(|&v| v as f32).collect();
    let img = blur(&img, width, height, p.blur_sigma);
    let img = smear(&img, width, height, p.motion_px, p.motion_dir);
    let mut img = interlace(&img, width, height, p.interlace_px);
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
    let blotches = unit_blotches(width, height, p.blotch_px, rng);
    let out: Vec<u8> = img
        .iter()
        .zip(&blotches)
        .map(|(&v, &b)| {
            let out = p.out_lo + (v - lo) * gain + p.out_noise * rng.normal() as f32 + p.blotch * b;
            out.round().clamp(0.0, 255.0) as u8
        })
        .collect();
    match p.video_quality {
        Some(q) => jpeg_round_trip(&out, width, height, q)
            .into_iter()
            .map(|v| {
                (v as f32 + p.link_noise * rng.normal() as f32)
                    .round()
                    .clamp(0.0, 255.0) as u8
            })
            .collect(),
        None => out,
    }
}

/// Shift every odd row sideways by `px` pixels (rounded), clamping at the edges.
pub fn interlace(img: &[f32], width: usize, height: usize, px: f32) -> Vec<f32> {
    let shift = px.round() as isize;
    if shift == 0 {
        return img.to_vec();
    }
    let mut out = img.to_vec();
    for y in (1..height).step_by(2) {
        for x in 0..width {
            let sx = (x as isize - shift).clamp(0, width as isize - 1) as usize;
            out[y * width + x] = img[y * width + sx];
        }
    }
    out
}

/// Encode as JPEG at `quality` and decode again.
pub fn jpeg_round_trip(grey: &[u8], width: usize, height: usize, quality: u8) -> Vec<u8> {
    use image::codecs::jpeg::JpegEncoder;
    let mut buf = Vec::new();
    JpegEncoder::new_with_quality(&mut buf, quality)
        .encode(
            grey,
            width as u32,
            height as u32,
            image::ExtendedColorType::L8,
        )
        .expect("encode JPEG");
    image::load_from_memory(&buf)
        .expect("decode JPEG")
        .to_luma8()
        .into_raw()
}

/// Average along a line of `length` pixels in direction `dir` (motion blur), sampled every
/// half pixel with clamped edges.
pub fn smear(img: &[f32], width: usize, height: usize, length: f32, dir: f32) -> Vec<f32> {
    if length < 0.5 {
        return img.to_vec();
    }
    let n = (2.0 * length).ceil() as usize + 1;
    let offsets: Vec<(isize, isize)> = (0..n)
        .map(|k| {
            let t = (k as f32 / (n - 1) as f32 - 0.5) * length;
            (
                (t * dir.cos()).round() as isize,
                (t * dir.sin()).round() as isize,
            )
        })
        .collect();
    let mut out = vec![0.0; img.len()];
    for y in 0..height as isize {
        for x in 0..width as isize {
            let mut acc = 0.0;
            for &(dx, dy) in &offsets {
                let sx = (x + dx).clamp(0, width as isize - 1) as usize;
                let sy = (y + dy).clamp(0, height as isize - 1) as usize;
                acc += img[sy * width + sx];
            }
            out[y as usize * width + x as usize] = acc / n as f32;
        }
    }
    out
}

/// White noise blurred over `sigma` pixels, rescaled to unit spread.
fn unit_blotches(width: usize, height: usize, sigma: f32, rng: &mut Rng) -> Vec<f32> {
    let white: Vec<f32> = (0..width * height).map(|_| rng.normal() as f32).collect();
    let b = blur(&white, width, height, sigma);
    let std = (b.iter().map(|v| v * v).sum::<f32>() / b.len() as f32).sqrt();
    b.iter().map(|v| v / std.max(1e-6)).collect()
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
            motion_px: 0.0,
            motion_dir: 0.0,
            interlace_px: 0.0,
            video_quality: None,
            link_noise: 0.0,
            noise: 0.0,
            out_noise: 0.0,
            blotch: 0.0,
            blotch_px: 1.0,
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
    fn smear_spreads_along_its_direction_only() {
        let mut img = vec![0.0; 81];
        img[40] = 9.0; // centre of 9x9
        let s = smear(&img, 9, 9, 4.0, 0.0); // horizontal
        assert!(s[39] > 0.0 && s[41] > 0.0);
        assert_eq!(s[31], 0.0); // nothing above
        assert!((s.iter().sum::<f32>() - 9.0).abs() < 1e-4);
    }

    #[test]
    fn interlace_shifts_odd_rows_only() {
        let img: Vec<f32> = (0..4 * 3).map(|k| (k % 4) as f32).collect(); // 4 wide, 3 high
        let out = interlace(&img, 4, 3, 1.0);
        assert_eq!(&out[0..4], &[0.0, 1.0, 2.0, 3.0]);
        assert_eq!(&out[4..8], &[0.0, 0.0, 1.0, 2.0]);
        assert_eq!(&out[8..12], &[0.0, 1.0, 2.0, 3.0]);
    }

    #[test]
    fn jpeg_round_trip_keeps_size_and_roughly_values() {
        let img: Vec<u8> = (0..64 * 64).map(|k| (k % 64 * 4) as u8).collect();
        let out = jpeg_round_trip(&img, 64, 64, 30);
        assert_eq!(out.len(), img.len());
        let err: f64 = img
            .iter()
            .zip(&out)
            .map(|(&a, &b)| (a as f64 - b as f64).abs())
            .sum::<f64>()
            / img.len() as f64;
        assert!(err < 10.0, "{err}");
    }

    #[test]
    fn percentile_ends() {
        let v: Vec<f32> = (0..=100).map(|x| x as f32).collect();
        assert_eq!(percentile(&v, 0.0), 0.0);
        assert_eq!(percentile(&v, 50.0), 50.0);
        assert_eq!(percentile(&v, 100.0), 100.0);
    }
}
