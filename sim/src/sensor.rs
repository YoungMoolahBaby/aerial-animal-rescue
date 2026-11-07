//! Thermal camera model: turns the renderer's temperature image into what an uncooled
//! 640x512 core puts out. No Bevy here.
//!
//! Order follows the camera: haze between camera and ground, optics blur, then noise and
//! fixed-pattern stripes in raw counts, then auto-gain (a percentile stretch, the main reason the same scene looks
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
    /// Detail enhancement (the DDE of DJI and FLIR cores): the image is split into its blur
    /// over `detail_px` pixels (the base) and the rest (the detail); the base's histogram
    /// is equalized, blended `equalize` of the way (0 = off), and the detail added back
    /// boosted `detail` times (1 = off). Equalizing squeezes the empty range between the
    /// ground and a small warm body, so the body's middle comes out little brighter than
    /// the brightest ground and its rim, from the detail, white: BAMBI's rimmed bodies on
    /// a crisp, busy ground.
    pub detail: f32,
    pub detail_px: f32,
    pub equalize: f32,
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
    /// Haze, fog or drizzle between camera and ground: the share of the scene's contrast
    /// lost before the sensor (0 = clear). The auto-gain stretches what's left back out,
    /// raw noise with it, so bodies come out faint in noise (BIRDSAI flight 361: 10 px
    /// people 50 grey levels above the ground, in 5 levels of noise). See `with_weather`.
    pub haze: f32,
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
            // one draw, as before (shots stay paired with older pools)
            noise: log_range(rng, 0.5, 4.0),
            haze: 0.0,
            out_noise: log_range(rng, 1.5, 9.0),
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
            detail: 1.0,
            detail_px: 0.0,
            equalize: 0.0,
        }
    }
}

impl SensorParams {
    /// A cleaner camera for forest shots, like BAMBI's (DJI thermal drones, saved as
    /// undistorted video): little noise (median neighbour difference 1 grey level), no
    /// interlace, a gain that keeps the ground darkish and lets warm bodies go white.
    pub fn forest(rng: &mut Rng) -> Self {
        let out_lo = rng.range(0.0, 30.0) as f32;
        Self {
            blur_sigma: log_range(rng, 0.5, 1.3),
            motion_px: if rng.chance(0.15) {
                rng.range(1.0, 3.0) as f32
            } else {
                0.0
            },
            motion_dir: rng.range(0.0, std::f64::consts::PI) as f32,
            interlace_px: 0.0,
            video_quality: rng.chance(0.3).then(|| rng.range(70.0, 90.0) as u8),
            link_noise: rng.range(0.2, 0.8) as f32,
            noise: log_range(rng, 0.2, 1.2),
            haze: 0.0,
            out_noise: log_range(rng, 0.3, 2.0),
            blotch: log_range(rng, 0.1, 1.5),
            blotch_px: rng.range(1.0, 3.0) as f32,
            fpn_col: rng.range(0.0, 0.3) as f32,
            fpn_row: rng.range(0.0, 0.3) as f32,
            agc_lo_pct: rng.range(0.2, 3.0) as f32,
            agc_hi_pct: rng.range(99.0, 99.95) as f32,
            out_lo,
            out_hi: (out_lo + rng.range(170.0, 240.0) as f32).min(255.0),
            jpeg_quality: rng.range(85.0, 97.0) as u8,
            // half the shots enhanced, from gently to BAMBI's strongest (white-rimmed
            // bodies on a crisp, busy ground)
            detail: 1.0,
            detail_px: log_range(rng, 1.5, 6.0),
            equalize: 0.0,
        }
        .enhanced(rng)
    }

    /// Detail enhancement in half the shots (`detail`).
    fn enhanced(mut self, rng: &mut Rng) -> Self {
        if rng.chance(0.5) {
            self.detail = log_range(rng, 1.5, 5.0);
            self.equalize = rng.range(0.3, 0.9) as f32;
        }
        self
    }
}

/// Side of a BAMBI frame: the 640x512 thermal image, undistorted and scaled into a
/// 1024x1024 frame with black above and below.
pub const BAMBI_SIZE: usize = 1024;

/// Where the picture starts and ends in BAMBI column `u` (-1 left edge .. 1 right):
/// undistortion bows the top and bottom edges, so the black bands are thicker in the
/// middle (rows 102 and 912) than at the sides (74 and 940); the same on every flight.
fn bambi_rows(u: f64) -> (f64, f64) {
    (102.0 - 28.0 * u * u, 912.0 + 28.0 * u * u)
}

/// A `width` x `height` grey frame as BAMBI stores it: stretched to 1024 wide, each column
/// into the rows `bambi_rows` gives it, black outside (bilinear).
pub fn bambi_frame(img: &[u8], width: usize, height: usize) -> Vec<u8> {
    let n = BAMBI_SIZE;
    let mut out = vec![0u8; n * n];
    let at = |x: isize, y: isize| {
        let x = x.clamp(0, width as isize - 1) as usize;
        let y = y.clamp(0, height as isize - 1) as usize;
        img[y * width + x] as f64
    };
    for x in 0..n {
        let u = (x as f64 + 0.5) / n as f64 * 2.0 - 1.0;
        let (top, bottom) = bambi_rows(u);
        let sx = (x as f64 + 0.5) / n as f64 * width as f64 - 0.5;
        for y in (top.ceil() as usize)..(bottom.floor() as usize).min(n) {
            let sy = (y as f64 + 0.5 - top) / (bottom - top) * height as f64 - 0.5;
            let (x0, y0) = (sx.floor(), sy.floor());
            let (fx, fy) = (sx - x0, sy - y0);
            let (x0, y0) = (x0 as isize, y0 as isize);
            let a = at(x0, y0) + (at(x0 + 1, y0) - at(x0, y0)) * fx;
            let b = at(x0, y0 + 1) + (at(x0 + 1, y0 + 1) - at(x0, y0 + 1)) * fx;
            out[y * n + x] = (a + (b - a) * fy).round().clamp(1.0, 255.0) as u8;
        }
    }
    out
}

/// A box (x1, y1, x2, y2) of a `width` x `height` frame, in `bambi_frame` pixels.
pub fn bambi_box(b: [f32; 4], width: usize, height: usize) -> [f32; 4] {
    let n = BAMBI_SIZE as f64;
    let sx = n / width as f64;
    let (x1, x2) = (b[0] as f64 * sx, b[2] as f64 * sx);
    let (mut y1, mut y2) = (f64::MAX, f64::MIN);
    // the rows stretch most at the sides, so check both ends and the middle if inside
    // (a box past the frame's sides takes the side's rows)
    let mut us = vec![
        (x1 / n * 2.0 - 1.0).clamp(-1.0, 1.0),
        (x2 / n * 2.0 - 1.0).clamp(-1.0, 1.0),
    ];
    if x1 < n / 2.0 && x2 > n / 2.0 {
        us.push(0.0);
    }
    for u in us {
        let (top, bottom) = bambi_rows(u);
        let scale = (bottom - top) / height as f64;
        y1 = y1.min(top + b[1] as f64 * scale);
        y2 = y2.max(top + b[3] as f64 * scale);
    }
    [x1 as f32, y1 as f32, x2 as f32, y2 as f32]
}

impl SensorParams {
    /// Draw the weather from its own stream (so the other draws, and the shots after,
    /// stay as they were): 35% of frames hazy or foggy.
    pub fn with_weather(mut self, rng: &mut Rng) -> Self {
        self.haze = if rng.chance(0.35) {
            rng.range(0.3, 0.8) as f32
        } else {
            0.0
        };
        self
    }

    /// The metadata's `weather` value.
    pub fn weather(&self) -> &'static str {
        if self.haze > 0.0 { "fog" } else { "clear" }
    }
}

/// Log-uniform in [lo, hi).
fn log_range(rng: &mut Rng, lo: f64, hi: f64) -> f32 {
    rng.range(lo.ln(), hi.ln()).exp() as f32
}

/// Apply the camera to a `width` x `height` grey render.
pub fn apply(raw: &[u8], width: usize, height: usize, p: &SensorParams, rng: &mut Rng) -> Vec<u8> {
    assert_eq!(raw.len(), width * height);
    let mut img: Vec<f32> = raw.iter().map(|&v| v as f32).collect();
    if p.haze > 0.0 {
        let mean = img.iter().sum::<f32>() / img.len() as f32;
        for v in &mut img {
            *v = mean + (*v - mean) * (1.0 - p.haze);
        }
    }
    let img = blur(&img, width, height, p.blur_sigma);
    let img = smear(&img, width, height, p.motion_px, p.motion_dir);
    // on the signal before the noise: the cameras' enhancement holds the noise back
    let img = enhance(&img, width, height, p);
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

/// Detail enhancement (`SensorParams::detail`): the base, the image's blur over
/// `detail_px`, part equalized within its own range, plus the detail boosted.
pub fn enhance(img: &[f32], width: usize, height: usize, p: &SensorParams) -> Vec<f32> {
    if p.detail == 1.0 && p.equalize == 0.0 {
        return img.to_vec();
    }
    let base = blur(img, width, height, p.detail_px);
    let (lo, hi) = base
        .iter()
        .fold((f32::MAX, f32::MIN), |(a, b), &v| (a.min(v), b.max(v)));
    let span = (hi - lo).max(1e-3);
    // the histogram's cumulative share at each bin's middle
    const BINS: usize = 512;
    let bin = |v: f32| (((v - lo) / span * BINS as f32) as usize).min(BINS - 1);
    let mut cdf = vec![0f32; BINS];
    for &v in &base {
        cdf[bin(v)] += 1.0;
    }
    let mut below = 0.0;
    for c in &mut cdf {
        let here = *c;
        *c = (below + here / 2.0) / base.len() as f32;
        below += here;
    }
    img.iter()
        .zip(&base)
        .map(|(&v, &b)| {
            let flat = lo + span * cdf[bin(b)];
            b + p.equalize * (flat - b) + p.detail * (v - b)
        })
        .collect()
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
    fn bambi_frame_has_bowed_black_bands() {
        let (w, h) = (640, 512);
        let out = bambi_frame(&vec![200u8; w * h], w, h);
        let n = BAMBI_SIZE;
        assert_eq!(out.len(), n * n);
        // first picture row: 102 in the middle column, 74 at the sides (BAMBI's frames)
        let first = |x: usize| (0..n).find(|&y| out[y * n + x] > 0).unwrap();
        let last = |x: usize| (0..n).rev().find(|&y| out[y * n + x] > 0).unwrap();
        assert_eq!(first(n / 2), 102);
        assert!((74..=76).contains(&first(0)));
        assert_eq!(last(n / 2), 911);
        assert!((938..=940).contains(&last(n - 1)));
        assert!(out[500 * n + 500] == 200 && out[10 * n + 500] == 0);
    }

    #[test]
    fn bambi_box_follows_the_frame() {
        let (w, h) = (640, 512);
        // a body in the middle of the frame stays in the middle, 1.6x as wide
        let b = bambi_box([310.0, 246.0, 330.0, 266.0], w, h);
        assert!((b[0] - 496.0).abs() < 0.01 && (b[2] - 528.0).abs() < 0.01);
        assert!(b[1] > 102.0 + 246.0 * 1.58 && b[3] < 102.0 + 266.0 * 1.6);
        // the whole picture maps onto the bowed band, widest at the sides
        let all = bambi_box([0.0, 0.0, 640.0, 512.0], w, h);
        assert_eq!(all, [0.0, 74.0, 1024.0, 940.0]);
    }

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
            haze: 0.0,
            out_noise: 0.0,
            blotch: 0.0,
            blotch_px: 1.0,
            detail: 1.0,
            detail_px: 0.0,
            equalize: 0.0,
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
    fn haze_buries_a_body_in_raw_noise() {
        // ground at 100, a 10x10 body at 140 in the middle, a hot roof corner at 250
        let (w, h) = (100, 100);
        let raw: Vec<u8> = (0..w * h)
            .map(|k| {
                let (x, y) = (k % w, k / w);
                if (45..55).contains(&x) && (45..55).contains(&y) {
                    140
                } else if x < 10 && y < 10 {
                    250
                } else {
                    100
                }
            })
            .collect();
        let snr = |haze: f32| {
            let p = SensorParams {
                blur_sigma: 0.0,
                motion_px: 0.0,
                motion_dir: 0.0,
                interlace_px: 0.0,
                video_quality: None,
                link_noise: 0.0,
                noise: 2.0,
                haze,
                out_noise: 0.0,
                blotch: 0.0,
                blotch_px: 1.0,
                detail: 1.0,
                detail_px: 0.0,
                equalize: 0.0,
                fpn_col: 0.0,
                fpn_row: 0.0,
                agc_lo_pct: 0.0,
                agc_hi_pct: 100.0,
                out_lo: 20.0,
                out_hi: 220.0,
                jpeg_quality: 90,
            };
            let out: Vec<f32> = apply(&raw, w, h, &p, &mut Rng::new(1))
                .iter()
                .map(|&v| v as f32)
                .collect();
            let at = |x: usize, y: usize| out[y * w + x];
            let body = (45..55).map(|i| at(i, 50)).sum::<f32>() / 10.0;
            let ground: Vec<f32> = (20..40).map(|i| at(i, 80)).collect();
            let g = ground.iter().sum::<f32>() / 20.0;
            let sd = (ground.iter().map(|v| (v - g) * (v - g)).sum::<f32>() / 20.0).sqrt();
            (body - g) / sd.max(0.1)
        };
        let (clear, foggy) = (snr(0.0), snr(0.7));
        assert!(foggy < clear * 0.6, "clear {clear}, foggy {foggy}");
        assert_eq!(SensorParams::random(&mut Rng::new(0)).weather(), "clear");
    }

    #[test]
    fn enhance_rims_a_warm_body() {
        // a 20x20 body at 200 on ground at 100
        let (w, h) = (60, 60);
        let img: Vec<f32> = (0..w * h)
            .map(|k| {
                let (x, y) = (k % w, k / w);
                if (20..40).contains(&x) && (20..40).contains(&y) {
                    200.0
                } else {
                    100.0
                }
            })
            .collect();
        let mut p = SensorParams::forest(&mut Rng::new(0));
        (p.detail, p.detail_px, p.equalize) = (1.0, 3.0, 0.0);
        assert_eq!(enhance(&img, w, h, &p), img);
        p.detail = 3.0;
        let e = enhance(&img, w, h, &p);
        let at = |x: usize, y: usize| e[y * w + x];
        // brighter rim than middle, a dark halo outside, flat ground far off unchanged
        assert!(
            at(21, 30) > at(30, 30) + 20.0,
            "{} {}",
            at(21, 30),
            at(30, 30)
        );
        assert!(at(18, 30) < 90.0, "{}", at(18, 30));
        assert!((at(5, 5) - 100.0).abs() < 1e-3);
    }

    #[test]
    fn equalizing_squeezes_the_gap_to_a_small_warm_body() {
        // ground ramping 90-110 across, a 6x6 body at 250
        let (w, h) = (60, 60);
        let img: Vec<f32> = (0..w * h)
            .map(|k| {
                let (x, y) = (k % w, k / w);
                if (27..33).contains(&x) && (27..33).contains(&y) {
                    250.0
                } else {
                    90.0 + 20.0 * x as f32 / w as f32
                }
            })
            .collect();
        let mut p = SensorParams::forest(&mut Rng::new(0));
        (p.detail, p.detail_px, p.equalize) = (1.0, 0.0, 1.0);
        let e = enhance(&img, w, h, &p);
        let ground_max = (0..w * h)
            .filter(|&k| img[k] < 200.0)
            .map(|k| e[k])
            .fold(f32::MIN, f32::max);
        let body = e[30 * w + 30];
        // the body is still the warmest thing, but only just
        assert!(
            body > ground_max && body - ground_max < 0.2 * (250.0 - 110.0),
            "{body} {ground_max}"
        );
        // and the range is kept
        assert!(e.iter().all(|&v| (89.0..=251.0).contains(&v)));
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
