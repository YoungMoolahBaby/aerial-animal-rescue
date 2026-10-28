//! Render N random shots and write them as a dataset pool (images, YOLO labels,
//! metadata.csv) that `scripts/split_dataset.py` reads like any other source.
//!
//!     cargo run --release --bin datagen -- --out ../data/interim/sim_phase1 --frames 20 --seed 0
//!
//! `--bodies assets/bodies` builds the kinds that have baked models (`scripts/bake_bodies.py`)
//! from them instead of primitives; for a seed the shots are the same either way.
//!
//! A window opens while it renders (Bevy's screenshot path needs one) and closes when done.
//! Each shot is captured twice: the thermal look, then the id look (`scene.rs`), each after
//! a few frames so the picture matches the scene. The thermal frame goes through the
//! sensor model and is saved as JPEG; the id frame gives the boxes. `sim_params.csv` next
//! to `metadata.csv` records each image's conditions and camera settings for analysis.
//!
//! Bevy compiles shaders during the first frames and draws nothing until then, so capture
//! starts after a warm-up that shows both looks, and a uniform capture is thrown away and
//! the shot retaken: a black frame with boxes would teach "people in the dark".

use std::fmt::Write as _;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use bevy::prelude::*;
use bevy::render::view::screenshot::{Screenshot, ScreenshotCaptured};
use image::codecs::jpeg::JpegEncoder;
use storm_sim::capture::{PixelBox, PoolWriter, Record, boxes_from_id_mask, id_mask_from_rgb};
use storm_sim::rng::Rng;
use storm_sim::scene::{self, BodyModels, ID_STEP, IMAGE_H, IMAGE_W, POOL, SceneCtl, Shot, World};
use storm_sim::sensor::{self, SensorParams};

const SETTLE_FRAMES: u32 = 4;
const WARMUP_FRAMES: u32 = 60;
const MAX_BLANK: usize = 20;
/// A target needs this many visible pixels to get a box (real labels go down to ~5 px).
const MIN_PIXELS: usize = 6;
/// Pixels added on every side of a box beyond the blur (see `write_shot`).
const LABEL_MARGIN: f32 = 2.0;

type Slot = Arc<Mutex<Option<Image>>>;

#[derive(Clone, Copy, Debug, PartialEq)]
enum Phase {
    Warmup(u32),
    Place,
    Settle(u32, Capture),
    Waiting(Capture),
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Capture {
    Thermal,
    Ids,
}

#[derive(Resource)]
struct Datagen {
    frames: usize,
    rng: Rng,
    writer: Option<PoolWriter>,
    params: String,
    blank: usize,
    index: usize,
    phase: Phase,
    shot: Option<Shot>,
    thermal: Option<Vec<u8>>,
    slot: Slot,
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let get = |name: &str| {
        args.iter()
            .position(|a| a == name)
            .and_then(|i| args.get(i + 1))
            .cloned()
    };
    let out = PathBuf::from(get("--out").expect("--out <pool folder> is required"));
    let frames: usize = get("--frames").map_or(20, |v| v.parse().expect("--frames is a number"));
    let seed: u64 = get("--seed").map_or(0, |v| v.parse().expect("--seed is a number"));
    let writer = PoolWriter::create(&out, "sim").unwrap_or_else(|e| panic!("{e}"));
    let only: Vec<String> = get("--body-kinds")
        .map(|k| k.split(',').map(str::to_string).collect())
        .unwrap_or_default();
    let models = get("--bodies").map_or_else(BodyModels::default, |d| {
        BodyModels::load(std::path::Path::new(&d), &only)
            .unwrap_or_else(|e| panic!("--bodies {d}: {e}"))
    });
    println!("body models: {}", models.summary());

    App::new()
        .add_plugins(DefaultPlugins.set(scene::window_plugin("storm sim: datagen")))
        .insert_resource(World::new(seed))
        .insert_resource(models)
        .insert_resource(Datagen {
            frames,
            rng: Rng::new(seed),
            writer: Some(writer),
            params: String::from(
                "image_id,night,air_c,sun_c,water,texture,vfov_deg,altitude_m,pitch_deg,\
                 targets,boxes,blur,motion_px,interlace_px,video_q,noise,out_noise,blotch,blotch_px,fpn_col,agc_lo,agc_hi,out_lo,out_hi,jpeg,bodies,haze\n",
            ),
            blank: 0,
            index: 0,
            phase: Phase::Warmup(WARMUP_FRAMES),
            shot: None,
            thermal: None,
            slot: Arc::new(Mutex::new(None)),
        })
        .add_systems(Startup, setup)
        .add_systems(Update, drive)
        .run();
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut images: ResMut<Assets<Image>>,
    mut mats: ResMut<Assets<StandardMaterial>>,
    world: Res<World>,
    models: Res<BodyModels>,
) {
    let kinds: Vec<_> = POOL
        .iter()
        .flat_map(|&(k, n)| std::iter::repeat_n(k, n))
        .collect();
    scene::spawn_scene(
        &mut commands,
        &mut meshes,
        &mut images,
        &mut mats,
        &world,
        &kinds,
        &models,
    );
}

fn drive(
    mut commands: Commands,
    mut dg: ResMut<Datagen>,
    world: Res<World>,
    mut ctl: SceneCtl,
    mut exit: MessageWriter<AppExit>,
) {
    if dg.index == dg.frames {
        if let Some(w) = dg.writer.take() {
            let n = w.len();
            let path = w.finish().expect("write metadata.csv");
            let params = path.with_file_name("sim_params.csv");
            std::fs::write(&params, &dg.params).expect("write sim_params.csv");
            info!("wrote {n} images, {}", path.display());
            exit.write(AppExit::Success);
        }
        return;
    }
    assert!(
        dg.blank <= MAX_BLANK,
        "{} blank captures: the scene isn't rendering",
        dg.blank
    );
    match dg.phase {
        Phase::Warmup(0) => {
            ctl.show_ids(false);
            dg.phase = Phase::Place;
        }
        Phase::Warmup(n) => {
            // compile the pipelines of both looks before the first capture
            if n == WARMUP_FRAMES / 2 {
                ctl.show_ids(true);
            }
            dg.phase = Phase::Warmup(n - 1);
        }
        Phase::Place => {
            let kinds = ctl.kinds();
            let shot = Shot::random(&mut dg.rng, &world, &kinds, &ctl.posed());
            ctl.apply(&shot);
            dg.shot = Some(shot);
            dg.phase = Phase::Settle(SETTLE_FRAMES, Capture::Thermal);
        }
        Phase::Settle(0, what) => {
            let slot = dg.slot.clone();
            commands.spawn(Screenshot::primary_window()).observe(
                move |ev: On<ScreenshotCaptured>| {
                    *slot.lock().unwrap() = Some(ev.image.clone());
                },
            );
            dg.phase = Phase::Waiting(what);
        }
        Phase::Settle(n, what) => dg.phase = Phase::Settle(n - 1, what),
        Phase::Waiting(what) => {
            let Some(img) = dg.slot.lock().unwrap().take() else {
                return;
            };
            let rgb = img
                .try_into_dynamic()
                .expect("screenshot to image")
                .to_rgb8();
            // Boxes are in window pixels; the window is the sensor size (scale factor 1).
            assert_eq!(
                (rgb.width(), rgb.height()),
                (IMAGE_W, IMAGE_H),
                "window is not the sensor size"
            );
            let raw = rgb.as_raw();
            // A thermal frame is never uniform; an id frame always has some green.
            let blank = match what {
                Capture::Thermal => raw.iter().all(|&p| p == raw[0]),
                Capture::Ids => raw.iter().all(|&p| p == 0),
            };
            if blank {
                warn!("blank {what:?} capture, retaking the shot");
                dg.blank += 1;
                ctl.show_ids(false);
                dg.phase = Phase::Place;
                return;
            }
            match what {
                Capture::Thermal => {
                    dg.thermal = Some(raw.chunks_exact(3).map(|p| p[0]).collect());
                    ctl.show_ids(true);
                    dg.phase = Phase::Settle(SETTLE_FRAMES, Capture::Ids);
                }
                Capture::Ids => {
                    ctl.show_ids(false);
                    if std::env::var_os("SIM_DEBUG").is_some() {
                        let dir = dg.writer.as_ref().unwrap().image_path("x", "png");
                        let dir = dir.parent().unwrap().with_file_name("debug");
                        std::fs::create_dir_all(&dir).unwrap();
                        rgb.save(dir.join(format!("{}_id.png", dg.index))).unwrap();
                        image::GrayImage::from_raw(IMAGE_W, IMAGE_H, dg.thermal.clone().unwrap())
                            .unwrap()
                            .save(dir.join(format!("{}_raw.png", dg.index)))
                            .unwrap();
                    }
                    let mask = id_mask_from_rgb(raw, ID_STEP);
                    let boxes: Vec<PixelBox> =
                        boxes_from_id_mask(&mask, IMAGE_W as usize, MIN_PIXELS)
                            .into_iter()
                            .filter_map(|(id, b, _)| {
                                Some(PixelBox {
                                    class_id: ctl.class_of(id)?,
                                    x1: b[0] as f32,
                                    y1: b[1] as f32,
                                    x2: b[2] as f32,
                                    y2: b[3] as f32,
                                })
                            })
                            .collect();
                    // what each body in the shot is built from (CC0 models: sim/bodies.csv)
                    let shot = dg.shot.as_ref().expect("a shot was placed");
                    let bodies: Vec<&str> = (1..=shot.targets.len() as u32)
                        .filter_map(|id| ctl.body_of(id, shot))
                        .collect();
                    let bodies = bodies.join(";");
                    write_shot(&mut dg, &world, &boxes, &bodies);
                    dg.phase = Phase::Place;
                }
            }
        }
    }
}

fn write_shot(dg: &mut Datagen, world: &World, boxes: &[PixelBox], bodies: &str) {
    let shot = dg.shot.take().expect("a shot was placed");
    let raw = dg.thermal.take().expect("thermal frame captured");
    // weather from its own stream, so the shots stay as in older pools
    let mut weather = Rng::new(world.seed.wrapping_mul(0x9E37_79B9) ^ (dg.index as u64) << 20);
    let p = SensorParams::random(&mut dg.rng).with_weather(&mut weather);
    let out = sensor::apply(&raw, IMAGE_W as usize, IMAGE_H as usize, &p, &mut dg.rng);
    // Match how the real data is labelled: people boxes there have a median short side of
    // 12-17 px where our tight id-mask boxes have 7 (same lens and altitude ranges), so
    // labellers leave a margin round the visible blob, which the optics blur already
    // widens by about sigma per side.
    let pad = p.blur_sigma.round() + LABEL_MARGIN;
    let boxes: Vec<PixelBox> = boxes
        .iter()
        .map(|b| PixelBox {
            x1: (b.x1 - pad).max(0.0),
            y1: (b.y1 - pad).max(0.0),
            x2: (b.x2 + pad).min(IMAGE_W as f32),
            y2: (b.y2 + pad).min(IMAGE_H as f32),
            ..*b
        })
        .collect();
    let boxes = &boxes[..];
    let image_id = format!("sim_w{}_{:06}", world.seed, dg.index);
    dg.index += 1;
    let c = &shot.conditions;
    let record = Record {
        // procedural world; bodies procedural or CC0 models (sim/bodies.csv, no conditions)
        source_license: "CC-BY-4.0".into(),
        modality: "thermal".into(),
        altitude_m: Some(shot.altitude_m),
        weather: p.weather().into(),
        scene_type: if c.water.is_some() { "flood" } else { "other" }.into(),
        sequence_id: format!("sim-world-{}", world.seed),
    };
    let w = dg.writer.as_mut().expect("writer open");
    let file = std::fs::File::create(w.image_path(&image_id, "jpg")).expect("create image");
    JpegEncoder::new_with_quality(std::io::BufWriter::new(file), p.jpeg_quality)
        .encode(&out, IMAGE_W, IMAGE_H, image::ExtendedColorType::L8)
        .expect("encode JPEG");
    w.add(&image_id, "jpg", IMAGE_W, IMAGE_H, boxes, &record)
        .expect("write label");
    writeln!(
        dg.params,
        "{image_id},{},{:.1},{:.1},{},{:.3},{:.1},{:.1},{:.1},{},{},{:.2},{:.2},{:.1},{},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2},{:.0},{:.0},{},{},{:.2}",
        c.night,
        c.air_c,
        c.sun_c,
        c.water.is_some(),
        c.texture,
        shot.vfov_deg,
        shot.altitude_m,
        shot.pitch_deg,
        shot.n_targets(),
        boxes.len(),
        p.blur_sigma,
        p.motion_px,
        p.interlace_px,
        p.video_quality.map_or(0, |q| q),
        p.noise,
        p.out_noise,
        p.blotch,
        p.blotch_px,
        p.fpn_col,
        p.agc_lo_pct,
        p.agc_hi_pct,
        p.out_lo,
        p.out_hi,
        p.jpeg_quality,
        bodies,
        p.haze,
    )
    .unwrap();
}
