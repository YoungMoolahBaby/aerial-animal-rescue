//! Render N random shots and write them as a dataset pool (images, YOLO labels,
//! metadata.csv) that `scripts/split_dataset.py` reads like any other source.
//!
//!     cargo run --release --bin datagen -- --out ../data/interim/sim_phase1 --frames 20 --seed 0
//!
//! `--modality rgb` renders daylight colour photos (1280x960, farm pool, `rgb.rs`) instead
//! of thermal frames (the default). `--modality forest` renders thermal frames like BAMBI's
//! (straight down over painted forest ground, `forest.rs`; deer, boar, people, dogs), saved
//! as BAMBI stores them: 1024x1024 with black bands above and below.
//! `--faint-share 0.4` makes the bodies faint in that share of shots (v7; default 0).
//! `--bodies assets/bodies` builds the kinds that have baked models (`scripts/bake_bodies.py`)
//! from them instead of primitives; for a seed the shots are the same either way.
//!
//! A window opens while it renders (Bevy's screenshot path needs one) and closes when done.
//! Each shot is captured twice: the picture (thermal or RGB), then the id look
//! (`scene.rs`), each after a few frames so the picture matches the scene. The picture
//! goes through its camera model and is saved as JPEG; the id frame gives the boxes.
//! `sim_params.csv` next to `metadata.csv` records each image's conditions and camera
//! settings for analysis.
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
use storm_sim::rgb::{self, RgbParams};
use storm_sim::rng::Rng;
use storm_sim::scene::{
    self, BodyModels, ClutterKind, ID_STEP, IMAGE_H, IMAGE_W, Modality, SceneCtl, Shot, World,
};
use storm_sim::sensor::{self, SensorParams};

const SETTLE_FRAMES: u32 = 4;
const WARMUP_FRAMES: u32 = 60;
const MAX_BLANK: usize = 20;
/// A target needs this many visible pixels to get a box (real labels go down to ~5 px).
const MIN_PIXELS: usize = 6;
/// Pixels added on every side of a box beyond the blur (see `write_shot`).
const LABEL_MARGIN: f32 = 2.0;
/// The real RGB cow boxes hug the animal (sharp 5 K photos), so one pixel at 1280.
const RGB_LABEL_MARGIN: f32 = 1.0;
/// BAMBI's boxes are drawn by hand with a wide margin: 5-7 px per side of 1024 beyond
/// what a model trained on 1 px margins puts round the animal, about the same at every
/// size (measured on BAMBI train, 2026-11-06), so about 4 render pixels more than that.
const FOREST_LABEL_MARGIN: f32 = 5.0;
const THERMAL_PARAMS: &str = "image_id,night,air_c,sun_c,water,texture,vfov_deg,altitude_m,pitch_deg,\
    targets,boxes,blur,motion_px,interlace_px,video_q,noise,out_noise,blotch,blotch_px,fpn_col,agc_lo,agc_hi,out_lo,out_hi,jpeg,bodies,haze,faint\n";
const FOREST_PARAMS: &str = "image_id,night,sun,covers,trees,debris,man_made,vfov_deg,altitude_m,pitch_deg,    targets,boxes,blur,motion_px,video_q,noise,out_noise,blotch,agc_lo,agc_hi,out_lo,out_hi,jpeg,window_lo,window_hi,detail,detail_px,equalize
";
const RGB_PARAMS: &str = "image_id,field,sun,sky,sun_elev_deg,vfov_deg,altitude_m,pitch_deg,targets,boxes,\
    bales,yard_sheds,yard_targets,wb_r,wb_b,median,vignette,circle,blur,haze,contrast,saturation,noise,jpeg,bodies\n";

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
    Picture,
    Ids,
}

#[derive(Resource)]
struct Datagen {
    frames: usize,
    faint_share: f64,
    modality: Modality,
    rng: Rng,
    writer: Option<PoolWriter>,
    params: String,
    blank: usize,
    index: usize,
    phase: Phase,
    shot: Option<Shot>,
    /// The captured picture: grey for thermal, RGB (3 bytes a pixel) for RGB.
    picture: Option<Vec<u8>>,
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
    let faint_share: f64 =
        get("--faint-share").map_or(0.0, |v| v.parse().expect("--faint-share is a number"));
    let modality = get("--modality").map_or(Modality::Thermal, |m| {
        Modality::parse(&m).expect("--modality is thermal, rgb or forest")
    });
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
        .add_plugins(DefaultPlugins.set(scene::window_plugin("storm sim: datagen", modality)))
        .insert_resource(World::new(seed))
        .insert_resource(models)
        .insert_resource(modality)
        .insert_resource(Datagen {
            frames,
            faint_share,
            modality,
            rng: Rng::new(seed),
            writer: Some(writer),
            params: String::from(match modality {
                Modality::Thermal => THERMAL_PARAMS,
                Modality::Rgb => RGB_PARAMS,
                Modality::Forest => FOREST_PARAMS,
            }),
            blank: 0,
            index: 0,
            phase: Phase::Warmup(WARMUP_FRAMES),
            shot: None,
            picture: None,
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
    modality: Res<Modality>,
) {
    scene::spawn_scene(
        &mut commands,
        &mut meshes,
        &mut images,
        &mut mats,
        &world,
        &modality.kinds(),
        &models,
        *modality,
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
            let (share, modality) = (dg.faint_share, dg.modality);
            let shot = Shot::random(&mut dg.rng, &world, &kinds, &ctl.posed(), share, modality);
            ctl.apply(&shot);
            dg.shot = Some(shot);
            dg.phase = Phase::Settle(SETTLE_FRAMES, Capture::Picture);
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
            let (width, height) = dg.modality.size();
            assert_eq!(
                (rgb.width(), rgb.height()),
                (width, height),
                "window is not the sensor size"
            );
            let raw = rgb.as_raw();
            // A picture is never uniform; an id frame always has some green.
            let blank = match what {
                Capture::Picture => raw.iter().all(|&p| p == raw[0]),
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
                Capture::Picture => {
                    dg.picture = Some(match dg.modality {
                        Modality::Thermal | Modality::Forest => {
                            raw.chunks_exact(3).map(|p| p[0]).collect()
                        }
                        Modality::Rgb => raw.clone(),
                    });
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
                        let picture = dg.picture.clone().unwrap();
                        let path = dir.join(format!("{}_raw.png", dg.index));
                        match dg.modality {
                            Modality::Thermal | Modality::Forest => {
                                image::GrayImage::from_raw(width, height, picture)
                                    .unwrap()
                                    .save(path)
                            }
                            Modality::Rgb => image::RgbImage::from_raw(width, height, picture)
                                .unwrap()
                                .save(path),
                        }
                        .unwrap();
                    }
                    let mask = id_mask_from_rgb(raw, ID_STEP);
                    let boxes: Vec<PixelBox> =
                        boxes_from_id_mask(&mask, width as usize, MIN_PIXELS)
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

/// Boxes grown by `pad` pixels on every side, clipped to the frame.
fn padded(boxes: &[PixelBox], pad: f32, width: u32, height: u32) -> Vec<PixelBox> {
    boxes
        .iter()
        .map(|b| PixelBox {
            x1: (b.x1 - pad).max(0.0),
            y1: (b.y1 - pad).max(0.0),
            x2: (b.x2 + pad).min(width as f32),
            y2: (b.y2 + pad).min(height as f32),
            ..*b
        })
        .collect()
}

fn write_shot(dg: &mut Datagen, world: &World, boxes: &[PixelBox], bodies: &str) {
    match dg.modality {
        Modality::Rgb => return write_rgb_shot(dg, world, boxes, bodies),
        Modality::Forest => return write_forest_shot(dg, world, boxes),
        Modality::Thermal => {}
    }
    let shot = dg.shot.take().expect("a shot was placed");
    let raw = dg.picture.take().expect("thermal frame captured");
    // weather from its own stream, so the shots stay as in older pools
    let mut weather = Rng::new(world.seed.wrapping_mul(0x9E37_79B9) ^ (dg.index as u64) << 20);
    let p = SensorParams::random(&mut dg.rng).with_weather(&mut weather);
    let out = sensor::apply(&raw, IMAGE_W as usize, IMAGE_H as usize, &p, &mut dg.rng);
    // Match how the real data is labelled: people boxes there have a median short side of
    // 12-17 px where our tight id-mask boxes have 7 (same lens and altitude ranges), so
    // labellers leave a margin round the visible blob, which the optics blur already
    // widens by about sigma per side.
    let pad = p.blur_sigma.round() + LABEL_MARGIN;
    let boxes = &padded(boxes, pad, IMAGE_W, IMAGE_H)[..];
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
        "{image_id},{},{:.1},{:.1},{},{:.3},{:.1},{:.1},{:.1},{},{},{:.2},{:.2},{:.1},{},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2},{:.0},{:.0},{},{},{:.2},{}",
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
        shot.faint,
    )
    .unwrap();
}

fn write_forest_shot(dg: &mut Datagen, world: &World, boxes: &[PixelBox]) {
    let shot = dg.shot.take().expect("a shot was placed");
    let raw = dg.picture.take().expect("thermal frame captured");
    let look = shot.forest.as_ref().expect("a forest shot");
    let p = SensorParams::forest(&mut dg.rng);
    let (w, h) = (IMAGE_W as usize, IMAGE_H as usize);
    let out = sensor::bambi_frame(&sensor::apply(&raw, w, h, &p, &mut dg.rng), w, h);
    let pad = p.blur_sigma.round() + FOREST_LABEL_MARGIN;
    let n = sensor::BAMBI_SIZE as u32;
    let boxes: Vec<PixelBox> = padded(boxes, pad, IMAGE_W, IMAGE_H)
        .iter()
        .map(|b| {
            let [x1, y1, x2, y2] = sensor::bambi_box([b.x1, b.y1, b.x2, b.y2], w, h);
            PixelBox {
                x1,
                y1,
                x2,
                y2,
                ..*b
            }
        })
        .collect();
    let image_id = format!("sim_forest_w{}_{:06}", world.seed, dg.index);
    dg.index += 1;
    let c = &shot.conditions;
    let record = Record {
        // procedural world and primitive bodies
        source_license: "CC-BY-4.0".into(),
        modality: "thermal".into(),
        altitude_m: Some(shot.altitude_m),
        weather: "clear".into(),
        scene_type: "forest".into(),
        sequence_id: format!("sim-forest-world-{}", world.seed),
    };
    let wr = dg.writer.as_mut().expect("writer open");
    let file = std::fs::File::create(wr.image_path(&image_id, "jpg")).expect("create image");
    JpegEncoder::new_with_quality(std::io::BufWriter::new(file), p.jpeg_quality)
        .encode(&out, n, n, image::ExtendedColorType::L8)
        .expect("encode JPEG");
    wr.add(&image_id, "jpg", n, n, &boxes, &record)
        .expect("write label");
    let covers = if look.covers[0] == look.covers[1] {
        look.covers[0].name().to_string()
    } else {
        format!("{}+{}", look.covers[0].name(), look.covers[1].name())
    };
    writeln!(
        dg.params,
        "{image_id},{},{},{covers},{},{},{},{:.1},{:.1},{:.1},{},{},{:.2},{:.2},{},{:.2},{:.2},{:.2},{:.2},{:.2},{:.0},{:.0},{},{:.1},{:.1},{:.2},{:.1},{:.2}",
        c.night,
        c.sun_dir[1] < 0.999,
        look.n_trees,
        look.n_debris,
        look.n_man_made,
        shot.vfov_deg,
        shot.altitude_m,
        shot.pitch_deg,
        shot.n_targets(),
        boxes.len(),
        p.blur_sigma,
        p.motion_px,
        p.video_quality.map_or(0, |q| q),
        p.noise,
        p.out_noise,
        p.blotch,
        p.agc_lo_pct,
        p.agc_hi_pct,
        p.out_lo,
        p.out_hi,
        p.jpeg_quality,
        shot.window.lo,
        shot.window.hi,
        p.detail,
        p.detail_px,
        p.equalize,
    )
    .unwrap();
}

fn write_rgb_shot(dg: &mut Datagen, world: &World, boxes: &[PixelBox], bodies: &str) {
    let (width, height) = Modality::Rgb.size();
    let shot = dg.shot.take().expect("a shot was placed");
    let raw = dg.picture.take().expect("RGB frame captured");
    let look = shot.rgb.as_ref().expect("an RGB shot");
    let yard = look.ground.layout.yard.as_ref();
    let p = RgbParams::random(&mut dg.rng);
    let out = rgb::apply(&raw, width as usize, height as usize, &p, &mut dg.rng);
    let pad = p.blur_sigma.round() + RGB_LABEL_MARGIN;
    let boxes = &padded(boxes, pad, width, height)[..];
    let image_id = format!("sim_rgb_w{}_{:06}", world.seed, dg.index);
    dg.index += 1;
    let record = Record {
        // procedural world and bodies (CC0 models if --bodies: sim/bodies.csv)
        source_license: "CC-BY-4.0".into(),
        modality: "rgb".into(),
        altitude_m: Some(shot.altitude_m),
        weather: "clear".into(),
        scene_type: if yard.is_some() {
            "farmyard"
        } else {
            "pasture"
        }
        .into(),
        sequence_id: format!("sim-rgb-world-{}", world.seed),
    };
    let w = dg.writer.as_mut().expect("writer open");
    let file = std::fs::File::create(w.image_path(&image_id, "jpg")).expect("create image");
    JpegEncoder::new_with_quality(std::io::BufWriter::new(file), p.jpeg_quality)
        .encode(&out, width, height, image::ExtendedColorType::Rgb8)
        .expect("encode JPEG");
    w.add(&image_id, "jpg", width, height, boxes, &record)
        .expect("write label");
    let d = &look.daylight;
    let bales = shot.clutter[Modality::Rgb.clutter_index(ClutterKind::Bale)]
        .iter()
        .flatten()
        .count();
    // bodies standing in the yard (in a pen, an open shed or on its mud)
    let in_yard = yard.map_or(0, |y| {
        shot.targets
            .iter()
            .flatten()
            .filter(|p| {
                let t = p.transform.translation;
                y.holds(t.x as f64, t.z as f64)
            })
            .count()
    });
    writeln!(
        dg.params,
        "{image_id},{},{:.2},{:.2},{:.0},{:.1},{:.1},{:.1},{},{},{},{},{},{:.3},{:.3},{:.0},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2},{},{}",
        look.ground.field.name(),
        d.sun,
        d.sky,
        d.sun_dir[1].asin().to_degrees(),
        shot.vfov_deg,
        shot.altitude_m,
        shot.pitch_deg,
        shot.n_targets(),
        boxes.len(),
        bales,
        yard.map_or(0, |y| y.sheds.len()),
        in_yard,
        p.wb[0],
        p.wb[2],
        p.median,
        p.vignette,
        p.circle,
        p.blur_sigma,
        p.haze,
        p.contrast,
        p.saturation,
        p.noise,
        p.jpeg_quality,
        bodies,
    )
    .unwrap();
}
