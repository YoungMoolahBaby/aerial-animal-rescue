//! Render N random shots and write them as a dataset pool (images, YOLO labels,
//! metadata.csv) that `scripts/split_dataset.py` reads like any other source.
//!
//!     cargo run --release --bin datagen -- --out ../data/interim/sim_phase0 --frames 20 --seed 0
//!
//! A window opens while it renders (Bevy's screenshot path needs one) and closes when done.
//! Every shot waits a few frames after moving the camera, so the picture and the boxes
//! come from the same scene. Bevy compiles shaders during the first frames and draws
//! nothing until then, so capture starts after a warm-up, and a blank (uniform) capture is
//! thrown away and retaken: a black frame with boxes would teach "people in the dark".

use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use bevy::prelude::*;
use bevy::render::view::screenshot::{Screenshot, ScreenshotCaptured};
use storm_sim::capture::{PixelBox, PoolWriter, Record};
use storm_sim::rng::Rng;
use storm_sim::scene::{self, AerialCamera, IMAGE_H, IMAGE_W, Shot, Target, World};
use storm_sim::world::Terrain;

const TARGETS: usize = 3;
const SETTLE_FRAMES: u32 = 4;
const WARMUP_FRAMES: u32 = 60;
const MAX_BLANK: usize = 20;

#[derive(Resource)]
struct Datagen {
    frames: usize,
    rng: Rng,
    writer: Arc<Mutex<Option<PoolWriter>>>,
    saved: Arc<AtomicUsize>,
    blank: Arc<AtomicUsize>,
    requested: usize,
    settle: u32,
    shot: Option<Shot>,
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

    App::new()
        .add_plugins(DefaultPlugins.set(scene::window_plugin("storm sim: datagen")))
        .insert_resource(World {
            terrain: Terrain::generate(seed, 600.0, 2.0, 12.0),
            seed,
        })
        .insert_resource(Datagen {
            frames,
            rng: Rng::new(seed),
            writer: Arc::new(Mutex::new(Some(writer))),
            saved: Arc::new(AtomicUsize::new(0)),
            blank: Arc::new(AtomicUsize::new(0)),
            requested: 0,
            settle: WARMUP_FRAMES,
            shot: None,
        })
        .add_systems(Startup, setup)
        .add_systems(Update, drive)
        .run();
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut mats: ResMut<Assets<StandardMaterial>>,
    world: Res<World>,
) {
    scene::spawn_scene(
        &mut commands,
        &mut meshes,
        &mut mats,
        &world.terrain,
        TARGETS,
    );
}

#[allow(clippy::too_many_arguments)]
fn drive(
    mut commands: Commands,
    mut dg: ResMut<Datagen>,
    world: Res<World>,
    mut camera_tf: Single<&mut Transform, With<AerialCamera>>,
    camera: Single<(&Camera, &GlobalTransform), With<AerialCamera>>,
    mut targets: Query<(&mut Transform, &Target), Without<AerialCamera>>,
    mut exit: MessageWriter<AppExit>,
) {
    let saved = dg.saved.load(Ordering::SeqCst);
    if saved == dg.frames {
        let writer = dg.writer.lock().unwrap().take();
        if let Some(w) = writer {
            let n = w.len();
            let path = w.finish().expect("write metadata.csv");
            info!("wrote {n} images, {}", path.display());
            exit.write(AppExit::Success);
        }
        return;
    }
    let blank = dg.blank.load(Ordering::SeqCst);
    assert!(
        blank <= MAX_BLANK,
        "{blank} blank captures: the scene isn't rendering"
    );
    if dg.requested > saved + blank {
        return; // waiting for the last screenshot
    }
    if dg.settle > 0 {
        dg.settle -= 1; // warm-up, or the new shot settling
        return;
    }
    // New shot: move everything, then let the transforms settle before capturing.
    if dg.shot.is_none() {
        let shot = Shot::random(&mut dg.rng, &world.terrain, TARGETS);
        scene::apply_shot(&shot, &mut camera_tf, &mut targets);
        dg.shot = Some(shot);
        dg.settle = SETTLE_FRAMES;
        return;
    }
    let shot = dg.shot.take().unwrap();
    let (cam, cam_global) = *camera;
    let boxes: Vec<PixelBox> = targets
        .iter()
        .filter_map(|(tf, t)| scene::project_target(cam, cam_global, tf, t, &world.terrain))
        .collect();
    let index = dg.requested;
    dg.requested += 1;
    let image_id = format!("sim_w{}_{:06}", world.seed, index); // ids skip retaken shots
    let record = Record {
        source_license: "CC-BY-4.0".into(), // procedural world, no third-party assets yet
        modality: "thermal".into(),
        altitude_m: Some(shot.altitude_m),
        weather: "unknown".into(),
        scene_type: "other".into(),
        sequence_id: format!("sim-world-{}", world.seed),
    };
    let (writer, saved, blank) = (dg.writer.clone(), dg.saved.clone(), dg.blank.clone());
    commands
        .spawn(Screenshot::primary_window())
        .observe(move |ev: On<ScreenshotCaptured>| {
            let img = ev
                .image
                .clone()
                .try_into_dynamic()
                .expect("screenshot to image")
                .to_luma8();
            // Boxes are in window pixels; the window is the sensor size (scale factor 1).
            assert_eq!(
                (img.width(), img.height()),
                (IMAGE_W, IMAGE_H),
                "window is not the sensor size"
            );
            let first = img.as_raw()[0];
            if img.as_raw().iter().all(|&p| p == first) {
                warn!("blank capture {image_id}, retaking");
                blank.fetch_add(1, Ordering::SeqCst);
                return;
            }
            let mut guard = writer.lock().unwrap();
            let w = guard.as_mut().expect("writer open");
            img.save(w.image_path(&image_id, "png"))
                .expect("save image");
            w.add(&image_id, "png", IMAGE_W, IMAGE_H, &boxes, &record)
                .expect("write label");
            saved.fetch_add(1, Ordering::SeqCst);
        });
}
