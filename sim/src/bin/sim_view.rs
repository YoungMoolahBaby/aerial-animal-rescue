//! Interactive viewer: the aerial camera's temperature render (before the sensor model).
//! Space draws a new random shot and prints its settings; I toggles the id look (each
//! target in its own shade of red, what the labels are made from).
//!
//!     cargo run --release --bin sim_view [-- --seed 1 --bodies assets/bodies]

use bevy::prelude::*;
use storm_sim::rng::Rng;
use storm_sim::scene::{self, BodyModels, POOL, SceneCtl, Shot, World};

#[derive(Resource)]
struct Shots(Rng);

fn main() {
    let seed = arg("--seed").and_then(|v| v.parse().ok()).unwrap_or(0);
    let only: Vec<String> = arg("--body-kinds")
        .map(|k| k.split(',').map(str::to_string).collect())
        .unwrap_or_default();
    let models = arg("--bodies").map_or_else(BodyModels::default, |d| {
        BodyModels::load(std::path::Path::new(&d), &only)
            .unwrap_or_else(|e| panic!("--bodies {d}: {e}"))
    });
    println!("body models: {}", models.summary());
    App::new()
        .add_plugins(DefaultPlugins.set(scene::window_plugin("storm sim: viewer")))
        .insert_resource(World::new(seed))
        .insert_resource(models)
        .insert_resource(Shots(Rng::new(seed)))
        .add_systems(Startup, setup)
        .add_systems(Update, controls)
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

fn controls(
    keys: Res<ButtonInput<KeyCode>>,
    mut started: Local<bool>,
    mut ids: Local<bool>,
    mut shots: ResMut<Shots>,
    world: Res<World>,
    mut ctl: SceneCtl,
) {
    if keys.just_pressed(KeyCode::KeyI) {
        *ids = !*ids;
        ctl.show_ids(*ids);
    }
    if *started && !keys.just_pressed(KeyCode::Space) {
        return;
    }
    *started = true;
    let kinds = ctl.kinds();
    let shot = Shot::random(&mut shots.0, &world, &kinds, &ctl.posed(), 0.0);
    let c = &shot.conditions;
    info!(
        "{} air {:.0} C{}, lens {:.0} deg, altitude {:.0} m, angle {:.0} deg, {} targets",
        if c.night { "night" } else { "day" },
        c.air_c,
        if c.water.is_some() { ", flooded" } else { "" },
        shot.vfov_deg,
        shot.altitude_m,
        shot.pitch_deg,
        shot.n_targets(),
    );
    ctl.apply(&shot);
}

fn arg(name: &str) -> Option<String> {
    let args: Vec<String> = std::env::args().collect();
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1))
        .cloned()
}
