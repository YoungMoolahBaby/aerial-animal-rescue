//! Interactive viewer: the aerial camera's picture, with each target's label box drawn as
//! a wireframe. Space draws a new random shot; the console prints its altitude, angle and
//! boxes.
//!
//!     cargo run --release --bin sim_view [-- --seed 1]

use bevy::prelude::*;
use storm_sim::rng::Rng;
use storm_sim::scene::{self, AerialCamera, Shot, Target, World};
use storm_sim::world::Terrain;

#[derive(Resource)]
struct Shots(Rng);

fn main() {
    let seed = arg("--seed").unwrap_or(0);
    App::new()
        .add_plugins(DefaultPlugins.set(scene::window_plugin("storm sim: viewer")))
        .insert_resource(World {
            terrain: Terrain::generate(seed, 600.0, 2.0, 12.0),
            seed,
        })
        .insert_resource(Shots(Rng::new(seed)))
        .add_systems(Startup, setup)
        .add_systems(Update, (next_shot, draw_boxes))
        .run();
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut mats: ResMut<Assets<StandardMaterial>>,
    world: Res<World>,
) {
    scene::spawn_scene(&mut commands, &mut meshes, &mut mats, &world.terrain, 3);
}

fn next_shot(
    keys: Res<ButtonInput<KeyCode>>,
    mut first: Local<bool>,
    mut shots: ResMut<Shots>,
    world: Res<World>,
    mut camera: Single<&mut Transform, With<AerialCamera>>,
    mut targets: Query<(&mut Transform, &Target), Without<AerialCamera>>,
) {
    if *first && !keys.just_pressed(KeyCode::Space) {
        return;
    }
    *first = true;
    let shot = Shot::random(&mut shots.0, &world.terrain, 3);
    info!(
        "altitude {:.0} m, angle {:.0} deg",
        shot.altitude_m, shot.pitch_deg
    );
    scene::apply_shot(&shot, &mut camera, &mut targets);
}

fn draw_boxes(
    mut gizmos: Gizmos,
    camera: Single<(&Camera, &GlobalTransform), With<AerialCamera>>,
    targets: Query<(&Transform, &Target), Without<AerialCamera>>,
    world: Res<World>,
) {
    let (cam, cam_tf) = *camera;
    for (tf, target) in &targets {
        let seen = scene::project_target(cam, cam_tf, tf, target, &world.terrain).is_some();
        let color = if seen {
            Color::srgb(0.0, 1.0, 0.0)
        } else {
            Color::srgb(1.0, 0.0, 0.0)
        };
        gizmos.cube(
            Transform::from_translation(tf.translation).with_scale(target.size),
            color,
        );
    }
}

fn arg(name: &str) -> Option<u64> {
    let args: Vec<String> = std::env::args().collect();
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1))
        .and_then(|v| v.parse().ok())
}
