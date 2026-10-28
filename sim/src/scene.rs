//! The Bevy side: terrain mesh, targets (people and animals), the aerial camera, and the
//! projection of targets to pixel boxes.
//!
//! Phase 0 renders a plain grey world with warm (white) people, unlit, so the image is
//! already thermal-like; the thermal sensor model comes in phase 1 (`sim/PLAN.md`).

use bevy::asset::RenderAssetUsages;
use bevy::core_pipeline::tonemapping::Tonemapping;
use bevy::mesh::{Indices, PrimitiveTopology};
use bevy::prelude::*;

use crate::capture::PixelBox;
use crate::rng::Rng;
use crate::world::Terrain;

/// Frame size of the thermal cameras in our real data (HIT-UAV, most BIRDSAI flights).
pub const IMAGE_W: u32 = 640;
pub const IMAGE_H: u32 = 512;
/// Vertical field of view, about a 640x512 thermal core with a 13 mm lens.
pub const VFOV_DEG: f32 = 37.0;
/// Our class ids (`configs/classes.yaml`).
pub const PERSON: u32 = 0;

#[derive(Resource)]
pub struct World {
    pub terrain: Terrain,
    pub seed: u64,
}

#[derive(Component)]
pub struct AerialCamera;

/// Something we label: its class and its size (full extents, metres).
#[derive(Component)]
pub struct Target {
    pub class_id: u32,
    pub size: Vec3,
}

/// One random camera pose and target layout, drawn from the ranges of the real data.
#[derive(Clone, Debug)]
pub struct Shot {
    /// Height of the camera above the ground it looks at, metres (real: 60-130).
    pub altitude_m: f64,
    /// Camera angle below the horizon, degrees: 90 looks straight down (real: 30-90).
    pub pitch_deg: f64,
    pub camera: Transform,
    pub targets: Vec<Vec3>,
}

impl Shot {
    pub fn random(rng: &mut Rng, terrain: &Terrain, n_targets: usize) -> Self {
        let aim = terrain.random_spot(rng, 150.0);
        let altitude_m = rng.range(60.0, 130.0);
        let pitch_deg = rng.range(30.0, 90.0);
        let yaw = rng.range(0.0, std::f64::consts::TAU);
        let back = altitude_m / pitch_deg.to_radians().tan();
        let (dir_x, dir_z) = (yaw.cos(), yaw.sin());
        let eye = Vec3::new(
            (aim[0] - dir_x * back) as f32,
            (aim[1] + altitude_m) as f32,
            (aim[2] - dir_z * back) as f32,
        );
        let target = Vec3::new(aim[0] as f32, aim[1] as f32, aim[2] as f32);
        // Looking straight down, "up" in the picture is the flight direction.
        let up = if pitch_deg > 85.0 {
            Vec3::new(dir_x as f32, 0.0, dir_z as f32)
        } else {
            Vec3::Y
        };
        let camera = Transform::from_translation(eye).looking_at(target, up);
        // Targets near the aim point, inside the picture: the ground footprint is at
        // least altitude * tan(fov/2) across.
        let spread = altitude_m * (VFOV_DEG as f64 / 2.0).to_radians().tan() * 0.8;
        let targets = (0..n_targets)
            .map(|_| {
                let (x, z) = (
                    aim[0] + rng.range(-spread, spread),
                    aim[2] + rng.range(-spread, spread),
                );
                Vec3::new(x as f32, terrain.height(x, z) as f32, z as f32)
            })
            .collect();
        Self {
            altitude_m,
            pitch_deg,
            camera,
            targets,
        }
    }
}

pub fn window_plugin(title: &str) -> WindowPlugin {
    WindowPlugin {
        primary_window: Some(Window {
            title: title.into(),
            // Exactly the sensor size, whatever the desktop scaling.
            resolution: bevy::window::WindowResolution::new(IMAGE_W, IMAGE_H)
                .with_scale_factor_override(1.0),
            resizable: false,
            ..default()
        }),
        ..default()
    }
}

/// Spawn terrain, camera and `n_targets` people. Positions are set by `apply_shot`.
pub fn spawn_scene(
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
    mats: &mut Assets<StandardMaterial>,
    terrain: &Terrain,
    n_targets: usize,
) {
    commands.spawn((
        Mesh3d(meshes.add(terrain_mesh(terrain))),
        MeshMaterial3d(mats.add(StandardMaterial {
            base_color: Color::srgb(0.35, 0.35, 0.35),
            unlit: true,
            cull_mode: None,
            ..default()
        })),
    ));
    commands.spawn((
        Camera3d::default(),
        Camera {
            clear_color: ClearColorConfig::Custom(Color::BLACK),
            ..default()
        },
        Projection::Perspective(PerspectiveProjection {
            fov: VFOV_DEG.to_radians(),
            near: 1.0,
            far: 3000.0,
            ..default()
        }),
        Tonemapping::None,
        AerialCamera,
    ));
    // A person: 0.5 m wide, 1.7 m tall, warm.
    let size = Vec3::new(0.5, 1.7, 0.5);
    let body = meshes.add(Capsule3d::new(size.x / 2.0, size.y - size.x));
    let warm = mats.add(StandardMaterial {
        base_color: Color::WHITE,
        unlit: true,
        ..default()
    });
    for _ in 0..n_targets {
        commands.spawn((
            Mesh3d(body.clone()),
            MeshMaterial3d(warm.clone()),
            Transform::default(),
            Target {
                class_id: PERSON,
                size,
            },
        ));
    }
}

/// Move the camera and targets to `shot`.
pub fn apply_shot(
    shot: &Shot,
    camera: &mut Transform,
    targets: &mut Query<(&mut Transform, &Target), Without<AerialCamera>>,
) {
    *camera = shot.camera;
    for ((mut tf, target), feet) in targets.iter_mut().zip(&shot.targets) {
        tf.translation = *feet + Vec3::Y * target.size.y / 2.0;
    }
}

/// Pixel box of a target from the corners of its bounding box, or `None` if it is off
/// screen, too small, or hidden behind the terrain. Phase 0 uses corners (slightly loose
/// boxes); phase 1 switches to an object-id mask for tight boxes under occlusion.
pub fn project_target(
    camera: &Camera,
    camera_tf: &GlobalTransform,
    target_tf: &Transform,
    target: &Target,
    terrain: &Terrain,
) -> Option<PixelBox> {
    let eye = camera_tf.translation();
    let c = target_tf.translation;
    let a = [eye.x as f64, eye.y as f64, eye.z as f64];
    if !terrain.line_of_sight(a, [c.x as f64, c.y as f64, c.z as f64], 0.5) {
        return None;
    }
    let h = target.size / 2.0;
    let mut points = Vec::with_capacity(8);
    for sx in [-1.0, 1.0] {
        for sy in [-1.0, 1.0] {
            for sz in [-1.0, 1.0] {
                let p = camera
                    .world_to_viewport(camera_tf, c + h * Vec3::new(sx, sy, sz))
                    .ok()?;
                points.push((p.x, p.y));
            }
        }
    }
    PixelBox::from_points(target.class_id, points, IMAGE_W, IMAGE_H, 1.0)
}

/// Bevy mesh of the terrain grid, with normals from CortenForge's height field.
pub fn terrain_mesh(terrain: &Terrain) -> Mesh {
    let (verts, w, d) = terrain.vertices();
    let pos: Vec<[f32; 3]> = verts
        .iter()
        .map(|p| [p[0] as f32, p[1] as f32, p[2] as f32])
        .collect();
    let normals: Vec<[f32; 3]> = verts
        .iter()
        .map(|p| {
            let n = terrain.normal(p[0], p[2]);
            [n[0] as f32, n[1] as f32, n[2] as f32]
        })
        .collect();
    let mut idx = Vec::with_capacity((w - 1) * (d - 1) * 6);
    for j in 0..d - 1 {
        for i in 0..w - 1 {
            let k = (j * w + i) as u32;
            let (r, b) = (k + 1, k + w as u32);
            idx.extend_from_slice(&[k, b, r, r, b, b + 1]);
        }
    }
    Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::default(),
    )
    .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, pos)
    .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, normals)
    .with_inserted_indices(Indices::U32(idx))
}
