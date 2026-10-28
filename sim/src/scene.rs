//! The Bevy side: terrain, roads, flood water, vegetation, rocks, houses and cars, the bodies
//! we label (people and animals), the aerial camera, and two looks for every mesh:
//!
//! - **thermal**: unlit grey = temperature through the shot's window (`thermal.rs`); the
//!   sensor model runs on the captured frame afterwards (`sensor.rs`);
//! - **id**: target `k` in red `k * ID_STEP`, everything else green with red 0. Captured
//!   right after the thermal frame, it gives a visible-pixel mask per target, so boxes are
//!   tight and hidden targets (behind a hill, a tree, under water) get no box.
//!
//! Bodies are built from primitives (capsules, spheres, cylinders): at 5-30 px the outline
//! and size matter, not the detail, and there are no third-party asset licenses to track.

use std::f32::consts::FRAC_PI_2;

use bevy::asset::RenderAssetUsages;
use bevy::core_pipeline::tonemapping::{DebandDither, Tonemapping};
use bevy::ecs::system::SystemParam;
use bevy::image::{ImageAddressMode, ImageSampler, ImageSamplerDescriptor};
use bevy::mesh::{Indices, PrimitiveTopology};
use bevy::prelude::*;
use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat};
use bevy::render::view::Msaa;

use crate::rng::Rng;
use crate::thermal::{Conditions, TempWindow, ground_texture, texture_gain};
use crate::world::Terrain;

/// Frame size of the thermal cameras in our real data (HIT-UAV, most BIRDSAI flights).
pub const IMAGE_W: u32 = 640;
pub const IMAGE_H: u32 = 512;
/// Vertical field of view range: 640x512 cores with 13-25 mm lenses (HIT-UAV's 25 mm is
/// about 20 deg).
pub const VFOV_DEG: (f64, f64) = (12.0, 37.0);
/// Red step between target ids in the id render.
pub const ID_STEP: u8 = 8;
/// Terrain: wide enough that oblique views from 130 m don't see past its edge.
pub const TERRAIN_M: f64 = 1600.0;
const CELL_M: f64 = 2.5;
const RELIEF_M: f64 = 12.0;
/// Ground texture: one tile every TILE_M metres, TEX_SIZE texels across.
const TILE_M: f32 = 24.0;
const TEX_SIZE: usize = 512;
const CLUTTER_MATS: usize = 8;

/// Unlabelled things in the scene: they hide targets and look like them (warm blobs,
/// crisp edges), so the model learns what isn't a body.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ClutterKind {
    Bush,
    Rock,
    House,
    Car,
}

/// Kinds and how many of each the scene holds (in `ClutterKind` order).
const CLUTTER: [(ClutterKind, usize); 4] = [
    (ClutterKind::Bush, 60),
    (ClutterKind::Rock, 10),
    (ClutterKind::House, 8),
    (ClutterKind::Car, 10),
];

/// What the pool of bodies holds; datagen picks from it per shot.
pub const POOL: [(Kind, usize); 10] = [
    (Kind::Person, 6),
    (Kind::Cattle, 3),
    (Kind::Horse, 2),
    (Kind::Deer, 3),
    (Kind::Sheep, 3),
    (Kind::Dog, 2),
    (Kind::Boar, 2),
    (Kind::Elephant, 3),
    (Kind::Giraffe, 3),
    (Kind::Lion, 2),
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Person,
    Cattle,
    Horse,
    Deer,
    Sheep,
    Dog,
    Boar,
    // wildlife, as in the thermal BIRDSAI flights (all `other_animal`)
    Elephant,
    Giraffe,
    Lion,
}

impl Kind {
    /// Our class id (`configs/classes.yaml`).
    pub fn class_id(self) -> u32 {
        match self {
            Kind::Person => 0,
            Kind::Dog => 1,
            Kind::Horse => 3,
            Kind::Cattle => 4,
            Kind::Sheep => 5,
            Kind::Deer | Kind::Boar | Kind::Elephant | Kind::Giraffe | Kind::Lion => 6,
        }
    }
}

/// Body proportions of a four-legged animal, metres.
struct Quadruped {
    length: f32,
    radius: f32,
    leg: f32,
    head: f32,
    neck_rise: f32,
}

fn quadruped(kind: Kind) -> Quadruped {
    let q = |length, radius, leg, head, neck_rise| Quadruped {
        length,
        radius,
        leg,
        head,
        neck_rise,
    };
    match kind {
        Kind::Cattle => q(2.0, 0.38, 0.75, 0.22, 0.25),
        Kind::Horse => q(2.0, 0.32, 0.95, 0.2, 0.7),
        Kind::Deer => q(1.3, 0.2, 0.8, 0.12, 0.5),
        Kind::Sheep => q(1.0, 0.27, 0.42, 0.12, 0.15),
        Kind::Dog => q(0.75, 0.13, 0.38, 0.1, 0.2),
        Kind::Boar => q(1.2, 0.28, 0.38, 0.17, 0.0),
        Kind::Elephant => q(3.2, 0.85, 1.3, 0.55, 0.1),
        Kind::Giraffe => q(2.0, 0.45, 1.7, 0.22, 2.0),
        Kind::Lion => q(1.7, 0.3, 0.65, 0.22, 0.15),
        Kind::Person => unreachable!("people aren't quadrupeds"),
    }
}

#[derive(Resource)]
pub struct World {
    pub terrain: Terrain,
    pub seed: u64,
    pub roads: Vec<Road>,
    /// Terrain vertices and normals (Bevy coordinates), for per-shot temperatures.
    pub vertices: Vec<[f64; 3]>,
    pub normals: Vec<[f64; 3]>,
}

impl World {
    pub fn new(seed: u64) -> Self {
        let terrain = Terrain::generate(seed, TERRAIN_M, CELL_M, RELIEF_M);
        let (vertices, _, _) = terrain.vertices();
        let normals = vertices
            .iter()
            .map(|p| terrain.normal(p[0], p[2]))
            .collect();
        let mut rng = Rng::new(seed ^ 0x0A0D_5EED);
        let roads = (0..1 + rng.index(4))
            .map(|_| Road::random(&mut rng))
            .collect();
        Self {
            terrain,
            seed,
            roads,
            vertices,
            normals,
        }
    }
}

/// A gently winding road or track: ground points (x, z) every few metres, and its width.
pub struct Road {
    pub points: Vec<[f64; 2]>,
    pub width: f64,
}

impl Road {
    /// Through a random spot near the middle (where shots look), 1.2 km long.
    fn random(rng: &mut Rng) -> Self {
        let start = [rng.range(-150.0, 150.0), rng.range(-150.0, 150.0)];
        let heading = rng.range(0.0, std::f64::consts::TAU);
        let step = 4.0;
        let half = (600.0 / step) as usize;
        let mut walk = |dir: f64| {
            let (mut p, mut h) = (start, heading + dir);
            let mut pts = Vec::with_capacity(half);
            for _ in 0..half {
                h += rng.range(-0.04, 0.04);
                p = [p[0] + step * h.cos(), p[1] + step * h.sin()];
                pts.push(p);
            }
            pts
        };
        let back = walk(std::f64::consts::PI);
        let ahead = walk(0.0);
        let mut points: Vec<[f64; 2]> = back.into_iter().rev().collect();
        points.push(start);
        points.extend(ahead);
        Self {
            points,
            width: rng.range(3.0, 7.0),
        }
    }
}

#[derive(Component)]
pub struct AerialCamera;

/// Parts of a body with their own temperature: thin limbs lose heat, the head is warm.
const TORSO: usize = 0;
const LIMB: usize = 1;
const HEAD: usize = 2;

/// A body we label. `id` is 1-based and is its colour in the id render.
#[derive(Component)]
pub struct Target {
    pub kind: Kind,
    pub id: u32,
    /// Thermal materials of its torso, limbs and head.
    thermal: [Handle<StandardMaterial>; 3],
}

#[derive(Component)]
pub struct Clutter {
    kind: ClutterKind,
}

#[derive(Component)]
pub struct WaterSurface;

/// The two materials of a mesh; `SceneCtl::show_ids` swaps between them.
#[derive(Component)]
pub struct Look {
    thermal: Handle<StandardMaterial>,
    id: Handle<StandardMaterial>,
}

#[derive(Resource)]
pub struct SceneHandles {
    terrain_mesh: Handle<Mesh>,
    texture: Handle<Image>,
    water: Handle<StandardMaterial>,
    roads: Handle<StandardMaterial>,
    /// Per `ClutterKind`, CLUTTER_MATS materials shared by the parts of its instances.
    clutter: Vec<Vec<Handle<StandardMaterial>>>,
}

/// Where one body goes in a shot.
#[derive(Clone, Debug)]
pub struct Placement {
    pub transform: Transform,
    /// Torso, limbs, head.
    pub temp_c: [f64; 3],
}

/// One random shot: camera, lens, conditions and where everything stands, drawn from the
/// ranges of the real data.
#[derive(Clone, Debug)]
pub struct Shot {
    /// Height of the camera above the ground it looks at, metres (real: 60-130).
    pub altitude_m: f64,
    /// Camera angle below the horizon, degrees: 90 looks straight down (real: 30-90).
    pub pitch_deg: f64,
    pub vfov_deg: f64,
    pub camera: Transform,
    pub conditions: Conditions,
    /// Per pool target, in spawn order; `None` = not in this shot.
    pub targets: Vec<Option<Placement>>,
    /// Per `ClutterKind`: where each instance goes (`None` = not in this shot) and the
    /// temperatures of its materials.
    pub clutter: Vec<Vec<Option<Transform>>>,
    pub clutter_c: Vec<Vec<f64>>,
    pub road_c: f64,
    /// Ground temperature per terrain vertex.
    pub ground_c: Vec<f64>,
    pub window: TempWindow,
}

impl Shot {
    pub fn random(rng: &mut Rng, world: &World, kinds: &[Kind]) -> Self {
        let terrain = &world.terrain;
        // log-uniform: narrow lenses (HIT-UAV's people are 15x17 px straight down from 60 m)
        // as often as wide ones
        let vfov_deg = rng.range(VFOV_DEG.0.ln(), VFOV_DEG.1.ln()).exp();
        let altitude_m = rng.range(60.0, 130.0);
        // keep the horizon out of the picture: the top edge looks at least 5 deg down
        let pitch_deg = rng.range((vfov_deg / 2.0 + 5.0).max(30.0), 90.0);
        let aim = terrain.random_spot(rng, 100.0);
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

        let flood = rng.chance(0.3).then(|| aim[1] + rng.range(-1.5, 1.0));
        let conditions = Conditions::random(rng, flood);
        let ground_c: Vec<f64> = world
            .vertices
            .iter()
            .zip(&world.normals)
            .map(|(p, n)| conditions.ground_c(world.seed, p[0], p[2], *n))
            .collect();
        let ground_at =
            |x: f64, z: f64| conditions.ground_c(world.seed, x, z, terrain.normal(x, z));

        // Ground half-height of the picture at the aim point (nadir); oblique views see more.
        let spread = altitude_m * (vfov_deg / 2.0).to_radians().tan();
        let near = |rng: &mut Rng, r: f64| (aim[0] + rng.range(-r, r), aim[2] + rng.range(-r, r));

        let chosen = choose_targets(rng, kinds);
        let (cx, cz) = near(rng, 0.7 * spread);
        let group_r = rng.range(2.0, 0.8 * spread);
        let targets = kinds
            .iter()
            .zip(&chosen)
            .map(|(&kind, &on)| {
                on.then(|| {
                    let (x, z) = (
                        cx + rng.range(-group_r, group_r),
                        cz + rng.range(-group_r, group_r),
                    );
                    let feet = Vec3::new(x as f32, terrain.height(x, z) as f32, z as f32);
                    let ground = ground_at(x, z);
                    let torso = conditions.body_c(rng, ground, kind == Kind::Person);
                    let limb = ground + (torso - ground) * rng.range(0.5, 0.85);
                    let head = ground + (torso - ground) * rng.range(1.0, 1.2);
                    Placement {
                        transform: pose(rng, kind, feet),
                        temp_c: [torso, limb, head],
                    }
                })
            })
            .collect();

        let mut clutter = Vec::new();
        let mut clutter_c = Vec::new();
        for (kind, max) in CLUTTER {
            // houses and cars come in some scenes only (a farm, a village, a road)
            let n = match kind {
                ClutterKind::Bush => rng.index(max + 1),
                ClutterKind::Rock => rng.chance(0.4) as usize * (1 + rng.index(max)),
                ClutterKind::House => rng.chance(0.35) as usize * (1 + rng.index(max)),
                ClutterKind::Car => rng.chance(0.4) as usize * (1 + rng.index(max)),
            };
            let placed: Vec<Option<Transform>> = (0..max)
                .map(|i| {
                    if i >= n {
                        return None;
                    }
                    let (x, z) = near(rng, 1.5 * spread);
                    let yaw = Quat::from_rotation_y(rng.range(0.0, std::f64::consts::TAU) as f32);
                    let ground = terrain.height(x, z);
                    let at = |y: f64, size: Vec3| {
                        Some(
                            Transform::from_xyz(x as f32, y as f32, z as f32)
                                .with_rotation(yaw)
                                .with_scale(size),
                        )
                    };
                    match kind {
                        ClutterKind::Bush => {
                            if rng.unit() > crate::thermal::vegetation(world.seed, x, z) + 0.15 {
                                return None; // bushes grow where the vegetation is
                            }
                            let tree = rng.chance(0.25);
                            let (rx, ry) = if tree {
                                (rng.range(1.5, 4.0), rng.range(3.0, 7.0))
                            } else {
                                let r = rng.range(0.4, 2.5);
                                (r, r * rng.range(0.4, 1.0))
                            };
                            let rz = rx * rng.range(0.7, 1.3);
                            let lift = if tree { 0.8 } else { 0.4 };
                            at(
                                ground + ry * lift,
                                Vec3::new(rx as f32, ry as f32, rz as f32),
                            )
                        }
                        ClutterKind::Rock => {
                            let r = rng.range(0.15, 0.7);
                            let size = Vec3::new(
                                r as f32,
                                (r * 0.6) as f32,
                                (r * rng.range(0.6, 1.4)) as f32,
                            );
                            at(ground, size)
                        }
                        ClutterKind::House => {
                            let (w, d, h) = (
                                rng.range(6.0, 14.0),
                                rng.range(6.0, 12.0),
                                rng.range(3.0, 6.0),
                            );
                            // sunk to the lowest corner, so it never floats on a slope
                            let low = [(-1.0, -1.0), (1.0, -1.0), (-1.0, 1.0), (1.0, 1.0)]
                                .iter()
                                .map(|(sx, sz)| terrain.height(x + sx * w / 2.0, z + sz * d / 2.0))
                                .fold(ground, f64::min);
                            let base = low - 0.5;
                            let top = ground + h;
                            at(
                                (base + top) / 2.0,
                                Vec3::new(w as f32, (top - base) as f32, d as f32),
                            )
                        }
                        ClutterKind::Car => at(ground + 0.7, Vec3::new(1.8, 1.5, 4.5)),
                    }
                })
                .collect();
            let temps: Vec<f64> = (0..CLUTTER_MATS)
                .map(|i| match kind {
                    ClutterKind::Bush => conditions.vegetation_c(rng),
                    ClutterKind::Rock => conditions.clutter_c(rng),
                    ClutterKind::House => conditions.building_c(rng),
                    ClutterKind::Car if i < CLUTTER_MATS / 2 => conditions.vehicle_c(rng),
                    ClutterKind::Car => conditions.glass_c(rng),
                })
                .collect();
            clutter.push(placed);
            clutter_c.push(temps);
        }
        let road_c = conditions.road_c(rng);

        let targets: Vec<Option<Placement>> = targets;
        let window = TempWindow::around(
            ground_c
                .iter()
                .copied()
                .chain(clutter_c.iter().flatten().copied())
                .chain([road_c])
                .chain(targets.iter().flatten().flat_map(|p| p.temp_c))
                .chain(conditions.water.map(|w| w.1)),
        );
        Self {
            altitude_m,
            pitch_deg,
            vfov_deg,
            camera,
            conditions,
            targets,
            clutter,
            clutter_c,
            road_c,
            ground_c,
            window,
        }
    }

    pub fn n_targets(&self) -> usize {
        self.targets.iter().flatten().count()
    }
}

/// Which pool targets are in a shot: ~30% empty frames (like the real data), else a group
/// of people, a herd of one kind, or a mix.
fn choose_targets(rng: &mut Rng, kinds: &[Kind]) -> Vec<bool> {
    let u = rng.unit();
    if u < 0.3 {
        return vec![false; kinds.len()];
    }
    if u < 0.5 {
        let n = 1 + rng.index(6);
        let mut k = 0;
        return kinds
            .iter()
            .map(|&kind| {
                let on = kind == Kind::Person && k < n;
                k += (kind == Kind::Person) as usize;
                on
            })
            .collect();
    }
    if u < 0.85 {
        let animals: Vec<Kind> = POOL
            .iter()
            .map(|p| p.0)
            .filter(|&k| k != Kind::Person)
            .collect();
        let herd = animals[rng.index(animals.len())];
        let n = 1 + rng.index(kinds.iter().filter(|&&k| k == herd).count());
        let mut k = 0;
        return kinds
            .iter()
            .map(|&kind| {
                let on = kind == herd && k < n;
                k += (kind == herd) as usize;
                on
            })
            .collect();
    }
    let mut chosen: Vec<bool> = kinds.iter().map(|_| rng.chance(0.25)).collect();
    if !chosen.contains(&true) {
        chosen[rng.index(kinds.len())] = true;
    }
    chosen
}

/// Random pose and size at `feet`: people stand, sit or lie; animals stand or lie.
fn pose(rng: &mut Rng, kind: Kind, feet: Vec3) -> Transform {
    let yaw = Quat::from_rotation_y(rng.range(0.0, std::f64::consts::TAU) as f32);
    let s = rng.range(0.85, 1.15) as f32;
    if kind == Kind::Person {
        let u = rng.unit();
        if u < 0.5 {
            Transform::from_translation(feet)
                .with_rotation(yaw)
                .with_scale(Vec3::splat(s))
        } else if u < 0.7 {
            Transform::from_translation(feet)
                .with_rotation(yaw)
                .with_scale(Vec3::new(s, s * 0.6, s))
        } else {
            // lying: the body's local y runs along the ground, its front-back depth is up
            Transform::from_translation(feet + Vec3::Y * 0.13 * s)
                .with_rotation(yaw * Quat::from_rotation_x(-FRAC_PI_2))
                .with_scale(Vec3::splat(s))
        }
    } else if rng.chance(0.75) {
        Transform::from_translation(feet)
            .with_rotation(yaw)
            .with_scale(Vec3::splat(s))
    } else {
        // lying: legs in the ground, belly near it
        let leg = quadruped(kind).leg * s;
        Transform::from_translation(feet - Vec3::Y * leg * 0.9)
            .with_rotation(yaw)
            .with_scale(Vec3::splat(s))
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

fn unlit(mats: &mut Assets<StandardMaterial>, color: Color) -> Handle<StandardMaterial> {
    mats.add(StandardMaterial {
        base_color: color,
        unlit: true,
        cull_mode: None,
        ..default()
    })
}

/// Spawn the camera, terrain, water, clutter and the body pool (`kinds`, in order; ids
/// 1, 2, ...). Everything is placed per shot by `SceneCtl::apply`.
pub fn spawn_scene(
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
    images: &mut Assets<Image>,
    mats: &mut Assets<StandardMaterial>,
    world: &World,
    kinds: &[Kind],
) {
    assert!(
        kinds.len() * ID_STEP as usize <= 255,
        "too many targets for the id render"
    );
    commands.spawn((
        Camera3d::default(),
        Camera {
            clear_color: ClearColorConfig::Custom(Color::BLACK),
            ..default()
        },
        Projection::Perspective(PerspectiveProjection {
            fov: (VFOV_DEG.1 as f32).to_radians(),
            near: 1.0,
            far: 4000.0,
            ..default()
        }),
        // Exact colours out: no tonemapping, no dither, no edge blending (the id render
        // must stay exact; the sensor model blurs the thermal frame instead).
        Tonemapping::None,
        DebandDither::Disabled,
        Msaa::Off,
        AerialCamera,
    ));
    // Green, so a real id frame is never all zeros (a blank capture is); terrain's vertex
    // colours scale it, red stays 0.
    let background = unlit(mats, Color::srgb(0.0, 1.0, 0.0));

    let texture = images.add(texture_image(ground_texture(
        world.seed, TEX_SIZE, 0.1, 0.0,
    )));
    let terrain_mesh = meshes.add(terrain_mesh(world));
    let ground = mats.add(StandardMaterial {
        base_color: Color::linear_rgb(texture_gain(), texture_gain(), texture_gain()),
        base_color_texture: Some(texture.clone()),
        unlit: true,
        cull_mode: None,
        ..default()
    });
    commands.spawn((
        Mesh3d(terrain_mesh.clone()),
        MeshMaterial3d(ground.clone()),
        Look {
            thermal: ground,
            id: background.clone(),
        },
    ));

    let water = unlit(mats, Color::BLACK);
    let size = TERRAIN_M as f32;
    commands.spawn((
        Mesh3d(meshes.add(Plane3d::default().mesh().size(size, size))),
        MeshMaterial3d(water.clone()),
        Transform::default(),
        Visibility::Hidden,
        WaterSurface,
        Look {
            thermal: water.clone(),
            id: background.clone(),
        },
    ));

    let roads = unlit(mats, Color::BLACK);
    commands.spawn((
        Mesh3d(meshes.add(road_mesh(world))),
        MeshMaterial3d(roads.clone()),
        Look {
            thermal: roads.clone(),
            id: background.clone(),
        },
    ));

    let block = meshes.add(Cuboid::new(1.0, 1.0, 1.0));
    let mut shapes = Rng::new(world.seed ^ 0x5_4A9E);
    let canopies: Vec<_> = (0..8)
        .map(|_| meshes.add(lumpy_blob(&mut shapes, 0.35, 0.3)))
        .collect();
    let stones: Vec<_> = (0..4)
        .map(|_| meshes.add(lumpy_blob(&mut shapes, 0.15, 0.15)))
        .collect();
    let mut clutter = Vec::new();
    for (kind, n) in CLUTTER {
        let set: Vec<_> = (0..CLUTTER_MATS)
            .map(|_| {
                mats.add(StandardMaterial {
                    base_color: Color::BLACK,
                    // canopies are mottled like the ground
                    base_color_texture: (kind == ClutterKind::Bush).then(|| texture.clone()),
                    unlit: true,
                    cull_mode: None,
                    ..default()
                })
            })
            .collect();
        for _ in 0..n {
            let parts = clutter_parts(&mut shapes, kind, &canopies, &stones, &block);
            commands
                .spawn((Transform::default(), Visibility::Hidden, Clutter { kind }))
                .with_children(|c| {
                    for (mesh, tf, mat) in parts {
                        let m = set[mat].clone();
                        c.spawn((
                            Mesh3d(mesh),
                            MeshMaterial3d(m.clone()),
                            tf,
                            Look {
                                thermal: m,
                                id: background.clone(),
                            },
                        ));
                    }
                });
        }
        clutter.push(set);
    }

    for (k, &kind) in kinds.iter().enumerate() {
        let id = k as u32 + 1;
        let thermal = [(); 3].map(|_| unlit(mats, Color::WHITE));
        let id_mat = unlit(mats, Color::srgb_u8(id as u8 * ID_STEP, 0, 0));
        let parts = body_parts(meshes, &mut shapes, kind);
        commands
            .spawn((
                Transform::default(),
                Visibility::Hidden,
                Target {
                    kind,
                    id,
                    thermal: thermal.clone(),
                },
            ))
            .with_children(|body| {
                for (mesh, tf, part) in parts {
                    body.spawn((
                        Mesh3d(mesh),
                        MeshMaterial3d(thermal[part].clone()),
                        tf,
                        Look {
                            thermal: thermal[part].clone(),
                            id: id_mat.clone(),
                        },
                    ));
                }
            });
    }
    commands.insert_resource(SceneHandles {
        terrain_mesh,
        texture,
        water,
        roads,
        clutter,
    });
}

/// Ribbons along the world's roads, a little above the ground (tracks, lanes, levees:
/// long crisp edges that the ground's vertex colours can't draw).
fn road_mesh(world: &World) -> Mesh {
    let (mut pos, mut idx) = (Vec::new(), Vec::new());
    for road in &world.roads {
        for (k, p) in road.points.iter().enumerate() {
            let q = road.points[(k + 1).min(road.points.len() - 1)];
            let r = road.points[k.saturating_sub(1)];
            let (dx, dz) = (q[0] - r[0], q[1] - r[1]);
            let len = (dx * dx + dz * dz).sqrt().max(1e-9);
            let (nx, nz) = (-dz / len * road.width / 2.0, dx / len * road.width / 2.0);
            let base = pos.len() as u32;
            for s in [-1.0, 1.0] {
                let (x, z) = (p[0] + s * nx, p[1] + s * nz);
                pos.push([
                    x as f32,
                    (world.terrain.height(x, z) + 0.25) as f32,
                    z as f32,
                ]);
            }
            if k + 1 < road.points.len() {
                idx.extend_from_slice(&[base, base + 2, base + 1, base + 1, base + 2, base + 3]);
            }
        }
    }
    let n = pos.len();
    Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::default(),
    )
    .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, pos)
    .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, vec![[0.0f32, 1.0, 0.0]; n])
    .with_inserted_attribute(Mesh::ATTRIBUTE_UV_0, vec![[0.0f32, 0.0]; n])
    .with_inserted_indices(Indices::U32(idx))
}

/// A unit sphere with its surface pushed in and out by 3D noise (`rough` = fraction of
/// the radius), so its outline is ragged at every scale, and vertex colours that vary its
/// temperature inside by up to `mottle` (a canopy has gaps, sunlit and shaded sides).
/// Finely subdivided (2,562 vertices) so the outline stays ragged up close.
fn lumpy_blob(rng: &mut Rng, rough: f64, mottle: f64) -> Mesh {
    use bevy::mesh::VertexAttributeValues;
    let mut mesh = Sphere::new(1.0).mesh().ico(4).expect("ico sphere");
    let (seed, freq) = (rng.next_u64(), rng.range(1.5, 3.0));
    let mut colors = Vec::new();
    if let Some(VertexAttributeValues::Float32x3(pos)) =
        mesh.attribute_mut(Mesh::ATTRIBUTE_POSITION)
    {
        for p in pos.iter_mut() {
            let (x, y, z) = (p[0] as f64, p[1] as f64, p[2] as f64);
            let r = 1.0 + rough * crate::world::fbm3(seed, x * freq, y * freq, z * freq, 4);
            *p = [(x * r) as f32, (y * r) as f32, (z * r) as f32];
            let shade = crate::world::fbm3(seed ^ 1, x * 5.0, y * 5.0, z * 5.0, 3);
            let m = (1.0 - mottle * (0.5 + 0.5 * shade)) as f32;
            colors.push([m, m, m, 1.0]);
        }
    }
    mesh.insert_attribute(Mesh::ATTRIBUTE_COLOR, colors);
    mesh.compute_smooth_normals();
    mesh
}

/// Parts of one clutter instance in unit space (the shot scales the whole thing), each
/// with an index into its kind's materials: the first half are the kind's main
/// temperatures, the second half its secondary ones (car glass, the other roof half).
/// A real model took smooth uniform discs and plain rectangles for animals (zero-shot
/// test, 2026-10-28), so clutter is made irregular and two-tone, unlike bodies.
fn clutter_parts(
    rng: &mut Rng,
    kind: ClutterKind,
    canopies: &[Handle<Mesh>],
    stones: &[Handle<Mesh>],
    block: &Handle<Mesh>,
) -> Vec<(Handle<Mesh>, Transform, usize)> {
    let half = CLUTTER_MATS / 2;
    match kind {
        ClutterKind::Bush => {
            // a clump of 1-4 lumpy blobs, each its own temperature: an irregular canopy
            (0..1 + rng.index(4))
                .map(|_| {
                    let blob = &canopies[rng.index(canopies.len())];
                    let r = rng.range(0.45, 0.8) as f32;
                    let tf = Transform::from_xyz(
                        rng.range(-0.55, 0.55) as f32,
                        rng.range(-0.2, 0.3) as f32,
                        rng.range(-0.55, 0.55) as f32,
                    )
                    .with_scale(Vec3::new(
                        r,
                        r * rng.range(0.7, 1.1) as f32,
                        r,
                    ));
                    (blob.clone(), tf, rng.index(CLUTTER_MATS))
                })
                .collect()
        }
        ClutterKind::Rock => vec![(
            stones[rng.index(stones.len())].clone(),
            Transform::default(),
            rng.index(half),
        )],
        ClutterKind::House => {
            // walls, and a roof in two halves along the ridge (sun side and shade side)
            let main = rng.index(half);
            vec![
                (
                    block.clone(),
                    Transform::from_scale(Vec3::new(0.98, 1.0, 0.98)),
                    main,
                ),
                (
                    block.clone(),
                    Transform::from_xyz(-0.25, 0.5, 0.0).with_scale(Vec3::new(0.5, 0.06, 1.0)),
                    main,
                ),
                (
                    block.clone(),
                    Transform::from_xyz(0.25, 0.5, 0.0).with_scale(Vec3::new(0.5, 0.06, 1.0)),
                    half + rng.index(half),
                ),
            ]
        }
        ClutterKind::Car => vec![
            (
                block.clone(),
                Transform::from_xyz(0.0, -0.2, 0.0).with_scale(Vec3::new(1.0, 0.55, 1.0)),
                rng.index(half),
            ),
            // cabin and glass: reflects the cold sky
            (
                block.clone(),
                Transform::from_xyz(0.0, 0.28, -0.05).with_scale(Vec3::new(0.85, 0.45, 0.5)),
                half + rng.index(half),
            ),
        ],
    }
}

/// Meshes of a body, relative to its feet (y up, facing +z). Each instance gets its own
/// pose, drawn once at spawn: people with arms at their sides or out; animals standing or
/// mid-stride, grazers sometimes with the head down. At 32-96 px the outline is what the
/// detector sees, and a stiff four-post table didn't look like real animals to it.
fn body_parts(
    meshes: &mut Assets<Mesh>,
    rng: &mut Rng,
    kind: Kind,
) -> Vec<(Handle<Mesh>, Transform, usize)> {
    if kind == Kind::Person {
        // 1.75 m tall, shoulders 0.44 m wide, 0.26 m front to back, and two arms
        let body = meshes.add(Capsule3d::new(0.22, 1.75 - 0.44));
        let arm = meshes.add(Capsule3d::new(0.055, 0.55));
        let mut parts = vec![(
            body,
            Transform::from_xyz(0.0, 0.875, 0.0).with_scale(Vec3::new(1.0, 1.0, 0.6)),
            TORSO,
        )];
        for side in [-1.0f32, 1.0] {
            let out = rng.range(0.05, 0.6) as f32; // radians away from the body
            let swing = rng.range(-0.4, 0.4) as f32; // forward or back
            parts.push((
                arm.clone(),
                Transform::from_xyz(side * (0.25 + 0.15 * out), 1.12, 0.0).with_rotation(
                    Quat::from_rotation_z(side * out) * Quat::from_rotation_x(swing),
                ),
                LIMB,
            ));
        }
        return parts;
    }
    let q = quadruped(kind);
    let torso_y = q.leg + q.radius * 0.8;
    let mut parts = vec![(
        meshes.add(Capsule3d::new(
            q.radius,
            (q.length - 2.0 * q.radius).max(0.01),
        )),
        Transform::from_xyz(0.0, torso_y, 0.0).with_rotation(Quat::from_rotation_x(FRAC_PI_2)),
        TORSO,
    )];
    let leg_len = q.leg + q.radius * 0.3;
    let leg = meshes.add(Cylinder::new(q.radius * 0.3, leg_len));
    // stride: diagonal legs swing together, front and back in opposite directions
    let stride = if rng.chance(0.6) {
        rng.range(0.15, 0.45) as f32
    } else {
        0.0
    };
    for (sx, sz) in [(-1.0f32, -1.0f32), (1.0, -1.0), (-1.0, 1.0), (1.0, 1.0)] {
        let swing = stride * sx * sz;
        let hip = Vec3::new(
            sx * q.radius * 0.6,
            leg_len,
            sz * (q.length / 2.0 - q.radius),
        );
        let dir = Vec3::new(0.0, -swing.cos(), swing.sin());
        parts.push((
            leg.clone(),
            Transform::from_translation(hip + dir * leg_len / 2.0)
                .with_rotation(Quat::from_rotation_arc(Vec3::Y, -dir)),
            LIMB,
        ));
    }
    let grazer = matches!(
        kind,
        Kind::Cattle | Kind::Horse | Kind::Deer | Kind::Sheep | Kind::Giraffe | Kind::Elephant
    );
    let shoulder = Vec3::new(0.0, torso_y, q.length / 2.0 - q.radius);
    let head = if grazer && rng.chance(0.4) {
        // head down to the grass, ahead of the shoulders
        Vec3::new(0.0, q.head, q.length / 2.0 + q.head + q.leg * 0.3)
    } else {
        Vec3::new(0.0, torso_y + q.neck_rise, q.length / 2.0 + q.head * 0.5)
    };
    let neck = head - shoulder;
    parts.push((
        meshes.add(Capsule3d::new(q.head * 0.6, neck.length())),
        Transform::from_translation((shoulder + head) / 2.0)
            .with_rotation(Quat::from_rotation_arc(Vec3::Y, neck.normalize())),
        TORSO,
    ));
    parts.push((
        meshes.add(Sphere::new(q.head)),
        Transform::from_translation(head),
        HEAD,
    ));
    parts
}

/// Where an entity is and whether it shows, plus one marker component.
type Placed<C> = (&'static mut Transform, &'static mut Visibility, &'static C);
/// The water surface, kept apart from every other `Transform` query.
type WaterOnly = (
    With<WaterSurface>,
    Without<AerialCamera>,
    Without<Target>,
    Without<Clutter>,
);

/// Everything a shot changes, for systems in both binaries.
#[derive(SystemParam)]
pub struct SceneCtl<'w, 's> {
    meshes: ResMut<'w, Assets<Mesh>>,
    images: ResMut<'w, Assets<Image>>,
    mats: ResMut<'w, Assets<StandardMaterial>>,
    handles: Res<'w, SceneHandles>,
    camera: Query<'w, 's, (&'static mut Transform, &'static mut Projection), With<AerialCamera>>,
    targets: Query<'w, 's, Placed<Target>, Without<AerialCamera>>,
    clutter: Query<'w, 's, Placed<Clutter>, (Without<AerialCamera>, Without<Target>)>,
    water: Query<'w, 's, (&'static mut Transform, &'static mut Visibility), WaterOnly>,
    looks: Query<'w, 's, (&'static Look, &'static mut MeshMaterial3d<StandardMaterial>)>,
}

impl SceneCtl<'_, '_> {
    /// Kinds of the body pool, in id order.
    pub fn kinds(&self) -> Vec<Kind> {
        let mut t: Vec<(u32, Kind)> = self
            .targets
            .iter()
            .map(|(_, _, t)| (t.id, t.kind))
            .collect();
        t.sort_by_key(|x| x.0);
        t.into_iter().map(|x| x.1).collect()
    }

    /// Class id of target `id`.
    pub fn class_of(&self, id: u32) -> Option<u32> {
        self.targets
            .iter()
            .find(|(_, _, t)| t.id == id)
            .map(|(_, _, t)| t.kind.class_id())
    }

    /// Move everything to `shot` and set every temperature.
    pub fn apply(&mut self, shot: &Shot) {
        let w = shot.window;
        let grey = |t: f64| {
            let v = w.linear(t);
            Color::linear_rgb(v, v, v)
        };
        let (mut cam_tf, mut proj) = self.camera.single_mut().expect("one aerial camera");
        *cam_tf = shot.camera;
        if let Projection::Perspective(p) = &mut *proj {
            p.fov = (shot.vfov_deg as f32).to_radians();
        }

        let colors: Vec<[f32; 4]> = shot
            .ground_c
            .iter()
            .map(|&t| {
                let v = w.linear(t);
                [v, v, v, 1.0]
            })
            .collect();
        if let Some(mut mesh) = self.meshes.get_mut(&self.handles.terrain_mesh) {
            mesh.insert_attribute(Mesh::ATTRIBUTE_COLOR, colors);
        }
        if let Some(mut img) = self.images.get_mut(&self.handles.texture) {
            let c = &shot.conditions;
            img.data = Some(rgba(&ground_texture(
                c.texture_seed,
                TEX_SIZE,
                c.texture,
                c.speckle,
            )));
        }

        let (mut water_tf, mut water_vis) = self.water.single_mut().expect("one water surface");
        match shot.conditions.water {
            Some((level, temp)) => {
                water_tf.translation = Vec3::new(0.0, level as f32, 0.0);
                *water_vis = Visibility::Inherited;
                if let Some(mut m) = self.mats.get_mut(&self.handles.water) {
                    m.base_color = grey(temp);
                }
            }
            None => *water_vis = Visibility::Hidden,
        }

        if let Some(mut m) = self.mats.get_mut(&self.handles.roads) {
            m.base_color = grey(shot.road_c);
        }
        for ((set, temps), (kind, _)) in self
            .handles
            .clutter
            .iter()
            .zip(&shot.clutter_c)
            .zip(CLUTTER)
        {
            // bushes share the ground texture, so they make up for its base too
            let gain = if kind == ClutterKind::Bush {
                texture_gain()
            } else {
                1.0
            };
            for (h, &t) in set.iter().zip(temps) {
                if let Some(mut m) = self.mats.get_mut(h) {
                    let v = w.linear(t) * gain;
                    m.base_color = Color::linear_rgb(v, v, v);
                }
            }
        }
        let mut places: Vec<_> = shot.clutter.iter().map(|v| v.iter()).collect();
        for (mut tf, mut vis, c) in &mut self.clutter {
            match places[c.kind as usize].next().and_then(|p| p.as_ref()) {
                Some(p) => {
                    *tf = *p;
                    *vis = Visibility::Inherited;
                }
                None => *vis = Visibility::Hidden,
            }
        }

        for (mut tf, mut vis, target) in &mut self.targets {
            match &shot.targets[target.id as usize - 1] {
                Some(p) => {
                    *tf = p.transform;
                    *vis = Visibility::Inherited;
                    for (h, &t) in target.thermal.iter().zip(&p.temp_c) {
                        if let Some(mut m) = self.mats.get_mut(h) {
                            m.base_color = grey(t);
                        }
                    }
                }
                None => *vis = Visibility::Hidden,
            }
        }
    }

    /// Switch every mesh to its id look (`true`) or its thermal look.
    pub fn show_ids(&mut self, ids: bool) {
        for (look, mut mat) in &mut self.looks {
            mat.0 = if ids {
                look.id.clone()
            } else {
                look.thermal.clone()
            };
        }
    }
}

fn rgba(grey: &[u8]) -> Vec<u8> {
    grey.iter().flat_map(|&g| [g, g, g, 255]).collect()
}

fn texture_image(grey: Vec<u8>) -> Image {
    let mut img = Image::new(
        Extent3d {
            width: TEX_SIZE as u32,
            height: TEX_SIZE as u32,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        rgba(&grey),
        TextureFormat::Rgba8UnormSrgb,
        RenderAssetUsages::default(),
    );
    img.sampler = ImageSampler::Descriptor(ImageSamplerDescriptor {
        address_mode_u: ImageAddressMode::Repeat,
        address_mode_v: ImageAddressMode::Repeat,
        ..ImageSamplerDescriptor::linear()
    });
    img
}

/// Bevy mesh of the terrain grid: normals from CortenForge's height field, texture
/// coordinates in tiles, and white vertex colours that each shot overwrites with the
/// ground temperature.
pub fn terrain_mesh(world: &World) -> Mesh {
    let (w, d) = (world.terrain.field.width(), world.terrain.field.depth());
    let verts = &world.vertices;
    let pos: Vec<[f32; 3]> = verts
        .iter()
        .map(|p| [p[0] as f32, p[1] as f32, p[2] as f32])
        .collect();
    let normals: Vec<[f32; 3]> = world
        .normals
        .iter()
        .map(|n| [n[0] as f32, n[1] as f32, n[2] as f32])
        .collect();
    let uvs: Vec<[f32; 2]> = verts
        .iter()
        .map(|p| [p[0] as f32 / TILE_M, p[2] as f32 / TILE_M])
        .collect();
    let colors = vec![[1.0f32; 4]; verts.len()];
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
    .with_inserted_attribute(Mesh::ATTRIBUTE_UV_0, uvs)
    .with_inserted_attribute(Mesh::ATTRIBUTE_COLOR, colors)
    .with_inserted_indices(Indices::U32(idx))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pool() -> Vec<Kind> {
        POOL.iter()
            .flat_map(|&(k, n)| std::iter::repeat_n(k, n))
            .collect()
    }

    #[test]
    fn shots_stay_in_the_real_ranges_and_some_are_empty() {
        let world = World::new(1);
        let kinds = pool();
        let mut rng = Rng::new(2);
        let mut empty = 0;
        for _ in 0..40 {
            let s = Shot::random(&mut rng, &world, &kinds);
            assert!((60.0..130.0).contains(&s.altitude_m));
            assert!(s.pitch_deg - s.vfov_deg / 2.0 >= 5.0);
            assert_eq!(s.targets.len(), kinds.len());
            assert_eq!(s.ground_c.len(), world.vertices.len());
            empty += (s.n_targets() == 0) as usize;
            for p in s.targets.iter().flatten() {
                for t in p.temp_c {
                    assert!((s.window.lo..=s.window.hi).contains(&t));
                }
            }
        }
        assert!((4..=24).contains(&empty), "{empty} empty shots of 40");
    }

    #[test]
    fn pool_fits_the_id_render() {
        assert!(pool().len() * (ID_STEP as usize) <= 255);
    }
}
