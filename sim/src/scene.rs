//! The Bevy side: terrain, roads, flood water, vegetation, rocks, houses and cars, the bodies
//! we label (people and animals), the aerial camera, and two looks for every mesh:
//!
//! - **thermal**: unlit grey = temperature through the shot's window (`thermal.rs`); the
//!   sensor model runs on the captured frame afterwards (`sensor.rs`);
//! - **id**: target `k` in red `k * ID_STEP`, everything else green with red 0. Captured
//!   right after the thermal frame, it gives a visible-pixel mask per target, so boxes are
//!   tight and hidden targets (behind a hill, a tree, under water) get no box.
//!
//! Bodies are built from primitives (capsules, spheres, cylinders) or, for the kinds with
//! baked models (`bodies.rs`, CC0 animals), from posed model meshes: at 5-30 px the outline
//! and size matter, not the detail.
//!
//! In an RGB run (`Modality::Rgb`) the first look is a daylight photo instead: lit
//! materials in the shot's colours (`rgb.rs`), a sun that casts shadows and a sky, a 4:3
//! camera looking (nearly) straight down like the DJI stills of the real RGB data, a farm
//! pool (cattle, horses, sheep, dogs, people) and round and square bales on stubble.

use std::f32::consts::FRAC_PI_2;
use std::path::Path;

use bevy::asset::RenderAssetUsages;
use bevy::camera::Exposure;
use bevy::core_pipeline::tonemapping::{DebandDither, Tonemapping};
use bevy::ecs::system::SystemParam;
use bevy::image::{ImageAddressMode, ImageSampler, ImageSamplerDescriptor};
use bevy::light::{
    CascadeShadowConfigBuilder, DirectionalLightShadowMap, GlobalAmbientLight, NotShadowCaster,
};
use bevy::mesh::{Indices, PrimitiveTopology};
use bevy::prelude::*;
use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat};
use bevy::render::view::Msaa;

use crate::bodies::{self, Model};
use crate::fields::{Layout, NoiseTable, OVERLAY_TEX};
use crate::forest::{CANOPY_TEX, FOREST_TEX, ForestLook};
use crate::rgb::{self, Coat, Daylight, Ground};
use crate::rng::Rng;
use crate::thermal::{Conditions, TempWindow, ground_texture, texture_gain};
use crate::world::Terrain;
use crate::yard::{FEEDS, Feed, Shed, Yard};

/// Frame size of the thermal cameras in our real data (HIT-UAV, most BIRDSAI flights).
pub const IMAGE_W: u32 = 640;
pub const IMAGE_H: u32 = 512;
/// Vertical field of view range: 640x512 cores with 13-25 mm lenses (HIT-UAV's 25 mm is
/// about 20 deg).
pub const VFOV_DEG: (f64, f64) = (12.0, 37.0);
/// RGB frames: 4:3 like the real DJI stills (4000x3000, 5280x3956), at the size training
/// sees them (`imgsz` 1280).
pub const RGB_W: u32 = 1280;
pub const RGB_H: u32 = 960;
/// Their wide lenses (77-84 deg diagonal) and heights: real cows are 13x21 px (median) at
/// that size, which a 2.4 m sim cow is from about 110 m.
const RGB_VFOV_DEG: (f64, f64) = (45.0, 60.0);
const RGB_ALTITUDE_M: (f64, f64) = (60.0, 150.0);
/// Full sun and full sky light (an overcast sky's), set so a white surface under both
/// renders just below 1.0 at the camera's `RGB_EV100`; the camera model re-exposes.
const SUN_LUX: f32 = 80_000.0;
const SKY_NITS: f32 = 25_000.0;
const RGB_EV100: f32 = 15.0;
const PATCH_TEXTURES: usize = 4;
/// How far the painted ground floats above the terrain, metres.
const PATCH_LIFT_M: f64 = 0.08;

/// Which camera a run renders for.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Resource)]
pub enum Modality {
    Thermal,
    Rgb,
    /// Thermal nadir video over forests and meadows, like BAMBI (`forest.rs`): written
    /// as thermal frames.
    Forest,
}

impl Modality {
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "thermal" => Some(Self::Thermal),
            "rgb" => Some(Self::Rgb),
            "forest" => Some(Self::Forest),
            _ => None,
        }
    }

    /// The metadata's `modality` value.
    pub fn name(self) -> &'static str {
        match self {
            Self::Thermal | Self::Forest => "thermal",
            Self::Rgb => "rgb",
        }
    }

    /// Frame width and height.
    pub fn size(self) -> (u32, u32) {
        match self {
            Self::Thermal | Self::Forest => (IMAGE_W, IMAGE_H),
            Self::Rgb => (RGB_W, RGB_H),
        }
    }

    /// The body pool, in id order.
    pub fn kinds(self) -> Vec<Kind> {
        let pool: &[(Kind, usize)] = match self {
            Self::Thermal => &POOL,
            Self::Rgb => &RGB_POOL,
            Self::Forest => &FOREST_POOL,
        };
        pool.iter()
            .flat_map(|&(k, n)| std::iter::repeat_n(k, n))
            .collect()
    }

    fn clutter(self) -> &'static [(ClutterKind, usize)] {
        match self {
            Self::Thermal => &CLUTTER,
            Self::Rgb => &RGB_CLUTTER,
            // trees, logs and rocks are painted into the ground (`forest.rs`)
            Self::Forest => &[],
        }
    }

    /// Where `kind` is in `clutter()` (and so in a shot's per-kind lists).
    pub fn clutter_index(self, kind: ClutterKind) -> usize {
        self.clutter()
            .iter()
            .position(|c| c.0 == kind)
            .expect("a clutter kind of this modality")
    }
}
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
    /// Round bales (lying or on end) and square ones, on harvested fields (RGB only): the
    /// real RGB model's false alarms come from such a field (ICAERUS flight 032).
    Bale,
    /// Cars, pickups, vans and tractors (RGB only; thermal keeps `Car`), often on a track.
    Vehicle,
    /// Hedge segments along hedged field edges (RGB only).
    Hedge,
    /// Farmyard (`yard.rs`, RGB only): closed sheds, open-fronted sheds (roof on posts),
    /// 3 m rail panels round the pens, ring feeders and troughs, and stacks by the
    /// buildings (wrapped bales, tyres, tanks).
    Shed,
    OpenShed,
    Rail,
    Feeder,
    Stack,
}

/// Kinds and how many of each the scene holds (in `ClutterKind` order).
const CLUTTER: [(ClutterKind, usize); 4] = [
    (ClutterKind::Bush, 60),
    (ClutterKind::Rock, 10),
    (ClutterKind::House, 8),
    (ClutterKind::Car, 10),
];
/// RGB: fewer vehicles than thermal's cars (the plain boxes were the real model's top
/// alarms on sim frames), and hedges.
const RGB_CLUTTER: [(ClutterKind, usize); 11] = [
    (ClutterKind::Bush, 60),
    (ClutterKind::Rock, 10),
    (ClutterKind::House, 8),
    (ClutterKind::Vehicle, 6),
    (ClutterKind::Bale, 40),
    (ClutterKind::Hedge, RGB_HEDGES),
    (ClutterKind::Shed, 3),
    (ClutterKind::OpenShed, 3),
    (ClutterKind::Rail, 240),
    (ClutterKind::Feeder, 12),
    (ClutterKind::Stack, 20),
];
const RGB_HEDGES: usize = 160;
/// One rail panel's length, metres (`ClutterKind::Rail`).
const RAIL_M: f64 = 3.0;
/// Share of yard shots whose whole herd is in the yard (else only bodies that landed on a
/// building move there).
const YARD_HERD: f64 = 0.8;
/// Share of yard herds that pack close and line up (`Yard::herd_slots`), else scattered.
const YARD_PACKED: f64 = 0.85;

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

/// The RGB pool: farm animals, people and dogs (cattle is what real RGB val can score;
/// the rest have no real data yet). 31 bodies fill the id render; herds of up to 12 cattle
/// like the real flights.
pub const RGB_POOL: [(Kind, usize); 5] = [
    (Kind::Person, 5),
    (Kind::Cattle, 12),
    (Kind::Horse, 5),
    (Kind::Sheep, 7),
    (Kind::Dog, 2),
];

/// The forest pool: herds of deer and boar like BAMBI's (up to a dozen in a frame), a few
/// people and dogs.
pub const FOREST_POOL: [(Kind, usize); 4] = [
    (Kind::Person, 3),
    (Kind::Dog, 2),
    (Kind::Deer, 12),
    (Kind::Boar, 12),
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

    /// Folder of its baked models under the `--bodies` directory.
    pub fn dir_name(self) -> &'static str {
        match self {
            Kind::Person => "person",
            Kind::Cattle => "cattle",
            Kind::Horse => "horse",
            Kind::Deer => "deer",
            Kind::Sheep => "sheep",
            Kind::Dog => "dog",
            Kind::Boar => "boar",
            Kind::Elephant => "elephant",
            Kind::Giraffe => "giraffe",
            Kind::Lion => "lion",
        }
    }
}

/// Baked models per kind; kinds without any keep their primitive bodies.
#[derive(Default, Resource)]
pub struct BodyModels(Vec<(Kind, Vec<Model>)>);

impl BodyModels {
    /// The models under `dir` (see `bodies.rs`) of the kinds named in `only` (folder names
    /// such as "person"; empty = every kind that has some).
    pub fn load(dir: &Path, only: &[String]) -> std::io::Result<Self> {
        let mut out = Vec::new();
        for &(kind, _) in &POOL {
            if !only.is_empty() && !only.iter().any(|k| k == kind.dir_name()) {
                continue;
            }
            let models = bodies::load_kind(dir, kind.dir_name())?;
            if !models.is_empty() {
                out.push((kind, models));
            }
        }
        Ok(Self(out))
    }

    /// One line per kind, for the log.
    pub fn summary(&self) -> String {
        self.0
            .iter()
            .map(|(k, m)| {
                let names: Vec<_> = m.iter().map(|m| m.name.as_str()).collect();
                format!("{}: {}", k.dir_name(), names.join(", "))
            })
            .collect::<Vec<_>>()
            .join("; ")
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

/// The sun (RGB runs only).
#[derive(Component)]
pub struct Sun;

/// Parts of a body with their own temperature: thin limbs lose heat, the head is warm.
const TORSO: usize = 0;
const LIMB: usize = 1;
const HEAD: usize = 2;

/// A body we label. `id` is 1-based and is its colour in the id render.
///
/// A body built from models has one child per model and pose (a variant); each shot shows
/// one of them (`variant_index`). A primitive body has a single variant.
#[derive(Component)]
pub struct Target {
    pub kind: Kind,
    pub id: u32,
    /// Per variant: "primitive", or the model and pose ("Cow:Walk@0.25").
    pub bodies: Vec<String>,
    variants: Vec<Entity>,
    /// Its posture is in the mesh (model people), so `pose` only turns and scales it.
    pub posed: bool,
    /// Picture (thermal or RGB) materials of its torso, limbs and head.
    picture: [Handle<StandardMaterial>; 3],
}

/// One way a target can look; children of a `Target`.
#[derive(Component)]
struct Variant;

#[derive(Component)]
pub struct Clutter {
    kind: ClutterKind,
}

#[derive(Component)]
pub struct WaterSurface;

/// The two materials of a mesh, the picture (thermal or RGB) and the id look;
/// `SceneCtl::show_ids` swaps between them.
#[derive(Component)]
pub struct Look {
    picture: Handle<StandardMaterial>,
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
    /// Black patches on white for piebald coats (RGB).
    patches: Vec<Handle<Image>>,
    /// RGB: the painted ground's mesh and image, and the noise it's painted from.
    overlay: Option<(Handle<Mesh>, Handle<Image>)>,
    /// Forest: the leaf canopy's mesh, its picture (grey + alpha) and its id mask.
    canopy: Option<(Handle<Mesh>, Handle<Image>, Handle<Image>)>,
    table: Option<NoiseTable>,
}

/// The colours of an RGB shot (`rgb.rs`).
#[derive(Clone, Debug)]
pub struct RgbLook {
    pub daylight: Daylight,
    pub ground: Ground,
    pub road: rgb::Rgb,
    /// Per `ClutterKind`, the colours of its materials.
    pub clutter: Vec<Vec<rgb::Rgb>>,
    /// Per pool target, in spawn order; `None` = not in this shot.
    pub coats: Vec<Option<Coat>>,
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
    /// Its bodies are faint: each stands out from the ground under it by only
    /// `FAINT_GREY` grey levels of the render (see `Shot::random`'s `faint_share`).
    pub faint: bool,
    /// Colours, for an RGB shot (the temperatures above are then unused).
    pub rgb: Option<RgbLook>,
    /// The painted ground of a forest shot.
    pub forest: Option<ForestLook>,
}

/// How far a faint body stands out from the ground under it, in grey levels of the
/// render: like the tiny people of BIRDSAI flight 361 (about 1.6x the ground's mottle at
/// body scale, where most sim bodies are 2.2x). Set in grey levels, not degrees: a body
/// 2 C above the ground is bright in a cool field and nearly gone next to a sun-baked roof.
const FAINT_GREY: (f64, f64) = (16.0, 60.0);

impl Shot {
    /// `posed[k]`: pool body k carries its posture in its mesh (see `Target::posed`);
    /// may be empty. `faint_share`: share of shots whose bodies are faint (v7 tried 0.4;
    /// not kept, see PLAN.md; thermal only).
    ///
    /// A thermal shot draws exactly what it drew before RGB existed, so thermal pools stay
    /// byte for byte the same.
    pub fn random(
        rng: &mut Rng,
        world: &World,
        kinds: &[Kind],
        posed: &[bool],
        faint_share: f64,
        modality: Modality,
    ) -> Self {
        if modality == Modality::Forest {
            return Self::forest(rng, world, kinds, posed);
        }
        let terrain = &world.terrain;
        let thermal = modality == Modality::Thermal;
        let (vfov_deg, altitude_m, pitch_deg) = if thermal {
            // log-uniform: narrow lenses (HIT-UAV's people are 15x17 px straight down from
            // 60 m) as often as wide ones
            let vfov_deg = rng.range(VFOV_DEG.0.ln(), VFOV_DEG.1.ln()).exp();
            let altitude_m = rng.range(60.0, 130.0);
            // keep the horizon out of the picture: the top edge looks at least 5 deg down
            let pitch_deg = rng.range((vfov_deg / 2.0 + 5.0).max(30.0), 90.0);
            (vfov_deg, altitude_m, pitch_deg)
        } else {
            // mapping and counting flights look straight down; some frames tilt a little
            let vfov_deg = rng.range(RGB_VFOV_DEG.0, RGB_VFOV_DEG.1);
            let (lo, hi) = RGB_ALTITUDE_M;
            let altitude_m = rng.range(lo.ln(), hi.ln()).exp();
            let pitch_deg = if rng.chance(0.8) {
                90.0
            } else {
                rng.range(70.0, 90.0)
            };
            (vfov_deg, altitude_m, pitch_deg)
        };
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

        // no flood in RGB yet: real RGB val is all pasture, so start where it can judge
        let flood = (thermal && rng.chance(0.3)).then(|| aim[1] + rng.range(-1.5, 1.0));
        let conditions = Conditions::random(rng, flood);
        let ground_c: Vec<f64> = if thermal {
            world
                .vertices
                .iter()
                .zip(&world.normals)
                .map(|(p, n)| conditions.ground_c(world.seed, p[0], p[2], *n))
                .collect()
        } else {
            Vec::new()
        };
        let ground_at =
            |x: f64, z: f64| conditions.ground_c(world.seed, x, z, terrain.normal(x, z));

        // Ground half-height of the picture at the aim point (nadir); oblique views see more.
        let spread = altitude_m * (vfov_deg / 2.0).to_radians().tan();
        let near = |rng: &mut Rng, r: f64| (aim[0] + rng.range(-r, r), aim[2] + rng.range(-r, r));

        // the RGB ground first: a farmyard in it changes the herd (thermal draws nothing)
        let ground = (!thermal).then(|| {
            let (centre, size) = footprint(&camera, vfov_deg, aim[1]);
            Ground::random(rng, centre, size)
        });
        let layout = ground.as_ref().map(|g| &g.layout);
        let chosen = choose_targets(rng, kinds, layout.is_some_and(|l| l.yard.is_some()));
        let (cx, cz) = near(rng, 0.7 * spread);
        let group_r = rng.range(2.0, 0.8 * spread);
        let targets = kinds
            .iter()
            .zip(&chosen)
            .enumerate()
            .map(|(k, (&kind, &on))| {
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
                        transform: pose(rng, kind, feet, posed.get(k) == Some(&true)),
                        temp_c: [torso, limb, head],
                    }
                })
            })
            .collect();

        let stubble =
            layout.is_some_and(|l| l.parcels.iter().any(|p| p.field == rgb::Field::Stubble));
        // nothing else stands in a farmyard or its mud (vehicles may park on its hard standing)
        let yard = layout.and_then(|l| l.yard.as_ref());
        let in_yard = |x: f64, z: f64| yard.is_some_and(|y| y.holds(x, z));
        let hedges: Vec<_> = layout
            .map_or_else(Vec::new, |l| l.hedge_segments(rng, RGB_HEDGES))
            .into_iter()
            .filter(|h| !in_yard(h.0, h.1))
            .collect();
        let yard_parts = yard.map(|y| yard_clutter(y, terrain));
        let mut clutter = Vec::new();
        let mut clutter_c = Vec::new();
        for &(kind, max) in modality.clutter() {
            // houses and cars come in some scenes only (a farm, a village, a road)
            let n = match kind {
                ClutterKind::Bush => rng.index(max + 1),
                ClutterKind::Rock => rng.chance(0.4) as usize * (1 + rng.index(max)),
                ClutterKind::House => rng.chance(0.35) as usize * (1 + rng.index(max)),
                ClutterKind::Car => rng.chance(0.4) as usize * (1 + rng.index(max)),
                ClutterKind::Bale => (stubble && rng.chance(0.7)) as usize * (1 + rng.index(max)),
                ClutterKind::Vehicle => rng.chance(0.3) as usize * (1 + rng.index(max)),
                ClutterKind::Hedge => hedges.len().min(max),
                ClutterKind::Shed
                | ClutterKind::OpenShed
                | ClutterKind::Rail
                | ClutterKind::Feeder
                | ClutterKind::Stack => yard_parts.as_ref().map_or(0, |p| p.count(kind)),
            };
            let placed: Vec<Option<Transform>> = (0..max)
                .map(|i| {
                    if i >= n {
                        return None;
                    }
                    match kind {
                        ClutterKind::Hedge => return Some(hedge(rng, terrain, hedges[i])),
                        ClutterKind::Vehicle => {
                            let v = vehicle(rng, terrain, layout, &near, spread);
                            let (x, z) = (v.translation.x as f64, v.translation.z as f64);
                            return (!yard.is_some_and(|y| y.blocks(x, z))).then_some(v);
                        }
                        ClutterKind::Bale => {
                            let b = bale(rng, terrain, layout?, &near, spread)?;
                            let (x, z) = (b.translation.x as f64, b.translation.z as f64);
                            return (!in_yard(x, z)).then_some(b);
                        }
                        ClutterKind::Shed
                        | ClutterKind::OpenShed
                        | ClutterKind::Rail
                        | ClutterKind::Feeder
                        | ClutterKind::Stack => {
                            return yard_parts.as_ref()?.at(kind, i);
                        }
                        _ => {}
                    }
                    let (x, z) = near(rng, 1.5 * spread);
                    if in_yard(x, z) {
                        return None;
                    }
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
                        ClutterKind::Bale
                        | ClutterKind::Vehicle
                        | ClutterKind::Hedge
                        | ClutterKind::Shed
                        | ClutterKind::OpenShed
                        | ClutterKind::Rail
                        | ClutterKind::Feeder
                        | ClutterKind::Stack => unreachable!("placed above"),
                    }
                })
                .collect();
            let temps: Vec<f64> = (0..CLUTTER_MATS)
                .map(|i| match kind {
                    ClutterKind::Bush | ClutterKind::Hedge => conditions.vegetation_c(rng),
                    ClutterKind::Rock | ClutterKind::Bale => conditions.clutter_c(rng),
                    ClutterKind::House | ClutterKind::Shed | ClutterKind::OpenShed => {
                        conditions.building_c(rng)
                    }
                    ClutterKind::Car | ClutterKind::Vehicle if i < CLUTTER_MATS / 2 => {
                        conditions.vehicle_c(rng)
                    }
                    ClutterKind::Car | ClutterKind::Vehicle => conditions.glass_c(rng),
                    ClutterKind::Rail | ClutterKind::Feeder | ClutterKind::Stack => {
                        conditions.clutter_c(rng)
                    }
                })
                .collect();
            clutter.push(placed);
            clutter_c.push(temps);
        }
        let road_c = conditions.road_c(rng);

        let mut targets: Vec<Option<Placement>> = targets;
        if let Some(y) = yard {
            yard_herd(rng, y, terrain, kinds, &mut targets);
        }
        // Faint shots draw from a stream of their own (seeded by where the camera looks),
        // so every other draw, and the shots after, stay as they were.
        let mut faint_rng = Rng::new(aim[0].to_bits() ^ aim[2].to_bits().rotate_left(29) ^ 0xFA17);
        let faint = thermal && targets.iter().any(Option::is_some) && faint_rng.chance(faint_share);
        if faint {
            let scene = TempWindow::around(
                ground_c
                    .iter()
                    .copied()
                    .chain(clutter_c.iter().flatten().copied())
                    .chain([road_c])
                    .chain(conditions.water.map(|w| w.1)),
            );
            let per_grey = (scene.hi - scene.lo) / 255.0;
            for p in targets.iter_mut().flatten() {
                let at = p.transform.translation;
                let ground = ground_at(at.x as f64, at.z as f64);
                let old = p.temp_c[0] - ground;
                let grey = faint_rng.range(FAINT_GREY.0.ln(), FAINT_GREY.1.ln()).exp();
                let scale = grey * per_grey / old.abs().max(1e-6);
                for t in &mut p.temp_c {
                    *t = ground + (*t - ground) * scale;
                }
            }
        }
        let window = TempWindow::around(
            ground_c
                .iter()
                .copied()
                .chain(clutter_c.iter().flatten().copied())
                .chain([road_c])
                .chain(targets.iter().flatten().flat_map(|p| p.temp_c))
                .chain(conditions.water.map(|w| w.1)),
        );
        let rgb = ground.map(|ground| RgbLook {
            daylight: Daylight::random(rng),
            ground,
            road: rgb::road(rng),
            clutter: modality
                .clutter()
                .iter()
                .map(|&(kind, _)| {
                    // a yard's rails are all painted alike
                    if kind == ClutterKind::Rail {
                        let paint = rgb::metal(rng);
                        return (0..CLUTTER_MATS)
                            .map(|_| rgb::vary(rng, paint, 0.04))
                            .collect();
                    }
                    (0..CLUTTER_MATS)
                        .map(|i| match kind {
                            ClutterKind::Bush | ClutterKind::Hedge => rgb::canopy(rng),
                            ClutterKind::Rock => rgb::stone(rng),
                            ClutterKind::House => rgb::building(rng),
                            ClutterKind::Car | ClutterKind::Vehicle if i < CLUTTER_MATS / 2 => {
                                rgb::car(rng)
                            }
                            ClutterKind::Car | ClutterKind::Vehicle => rgb::glass(rng),
                            ClutterKind::Bale => rgb::bale(rng),
                            // first half roofs, metal, film; second half walls, hay, tyres
                            ClutterKind::Shed | ClutterKind::OpenShed if i < CLUTTER_MATS / 2 => {
                                rgb::roof(rng)
                            }
                            ClutterKind::Shed | ClutterKind::OpenShed => rgb::wall(rng),
                            ClutterKind::Rail => rgb::metal(rng),
                            ClutterKind::Feeder if i < CLUTTER_MATS / 2 => rgb::metal(rng),
                            ClutterKind::Feeder => rgb::fodder(rng),
                            ClutterKind::Stack if i < CLUTTER_MATS / 2 => rgb::film(rng),
                            ClutterKind::Stack => rgb::rubber(rng),
                        })
                        .collect()
                })
                .collect(),
            coats: kinds
                .iter()
                .zip(&targets)
                .map(|(&kind, p)| p.as_ref().map(|_| Coat::random(rng, kind)))
                .collect(),
        });
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
            faint,
            rgb,
            forest: None,
        }
    }

    pub fn n_targets(&self) -> usize {
        self.targets.iter().flatten().count()
    }

    /// A forest shot like BAMBI's: straight down over painted forest ground (`forest.rs`),
    /// a herd of deer or boar (sometimes people and a dog) much warmer than the ground.
    fn forest(rng: &mut Rng, world: &World, kinds: &[Kind], posed: &[bool]) -> Self {
        let terrain = &world.terrain;
        // a wide thermal lens; the frame's short side covers 15-60 m, so a deer is about
        // 13-50 px long in the 640x512 render (BAMBI: median box side 42 px of 1024)
        let vfov_deg = rng.range(45.0, 52.0);
        let cover_m = rng.range(10.0_f64.ln(), 60.0_f64.ln()).exp();
        let altitude_m = cover_m / 2.0 / (vfov_deg / 2.0).to_radians().tan();
        let pitch_deg = rng.range(87.0, 90.0);
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
        let up = Vec3::new(dir_x as f32, 0.0, dir_z as f32);
        let camera = Transform::from_translation(eye).looking_at(target, up);

        // dawn, day or night; by day the sun casts the crowns' shadows
        let night = rng.chance(0.3);
        let sun = (!night && rng.chance(0.7)).then(|| {
            let elevation = rng.range(10.0_f64, 55.0).to_radians();
            let azimuth = rng.range(0.0, std::f64::consts::TAU);
            [
                elevation.cos() * azimuth.cos(),
                elevation.sin(),
                elevation.cos() * azimuth.sin(),
            ]
        });
        let (centre, size) = footprint(&camera, vfov_deg, aim[1]);
        let table = NoiseTable::new(world.seed ^ 0xF0_4E57);
        let mut look = ForestLook::random(rng, centre, size, sun, &table);
        if let Some(c) = &mut look.canopy {
            // low flights fly over the canopy, not through it
            c.height_m = c.height_m.min(0.6 * altitude_m);
        }

        let chosen = forest_targets(rng, kinds);
        let spread = cover_m / 2.0;
        let (cx, cz) = (
            aim[0] + rng.range(-0.8, 0.8) * spread,
            aim[2] + rng.range(-0.8, 0.8) * spread,
        );
        // herds stand close: a few metres for boar, wider for deer
        let herd_r = rng.range(1.5, 12.0_f64.min(spread));
        let targets = kinds
            .iter()
            .zip(&chosen)
            .enumerate()
            .map(|(k, (&kind, &on))| {
                on.then(|| {
                    let (x, z) = (
                        cx + rng.normal() * herd_r / 2.0,
                        cz + rng.normal() * herd_r / 2.0,
                    );
                    let feet = Vec3::new(x as f32, terrain.height(x, z) as f32, z as f32);
                    let ground = look.ground_at(x, z);
                    // BAMBI's animals are near white on a darker ground: mostly many
                    // degrees warmer than anything around them, a few faint
                    let (lo, hi) = if kind == Kind::Person {
                        (3.0_f64, 20.0_f64)
                    } else {
                        (3.0, 25.0)
                    };
                    let torso = ground + rng.range(lo.ln(), hi.ln()).exp();
                    let limb = ground + (torso - ground) * rng.range(0.5, 0.85);
                    let head = ground + (torso - ground) * rng.range(1.0, 1.2);
                    let mut transform = pose(rng, kind, feet, posed.get(k) == Some(&true));
                    // red and fallow deer are bigger than the roe-sized primitive
                    if kind == Kind::Deer {
                        transform.scale *= rng.range(0.9, 1.5) as f32;
                    }
                    Placement {
                        transform,
                        temp_c: [torso, limb, head],
                    }
                })
            })
            .collect();

        // bodies warmer than the window's top come out white, as BAMBI's mostly do
        let leaves = (look.canopy_percentile(0.1), look.canopy_percentile(99.9));
        let window = TempWindow {
            lo: look.percentile(0.1).min(leaves.0.unwrap_or(f64::MAX)) - 0.5,
            hi: look.percentile(99.9).max(leaves.1.unwrap_or(f64::MIN)) + rng.range(1.0, 8.0),
        };
        // mottle on the bodies (the ground texture, `apply_temperatures`)
        let (texture, speckle, texture_seed) =
            (rng.range(0.05, 0.3), rng.range(0.0, 0.15), rng.next_u64());
        let conditions = Conditions {
            night,
            air_c: look.base_c,
            sun_c: 0.0,
            sun_dir: sun.unwrap_or([0.0, 1.0, 0.0]),
            cooling_c: 0.0,
            patch_c: 0.0,
            texture,
            speckle,
            texture_seed,
            water: None,
        };
        Self {
            altitude_m,
            pitch_deg,
            vfov_deg,
            camera,
            conditions,
            targets,
            clutter: Vec::new(),
            clutter_c: Vec::new(),
            road_c: look.base_c,
            ground_c: vec![look.base_c; world.vertices.len()],
            window,
            faint: false,
            rgb: None,
            forest: Some(look),
        }
    }
}

/// Who is in a forest shot: mostly a herd of deer or of boar (1-12), sometimes people
/// with a dog, a mixed group, or nobody.
fn forest_targets(rng: &mut Rng, kinds: &[Kind]) -> Vec<bool> {
    let u = rng.unit();
    let first_n = |kind: Kind, n: usize| {
        let mut k = 0;
        kinds
            .iter()
            .map(|&x| {
                let on = x == kind && k < n;
                k += (x == kind) as usize;
                on
            })
            .collect::<Vec<bool>>()
    };
    if u < 0.12 {
        return vec![false; kinds.len()];
    }
    if u < 0.22 {
        let people = first_n(Kind::Person, 1 + rng.index(3));
        let dog = first_n(Kind::Dog, rng.chance(0.5) as usize);
        return people.iter().zip(&dog).map(|(a, b)| *a || *b).collect();
    }
    if u < 0.95 {
        let herd = if rng.chance(0.5) {
            Kind::Deer
        } else {
            Kind::Boar
        };
        let most = kinds.iter().filter(|&&k| k == herd).count();
        return first_n(herd, 1 + rng.index(most));
    }
    let mut chosen: Vec<bool> = kinds.iter().map(|_| rng.chance(0.25)).collect();
    if !chosen.contains(&true) {
        chosen[rng.index(kinds.len())] = true;
    }
    chosen
}

/// The ground the camera sees: a square (centre x, z and side, metres) around where the
/// frame's corners meet the plane at `ground_y`, with a margin for the terrain's relief.
fn footprint(camera: &Transform, vfov_deg: f64, ground_y: f64) -> ([f64; 2], f64) {
    let tan_v = (vfov_deg / 2.0).to_radians().tan() as f32;
    let tan_h = tan_v * RGB_W as f32 / RGB_H as f32;
    let eye = camera.translation;
    let (mut lo, mut hi) = ([f64::MAX; 2], [f64::MIN; 2]);
    for (sx, sy) in [(-1.0, -1.0), (1.0, -1.0), (-1.0, 1.0), (1.0, 1.0)] {
        let dir = camera.rotation * Vec3::new(sx * tan_h, sy * tan_v, -1.0);
        // the top edge looks at least 40 deg down in RGB shots, so every ray hits
        let t = (ground_y as f32 - eye.y) / dir.y.min(-1e-3);
        let p = eye + dir * t;
        for (k, v) in [p.x as f64, p.z as f64].into_iter().enumerate() {
            lo[k] = lo[k].min(v);
            hi[k] = hi[k].max(v);
        }
    }
    let centre = [(lo[0] + hi[0]) / 2.0, (lo[1] + hi[1]) / 2.0];
    let size = ((hi[0] - lo[0]).max(hi[1] - lo[1]) * 1.1 + 20.0).min(600.0);
    (centre, size)
}

/// A hedge segment at (x, z) running along `a`: 4.5-6.5 m long, 1.5-3.5 m high.
fn hedge(rng: &mut Rng, terrain: &Terrain, (x, z, a): (f64, f64, [f64; 2])) -> Transform {
    let (len, h, w) = (
        rng.range(4.5, 6.5),
        rng.range(1.5, 3.5),
        rng.range(1.5, 2.8),
    );
    // local x along the hedge
    let yaw = (-a[1]).atan2(a[0]) as f32;
    Transform::from_xyz(x as f32, (terrain.height(x, z) + h * 0.4) as f32, z as f32)
        .with_rotation(Quat::from_rotation_y(yaw))
        .with_scale(Vec3::new(len as f32, h as f32, w as f32))
}

/// A vehicle: parked on a track, along it, 60% of the time if there is one; else anywhere
/// near. Its parts carry its size (`clutter_parts`).
fn vehicle(
    rng: &mut Rng,
    terrain: &Terrain,
    layout: Option<&Layout>,
    near: &impl Fn(&mut Rng, f64) -> (f64, f64),
    spread: f64,
) -> Transform {
    let spot = layout.and_then(|l| {
        if rng.chance(0.6) {
            l.track_spot(rng)
        } else {
            None
        }
    });
    let (x, z, yaw) = match spot {
        // local z along the track
        Some((x, z, a)) => (x, z, a[0].atan2(a[1])),
        None => {
            let (x, z) = near(rng, 1.5 * spread);
            (x, z, rng.range(0.0, std::f64::consts::TAU))
        }
    };
    Transform::from_xyz(x as f32, terrain.height(x, z) as f32, z as f32)
        .with_rotation(Quat::from_rotation_y(yaw as f32))
}

/// A bale on a stubble field (`None` if the spot isn't one), moved onto the middle of the
/// nearest swath if the field has them: balers leave bales in the swath rows.
fn bale(
    rng: &mut Rng,
    terrain: &Terrain,
    layout: &Layout,
    near: &impl Fn(&mut Rng, f64) -> (f64, f64),
    spread: f64,
) -> Option<Transform> {
    let (mut x, mut z) = near(rng, 1.5 * spread);
    let yaw = rng.range(0.0, std::f64::consts::TAU) as f32;
    let s = rng.range(0.9, 1.1) as f32;
    let p = layout.parcel_at(x, z);
    if p.field != rgb::Field::Stubble {
        return None;
    }
    if let Some((spacing, _)) = p.swaths {
        // t runs across the rows, along (dir.z, -dir.x)
        let t = x * p.dir[1] - z * p.dir[0];
        let shift = (t / spacing).floor() * spacing + spacing / 2.0 - t;
        x += p.dir[1] * shift;
        z -= p.dir[0] * shift;
        // the move can cross a field edge
        if layout.parcel_at(x, z).field != rgb::Field::Stubble {
            return None;
        }
    }
    Some(
        Transform::from_xyz(x as f32, terrain.height(x, z) as f32, z as f32)
            .with_rotation(Quat::from_rotation_y(yaw))
            .with_scale(Vec3::splat(s)),
    )
}

/// Where a yard's sheds, rail panels, feeders and stacks go, per kind.
struct YardParts {
    sheds: Vec<Transform>,
    open_sheds: Vec<Transform>,
    rails: Vec<Transform>,
    /// By mesh slot: `yard::FEEDS` in order (ring feeders, troughs, tanks); empty slots
    /// are `None`.
    feeders: Vec<Option<Transform>>,
    stacks: Vec<Transform>,
}

impl YardParts {
    /// How many meshes of `kind` the shot may place.
    fn count(&self, kind: ClutterKind) -> usize {
        match kind {
            ClutterKind::Shed => self.sheds.len(),
            ClutterKind::OpenShed => self.open_sheds.len(),
            ClutterKind::Rail => self.rails.len(),
            ClutterKind::Feeder => self.feeders.len(),
            ClutterKind::Stack => self.stacks.len(),
            _ => 0,
        }
    }

    fn at(&self, kind: ClutterKind, i: usize) -> Option<Transform> {
        match kind {
            ClutterKind::Shed => self.sheds.get(i).copied(),
            ClutterKind::OpenShed => self.open_sheds.get(i).copied(),
            ClutterKind::Rail => self.rails.get(i).copied(),
            ClutterKind::Feeder => self.feeders.get(i).copied().flatten(),
            ClutterKind::Stack => self.stacks.get(i).copied(),
            _ => None,
        }
    }
}

/// A yard's feeders by mesh slot: each kind in its own block of slots (`yard::FEEDS`).
fn feeder_slots(
    y: &Yard,
    on_ground: impl Fn([f64; 2], f64) -> Transform,
) -> Vec<Option<Transform>> {
    let mut slots = Vec::new();
    for (feed, max) in FEEDS {
        let mut these: Vec<Option<Transform>> = y
            .feeders
            .iter()
            .filter(|f| f.2 == feed)
            .map(|f| Some(on_ground(f.0, f.1)))
            .collect();
        these.resize(max, None);
        slots.extend(these);
    }
    slots
}

/// The yard's 3D parts: sheds scaled to their size and sunk to their lowest corner (local
/// x along the row, z towards the pens), rails in 3 m panels (shorter to fit a run).
fn yard_clutter(y: &Yard, terrain: &Terrain) -> YardParts {
    let on_ground = |p: [f64; 2], turn: f64| {
        let (x, z) = y.world(p);
        Transform::from_xyz(x as f32, terrain.height(x, z) as f32, z as f32)
            .with_rotation(Quat::from_rotation_y(y.yaw(turn) as f32))
    };
    let shed = |s: &Shed| {
        let r = &s.rect;
        let corners = [(-1.0, -1.0), (1.0, -1.0), (-1.0, 1.0), (1.0, 1.0)].map(|(a, b)| {
            let (x, z) = y.world([r.c[0] + a * r.half[0], r.c[1] + b * r.half[1]]);
            terrain.height(x, z)
        });
        let (x, z) = y.world(r.c);
        let ground = terrain.height(x, z);
        let base = corners.iter().copied().fold(ground, f64::min) - 0.5;
        let top = ground + s.height;
        Transform::from_xyz(x as f32, ((base + top) / 2.0) as f32, z as f32)
            .with_rotation(Quat::from_rotation_y(y.yaw(0.0) as f32))
            .with_scale(Vec3::new(
                (2.0 * r.half[0]) as f32,
                (top - base) as f32,
                (2.0 * r.half[1]) as f32,
            ))
    };
    let mut rails = Vec::new();
    for (p, q) in y.rail_runs() {
        let (du, dv) = (q[0] - p[0], q[1] - p[1]);
        let len = du.hypot(dv);
        let n = (len / RAIL_M).ceil().max(1.0) as usize;
        let turn = dv.atan2(du);
        for k in 0..n {
            let f = (k as f64 + 0.5) / n as f64;
            let tf = on_ground([p[0] + du * f, p[1] + dv * f], turn);
            rails.push(tf.with_scale(Vec3::new((len / n as f64 / RAIL_M) as f32, 1.0, 1.0)));
        }
    }
    YardParts {
        sheds: y.sheds.iter().filter(|s| !s.open).map(shed).collect(),
        open_sheds: y.sheds.iter().filter(|s| s.open).map(shed).collect(),
        rails,
        feeders: feeder_slots(y, on_ground),
        stacks: y.stacks.iter().map(|&(p, t)| on_ground(p, t)).collect(),
    }
}

/// Moves the shot's bodies into the yard: the whole herd most of the time, else only those
/// that landed on a building. Most of the herd packs into one area (a pen, the front of an
/// open shed, the mud): usually close and lined up (`Yard::herd_slots`: heads in round a
/// ring feeder, rows facing a rail), else scattered; the rest anywhere in the yard, centres
/// at least 1.4 m apart. People never take a packed slot.
fn yard_herd(
    rng: &mut Rng,
    y: &Yard,
    terrain: &Terrain,
    kinds: &[Kind],
    targets: &mut [Option<Placement>],
) {
    let areas = y.herd_areas();
    if areas.is_empty() {
        return;
    }
    let pick_area = |rng: &mut Rng| {
        let total: f64 = areas.iter().map(|a| a.0).sum();
        let mut u = rng.unit() * total;
        for (w, a) in &areas {
            if u < *w {
                return *a;
            }
            u -= w;
        }
        areas[areas.len() - 1].1
    };
    let all = rng.chance(YARD_HERD);
    let home = pick_area(rng);
    let n = targets.iter().flatten().count();
    let slots = if rng.chance(YARD_PACKED) {
        y.herd_slots(rng, &home, n)
    } else {
        Vec::new()
    };
    let mut free = slots.iter();
    // packed cows are placed exactly; everyone else keeps clear of all the slots
    let mut taken: Vec<(f64, f64)> = slots.iter().map(|s| y.world(s.p)).collect();
    for (&kind, p) in kinds.iter().zip(targets.iter_mut()) {
        let Some(p) = p else { continue };
        let t = p.transform.translation;
        let (x, z) = (t.x as f64, t.z as f64);
        if !all && !y.blocks(x, z) {
            continue;
        }
        // keep its pose: lying bodies stay as far into the ground
        let lift = t.y as f64 - terrain.height(x, z);
        let animal = kind != Kind::Person;
        let home_bound = rng.chance(0.8);
        if animal
            && home_bound
            && let Some(s) = free.next()
        {
            let (sx, sz) = y.world(s.p);
            let lift = if s.stand { lift.max(0.0) } else { lift };
            p.transform.translation =
                Vec3::new(sx as f32, (terrain.height(sx, sz) + lift) as f32, sz as f32);
            p.transform.rotation = facing(y, s.facing);
            continue;
        }
        let area = if home_bound { home } else { pick_area(rng) };
        let spot = (0..12)
            .map(|k| {
                // a full area overflows into the rest of the yard
                let a = if k < 8 { area } else { pick_area(rng) };
                y.spot_in(rng, &a)
            })
            .find(|s| {
                let (u, v) = y.local(s.0, s.1);
                y.off_feeders([u, v]) && taken.iter().all(|q| (q.0 - s.0).hypot(q.1 - s.1) > 1.4)
            });
        let Some((sx, sz)) = spot else { continue };
        p.transform.translation =
            Vec3::new(sx as f32, (terrain.height(sx, sz) + lift) as f32, sz as f32);
        taken.push((sx, sz));
    }
}

/// Turn that points a body's front (+z) along yard direction `facing`.
fn facing(y: &Yard, facing: f64) -> Quat {
    Quat::from_rotation_y(y.yaw(facing - std::f64::consts::FRAC_PI_2) as f32)
}

/// Which pool targets are in a shot: ~30% empty frames (like the real data), else a group
/// of people, a herd of one kind, or a mix. Herds in a farmyard are mostly cattle and run
/// bigger (the larger of two draws): real yard cows stand in groups of 6 at the median, a third of them in 10 or
/// more (train flights 060, 062, Derval 012/014).
fn choose_targets(rng: &mut Rng, kinds: &[Kind], yard: bool) -> Vec<bool> {
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
        // the pool's animal kinds in pool order (for the thermal pool, POOL's order)
        let mut animals: Vec<Kind> = Vec::new();
        for &k in kinds {
            if k != Kind::Person && !animals.contains(&k) {
                animals.push(k);
            }
        }
        // farmyards are mostly cattle yards
        let herd = if yard && animals.contains(&Kind::Cattle) && rng.chance(0.7) {
            Kind::Cattle
        } else {
            animals[rng.index(animals.len())]
        };
        let most = kinds.iter().filter(|&&k| k == herd).count();
        let mut n = 1 + rng.index(most);
        if yard {
            n = n.max(1 + rng.index(most));
        }
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

/// Random pose and size at `feet`: people stand, sit or lie; animals stand or lie. A
/// `posed` body (its posture is in its mesh) is only turned and scaled, with the same
/// random draws, so shots match those of primitive bodies.
fn pose(rng: &mut Rng, kind: Kind, feet: Vec3, posed: bool) -> Transform {
    let yaw = Quat::from_rotation_y(rng.range(0.0, std::f64::consts::TAU) as f32);
    let s = rng.range(0.85, 1.15) as f32;
    if kind == Kind::Person {
        let u = rng.unit();
        if u < 0.5 || posed {
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

pub fn window_plugin(title: &str, modality: Modality) -> WindowPlugin {
    let (w, h) = modality.size();
    WindowPlugin {
        primary_window: Some(Window {
            title: title.into(),
            // Exactly the sensor size, whatever the desktop scaling.
            resolution: bevy::window::WindowResolution::new(w, h).with_scale_factor_override(1.0),
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

/// A picture material: unlit (the colour is the temperature's grey) for thermal, lit and
/// matt for RGB (fields, coats and roofs barely shine from 100 m).
fn picture(mats: &mut Assets<StandardMaterial>, modality: Modality) -> Handle<StandardMaterial> {
    mats.add(StandardMaterial {
        base_color: Color::BLACK,
        unlit: modality != Modality::Rgb,
        perceptual_roughness: 0.95,
        reflectance: 0.15,
        cull_mode: None,
        ..default()
    })
}

/// Spawn the camera, terrain, water, clutter and the body pool (`kinds`, in order; ids
/// 1, 2, ...); for RGB also the sun. Everything is placed per shot by `SceneCtl::apply`.
#[allow(clippy::too_many_arguments)]
pub fn spawn_scene(
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
    images: &mut Assets<Image>,
    mats: &mut Assets<StandardMaterial>,
    world: &World,
    kinds: &[Kind],
    models: &BodyModels,
    modality: Modality,
) {
    assert!(
        kinds.len() * ID_STEP as usize <= 255,
        "too many targets for the id render"
    );
    let rgb = modality == Modality::Rgb;
    let forest = modality == Modality::Forest;
    let camera = commands
        .spawn((
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
            // must stay exact; the sensor model blurs the thermal frame instead). RGB
            // pictures get MSAA (`show_ids` turns it off for the id look), like a photo
            // scaled down from 5 K.
            Tonemapping::None,
            DebandDither::Disabled,
            Msaa::Off,
            AerialCamera,
        ))
        .id();
    if rgb {
        commands
            .entity(camera)
            .insert((Msaa::Sample4, Exposure { ev100: RGB_EV100 }));
        commands.spawn((
            DirectionalLight {
                shadow_maps_enabled: true,
                ..default()
            },
            // from the camera (60-150 m up) to the far edge of an oblique frame
            CascadeShadowConfigBuilder {
                num_cascades: 2,
                minimum_distance: 1.0,
                first_cascade_far_bound: 180.0,
                maximum_distance: 500.0,
                ..default()
            }
            .build(),
            Transform::default(),
            Sun,
        ));
        // ~6 cm per texel near the camera: a cow's shadow keeps its outline
        commands.insert_resource(DirectionalLightShadowMap { size: 4096 });
    }
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
        unlit: !rgb,
        perceptual_roughness: 1.0,
        reflectance: 0.1,
        cull_mode: None,
        ..default()
    });
    commands.spawn((
        Mesh3d(terrain_mesh.clone()),
        MeshMaterial3d(ground.clone()),
        Look {
            picture: ground,
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
            picture: water.clone(),
            id: background.clone(),
        },
    ));

    let roads = picture(mats, modality);
    // RGB and forest paint their tracks and roads into the ground image instead
    if modality == Modality::Thermal {
        commands.spawn((
            Mesh3d(meshes.add(road_mesh(world))),
            MeshMaterial3d(roads.clone()),
            Look {
                picture: roads.clone(),
                id: background.clone(),
            },
        ));
    }
    // RGB and forest: the painted ground (`fields.rs`, `forest.rs`) on a copy of the
    // terrain under the camera, a few centimetres up; rebuilt every shot
    let overlay = (rgb || forest).then(|| {
        let image = images.add(overlay_image(if forest { FOREST_TEX } else { OVERLAY_TEX }));
        let mesh = meshes.add(terrain_patch(world, [0.0, 0.0], 10.0, PATCH_LIFT_M));
        let mat = mats.add(StandardMaterial {
            base_color: Color::WHITE,
            base_color_texture: Some(image.clone()),
            // forest: the texture's grey is the temperature, as for every thermal material
            unlit: forest,
            perceptual_roughness: 1.0,
            reflectance: 0.1,
            cull_mode: None,
            ..default()
        });
        commands.spawn((
            Mesh3d(mesh.clone()),
            MeshMaterial3d(mat.clone()),
            NotShadowCaster,
            Look {
                picture: mat,
                id: background.clone(),
            },
        ));
        (mesh, image)
    });
    // forest: a leaf canopy above the ground, cut out by its alpha; in the id look its
    // leaves are background, so bodies under them get no pixels
    let canopy = forest.then(|| {
        let picture = images.add(overlay_image(CANOPY_TEX));
        let mask = images.add(overlay_image(CANOPY_TEX));
        let mesh = meshes.add(terrain_patch(world, [0.0, 0.0], 10.0, 10.0));
        let leaves = |mats: &mut Assets<StandardMaterial>, color: Color, image: &Handle<Image>| {
            mats.add(StandardMaterial {
                base_color: color,
                base_color_texture: Some(image.clone()),
                unlit: true,
                alpha_mode: AlphaMode::Mask(0.5),
                cull_mode: None,
                ..default()
            })
        };
        let pic = leaves(mats, Color::WHITE, &picture);
        let id = leaves(mats, Color::srgb(0.0, 1.0, 0.0), &mask);
        commands.spawn((
            Mesh3d(mesh.clone()),
            MeshMaterial3d(pic.clone()),
            Look { picture: pic, id },
        ));
        (mesh, picture, mask)
    });

    let block = meshes.add(Cuboid::new(1.0, 1.0, 1.0));
    let cylinder = meshes.add(Cylinder::new(0.5, 1.0));
    let hoop = meshes.add(Torus::new(0.94, 1.06));
    let tyre = meshes.add(Torus::new(0.25, 0.51));
    let mut shapes = Rng::new(world.seed ^ 0x5_4A9E);
    let canopies: Vec<_> = (0..8)
        .map(|_| meshes.add(lumpy_blob(&mut shapes, 0.35, 0.3)))
        .collect();
    let stones: Vec<_> = (0..4)
        .map(|_| meshes.add(lumpy_blob(&mut shapes, 0.15, 0.15)))
        .collect();
    // RGB canopies: leaf clusters and gaps, much coarser contrast than grass
    let foliage = rgb.then(|| {
        images.add(texture_image(ground_texture(
            world.seed ^ 0xF0_11A6E,
            TEX_SIZE,
            0.35,
            0.25,
        )))
    });
    let sheets = rgb.then(|| images.add(texture_image(sheet_texture(world.seed ^ 0x5EE7))));
    let mut clutter = Vec::new();
    for &(kind, n) in modality.clutter() {
        let set: Vec<_> = (0..CLUTTER_MATS)
            .map(|_| {
                let m = picture(mats, modality);
                // canopies are mottled like the ground (thermal) or leafy (RGB)
                if matches!(kind, ClutterKind::Bush | ClutterKind::Hedge) {
                    let t = foliage.clone().unwrap_or_else(|| texture.clone());
                    mats.get_mut(&m).unwrap().base_color_texture = Some(t);
                }
                if matches!(kind, ClutterKind::Shed | ClutterKind::OpenShed) {
                    mats.get_mut(&m).unwrap().base_color_texture = sheets.clone();
                }
                m
            })
            .collect();
        for slot in 0..n {
            let shapes_of = ClutterShapes {
                canopies: &canopies,
                stones: &stones,
                block: &block,
                cylinder: &cylinder,
                hoop: &hoop,
                tyre: &tyre,
            };
            let parts = clutter_parts(&mut shapes, kind, slot, &shapes_of);
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
                                picture: m,
                                id: background.clone(),
                            },
                        ));
                    }
                });
        }
        clutter.push(set);
    }

    // every model and pose of a kind, meshed once and shared by its pool bodies
    let model_variants: Vec<(Kind, Vec<BodyVariant>)> = models
        .0
        .iter()
        .map(|(kind, m)| (*kind, model_variants(meshes, m)))
        .collect();
    for (k, &kind) in kinds.iter().enumerate() {
        let id = k as u32 + 1;
        let materials = [(); 3].map(|_| picture(mats, modality));
        // forest bodies are mottled (fur, wet patches; BAMBI's close boar are not flat white)
        if forest {
            for m in &materials {
                mats.get_mut(m).unwrap().base_color_texture = Some(texture.clone());
            }
        }
        let id_mat = unlit(mats, Color::srgb_u8(id as u8 * ID_STEP, 0, 0));
        // drawn for model kinds too, so the primitive bodies after them stay the same
        let primitive = body_parts(meshes, &mut shapes, kind);
        let modelled = model_variants.iter().find(|(k, _)| *k == kind);
        let variants = match modelled {
            Some((_, v)) => v.clone(),
            None => vec![("primitive".to_string(), primitive)],
        };
        let target = commands
            .spawn((Transform::default(), Visibility::Hidden))
            .id();
        let entities: Vec<Entity> = variants
            .iter()
            .enumerate()
            .map(|(v, (_, parts))| {
                let shown = if v == 0 {
                    Visibility::Inherited
                } else {
                    Visibility::Hidden
                };
                commands
                    .spawn((Transform::default(), shown, Variant, ChildOf(target)))
                    .with_children(|body| {
                        for (mesh, tf, part) in parts {
                            body.spawn((
                                Mesh3d(mesh.clone()),
                                MeshMaterial3d(materials[*part].clone()),
                                *tf,
                                Look {
                                    picture: materials[*part].clone(),
                                    id: id_mat.clone(),
                                },
                            ));
                        }
                    })
                    .id()
            })
            .collect();
        commands.entity(target).insert(Target {
            kind,
            id,
            bodies: variants.into_iter().map(|(name, _)| name).collect(),
            variants: entities,
            posed: kind == Kind::Person && modelled.is_some(),
            picture: materials.clone(),
        });
    }
    let patches = if rgb {
        (0..PATCH_TEXTURES as u64)
            .map(|k| images.add(patch_image(world.seed ^ (0x9A7C + k))))
            .collect()
    } else {
        Vec::new()
    };
    commands.insert_resource(SceneHandles {
        terrain_mesh,
        texture,
        water,
        roads,
        clutter,
        patches,
        overlay,
        canopy,
        table: rgb.then(|| NoiseTable::new(world.seed ^ 0x7AB1E)),
    });
}

/// The painted ground image (`fields.rs`, `forest.rs`), `n` texels square, replaced every
/// shot.
fn overlay_image(n: usize) -> Image {
    let n = n as u32;
    let mut img = Image::new(
        Extent3d {
            width: n,
            height: n,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        vec![128; (n * n * 4) as usize],
        TextureFormat::Rgba8UnormSrgb,
        RenderAssetUsages::default(),
    );
    img.sampler = ImageSampler::Descriptor(ImageSamplerDescriptor::linear());
    img
}

/// The terrain cells under a square (`centre`, `size` metres across), `lift` metres above
/// the terrain, with texture coordinates running 0-1 across the square. Built from the
/// terrain's own vertices, so it follows the ground exactly and never dips into it.
fn terrain_patch(world: &World, centre: [f64; 2], size: f64, lift: f64) -> Mesh {
    let (w, d) = (world.terrain.field.width(), world.terrain.field.depth());
    let v0 = world.vertices[0];
    let (x0, z0) = (centre[0] - size / 2.0, centre[1] - size / 2.0);
    let cell = |v: f64, origin: f64, n: usize| ((v - origin) / CELL_M).clamp(0.0, (n - 1) as f64);
    let (i0, i1) = (
        cell(x0, v0[0], w).floor() as usize,
        cell(x0 + size, v0[0], w).ceil() as usize,
    );
    let (j0, j1) = (
        cell(z0, v0[2], d).floor() as usize,
        cell(z0 + size, v0[2], d).ceil() as usize,
    );
    let (mut pos, mut normals, mut uvs) = (Vec::new(), Vec::new(), Vec::new());
    for j in j0..=j1 {
        for i in i0..=i1 {
            let p = world.vertices[j * w + i];
            let n = world.normals[j * w + i];
            pos.push([p[0] as f32, (p[1] + lift) as f32, p[2] as f32]);
            normals.push([n[0] as f32, n[1] as f32, n[2] as f32]);
            uvs.push([((p[0] - x0) / size) as f32, ((p[2] - z0) / size) as f32]);
        }
    }
    let cols = (i1 - i0 + 1) as u32;
    let mut idx = Vec::new();
    for j in 0..(j1 - j0) as u32 {
        for i in 0..cols - 1 {
            let k = j * cols + i;
            let (r, b) = (k + 1, k + cols);
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
    .with_inserted_indices(Indices::U32(idx))
}

/// Black patches on white (Holstein cattle, piebald horses and dogs), wrapped round a
/// primitive torso by its texture coordinates: a few big blobs around it, a couple along.
fn patch_image(seed: u64) -> Image {
    let (w, h) = (64usize, 32usize);
    let mut grey = Vec::with_capacity(w * h);
    for j in 0..h {
        for i in 0..w {
            let (u, v) = (i as f64 / w as f64, j as f64 / h as f64);
            let n = crate::thermal::tiling_fbm(seed, u, v, 4.0);
            grey.push(if n > 0.05 { 18 } else { 255 });
        }
    }
    let mut img = Image::new(
        Extent3d {
            width: w as u32,
            height: h as u32,
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

/// The meshes clutter is built from.
struct ClutterShapes<'a> {
    canopies: &'a [Handle<Mesh>],
    stones: &'a [Handle<Mesh>],
    /// Unit cube and unit cylinder (radius 0.5, height 1).
    block: &'a Handle<Mesh>,
    cylinder: &'a Handle<Mesh>,
    /// A ring feeder's hoop (1 m radius) and a tyre (0.38 m), lying flat.
    hoop: &'a Handle<Mesh>,
    tyre: &'a Handle<Mesh>,
}

/// Parts of one clutter instance in unit space (the shot scales the whole thing; bales
/// come at their real size), each with an index into its kind's materials: the first half
/// are the kind's main temperatures, the second half its secondary ones (car glass, the
/// other roof half). A real model took smooth uniform discs and plain rectangles for
/// animals (zero-shot test, 2026-10-28), so clutter is made irregular and two-tone, unlike
/// bodies. Bales are the exception: they are the smooth look-alikes real fields have.
fn clutter_parts(
    rng: &mut Rng,
    kind: ClutterKind,
    slot: usize,
    shapes: &ClutterShapes,
) -> Vec<(Handle<Mesh>, Transform, usize)> {
    let ClutterShapes {
        canopies,
        stones,
        block,
        ..
    } = *shapes;
    let half = CLUTTER_MATS / 2;
    let b = |x: f32, y: f32, z: f32, sx: f32, sy: f32, sz: f32, m: usize| {
        (
            block.clone(),
            Transform::from_xyz(x, y, z).with_scale(Vec3::new(sx, sy, sz)),
            m,
        )
    };
    match kind {
        ClutterKind::Shed | ClutterKind::OpenShed => {
            // unit box (the shot scales it), x along the row, +z towards the pens; a
            // low-pitched roof in two halves along the ridge, some with lighter roof lights
            let (roof, wall) = (rng.index(half), half + rng.index(half));
            let pitch = rng.range(0.06, 0.14) as f32;
            let mut parts = vec![
                (
                    block.clone(),
                    Transform::from_xyz(0.0, 0.47, -0.25)
                        .with_rotation(Quat::from_rotation_x(pitch))
                        .with_scale(Vec3::new(1.02, 0.03, 0.53)),
                    roof,
                ),
                (
                    block.clone(),
                    Transform::from_xyz(0.0, 0.47, 0.25)
                        .with_rotation(Quat::from_rotation_x(-pitch))
                        .with_scale(Vec3::new(1.02, 0.03, 0.53)),
                    roof,
                ),
            ];
            if rng.chance(0.4) {
                let light = (roof + 1 + rng.index(half - 1)) % half;
                let n = 3 + rng.index(6);
                for k in 0..n {
                    let x = (k as f32 + 0.5) / n as f32 - 0.5;
                    let z = if rng.chance(0.5) { 0.25 } else { -0.25 };
                    let tilt = if z > 0.0 { -pitch } else { pitch };
                    parts.push((
                        block.clone(),
                        Transform::from_xyz(x, 0.49, z)
                            .with_rotation(Quat::from_rotation_x(tilt))
                            .with_scale(Vec3::new(0.02, 0.03, 0.4)),
                        light,
                    ));
                }
            }
            if kind == ClutterKind::Shed {
                parts.push(b(0.0, -0.03, 0.0, 0.98, 0.94, 0.98, wall));
            } else {
                // back wall, side walls with an open gap above, posts along the front
                parts.push(b(0.0, -0.05, -0.49, 0.98, 0.9, 0.02, wall));
                for x in [-0.49, 0.49] {
                    parts.push(b(x, -0.2, -0.1, 0.02, 0.6, 0.78, wall));
                }
                let n = 4 + rng.index(5);
                for k in 0..=n {
                    let x = k as f32 / n as f32 - 0.5;
                    parts.push(b(x * 0.98, -0.03, 0.48, 0.012, 0.94, 0.02, wall));
                }
            }
            parts
        }
        ClutterKind::Rail => {
            // a 3 m panel along local x (metres): posts at the ends, two or three rails
            let m = rng.index(CLUTTER_MATS);
            let len = RAIL_M as f32;
            let mut parts = vec![
                b(-len / 2.0, 0.65, 0.0, 0.1, 1.3, 0.1, m),
                b(len / 2.0, 0.65, 0.0, 0.1, 1.3, 0.1, m),
            ];
            let rails: &[f32] = if rng.chance(0.6) {
                &[0.45, 0.8, 1.15]
            } else {
                &[0.6, 1.1]
            };
            for &y in rails {
                parts.push(b(0.0, y, 0.0, len, 0.07, 0.07, m));
            }
            parts
        }
        ClutterKind::Feeder => {
            let (metal, fodder) = (rng.index(half), half + rng.index(half));
            // which feeder is fixed by the slot, so the shot knows where the rings are
            let mut feed = Feed::Tank;
            let mut first = 0;
            for (f, max) in FEEDS {
                if slot < first + max {
                    feed = f;
                    break;
                }
                first += max;
            }
            if feed == Feed::Ring {
                // ring feeder round a bale of hay
                vec![
                    (
                        shapes.hoop.clone(),
                        Transform::from_xyz(0.0, 0.9, 0.0),
                        metal,
                    ),
                    (
                        shapes.hoop.clone(),
                        Transform::from_xyz(0.0, 0.3, 0.0),
                        metal,
                    ),
                    (
                        shapes.cylinder.clone(),
                        Transform::from_xyz(0.0, 0.5, 0.0).with_scale(Vec3::new(1.5, 1.0, 1.5)),
                        fodder,
                    ),
                ]
            } else if feed == Feed::Trough {
                // trough
                vec![b(0.0, 0.3, 0.0, 3.0, 0.5, 0.65, metal)]
            } else {
                // water tank
                vec![(
                    shapes.cylinder.clone(),
                    Transform::from_xyz(0.0, 0.45, 0.0).with_scale(Vec3::new(1.6, 0.9, 1.6)),
                    fodder,
                )]
            }
        }
        ClutterKind::Stack => {
            let (film, rubber) = (rng.index(half), half + rng.index(half));
            let u = rng.unit();
            if u < 0.7 {
                // a wrapped bale, lying along the row or on end
                let tf = if rng.chance(0.6) {
                    Transform::from_xyz(0.0, 0.62, 0.0)
                        .with_rotation(Quat::from_rotation_z(FRAC_PI_2))
                        .with_scale(Vec3::new(1.25, 1.2, 1.25))
                } else {
                    Transform::from_xyz(0.0, 0.6, 0.0).with_scale(Vec3::new(1.25, 1.2, 1.25))
                };
                vec![(shapes.cylinder.clone(), tf, film)]
            } else if u < 0.9 {
                // a stack of tyres
                (0..2 + rng.index(4))
                    .map(|k| {
                        (
                            shapes.tyre.clone(),
                            Transform::from_xyz(0.0, 0.12 + 0.24 * k as f32, 0.0),
                            rubber,
                        )
                    })
                    .collect()
            } else {
                // a tank in its cage
                vec![
                    b(0.0, 0.6, 0.0, 1.0, 1.0, 1.2, film),
                    b(0.0, 0.08, 0.0, 1.1, 0.16, 1.3, rubber),
                ]
            }
        }
        ClutterKind::Hedge => {
            // 2-3 canopy blobs along local x, overlapping into one ragged segment
            let k = 2 + rng.index(2);
            (0..k)
                .map(|j| {
                    let x = (j as f32 + 0.5) / k as f32 - 0.5;
                    let tf = Transform::from_xyz(x, rng.range(-0.1, 0.1) as f32, 0.0).with_scale(
                        Vec3::new(
                            rng.range(0.35, 0.55) as f32,
                            rng.range(0.4, 0.55) as f32,
                            rng.range(0.4, 0.55) as f32,
                        ),
                    );
                    (
                        canopies[rng.index(canopies.len())].clone(),
                        tf,
                        rng.index(CLUTTER_MATS),
                    )
                })
                .collect()
        }
        ClutterKind::Vehicle => {
            // metres, length along local z, front at +z; body colour and dark glass (also
            // tyres and a pickup's bed liner)
            let (body, glass) = (rng.index(half), half + rng.index(half));
            let b = |x: f32, y: f32, z: f32, sx: f32, sy: f32, sz: f32, m: usize| {
                (
                    block.clone(),
                    Transform::from_xyz(x, y, z).with_scale(Vec3::new(sx, sy, sz)),
                    m,
                )
            };
            let wheel = |x: f32, y: f32, z: f32, r: f32, w: f32| {
                (
                    shapes.cylinder.clone(),
                    Transform::from_xyz(x, y, z)
                        .with_rotation(Quat::from_rotation_z(FRAC_PI_2))
                        .with_scale(Vec3::new(2.0 * r, w, 2.0 * r)),
                    glass,
                )
            };
            let u = rng.unit();
            if u < 0.45 {
                // car: body, roof, windscreen, rear window, side glass under the roof edge
                vec![
                    b(0.0, 0.45, 0.0, 1.8, 0.7, 4.4, body),
                    b(0.0, 1.0, -0.2, 1.56, 0.35, 1.6, glass),
                    b(0.0, 1.2, -0.2, 1.5, 0.08, 1.5, body),
                    b(0.0, 0.92, 0.9, 1.5, 0.5, 0.6, glass),
                    b(0.0, 0.92, -1.25, 1.5, 0.45, 0.45, glass),
                ]
            } else if u < 0.7 {
                // pickup: cab, and an open bed with a dark liner
                vec![
                    b(0.0, 0.5, 0.0, 1.9, 0.8, 5.2, body),
                    b(0.0, 1.15, 0.6, 1.76, 0.5, 1.5, glass),
                    b(0.0, 1.42, 0.6, 1.7, 0.08, 1.35, body),
                    b(0.0, 0.93, -1.4, 1.6, 0.05, 2.1, glass),
                    b(-0.9, 1.05, -1.4, 0.1, 0.3, 2.2, body),
                    b(0.9, 1.05, -1.4, 0.1, 0.3, 2.2, body),
                    b(0.0, 1.05, -2.5, 1.8, 0.3, 0.1, body),
                ]
            } else if u < 0.9 {
                // van: a tall box with the windscreen at its front edge
                vec![
                    b(0.0, 1.05, 0.0, 2.0, 2.0, 5.0, body),
                    b(0.0, 1.75, 2.35, 1.9, 0.5, 0.35, glass),
                ]
            } else {
                // tractor: bonnet, glass cab under a roof, big rear wheels, small front ones
                vec![
                    b(0.0, 1.2, 0.9, 0.9, 0.9, 2.2, body),
                    b(0.0, 1.9, -0.7, 1.5, 1.4, 1.5, glass),
                    b(0.0, 2.65, -0.7, 1.6, 0.08, 1.7, body),
                    wheel(-1.05, 0.8, -0.8, 0.8, 0.55),
                    wheel(1.05, 0.8, -0.8, 0.8, 0.55),
                    wheel(-0.85, 0.5, 1.5, 0.5, 0.35),
                    wheel(0.85, 0.5, 1.5, 0.5, 0.35),
                ]
            }
        }
        ClutterKind::Bale => {
            // round (1.5 m across, 1.2 m wide) lying on its side or on end, or square
            // (2.4 x 1.2 x 0.9 m)
            let u = rng.unit();
            let (mesh, tf) = if u < 0.6 {
                let tf = Transform::from_xyz(0.0, 0.72, 0.0)
                    .with_rotation(Quat::from_rotation_z(FRAC_PI_2))
                    .with_scale(Vec3::new(1.5, 1.2, 1.5));
                (shapes.cylinder, tf)
            } else if u < 0.8 {
                let tf = Transform::from_xyz(0.0, 0.6, 0.0).with_scale(Vec3::new(1.5, 1.2, 1.5));
                (shapes.cylinder, tf)
            } else {
                let tf = Transform::from_xyz(0.0, 0.45, 0.0).with_scale(Vec3::new(2.4, 0.9, 1.2));
                (shapes.block, tf)
            };
            vec![(mesh.clone(), tf, rng.index(CLUTTER_MATS))]
        }
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

/// A name ("Cow:Walk@0.25") and the meshes of one way a body can look.
type BodyVariant = (String, Vec<(Handle<Mesh>, Transform, usize)>);

/// Every model and pose of a kind as meshes (already facing +z, feet at 0).
fn model_variants(meshes: &mut Assets<Mesh>, models: &[Model]) -> Vec<BodyVariant> {
    let mut out = Vec::new();
    for model in models {
        for pose in &model.poses {
            let parts = pose
                .parts
                .iter()
                .enumerate()
                .filter(|(_, p)| !p.indices.is_empty())
                .map(|(k, p)| {
                    let mesh = Mesh::new(
                        PrimitiveTopology::TriangleList,
                        RenderAssetUsages::default(),
                    )
                    .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, p.positions.clone())
                    .with_inserted_indices(Indices::U32(p.indices.clone()))
                    .with_computed_smooth_normals();
                    (meshes.add(mesh), Transform::default(), k)
                })
                .collect();
            out.push((format!("{}:{}", model.name, pose.name), parts));
        }
    }
    out
}

/// Which of `n` variants a placed body shows: a hash of where it stands, so picking one
/// takes no random draws (shots stay identical whatever the bodies are built from).
pub fn variant_index(p: &Placement, n: usize) -> usize {
    let t = p.transform.translation;
    let mut h = (t.x.to_bits() as u64) << 32 | t.z.to_bits() as u64;
    h ^= h >> 33;
    h = h.wrapping_mul(0xff51_afd7_ed55_8ccd);
    h ^= h >> 33;
    (h % n.max(1) as u64) as usize
}

/// Where an entity is and whether it shows, plus one marker component.
type Placed<C> = (&'static mut Transform, &'static mut Visibility, &'static C);
/// Body variants, kept apart from the other `Visibility` queries.
type VariantOnly = (
    With<Variant>,
    Without<AerialCamera>,
    Without<Target>,
    Without<Clutter>,
    Without<WaterSurface>,
);
/// The water surface, kept apart from every other `Transform` query.
type WaterOnly = (
    With<WaterSurface>,
    Without<AerialCamera>,
    Without<Target>,
    Without<Clutter>,
);
/// The sun, kept apart from every other `Transform` query.
type SunOnly = (
    With<Sun>,
    Without<AerialCamera>,
    Without<Target>,
    Without<Clutter>,
    Without<WaterSurface>,
);
type CameraParts = (
    &'static mut Transform,
    &'static mut Projection,
    &'static mut Msaa,
);

/// Everything a shot changes, for systems in both binaries.
#[derive(SystemParam)]
pub struct SceneCtl<'w, 's> {
    meshes: ResMut<'w, Assets<Mesh>>,
    images: ResMut<'w, Assets<Image>>,
    mats: ResMut<'w, Assets<StandardMaterial>>,
    handles: Res<'w, SceneHandles>,
    modality: Res<'w, Modality>,
    world: Res<'w, World>,
    sky: ResMut<'w, GlobalAmbientLight>,
    camera: Query<'w, 's, CameraParts, With<AerialCamera>>,
    sun: Query<'w, 's, (&'static mut Transform, &'static mut DirectionalLight), SunOnly>,
    targets: Query<'w, 's, Placed<Target>, Without<AerialCamera>>,
    clutter: Query<'w, 's, Placed<Clutter>, (Without<AerialCamera>, Without<Target>)>,
    water: Query<'w, 's, (&'static mut Transform, &'static mut Visibility), WaterOnly>,
    variants: Query<'w, 's, &'static mut Visibility, VariantOnly>,
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

    /// Per pool body, in id order: is its posture in its mesh (see `Target::posed`)?
    pub fn posed(&self) -> Vec<bool> {
        let mut t: Vec<(u32, bool)> = self
            .targets
            .iter()
            .map(|(_, _, t)| (t.id, t.posed))
            .collect();
        t.sort_by_key(|x| x.0);
        t.into_iter().map(|x| x.1).collect()
    }

    /// What target `id` looks like in `shot` (see `Target::bodies`); `None` if it isn't in it.
    pub fn body_of(&self, id: u32, shot: &Shot) -> Option<&str> {
        let p = shot.targets.get(id as usize - 1)?.as_ref()?;
        let (_, _, t) = self.targets.iter().find(|(_, _, t)| t.id == id)?;
        Some(t.bodies[variant_index(p, t.bodies.len())].as_str())
    }

    /// Class id of target `id`.
    pub fn class_of(&self, id: u32) -> Option<u32> {
        self.targets
            .iter()
            .find(|(_, _, t)| t.id == id)
            .map(|(_, _, t)| t.kind.class_id())
    }

    /// Move everything to `shot` and set every temperature (or colour, for RGB).
    pub fn apply(&mut self, shot: &Shot) {
        let (mut cam_tf, mut proj, _) = self.camera.single_mut().expect("one aerial camera");
        *cam_tf = shot.camera;
        if let Projection::Perspective(p) = &mut *proj {
            p.fov = (shot.vfov_deg as f32).to_radians();
        }
        let mut places: Vec<_> = shot.clutter.iter().map(|v| v.iter()).collect();
        for (mut tf, mut vis, c) in &mut self.clutter {
            let k = self.modality.clutter_index(c.kind);
            match places[k].next().and_then(|p| p.as_ref()) {
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
                    let shown = variant_index(p, target.variants.len());
                    for (v, &e) in target.variants.iter().enumerate() {
                        if let Ok(mut vv) = self.variants.get_mut(e) {
                            *vv = if v == shown {
                                Visibility::Inherited
                            } else {
                                Visibility::Hidden
                            };
                        }
                    }
                }
                None => *vis = Visibility::Hidden,
            }
        }
        match &shot.rgb {
            None => self.apply_temperatures(shot),
            Some(look) => self.apply_colours(look),
        }
        if let (Some(look), Some((mesh, image))) = (&shot.forest, &self.handles.overlay) {
            let w = shot.window;
            if let Some(mut img) = self.images.get_mut(image) {
                img.data = Some(look.rgba(|t| w.grey(t)));
            }
            if let Some(mut m) = self.meshes.get_mut(mesh) {
                *m = terrain_patch(&self.world, look.centre, look.size, PATCH_LIFT_M);
            }
        }
        if let (Some(look), Some((mesh, picture, mask))) = (&shot.forest, &self.handles.canopy) {
            let w = shot.window;
            if let Some(mut img) = self.images.get_mut(picture) {
                img.data = Some(look.canopy_rgba(|t| w.grey(t)));
            }
            if let Some(mut img) = self.images.get_mut(mask) {
                img.data = Some(look.canopy_mask());
            }
            let lift = look.canopy.as_ref().map_or(1.0, |c| c.height_m);
            if let Some(mut m) = self.meshes.get_mut(mesh) {
                *m = terrain_patch(&self.world, look.centre, look.size, lift);
            }
        }
    }

    /// Thermal: every material's grey is its temperature in the shot's window.
    fn apply_temperatures(&mut self, shot: &Shot) {
        let w = shot.window;
        let grey = |t: f64| {
            let v = w.linear(t);
            Color::linear_rgb(v, v, v)
        };
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
        for (_, _, target) in &self.targets {
            let Some(p) = &shot.targets[target.id as usize - 1] else {
                continue;
            };
            // forest bodies carry the ground texture (mottle), so they make up for its base
            let gain = if shot.forest.is_some() {
                texture_gain()
            } else {
                1.0
            };
            for (h, &t) in target.picture.iter().zip(&p.temp_c) {
                if let Some(mut m) = self.mats.get_mut(h) {
                    let v = w.linear(t) * gain;
                    m.base_color = Color::linear_rgb(v, v, v);
                }
            }
        }
    }

    /// RGB: albedos (`rgb.rs`) under the shot's sun and sky.
    fn apply_colours(&mut self, look: &RgbLook) {
        let lin = |c: rgb::Rgb| Color::linear_rgb(c[0], c[1], c[2]);
        let d = &look.daylight;
        if let Ok((mut tf, mut sun)) = self.sun.single_mut() {
            let dir = Vec3::new(
                d.sun_dir[0] as f32,
                d.sun_dir[1] as f32,
                d.sun_dir[2] as f32,
            );
            *tf = Transform::from_translation(dir).looking_at(Vec3::ZERO, Vec3::Y);
            sun.illuminance = SUN_LUX * d.sun;
            sun.color = lin(d.sun_tint);
        }
        self.sky.color = lin(d.sky_tint);
        self.sky.brightness = SKY_NITS * d.sky;

        let colors: Vec<[f32; 4]> = self
            .world
            .vertices
            .iter()
            .map(|p| {
                let c = look.ground.at(p[0], p[2]);
                [c[0], c[1], c[2], 1.0]
            })
            .collect();
        if let Some(mut mesh) = self.meshes.get_mut(&self.handles.terrain_mesh) {
            mesh.insert_attribute(Mesh::ATTRIBUTE_COLOR, colors);
        }
        if let Some(mut img) = self.images.get_mut(&self.handles.texture) {
            img.data = Some(rgb::ground_texture(&look.ground, TEX_SIZE));
        }
        if let (Some((mesh, image)), Some(table)) = (&self.handles.overlay, &self.handles.table) {
            let l = &look.ground.layout;
            if let Some(mut img) = self.images.get_mut(image) {
                img.data = Some(crate::fields::paint(l, table, OVERLAY_TEX));
            }
            if let Some(mut m) = self.meshes.get_mut(mesh) {
                *m = terrain_patch(&self.world, l.centre, l.size, PATCH_LIFT_M);
            }
        }
        let (_, mut water_vis) = self.water.single_mut().expect("one water surface");
        *water_vis = Visibility::Hidden;
        if let Some(mut m) = self.mats.get_mut(&self.handles.roads) {
            m.base_color = lin(look.road);
        }
        for ((set, colours), &(kind, _)) in self
            .handles
            .clutter
            .iter()
            .zip(&look.clutter)
            .zip(RGB_CLUTTER.iter())
        {
            // canopies carry the foliage texture, so they make up for its base
            let gain = match kind {
                ClutterKind::Bush | ClutterKind::Hedge => texture_gain(),
                ClutterKind::Shed | ClutterKind::OpenShed => SHEET_GAIN,
                _ => 1.0,
            };
            for (h, &c) in set.iter().zip(colours) {
                if let Some(mut m) = self.mats.get_mut(h) {
                    m.base_color = lin(c.map(|v| v * gain));
                }
            }
        }
        for (_, _, target) in &self.targets {
            let Some(coat) = &look.coats[target.id as usize - 1] else {
                continue;
            };
            // baked models have no texture coordinates, so patches are for primitives
            let patched = coat.patches && target.bodies[0] == "primitive";
            let patch = &self.handles.patches[target.id as usize % PATCH_TEXTURES];
            for (k, (h, &c)) in target.picture.iter().zip(&coat.parts).enumerate() {
                if let Some(mut m) = self.mats.get_mut(h) {
                    m.base_color = lin(c);
                    m.base_color_texture = (patched && k == TORSO).then(|| patch.clone());
                }
            }
        }
    }

    /// Switch every mesh to its id look (`true`) or its picture look. RGB pictures are
    /// drawn with MSAA, the id look never (its colours must stay exact).
    pub fn show_ids(&mut self, ids: bool) {
        for (look, mut mat) in &mut self.looks {
            mat.0 = if ids {
                look.id.clone()
            } else {
                look.picture.clone()
            };
        }
        if *self.modality == Modality::Rgb {
            let (_, _, mut msaa) = self.camera.single_mut().expect("one aerial camera");
            *msaa = if ids { Msaa::Off } else { Msaa::Sample4 };
        }
    }
}

fn rgba(grey: &[u8]) -> Vec<u8> {
    grey.iter().flat_map(|&g| [g, g, g, 255]).collect()
}

/// Weathered sheeting for shed roofs and walls (grey, sRGB, TEX_SIZE square, tiling):
/// sheets with a darker seam between them, each a little lighter or darker, faint streaks
/// down them, and broad dirt and lichen. Mean about `1 / SHEET_GAIN` (linear).
fn sheet_texture(seed: u64) -> Vec<u8> {
    let n = TEX_SIZE;
    let sheet = 32;
    let mut out = Vec::with_capacity(n * n);
    for j in 0..n {
        for i in 0..n {
            let (u, v) = (i as f64 / n as f64, j as f64 / n as f64);
            let seam = if i % sheet == 0 { 0.8 } else { 1.0 };
            let tone = 0.95 + 0.1 * (crate::rng::lattice(seed, (i / sheet) as i64, 0) - 0.5);
            let streak = 1.0 + 0.06 * (crate::rng::lattice(seed ^ 1, i as i64, 0) - 0.5);
            let dirt = (crate::thermal::tiling_fbm(seed ^ 2, u, v, 3.0) + 0.2).clamp(0.0, 1.0);
            let lin = 0.95 * seam * tone * streak * (1.0 - 0.25 * dirt);
            out.push((rgb::linear_to_srgb(lin.clamp(0.0, 1.0) as f32) * 255.0).round() as u8);
        }
    }
    out
}

/// Undoes `sheet_texture`'s darkening on the base colour.
const SHEET_GAIN: f32 = 1.15;

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
            let s = Shot::random(&mut rng, &world, &kinds, &[], 0.0, Modality::Thermal);
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
    fn posed_people_change_only_their_posture() {
        let world = World::new(1);
        let kinds = pool();
        let posed: Vec<bool> = kinds.iter().map(|&k| k == Kind::Person).collect();
        let (mut a, mut b) = (Rng::new(3), Rng::new(3));
        for _ in 0..20 {
            let s = Shot::random(&mut a, &world, &kinds, &[], 0.0, Modality::Thermal);
            let t = Shot::random(&mut b, &world, &kinds, &posed, 0.0, Modality::Thermal);
            assert_eq!(s.camera, t.camera);
            for ((p, q), &kind) in s.targets.iter().zip(&t.targets).zip(&kinds) {
                let (Some(p), Some(q)) = (p, q) else {
                    assert_eq!(p.is_some(), q.is_some());
                    continue;
                };
                assert_eq!(p.temp_c, q.temp_c);
                assert_eq!(p.transform.translation.xz(), q.transform.translation.xz());
                if kind == Kind::Person {
                    // upright: no squash, no tipping over
                    assert!((q.transform.scale.y - q.transform.scale.x).abs() < 1e-6);
                    assert!((q.transform.rotation * Vec3::Y).y > 0.999);
                } else {
                    assert_eq!(p.transform, q.transform);
                }
            }
        }
    }

    #[test]
    fn variants_are_picked_in_range_and_spread() {
        let mut seen = [0usize; 5];
        for i in 0..200 {
            let p = Placement {
                transform: Transform::from_xyz(i as f32 * 3.7, 0.0, i as f32 * -1.3),
                temp_c: [0.0; 3],
            };
            seen[variant_index(&p, 5)] += 1;
        }
        assert!(seen.iter().all(|&n| n > 15), "{seen:?}");
        let p = Placement {
            transform: Transform::default(),
            temp_c: [0.0; 3],
        };
        assert_eq!(variant_index(&p, 1), 0);
    }

    #[test]
    fn faint_shots_keep_bodies_just_above_the_ground() {
        let world = World::new(1);
        let kinds = pool();
        let mut rng = Rng::new(5);
        let (mut faint, mut with_targets) = (0, 0);
        for _ in 0..80 {
            let s = Shot::random(&mut rng, &world, &kinds, &[], 0.4, Modality::Thermal);
            if s.n_targets() == 0 {
                assert!(!s.faint);
                continue;
            }
            with_targets += 1;
            if !s.faint {
                continue;
            }
            faint += 1;
            let c = &s.conditions;
            for p in s.targets.iter().flatten() {
                let t = p.transform.translation;
                let (x, z) = (t.x as f64, t.z as f64);
                let ground = c.ground_c(world.seed, x, z, world.terrain.normal(x, z));
                let grey = (s.window.grey(p.temp_c[0]) - s.window.grey(ground)).abs() * 255.0;
                assert!((11.0..=61.0).contains(&grey), "{grey} grey levels");
            }
        }
        let share = faint as f64 / with_targets as f64;
        assert!(
            (0.2..0.6).contains(&share),
            "{faint} faint of {with_targets}"
        );
    }

    #[test]
    fn pool_fits_the_id_render() {
        assert_eq!(pool(), Modality::Thermal.kinds());
        for m in [Modality::Thermal, Modality::Rgb] {
            assert!(m.kinds().len() * (ID_STEP as usize) <= 255);
        }
    }

    #[test]
    fn rgb_shots_look_down_in_colour_with_bales_on_stubble() {
        let world = World::new(1);
        let kinds = Modality::Rgb.kinds();
        let mut rng = Rng::new(6);
        let bale = Modality::Rgb.clutter_index(ClutterKind::Bale);
        let (mut nadir, mut bales_off_stubble, mut empty) = (0, 0, 0);
        for _ in 0..60 {
            let s = Shot::random(&mut rng, &world, &kinds, &[], 0.4, Modality::Rgb);
            let look = s.rgb.as_ref().expect("RGB shots carry colours");
            assert!((60.0..150.0).contains(&s.altitude_m));
            assert!(s.conditions.water.is_none() && !s.faint);
            nadir += (s.pitch_deg == 90.0) as usize;
            empty += (s.n_targets() == 0) as usize;
            for (p, c) in s.targets.iter().zip(&look.coats) {
                assert_eq!(p.is_some(), c.is_some());
            }
            for b in s.clutter[bale].iter().flatten() {
                let (x, z) = (b.translation.x as f64, b.translation.z as f64);
                let field = look.ground.layout.parcel_at(x, z).field;
                bales_off_stubble += (field != rgb::Field::Stubble) as usize;
            }
            assert_eq!(look.clutter.len(), s.clutter.len());
        }
        assert!((36..60).contains(&nadir), "{nadir} nadir of 60");
        assert!((6..=36).contains(&empty), "{empty} empty of 60");
        assert_eq!(bales_off_stubble, 0);
    }

    #[test]
    fn packed_cows_face_where_their_slot_says() {
        let mut rng = Rng::new(3);
        for _ in 0..20 {
            let a = rng.range(0.0, std::f64::consts::TAU);
            let y = Yard::random(&mut rng, [5.0, 7.0], [a.cos(), a.sin()]);
            let f = rng.range(0.0, std::f64::consts::TAU);
            let front = facing(&y, f) * Vec3::Z;
            let (x0, z0) = y.world([0.0, 0.0]);
            let (x1, z1) = y.world([f.cos(), f.sin()]);
            assert!((front.x as f64 - (x1 - x0)).abs() < 1e-5);
            assert!((front.z as f64 - (z1 - z0)).abs() < 1e-5);
        }
    }

    #[test]
    fn yard_shots_put_herds_in_pens_and_keep_clutter_out() {
        let world = World::new(3);
        let kinds = Modality::Rgb.kinds();
        let mut rng = Rng::new(11);
        let m = Modality::Rgb;
        let (mut yards, mut herds, mut in_yard, mut bodies) = (0, 0, 0, 0);
        for _ in 0..80 {
            let s = Shot::random(&mut rng, &world, &kinds, &[], 0.0, m);
            let Some(y) = s.rgb.as_ref().unwrap().ground.layout.yard.as_ref() else {
                continue;
            };
            yards += 1;
            let sheds = s.clutter[m.clutter_index(ClutterKind::Shed)]
                .iter()
                .flatten()
                .count()
                + s.clutter[m.clutter_index(ClutterKind::OpenShed)]
                    .iter()
                    .flatten()
                    .count();
            assert_eq!(sheds, y.sheds.len());
            for kind in [ClutterKind::Bush, ClutterKind::House, ClutterKind::Hedge] {
                for t in s.clutter[m.clutter_index(kind)].iter().flatten() {
                    assert!(!y.covers(t.translation.x as f64, t.translation.z as f64));
                }
            }
            let n = s.n_targets();
            if n > 0 {
                herds += 1;
                bodies += n;
                in_yard += s
                    .targets
                    .iter()
                    .flatten()
                    .filter(|p| {
                        y.holds(
                            p.transform.translation.x as f64,
                            p.transform.translation.z as f64,
                        )
                    })
                    .count();
            }
        }
        assert!((15..45).contains(&yards), "{yards} yards in 80 shots");
        assert!(herds > 5);
        // most bodies of a yard shot stand in it
        assert!(in_yard * 2 > bodies, "{in_yard} of {bodies} in the yard");
    }

    #[test]
    fn rgb_footprint_holds_the_whole_frame() {
        let world = World::new(2);
        let kinds = Modality::Rgb.kinds();
        let mut rng = Rng::new(8);
        for _ in 0..40 {
            let s = Shot::random(&mut rng, &world, &kinds, &[], 0.0, Modality::Rgb);
            let l = &s.rgb.as_ref().unwrap().ground.layout;
            // the ground under the frame centre is inside the painted square, and the
            // square is at least the nadir frame's width
            let fwd = s.camera.rotation * Vec3::NEG_Z;
            let eye = s.camera.translation;
            let t = -(s.altitude_m as f32) / fwd.y;
            let c = eye + fwd * t;
            assert!((c.x as f64 - l.centre[0]).abs() < l.size / 2.0);
            assert!((c.z as f64 - l.centre[1]).abs() < l.size / 2.0);
            let width = 2.0 * s.altitude_m * (s.vfov_deg / 2.0).to_radians().tan() * 4.0 / 3.0;
            assert!(l.size >= width, "{} < {width}", l.size);
        }
    }
}
