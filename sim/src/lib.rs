//! Storm sim for aerial-animal-rescue. See `sim/PLAN.md`.
//!
//! Only `scene` depends on Bevy. `world` (terrain, on CortenForge), `thermal` (temperatures
//! per shot), `sensor` (the thermal camera model), `bodies` (baked CC0 animal models) and
//! `capture` (rendered frames -> dataset pool) work with any renderer; `capture` is meant
//! to move into a CortenForge crate later (see the Suggestions in `sim/FINDINGS.md`).

pub mod bodies;
pub mod capture;
pub mod rng;
pub mod scene;
pub mod sensor;
pub mod thermal;
pub mod world;
