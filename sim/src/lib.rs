//! Storm sim for aerial-animal-rescue. See `sim/PLAN.md`.
//!
//! `world` and `capture` don't depend on Bevy: terrain and placement use CortenForge, and
//! capture turns any renderer's output into a dataset pool. `capture` is meant to move into
//! a CortenForge crate later (see the Suggestions in `sim/FINDINGS.md`). `scene` is the
//! Bevy side.

pub mod capture;
pub mod rng;
pub mod scene;
pub mod world;
