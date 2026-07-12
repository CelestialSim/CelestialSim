//! Pure CPU math for CelestialSim: the base icosphere and the triangular
//! clipmap patch descriptors (constant cost w.r.t. triangle count).
//!
//! This crate holds the LOD invariant test `visible_cells_meet_screen_error`;
//! any change to the descriptor math must keep it green.

pub mod chunk_cache;
pub mod clipmap;
pub mod cull;
pub mod icosphere;
pub mod quadtree;
pub mod scatter;
pub mod surface_cache;
