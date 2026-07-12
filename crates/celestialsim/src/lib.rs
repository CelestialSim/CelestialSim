//! Celestial v5 — GPU-realized chunked-quadtree terrain.
//!
//! The CPU selects a screen-space-error cut over a fixed triangular quadtree
//! (constant cost), and Slang compute shaders realize + bake each chunk's
//! geometry and surface-detail atlas straight into an indirect MultiMesh on the
//! main rendering device — no CPU readback. Geometry and texture are cached
//! per chunk (`ChunkCache`) so only newly-visible chunks are re-realized.

use godot::prelude::*;

pub mod async_bake;
/// Task 10 chunk-realize GPU verification node — debug-only; runs `ChunkRealize`
/// on a local RenderingDevice and checks positions vs `chunk_subvertex_base`.
#[cfg(debug_assertions)]
pub mod chunk_gpu_test;
pub mod chunk_descriptors;
pub mod chunk_mesh;
pub mod chunk_nodes;
pub mod chunk_pipeline;
pub mod custom_surface;
pub mod bake_pool;
pub mod descriptors;
pub mod gpu;
pub mod builder;
pub mod noise_provider;
#[cfg(debug_assertions)]
pub mod quadtree_debug;
pub mod celestial;
pub mod scatter_descriptors;
pub mod scatter_layer;
pub mod scatter_mesh;
pub mod surface;
pub mod water;
pub mod water_runtime;
/// Debug-only standalone single-tile texture viewer (`CelestialTileViewer`):
/// re-bakes one icosphere face's surface detail into a `W×W` texture as the user
/// zooms/pans, isolated from the clipmap. Excluded from release builds.
#[cfg(debug_assertions)]
pub mod tile_viewer;

struct CelestialExtension;

#[gdextension]
unsafe impl ExtensionLibrary for CelestialExtension {}
