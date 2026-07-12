//! **CelestialSim** — a Godot GDExtension that renders planetary bodies with
//! adaptive-LOD, chunked-quadtree terrain.
//!
//! Each frame the CPU selects a screen-space-error cut over a *fixed* triangular
//! quadtree — its cost depends on the number of chunks in view, not on the
//! triangle count — and Slang compute shaders realize each chunk's geometry and
//! bake its surface-detail atlas straight into an indirect `MultiMesh` on Godot's
//! main `RenderingDevice`. There is **no CPU readback**: geometry and colour never
//! leave the GPU. Chunks are cached in stable pool slots
//! (`celestial_algo::chunk_cache::ChunkCache`), so only newly-visible chunks are
//! realized and revisited terrain costs nothing to redraw.
//!
//! # Most readers want the guide, not the API
//!
//! The user documentation — install, first planet, custom terrain, scatter — is at
//! <https://celestialsim.github.io/CelestialSim/>. This page is the Rust reference
//! for the extension's internals.
//!
//! # The three types you touch from Godot
//!
//! * [`celestial::Celestial`] — the planet `Node3D`. Add one to a scene, set its
//!   radius / LOD knobs, and it streams terrain around the active camera.
//! * [`builder::CesBuilder`] — a `Resource` assigned to the planet's `builder`
//!   property: it *is* the terrain source (built-in GPU or CPU noise, your own
//!   `.glsl`, or GDScript), and it also carries the planet's ocean settings.
//! * [`scatter_layer::CesScatterLayer`] — a `Resource` per scattered mesh (grass,
//!   trees, rocks), placed entirely on the GPU; layers go in the planet's
//!   `scatter_layers` array.
//!
//! # Module map
//!
//! Pipeline (the render-thread GPU work):
//! * [`chunk_pipeline`] — wires the `upload → realize → bake` computation graph and
//!   runs it in one render-thread job.
//! * [`chunk_nodes`] — one self-contained `PipelineNode` per operation.
//! * [`gpu`] — thin helpers over Godot's main `RenderingDevice` (render thread only).
//!
//! CPU → GPU packing:
//! * [`chunk_descriptors`] — std430 packing of chunk descriptors and per-instance data.
//! * [`descriptors`] — the shared terrain-noise params (`HeightGpu` + `TextureGpu`).
//! * [`scatter_descriptors`] — the scatter passes' params/aux/visibility buffers.
//!
//! Surface / terrain:
//! * [`surface`] — the CPU-surface traits (a baked per-chunk colour/height/normal patch).
//! * [`custom_surface`] — assembles a user `.glsl` into the realize/bake shaders.
//! * [`noise_provider`] — the built-in noise evaluated on the CPU.
//! * [`chunk_mesh`] — the reference chunk mesh the indirect `MultiMesh` instances.
//!
//! Scatter:
//! * [`scatter_mesh`] — the procedural grass-blade mesh.
//!
//! Water:
//! * [`water`] / [`water_runtime`] — the analytic ocean proxy and its per-frame update.
//!
//! Async CPU bake:
//! * [`async_bake`] — the thread-safe hand-back queue behind `CesBuilder::submit_chunk`.
//! * [`bake_pool`] — the worker pool that bakes chunk surfaces off the main thread.
//!
//! Debug-only (`#[cfg(debug_assertions)]`, excluded from release builds):
//! [`chunk_gpu_test`], [`quadtree_debug`], [`tile_viewer`].
//!
//! The pure-CPU LOD math — quadtree selection and the chunk cache — lives in the
//! sibling `celestial-algo` crate, and the engine-agnostic GPU computation graph in
//! `celestial-graph`.
//!
//! # Shaders
//!
//! The compute shaders are Slang sources under `crates/celestialsim/shaders/`, with
//! their SPIR-V **committed** and baked into the cdylib — a normal build needs no
//! `slangc`. Regenerate with `SLANG_RECOMPILE=1 cargo build -p celestialsim` after
//! editing a `.slang`.

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
