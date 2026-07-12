//! Neutral CPU-baked surface type + the provider trait that supplies it.
//!
//! The chunked planet bakes each chunk's surface into three per-slot GPU
//! buffers — colour (rgba8), height (`f32`, in the provider's own vertical unit)
//! and normal (rgba8-packed) — and the realize/bake shaders read them when the
//! surface is enabled. TWO paths fill those buffers:
//!
//! * **GPU procedural noise** — a compute shader computes them in place (the
//!   default; no provider, `surface_enabled == 0`).
//! * **CPU async** — a worker pool ([`crate::bake_pool::BakePool`]) calls a
//!   [`CpuSurfaceProvider`] off the main thread to produce a [`ChunkSurface`],
//!   which the planet uploads into buffers 4/5/6 and the shaders read when the
//!   surface is enabled. The built-in provider is [`crate::noise_provider`]
//!   (CPU fBm terrain), used by a [`crate::builder::CesBuilder`] in
//!   [`crate::builder::BuilderMode::CpuNoise`] mode.
//!
//! This module holds ONLY the neutral data type and the trait — no concrete
//! surface source, no Godot rendering.

use std::collections::HashSet;

use godot::builtin::Vector3;

use celestial_algo::clipmap::FaceFrame;
use celestial_algo::quadtree::{Chunk, ChunkId};

/// A resampled per-chunk surface: `color` is `tile_res·tile_res·4` RGBA8,
/// `height` is `tile_res·tile_res` elevations (the provider's vertical unit,
/// scaled by [`CpuSurfaceProvider::height_scale`] on the GPU), and `normal` is
/// `tile_res·tile_res·4` rgba8-packed world normals — all row-major. The chunk's
/// real footprint is the lower-left triangle, but the WHOLE square holds valid
/// data (texels past the diagonal replicate the nearest diagonal sample; see
/// [`crate::surface_tiles::build_patch`]).
#[derive(Clone, Debug)]
pub struct ChunkSurface {
    pub color: Vec<u8>,
    pub height: Vec<f32>,
    pub normal: Vec<u8>,
}

/// A source of CPU-baked chunk surfaces (colour/height/normal). Implementors run
/// on the bake worker pool ([`bake`](Self::bake) is called off the main thread,
/// so it must be `Send + Sync` and use interior mutability for any shared state).
/// The other hooks run on the main thread.
pub trait CpuSurfaceProvider: Send + Sync {
    /// Resample this chunk's surface at `tile_res × tile_res`. Called on a bake
    /// worker thread. A streaming provider may also enqueue any data it still
    /// needs (a side effect) so [`poll_refresh`](Self::poll_refresh) can later
    /// report the chunk for a re-bake once that data arrives.
    fn bake(&self, frame: &FaceFrame, chunk: &Chunk, tile_res: u32) -> ChunkSurface;

    /// Surface displacement at `dir`, in the height buffer's unit (multiply by
    /// [`height_scale`](Self::height_scale) · radius for world units). Steers LOD
    /// toward the displaced surface. `None` ⇒ no data (treated as sea level).
    fn sample_height(&self, _dir: Vector3) -> Option<f32> {
        None
    }

    /// Chunks whose awaited data has arrived and should be re-baked (streaming).
    /// Called each frame on the main thread; a non-streaming provider returns
    /// empty. Returns whole [`Chunk`]s (not just ids) so the caller can re-queue
    /// a bake directly.
    fn poll_refresh(&self) -> Vec<Chunk> {
        Vec::new()
    }

    /// Camera-driven cancellation hint: the set of chunk ids still in view.
    /// Streaming providers drop bookkeeping / downloads for anything else.
    fn set_wanted(&self, _ids: &HashSet<ChunkId>) {}

    /// Is enough data present for the surface to be shown (the GPU
    /// `surface_enabled` gate)? A streaming provider returns `false` until its base
    /// map lands; a self-contained provider (noise) is always ready.
    fn base_ready(&self) -> bool {
        true
    }

    /// Per-height-unit displaced-radius factor for the GPU: the surface radius
    /// becomes `radius · (1 + height · height_scale)`.
    fn height_scale(&self) -> f32 {
        0.0
    }

    /// External resource fetches in flight (streaming providers; for the HUD).
    fn resources_in_flight(&self) -> usize {
        0
    }

    /// Absolute path of any on-disk cache this provider uses (for the HUD).
    fn cache_dir(&self) -> Option<String> {
        None
    }
}

