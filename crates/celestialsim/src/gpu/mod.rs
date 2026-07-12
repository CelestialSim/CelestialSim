//! Thin helpers over Godot's main `RenderingDevice`.
//!
//! Every function here must run on the **render thread** (the buffer RIDs we
//! touch belong to the global device) — callers go through
//! `RenderingServer::call_on_render_thread`, CEL-58 style.

pub mod chunk_gpu;
pub mod device;
pub mod owned;

/// Width (texels) of the GPU attribute/atlas textures: chunk vertex pools and
/// per-chunk detail atlases are 1-D runs wrapped at this width. Shaders read it
/// as `attr_w` to recover 2-D texel coords from a linear index.
pub const ATTR_TEX_WIDTH: u32 = 4096;
