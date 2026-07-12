//! `celestial/chunk-surface-custom` — fill the per-slot surface buffers from a
//! user's runtime-compiled GLSL (`crate::custom_surface`).
//!
//! Runs between upload and realize: one thread per tile texel of each realized
//! chunk writes `surface_color/height/normal`, which realize (displacement) and
//! bake (albedo/normal) then read via the existing `surface_enabled` path. A
//! no-op unless a custom GPU surface is installed AND its shader compiled — the
//! params transfer happens in the upload node (`upload_custom_params`), so this
//! node only records the dispatch inside the compute list.

use crate::chunk_nodes::ChunkNode;
use crate::chunk_pipeline::ChunkCtx;

pub struct ChunkSurfaceCustom;

impl ChunkNode for ChunkSurfaceCustom {
    fn name(&self) -> &'static str {
        "celestial/chunk-surface-custom"
    }

    fn record(&mut self, ctx: &mut ChunkCtx<'_>) {
        let count = ctx.realize_count;
        if count == 0 || !ctx.gpu.custom_ready() {
            return;
        }
        let list = ctx.list();
        let mut rd = ctx.rd.clone();
        ctx.gpu.record_custom_surface(&mut rd, list, count);
    }
}
