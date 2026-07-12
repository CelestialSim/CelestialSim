//! `celestial/chunk-bake` — Phase-4 compute dispatch that bakes per-chunk
//! colour + normal detail tiles into the atlas textures. One thread per tile
//! texel (`realize_count * tile_res²`). Runs after `chunk-realize`; reads the
//! same uploaded descriptors and writes the colour/normal atlases the surface
//! fragment shader samples.

use crate::chunk_nodes::ChunkNode;
use crate::chunk_pipeline::ChunkCtx;

pub struct ChunkBake;

impl ChunkNode for ChunkBake {
    fn name(&self) -> &'static str {
        "celestial/chunk-bake"
    }

    fn record(&mut self, ctx: &mut ChunkCtx<'_>) {
        let count = ctx.realize_count;
        if count == 0 {
            return;
        }
        let tr = ctx.gpu.tile_res();
        let texel_threads = count * tr * tr;
        let list = ctx.list();
        let mut rd = ctx.rd.clone();
        ctx.gpu.record_bake(&mut rd, list, texel_threads);
    }
}
