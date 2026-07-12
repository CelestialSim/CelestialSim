//! `celestial/chunk-scatter-place` — CEL-73 scatter placement dispatch. One
//! thread per candidate of each realize-batch chunk, per layer. Runs only when
//! chunks were (re)realized (`realize_count > 0`); the pool it writes is CACHED
//! per slot, so camera moves and density/height edits never re-run it.

use crate::chunk_nodes::ChunkNode;
use crate::chunk_pipeline::ChunkCtx;

pub struct ChunkScatterPlace;

impl ChunkNode for ChunkScatterPlace {
    fn name(&self) -> &'static str {
        "celestial/chunk-scatter-place"
    }

    fn record(&mut self, ctx: &mut ChunkCtx<'_>) {
        let count = ctx.realize_count;
        if count == 0 || ctx.gpu.scatter_layer_count() == 0 {
            return;
        }
        let list = ctx.list();
        let mut rd = ctx.rd.clone();
        for li in 0..ctx.gpu.scatter_layer_count() {
            ctx.gpu.record_scatter_place(&mut rd, list, li, count);
        }
    }
}
