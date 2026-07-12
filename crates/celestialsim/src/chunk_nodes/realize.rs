//! `celestial/chunk-realize` — compute dispatch that realizes chunk vertices
//! from the uploaded descriptors into the shared vertex pool. One thread per
//! pool vertex (`realize_count * verts_per_chunk`). Mirrors `nodes/realize.rs`.

use crate::chunk_nodes::ChunkNode;
use crate::chunk_pipeline::ChunkCtx;

pub struct ChunkRealize;

impl ChunkNode for ChunkRealize {
    fn name(&self) -> &'static str {
        "celestial/chunk-realize"
    }

    fn record(&mut self, ctx: &mut ChunkCtx<'_>) {
        let count = ctx.realize_count;
        if count == 0 {
            return;
        }
        let vert_threads = count * ctx.gpu.verts_per_chunk();
        let list = ctx.list();
        let mut rd = ctx.rd.clone();
        ctx.gpu.record_realize(&mut rd, list, vert_threads);
    }
}
