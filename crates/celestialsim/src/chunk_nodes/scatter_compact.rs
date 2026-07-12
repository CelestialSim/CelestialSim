//! `celestial/chunk-scatter-compact` — CEL-73 scatter draw-list build. One
//! thread per cached pool candidate of each VISIBLE chunk, per layer: gates by
//! the LIVE `density`/height sliders and appends surviving transforms into
//! the layer's indirect MultiMesh (count into the command buffer — no
//! readback). This is the ONLY GPU work a slider edit re-runs.

use crate::chunk_nodes::ChunkNode;
use crate::chunk_pipeline::ChunkCtx;

pub struct ChunkScatterCompact;

impl ChunkNode for ChunkScatterCompact {
    fn name(&self) -> &'static str {
        "celestial/chunk-scatter-compact"
    }

    fn record(&mut self, ctx: &mut ChunkCtx<'_>) {
        let vis = ctx.vis_count;
        if vis == 0 || ctx.gpu.scatter_layer_count() == 0 {
            return;
        }
        let list = ctx.list();
        let mut rd = ctx.rd.clone();
        for li in 0..ctx.gpu.scatter_layer_count() {
            ctx.gpu.record_scatter_compact(&mut rd, list, li, vis);
        }
    }
}
