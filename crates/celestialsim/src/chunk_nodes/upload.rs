//! `celestial/chunk-upload` — CPU → GPU chunk descriptor + params + instance
//! buffer updates (transfers, outside any compute list). Mirrors `nodes/upload.rs`.

use crate::chunk_nodes::ChunkNode;
use crate::chunk_pipeline::ChunkCtx;

pub struct ChunkUpload;

impl ChunkNode for ChunkUpload {
    fn name(&self) -> &'static str {
        "celestial/chunk-upload"
    }

    fn record(&mut self, ctx: &mut ChunkCtx<'_>) {
        // CPU → GPU buffer updates must run outside any open compute list.
        ctx.close_list();
        let mut rd = ctx.rd.clone();
        let terrain = ctx.terrain;
        if let Some(stage) = ctx.stage.take() {
            // Set the surface flags BEFORE upload_params so the repacked ChunkParams
            // carries the global toggle + height scale. A custom GPU surface owns
            // the toggle itself: on only once its shader compiled (else the buffers
            // it fills are never written → fall back to procedural), with the
            // layer's height scale.
            if ctx.gpu.custom_requested() {
                let hs = ctx.gpu.custom_height_scale();
                let on = if ctx.gpu.custom_ready() { 1.0 } else { 0.0 };
                ctx.gpu.set_surface(on, hs);
            } else {
                ctx.gpu.set_surface(stage.surface_enabled, stage.surface_height_scale);
            }
            ctx.gpu.upload_descs(&mut rd, &stage.desc_bytes, stage.realize_count);
            ctx.gpu.upload_params(&mut rd, stage.realize_count, &terrain);
            ctx.gpu.upload_instances(&mut rd, &stage.instance_bytes, stage.instance_count);
            // Custom GPU surface: refresh its params UBO (chunk_count + live knobs)
            // for the surface-custom dispatch. Transfer — must be outside a list.
            ctx.gpu.upload_custom_params(&mut rd, stage.realize_count);
            // Upload each realized chunk's CPU-surface color+height patch into its atlas slot.
            for (slot, color, height, normal) in &stage.surface_patches {
                ctx.gpu.upload_surface(&mut rd, *slot, color, height, normal);
            }
            // CEL-73 scatter staging: aux paths, visible list, per-layer params
            // (fresh density/height each stage), and counter/draw-count zeroing.
            ctx.gpu.upload_scatter(
                &mut rd,
                &stage.scatter_aux_bytes,
                &stage.scatter_vis_bytes,
                &stage.scatter_layer_params,
            );
            ctx.realize_count = stage.realize_count;
            ctx.vis_count = stage.scatter_vis_count;
        }
    }
}
