//! The chunk pipeline recording context + render-thread job.
//!
//! Mirrors `pipeline.rs` (`Ctx`/`PlanetJob`) for the chunk (grass/foliage)
//! render path. Two nodes — `chunk-upload` → `chunk-realize` — wired through the
//! descriptor/instance/verts resources, executed inside one render-thread job.

use std::collections::HashMap;

use celestial_graph::{Graph, NodeId, RecordCtx};
use godot::classes::{RefCounted, RenderingDevice, RenderingServer};
use godot::prelude::*;

use crate::chunk_nodes::{build_chunk_pipeline, ChunkNode};
use crate::descriptors::TerrainGpu;
use crate::gpu::chunk_gpu::ChunkGpuResources;

/// One chunk batch's CPU-staged data, handed from the main thread to the render
/// thread. `desc_bytes` is `pack_chunks` output; `instance_bytes` is
/// `pack_instances` output.
pub struct ChunkStage {
    pub desc_bytes: Vec<u8>,
    pub realize_count: u32,
    pub instance_bytes: Vec<u8>,
    pub instance_count: u32,
    /// Global CPU-surface toggle (1.0 once the provider's base is ready, else 0.0).
    pub surface_enabled: f32,
    /// Per-metre displaced-radius factor for CPU-surface heights.
    pub surface_height_scale: f32,
    /// Per realized chunk: (slot, rgba8 color [tile_res²·4], elevation metres
    /// [tile_res²], rgba8-packed world normal [tile_res²·4]).
    pub surface_patches: Vec<(u32, Vec<u8>, Vec<f32>, Vec<u8>)>,
    /// CEL-73 scatter: per-realize-chunk `{path_lo, path_hi, face, pad}` aux
    /// (`pack_scatter_aux`), parallel to `desc_bytes`. Empty when no layers.
    pub scatter_aux_bytes: Vec<u8>,
    /// CEL-73 scatter: per-visible-instance `{slot, depth}` list
    /// (`pack_scatter_vis`) the compact pass gathers from.
    pub scatter_vis_bytes: Vec<u8>,
    pub scatter_vis_count: u32,
    /// CEL-73 scatter: one packed `ScatterParamsGpu` snapshot per layer —
    /// re-read every stage so density/height edits flow through as pure
    /// uniform updates.
    pub scatter_layer_params: Vec<Vec<u8>>,
}

/// Bytes of one packed `ChunkGpu` descriptor in `desc_bytes`.
pub const CHUNK_DESC_STRIDE: usize = 96;
/// Bytes of one packed scatter-aux entry (`ScatterAuxGpu`), parallel to desc.
pub const SCATTER_AUX_STRIDE: usize = 16;

/// Merge a newly staged batch into a still-pending one instead of replacing it.
///
/// The main thread stages every frame while streaming, but the render thread
/// consumes asynchronously; a plain `stage = Some(new)` DROPPED any unconsumed
/// batch — those chunks were already marked clean in the cache and kept their
/// slots, so they were drawn forever with uninitialized pool memory (the giant
/// garbage-triangle walls). Realize work is cumulative (descriptors + surface
/// patches + scatter aux append); the instance set, surface globals and scatter
/// visible/params are snapshots (latest wins). If the merged batch exceeds
/// `budget_chunks` (render thread stalled for hundreds of frames), the OLDEST
/// realizes are dropped — they re-dirty naturally.
pub fn merge_stage(pending: &mut ChunkStage, new: ChunkStage, budget_chunks: usize) {
    pending.desc_bytes.extend_from_slice(&new.desc_bytes);
    pending.realize_count += new.realize_count;
    pending.surface_patches.extend(new.surface_patches);
    pending.scatter_aux_bytes.extend_from_slice(&new.scatter_aux_bytes);
    pending.instance_bytes = new.instance_bytes;
    pending.instance_count = new.instance_count;
    pending.surface_enabled = new.surface_enabled;
    pending.surface_height_scale = new.surface_height_scale;
    // Scatter visible list + per-layer params are per-frame snapshots.
    pending.scatter_vis_bytes = new.scatter_vis_bytes;
    pending.scatter_vis_count = new.scatter_vis_count;
    pending.scatter_layer_params = new.scatter_layer_params;

    let over = (pending.realize_count as usize).saturating_sub(budget_chunks);
    if over > 0 {
        pending.desc_bytes.drain(0..over * CHUNK_DESC_STRIDE);
        // surface_patches / scatter_aux parallel the realize list 1:1.
        let drop_patches = over.min(pending.surface_patches.len());
        pending.surface_patches.drain(0..drop_patches);
        let drop_aux = (over * SCATTER_AUX_STRIDE).min(pending.scatter_aux_bytes.len());
        pending.scatter_aux_bytes.drain(0..drop_aux);
        pending.realize_count = budget_chunks as u32;
    }

    // CEL-91: patches are not always 1:1 with realizes (a realize whose surface
    // was rejected/missing contributes no patch), so the `over` drain above does
    // NOT by itself bound them — a stalled render thread could accumulate
    // arbitrarily many, each ~0.75 MiB at tile_res 256. Cap them explicitly at
    // one pool's worth: the OLDEST go, exactly as with the realizes. A dropped
    // patch only means the slot keeps its previous surface until the chunk
    // re-dirties.
    if pending.surface_patches.len() > budget_chunks {
        let excess = pending.surface_patches.len() - budget_chunks;
        pending.surface_patches.drain(0..excess);
    }
}

/// Production recording context for the chunk pipeline: wraps the main
/// `RenderingDevice`, the chunk GPU resources, and the staged upload for this
/// execute. Mirrors `Ctx<'a>` from `pipeline.rs`.
pub struct ChunkCtx<'a> {
    pub rd: Gd<RenderingDevice>,
    pub gpu: &'a mut ChunkGpuResources,
    pub stage: &'a mut Option<ChunkStage>,
    pub terrain: TerrainGpu,
    pub res: u32,
    /// Chunks to realize this execute; set by the upload node from the stage and
    /// read by the realize node to size its dispatch (`count * verts_per_chunk`).
    pub realize_count: u32,
    /// Visible instances this execute (CEL-73); set by the upload node, read by
    /// the scatter-compact node to size its dispatch (`vis * capacity`).
    pub vis_count: u32,
    list: Option<i64>,
    /// Timestamp labels captured this execute (for GPU timing parity with `Ctx`).
    pub captured: Vec<&'static str>,
}

impl<'a> ChunkCtx<'a> {
    pub fn new(
        rd: Gd<RenderingDevice>,
        gpu: &'a mut ChunkGpuResources,
        stage: &'a mut Option<ChunkStage>,
        terrain: TerrainGpu,
        res: u32,
    ) -> Self {
        Self {
            rd,
            gpu,
            stage,
            terrain,
            res,
            realize_count: 0,
            vis_count: 0,
            list: None,
            captured: Vec::new(),
        }
    }

    pub fn list(&mut self) -> i64 {
        *self.list.get_or_insert_with(|| self.rd.compute_list_begin())
    }

    /// Timestamps must sit between compute lists, so close any open list.
    pub fn close_list(&mut self) {
        if self.list.take().is_some() {
            self.rd.compute_list_end();
        }
    }

    /// Close the list and capture the end marker; returns the captured labels.
    pub fn finish(mut self) -> Vec<&'static str> {
        self.close_list();
        self.rd.capture_timestamp("celestial/chunk-end");
        self.captured.push("celestial/chunk-end");
        self.captured
    }
}

impl RecordCtx for ChunkCtx<'_> {
    fn timestamp(&mut self, label: &str) {
        self.close_list();
        self.rd.capture_timestamp(label);
        if let Some(stat) = STATIC_LABELS.iter().find(|l| **l == label) {
            self.captured.push(stat);
        }
    }
}

const STATIC_LABELS: [&str; 6] = [
    "celestial/chunk-upload",
    "celestial/chunk-surface-custom",
    "celestial/chunk-realize",
    "celestial/chunk-bake",
    "celestial/chunk-scatter-place",
    "celestial/chunk-scatter-compact",
];

#[cfg(test)]
mod stage_tests {
    use super::*;

    fn stage(chunks: u32, tag: u8) -> ChunkStage {
        ChunkStage {
            desc_bytes: vec![tag; chunks as usize * CHUNK_DESC_STRIDE],
            realize_count: chunks,
            instance_bytes: vec![tag; 8],
            instance_count: chunks,
            surface_enabled: tag as f32,
            surface_height_scale: tag as f32 * 0.5,
            surface_patches: (0..chunks).map(|s| (s, vec![tag; 4], vec![tag as f32], vec![tag; 4])).collect(),
            scatter_aux_bytes: vec![tag; chunks as usize * SCATTER_AUX_STRIDE],
            scatter_vis_bytes: vec![tag; 8],
            scatter_vis_count: chunks,
            scatter_layer_params: vec![vec![tag; 4]],
        }
    }

    #[test]
    fn desc_stride_matches_chunk_gpu() {
        assert_eq!(
            CHUNK_DESC_STRIDE,
            std::mem::size_of::<crate::chunk_descriptors::ChunkGpu>()
        );
    }

    #[test]
    fn merge_appends_realizes_and_snapshots_instances() {
        // Regression for the dropped-batch bug: merging must KEEP the pending
        // batch's realize work and only replace the per-frame snapshot parts.
        let mut pending = stage(3, 1);
        merge_stage(&mut pending, stage(2, 2), 1024);
        assert_eq!(pending.realize_count, 5, "realizes accumulate");
        assert_eq!(pending.desc_bytes.len(), 5 * CHUNK_DESC_STRIDE);
        assert_eq!(&pending.desc_bytes[..CHUNK_DESC_STRIDE], &[1u8; CHUNK_DESC_STRIDE][..]);
        assert_eq!(pending.surface_patches.len(), 5);
        // Snapshot parts take the NEWEST values.
        assert_eq!(pending.instance_bytes, vec![2u8; 8]);
        assert_eq!(pending.instance_count, 2);
        assert_eq!(pending.surface_enabled, 2.0);
    }

    #[test]
    fn merge_overflow_drops_oldest_realizes() {
        let mut pending = stage(3, 1);
        merge_stage(&mut pending, stage(2, 2), 4); // budget 4 < 3+2
        assert_eq!(pending.realize_count, 4);
        assert_eq!(pending.desc_bytes.len(), 4 * CHUNK_DESC_STRIDE);
        // The oldest (tag 1) descriptor was dropped; the newest survive.
        assert_eq!(
            &pending.desc_bytes[3 * CHUNK_DESC_STRIDE..],
            &[2u8; CHUNK_DESC_STRIDE][..]
        );
        assert_eq!(pending.surface_patches.len(), 4);
    }

    /// CEL-91: a stalled render thread must not let the merged batch accumulate
    /// surface patches without bound — each is ~0.75 MiB at tile_res 256. The
    /// cap is one pool's worth (`budget_chunks`), oldest dropped.
    #[test]
    fn merge_caps_surface_patches_at_the_budget() {
        const BUDGET: usize = 4;
        let mut pending = stage(0, 0);
        for tag in 1..=50u8 {
            merge_stage(&mut pending, stage(3, tag), BUDGET);
            assert!(pending.realize_count as usize <= BUDGET);
            assert!(
                pending.surface_patches.len() <= BUDGET,
                "surface patches ({}) grew past the budget",
                pending.surface_patches.len()
            );
        }
        assert_eq!(pending.realize_count as usize, BUDGET);
        assert_eq!(pending.desc_bytes.len(), BUDGET * CHUNK_DESC_STRIDE);
        assert_eq!(pending.scatter_aux_bytes.len(), BUDGET * SCATTER_AUX_STRIDE);
    }

    /// Patches are NOT strictly 1:1 with realizes (a stage can carry patches with
    /// realize_count under budget), so the realize-overflow drain alone does not
    /// bound them. They must be capped on their own.
    #[test]
    fn patches_are_capped_even_when_the_realize_count_is_under_budget() {
        const BUDGET: usize = 4;
        let mut pending = stage(0, 0);
        for tag in 1..=20u8 {
            let mut s = stage(3, tag);
            s.realize_count = 0; // no realize overflow — patches only
            s.desc_bytes.clear();
            s.scatter_aux_bytes.clear();
            merge_stage(&mut pending, s, BUDGET);
        }
        assert_eq!(pending.realize_count, 0);
        assert_eq!(pending.surface_patches.len(), BUDGET, "patch memory must stay bounded");
        // The NEWEST patches survive: the last batch (tag 20) plus one from 19.
        let tags: Vec<f32> = pending.surface_patches.iter().map(|p| p.2[0]).collect();
        assert_eq!(tags, vec![19.0, 20.0, 20.0, 20.0]);
    }
}

/// Render-thread job owning the chunk graph + GPU resources. (CEL-58 pattern:
/// a separate `RefCounted` so inline callbacks can't re-enter the node's borrow.)
#[derive(GodotClass)]
#[class(base = RefCounted, no_init)]
pub struct CesChunkJob {
    base: Base<RefCounted>,
    pub gpu: ChunkGpuResources,
    graph: Graph<Box<dyn ChunkNode>>,
    upload_node: NodeId,
    /// Staged by the main thread; drained on the render thread.
    pub stage: Option<ChunkStage>,
    /// Current terrain params (uploaded with the batch).
    pub terrain: TerrainGpu,
    pub res: u32,
    /// Number of graph executes that actually ran (consumed a stage). Stays flat
    /// on a stationary camera — the no-readback / cache win the HUD verifies.
    pub executes: u64,
    /// Timestamp labels captured last execute (drained the next execute).
    pending_labels: Vec<&'static str>,
    /// Per-stage GPU milliseconds from the previous execute: `(node name, ms)`
    /// for each graph node that ran (`chunk-upload`, `chunk-realize`, and any
    /// future pass). Read by the node for the HUD breakdown.
    pub gpu_ms: Vec<(&'static str, f64)>,
}

impl CesChunkJob {
    /// Build a job targeting `mm_rid` (or `Rid::Invalid` to self-create the
    /// multimesh) with capacity `budget` chunks at resolution `res`.
    pub fn create(
        mm_rid: Rid,
        budget: u32,
        res: u32,
        tile_res: u32,
        radius: f32,
        bump_enable: f32,
        terrain: TerrainGpu,
        scatter: Vec<crate::gpu::chunk_gpu::ScatterConfig>,
    ) -> Gd<Self> {
        let mut graph: Graph<Box<dyn ChunkNode>> = Graph::new();
        let reg = build_chunk_pipeline(&mut graph);
        let mut gpu = ChunkGpuResources::new(mm_rid, budget, res, tile_res, radius);
        gpu.set_bump_enable(bump_enable);
        gpu.set_scatter_configs(scatter);
        Gd::from_init_fn(|base| Self {
            base,
            gpu,
            graph,
            upload_node: reg.upload,
            stage: None,
            terrain,
            res,
            executes: 0,
            pending_labels: Vec::new(),
            gpu_ms: Vec::new(),
        })
    }
}

#[godot_api]
impl CesChunkJob {
    /// Render-thread entry point. Retries silently until resources are ready.
    #[func]
    fn run(&mut self) {
        if self.stage.is_none() {
            return;
        }
        let rs = RenderingServer::singleton();
        let Some(mut rd) = rs.get_rendering_device() else { return };
        // Drain the PREVIOUS execute's GPU timestamps into per-stage ms.
        self.drain_timestamps(&mut rd);
        if !self.gpu.ensure_ready(&mut rd) {
            if self.gpu.init_failed() {
                self.stage = None;
            }
            return; // retry next frame
        }
        self.graph.mark_dirty(self.upload_node);
        let mut ctx = ChunkCtx::new(rd, &mut self.gpu, &mut self.stage, self.terrain, self.res);
        self.graph.execute(&mut ctx);
        self.pending_labels = ctx.finish();
        self.executes += 1;
    }

    /// Convert the previous execute's timestamp pairs into per-stage milliseconds.
    /// Each consecutive label pair (`chunk-upload`→`chunk-realize`→`chunk-end`,
    /// plus any future stage) becomes one `(name, ms)` entry. Mirrors
    /// `PlanetJob::drain_timestamps`. Timestamps are read by name, so this is
    /// robust even if the device's capture list is shared with other jobs.
    fn drain_timestamps(&mut self, rd: &mut Gd<RenderingDevice>) {
        if self.pending_labels.is_empty() {
            return;
        }
        let n = rd.get_captured_timestamps_count();
        if n == 0 {
            return;
        }
        let mut times: HashMap<String, u64> = HashMap::new();
        for i in 0..n {
            let name = rd.get_captured_timestamp_name(i).to_string();
            times.insert(name, rd.get_captured_timestamp_gpu_time(i));
        }
        let labels = std::mem::take(&mut self.pending_labels);
        let mut out = Vec::new();
        for pair in labels.windows(2) {
            if let (Some(&a), Some(&b)) = (times.get(pair[0]), times.get(pair[1])) {
                out.push((pair[0], (b.saturating_sub(a)) as f64 / 1.0e6));
            }
        }
        if !out.is_empty() {
            self.gpu_ms = out;
        }
    }

    /// TEMP DEBUG (render-thread): read back the vertex pool and print each
    /// used slot's radius range, to see which chunks sit at inconsistent radii.
    #[func]
    fn debug_dump_radii(&mut self, slots: PackedInt32Array) {
        let rs = RenderingServer::singleton();
        let Some(mut rd) = rs.get_rendering_device() else { return };
        let buf = self.gpu.pos_buf();
        if buf.is_invalid() {
            return;
        }
        let vpc = self.gpu.verts_per_chunk() as usize;
        let data = rd.buffer_get_data(buf);
        let bytes = data.as_slice();
        let floats: &[f32] = bytemuck::cast_slice(bytes);
        // pos_tex is what the material ACTUALLY draws from — compare per texel.
        let tex_data = rd.texture_get_data(self.gpu.pos_tex(), 0);
        let tex_bytes = tex_data.as_slice();
        let tex: &[f32] = bytemuck::cast_slice(tex_bytes);
        godot_print!(
            "[radii] verts_buf {} floats | pos_tex {} floats ({} texels)",
            floats.len(),
            tex.len(),
            tex.len() / 4
        );
        // NaN hunt: min/max and |a-b| comparisons are NaN-blind, so count
        // non-finite components explicitly, per slot, in BOTH sources.
        let mut nan_slots: Vec<String> = Vec::new();
        let mut clean = 0u32;
        for &slot in slots.as_slice() {
            let s = slot as usize;
            let base = s * vpc * 4;
            if base + vpc * 4 > floats.len() {
                continue;
            }
            let mut buf_nan = 0usize;
            let mut tex_nan = 0usize;
            let mut first_v = usize::MAX;
            for v in 0..vpc {
                let gv = s * vpc + v;
                let b = &floats[base + v * 4..base + v * 4 + 3];
                if b.iter().any(|x| !x.is_finite()) {
                    buf_nan += 1;
                    if first_v == usize::MAX {
                        first_v = v;
                    }
                }
                if gv * 4 + 3 <= tex.len()
                    && tex[gv * 4..gv * 4 + 3].iter().any(|x| !x.is_finite())
                {
                    tex_nan += 1;
                }
            }
            if buf_nan > 0 || tex_nan > 0 {
                if nan_slots.len() < 16 {
                    nan_slots.push(format!(
                        "slot {slot}: buf {buf_nan}/{vpc} tex {tex_nan}/{vpc} first v{first_v}"
                    ));
                }
            } else {
                clean += 1;
            }
        }
        godot_print!(
            "[radii] NON-FINITE check: {} clean / {} total | {}",
            clean,
            slots.len(),
            nan_slots.join(" | ")
        );

        // WHERE is each tail slot's geometry: mean interior direction → lat/lon
        // (the tail of the drawn list is the deepest / nearest chunks). If a
        // slot's location isn't where the cut says its chunk is, the pool holds
        // a STALE chunk (realize never landed for the reassignment).
        let interior = (self.res as usize + 1) * (self.res as usize + 2) / 2;
        let show = slots.as_slice().len().saturating_sub(10);
        for &slot in &slots.as_slice()[show..] {
            let s = slot as usize;
            let base = s * vpc * 4;
            if base + vpc * 4 > floats.len() {
                continue;
            }
            let (mut sx, mut sy, mut sz) = (0f64, 0f64, 0f64);
            let (mut rmin, mut rmax) = (f32::MAX, f32::MIN);
            for v in 0..interior {
                let p = &floats[base + v * 4..base + v * 4 + 3];
                let r = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
                rmin = rmin.min(r);
                rmax = rmax.max(r);
                sx += p[0] as f64;
                sy += p[1] as f64;
                sz += p[2] as f64;
            }
            let len = (sx * sx + sy * sy + sz * sz).sqrt().max(1e-9);
            let lat = (sy / len).clamp(-1.0, 1.0).asin().to_degrees();
            let lon = sz.atan2(sx).to_degrees();
            godot_print!(
                "[radii] GPU slot {slot}: lat {lat:.2} lon {lon:.2} r[{rmin:.1},{rmax:.1}]"
            );
        }

        // Indirect draw truth: what instance count does the COMMAND BUFFER hold,
        // and which slots does the GPU-side instance buffer actually carry?
        let rs2 = RenderingServer::singleton();
        let mm = self.gpu.mm_rid();
        let cmd = rs2.multimesh_get_command_buffer_rd_rid(mm);
        let inst = rs2.multimesh_get_buffer_rd_rid(mm);
        if cmd.is_valid() {
            let cb = rd.buffer_get_data(cmd);
            let c = cb.as_slice();
            if c.len() >= 20 {
                let words: Vec<u32> = c[..20]
                    .chunks(4)
                    .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                    .collect();
                godot_print!("[radii] indirect cmd words: {:?}", words);
            }
        }
        if inst.is_valid() {
            let ib = rd.buffer_get_data(inst);
            let f: &[f32] = bytemuck::cast_slice(ib.as_slice());
            let stride = 16usize; // 12 transform + 4 custom floats
            let n = (slots.len() as usize).min(f.len() / stride); // ACTIVE instances
            let take = |i: usize| (f[i * stride + 12], f[i * stride + 13]);
            let mut head: Vec<String> = Vec::new();
            for i in 0..n.min(4) {
                let (s, m) = take(i);
                head.push(format!("i{i}=slot{:.0}/m{m:.2}", s));
            }
            let mut tail: Vec<String> = Vec::new();
            for i in n.saturating_sub(6)..n {
                let (s, m) = take(i);
                tail.push(format!("i{i}=slot{:.0}/m{m:.2}", s));
            }
            godot_print!(
                "[radii] active instances {} | head {} | tail {}",
                n,
                head.join(" "),
                tail.join(" ")
            );
            // Raw transform rows — pack_instances writes identity; anything
            // else means the renderer reads a different layout.
            for &i in [0usize, n / 2, n.saturating_sub(1)].iter() {
                let t = &f[i * stride..i * stride + 12];
                godot_print!(
                    "[radii] i{i} xform [{:.2} {:.2} {:.2} {:.2} | {:.2} {:.2} {:.2} {:.2} | {:.2} {:.2} {:.2} {:.2}]",
                    t[0], t[1], t[2], t[3], t[4], t[5], t[6], t[7], t[8], t[9], t[10], t[11]
                );
            }
        }
    }
}
