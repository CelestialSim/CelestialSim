//! All RD resources for one chunk multimesh: the `ChunkRealize` compute
//! pipeline + the shared vertex pool (positions buffer + attribute texture) +
//! the descriptor/params buffers + uniform set + an indirect `MultiMesh`.
//! Render-thread only. The chunk realize is a single vertex pass: it reads
//! `chunks`/`params` and writes the vertex pool — it does NOT bind the
//! multimesh, so the compute path is multimesh-independent.

use std::sync::Arc;

use bytemuck::Zeroable;
use godot::classes::rendering_device::UniformType;
use godot::classes::rendering_server::MultimeshTransformFormat;
use godot::classes::{
    ArrayMesh, Material, MultiMesh, RdUniform, RenderingDevice, RenderingServer, StandardMaterial3D,
};
use godot::prelude::*;

use super::device;
use super::owned::{
    LocalDeviceSink, MainDeviceSink, Owned, RdBuffer, RdPipeline, RdShader, RdTexture, RdUniformSet,
    RidSink,
};
use crate::chunk_descriptors::{pack_params, verts_per_chunk};
use crate::chunk_mesh::reference_chunk_mesh;
use crate::descriptors::TerrainGpu;
use crate::gpu::ATTR_TEX_WIDTH;

const CHUNK_REALIZE_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/ChunkRealize.spv"));
const CHUNK_TILE_BAKE_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/ChunkTileBake.spv"));
const SCATTER_PLACE_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/ScatterPlace.spv"));
const SCATTER_COMPACT_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/ScatterCompact.spv"));

/// Static config of one scatter layer's GPU side (CEL-73): the target indirect
/// MultiMesh + the per-slot candidate capacity (`k_per_cell * 4^S_MAX`) + the
/// instance cap the compact pass clamps to.
#[derive(Clone, Copy, Debug)]
pub struct ScatterConfig {
    pub mm_rid: Rid,
    pub capacity: u32,
    pub max_instances: u32,
}

/// RD resources owned by one scatter layer: the cached candidate pool
/// (`budget * capacity` records × 64 B), its params block, the compact pass's
/// atomic counter, and the two uniform sets.
///
/// FIELD ORDER IS THE DROP ORDER — see the drop-order contract in `gpu::owned`:
/// the uniform sets (dependents of the scatter shaders AND of the buffers below)
/// must be declared, hence freed, first.
struct ScatterLayerGpu {
    cfg: ScatterConfig,
    place_set: RdUniformSet,
    compact_set: RdUniformSet,
    pool_buf: RdBuffer,
    params_buf: RdBuffer,
    counter_buf: RdBuffer,
}

/// GPU resources owned by one chunk multimesh.
///
/// FIELD ORDER IS THE DROP ORDER (see the DROP-ORDER CONTRACT in `gpu::owned`):
/// **uniform sets → scatter layers (whose own sets come first) → pipelines →
/// shaders → buffers/textures**. RD frees dependents together with their parent,
/// so a set/pipeline must go before the shader or buffer it was built from.
/// Nothing enforces this but the declaration order below — do not reshuffle.
pub struct ChunkGpuResources {
    /// Who frees this set's RIDs (main device = deferred to the render thread;
    /// local device = no-op). Every `Owned` field holds a clone.
    sink: Arc<dyn RidSink>,

    mm_rid: Rid,
    /// Max chunk count (vertex-pool slots): the pool holds `budget * vpc` verts.
    budget: u32,
    res: u32,
    radius: f32,
    attr_w: u32,
    /// Vertices per chunk (`verts_per_chunk(res)`).
    vpc: u32,
    /// Phase 4: per-chunk detail-tile resolution (atlas is `tile_res²` / slot).
    tile_res: u32,
    /// Detail-normal bump enable (1.0 on, 0.0 off) — debug toggle passed through
    /// `ChunkParams` to `ChunkTileBake.slang`. Defaults to 1.0 (production look).
    bump_enable: f32,
    /// CPU-surface global toggle passed through `ChunkParams.surface_enabled`
    /// (1.0 on / 0.0 off). Defaults to 0.0 (procedural, identical to today).
    surface_enabled: f32,
    /// CPU-surface displaced-radius factor per meter, passed through
    /// `ChunkParams.surface_height_scale`. Defaults to 0.0.
    surface_height_scale: f32,

    // ---- 1. uniform sets (dependents — freed first) ----
    realize_set: RdUniformSet,
    bake_set: RdUniformSet,
    custom_set: RdUniformSet,

    /// CEL-73 scatter passes' per-layer resources (each holds its own sets first,
    /// then its buffers). Declared here so a layer's sets die before the scatter
    /// shaders below. Empty configs = scatter disabled.
    scatter_layers: Vec<ScatterLayerGpu>,

    // ---- 2. pipelines ----
    pipeline: RdPipeline,
    /// Phase-4 tile-bake compute pipeline (`ChunkTileBake.spv`).
    bake_pipeline: RdPipeline,
    custom_pipeline: RdPipeline,
    scatter_pipeline: RdPipeline,
    compact_pipeline: RdPipeline,

    // ---- 3. shaders (parents of the sets + pipelines above) ----
    shader: RdShader,
    bake_shader: RdShader,
    custom_shader: RdShader,
    scatter_shader: RdShader,
    compact_shader: RdShader,

    // ---- 4. buffers / textures (parents of the sets above) ----
    /// Per-chunk descriptors (`ChunkGpu` batch), binding 0.
    desc_buf: RdBuffer,
    /// `ChunkParams` (header + embedded `TerrainGpu` + tile_res), binding 1.
    params_buf: RdBuffer,
    /// Per-vertex attributes (2 rgba16f texels: color+layer, normal+height),
    /// binding 2 — sampled by the surface material.
    verts_tex: RdTexture,
    /// Per-vertex positions (1 float4: pos+height), binding 3 — the readback
    /// source the verification compares against `chunk_subvertex_base`.
    verts_buf: RdBuffer,
    /// Per-vertex world positions (1 rgba32f texel/vertex), binding 4 — the
    /// SAMPLEABLE position store the surface material reads VERTEX from (a
    /// spatial shader can't sample the `verts_buf` storage buffer).
    pos_tex: RdTexture,
    /// Per-chunk colour detail atlas (rgba8, `budget*tile_res²` texels wrapped
    /// at `attr_w`) — sampled by the surface fragment shader.
    color_atlas: RdTexture,
    /// Per-chunk world-normal detail atlas (rgba8, normal encoded `*0.5+0.5`).
    normal_atlas: RdTexture,
    /// Per-chunk CPU-surface colour storage buffer: one rgba8-packed `u32` per
    /// tile texel (`budget*tile_res²` u32s), indexed by linear texel `g`. Bound at
    /// bake-set binding 4.
    surface_color_buf: RdBuffer,
    /// Per-chunk CPU-surface NORMAL storage buffer: one rgba8-packed `u32`
    /// per tile texel (CPU-computed, curvature-correct).
    surface_normal_buf: RdBuffer,
    /// Per-chunk CPU-surface height storage buffer: one `f32` (meters) per tile
    /// texel, indexed by linear texel `g`. Bound at bake-set binding 5 and
    /// realize-set binding 5.
    surface_height_buf: RdBuffer,
    /// std430 `{chunk_count, tile_res, water_height, height_scale, cels_user[16]}`.
    custom_params_buf: RdBuffer,
    /// Per-realize-chunk `{path_lo, path_hi, face, pad}` aux (16 B × budget).
    scatter_aux_buf: RdBuffer,
    /// Per-visible-instance `{slot, depth}` gather list (8 B × budget).
    scatter_vis_buf: RdBuffer,

    // ---- plain state ----
    /// Custom **GPU** surface (user GLSL). Assembled source set before
    /// `ensure_ready`; the pipeline is compiled AT RUNTIME by Godot
    /// (`device::compute_pipeline_from_glsl`) and, when it succeeds, a per-slot
    /// dispatch fills `surface_color/height/normal` from the user's terrain
    /// functions (the `celestial/chunk-surface-custom` node). `None` source =
    /// no custom surface (procedural or CPU-provider path).
    custom_source: Option<String>,
    /// The layer's normalized sea level, fed to the template as `CELS_WATER_HEIGHT`.
    custom_water_height: f32,
    /// The layer's geometry displacement multiplier: used as
    /// `surface_height_scale` (realize does `radius·(1 + h·scale)`) and passed to
    /// the template as `CELS_HEIGHT_SCALE`.
    custom_height_scale: f32,
    /// Generic user params (one per `@export var name: float` on a GPU builder),
    /// packed into the tail of the custom params buffer as `cels_user[16]`.
    custom_user_params: Vec<f32>,
    custom_built: bool,
    /// Runtime GLSL compile failed — fall back to procedural (do NOT crash the
    /// planet). Distinct from `init_failed`, which disables the whole node.
    custom_failed: bool,
    scatter_cfgs: Vec<ScatterConfig>,
    scatter_built: bool,

    /// Compute-only mode: build the realize pipeline + vertex pool only, with
    /// no `MultiMesh` (used for the local-device readback verification, where
    /// the server-side multimesh — tied to the main device — has no role).
    compute_only: bool,
    /// Whether the pipeline + pool + set are built.
    built: bool,
    init_failed: bool,
    /// Keep-alive for the server-side multimesh (created here when `mm_rid` was
    /// `Invalid`); its RD buffers live as long as this `Gd` does. SERVER-side
    /// RIDs (RenderingServer / Resource), not RenderingDevice ones — freed by
    /// dropping the `Gd`, not through the sink.
    multimesh: Option<Gd<MultiMesh>>,
    /// Keep-alives so the reference mesh + material outlive the multimesh.
    _template: Option<Gd<ArrayMesh>>,
    _material: Option<Gd<Material>>,
}

impl ChunkGpuResources {
    /// Construct an (unbuilt) resource set on the MAIN `RenderingDevice`.
    /// `mm_rid` is the target multimesh, or `Rid::Invalid` to have `ensure_ready`
    /// create its own (CEL-58 order).
    pub fn new(mm_rid: Rid, budget: u32, res: u32, tile_res: u32, radius: f32) -> Self {
        Self::with_sink(mm_rid, budget, res, tile_res, radius, false, MainDeviceSink::new())
    }

    /// Compute-only resources (no `MultiMesh`) on a LOCAL `RenderingDevice`, for
    /// the readback verification. `ensure_ready` builds just the realize pipeline
    /// + pool; the local device frees its own resources when it dies, so the sink
    /// is a no-op.
    pub fn new_compute_only(budget: u32, res: u32, tile_res: u32, radius: f32) -> Self {
        Self::with_sink(Rid::Invalid, budget, res, tile_res, radius, true, LocalDeviceSink::new())
    }

    /// Construct with an explicit [`RidSink`] (tests inject a spy).
    pub fn with_sink(
        mm_rid: Rid,
        budget: u32,
        res: u32,
        tile_res: u32,
        radius: f32,
        compute_only: bool,
        sink: Arc<dyn RidSink>,
    ) -> Self {
        Self {
            mm_rid,
            budget: budget.max(1),
            res,
            radius,
            attr_w: ATTR_TEX_WIDTH,
            vpc: verts_per_chunk(res),
            tile_res: tile_res.max(1),
            bump_enable: 0.0,
            surface_enabled: 0.0,
            surface_height_scale: 0.0,
            realize_set: Owned::invalid(sink.clone()),
            bake_set: Owned::invalid(sink.clone()),
            custom_set: Owned::invalid(sink.clone()),
            scatter_layers: Vec::new(),
            pipeline: Owned::invalid(sink.clone()),
            bake_pipeline: Owned::invalid(sink.clone()),
            custom_pipeline: Owned::invalid(sink.clone()),
            scatter_pipeline: Owned::invalid(sink.clone()),
            compact_pipeline: Owned::invalid(sink.clone()),
            shader: Owned::invalid(sink.clone()),
            bake_shader: Owned::invalid(sink.clone()),
            custom_shader: Owned::invalid(sink.clone()),
            scatter_shader: Owned::invalid(sink.clone()),
            compact_shader: Owned::invalid(sink.clone()),
            desc_buf: Owned::invalid(sink.clone()),
            params_buf: Owned::invalid(sink.clone()),
            verts_tex: Owned::invalid(sink.clone()),
            verts_buf: Owned::invalid(sink.clone()),
            pos_tex: Owned::invalid(sink.clone()),
            color_atlas: Owned::invalid(sink.clone()),
            normal_atlas: Owned::invalid(sink.clone()),
            surface_color_buf: Owned::invalid(sink.clone()),
            surface_normal_buf: Owned::invalid(sink.clone()),
            surface_height_buf: Owned::invalid(sink.clone()),
            custom_params_buf: Owned::invalid(sink.clone()),
            scatter_aux_buf: Owned::invalid(sink.clone()),
            scatter_vis_buf: Owned::invalid(sink.clone()),
            custom_source: None,
            custom_water_height: 0.45,
            custom_height_scale: 1.0,
            custom_user_params: Vec::new(),
            custom_built: false,
            custom_failed: false,
            scatter_cfgs: Vec::new(),
            scatter_built: false,
            compute_only,
            built: false,
            init_failed: false,
            multimesh: None,
            _template: None,
            _material: None,
            sink,
        }
    }

    /// Set the detail-normal bump enable (1.0 on, 0.0 off) before the params are
    /// packed. Takes effect on the next `upload_params`/bake.
    pub fn set_bump_enable(&mut self, v: f32) {
        self.bump_enable = v;
    }

    /// Set the CPU-surface provider params (global toggle +
    /// displaced-radius factor per unit) before the params are packed. Takes
    /// effect on the next `upload_params`/bake.
    pub fn set_surface(&mut self, enabled: f32, height_scale: f32) {
        self.surface_enabled = enabled;
        self.surface_height_scale = height_scale;
    }

    /// Install a custom **GPU** surface: assembled GLSL `source`
    /// (`crate::custom_surface::assemble_source` output) plus the layer's
    /// `water_height` / `height_scale`. Call before the first `ensure_ready`; the
    /// pipeline is compiled lazily on the render thread. A shader-file change
    /// rebuilds the whole job, so this is only ever set once per job.
    pub fn set_custom_surface(
        &mut self,
        source: String,
        water_height: f32,
        height_scale: f32,
        user_params: Vec<f32>,
    ) {
        debug_assert!(!self.custom_built, "custom surface must be set before ensure_ready");
        self.custom_source = Some(source);
        self.custom_water_height = water_height;
        self.custom_height_scale = height_scale;
        self.custom_user_params = user_params;
    }

    /// Update the custom surface's LIVE knobs (water level + geometry scale +
    /// generic user params) without recompiling. Picked up by the next
    /// `upload_custom_params`; pair with a cache `invalidate_all` so resident
    /// chunks re-fill with the new values. No-op if no custom surface is
    /// installed.
    pub fn set_custom_knobs(&mut self, water_height: f32, height_scale: f32, user_params: Vec<f32>) {
        self.custom_water_height = water_height;
        self.custom_height_scale = height_scale;
        self.custom_user_params = user_params;
    }

    /// Was a custom GPU surface requested (source installed)?
    pub fn custom_requested(&self) -> bool {
        self.custom_source.is_some()
    }

    /// Is the custom GPU surface compiled and ready to dispatch? `false` while
    /// still building, or permanently after a compile error (fall back to
    /// procedural).
    pub fn custom_ready(&self) -> bool {
        self.custom_built && !self.custom_failed && self.custom_pipeline.is_valid()
    }

    /// The sink every owned RID of this set is released through.
    fn sink(&self) -> Arc<dyn RidSink> {
        self.sink.clone()
    }

    /// The custom layer's geometry displacement multiplier (used as
    /// `surface_height_scale` for the realize displacement).
    pub fn custom_height_scale(&self) -> f32 {
        self.custom_height_scale
    }

    /// Configure the scatter layers (CEL-73). Call before the first
    /// `ensure_ready`; layer resources are built there once every target
    /// multimesh's RD buffers exist.
    pub fn set_scatter_configs(&mut self, cfgs: Vec<ScatterConfig>) {
        debug_assert!(!self.scatter_built, "scatter configs must be set before ensure_ready");
        self.scatter_cfgs = cfgs;
    }

    /// Rows in the verts texture (2 texels per vertex) for the whole pool.
    fn verts_tex_rows(&self) -> u32 {
        (self.budget as u64 * self.vpc as u64 * 2).div_ceil(self.attr_w as u64).max(1) as u32
    }

    /// Rows in the position texture (1 texel per vertex) for the whole pool.
    fn pos_tex_rows(&self) -> u32 {
        (self.budget as u64 * self.vpc as u64).div_ceil(self.attr_w as u64).max(1) as u32
    }

    /// Texels per atlas slot (`tile_res²`).
    fn tile_texels(&self) -> u64 {
        self.tile_res as u64 * self.tile_res as u64
    }

    /// Rows in a detail atlas (1 texel per tile texel) for the whole budget.
    fn atlas_rows(&self) -> u32 {
        (self.budget as u64 * self.tile_texels()).div_ceil(self.attr_w as u64).max(1) as u32
    }

    /// Build the pipeline + vertex pool + uniform set + multimesh. Idempotent;
    /// returns false (retry next frame) until everything — including the
    /// multimesh's lazily-created command buffer — exists.
    pub fn ensure_ready(&mut self, rd: &mut Gd<RenderingDevice>) -> bool {
        if self.init_failed {
            return false;
        }
        if !self.built {
            // CEL-58 alloc order: pipeline → pool buffers/texture → set, then
            // the multimesh (indirect alloc BEFORE set_mesh).
            let sink = self.sink();
            let Some((shader, pipeline)) =
                device::compute_pipeline(rd, &sink, CHUNK_REALIZE_SPV, "ChunkRealize")
            else {
                self.init_failed = true;
                return false;
            };
            self.shader = shader;
            self.pipeline = pipeline;

            // Phase-4 bake pipeline (shares the desc/params buffers).
            let Some((bake_shader, bake_pipeline)) =
                device::compute_pipeline(rd, &sink, CHUNK_TILE_BAKE_SPV, "ChunkTileBake")
            else {
                self.init_failed = true;
                return false;
            };
            self.bake_shader = bake_shader;
            self.bake_pipeline = bake_pipeline;

            let desc_bytes =
                self.budget as usize * std::mem::size_of::<crate::chunk_descriptors::ChunkGpu>();
            self.desc_buf = device::storage_buffer(rd, &sink, &vec![0u8; desc_bytes]);
            // 96-byte ChunkParams; pre-fill with a terrain-off header.
            let params0 = pack_params(
                self.res,
                self.vpc,
                self.attr_w,
                0,
                self.tile_res,
                self.bump_enable,
                &TerrainGpu::zeroed(),
                0.0,
                0.0,
            );
            self.params_buf = device::storage_buffer(rd, &sink, &params0);
            self.verts_buf =
                device::storage_buffer_empty(rd, &sink, self.budget as u64 * self.vpc as u64 * 16);
            self.verts_tex =
                device::attribute_texture(rd, &sink, self.attr_w, self.verts_tex_rows());
            self.pos_tex = device::position_texture(rd, &sink, self.attr_w, self.pos_tex_rows());
            self.color_atlas = device::atlas_texture(rd, &sink, self.attr_w, self.atlas_rows());
            self.normal_atlas = device::atlas_texture(rd, &sink, self.attr_w, self.atlas_rows());

            // Per-chunk CPU-surface color/height storage buffers: one u32 / one
            // f32 per tile texel across the whole budget (4 bytes each).
            let surface_bytes = self.budget as u64 * self.tile_texels() * 4;
            self.surface_color_buf = device::storage_buffer_empty(rd, &sink, surface_bytes);
            self.surface_height_buf = device::storage_buffer_empty(rd, &sink, surface_bytes);
            self.surface_normal_buf = device::storage_buffer_empty(rd, &sink, surface_bytes);

            let uniforms: Array<Gd<RdUniform>> = [
                device::uniform(UniformType::STORAGE_BUFFER, 0, self.desc_buf.rid()),
                device::uniform(UniformType::STORAGE_BUFFER, 1, self.params_buf.rid()),
                device::uniform(UniformType::IMAGE, 2, self.verts_tex.rid()),
                device::uniform(UniformType::STORAGE_BUFFER, 3, self.verts_buf.rid()),
                device::uniform(UniformType::IMAGE, 4, self.pos_tex.rid()),
                device::uniform(UniformType::STORAGE_BUFFER, 5, self.surface_height_buf.rid()),
            ]
            .into_iter()
            .collect();
            self.realize_set =
                Owned::new(rd.uniform_set_create(&uniforms, self.shader.rid(), 0), sink.clone());

            // Bake set (set 0 against the bake shader): desc + params + atlases.
            let bake_uniforms: Array<Gd<RdUniform>> = [
                device::uniform(UniformType::STORAGE_BUFFER, 0, self.desc_buf.rid()),
                device::uniform(UniformType::STORAGE_BUFFER, 1, self.params_buf.rid()),
                device::uniform(UniformType::IMAGE, 2, self.color_atlas.rid()),
                device::uniform(UniformType::IMAGE, 3, self.normal_atlas.rid()),
                device::uniform(UniformType::STORAGE_BUFFER, 4, self.surface_color_buf.rid()),
                device::uniform(UniformType::STORAGE_BUFFER, 5, self.surface_height_buf.rid()),
                device::uniform(UniformType::STORAGE_BUFFER, 6, self.surface_normal_buf.rid()),
            ]
            .into_iter()
            .collect();
            self.bake_set = Owned::new(
                rd.uniform_set_create(&bake_uniforms, self.bake_shader.rid(), 0),
                sink.clone(),
            );

            if !self.compute_only && self.mm_rid.is_invalid() {
                self.create_multimesh();
            }
            self.built = true;
        }

        // Custom GPU surface (user GLSL): compile once, on the render thread, now
        // that the desc + surface buffers exist. A compile error disables the
        // custom path (falls back to procedural) WITHOUT failing the planet.
        if self.custom_source.is_some() && !self.custom_built && !self.custom_failed {
            self.ensure_custom_ready(rd);
        }

        if self.compute_only {
            // No multimesh to wait on. Scatter (CEL-73) still builds its
            // PLACE side for the local-device verification (configs carry
            // `Rid::Invalid` multimeshes, so no compact sets are created).
            if !self.scatter_cfgs.is_empty() && !self.scatter_built {
                return self.ensure_scatter_ready(rd);
            }
            return true;
        }

        // The multimesh's RD buffers are created lazily by the renderer.
        let rs = RenderingServer::singleton();
        if rs.multimesh_get_buffer_rd_rid(self.mm_rid).is_invalid()
            || rs.multimesh_get_command_buffer_rd_rid(self.mm_rid).is_invalid()
        {
            return false;
        }

        // CEL-73 second stage: scatter pipelines + per-layer pools/sets, once
        // every layer's lazily-created multimesh RD buffers exist too.
        if !self.scatter_cfgs.is_empty() && !self.scatter_built {
            if !self.ensure_scatter_ready(rd) {
                return false;
            }
        }
        true
    }

    /// Build the scatter pipelines + per-layer resources (CEL-73). Returns
    /// false (retry next frame) while any layer's multimesh RD buffers are
    /// still pending.
    fn ensure_scatter_ready(&mut self, rd: &mut Gd<RenderingDevice>) -> bool {
        let rs = RenderingServer::singleton();
        for cfg in &self.scatter_cfgs {
            // A config without a multimesh (compute-only verification) skips
            // the wait — its compact set simply stays Invalid.
            if cfg.mm_rid.is_valid()
                && (rs.multimesh_get_buffer_rd_rid(cfg.mm_rid).is_invalid()
                    || rs.multimesh_get_command_buffer_rd_rid(cfg.mm_rid).is_invalid())
            {
                return false;
            }
        }

        let sink = self.sink();
        let Some((shader, pipeline)) =
            device::compute_pipeline(rd, &sink, SCATTER_PLACE_SPV, "ScatterPlace")
        else {
            self.init_failed = true;
            return false;
        };
        self.scatter_shader = shader;
        self.scatter_pipeline = pipeline;
        let Some((cshader, cpipeline)) =
            device::compute_pipeline(rd, &sink, SCATTER_COMPACT_SPV, "ScatterCompact")
        else {
            self.init_failed = true;
            return false;
        };
        self.compact_shader = cshader;
        self.compact_pipeline = cpipeline;

        self.scatter_aux_buf = device::storage_buffer_empty(rd, &sink, self.budget as u64 * 16);
        // 1 uint (slot) per visible chunk.
        self.scatter_vis_buf = device::storage_buffer_empty(rd, &sink, self.budget as u64 * 4);

        let cfgs = std::mem::take(&mut self.scatter_cfgs);
        for cfg in &cfgs {
            let pool_bytes = self.budget as u64 * cfg.capacity as u64 * 64;
            let pool_buf = device::storage_buffer_empty(rd, &sink, pool_bytes);
            let params_buf = device::storage_buffer(
                rd,
                &sink,
                &[0u8; std::mem::size_of::<crate::scatter_descriptors::ScatterParamsGpu>()],
            );
            let counter_buf = device::storage_buffer_empty(rd, &sink, 4);

            let mut layer = ScatterLayerGpu {
                cfg: *cfg,
                place_set: Owned::invalid(sink.clone()),
                compact_set: Owned::invalid(sink.clone()),
                pool_buf,
                params_buf,
                counter_buf,
            };
            self.build_scatter_sets(rd, &mut layer);
            self.scatter_layers.push(layer);
        }
        self.scatter_built = true;
        true
    }

    /// Compile the user's custom-surface GLSL and build its uniform set (render
    /// thread). On a compile error, set `custom_failed` and return — the planet
    /// keeps running on the procedural path. Its inputs (desc + surface buffers)
    /// are created in the base `built` block, so this can run right after.
    fn ensure_custom_ready(&mut self, rd: &mut Gd<RenderingDevice>) {
        let Some(src) = self.custom_source.clone() else { return };
        let sink = self.sink();
        let Some((shader, pipeline)) =
            device::compute_pipeline_from_glsl(rd, &sink, &src, "CustomSurface")
        else {
            self.custom_failed = true;
            return;
        };
        self.custom_shader = shader;
        self.custom_pipeline = pipeline;
        let params0 = crate::custom_surface::pack_params(
            0,
            self.tile_res,
            self.custom_water_height,
            self.custom_height_scale,
            &self.custom_user_params,
        );
        self.custom_params_buf = device::storage_buffer(rd, &sink, &params0);
        let uniforms: Array<Gd<RdUniform>> = [
            device::uniform(UniformType::STORAGE_BUFFER, 0, self.desc_buf.rid()),
            device::uniform(UniformType::STORAGE_BUFFER, 1, self.surface_color_buf.rid()),
            device::uniform(UniformType::STORAGE_BUFFER, 2, self.surface_height_buf.rid()),
            device::uniform(UniformType::STORAGE_BUFFER, 3, self.surface_normal_buf.rid()),
            device::uniform(UniformType::STORAGE_BUFFER, 4, self.custom_params_buf.rid()),
        ]
        .into_iter()
        .collect();
        self.custom_set =
            Owned::new(rd.uniform_set_create(&uniforms, self.custom_shader.rid(), 0), sink.clone());
        self.custom_built = true;
    }

    /// (Re)create one layer's place/compact uniform sets. The compact set binds
    /// the layer's multimesh RD buffers, which the renderer can RECREATE when
    /// the multimesh is first actually drawn — invalidating the set — so this
    /// is also called from `upload_scatter` whenever a set has gone invalid. The
    /// assignment of a fresh set over a live one DROPS the old handle, which frees
    /// the stale set (this used to leak).
    fn build_scatter_sets(&self, rd: &mut Gd<RenderingDevice>, layer: &mut ScatterLayerGpu) {
        let rs = RenderingServer::singleton();
        let sink = self.sink();
        let place_uniforms: Array<Gd<RdUniform>> = [
            device::uniform(UniformType::STORAGE_BUFFER, 0, self.desc_buf.rid()),
            device::uniform(UniformType::STORAGE_BUFFER, 1, self.scatter_aux_buf.rid()),
            device::uniform(UniformType::STORAGE_BUFFER, 2, layer.params_buf.rid()),
            device::uniform(UniformType::STORAGE_BUFFER, 3, layer.pool_buf.rid()),
            // CPU-surface elevation grid: place samples the SAME heightmap
            // realize displaces the terrain with (procedural route ignores it).
            device::uniform(UniformType::STORAGE_BUFFER, 4, self.surface_height_buf.rid()),
        ]
        .into_iter()
        .collect();
        layer.place_set = Owned::new(
            rd.uniform_set_create(&place_uniforms, self.scatter_shader.rid(), 0),
            sink.clone(),
        );

        if layer.cfg.mm_rid.is_invalid() {
            // Compute-only verification: place side only.
            layer.compact_set = Owned::invalid(sink);
            return;
        }
        let mm_buf = rs.multimesh_get_buffer_rd_rid(layer.cfg.mm_rid);
        let cmd_buf = rs.multimesh_get_command_buffer_rd_rid(layer.cfg.mm_rid);
        let compact_uniforms: Array<Gd<RdUniform>> = [
            device::uniform(UniformType::STORAGE_BUFFER, 0, layer.params_buf.rid()),
            device::uniform(UniformType::STORAGE_BUFFER, 1, layer.pool_buf.rid()),
            device::uniform(UniformType::STORAGE_BUFFER, 2, self.scatter_vis_buf.rid()),
            device::uniform(UniformType::STORAGE_BUFFER, 3, layer.counter_buf.rid()),
            device::uniform(UniformType::STORAGE_BUFFER, 4, mm_buf),
            device::uniform(UniformType::STORAGE_BUFFER, 5, cmd_buf),
        ]
        .into_iter()
        .collect();
        layer.compact_set = Owned::new(
            rd.uniform_set_create(&compact_uniforms, self.compact_shader.rid(), 0),
            sink,
        );
    }

    /// Upload one execute's scatter staging (CEL-73): the realize batch's aux
    /// paths, the visible `{slot, depth}` list, and each layer's fresh params
    /// snapshot; zero each layer's compact counter + indirect draw count so the
    /// compact pass rebuilds the draw list from scratch. Also heals any uniform
    /// set the renderer invalidated by recreating a multimesh buffer.
    pub fn upload_scatter(
        &mut self,
        rd: &mut Gd<RenderingDevice>,
        aux_bytes: &[u8],
        vis_bytes: &[u8],
        layer_params: &[Vec<u8>],
    ) {
        if !self.scatter_built {
            return;
        }
        let mut layers = std::mem::take(&mut self.scatter_layers);
        for layer in &mut layers {
            if !rd.uniform_set_is_valid(layer.place_set.rid())
                || (layer.cfg.mm_rid.is_valid() && !rd.uniform_set_is_valid(layer.compact_set.rid()))
            {
                self.build_scatter_sets(rd, layer);
            }
        }
        self.scatter_layers = layers;
        if !aux_bytes.is_empty() {
            let n = aux_bytes.len().min(self.budget as usize * 16);
            rd.buffer_update(
                self.scatter_aux_buf.rid(),
                0,
                n as u32,
                &PackedByteArray::from(&aux_bytes[..n]),
            );
        }
        if !vis_bytes.is_empty() {
            let n = vis_bytes.len().min(self.budget as usize * 4);
            rd.buffer_update(
                self.scatter_vis_buf.rid(),
                0,
                n as u32,
                &PackedByteArray::from(&vis_bytes[..n]),
            );
        }
        let rs = RenderingServer::singleton();
        let zero = PackedByteArray::from(&0u32.to_le_bytes()[..]);
        for (layer, params) in self.scatter_layers.iter().zip(layer_params.iter()) {
            rd.buffer_update(
                layer.params_buf.rid(),
                0,
                params.len() as u32,
                &PackedByteArray::from(&params[..]),
            );
            rd.buffer_update(layer.counter_buf.rid(), 0, 4, &zero);
            if layer.cfg.mm_rid.is_valid() {
                let cmd = rs.multimesh_get_command_buffer_rd_rid(layer.cfg.mm_rid);
                if cmd.is_valid() {
                    rd.buffer_update(cmd, 4, 4, &zero);
                }
            }
        }
    }

    /// Record layer `li`'s place dispatch: one thread per candidate of each
    /// realize-batch chunk, clamped to the pool capacity (device-loss guard).
    pub fn record_scatter_place(
        &self,
        rd: &mut Gd<RenderingDevice>,
        list: i64,
        li: usize,
        chunk_count: u32,
    ) {
        let Some(layer) = self.scatter_layers.get(li) else { return };
        let max_threads = self.budget * layer.cfg.capacity;
        let threads = (chunk_count * layer.cfg.capacity).min(max_threads);
        let groups = threads.div_ceil(64);
        if groups == 0 {
            return;
        }
        rd.compute_list_bind_compute_pipeline(list, self.scatter_pipeline.rid());
        rd.compute_list_bind_uniform_set(list, layer.place_set.rid(), 0);
        rd.compute_list_dispatch(list, groups, 1, 1);
        rd.compute_list_add_barrier(list);
    }

    /// Record layer `li`'s compact dispatch: one thread per cached candidate of
    /// each VISIBLE chunk, clamped to the pool capacity.
    pub fn record_scatter_compact(
        &self,
        rd: &mut Gd<RenderingDevice>,
        list: i64,
        li: usize,
        vis_count: u32,
    ) {
        let Some(layer) = self.scatter_layers.get(li) else { return };
        if !layer.compact_set.is_valid() {
            return; // compute-only verification: no multimesh to compact into
        }
        let max_threads = self.budget * layer.cfg.capacity;
        let threads = (vis_count * layer.cfg.capacity).min(max_threads);
        let groups = threads.div_ceil(64);
        if groups == 0 {
            return;
        }
        rd.compute_list_bind_compute_pipeline(list, self.compact_pipeline.rid());
        rd.compute_list_bind_uniform_set(list, layer.compact_set.rid(), 0);
        rd.compute_list_dispatch(list, groups, 1, 1);
        rd.compute_list_add_barrier(list);
    }

    /// Number of built scatter layers.
    pub fn scatter_layer_count(&self) -> usize {
        self.scatter_layers.len()
    }

    /// Layer `li`'s cached candidate pool buffer (readback for the debug-only
    /// GPU-vs-CPU placement verification).
    pub fn scatter_pool_buf(&self, li: usize) -> Rid {
        self.scatter_layers.get(li).map_or(Rid::Invalid, |l| l.pool_buf.rid())
    }

    /// Allocate a server-side indirect MultiMesh (CEL-58 order) and keep its
    /// `Gd` alive.
    fn create_multimesh(&mut self) {
        let mut rs = RenderingServer::singleton();
        let material: Gd<Material> = StandardMaterial3D::new_gd().upcast();
        let template = reference_chunk_mesh(self.res, &material);
        let multimesh = MultiMesh::new_gd();
        let mm_rid = multimesh.get_rid();
        rs.multimesh_allocate_data_ex(mm_rid, self.budget as i32, MultimeshTransformFormat::TRANSFORM_3D)
            .custom_data_format(true)
            .use_indirect(true)
            .done();
        rs.multimesh_set_mesh(mm_rid, template.get_rid());
        self.mm_rid = mm_rid;
        self.multimesh = Some(multimesh);
        self._template = Some(template);
        self._material = Some(material);
    }

    /// Upload one batch of packed chunk descriptors (`count` × `ChunkGpu`).
    ///
    /// Guards `count <= budget`: an over-budget batch is clamped to the desc
    /// buffer's capacity so it can never overrun the descriptor buffer / vertex
    /// pool (the stale-oversized-buffer shape that caused a KRACKAN1 device-loss).
    pub fn upload_descs(&mut self, rd: &mut Gd<RenderingDevice>, bytes: &[u8], count: u32) {
        if bytes.is_empty() {
            return;
        }
        debug_assert!(
            count <= self.budget,
            "chunk realize count {count} exceeds budget {} — would overrun desc buffer",
            self.budget
        );
        let max = self.budget as usize * std::mem::size_of::<crate::chunk_descriptors::ChunkGpu>();
        let n = bytes.len().min(max);
        rd.buffer_update(self.desc_buf.rid(), 0, n as u32, &PackedByteArray::from(&bytes[..n]));
    }

    /// Rewrite the whole `ChunkParams` for a realize batch of `chunk_count`
    /// chunks with the given terrain.
    pub fn upload_params(
        &mut self,
        rd: &mut Gd<RenderingDevice>,
        chunk_count: u32,
        terrain: &TerrainGpu,
    ) {
        let bytes = pack_params(
            self.res,
            self.vpc,
            self.attr_w,
            chunk_count,
            self.tile_res,
            self.bump_enable,
            terrain,
            self.surface_enabled,
            self.surface_height_scale,
        );
        rd.buffer_update(
            self.params_buf.rid(),
            0,
            bytes.len() as u32,
            &PackedByteArray::from(&bytes[..]),
        );
    }

    /// Update only the embedded terrain (offset 16, 64 bytes) of `ChunkParams`.
    pub fn upload_terrain(&mut self, rd: &mut Gd<RenderingDevice>, terrain: &TerrainGpu) {
        let bytes = bytemuck::bytes_of(terrain);
        rd.buffer_update(self.params_buf.rid(), 16, bytes.len() as u32, &PackedByteArray::from(bytes));
    }

    /// Upload one chunk's CPU-surface color + height patch into the per-slot
    /// region of the surface storage buffers.
    ///
    /// `color_bytes` is `tile_res²` rgba8-packed texels (little-endian, 4 bytes
    /// each); `height` is `tile_res²` elevations in meters. Both are indexed by the
    /// linear texel `g = slot*tile_texels + ty*tile_res + tx`, matching the `g`
    /// `ChunkTileBake.slang` computes for `color_atlas`. No-op if `slot >= budget`,
    /// either buffer is unbuilt, or the inputs are empty.
    pub fn upload_surface(
        &mut self,
        rd: &mut Gd<RenderingDevice>,
        slot: u32,
        color_bytes: &[u8],
        height: &[f32],
        normal_bytes: &[u8],
    ) {
        if slot >= self.budget
            || !self.surface_color_buf.is_valid()
            || !self.surface_height_buf.is_valid()
            || !self.surface_normal_buf.is_valid()
        {
            return;
        }
        let off = slot as u64 * self.tile_texels() * 4;
        if !color_bytes.is_empty() {
            rd.buffer_update(
                self.surface_color_buf.rid(),
                off as u32,
                color_bytes.len() as u32,
                &PackedByteArray::from(color_bytes),
            );
        }
        if !height.is_empty() {
            let h: &[u8] = bytemuck::cast_slice(height);
            rd.buffer_update(
                self.surface_height_buf.rid(),
                off as u32,
                h.len() as u32,
                &PackedByteArray::from(h),
            );
        }
        if !normal_bytes.is_empty() {
            rd.buffer_update(
                self.surface_normal_buf.rid(),
                off as u32,
                normal_bytes.len() as u32,
                &PackedByteArray::from(normal_bytes),
            );
        }
    }

    /// Upload per-instance data into the multimesh buffer and set the indirect
    /// draw instance count (`cmd[1]`, byte offset 4). Mirrors
    /// `PlanetGpu::upload_face` / `validate_instances`' command-buffer write.
    pub fn upload_instances(&mut self, rd: &mut Gd<RenderingDevice>, bytes: &[u8], count: u32) {
        let rs = RenderingServer::singleton();
        let buffer = rs.multimesh_get_buffer_rd_rid(self.mm_rid);
        let command = rs.multimesh_get_command_buffer_rd_rid(self.mm_rid);
        if buffer.is_invalid() || command.is_invalid() {
            return;
        }
        if !bytes.is_empty() {
            rd.buffer_update(buffer, 0, bytes.len() as u32, &PackedByteArray::from(bytes));
        }
        let c = count.min(self.budget);
        rd.buffer_update(command, 4, 4, &PackedByteArray::from(&c.to_le_bytes()[..]));
    }

    /// Record the realize dispatch into `list`: one thread per pool vertex
    /// (`vert_threads` total), then a barrier so readers see the writes.
    pub fn record_realize(&self, rd: &mut Gd<RenderingDevice>, list: i64, vert_threads: u32) {
        // Guard the dispatch against an over-budget batch: clamp to the pool's
        // vertex capacity so the realize can never write past `verts`/`pos_tex`
        // (defensive against the stale-oversized-buffer device-loss shape).
        let max_threads = self.budget * self.vpc;
        debug_assert!(
            vert_threads <= max_threads,
            "chunk realize vert_threads {vert_threads} exceeds pool capacity {max_threads}"
        );
        let vert_threads = vert_threads.min(max_threads);
        let groups = vert_threads.div_ceil(64);
        if groups == 0 {
            return;
        }
        rd.compute_list_bind_compute_pipeline(list, self.pipeline.rid());
        rd.compute_list_bind_uniform_set(list, self.realize_set.rid(), 0);
        rd.compute_list_dispatch(list, groups, 1, 1);
        rd.compute_list_add_barrier(list);
    }

    /// Record the Phase-4 tile-bake dispatch into `list`: one thread per tile
    /// texel (`texel_threads` total = `realize_count * tile_res²`), clamped to
    /// the atlas capacity so a stale/over-budget batch can never write past the
    /// atlas (the same defensive guard as `record_realize`).
    pub fn record_bake(&self, rd: &mut Gd<RenderingDevice>, list: i64, texel_threads: u32) {
        let max_threads = (self.budget as u64 * self.tile_texels()).min(u32::MAX as u64) as u32;
        debug_assert!(
            texel_threads <= max_threads,
            "chunk bake texel_threads {texel_threads} exceeds atlas capacity {max_threads}"
        );
        let texel_threads = texel_threads.min(max_threads);
        let groups = texel_threads.div_ceil(64);
        if groups == 0 {
            return;
        }
        rd.compute_list_bind_compute_pipeline(list, self.bake_pipeline.rid());
        rd.compute_list_bind_uniform_set(list, self.bake_set.rid(), 0);
        rd.compute_list_dispatch(list, groups, 1, 1);
        rd.compute_list_add_barrier(list);
    }

    /// Refresh the custom-surface params UBO with this batch's `chunk_count`
    /// (and the live water/height knobs). A buffer transfer — call from the
    /// upload node, OUTSIDE any compute list. No-op unless the custom pipeline
    /// is ready.
    pub fn upload_custom_params(&self, rd: &mut Gd<RenderingDevice>, chunk_count: u32) {
        if !self.custom_ready() || !self.custom_params_buf.is_valid() {
            return;
        }
        let bytes = crate::custom_surface::pack_params(
            chunk_count.min(self.budget),
            self.tile_res,
            self.custom_water_height,
            self.custom_height_scale,
            &self.custom_user_params,
        );
        rd.buffer_update(
            self.custom_params_buf.rid(),
            0,
            bytes.len() as u32,
            &PackedByteArray::from(&bytes[..]),
        );
    }

    /// Record the custom-surface dispatch into `list`: one thread per tile texel
    /// of each realized chunk (`chunk_count · tile_res²`), clamped to the surface
    /// buffer capacity. No-op unless the custom pipeline is ready.
    pub fn record_custom_surface(&self, rd: &mut Gd<RenderingDevice>, list: i64, chunk_count: u32) {
        if !self.custom_ready() {
            return;
        }
        let max_threads = (self.budget as u64 * self.tile_texels()).min(u32::MAX as u64) as u32;
        let threads = ((chunk_count as u64 * self.tile_texels()).min(max_threads as u64)) as u32;
        let groups = threads.div_ceil(64);
        if groups == 0 {
            return;
        }
        rd.compute_list_bind_compute_pipeline(list, self.custom_pipeline.rid());
        rd.compute_list_bind_uniform_set(list, self.custom_set.rid(), 0);
        rd.compute_list_dispatch(list, groups, 1, 1);
        rd.compute_list_add_barrier(list);
    }

    /// Vertices per chunk (pool stride per slot).
    pub fn verts_per_chunk(&self) -> u32 {
        self.vpc
    }

    /// Per-chunk detail-tile resolution (atlas texels per slot = `tile_res²`).
    pub fn tile_res(&self) -> u32 {
        self.tile_res
    }

    /// The per-chunk colour detail atlas (rgba8) — sampled by the surface shader.
    pub fn color_atlas(&self) -> Rid {
        self.color_atlas.rid()
    }

    /// The per-chunk world-normal detail atlas (rgba8, encoded `*0.5+0.5`).
    pub fn normal_atlas(&self) -> Rid {
        self.normal_atlas.rid()
    }

    /// Planet radius (for the verification envelope check).
    pub fn radius(&self) -> f32 {
        self.radius
    }

    /// The vertex-pool positions buffer (1 float4/vertex: pos+height) — readback.
    pub fn pos_buf(&self) -> Rid {
        self.verts_buf.rid()
    }

    /// The per-vertex attribute texture (color+layer, normal+height).
    pub fn attr_tex(&self) -> Rid {
        self.verts_tex.rid()
    }

    /// The per-vertex SAMPLEABLE world-position texture (rgba32f); the surface
    /// material reads VERTEX from this.
    pub fn pos_tex(&self) -> Rid {
        self.pos_tex.rid()
    }

    /// The target multimesh.
    pub fn mm_rid(&self) -> Rid {
        self.mm_rid
    }

    /// True once `ensure_ready` failed permanently (shader/pipeline build error).
    pub fn init_failed(&self) -> bool {
        self.init_failed
    }

}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::owned::{Owned, RidSink};
    use std::sync::{Arc, Mutex};

    /// Records every RID handed to it, in order.
    #[derive(Default)]
    struct SpySink {
        freed: Mutex<Vec<Rid>>,
    }

    impl SpySink {
        fn freed(&self) -> Vec<Rid> {
            self.freed.lock().unwrap().clone()
        }
    }

    impl RidSink for SpySink {
        fn free(&self, rid: Rid) {
            self.freed.lock().unwrap().push(rid);
        }
    }

    /// Ownership contract (CEL-91): dropping the resource set frees every RD RID it
    /// holds exactly once, and frees dependents (uniform sets) before the parents
    /// (shaders / buffers) they were created from — see `gpu::owned`'s drop-order
    /// contract. No GPU needed: the sink is a spy.
    #[test]
    fn dropping_resources_frees_every_rid_once_dependents_first() {
        let spy = Arc::new(SpySink::default());
        let sink: Arc<dyn RidSink> = spy.clone();

        {
            let mut r =
                ChunkGpuResources::with_sink(Rid::Invalid, 4, 8, 8, 1.0, true, sink.clone());
            // Uniform sets (dependents).
            r.realize_set = Owned::new(Rid::new(101), sink.clone());
            r.bake_set = Owned::new(Rid::new(102), sink.clone());
            r.custom_set = Owned::new(Rid::new(103), sink.clone());
            // Pipelines.
            r.pipeline = Owned::new(Rid::new(201), sink.clone());
            r.bake_pipeline = Owned::new(Rid::new(202), sink.clone());
            // Shaders (parents of the sets/pipelines).
            r.shader = Owned::new(Rid::new(301), sink.clone());
            r.bake_shader = Owned::new(Rid::new(302), sink.clone());
            r.custom_shader = Owned::new(Rid::new(303), sink.clone());
            r.scatter_shader = Owned::new(Rid::new(304), sink.clone());
            r.compact_shader = Owned::new(Rid::new(305), sink.clone());
            // Buffers / textures (parents).
            r.desc_buf = Owned::new(Rid::new(401), sink.clone());
            r.params_buf = Owned::new(Rid::new(402), sink.clone());
            r.verts_buf = Owned::new(Rid::new(403), sink.clone());
            r.verts_tex = Owned::new(Rid::new(404), sink.clone());
            r.pos_tex = Owned::new(Rid::new(405), sink.clone());
            // One scatter layer: its sets are dependents of the scatter shaders and
            // of its own pool/params/counter buffers.
            r.scatter_layers.push(ScatterLayerGpu {
                cfg: ScatterConfig { mm_rid: Rid::Invalid, capacity: 1, max_instances: 1 },
                place_set: Owned::new(Rid::new(111), sink.clone()),
                compact_set: Owned::new(Rid::new(112), sink.clone()),
                pool_buf: Owned::new(Rid::new(411), sink.clone()),
                params_buf: Owned::new(Rid::new(412), sink.clone()),
                counter_buf: Owned::new(Rid::new(413), sink.clone()),
            });
            assert!(spy.freed().is_empty(), "nothing freed while alive");
        }

        let freed = spy.freed();
        let expected: Vec<Rid> = [
            101, 102, 103, 111, 112, 201, 202, 301, 302, 303, 304, 305, 401, 402, 403, 404, 405,
            411, 412, 413,
        ]
        .iter()
        .map(|&n| Rid::new(n))
        .collect();

        // Freed exactly once each (no double-free), and every RID accounted for.
        let mut sorted = freed.clone();
        sorted.sort_by_key(|r| r.to_u64());
        let mut want = expected.clone();
        want.sort_by_key(|r| r.to_u64());
        assert_eq!(sorted, want, "every owned RID freed exactly once");

        let pos = |n: u64| freed.iter().position(|r| *r == Rid::new(n)).unwrap();
        // Dependents before parents.
        assert!(pos(101) < pos(301), "realize_set freed before its shader");
        assert!(pos(102) < pos(302), "bake_set freed before its shader");
        assert!(pos(103) < pos(303), "custom_set freed before its shader");
        assert!(pos(111) < pos(304), "place_set freed before the scatter shader");
        assert!(pos(112) < pos(305), "compact_set freed before the compact shader");
        assert!(pos(111) < pos(411), "place_set freed before the layer pool buffer");
        assert!(pos(101) < pos(401), "realize_set freed before the desc buffer");
        assert!(pos(201) < pos(301), "pipeline freed before its shader");
    }
}
