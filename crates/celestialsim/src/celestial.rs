//! The `Celestial` node (Phase 2, CEL-62): a rendered, navigable
//! chunked quadtree planet. Strictly additive to the clipmap.
//!
//! Each frame the CPU selects the visible chunk cut (`select_chunks`), maps it to
//! stable GPU slots (`ChunkCache`), and — only when the cut changes — stages the
//! dirty descriptors + instances for a render-thread `CesChunkJob` that realizes
//! chunk vertices into a shared vertex pool (positions in a sampleable `pos_tex`,
//! attributes in `verts_tex`). The chunk MultiMesh's `terrain_chunk.gdshader`
//! material samples those textures, so geometry/colour come straight from the GPU
//! with no CPU readback. On a stationary camera the cut is identical, the cache
//! reports zero realizes, and no job is scheduled — `executes` stops growing.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Instant;

use celestial_algo::chunk_cache::CacheDiff;
use celestial_algo::surface_cache::SurfaceCache;
use celestial_algo::quadtree::{
    base_face_frames, morph_factor, select_chunks_displaced, Chunk, ChunkId, SurfaceFn,
};
use godot::classes::file_access::ModeFlags;
use godot::classes::rendering_server::MultimeshTransformFormat;
use godot::register::info::{PropertyInfo, PropertyUsageFlags};
use godot::classes::{
    INode3D, MultiMesh, MultiMeshInstance3D, Node3D, RenderingServer, Resource, Shader,
    ShaderMaterial, Texture2Drd,
};
use godot::prelude::*;

use crate::async_bake::{self, SubmitQueue, TAG_MASK};
use crate::bake_pool::BakePool;
use crate::chunk_descriptors::{pack_chunks, pack_instances, verts_per_chunk};
use crate::chunk_mesh::reference_chunk_mesh;
use crate::chunk_pipeline::{CesChunkJob, ChunkStage};
use crate::descriptors::{assemble, HeightGpu, TextureGpu};
use crate::gpu::chunk_gpu::ScatterConfig;
use crate::gpu::ATTR_TEX_WIDTH;
use crate::builder::{BuilderRoute, CesBuilder};
use crate::scatter_descriptors::{pack_scatter_aux, pack_scatter_params, pack_scatter_vis};
use crate::scatter_layer::CesScatterLayer;
use crate::surface::{ChunkSurface, CpuSurfaceProvider};

const CHUNK_SHADER: &str = "res://addons/celestialsim/terrain_chunk.gdshader";

#[derive(GodotClass)]
#[class(base = Node3D, tool, init)]
pub struct Celestial {
    base: Base<Node3D>,

    /// Sphere radius.
    #[export]
    #[init(val = 1000.0)]
    radius: f32,
    /// Screen-error LOD threshold (chunk edge / distance).
    #[export(range = (0.005, 0.5, 0.005))]
    #[init(val = 0.02)]
    screen_error: f32,
    /// Chunk edge resolution (segments per chunk edge).
    #[export(range = (2.0, 32.0, 1.0))]
    #[init(val = 16)]
    chunk_res: i64,
    /// Phase 4: per-chunk detail-tile resolution. Colour + normal detail is baked
    /// at `tile_res × tile_res` per chunk (independent of `chunk_res`) and sampled
    /// per-pixel, so surface shading is crisper than the geometry grid. NOTE: the
    /// detail atlas costs `tile_res² × 16 B` per resident chunk (detail atlases + surface buffers), so a larger
    /// `tile_res` means fewer chunks fit in `vram_budget_gib` (see `effective_budget`).
    #[export(range = (8.0, 1024.0, 1.0))]
    #[init(val = 32)]
    tile_res: i64,
    /// Maximum quadtree depth. 0 = no subdivision (the 20 base faces only — "LOD 0").
    #[export(range = (0.0, 20.0, 1.0))]
    #[init(val = 16)]
    max_depth: i64,
    /// GPU VRAM budget for the resident chunk pools, in **GiB**. You set the
    /// gigabytes, not a slot count: the number of resident chunk slots is derived
    /// as `vram / per-chunk bytes`, where per-chunk = `verts_per_chunk×48 B`
    /// (geometry) + `tile_res²×16 B` (detail atlases + surface buffers). So a bigger `tile_res`
    /// simply means fewer chunks fit in the same budget — no manual rebalancing.
    /// (Also clamped so the atlas texture height stays within the GPU's limit.)
    #[export(range = (0.05, 8.0, 0.05))]
    #[init(val = 1.0)]
    vram_budget_gib: f32,
    /// Geomorphing (Phase 5): smoothly blend each chunk between its full-detail
    /// grid and its coarser (parent-resolution) sublattice as the camera distance
    /// crosses the LOD band, so detail fades in/out instead of popping when the
    /// quadtree subdivides/merges. The per-chunk morph factor rides in the
    /// per-frame instance buffer (`INSTANCE_CUSTOM.g`) and is applied in the
    /// surface vertex shader, so the realize/bake cache is untouched. When off,
    /// every chunk gets `morph = 1` (full detail, no blend) and the instance
    /// buffer is NOT re-uploaded on camera movement.
    #[export]
    #[init(val = true)]
    geomorph: bool,
    /// Tint chunks by slot so the chunk tiling is visible.
    #[export]
    #[init(val = false)]
    lod_colors: bool,
    #[export]
    #[init(val = true)]
    debug_log: bool,
    /// Horizon (back-of-planet) culling: skip selecting/realizing/baking/drawing
    /// chunks that are fully beyond the planet's horizon. Removes the entire far
    /// hemisphere; and because the horizon is close when the camera is near the
    /// surface, it also drops distant near-ground chunks — shrinking the resident
    /// set (and the VRAM/`tile_res` it can afford).
    #[export]
    #[init(val = true)]
    horizon_cull: bool,
    /// Terrain-height slack for horizon culling, as a fraction of `radius`: a
    /// patch is kept if terrain up to `radius × cull_height_margin` above the
    /// surface could peek over the horizon. Raise if tall terrain pops in at the
    /// horizon; lower to cull more aggressively. (HQ max displacement ≈ 0.3·r.)
    #[export(range = (0.0, 1.0, 0.01))]
    #[init(val = 0.3)]
    cull_height_margin: f32,
    /// Max NEW chunks to realize+bake per frame. A large influx (teleport, fast
    /// turn, first fill) otherwise bakes every newly-visible chunk in one frame and
    /// spikes frame time; capping it spreads the work over frames (the rest pop in
    /// over the next few frames). Lower = smoother under influx, slower fill-in.
    /// Dirty re-bakes (streamed-tile arrivals) share this cap; already-resident
    /// chunks are always drawn (stale until their re-bake turn).
    #[export(range = (1.0, 4096.0, 1.0))]
    #[init(val = 48)]
    max_bakes_per_frame: i64,
    /// TEST (temporary): bypass the per-chunk cache and re-realize + re-bake every
    /// VISIBLE chunk EVERY frame (no persistence, no eviction). Lets you probe the
    /// raw per-frame realize+bake cost at high `tile_res`/`chunk_res` without the
    /// cache size limiting things. Off = normal cached path.
    #[export]
    #[init(val = false)]
    recompute_every_frame: bool,

    /// Scatter layers (CEL-73): each layer scatters one mesh over the planet on
    /// a stable world lattice, with LIVE `density` + `min_height`/`max_height`
    /// sliders — edits re-run only the scatter-compact dispatch. A layer with no
    /// mesh is INACTIVE (for grass assign `addons/celestialsim/grass_blade.tres`).
    /// `lod_level` sets density/reach; adding/removing layers (or changing
    /// `instances_per_cell`/`max_instances`) rebuilds the GPU job.
    #[export]
    scatter_layers: Array<Gd<CesScatterLayer>>,

    /// The terrain **builder** (one active at a time). Its TYPE (a `CesBuilder`
    /// subclass) decides how the surface is produced. A new planet starts with a
    /// GPU-example builder (added in [`ready`]); **clear it to render a plain
    /// white sphere**. Swapping or editing the builder reshades/rebuilds live.
    #[var(get = get_builder, set = set_builder)]
    #[export]
    builder: Option<Gd<CesBuilder>>,

    /// Hidden, storage-only guard: add the default builder the first time a
    /// builder-less planet is readied, then never again (so clearing the builder
    /// stays white). Not shown in the inspector — see `on_validate_property`.
    #[export]
    #[init(val = true)]
    auto_add_builder: bool,

    /// Analytic planetary water proxy (created lazily). Its toggle, water level,
    /// and look params all come from the active [`CesBuilder`], so water config
    /// travels with the terrain builder — see `update_water`.
    water: Option<crate::water_runtime::WaterRuntime>,

    /// The CPU-surface provider driving colour/height/normal (only for a
    /// `CpuNoise` builder → [`NoiseProvider`]); `None` for every other path.
    /// Shared (`Arc`) with the bake pool's worker threads.
    provider: Option<Arc<dyn CpuSurfaceProvider>>,
    /// Worker pool resampling chunk surfaces off the main thread (the fast-
    /// flight stutter fix): chunks are admitted only once their surface is ready.
    bake_pool: Option<BakePool>,
    /// Baked surfaces waiting for admission, keyed by chunk. Each entry is
    /// stamped with the `param_epoch` current when it arrived; a surface whose
    /// stamp is older than the CURRENT epoch was requested before the latest
    /// param edit and is rejected when it tries to be rendered.
    ///
    /// BOUNDED (CEL-91): each surface is `12 × tile_res²` bytes (0.75 MiB at
    /// tile_res 256), and this used to be a plain `HashMap` pruned only against
    /// the quadtree cut — so it grew to the cut size (1000-3000 chunks = GBs of
    /// RSS) no matter what `vram_budget_gib` said. It is now a FIFO
    /// [`SurfaceCache`] capped at `effective_budget()` — the SAME slot count the
    /// budget gives the GPU pool — so `vram_budget_gib` bounds the CPU side too.
    /// An evicted (never-admitted) surface simply re-bakes when its chunk is
    /// still in the cut.
    #[init(val = SurfaceCache::new(1))]
    ready_surfaces: SurfaceCache<(u64, ChunkSurface)>,
    /// The "last parameter update time": bumped the INSTANT a live builder edit
    /// is detected (not when the coalesced reshade is applied — that can lag
    /// behind by frames while a realize stage is in flight, a window in which
    /// old-param bakes used to slip through). Every ready surface carries the
    /// epoch it arrived under; render-time consumption rejects older stamps.
    param_epoch: u64,
    /// Resident chunks whose baked CPU surface was made with STALE params and must
    /// be re-baked. Owned here, NOT inferred from the cache's dirty flag: `update`
    /// consumes `dirty` as soon as it re-realizes a chunk (even though that realize
    /// used the old surface, because the new bake had not landed yet), while the
    /// per-frame bake budget only re-requests a handful of chunks. Chunks beyond
    /// that budget would lose `dirty` before ever being re-requested and keep
    /// old-param terrain forever. An id stays here until its FRESH surface actually
    /// arrives, so every resident chunk is guaranteed to refresh eventually.
    stale_surfaces: HashSet<ChunkId>,
    /// Tracks the provider's base-ready edge so the one-time "switch the initial
    /// procedural view to the baked surface" re-bake fires exactly once.
    was_base_ready: bool,
    /// The active `CpuCustom` builder (if any): its batched GDScript
    /// `height`/`color`/`normal` are called once per chunk, synchronously on the
    /// main thread during staging (realize count is already throttled per
    /// frame). Mutually exclusive with the other paths. See
    /// `docs/custom_terrain_cpu_gdscript.md`.
    #[init(val = None)]
    gd_baker: Option<Gd<CesBuilder>>,

    /// The active `CpuCustomAsync` builder (CEL-86), if any: the planet hands it
    /// batches of chunks via `_bake_requested` and never waits. Mutually
    /// exclusive with `gd_baker`, the provider, and the custom GPU surface.
    #[init(val = None)]
    gd_async_baker: Option<Gd<CesBuilder>>,
    /// That builder's hand-back queue, drained on the main thread each frame.
    /// Held separately so the drain doesn't need to `bind()` the builder while
    /// a worker thread may be pushing into it.
    #[init(val = None)]
    gd_submits: Option<Arc<SubmitQueue>>,
    /// Chunks handed to the async builder that haven't been submitted back.
    /// Pruned against the cut, so a chunk the camera flew past is simply
    /// re-requested if the player returns. This is the whole cancellation story.
    gd_outstanding: HashSet<ChunkId>,

    cache: Option<celestial_algo::chunk_cache::ChunkCache>,
    job: Option<Gd<CesChunkJob>>,
    run_cb: Option<Callable>,
    /// CEL-91: owned through `IndirectMultiMesh` so Godot's leaked indirect command
    /// buffer is released when the multimesh goes away.
    multimesh: Option<crate::gpu::owned::IndirectMultiMesh>,
    mmi: Option<Gd<MultiMeshInstance3D>>,
    material: Option<Gd<ShaderMaterial>>,
    _template: Option<Gd<godot::classes::ArrayMesh>>,

    /// CEL-73 scatter: one indirect MultiMesh child per layer (parallel vecs),
    /// the mesh keep-alives (RS `set_mesh` doesn't refcount), and the per-layer
    /// structural snapshot used to classify `changed` edits.
    scatter_mmis: Vec<Gd<MultiMeshInstance3D>>,
    scatter_mms: Vec<crate::gpu::owned::IndirectMultiMesh>,
    scatter_meshes: Vec<Gd<godot::classes::Mesh>>,
    scatter_snapshot: Vec<ScatterSnapshot>,
    /// Set by any layer's `changed` signal; consumed at staging (live edits) or
    /// by `check_scatter_structure` (structural edits).
    scatter_dirty: bool,

    /// The pos/verts texture RIDs currently wired into the material.
    #[init(val = Rid::Invalid)]
    wired_pos: Rid,
    #[init(val = Rid::Invalid)]
    wired_verts: Rid,
    /// The colour/normal detail atlas RIDs currently wired into the material.
    #[init(val = Rid::Invalid)]
    wired_color: Rid,
    #[init(val = Rid::Invalid)]
    wired_normal: Rid,
    /// Visible slots staged last time (skip re-staging an identical cut).
    last_slots: Vec<u32>,
    /// Camera (local-space) position last frame. Geomorph factors change whenever
    /// the camera moves even if the cut is identical, so when geomorph is on a
    /// camera move re-uploads ONLY the instance buffer (no realize/bake). `None`
    /// until the first frame.
    last_cam: Option<Vector3>,

    last_select_ms: f64,
    last_update_ms: f64,
    last_patch_ms: f64,
    last_stage_ms: f64,
    last_realize_count: i64,
    last_visible_count: i64,
    /// Depth of every chunk in the last cut (for `cut_report`).
    last_cut_depths: Vec<u8>,
    /// Centroid of every chunk in the last cut (debug: expected slot content).
    last_cut_centroids: Vec<Vector3>,
    status_accum: f64,

    /// Params baked into the built job/cache. A later change re-applies them:
    /// `chunk_res`/`tile_res`/`vram_budget_gib` shape the GPU buffers → full
    /// rebuild; `radius` is baked into every realized chunk's geometry (and its
    /// scattered instances) → invalidate the cache so all resident chunks
    /// re-realize at the new radius. Set at build in `ensure_job`; only read
    /// once the job exists. (`screen_error`/`max_depth`/culling need nothing —
    /// they change the per-frame cut directly.)
    built_radius: f32,
    built_res: i64,
    built_tile_res: i64,
    built_budget_gib: f32,
    /// Assembled custom-GPU-surface GLSL installed in the current job (`None` =
    /// no custom surface). A `changed` that yields a DIFFERENT source (shader
    /// file edited / added / removed) forces a rebuild to recompile; an
    /// identical source with new water/height knobs is a live edit.
    built_custom_source: Option<String>,
    /// The routing path the current job was built for (`None` = white / no
    /// builder). A `changed` that flips the route forces a rebuild.
    #[init(val = None)]
    built_builder_kind: Option<BuilderRoute>,
    /// Last-polled fingerprint of the active builder's `@export` FLOAT values
    /// (noise knobs / custom params). `None` until primed and after every
    /// rebuild. Godot does NOT emit `changed` for a plain `@export` edit, so
    /// `poll_builder_params` detects value edits by comparing this each frame —
    /// a builder needs no `emit_changed()` setter for its sliders to reshade.
    #[init(val = None)]
    last_param_values: Option<Vec<f32>>,
    /// A live builder edit is waiting to be applied (set by `on_builder_changed`,
    /// consumed in `process`). Deferring here COALESCES a burst of `changed`
    /// signals (e.g. dragging an inspector slider) into a single reshade, and
    /// gates it on no realize being in flight — so we never stack re-realizes.
    #[init(val = false)]
    terrain_reshade_pending: bool,
    /// The next `update_throttled_gated` should ignore `max_bakes_per_frame` and
    /// re-realize every dirty chunk in ONE frame. Set when a param edit is
    /// applied so the whole planet updates at once (no visible per-chunk stagger).
    #[init(val = false)]
    force_full_reshade: bool,
    /// Last-seen modified time of a `GpuCustom` builder's `.glsl` (editor only),
    /// so saving the shader file auto-recompiles. `0` = not yet baselined.
    #[init(val = 0)]
    shader_mtime: u64,
    /// Frame counter that throttles the shader-file mtime poll (see
    /// `poll_shader_reload`).
    #[init(val = 0)]
    shader_poll_ticks: u32,
}

#[godot_api]
impl INode3D for Celestial {
    /// Give a fresh planet its default `CesGPUNoiseExample` builder so it renders
    /// terrain immediately (safe here — the node is in the tree, unlike an
    /// instantiated property default). Fires once via the storage-only
    /// `auto_add_builder` guard, so clearing the builder and saving keeps the
    /// white planet.
    fn ready(&mut self) {
        if self.auto_add_builder && self.builder.is_none() {
            let b = CesBuilder::new_gd();
            // Attach the CesGPUNoiseExample script (built-in GPU noise). Loaded by PATH
            // so it works even before the GDScript class globals are registered;
            // falls back to a bare custom builder if the addon script is missing.
            if let Ok(script) = godot::tools::try_load::<godot::classes::Script>(
                "res://addons/celestialsim/builders/ces_gpu_noise.gd",
            ) {
                b.clone().upcast::<Object>().set_script(&script);
            }
            self.builder = Some(b);
        }
        self.auto_add_builder = false;
    }

    /// Hide the internal `auto_add_builder` guard from the inspector (kept for
    /// storage only), so it isn't a user-facing setting.
    fn on_validate_property(&self, property: &mut PropertyInfo) {
        if property.property_name.to_string() == "auto_add_builder" {
            property.usage = PropertyUsageFlags::STORAGE;
        }
    }

    // NOTE: deliberately NO `on_notification` teardown hook. Freeing the planet
    // already releases its GPU pool: `job` is a `Gd<CesChunkJob>` field, so
    // dropping `Celestial` drops the job, which drops `ChunkGpuResources`, whose
    // `Owned<K>` handles queue their RIDs for the render-thread free (CEL-91).
    // Adding an `on_notification` here is actively harmful: gdext `bind_mut()`s
    // the instance for EVERY notification before the handler filters it, so a
    // notification delivered while `process()` holds the borrow (the water
    // runtime adds a child mid-process) panics with a double-borrow abort.
    fn process(&mut self, delta: f64) {
        // CEL-73: layer-list / structural scatter edits rebuild the job before
        // this frame's ensure_job; live edits only mark the stage dirty.
        self.check_scatter_structure();
        // Re-apply planet param edits (radius/chunk_res/tile_res/vram) so ALL
        // resident chunks pick them up, not just newly-visible ones.
        self.reapply_param_changes();
        // Editor: auto-recompile when a custom `.glsl` is edited on disk. MUST be
        // before `ensure_job` — it may `teardown_job`, and ensure_job rebuilds it
        // this same frame (otherwise the rest of `process` unwraps a None cache).
        self.poll_shader_reload();
        self.ensure_job();
        // Idempotently connect each mesh layer's `changed` signal so a live
        // slider edit (or a layer assigned after the node was ready) reshades.
        self.connect_builders();
        // Detect @export knob/param edits (which emit no `changed` signal) and
        // flag a coalesced reshade — no `emit_changed()` setter needed.
        self.poll_builder_params();

        // Apply a coalesced live builder edit — but only once the previous
        // re-realize has been consumed by the render thread (stage empty). A
        // slider drag that fires many `changed` signals therefore collapses to
        // one reshade per completed frame, always at the latest value.
        if self.terrain_reshade_pending {
            let in_flight = self.job.as_ref().map(|j| j.bind().stage.is_some()).unwrap_or(false);
            if !in_flight {
                self.terrain_reshade_pending = false;
                self.apply_terrain_reshade();
            }
        }

        let Some(cam) = self.camera_local() else { return };

        let res = self.chunk_res.clamp(2, 32) as u32;
        let tile_res = self.tile_res.clamp(8, 1024) as u32;
        let budget = self.effective_budget();
        let frames = base_face_frames(self.radius);

        let t0 = Instant::now();
        // Horizon culling: drop patches fully behind the planet's horizon. The
        // margin keeps tall terrain that can peek over (max displacement ≈
        // radius × height_scale × 1.2; the export defaults cover the HQ terrain).
        let cull = if self.horizon_cull {
            Some(self.radius * self.cull_height_margin.max(0.0))
        } else {
            None
        };
        let se = self.screen_error;
        let geomorph = self.geomorph;
        // LOD distances are measured to the DISPLACED surface when CPU-surface
        // height data is present: with the bare-sphere distance a camera standing on
        // elevated terrain (e.g. a ~120 m peak × exaggeration) could never
        // bring `dist` under the local terrain height, capping the reachable
        // depth — the "detail stops sharpening near the ground" bug. The block
        // scopes the tile-cache borrow so the fetcher poll below can borrow mut.
        let (cut, morphs_cut) = {
            let surf_impl;
            // LOD is measured to the DISPLACED surface once the provider is
            // ready, so a camera standing on elevated terrain can still descend
            // to full depth (the "detail stops sharpening near the ground" bug).
            let radius = self.radius;
            let surface: Option<&SurfaceFn<'_>> = match self.provider.as_ref() {
                Some(p) if p.base_ready() => {
                    let provider = Arc::clone(p);
                    let scale = provider.height_scale();
                    surf_impl =
                        move |pt: Vector3| provider.sample_height(pt).unwrap_or(0.0) * scale * radius;
                    Some(&surf_impl)
                }
                _ => None,
            };

            let cut = select_chunks_displaced(
                &frames,
                cam,
                self.screen_error,
                res,
                self.max_depth.clamp(0, 20) as u8,
                cull,
                surface,
            );

            // Per-chunk geomorph factor (Phase 5), parallel to `cut`. Computed
            // from the SAME geometry `descend` uses (displaced centroid distance
            // + analytic tri-edge) so morph reaches 0 exactly as the parent takes
            // over — a seamless handover. Geomorph off => full detail (morph 1).
            let morphs: Vec<f32> = if geomorph {
                cut.iter()
                    .map(|c| {
                        let frame = &frames[c.id.face as usize];
                        // Gnomonic centroid — the corners are gnomonic, and the
                        // realize projection now matches them.
                        let pc = ((c.corners[0] + c.corners[1] + c.corners[2]) / 3.0)
                            .normalized()
                            * frame.radius;
                        let pc = match surface {
                            Some(f) => pc * (1.0 + f(pc) / pc.length().max(1.0e-6)),
                            None => pc,
                        };
                        let dist = (pc - cam).length();
                        morph_factor(frame.edge_len(), c.id.depth, res, dist, se)
                    })
                    .collect()
            } else {
                vec![1.0; cut.len()]
            };
            (cut, morphs)
        };
        self.last_select_ms = t0.elapsed().as_secs_f64() * 1000.0;

        // Ask the provider for any chunks whose awaited data has arrived
        // (streaming) and re-bake them; when the fresh surface lands (below) the
        // resident chunk is marked dirty and re-realized with it. The provider
        // owns all source-specific streaming — the planet just drives the trait.
        let refresh = self.provider.as_ref().map(|p| p.poll_refresh()).unwrap_or_default();
        let base_ready = self.provider.as_ref().map(|p| p.base_ready()).unwrap_or(false);
        if let Some(pool) = self.bake_pool.as_mut() {
            for chunk in &refresh {
                pool.request(&frames[chunk.id.face as usize], chunk);
            }
            // One-time handover: when the provider first becomes base-ready,
            // re-bake the whole current cut so the initial procedural view turns
            // into the baked surface.
            if base_ready && !self.was_base_ready {
                for chunk in &cut {
                    pool.request(&frames[chunk.id.face as usize], chunk);
                }
            }
        }
        self.was_base_ready = base_ready;

        // Drain finished bakes: stash the surface for admission/staging and mark
        // already-resident chunks dirty so they re-realize with the new data.
        let baked = self.bake_pool.as_mut().map(|p| p.poll()).unwrap_or_default();
        for r in baked {
            // `None` = cancelled (chunk flew out of view while queued); the
            // poll already released its in-flight entry — nothing to stage.
            let Some(surface) = r.surface else { continue };
            let id = r.chunk.id;
            self.ready_surfaces.insert(id, (self.param_epoch, surface));
            // A FRESH surface has landed (the pool drops results baked from stale
            // params), so this chunk is no longer stale.
            self.stale_surfaces.remove(&id);
            if let Some(c) = self.cache.as_mut() {
                if c.slot_of(id).is_some() {
                    c.mark_dirty(id);
                }
            }
        }

        // Enqueue bakes for cut chunks that NEED a surface: either not resident
        // yet (awaiting admission), or resident but DIRTY — a live param edit
        // calls `invalidate_all`, which keeps chunks RESIDENT and only marks them
        // dirty. Gating on `!resident` alone therefore never re-baked those
        // chunks: they re-realized against their stale surface and kept
        // old-param terrain, so only the chunks the camera happened to evict and
        // re-admit ever picked up the new values (the "some old chunks survive an
        // edit" bug). Skipped if a surface is already ready or in flight.
        // Bounded per frame + backpressure so a fast flight can't flood the queue.
        //
        // The production bounds are DERIVED from the consumption throttle
        // (`max_bakes_per_frame`, the rate at which admission actually drains
        // ready surfaces). Hard-coded 24-per-frame / 64-in-flight let the pool
        // produce ~3× what the planet consumed, so `ready_surfaces` saturated at
        // the cut bound (GBs). Producing at most one frame's worth of admissions,
        // with ~2 frames queued, keeps the ready set small by construction.
        let max_requests: usize = self.max_bakes_per_frame.clamp(1, 64) as usize;
        let max_in_flight: usize = max_requests * 2;
        if let Some(pool) = self.bake_pool.as_mut() {
            let cache = self.cache.as_ref();
            let mut requested = 0;
            for chunk in &cut {
                if pool.in_flight_len() >= max_in_flight || requested >= max_requests {
                    break;
                }
                let id = chunk.id;
                // Needs a surface if it is not resident yet, OR its surface was
                // baked from params that have since been edited. `stale_surfaces`
                // (unlike the cache's `dirty`) is only cleared when the FRESH bake
                // actually lands, so a chunk pushed past this frame's budget stays
                // queued for a re-bake instead of being silently starved.
                let resident = cache.map(|c| c.slot_of(id).is_some()).unwrap_or(false);
                let needs_surface = !resident || self.stale_surfaces.contains(&id);
                if needs_surface && !self.ready_surfaces.contains(&id) && !pool.in_flight(id) {
                    pool.request(&frames[chunk.id.face as usize], chunk);
                    requested += 1;
                }
            }
        }

        // CEL-86 async GDScript bake: take back whatever the builder finished
        // (from its threads / downloads), then hand it the chunks still missing.
        // Both are no-ops unless a `CpuCustomAsync` builder is active.
        self.drain_gd_submissions(&cut, tile_res);
        self.request_gd_bakes(&cut, tile_res);

        let tu = Instant::now();
        let diff = if self.recompute_every_frame {
            // TEST: no cache. Pack the visible cut into slots 0..N and mark ALL of
            // them for realize+bake every frame — only the visible chunks, never
            // persisted. Skip `cache.update` entirely so its state can't interfere.
            let n = (cut.len() as u32).min(budget) as usize;
            let realize: Vec<(u32, Chunk)> =
                cut.iter().take(n).cloned().enumerate().map(|(i, c)| (i as u32, c)).collect();
            let visible_slots: Vec<u32> = (0..n as u32).collect();
            let visible_cut_idx: Vec<Option<usize>> = (0..n).map(Some).collect();
            CacheDiff { realize, evicted: Vec::new(), visible_slots, visible_cut_idx }
        } else {
            // Throttle new-chunk admission so a big influx spreads its realize
            // over frames, and GATE admission on the worker-baked patch being
            // ready (CPU surface on) — unadmitted chunks stay covered by ancestor
            // stand-ins, so waiting for the bake is invisible.
            // A param reshade re-realizes EVERY dirty chunk in one frame
            // (ignoring the per-frame bake throttle) so the planet updates at
            // once; normal camera streaming keeps the throttle.
            let max_new = if self.force_full_reshade {
                self.force_full_reshade = false;
                budget
            } else {
                self.max_bakes_per_frame.clamp(1, i32::MAX as i64) as u32
            };
            // Admission waits on a ready surface for BOTH off-main-thread paths
            // (the Rust bake pool and the async GDScript builder); until then a
            // coarse ancestor stand-in covers the chunk.
            let surface_on = self.bake_pool.is_some() || self.gd_async_baker.is_some();
            let ready = &self.ready_surfaces;
            // Admission requires a surface requested AFTER the last param edit;
            // an older stamp is as good as no surface (the chunk keeps its
            // ancestor stand-in until the fresh bake lands).
            let epoch = self.param_epoch;
            self.cache.as_mut().unwrap().update_throttled_gated(&cut, max_new, &|id| {
                !surface_on || ready.get(&id).is_some_and(|(e, _)| *e == epoch)
            })
        };
        self.last_update_ms = tu.elapsed().as_secs_f64() * 1000.0;
        self.last_realize_count = diff.realize.len() as i64;
        self.last_visible_count = diff.visible_slots.len() as i64;
        self.last_cut_depths.clear();
        self.last_cut_depths.extend(cut.iter().map(|c| c.id.depth));
        self.last_cut_centroids.clear();
        self.last_cut_centroids
            .extend(cut.iter().map(|c| (c.corners[0] + c.corners[1] + c.corners[2]) / 3.0));

        // Morph factors aligned 1:1 with `diff.visible_slots` (the DRAWN set).
        // `visible_cut_idx[k] == Some(i)` maps drawn instance k to `cut[i]` (its
        // morph); `None` is a coarse ancestor stand-in covering not-yet-admitted
        // chunks — drawn at full detail (morph 1.0).
        let drawn_morphs: Vec<f32> = if !geomorph {
            vec![1.0; diff.visible_slots.len()]
        } else {
            diff.visible_cut_idx
                .iter()
                .map(|idx| idx.map(|i| morphs_cut[i]).unwrap_or(1.0))
                .collect()
        };
        debug_assert_eq!(drawn_morphs.len(), diff.visible_slots.len(), "morphs must align with drawn slots");

        let has_scatter = !self.scatter_mms.is_empty();

        // Stage when something changed: new/re-realized chunks, the visible set
        // (instances) moved, OR — with geomorph on — the camera moved with an
        // unchanged cut (the morph factors changed, so the small instance buffer
        // must be re-uploaded; realize/bake are skipped because `realize_count`
        // stays = new chunks only, preserving the cache). A stationary camera =>
        // identical cut + no move => no stage, no schedule.
        let cam_moved = self.last_cam.map_or(true, |p| (p - cam).length() > 1.0e-3);
        let slots_changed = diff.visible_slots != self.last_slots;
        let geomorph_restage = geomorph && cam_moved && !diff.visible_slots.is_empty();
        // CEL-73: a live density/height edit stages with realize_count = 0 —
        // only the scatter params change, so only scatter-compact does real work.
        let scatter_restage = has_scatter && self.scatter_dirty;
        if !diff.realize.is_empty() || slots_changed || geomorph_restage || scatter_restage {
            // Defensive budget clamp (the cache already bounds slots < budget).
            let realize: Vec<_> =
                diff.realize.iter().take(budget as usize).cloned().collect();
            let mut slots = diff.visible_slots.clone();
            slots.truncate(budget as usize);
            // `drawn_morphs` is already aligned 1:1 with `diff.visible_slots`.
            let mut morphs = drawn_morphs.clone();
            morphs.truncate(slots.len());

            let desc_bytes = pack_chunks(&frames, &realize, res);
            let instance_bytes = pack_instances(&slots, &morphs);

            // Pull each realized chunk's CPU-baked surface (colour/height/normal)
            // from the ready set into the stage. Admission was gated on the
            // surface being ready, so the hit is guaranteed; a defensive miss just
            // skips the upload (the slot keeps its previous surface). Missing-data
            // streaming is now handled inside `provider.bake`.
            let (surface_enabled, surface_height_scale) = if let Some(baker) = self.gd_baker.as_ref() {
                // GDScript baker: surface always on; optional `height_scale`
                // property (else 1.0 = the returned fraction displaces directly).
                let hs = baker.get("height_scale").try_to::<f32>().unwrap_or(1.0);
                (1.0f32, hs)
            } else if let Some(baker) = self.gd_async_baker.as_ref() {
                // Async GDScript baker: same `height_scale`, but the surface is
                // hidden (procedural fallback shows) until the builder's optional
                // `_base_ready` says its base data has landed.
                let hs = baker.get("height_scale").try_to::<f32>().unwrap_or(1.0);
                (if Self::gd_base_ready(baker) { 1.0 } else { 0.0 }, hs)
            } else {
                match self.provider.as_ref() {
                    Some(p) => (if p.base_ready() { 1.0 } else { 0.0 }, p.height_scale()),
                    None => (0.0f32, 0.0f32),
                }
            };
            let mut surface_patches = Vec::new();
            let tp = Instant::now();
            if let Some(mut baker) = self.gd_baker.clone() {
                // Bake each newly-realized chunk synchronously (realize is already
                // throttled per frame by admission), main thread.
                for (slot, chunk) in &realize {
                    let surface = Self::gd_bake_chunk(&mut baker, chunk, tile_res);
                    surface_patches.push((*slot, surface.color, surface.height, surface.normal));
                }
            } else if self.provider.is_some() || self.gd_async_baker.is_some() {
                for (slot, chunk) in &realize {
                    let Some((epoch, surface)) = self.ready_surfaces.remove(&chunk.id) else {
                        continue;
                    };
                    // Requested before the last param edit → reject at render
                    // time. The slot keeps its previous surface (or the ancestor
                    // stand-in) and the chunk is re-requested via the
                    // not-ready/stale paths with the current params.
                    if epoch != self.param_epoch {
                        continue;
                    }
                    surface_patches.push((*slot, surface.color, surface.height, surface.normal));
                }
            }
            self.last_patch_ms = tp.elapsed().as_secs_f64() * 1000.0;

            let ts = Instant::now();
            // CEL-73 scatter staging: aux paths for the realize batch, the
            // visible-slot gather list, and a fresh per-layer params snapshot
            // (density/height re-read every stage → live sliders).
            let (scatter_aux_bytes, scatter_vis_bytes, scatter_vis_count, scatter_layer_params) =
                if has_scatter {
                    // Scatter placement displaces instances with the SAME noise +
                    // params as ChunkRealize, so it must read the builder's terrain
                    // params (not HeightGpu::default) — otherwise instances sit on a
                    // different surface than the realized ground and float. Re-read
                    // every stage so a builder edit (re-realize → re-place) tracks.
                    let terrain = self.terrain_params();
                    // ACTIVE layers only (mesh assigned) — parallel to the
                    // GPU layer list built in `build_scatter_children`.
                    let params: Vec<Vec<u8>> = self
                        .scatter_layers
                        .iter_shared()
                        .filter(|l| l.bind().mesh.is_some())
                        .map(|layer| {
                            let l = layer.bind();
                            let k = l.instances_per_cell.clamp(1, 64) as u32;
                            // Disabled layer: density 0 fails every hash01
                            // gate — zero instances, cached placement kept.
                            let density =
                                if l.enabled { l.density.clamp(0.0, 1.0) } else { 0.0 };
                            pack_scatter_params(
                                realize.len() as u32,
                                celestial_algo::scatter::capacity(k),
                                k,
                                l.lod_level.clamp(0, 20) as u32,
                                self.radius,
                                density,
                                l.max_instances.clamp(64, 4_000_000) as u32,
                                l.seed as u32,
                                slots.len() as u32,
                                l.min_height.clamp(0.0, 1.0),
                                l.max_height.clamp(0.0, 1.0),
                                l.scale.max(0.0),
                                // CPU-surface route: place must sample the SAME
                                // baked heightmap realize displaced with, or the
                                // instances sit on the procedural noise instead
                                // (floating/buried, wrong height gates).
                                surface_enabled,
                                surface_height_scale,
                                tile_res,
                                &terrain,
                            )
                        })
                        .collect();
                    (pack_scatter_aux(&realize), pack_scatter_vis(&slots), slots.len() as u32, params)
                } else {
                    (Vec::new(), Vec::new(), 0, Vec::new())
                };

            if let Some(job) = &mut self.job {
                let new_stage = ChunkStage {
                    desc_bytes,
                    realize_count: realize.len() as u32,
                    instance_bytes,
                    instance_count: slots.len() as u32,
                    surface_enabled,
                    surface_height_scale,
                    surface_patches,
                    scatter_aux_bytes,
                    scatter_vis_bytes,
                    scatter_vis_count,
                    scatter_layer_params,
                };
                // MERGE into any still-pending batch — plain overwrite dropped
                // the unconsumed realizes, leaving their (cache-clean, drawn)
                // slots holding uninitialized pool memory forever.
                let mut job = job.bind_mut();
                match &mut job.stage {
                    Some(pending) => {
                        crate::chunk_pipeline::merge_stage(pending, new_stage, budget as usize)
                    }
                    slot @ None => *slot = Some(new_stage),
                }
            }
            self.last_slots = diff.visible_slots.clone();
            self.scatter_dirty = false;
            self.last_stage_ms = ts.elapsed().as_secs_f64() * 1000.0;
        }
        self.last_cam = Some(cam);

        // Camera-driven cancellation: tell both worker pools what is still in
        // view so queued work for flown-past terrain is skipped, and drop the
        // matching main-thread bookkeeping. Everything released here is simply
        // re-requested if the player comes back.
        if self.bake_pool.is_some()
            || !self.ready_surfaces.is_empty()
            || !self.gd_outstanding.is_empty()
        {
            let cut_ids: HashSet<ChunkId> = cut.iter().map(|c| c.id).collect();
            // Baked surfaces whose chunk left the cut before admission.
            self.ready_surfaces.retain(|id, _| cut_ids.contains(id));
            // NOTE: deliberately NOT retained against `cut_ids`. A chunk can leave
            // the cut while staying RESIDENT in the LRU cache; dropping its stale
            // mark here would mean that when it re-enters view it is resident and
            // "not stale", so it is never re-requested and keeps its old-param
            // surface forever — which is exactly what made panning around during a
            // slider drag leave old chunks behind. Staleness ends only when the
            // fresh surface lands, or when the chunk is EVICTED (below) — an
            // evicted chunk re-bakes anyway via the not-resident path.
            if let Some(c) = self.cache.as_ref() {
                self.stale_surfaces.retain(|id| c.slot_of(*id).is_some());
            }
            // Ditto for chunks the async GDScript builder still owes us: forget
            // them, so a return trip re-requests them. A submission that lands
            // anyway is dropped for the same reason (not in the cut).
            self.gd_outstanding.retain(|id| cut_ids.contains(id));
            if let Some(pool) = &self.bake_pool {
                pool.set_wanted(cut_ids.clone());
            }
            // The provider prunes its own streaming bookkeeping to the cut.
            if let Some(p) = self.provider.as_ref() {
                p.set_wanted(&cut_ids);
            }
        }

        // Perf forensics (CELESTIAL_PERF=1): name the slow section of any frame
        // whose planet-side work exceeded ~15 ms.
        if std::env::var_os("CELESTIAL_PERF").is_some() {
            let total = t0.elapsed().as_secs_f64() * 1000.0;
            if total > 15.0 {
                eprintln!(
                    "[perf] frame {total:.1} ms | select {:.1} | cache {:.1} | patches {:.1} ms ({} baked) | stage {:.1}",
                    self.last_select_ms,
                    self.last_update_ms,
                    self.last_patch_ms,
                    self.last_realize_count,
                    self.last_stage_ms,
                );
            }
        }

        self.pump();
        self.wire_textures();
        self.update_water();

        self.status_accum += delta;
        if self.debug_log && self.status_accum >= 5.0 {
            self.status_accum = 0.0;
            let (resident, executes) = self
                .job
                .as_ref()
                .map(|j| (self.cache.as_ref().unwrap().resident_count(), j.bind().executes))
                .unwrap_or((0, 0));
            godot_print!(
                "[Celestial] chunks {} | resident {resident} | realize {} | executes {executes} | select {:.0} us",
                cut.len(),
                self.last_realize_count,
                self.last_select_ms * 1000.0,
            );
        }
    }
}

#[godot_api]
impl Celestial {
    /// A builder's `changed` signal lands here. A STRUCTURAL change — which
    /// builder is first-enabled, its [`BuilderRoute`], or a `GpuCustom` shader
    /// source — rebuilds the job (so the pipeline / compiled shader / provider is
    /// re-wired). A pure param edit is applied live, by kind, then every resident
    /// chunk is invalidated so it re-realizes with the new values.
    #[func]
    fn get_builder(&self) -> Option<Gd<CesBuilder>> {
        self.builder.clone()
    }

    /// Swap the terrain builder (assign a different one, or clear it → white).
    /// Rebuilds the job so the new type/params take effect immediately; the next
    /// frame re-wires the new builder's `changed` signal (via `connect_builders`).
    #[func]
    fn set_builder(&mut self, v: Option<Gd<CesBuilder>>) {
        self.builder = v;
        // Rebuild from scratch (kind/provider/compiled shader may all differ).
        self.teardown_job();
        self.built_builder_kind = None;
        self.built_custom_source = None;
    }

    #[func]
    fn on_builder_changed(&mut self) {
        let new_kind = self.active_builder().map(|b| crate::builder::route_of(&b));
        let new_custom = self.build_custom_surface();
        let new_src = new_custom.as_ref().map(|(s, ..)| s.clone());

        // Structural change → rebuild (ensure_job re-wires next frame).
        if new_kind != self.built_builder_kind || new_src != self.built_custom_source {
            self.teardown_job();
            return;
        }

        // Live param edit — don't do the heavy work here. Flag it and let
        // `process` apply it once, when no realize is in flight (so a slider
        // drag firing many `changed` signals collapses to a single reshade at
        // the latest value instead of stacking a re-realize per signal).
        self.note_param_update();
    }

    /// Record "the parameters just changed" — the LAST-UPDATE timestamp every
    /// in-flight surface request is judged against. Called the INSTANT an edit
    /// is detected, unlike `apply_terrain_reshade` (which is coalesced and can
    /// lag by frames while a realize stage is in flight — a window in which
    /// old-param bakes used to arrive, get accepted and rendered).
    ///
    /// From this moment on:
    /// - results for bakes REQUESTED before now are rejected on arrival
    ///   ([`BakePool::poll`] compares each result's request generation);
    /// - surfaces ALREADY delivered are dropped here, and — belt and braces —
    ///   anything that slips through is rejected again at render time by its
    ///   `param_epoch` stamp.
    ///
    /// For the CPU-noise route the fresh provider is also installed
    /// immediately, so every re-request from this frame on bakes with the NEW
    /// params (deferring the swap to the coalesced reshade would stamp
    /// old-param bakes as fresh).
    fn note_param_update(&mut self) {
        self.param_epoch += 1;
        self.ready_surfaces.clear();
        if self.bake_pool.is_some() {
            if let Some(provider) = self.build_provider() {
                if let Some(pool) = self.bake_pool.as_mut() {
                    pool.set_provider(Arc::clone(&provider));
                }
                self.provider = Some(provider);
            }
        }
        self.terrain_reshade_pending = true;
    }

    /// Apply a pending live builder edit: push the new params into the running
    /// job by route, invalidate every resident chunk, and request a throttle-free
    /// re-realize so the whole planet updates in ONE frame. Called from `process`
    /// only when nothing is in flight (see `terrain_reshade_pending`).
    fn apply_terrain_reshade(&mut self) {
        match self.active_builder().map(|b| crate::builder::route_of(&b)) {
            Some(BuilderRoute::GpuCustom) => {
                if let Some((_, w, hs, vals)) = self.build_custom_surface() {
                    if let Some(job) = self.job.as_mut() {
                        job.bind_mut().gpu.set_custom_knobs(w, hs, vals);
                    }
                }
            }
            Some(BuilderRoute::CpuNoise) => {
                // Rebuild the provider with the edited params and swap it into the
                // running bake pool (no thread respawn); drop stale baked surfaces.
                if let Some(provider) = self.build_provider() {
                    // `set_provider` opens a new generation and releases in-flight
                    // bakes, so bakes started under the OLD params can neither be
                    // applied nor block their chunk from being re-baked.
                    if let Some(pool) = self.bake_pool.as_mut() {
                        pool.set_provider(Arc::clone(&provider));
                    }
                    self.provider = Some(provider);
                }
                self.ready_surfaces.clear();
                self.was_base_ready = false;
            }
            Some(BuilderRoute::GpuNoise) => {
                let terrain = self.terrain_params();
                if let Some(job) = self.job.as_mut() {
                    job.bind_mut().terrain = terrain;
                }
            }
            Some(BuilderRoute::CpuCustomAsync) => {
                // Every surface in flight was baked with the OLD params: drop the
                // queued submissions and the ready set, and forget what we asked
                // for, so the invalidate below re-requests the whole cut.
                if let Some(q) = self.gd_submits.as_ref() {
                    q.clear();
                }
                self.gd_outstanding.clear();
                self.ready_surfaces.clear();
            }
            // CpuCustom (GDScript funcs re-run per re-bake) and None (white) need
            // nothing beyond the invalidate below.
            _ => {}
        }

        if let Some(cache) = self.cache.as_mut() {
            // Old-param terrain must never reappear: every OFF-SCREEN resident
            // holds a surface baked with the old params, and a revisit would
            // draw it for the frames its re-bake takes. Evict them all — a
            // revisited zone then streams in exactly like a first visit (fresh
            // ancestor stand-in refining), never showing stale data. On-screen
            // chunks stay resident and re-bake in place (no LOD flash).
            cache.evict_offscreen();
            let resident = cache.invalidate_all();
            // Every resident chunk's baked surface was made with the OLD params.
            // Remember them ALL: the cache's `dirty` flag is cleared by the very
            // next `update` (which re-realizes them against those stale surfaces),
            // so it cannot survive long enough to drive a budgeted re-bake.
            if self.bake_pool.is_some() {
                self.stale_surfaces = resident.into_iter().map(|(_, id)| id).collect();
            }
        }
        self.force_full_reshade = true;
        self.last_slots.clear();
    }

    /// Detect live edits to the active builder's `@export` FLOAT knobs/params by
    /// fingerprinting their values each frame. Godot does NOT emit a resource's
    /// `changed` signal for a plain `@export` edit, so — instead of requiring a
    /// `set(v): x = v; emit_changed()` on every param — we poll and flag a
    /// reshade when the fingerprint changes. Coalesced like `on_builder_changed`
    /// (via `terrain_reshade_pending`), so a slider drag is one reshade/frame.
    /// Base fields (`device`/`shader_file`/`builtin_shader`) are
    /// native, not script vars, so the `SCRIPT_VARIABLE` filter skips them — and
    /// they carry their own `changed` (structural rebuild) via native setters.
    fn poll_builder_params(&mut self) {
        let Some(b) = self.active_builder() else {
            self.last_param_values = None;
            return;
        };
        let obj = b.upcast::<Object>();
        let script_var = PropertyUsageFlags::SCRIPT_VARIABLE.ord() as i64;
        let float_ty = VariantType::FLOAT.ord as i64;
        // NOTE: bind the property list to a variable — iterating the temporary
        // `obj.get_property_list().iter_shared()` yields NOTHING (the Array is
        // dropped before iteration), which silently returned an empty fingerprint.
        let plist = obj.get_property_list();
        let mut vals: Vec<f32> = Vec::new();
        for entry in plist.iter_shared() {
            let usage = entry.get("usage").and_then(|v| v.try_to::<i64>().ok()).unwrap_or(0);
            if usage & script_var == 0 {
                continue;
            }
            let vtype = entry.get("type").and_then(|v| v.try_to::<i64>().ok()).unwrap_or(0);
            if vtype != float_ty {
                continue;
            }
            // The property-list "name" is a String variant, NOT StringName —
            // read it as GString (try_to::<StringName> fails / to() panics on it).
            if let Some(name) = entry.get("name").and_then(|v| v.try_to::<GString>().ok()) {
                let name = name.to_string();
                vals.push(obj.get(&name).try_to::<f32>().unwrap_or(0.0));
            }
        }
        match &self.last_param_values {
            // First observation (or just after a rebuild): baseline, don't reshade.
            None => self.last_param_values = Some(vals),
            Some(prev) if *prev != vals => {
                self.last_param_values = Some(vals);
                self.note_param_update();
            }
            _ => {}
        }
    }

    /// EDITOR ONLY: watch a `GpuCustom` builder's `.glsl` for external edits and
    /// rebuild (recompile) when its modified time changes — so saving the shader
    /// updates the viewport without re-assigning the builder. Polled every ~15
    /// frames; `teardown_job` resets the baseline so a rebuild re-baselines.
    fn poll_shader_reload(&mut self) {
        if !godot::classes::Engine::singleton().is_editor_hint() {
            return;
        }
        self.shader_poll_ticks = self.shader_poll_ticks.wrapping_add(1);
        if self.shader_poll_ticks % 15 != 0 {
            return;
        }
        let Some(b) = self.active_builder() else { return };
        if crate::builder::route_of(&b) != BuilderRoute::GpuCustom {
            self.shader_mtime = 0;
            return;
        }
        let path = b.bind().shader_file.clone();
        if path.is_empty() {
            self.shader_mtime = 0;
            return;
        }
        let mtime = godot::classes::FileAccess::get_modified_time(&path);
        if self.shader_mtime == 0 {
            self.shader_mtime = mtime; // first sighting → baseline, no rebuild
        } else if mtime != self.shader_mtime {
            // Source changed on disk → full rebuild recompiles the shader; force
            // an un-throttled re-realize so the edit shows at once. `teardown_job`
            // clears `shader_mtime`, so the next poll re-baselines to `mtime`.
            self.teardown_job();
            self.force_full_reshade = true;
        }
    }

    /// Chunks currently streaming toward full detail: queued/running background
    /// bakes plus baked patches waiting for cache admission. `0` ⇒ every chunk
    /// in view is resident at its target LOD (nothing left to arrive).
    #[func]
    fn chunks_in_queue(&self) -> i64 {
        let baking = self.bake_pool.as_ref().map(|p| p.in_flight_len()).unwrap_or(0);
        (baking + self.ready_surfaces.len()) as i64
    }

    /// Absolute path of the provider's on-disk cache (empty when none).
    #[func]
    fn tile_cache_path(&self) -> GString {
        match self.provider.as_ref().and_then(|p| p.cache_dir()) {
            Some(d) => GString::from(d.as_str()),
            None => GString::new(),
        }
    }

    /// Provider resource fetches currently in flight (network + decode).
    #[func]
    fn tiles_in_flight(&self) -> i64 {
        self.provider.as_ref().map(|p| p.resources_in_flight() as i64).unwrap_or(0)
    }

    /// Division-algorithm debug: per-depth histogram of the last selected cut,
    /// plus how many of those chunks are actually DRAWN (resident) — any gap is
    /// a hole on screen (unadmitted/throttled chunks). Example output:
    /// `cut 271 drawn 268 | d7:12 d8:24 ... d16:40`.
    #[func]
    fn cut_report(&self) -> GString {
        let mut hist = [0u32; 24];
        for &d in &self.last_cut_depths {
            hist[(d as usize).min(23)] += 1;
        }
        let parts: Vec<String> = hist
            .iter()
            .enumerate()
            .filter(|(_, &c)| c > 0)
            .map(|(d, &c)| format!("d{d}:{c}"))
            .collect();
        GString::from(
            format!(
                "cut {} drawn {} | {}",
                self.last_cut_depths.len(),
                self.last_visible_count,
                parts.join(" ")
            )
            .as_str(),
        )
    }

    /// Radius (world units, from the planet centre) of the RENDERED ground in
    /// direction `dir`: the bare sphere plus the CPU-surface displacement,
    /// sampled by the provider's `sample_height`. Use this to place
    /// cameras/objects on the surface instead of guessing an altitude — the
    /// ground can legitimately sit kilometres above the sphere. Returns the bare
    /// radius when no CPU-surface provider is active.
    #[func]
    fn ground_radius_at(&self, dir: Vector3) -> f32 {
        let h = match self.provider.as_ref() {
            // No clamp: `ChunkRealize` displaces by the raw sampled height (the
            // seabed dips BELOW the sphere), so clamping here would report a
            // ground radius the renderer never draws.
            Some(p) => p.sample_height(dir).unwrap_or(0.0) * p.height_scale(),
            None => 0.0,
        };
        self.radius * (1.0 + h)
    }

    /// TEMP DEBUG: schedule a render-thread dump of every drawn slot's vertex
    /// radius range (see `CesChunkJob::debug_dump_radii`).
    #[func]
    fn debug_dump_radii(&self) {
        let Some(job) = &self.job else { return };
        // EXPECTED: the DEEPEST chunks (the near field — where the breakage
        // is). Print their cut index / slot / location, and have the GPU dump
        // report the same slots' actual geometry.
        let mut deep_slots: Vec<u32> = Vec::new();
        for (i, (&d, c)) in self
            .last_cut_depths
            .iter()
            .zip(self.last_cut_centroids.iter())
            .enumerate()
        {
            if d < 13 {
                continue;
            }
            let cn = c.normalized();
            let lat = (cn.y.clamp(-1.0, 1.0)).asin().to_degrees();
            let lon = cn.z.atan2(cn.x).to_degrees();
            let slot = self.last_slots.get(i).copied().unwrap_or(u32::MAX);
            godot_print!(
                "[radii] CPU cut[{i}] d{d} slot {slot}: lat {lat:.2} lon {lon:.2}"
            );
            if slot != u32::MAX {
                deep_slots.push(slot);
            }
        }
        let slots = PackedInt32Array::from_iter(deep_slots.iter().map(|&s| s as i32));
        let cb = Callable::from_object_method(job, "debug_dump_radii").bind(&[slots.to_variant()]);
        RenderingServer::singleton().call_on_render_thread(&cb);
    }

    /// Any scatter layer's `changed` signal lands here (CEL-73); the edit is
    /// classified next frame in `check_scatter_structure` / the stage packer.
    #[func]
    fn on_scatter_layer_changed(&mut self) {
        self.scatter_dirty = true;
    }

    #[func]
    fn resident_count(&self) -> i64 {
        self.cache.as_ref().map(|c| c.resident_count() as i64).unwrap_or(0)
    }

    /// Chunks realized in the last staged batch (0 once the camera settles).
    #[func]
    fn realize_count(&self) -> i64 {
        self.last_realize_count
    }

    /// Graph executes that actually consumed a stage (flat when stationary).
    #[func]
    fn executes(&self) -> i64 {
        self.job.as_ref().map(|j| j.bind().executes as i64).unwrap_or(0)
    }

    /// Last CPU selection time in milliseconds.
    #[func]
    fn select_ms(&self) -> f64 {
        self.last_select_ms
    }

    /// Last `ChunkCache::update` time in ms (eviction/bookkeeping cost).
    #[func]
    fn update_ms(&self) -> f64 {
        self.last_update_ms
    }

    /// Per-stage GPU time of the last `CesChunkJob` execute, e.g.
    /// `"upload 0.01 + realize 0.42 ms"`. Each graph node (compute stage) is a
    /// term — a future pass (e.g. a separate normals/noise shader) appears here
    /// automatically. `"idle"` when no realize ran since the camera last moved;
    /// this is the realize COMPUTE cost, distinct from the every-frame render
    /// (rasterization) cost shown as `render gpu`.
    #[func]
    fn gpu_report(&self) -> GString {
        let Some(job) = &self.job else { return GString::from("n/a") };
        let gpu_ms = &job.bind().gpu_ms;
        if gpu_ms.is_empty() {
            return GString::from("idle");
        }
        let parts: Vec<String> = gpu_ms
            .iter()
            .map(|(name, ms)| {
                let short = name.strip_prefix("celestial/chunk-").unwrap_or(name);
                format!("{short} {ms:.2}")
            })
            .collect();
        GString::from(format!("{} ms", parts.join(" + ")).as_str())
    }

    /// Triangles currently drawn = visible chunks × `chunk_res²`.
    #[func]
    fn triangle_count(&self) -> i64 {
        let res = self.chunk_res.clamp(2, 32);
        self.last_visible_count * res * res
    }

    /// VRAM (bytes) the GPU pools reserve for `budget` slots. Per-vertex the
    /// pool costs pos_tex(rgba32f=16) + verts_tex(2×rgba16f=16) + verts_buf
    /// (float4=16) = 48 B × `verts_per_chunk(res)`. Phase 4 adds the detail
    /// atlases: colour(rgba8=4) + normal(rgba8=4) = 8 B × `tile_res²` per slot.
    #[func]
    fn pool_vram_bytes(&self) -> i64 {
        self.effective_budget() as i64 * self.per_slot_bytes()
    }

    /// Resident chunk SLOT count derived from the VRAM budget. Each slot costs
    /// `verts_per_chunk(chunk_res)×48 B` (geometry) + `tile_res²×16 B` (detail atlases +
    /// normal atlas); `slots = vram_budget / per_slot`. Also clamped so the atlas
    /// texture height (`slots×tile_res²/ATTR_TEX_WIDTH`) stays within the GPU's max
    /// texture dimension (the atlas is one strip of width `ATTR_TEX_WIDTH`).
    #[func]
    fn effective_budget(&self) -> u32 {
        let tile_res = self.tile_res.clamp(8, 1024) as i64;
        let vram = (self.vram_budget_gib.max(0.01) as f64 * 1024.0 * 1024.0 * 1024.0) as i64;
        let from_vram = (vram / self.per_slot_bytes()).max(1);
        // Atlas height = slots × tile_res² / width must fit the GPU texture limit.
        let height_cap =
            (MAX_ATLAS_TEX_HEIGHT * ATTR_TEX_WIDTH as i64 / (tile_res * tile_res)).max(1);
        from_vram.min(height_cap) as u32
    }

    /// VRAM one resident chunk slot reserves: geometry pool + detail atlases +
    /// the CPU-surface colour/height buffers (see
    /// [`crate::chunk_descriptors::per_slot_bytes`]).
    fn per_slot_bytes(&self) -> i64 {
        let res = self.chunk_res.clamp(2, 32) as u32;
        let tile_res = self.tile_res.clamp(8, 1024) as u32;
        crate::chunk_descriptors::per_slot_bytes(res, tile_res)
    }
}

/// Structural snapshot of one scatter layer (CEL-73), used to classify edits:
/// capacity-affecting fields rebuild the job, lod_level/seed/scale re-place
/// resident chunks (place-side transform), a mesh swap rebinds in place.
/// density/min_height/max_height are compact-side (no snapshot needed).
struct ScatterSnapshot {
    layer_id: i64,
    lod_level: i64,
    instances_per_cell: i64,
    max_instances: i64,
    seed: i64,
    scale: f32,
    mesh_rid: Rid,
}

/// Max atlas-strip texture height (texels). The detail atlas is a single texture
/// of width `ATTR_TEX_WIDTH`; its height `slots×tile_res²/width` must stay within
/// this so allocation can't exceed the GPU's max texture dimension.
const MAX_ATLAS_TEX_HEIGHT: i64 = 16384;

impl Celestial {
    /// The active builder (drives the terrain), or `None` (→ white planet) when
    /// unset.
    fn active_builder(&self) -> Option<Gd<CesBuilder>> {
        self.builder.clone()
    }

    /// The terrain params for the inline example shader. ONLY the built-in
    /// Noise modes drive the base terrain with the noise knobs; the Custom modes
    /// (and no builder) return a DISABLED terrain (`enabled = 0`) so the base is
    /// a plain sphere. That way a Custom builder whose surface isn't active yet
    /// (no `shader_file`, a shader error, or a chunk not baked) shows a white
    /// sphere — never the example noise leaking through.
    fn terrain_params(&self) -> crate::descriptors::TerrainGpu {
        let noise_builder = self.active_builder().filter(|b| {
            matches!(
                crate::builder::route_of(b),
                BuilderRoute::GpuNoise | BuilderRoute::CpuNoise
            )
        });
        if let Some(b) = noise_builder {
            let l = b.bind();
            assemble(&l.to_height_gpu(), &l.to_texture_gpu())
        } else {
            let h = HeightGpu { enabled: 0.0, ..HeightGpu::default() };
            assemble(&h, &TextureGpu::default())
        }
    }

    /// Idempotently connect the builder's `changed` signal to the reshade /
    /// rebuild handler (skips if already wired), so editor edits are live.
    fn connect_builders(&mut self) {
        let Some(builder) = self.builder.clone() else { return };
        let self_gd = self.to_gd();
        let callable = Callable::from_object_method(&self_gd, "on_builder_changed");
        let mut res = builder.upcast::<Resource>();
        if !res.is_connected("changed", &callable) {
            res.connect("changed", &callable);
        }
    }

    /// Lazily build the MultiMesh + instance + chunk material + render-thread job.
    fn ensure_job(&mut self) {
        if self.job.is_some() {
            return;
        }
        let res = self.chunk_res.clamp(2, 32) as u32;
        let tile_res = self.tile_res.clamp(8, 1024) as u32;
        let budget = self.effective_budget();
        godot_print!(
            "[Celestial] vram budget {:.2} GiB | chunk_res {res} tile_res {tile_res} | {budget} resident chunk slots (~{} MB pools, {} KB/slot)",
            self.vram_budget_gib,
            self.pool_vram_bytes() / 1_048_576,
            self.per_slot_bytes() / 1024,
        );

        // Material + reference chunk mesh + indirect MultiMesh (CEL-58 order:
        // allocate indirect BEFORE set_mesh so the command buffer is created).
        let mut material = make_material(res, tile_res, self.lod_colors);
        let template = reference_chunk_mesh(res, &material.clone().upcast());

        let mut rs = RenderingServer::singleton();
        let multimesh = MultiMesh::new_gd();
        let mm_rid = multimesh.get_rid();
        let mut mmi = MultiMeshInstance3D::new_alloc();
        mmi.set_multimesh(&multimesh);
        self.base_mut().add_child(&mmi);

        rs.multimesh_allocate_data_ex(mm_rid, budget as i32, MultimeshTransformFormat::TRANSFORM_3D)
            .custom_data_format(true)
            .use_indirect(true)
            .done();
        rs.multimesh_set_mesh(mm_rid, template.get_rid());
        // Transforms are GPU-only (identity) and VERTEX comes from pos_tex, so
        // give an explicit AABB covering the displaced sphere envelope.
        let m = self.radius * 1.3;
        rs.multimesh_set_custom_aabb(
            mm_rid,
            Aabb { position: Vector3::splat(-m), size: Vector3::splat(2.0 * m) },
        );

        // CEL-73: one indirect MultiMesh child per scatter layer (same CEL-58
        // allocate-before-set_mesh order as the terrain multimesh above).
        let scatter_cfgs = self.build_scatter_children(&mut rs, m);

        // Remember which builder kind this job is wired for (rebuild on a flip).
        self.built_builder_kind = self.active_builder().map(|b| crate::builder::route_of(&b));

        // Terrain from the first enabled builder (disabled → white when none).
        let terrain = self.terrain_params();
        // Detail-normal bump removed (it scattered lit speckles across the
        // surface / lakes); pass 0.0 so the baked normal is the smooth FD normal.
        let mut job = CesChunkJob::create(
            mm_rid,
            budget,
            res,
            tile_res,
            self.radius,
            0.0,
            terrain,
            scatter_cfgs,
        );

        // Custom GPU surface (user GLSL) takes precedence over the CPU provider:
        // install it into the job (compiled lazily on the render thread) BEFORE
        // the first run, and remember its source so a later shader-file edit can
        // be told apart from a live knob edit.
        let custom_surface = self.build_custom_surface();
        if let Some((src, w, hs, vals)) = &custom_surface {
            job.bind_mut().gpu.set_custom_surface(src.clone(), *w, *hs, vals.clone());
        }
        self.built_custom_source = custom_surface.as_ref().map(|(s, ..)| s.clone());

        // Secondary GDScript CPU bakers — only when no GPU custom surface owns
        // the surface buffers. `gd_baker` bakes synchronously at staging;
        // `gd_async_baker` is handed chunks and submits them back whenever it
        // likes. `route_of` picks between them, so they are never both set.
        if custom_surface.is_none() {
            self.gd_baker = self.find_gd_baker();
            self.gd_async_baker = self.find_gd_async_baker();
        } else {
            self.gd_baker = None;
            self.gd_async_baker = None;
        }
        self.gd_submits = self.gd_async_baker.as_ref().map(|b| b.bind().submits());
        self.gd_outstanding.clear();

        // A script defining BOTH contracts gets the async one; say so rather than
        // silently ignoring the `height` the author clearly wrote.
        if let Some(b) = self.gd_async_baker.as_ref() {
            if b.clone().upcast::<Object>().has_method("height") {
                godot_warn!(
                    "[Celestial] builder defines both `_bake_requested` and \
                     `height`; using the async `_bake_requested` path and ignoring `height`."
                );
            }
        }

        self.run_cb = Some(Callable::from_object_method(&job, "run"));

        // Avoid an unused-variable warning while keeping the material alive.
        material.set_shader_parameter("attr_w", &(ATTR_TEX_WIDTH as i32).to_variant());

        self.cache = Some(celestial_algo::chunk_cache::ChunkCache::new(budget));
        // CEL-91: the CPU surface cache gets the SAME budget as the GPU slot pool,
        // so `vram_budget_gib` bounds BOTH sides of the memory.
        self.ready_surfaces.set_capacity(budget as usize);
        self.job = Some(job);
        self.multimesh =
            Some(crate::gpu::owned::IndirectMultiMesh::new(multimesh, crate::gpu::owned::MainDeviceSink::new()));
        self.mmi = Some(mmi);
        self.material = Some(material);
        self._template = Some(template);
        self.last_slots.clear();
        self.last_cam = None;

        // Select the CPU-surface provider from the layers (the first CpuNoise
        // mesh layer, else none). When present, spin up the shared provider +
        // off-thread bake pool; absent ⇒ GPU procedural path. A custom GPU
        // surface owns the surface buffers itself, so skip the CPU provider then.
        // Likewise skip when either GDScript CPU baker is active (they own the
        // surface buffers themselves).
        if custom_surface.is_none() && self.gd_baker.is_none() && self.gd_async_baker.is_none() {
        if let Some(provider) = self.build_provider() {
            // A surface bake costs several ms and CPU generation is the
            // bottleneck, so scale workers with the machine: half the logical
            // cores (leaving the other half for the render/main threads and the
            // tile fetchers), floor of 2. Off-thread baking keeps fast flight
            // smooth (admission waits on readiness).
            let workers = std::thread::available_parallelism()
                .map(|n| (n.get() / 2).max(2))
                .unwrap_or(3);
            // The result channel is bounded by the same in-flight ceiling the
            // request loop uses (2 frames' worth of admissions), so finished-but-
            // undrained surfaces can never pile up on the heap (CEL-91).
            let result_cap = (self.max_bakes_per_frame.clamp(1, 64) as usize) * 2;
            self.bake_pool =
                Some(BakePool::new(Arc::clone(&provider), tile_res, workers, result_cap));
            self.provider = Some(provider);
        }
        }

        // Baseline for detecting later param edits (see `reapply_param_changes`).
        self.built_radius = self.radius;
        self.built_res = self.chunk_res;
        self.built_tile_res = self.tile_res;
        self.built_budget_gib = self.vram_budget_gib;
    }

    /// A CPU-surface provider for the first enabled builder, or `None`. Only a
    /// [`BuilderRoute::CpuNoise`] builder yields one (the built-in
    /// [`NoiseProvider`], baked on the worker pool); every other kind returns
    /// `None` (their surface comes from a different path).
    fn build_provider(&self) -> Option<Arc<dyn CpuSurfaceProvider>> {
        let b = self.active_builder()?;
        if crate::builder::route_of(&b) == BuilderRoute::CpuNoise {
            let l = b.bind();
            Some(Arc::new(crate::noise_provider::NoiseProvider::new(l.to_noise_params(self.radius))))
        } else {
            None
        }
    }

    /// The custom **GPU** surface to install when the first enabled builder is
    /// [`BuilderRoute::GpuCustom`]:
    /// `(assembled_glsl, water_height, height_scale, user_param_values)`, or
    /// `None`. Reads the builder's `res://` `.glsl`, enumerates its own
    /// `@export var name: float` script vars (surfaced to the shader as
    /// `#define NAME`), and splices both into the library template
    /// ([`crate::custom_surface::assemble_source_with_params`]); a missing file
    /// or a source that omits a required function is logged and treated as `None`
    /// (→ white) so a typo never crashes the planet.
    fn build_custom_surface(&self) -> Option<(String, f32, f32, Vec<f32>)> {
        let b = self.active_builder()?;
        if crate::builder::route_of(&b) != BuilderRoute::GpuCustom {
            return None;
        }
        let l = b.bind();
        if l.shader_file.is_empty() {
            return None;
        }
        let path = l.shader_file.clone();
        let water = l.water_height();
        let hs = l.height_scale();
        drop(l);
        let Some(file) = godot::classes::FileAccess::open(&path, ModeFlags::READ) else {
            godot_error!("[Celestial] custom surface: cannot open shader file {path}");
            return None;
        };
        let user_glsl = file.get_as_text().to_string();

        // Enumerate the builder's OWN @export float script vars (its params),
        // in declaration order, excluding the base builder fields. Each becomes
        // a `#define UPPERCASE_NAME` in the assembled shader (value packed into
        // the params UBO's cels_user tail in the same order).
        let excluded = ["shader_file", "device", "builtin_shader"];
        let mut names: Vec<String> = Vec::new();
        let mut values: Vec<f32> = Vec::new();
        let obj = b.clone().upcast::<Object>();
        let script_var = PropertyUsageFlags::SCRIPT_VARIABLE.ord() as i64;
        // Bind the list to a variable — iterating the temporary yields nothing.
        let plist = obj.get_property_list();
        for entry in plist.iter_shared() {
            // Parse property-list dict values with `try_to` (NOT `to`, which
            // panics on an unexpected variant type — hit in practice).
            let usage = entry.get("usage").and_then(|v| v.try_to::<i64>().ok()).unwrap_or(0);
            if usage & script_var == 0 {
                continue;
            }
            let vtype = entry.get("type").and_then(|v| v.try_to::<i64>().ok()).unwrap_or(0);
            if vtype != VariantType::FLOAT.ord as i64 {
                continue;
            }
            let name = entry
                .get("name")
                .and_then(|v| v.try_to::<GString>().ok())
                .map(|g| g.to_string())
                .unwrap_or_default();
            if name.is_empty() || excluded.contains(&name.as_str()) {
                continue;
            }
            let val = obj.get(&name).try_to::<f32>().unwrap_or(0.0);
            names.push(name);
            values.push(val);
        }

        match crate::custom_surface::assemble_source_with_params(&user_glsl, &names) {
            Ok(src) => Some((src, water, hs, values)),
            Err(e) => {
                godot_error!("[Celestial] custom surface ({path}): {e}");
                None
            }
        }
    }

    // ---- CEL-86: the async GDScript bake ------------------------------------

    /// This planet's handle tag: distinct planets sharing one builder `.tres`
    /// stamp different tags, so a submission can never be applied to the wrong
    /// planet. Derived from the (unique, stable) Godot instance id.
    fn planet_tag(&self) -> u16 {
        (self.to_gd().instance_id().to_i64() as u64 & TAG_MASK) as u16
    }

    /// The active `CpuCustomAsync` builder, if the first enabled builder is one.
    fn find_gd_async_baker(&self) -> Option<Gd<CesBuilder>> {
        let b = self.active_builder()?;
        (crate::builder::route_of(&b) == BuilderRoute::CpuCustomAsync).then_some(b)
    }

    /// Ask an async builder whether its base data has landed. A builder that
    /// doesn't define `_base_ready` is always ready (the common, non-streaming
    /// case); a streaming one returns `false` until its coarse base map arrives,
    /// and the planet shows the procedural fallback meanwhile.
    fn gd_base_ready(baker: &Gd<CesBuilder>) -> bool {
        let mut obj = baker.clone().upcast::<Object>();
        if !obj.has_method(crate::builder::BASE_READY) {
            return true;
        }
        obj.call(crate::builder::BASE_READY, &[]).try_to::<bool>().unwrap_or(true)
    }

    /// Take back everything the async builder finished since last frame.
    ///
    /// A submission is applied only if its chunk is still in `cut` — which both
    /// implements cancellation (fly past a chunk and its late result is dropped)
    /// and guarantees we have the chunk's corners for the finite-difference
    /// normal. Applying one to an ALREADY-RESIDENT chunk marks it dirty so it
    /// re-realizes with the new surface: that is the streaming refinement path
    /// (coarse tile now, finer tile when the download lands).
    fn drain_gd_submissions(&mut self, cut: &[Chunk], tile_res: u32) {
        let Some(queue) = self.gd_submits.clone() else { return };
        let raws = queue.drain();
        if raws.is_empty() {
            return;
        }
        let tag = self.planet_tag();
        let by_id: HashMap<ChunkId, &Chunk> = cut.iter().map(|c| (c.id, c)).collect();

        for raw in &raws {
            // Unknown / stale / wrong-planet handle: silently dropped, by design.
            let Some(id) = async_bake::handle_decode(tag, raw.handle) else { continue };
            let Some(chunk) = by_id.get(&id) else {
                self.gd_outstanding.remove(&id);
                continue;
            };

            let sub = match async_bake::validate_submission(raw, tile_res) {
                Ok(s) => s,
                Err(e) => {
                    // A bad payload inside a bake loop would print thousands of
                    // identical lines a second; say it once, then stay quiet.
                    if queue.should_report_error() {
                        godot_error!(
                            "[Celestial] submit_chunk rejected: {e}. \
                             Further submit_chunk errors from this builder are suppressed."
                        );
                    }
                    self.gd_outstanding.remove(&id);
                    continue;
                }
            };

            let normal = sub.normal.unwrap_or_else(|| {
                let dirs = async_bake::chunk_dirs(chunk.corners, tile_res);
                async_bake::fd_normals(chunk.corners, &dirs, &sub.height, tile_res)
            });
            let surface = ChunkSurface { color: sub.color, height: sub.height, normal };
            self.ready_surfaces.insert(id, (self.param_epoch, surface));
            self.gd_outstanding.remove(&id);
            // Already drawn? Re-realize it with the fresher surface.
            if let Some(c) = self.cache.as_mut() {
                if c.slot_of(id).is_some() {
                    c.mark_dirty(id);
                }
            }
        }
    }

    /// Hand the async builder the cut chunks it hasn't produced yet — those not
    /// resident, with no ready surface and not already outstanding. Bounded by
    /// the same constants the Rust bake pool uses, so a fast flight can't flood
    /// the builder with chunks it will never draw. One call per frame; the
    /// builder must not block in it.
    fn request_gd_bakes(&mut self, cut: &[Chunk], tile_res: u32) {
        let Some(mut baker) = self.gd_async_baker.clone() else { return };
        let tag = self.planet_tag();
        let cache = self.cache.as_ref();

        let mut requests = VarArray::new();
        let mut fresh: Vec<ChunkId> = Vec::new();
        for chunk in cut {
            if self.gd_outstanding.len() + fresh.len() >= 64 || fresh.len() >= 24 {
                break;
            }
            let id = chunk.id;
            let resident = cache.map(|c| c.slot_of(id).is_some()).unwrap_or(false);
            if resident || self.ready_surfaces.contains(&id) || self.gd_outstanding.contains(&id)
            {
                continue;
            }
            // Too deep to encode a handle for: never handed over (it keeps its
            // ancestor stand-in). `MAX_HANDLE_DEPTH` is 20, well past any cut.
            let Some(handle) = async_bake::handle_encode(tag, id) else { continue };

            let mut d = VarDictionary::new();
            d.set(&"handle".to_variant(), &handle.to_variant());
            d.set(
                &"corners".to_variant(),
                &PackedVector3Array::from(&chunk.corners[..]).to_variant(),
            );
            d.set(&"tile_res".to_variant(), &(tile_res as i64).to_variant());
            d.set(&"depth".to_variant(), &(id.depth as i64).to_variant());
            requests.push(&d.to_variant());
            fresh.push(id);
        }

        if fresh.is_empty() {
            return;
        }
        self.gd_outstanding.extend(fresh);
        baker.call(crate::builder::BAKE_REQUESTED, &[requests.to_variant()]);
    }

    /// The active `CpuCustom` builder (its GDScript `height`/`color`/`normal`
    /// overrides are called per texel), when the first enabled builder is
    /// [`BuilderRoute::CpuCustom`] — else `None`.
    fn find_gd_baker(&self) -> Option<Gd<CesBuilder>> {
        let b = self.active_builder()?;
        if crate::builder::route_of(&b) == BuilderRoute::CpuCustom {
            Some(b)
        } else {
            None
        }
    }

    /// Bake one chunk's surface by calling the `CpuCustom` builder's **batched**
    /// `height` / `color` / `normal` — ONE call each per chunk (main thread),
    /// passing a `PackedVector3Array` of texel directions and getting a packed
    /// array back. This collapses the per-texel Rust→GDScript FFI (`tile_res²`
    /// calls) into a handful, the dominant cost of the GDScript path. Uses the
    /// SAME direction mapping the shaders use; the normal is finite-differenced
    /// from the height grid unless the builder returned its own. `height_scale`
    /// is applied by the GPU (as `surface_height_scale`), so we store the raw
    /// fraction here — matching the GPU custom path.
    fn gd_bake_chunk(
        builder: &mut Gd<CesBuilder>,
        chunk: &Chunk,
        tile_res: u32,
    ) -> ChunkSurface {
        let n = tile_res as usize;
        // Texel-centre world directions (folded onto the chunk triangle),
        // computed once and reused for every batched call and the FD normal.
        // Same helper the async route exposes to GDScript as `chunk_dirs`.
        let dirs = async_bake::chunk_dirs(chunk.corners, tile_res);
        let dirs_packed = PackedVector3Array::from(&dirs[..]);

        // Which optional functions the GDScript builder defines (missing height =
        // flat, missing color = white, missing normal = finite-differenced).
        let res_obj = builder.clone().upcast::<Object>();
        let has_height = res_obj.has_method("height");
        let has_color = res_obj.has_method("color");
        let has_normal = res_obj.has_method("normal");

        // height(dirs) -> PackedFloat32Array — one call. A short/absent return
        // just leaves the missing texels flat (0).
        let mut height = vec![0f32; n * n];
        if has_height {
            let hp = builder
                .call("height", &[dirs_packed.to_variant()])
                .try_to::<PackedFloat32Array>()
                .unwrap_or_default();
            let hs = hp.as_slice();
            for (i, h) in height.iter_mut().enumerate() {
                *h = hs.get(i).copied().unwrap_or(0.0);
            }
        }

        // color(dirs, heights) -> PackedColorArray — one call (missing = white).
        let white = Color::from_rgba(1.0, 1.0, 1.0, 1.0);
        let mut color = vec![255u8; n * n * 4];
        if has_color {
            let heights_packed = PackedFloat32Array::from(&height[..]);
            let cp = builder
                .call("color", &[dirs_packed.to_variant(), heights_packed.to_variant()])
                .try_to::<PackedColorArray>()
                .unwrap_or_default();
            let cs = cp.as_slice();
            for i in 0..(n * n) {
                let col = cs.get(i).copied().unwrap_or(white);
                color[i * 4] = (col.r.clamp(0.0, 1.0) * 255.0) as u8;
                color[i * 4 + 1] = (col.g.clamp(0.0, 1.0) * 255.0) as u8;
                color[i * 4 + 2] = (col.b.clamp(0.0, 1.0) * 255.0) as u8;
                color[i * 4 + 3] = 255;
            }
        }

        // Optional normal(dirs, heights) -> PackedVector3Array — one call; each
        // texel with a returned normal uses it, the rest fall back to the FD.
        let overridden = if has_normal {
            let heights_packed = PackedFloat32Array::from(&height[..]);
            builder
                .call("normal", &[dirs_packed.to_variant(), heights_packed.to_variant()])
                .try_to::<PackedVector3Array>()
                .ok()
        } else {
            None
        };
        let over = overridden.as_ref().map(|a| a.as_slice());

        // Normals: each texel the builder gave one for uses it; the rest are
        // finite-differenced from the height grid (shared with the async route).
        let normal = async_bake::pack_normals(chunk.corners, &dirs, &height, tile_res, over);
        ChunkSurface { color, height, normal }
    }

    /// Re-apply planet params changed since the job was built (editor live-edit
    /// or a runtime setter). Buffer-shaping params (`chunk_res`, `tile_res`,
    /// `vram_budget_gib`) force a full rebuild. `radius` is baked into every
    /// realized chunk's geometry — and into the world positions of its
    /// scattered instances — so a change invalidates the whole chunk cache:
    /// every resident chunk re-realizes (and re-places its scatter) at the new
    /// radius over the next frames, instead of only newly-visible chunks
    /// picking it up. Also refreshes the draw AABB envelope.
    fn reapply_param_changes(&mut self) {
        if self.job.is_none() {
            return;
        }
        if self.chunk_res != self.built_res
            || self.tile_res != self.built_tile_res
            || self.vram_budget_gib != self.built_budget_gib
        {
            // Re-cap the CPU surface cache with the edited sizing right away
            // (CEL-91): shrinking the budget must free the held surfaces NOW, not
            // only once `ensure_job` rebuilds. `teardown_job` clears the entries;
            // this is what keeps the *capacity* in step with the new budget.
            let budget = self.effective_budget() as usize;
            self.ready_surfaces.set_capacity(budget);
            self.teardown_job(); // ensure_job rebuilds with the new sizing
            return;
        }
        if self.radius != self.built_radius {
            self.built_radius = self.radius;
            if let Some(cache) = self.cache.as_mut() {
                cache.invalidate_all();
            }
            let m = self.radius * 1.3;
            let aabb = Aabb { position: Vector3::splat(-m), size: Vector3::splat(2.0 * m) };
            let mut rs = RenderingServer::singleton();
            if let Some(mm) = &self.multimesh {
                rs.multimesh_set_custom_aabb(mm.get_rid(), aabb);
            }
            for smm in &self.scatter_mms {
                rs.multimesh_set_custom_aabb(smm.get_rid(), aabb);
            }
        }
    }

    /// Build one indirect MultiMesh child per scatter layer (CEL-73), connect
    /// each layer's `changed` signal, and snapshot the structural params for
    /// later edit classification. Returns the GPU configs for the job.
    fn build_scatter_children(
        &mut self,
        rs: &mut Gd<RenderingServer>,
        aabb_margin: f32,
    ) -> Vec<ScatterConfig> {
        self.scatter_mmis.clear();
        self.scatter_mms.clear();
        self.scatter_meshes.clear();
        self.scatter_snapshot.clear();
        let self_gd = self.to_gd();
        let mut cfgs = Vec::new();
        let layers: Vec<Gd<CesScatterLayer>> = self.scatter_layers.iter_shared().collect();
        for (i, layer) in layers.into_iter().enumerate() {
            // Every layer gets the `changed` connection (so assigning a mesh
            // to an inactive layer later triggers the structure check)...
            let callable = Callable::from_object_method(&self_gd, "on_scatter_layer_changed");
            let mut layer_mut = layer.clone();
            if !layer_mut.is_connected("changed", &callable) {
                layer_mut.connect("changed", &callable);
            }

            let (k, max_inst, mesh, snap) = {
                let l = layer.bind();
                // ...but only layers WITH a mesh become GPU layers. No mesh =
                // inactive: renders nothing (no silent fallback).
                let Some(mesh) = l.mesh.clone() else { continue };
                let k = l.instances_per_cell.clamp(1, 64) as u32;
                let max_inst = l.max_instances.clamp(64, 4_000_000) as u32;
                let snap = ScatterSnapshot {
                    layer_id: layer.instance_id().to_i64(),
                    lod_level: l.lod_level,
                    instances_per_cell: l.instances_per_cell,
                    max_instances: l.max_instances,
                    seed: l.seed,
                    scale: l.scale,
                    mesh_rid: mesh.get_rid(),
                };
                (k, max_inst, mesh, snap)
            };

            let multimesh = MultiMesh::new_gd();
            let mm_rid = multimesh.get_rid();
            let mut mmi = MultiMeshInstance3D::new_alloc();
            let display = layer.bind().layer_name.to_string();
            if display.is_empty() {
                mmi.set_name(&format!("CesScatterLayer{i}"));
            } else {
                mmi.set_name(&format!("Scatter_{display}"));
            }
            mmi.set_multimesh(&multimesh);
            self.base_mut().add_child(&mmi);

            // CEL-58 order: allocate indirect BEFORE set_mesh so the renderer
            // creates the indirect command buffer the compact pass writes.
            rs.multimesh_allocate_data_ex(
                mm_rid,
                max_inst as i32,
                MultimeshTransformFormat::TRANSFORM_3D,
            )
            .use_indirect(true)
            .done();
            rs.multimesh_set_mesh(mm_rid, mesh.get_rid());
            rs.multimesh_set_custom_aabb(
                mm_rid,
                Aabb {
                    position: Vector3::splat(-aabb_margin),
                    size: Vector3::splat(2.0 * aabb_margin),
                },
            );

            cfgs.push(ScatterConfig {
                mm_rid,
                capacity: celestial_algo::scatter::capacity(k),
                max_instances: max_inst,
            });
            self.scatter_mmis.push(mmi);
            self.scatter_mms.push(crate::gpu::owned::IndirectMultiMesh::new(
                multimesh,
                crate::gpu::owned::MainDeviceSink::new(),
            ));
            self.scatter_meshes.push(mesh);
            self.scatter_snapshot.push(snap);
        }
        // First stage after a (re)build must carry scatter params.
        self.scatter_dirty = !self.scatter_mms.is_empty();
        cfgs
    }

    /// Classify scatter edits (CEL-73). Layer add/remove or a capacity-affecting
    /// change (`instances_per_cell`, `max_instances`) tears the job down for a
    /// rebuild next frame; `lod_level`/`seed`/`scale` re-place resident chunks
    /// via `invalidate_all`; a mesh swap rebinds in place. density/height need
    /// nothing here — they flow through the next stage's params snapshot.
    fn check_scatter_structure(&mut self) {
        if self.job.is_none() {
            return;
        }
        // The ACTIVE set (layers with a mesh) must match the snapshot: a layer
        // added/removed/reordered — or gaining/losing its mesh — rebuilds.
        let active: Vec<Gd<CesScatterLayer>> = self
            .scatter_layers
            .iter_shared()
            .filter(|l| l.bind().mesh.is_some())
            .collect();
        let ids: Vec<i64> = active.iter().map(|l| l.instance_id().to_i64()).collect();
        let snap_ids: Vec<i64> = self.scatter_snapshot.iter().map(|s| s.layer_id).collect();
        if ids != snap_ids {
            self.teardown_job();
            return;
        }
        if !self.scatter_dirty {
            return;
        }
        for (i, layer) in active.into_iter().enumerate() {
            let (k, max_inst, lod, seed, scale, mesh) = {
                let l = layer.bind();
                (
                    l.instances_per_cell,
                    l.max_instances,
                    l.lod_level,
                    l.seed,
                    l.scale,
                    l.mesh.clone().expect("active layer has a mesh"),
                )
            };
            let snap_k = self.scatter_snapshot[i].instances_per_cell;
            let snap_max = self.scatter_snapshot[i].max_instances;
            let snap_lod = self.scatter_snapshot[i].lod_level;
            let snap_seed = self.scatter_snapshot[i].seed;
            let snap_scale = self.scatter_snapshot[i].scale;
            let snap_mesh_rid = self.scatter_snapshot[i].mesh_rid;
            if k != snap_k || max_inst != snap_max {
                self.teardown_job();
                return;
            }
            // lod_level/seed/scale are baked into the cached placement, so an
            // edit must re-place resident chunks (GPU-only; slots preserved, the
            // next updates push them back through realize → scatter-place).
            // Height gates + density are compact-side (params snapshot).
            if lod != snap_lod || seed != snap_seed || scale != snap_scale {
                if let Some(cache) = self.cache.as_mut() {
                    cache.invalidate_all();
                }
                self.scatter_snapshot[i].lod_level = lod;
                self.scatter_snapshot[i].seed = seed;
                self.scatter_snapshot[i].scale = scale;
            }
            if mesh.get_rid() != snap_mesh_rid {
                // Mesh swap (Some -> Some): rebind in place, no rebuild.
                RenderingServer::singleton()
                    .multimesh_set_mesh(self.scatter_mms[i].get_rid(), mesh.get_rid());
                self.scatter_snapshot[i].mesh_rid = mesh.get_rid();
                self.scatter_meshes[i] = mesh;
            }
        }
    }

    /// Tear the job + all MultiMesh children down; `ensure_job` rebuilds next
    /// frame (used for structural scatter edits). The chunk cache restarts
    /// empty, so terrain re-realizes over the following frames (amortized by
    /// `MAX_BAKES_PER_FRAME`).
    fn teardown_job(&mut self) {
        // Dropping the job drops its `ChunkGpuResources`, whose `Owned<K>` handles
        // queue their RIDs on the render-thread free-drain (CEL-91). No deferred
        // `dispose` Callable: a Callable holds no strong ref, so the job used to die
        // before it ever ran — leaking the whole GPU pool.
        self.job = None;
        self.run_cb = None;
        self.cache = None;
        // Clear the CPU-surface state too: `ensure_job` repopulates it from the
        // CURRENT builder next frame, so leaving stale values here means a swap
        // to a builder that needs none (e.g. CpuNoise → GpuNoise) keeps the
        // old bake pool alive and it keeps painting the previous surface. Every
        // rebuild path funnels through here, so this is the one place to reset.
        self.provider = None;
        self.bake_pool = None;
        self.gd_baker = None;
        self.gd_async_baker = None;
        // Drop anything the old builder's workers already queued: it was baked
        // against the old params/geometry and must not land on the new job.
        if let Some(q) = self.gd_submits.take() {
            q.clear();
        }
        self.gd_outstanding.clear();
        self.ready_surfaces.clear();
        self.stale_surfaces.clear();
        self.was_base_ready = false;
        if let Some(mut mmi) = self.mmi.take() {
            mmi.queue_free();
        }
        self.multimesh = None;
        self.material = None;
        self._template = None;
        for mmi in &mut self.scatter_mmis {
            mmi.queue_free();
        }
        self.scatter_mmis.clear();
        self.scatter_mms.clear();
        self.scatter_meshes.clear();
        self.scatter_snapshot.clear();
        self.wired_pos = Rid::Invalid;
        self.wired_verts = Rid::Invalid;
        self.wired_color = Rid::Invalid;
        self.wired_normal = Rid::Invalid;
        self.last_slots.clear();
        self.last_cam = None;
        self.scatter_dirty = false;
        // Re-baseline the shader-file watcher after any rebuild.
        self.shader_mtime = 0;
        // Re-prime the param poll: the rebuild already applied current values, so
        // the next poll should baseline (not fire a spurious reshade).
        self.last_param_values = None;
    }

    /// Schedule the render-thread run whenever a stage is pending (retries until
    /// the lazily-created multimesh buffers exist and the stage is consumed).
    fn pump(&mut self) {
        let pending = self.job.as_ref().map(|j| j.bind().stage.is_some()).unwrap_or(false);
        if pending {
            if let Some(cb) = self.run_cb.clone() {
                RenderingServer::singleton().call_on_render_thread(&cb);
            }
        }
    }

    /// Wire the GPU-created pos/verts + detail-atlas textures into the material
    /// once they exist.
    fn wire_textures(&mut self) {
        let (pos, verts, color, normal) = match &self.job {
            Some(job) => {
                let g = &job.bind().gpu;
                (g.pos_tex(), g.attr_tex(), g.color_atlas(), g.normal_atlas())
            }
            None => return,
        };
        if self.material.is_none() {
            return;
        }
        for (rid, wired, name) in [
            (pos, &mut self.wired_pos, "pos_tex"),
            (verts, &mut self.wired_verts, "verts_tex"),
            (color, &mut self.wired_color, "color_atlas"),
            (normal, &mut self.wired_normal, "normal_atlas"),
        ] {
            if rid.is_valid() && rid != *wired {
                let mut tex = Texture2Drd::new_gd();
                tex.set_texture_rd_rid(rid);
                self.material.as_mut().unwrap().set_shader_parameter(name, &tex.to_variant());
                *wired = rid;
            }
        }
    }

    /// Create (once) and update the analytic water proxy from the ACTIVE builder.
    /// The builder owns the water config, so water travels with the terrain: the
    /// sea sits at the builder's `water_height` (same formula for noise & custom,
    /// so the level means the same everywhere), it draws only when the builder's
    /// `water_enabled` is on, and the look comes from the builder's water exports.
    fn update_water(&mut self) {
        if self.water.is_none() {
            let mut parent = self.to_gd().upcast::<Node3D>();
            self.water = Some(crate::water_runtime::WaterRuntime::create(&mut parent));
        }

        let builder = self.active_builder();
        let (enabled, water_radius, params) = match builder {
            Some(b) => {
                let l = b.bind();
                let wr = crate::water::water_radius(self.radius, l.water_height(), l.height_scale());
                let params = crate::water_runtime::WaterParams {
                    deep_color: l.water_deep_color,
                    shallow_color: l.water_shallow_color,
                    wave_strength: l.water_wave_strength,
                    wave_scale: l.water_wave_scale,
                    wave_speed: l.water_wave_speed,
                    underwater_color: l.water_underwater_color,
                    underwater_density: l.water_underwater_density,
                    sun_dir: self.sun_direction(),
                };
                (l.water_enabled, wr, params)
            }
            // No builder → white planet, no water. `params` is unused when
            // disabled (the runtime early-outs on `visible == false`).
            None => (
                false,
                self.radius,
                crate::water_runtime::WaterParams {
                    deep_color: Color::from_rgb(0.05, 0.22, 0.42),
                    shallow_color: Color::from_rgb(0.20, 0.55, 0.70),
                    wave_strength: 0.55,
                    wave_scale: 0.15,
                    wave_speed: 0.04,
                    underwater_color: Color::from_rgb(0.04, 0.16, 0.28),
                    underwater_density: 0.02,
                    sun_dir: Vector3::UP,
                },
            ),
        };
        let center = self.base().get_global_position();
        if let Some(water) = self.water.as_mut() {
            water.update(center, water_radius, params, enabled);
        }
    }

    /// World-space direction pointing TOWARD the sun: the +Z (BACK) axis of the
    /// first `DirectionalLight3D` in the scene (a directional light emits along
    /// its −Z/FORWARD). Falls back to a fixed key direction if none is found.
    fn sun_direction(&self) -> Vector3 {
        use godot::classes::DirectionalLight3D;
        let fallback = || Vector3::new(0.4, 0.7, 0.55).normalized();
        let Some(root) = self.base().get_tree().get_root() else {
            return fallback();
        };
        let root: Gd<Node> = root.upcast();
        let light = root
            .find_children_ex("*")
            .type_("DirectionalLight3D")
            .recursive(true)
            .owned(false)
            .done()
            .iter_shared()
            .find_map(|n: Gd<Node>| n.try_cast::<DirectionalLight3D>().ok());
        match light {
            Some(l) => (l.get_global_transform().basis * Vector3::BACK).normalized(),
            None => fallback(),
        }
    }

    /// Active camera position in this node's local space.
    fn camera_local(&self) -> Option<Vector3> {
        use godot::classes::{EditorInterface, Engine};
        let global = if Engine::singleton().is_editor_hint() {
            EditorInterface::singleton()
                .get_editor_viewport_3d()
                .and_then(|vp| vp.get_camera_3d())
                .map(|c| c.get_global_position())
        } else {
            self.base()
                .get_viewport()
                .and_then(|vp| vp.get_camera_3d())
                .map(|c| c.get_global_position())
        }?;
        Some(self.base().get_global_transform().affine_inverse() * global)
    }
}

/// Build the chunk surface material (VERTEX from pos_tex by slot+VERTEX_ID;
/// colour + normal per-pixel from the detail atlas indexed by slot + UV).
fn make_material(res: u32, tile_res: u32, lod_colors: bool) -> Gd<ShaderMaterial> {
    let shader = godot::tools::load::<Shader>(CHUNK_SHADER);
    let mut mat = ShaderMaterial::new_gd();
    mat.set_shader(&shader);
    mat.set_shader_parameter("attr_w", &(ATTR_TEX_WIDTH as i32).to_variant());
    mat.set_shader_parameter("verts_per_chunk", &(verts_per_chunk(res) as i32).to_variant());
    // chunk_res drives the surface shader's geomorph even-sublattice decode.
    mat.set_shader_parameter("chunk_res", &(res as i32).to_variant());
    mat.set_shader_parameter("tile_res", &(tile_res as i32).to_variant());
    mat.set_shader_parameter("lod_colors", &lod_colors.to_variant());
    mat
}
