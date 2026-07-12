//! Debug-only standalone single-tile texture viewer (`CelestialTileViewer`).
//!
//! A flat 2-D view of ONE icosphere face's baked surface colour, isolated from
//! the clipmap/icosphere. The `TileViewer` compute shader bakes the SAME albedo
//! + grass/rock detail as `ChunkTileBake.slang` over a SQUARE barycentric region
//! of one face into a `W×W` rgba8 texture. Each scroll/pan re-dispatches the bake
//! over a smaller/shifted barycentric window (no caching), so the user can see
//! how the detail noise holds up as they zoom toward rock/grass scale.
//!
//! Additive and read-only with respect to the production chunk path: it only
//! reuses the shared shading code + `TerrainGpu` defaults. GPU work runs on the
//! main `RenderingDevice` via `RenderingServer::call_on_render_thread`. The GPU
//! resources live in a separate `RefCounted` [`TileViewerJob`] (the CEL-58
//! pattern): the render-thread callback borrows the job, not the node, so a
//! synchronous `call_on_render_thread` can't re-enter the node's borrow.

use std::sync::Arc;

use bytemuck::Zeroable;
use godot::classes::notify::NodeNotification;
use godot::classes::rendering_device::UniformType;
use godot::classes::rendering_server::MultimeshTransformFormat;
use godot::classes::{
    ArrayMesh, INode, INode3D, MultiMesh, MultiMeshInstance3D, Node, Node3D, RdUniform, RefCounted,
    RenderingDevice, RenderingServer, Shader, ShaderMaterial, Texture2Drd,
};
use godot::prelude::*;

use celestial_algo::quadtree::{base_face_frames, Bary, Chunk, ChunkId};

use crate::chunk_descriptors::{pack_chunks, pack_instances, verts_per_chunk};
use crate::chunk_mesh::reference_chunk_mesh;
use crate::chunk_pipeline::{CesChunkJob, ChunkStage};
use crate::descriptors::{assemble, HeightGpu, TerrainGpu, TextureGpu};
use crate::gpu::ATTR_TEX_WIDTH;
use crate::gpu::device;
use crate::gpu::owned::{
    MainDeviceSink, Owned, RdBuffer, RdPipeline, RdShader, RdTexture, RdUniformSet, RidSink,
};

const TILE_VIEWER_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/TileViewer.spv"));

/// Sphere radius the viewer bakes against (matches the HQ scene / chunk tests).
const RADIUS: f32 = 1000.0;

/// std430 params for `TileViewer.slang` — byte-identical to `struct
/// TileViewerParams` there (176 bytes). The `tile_viewer_params_layout` test
/// locks the offsets. Field offsets: a 0, b 16, c 32, sub0 48, sub1 64,
/// sub2 80, width 96, tex_res 100, terrain 112.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct TileViewerParams {
    /// Face corner A (xyz world) + sphere radius (w).
    pub a: [f32; 4],
    /// Face corner B (xyz), w unused.
    pub b: [f32; 4],
    /// Face corner C (xyz), w unused.
    pub c: [f32; 4],
    /// Viewed sub-triangle corner 0 in face barycentric (wb, wc in xy) → UV (0,0).
    pub sub0: [f32; 4],
    /// Sub-triangle corner 1 (wb, wc) → UV (1,0).
    pub sub1: [f32; 4],
    /// Sub-triangle corner 2 (wb, wc) → UV (0,1).
    pub sub2: [f32; 4],
    /// Output texture edge (texels).
    pub width: u32,
    /// Texture-resolution quantisation grid.
    pub tex_res: u32,
    /// Padding to 16-align the trailing terrain block.
    pub _pad: [u32; 2],
    /// Embedded 56-byte terrain block (shared with the chunk path).
    pub terrain: TerrainGpu,
    /// Tail padding keeping the struct a 16-byte multiple (std430 stride rule).
    pub _pad2: [f32; 2],
}

/// The viewer's GPU resources on the MAIN `RenderingDevice` (render-thread only).
///
/// FIELD ORDER IS THE DROP ORDER (see the DROP-ORDER CONTRACT in `gpu::owned`):
/// the uniform set (a dependent) first, then the pipeline, then the shader and
/// the buffers/textures it derives from.
struct TileViewerGpu {
    sink: Arc<dyn RidSink>,
    set: RdUniformSet,
    pipeline: RdPipeline,
    shader: RdShader,
    /// `W×W` rgba8 colour output (STORAGE|SAMPLING|CAN_COPY_FROM) — wrapped in a
    /// `Texture2DRD` for display.
    tex: RdTexture,
    /// `W×W` rgba8 world-normal output (encoded *0.5+0.5) for the flat+normal mode.
    nrm: RdTexture,
    params_buf: RdBuffer,
    width: u32,
    built: bool,
    init_failed: bool,
}

impl Default for TileViewerGpu {
    fn default() -> Self {
        // Main device: frees are queued and drained on the render thread.
        let sink: Arc<dyn RidSink> = MainDeviceSink::new();
        Self {
            set: Owned::invalid(sink.clone()),
            pipeline: Owned::invalid(sink.clone()),
            shader: Owned::invalid(sink.clone()),
            tex: Owned::invalid(sink.clone()),
            nrm: Owned::invalid(sink.clone()),
            params_buf: Owned::invalid(sink.clone()),
            width: 0,
            built: false,
            init_failed: false,
            sink,
        }
    }
}

impl TileViewerGpu {
    /// Build the pipeline + output texture + params buffer + uniform set. Idempotent.
    fn ensure_ready(&mut self, rd: &mut Gd<RenderingDevice>, width: u32) -> bool {
        if self.init_failed {
            return false;
        }
        if self.built {
            return true;
        }
        let sink = self.sink.clone();
        let Some((shader, pipeline)) =
            device::compute_pipeline(rd, &sink, TILE_VIEWER_SPV, "TileViewer")
        else {
            self.init_failed = true;
            return false;
        };
        self.shader = shader;
        self.pipeline = pipeline;
        // rgba8 STORAGE|SAMPLING|CAN_COPY_FROM square textures (atlas_texture fits).
        self.tex = device::atlas_texture(rd, &sink, width, width);
        self.nrm = device::atlas_texture(rd, &sink, width, width);
        self.params_buf =
            device::storage_buffer(rd, &sink, &vec![0u8; std::mem::size_of::<TileViewerParams>()]);
        let uniforms: Array<Gd<RdUniform>> = [
            device::uniform(UniformType::STORAGE_BUFFER, 0, self.params_buf.rid()),
            device::uniform(UniformType::IMAGE, 1, self.tex.rid()),
            device::uniform(UniformType::IMAGE, 2, self.nrm.rid()),
        ]
        .into_iter()
        .collect();
        self.set = Owned::new(rd.uniform_set_create(&uniforms, self.shader.rid(), 0), sink);
        self.width = width;
        self.built = true;
        true
    }

    /// Upload params and record a `W*W`-thread bake into the output texture.
    fn dispatch(&mut self, rd: &mut Gd<RenderingDevice>, params: &TileViewerParams) {
        let bytes = bytemuck::bytes_of(params);
        rd.buffer_update(
            self.params_buf.rid(),
            0,
            bytes.len() as u32,
            &PackedByteArray::from(bytes),
        );
        let total = self.width.saturating_mul(self.width);
        let groups = total.div_ceil(64);
        if groups == 0 {
            return;
        }
        let list = rd.compute_list_begin();
        rd.compute_list_bind_compute_pipeline(list, self.pipeline.rid());
        rd.compute_list_bind_uniform_set(list, self.set.rid(), 0);
        rd.compute_list_dispatch(list, groups, 1, 1);
        rd.compute_list_end();
    }
}

/// Render-thread job owning the viewer's GPU resources + the pending view. A
/// separate `RefCounted` so the render-thread callback never re-enters the
/// node's borrow (mirrors `CesChunkJob`).
#[derive(GodotClass)]
#[class(base = RefCounted, no_init)]
pub struct TileViewerJob {
    base: Base<RefCounted>,
    gpu: TileViewerGpu,
    /// The view to bake on the next render-thread run.
    params: TileViewerParams,
    /// Set by `set_view`, drained by `run`.
    pending: bool,
}

impl TileViewerJob {
    fn create() -> Gd<Self> {
        Gd::from_init_fn(|base| Self {
            base,
            gpu: TileViewerGpu::default(),
            params: TileViewerParams::zeroed(),
            pending: false,
        })
    }

    /// The colour output texture RID (`Invalid` until the first run builds it).
    fn tex_rid(&self) -> Rid {
        self.gpu.tex.rid()
    }

    /// The normal output texture RID (`Invalid` until the first run builds it).
    fn nrm_rid(&self) -> Rid {
        self.gpu.nrm.rid()
    }
}

#[godot_api]
impl TileViewerJob {
    /// Render-thread entry point: ensure resources, then bake the pending view.
    #[func]
    fn run(&mut self) {
        if !self.pending {
            return;
        }
        let rs = RenderingServer::singleton();
        let Some(mut rd) = rs.get_rendering_device() else { return };
        let w = self.params.width;
        if !self.gpu.ensure_ready(&mut rd, w) {
            return; // retry next request
        }
        let params = self.params;
        self.gpu.dispatch(&mut rd, &params);
        self.pending = false;
    }
}

/// Debug node driving the single-tile texture viewer. Exposed to the editor as a
/// `tool` so the bundled `debug/tile_viewer.tscn` scene can run it standalone.
#[derive(GodotClass)]
#[class(base = Node, tool, init, internal)]
pub struct CelestialTileViewer {
    base: Base<Node>,

    /// Output texture edge in texels (square `W×W` bake).
    #[export]
    #[init(val = 1024)]
    width: i64,
    /// Which of the 20 icosphere faces to view.
    #[export]
    #[init(val = 7)]
    face: i64,
    /// Texture (bake) resolution: the surface is sampled on a `tex_res × tex_res`
    /// grid (carried in the params `c.w` slot), so the view shows the detail AT
    /// that texture resolution regardless of the output texel count. Cycled by R
    /// via `set_tex_res` (internal — not an `#[export]`, which would auto-generate
    /// a colliding `set_tex_res`).
    #[init(val = 64)]
    tex_res: i64,

    /// Viewed sub-triangle corners in face barycentric (wb, wc) — set by
    /// `set_region`. Defaults to the full face triangle so the standalone node
    /// shows the whole face if no region is supplied.
    #[init(val = Vector2::new(0.0, 0.0))]
    sub0: Vector2,
    #[init(val = Vector2::new(1.0, 0.0))]
    sub1: Vector2,
    #[init(val = Vector2::new(0.0, 1.0))]
    sub2: Vector2,

    /// Detail-normal bump enable (1.0 on, 0.0 off) for the baked normal map —
    /// carried to the shader in the spare `b.w` slot. Default off (bump removed
    /// from production); the scene's B key re-enables it for comparison.
    #[init(val = 0.0)]
    bump_enable: f32,

    job: Option<Gd<TileViewerJob>>,
    run_cb: Option<Callable>,
}

#[godot_api]
impl INode for CelestialTileViewer {
    fn on_notification(&mut self, what: NodeNotification) {
        if matches!(what, NodeNotification::EXIT_TREE | NodeNotification::PREDELETE) {
            // Dropping the job drops its `Owned<K>` handles, which queue the RID
            // frees on the render thread (CEL-91).
            self.job = None;
            self.run_cb = None;
        }
    }
}

#[godot_api]
impl CelestialTileViewer {
    /// Set the face and the viewed sub-triangle (barycentric corners, matching
    /// the chunk's region) and schedule a render-thread re-bake. Corner 0 → UV
    /// (0,0), corner 1 → UV (1,0), corner 2 → UV (0,1), to match the display
    /// mesh's vertex UVs.
    #[func]
    fn set_region(&mut self, face: i64, c0: Vector2, c1: Vector2, c2: Vector2) {
        self.face = face;
        self.sub0 = c0;
        self.sub1 = c1;
        self.sub2 = c2;
        self.rebake();
    }

    /// Set the texture (bake) resolution and re-bake. Cycled by the `R` keybind.
    #[func]
    fn set_tex_res(&mut self, n: i64) {
        self.tex_res = n.clamp(1, 4096);
        self.rebake();
    }

    /// Build params from the stored region + resolution and schedule the bake.
    fn rebake(&mut self) {
        self.ensure_job();

        let frames = base_face_frames(RADIUS);
        let fi = self.face.clamp(0, frames.len() as i64 - 1) as usize;
        let frame = &frames[fi];
        let terrain = assemble(&HeightGpu::default(), &TextureGpu::default());
        let params = TileViewerParams {
            a: [frame.a.x, frame.a.y, frame.a.z, frame.radius],
            // b.w carries the detail-normal bump enable (1.0 on, 0.0 off).
            b: [frame.b.x, frame.b.y, frame.b.z, self.bump_enable],
            c: [frame.c.x, frame.c.y, frame.c.z, 0.0],
            sub0: [self.sub0.x, self.sub0.y, 0.0, 0.0],
            sub1: [self.sub1.x, self.sub1.y, 0.0, 0.0],
            sub2: [self.sub2.x, self.sub2.y, 0.0, 0.0],
            width: self.width.clamp(16, 4096) as u32,
            tex_res: self.tex_res.clamp(1, 4096) as u32,
            _pad: [0, 0],
            terrain,
            _pad2: [0.0, 0.0],
        };

        if let Some(job) = &mut self.job {
            let mut j = job.bind_mut();
            j.params = params;
            j.pending = true;
        }
        if let Some(cb) = self.run_cb.clone() {
            RenderingServer::singleton().call_on_render_thread(&cb);
        }
    }

    /// Current texture (bake) resolution (for the HUD label).
    #[func]
    fn tex_resolution(&self) -> i64 {
        self.tex_res
    }

    /// Enable/disable the baked detail-normal bump (1.0 on, 0.0 off) and re-bake.
    #[func]
    fn set_bump(&mut self, enable: f32) {
        self.bump_enable = if enable > 0.5 { 1.0 } else { 0.0 };
        self.rebake();
    }

    /// The output texture RID (wrap in a `Texture2DRD` for display). `Invalid`
    /// until the first render-thread run has built resources.
    #[func]
    fn texture_rid(&self) -> Rid {
        self.job.as_ref().map(|j| j.bind().tex_rid()).unwrap_or(Rid::Invalid)
    }

    /// The baked world-normal texture RID (wrap in a `Texture2DRD`). `Invalid`
    /// until the first render-thread run has built resources.
    #[func]
    fn normal_texture_rid(&self) -> Rid {
        self.job.as_ref().map(|j| j.bind().nrm_rid()).unwrap_or(Rid::Invalid)
    }

    /// Lazily create the render-thread job + its run callable.
    fn ensure_job(&mut self) {
        if self.job.is_some() {
            return;
        }
        let job = TileViewerJob::create();
        self.run_cb = Some(Callable::from_object_method(&job, "run"));
        self.job = Some(job);
    }
}

// ───────────────────────── Mode B: real chunk pipeline (FLAT) ─────────────────
//
// `CelestialTileChunk` renders ONE chunk through the *production* chunk path
// (`ChunkRealize` + `ChunkTileBake` + `terrain_chunk.gdshader`) so the viewer can
// A/B-compare the per-pixel baked tile (mode A) against the actual surface the
// planet draws (mode B). It is the same machinery `celestial.rs` drives,
// reduced to a single chunk laid out flat-on to the camera.
//
// NOTE on "flat": the production realize/bake shaders gate ALL terrain detail
// (displacement, grass/rock albedo, detail normals) on `radius > 0` — a literal
// `radius == 0` descriptor renders as a featureless flat-green triangle that is
// invariant to `chunk_res`. To make the A/B comparison meaningful (and to show
// resolution-dependent faceting) this node defaults to the real sphere frame
// (`flat_face == false`) but covers only a small sub-triangle of the face
// (`face_fraction`), so the patch is nearly planar yet runs the full detail path.
// Set `flat_face = true` to reproduce the literal `radius == 0` flat path.

const CHUNK_SHADER: &str = "res://addons/celestialsim/terrain_chunk.gdshader";

/// Debug node (mode B): one production chunk realized + baked + drawn flat-on.
#[derive(GodotClass)]
#[class(base = Node3D, tool, init, internal)]
pub struct CelestialTileChunk {
    base: Base<Node3D>,

    /// Which of the 20 icosphere faces this chunk lives on.
    #[export]
    #[init(val = 7)]
    face: i64,
    /// Sphere radius (matches the HQ scene / mode A).
    #[export]
    #[init(val = 1000.0)]
    radius: f32,
    /// Triangle resolution: the chunk is `chunk_res × chunk_res` triangles.
    #[export]
    #[init(val = 64)]
    chunk_res: i64,
    /// Per-chunk detail-tile resolution (colour/normal atlas), as in production.
    /// Cycled by R via `set_tile_res` (internal — not `#[export]`, which would
    /// auto-generate a colliding `set_tile_res`).
    #[init(val = 64)]
    tile_res: i64,
    /// Detail-normal bump enable (1.0 on, 0.0 off) — debug toggle that re-bakes
    /// the chunk's normal atlas with/without the high-frequency bump. Default off
    /// (the bump was removed from production); the scene's B key re-enables it.
    #[init(val = 0.0)]
    bump_enable: f32,
    /// Sub-triangle size in barycentric units (1.0 = the whole face). 1.0 keeps
    /// the chunk's full sphere curvature + displacement so low `chunk_res`
    /// tessellation faceting is visible; smaller values isolate a flatter patch.
    #[export(range = (0.02, 1.0, 0.01))]
    #[init(val = 1.0)]
    face_fraction: f32,
    /// Literal flat path: pass `radius = 0` to the descriptor. The production
    /// shader then emits featureless flat-green (detail is gated on `radius > 0`).
    #[export]
    #[init(val = false)]
    flat_face: bool,

    job: Option<Gd<CesChunkJob>>,
    run_cb: Option<Callable>,
    multimesh: Option<Gd<MultiMesh>>,
    mmi: Option<Gd<MultiMeshInstance3D>>,
    material: Option<Gd<ShaderMaterial>>,
    _template: Option<Gd<ArrayMesh>>,

    #[init(val = Rid::Invalid)]
    wired_pos: Rid,
    #[init(val = Rid::Invalid)]
    wired_verts: Rid,
    #[init(val = Rid::Invalid)]
    wired_color: Rid,
    #[init(val = Rid::Invalid)]
    wired_normal: Rid,

    built: bool,
}

#[godot_api]
impl INode3D for CelestialTileChunk {
    fn process(&mut self, _delta: f64) {
        self.ensure_built();
        self.pump();
        self.wire_textures();
    }

    fn on_notification(&mut self, what: godot::classes::notify::Node3DNotification) {
        use godot::classes::notify::Node3DNotification as N;
        if matches!(what, N::EXIT_TREE | N::PREDELETE) {
            // The job's `Owned<K>` handles free themselves on drop (CEL-91).
            self.job = None;
            self.run_cb = None;
        }
    }
}

#[godot_api]
impl CelestialTileChunk {
    /// Rebuild the chunk at a new triangle resolution (re-mesh + re-realize +
    /// re-bake). Called by the viewer's `R` keybind.
    #[func]
    fn set_resolution(&mut self, res: i64) {
        let r = res.clamp(2, 1024);
        if r == self.chunk_res && self.built {
            return;
        }
        self.chunk_res = r;
        self.teardown();
    }

    /// Set the per-chunk detail-tile (texture) resolution and rebuild the chunk
    /// so the new tile_res flows through ChunkTileBake. Triangles are unchanged.
    #[func]
    fn set_tile_res(&mut self, n: i64) {
        let r = n.clamp(8, 1024);
        if r == self.tile_res && self.built {
            return;
        }
        self.tile_res = r;
        self.teardown();
    }

    /// Enable/disable the detail-normal bump (1.0 on, 0.0 off) and rebuild so the
    /// normal atlas is re-baked. Lets the debug scene toggle bumps entirely.
    #[func]
    fn set_bump(&mut self, enable: f32) {
        let e = if enable > 0.5 { 1.0 } else { 0.0 };
        if e == self.bump_enable && self.built {
            return;
        }
        self.bump_enable = e;
        self.teardown();
    }

    /// Current triangle resolution (for the HUD label).
    #[func]
    fn chunk_resolution(&self) -> i64 {
        self.chunk_res
    }

    /// Current detail-tile resolution (for the HUD label).
    #[func]
    fn tile_resolution(&self) -> i64 {
        self.tile_res
    }

    /// World-space centroid of the chunk's base triangle (camera look-at target).
    #[func]
    fn chunk_centroid(&self) -> Vector3 {
        let c = self.world_corners();
        (c[0] + c[1] + c[2]) / 3.0
    }

    /// Outward (away-from-origin) face normal of the chunk's base triangle.
    #[func]
    fn chunk_normal(&self) -> Vector3 {
        let c = self.world_corners();
        let cen = (c[0] + c[1] + c[2]) / 3.0;
        let n = (c[1] - c[0]).cross(c[2] - c[0]).normalized();
        if n.dot(cen) < 0.0 {
            -n
        } else {
            n
        }
    }

    /// An in-plane "up" axis (toward corner A) for orienting the camera.
    #[func]
    fn chunk_up(&self) -> Vector3 {
        let c = self.world_corners();
        let cen = (c[0] + c[1] + c[2]) / 3.0;
        (c[0] - cen).normalized()
    }

    /// Circumradius of the base triangle (used to frame the camera distance).
    #[func]
    fn chunk_extent(&self) -> f32 {
        let c = self.world_corners();
        let cen = (c[0] + c[1] + c[2]) / 3.0;
        c.iter().map(|p| (*p - cen).length()).fold(0.0_f32, f32::max)
    }

    /// This chunk's face index (so Mode A bakes the same face).
    #[func]
    fn face_id(&self) -> i64 {
        self.face_index() as i64
    }

    /// The chunk's three sub-triangle corners in face barycentric (wb, wc), in
    /// the same order as `world_corners_packed` — fed to `TileViewer::set_region`.
    #[func]
    fn bary_corners(&self) -> PackedVector2Array {
        let b = self.sub_bary();
        PackedVector2Array::from(&[
            Vector2::new(b[0].wb, b[0].wc),
            Vector2::new(b[1].wb, b[1].wc),
            Vector2::new(b[2].wb, b[2].wc),
        ])
    }

    /// The chunk's three world-space corners (flat triangle for Mode A's mesh).
    #[func]
    fn world_corners_packed(&self) -> PackedVector3Array {
        let c = self.world_corners();
        PackedVector3Array::from(&[c[0], c[1], c[2]])
    }
}

impl CelestialTileChunk {
    /// Descriptor radius: 0 in literal flat mode, else the sphere radius.
    fn effective_radius(&self) -> f32 {
        if self.flat_face {
            0.0
        } else {
            self.radius
        }
    }

    /// The chunk's three barycentric corners: a `face_fraction`-scaled triangle
    /// centred on the face centroid (so the patch is small and nearly planar).
    fn sub_bary(&self) -> [Bary; 3] {
        let s = self.face_fraction.clamp(0.02, 1.0);
        let cb = 1.0 / 3.0;
        let lerp = |wb: f32, wc: f32| Bary { wb: cb + s * (wb - cb), wc: cb + s * (wc - cb) };
        [lerp(0.0, 0.0), lerp(1.0, 0.0), lerp(0.0, 1.0)]
    }

    /// Face index clamped to the valid 0..20 range.
    fn face_index(&self) -> usize {
        self.face.clamp(0, 19) as usize
    }

    /// The icosphere face frames with the descriptor radius applied (corners stay
    /// at sphere scale; `radius == 0` just makes `project_bary` linear == flat).
    fn frames(&self) -> Vec<celestial_algo::clipmap::FaceFrame> {
        let mut frames = base_face_frames(self.radius);
        let er = self.effective_radius();
        for f in &mut frames {
            f.radius = er;
        }
        frames
    }

    /// World positions of the chunk's three corners.
    fn world_corners(&self) -> [Vector3; 3] {
        let frames = self.frames();
        let f = &frames[self.face_index()];
        let b = self.sub_bary();
        [
            f.project_bary(b[0].wb, b[0].wc),
            f.project_bary(b[1].wb, b[1].wc),
            f.project_bary(b[2].wb, b[2].wc),
        ]
    }

    /// Lazily build the material + reference mesh + indirect MultiMesh + render
    /// job, then stage the single chunk. Mirrors `celestial::ensure_job`.
    fn ensure_built(&mut self) {
        if self.built {
            return;
        }
        let res = self.chunk_res.clamp(2, 1024) as u32;
        let tile_res = self.tile_res.clamp(8, 1024) as u32;

        let mut material = make_chunk_material(res, tile_res);
        let template = reference_chunk_mesh(res, &material.clone().upcast());

        let mut rs = RenderingServer::singleton();
        let multimesh = MultiMesh::new_gd();
        let mm_rid = multimesh.get_rid();
        let mut mmi = MultiMeshInstance3D::new_alloc();
        mmi.set_multimesh(&multimesh);
        // Hidden until `wire_textures` binds the GPU atlases — they only become
        // valid after the render-thread job runs (next frame), and drawing with
        // an unwired atlas trips a "binding 1 invalid" error every rebuild.
        mmi.set_visible(false);
        self.base_mut().add_child(&mmi);

        rs.multimesh_allocate_data_ex(mm_rid, 1, MultimeshTransformFormat::TRANSFORM_3D)
            .custom_data_format(true)
            .use_indirect(true)
            .done();
        rs.multimesh_set_mesh(mm_rid, template.get_rid());
        let m = self.radius.max(1.0) * 1.4;
        rs.multimesh_set_custom_aabb(
            mm_rid,
            Aabb { position: Vector3::splat(-m), size: Vector3::splat(2.0 * m) },
        );

        // Terrain ON (HQ defaults) so the production detail path runs.
        let terrain = assemble(&HeightGpu::default(), &TextureGpu::default());
        let job = CesChunkJob::create(mm_rid, 1, res, tile_res, self.radius, self.bump_enable, terrain, Vec::new());
        self.run_cb = Some(Callable::from_object_method(&job, "run"));

        material.set_shader_parameter("attr_w", &(ATTR_TEX_WIDTH as i32).to_variant());

        self.job = Some(job);
        self.multimesh = Some(multimesh);
        self.mmi = Some(mmi);
        self.material = Some(material);
        self._template = Some(template);
        self.built = true;

        self.stage_chunk(res);
    }

    /// Stage the single flat chunk for the render-thread job.
    fn stage_chunk(&mut self, res: u32) {
        let frames = self.frames();
        let fi = self.face_index();
        let f = &frames[fi];
        let bary = self.sub_bary();
        let corners = [
            f.project_bary(bary[0].wb, bary[0].wc),
            f.project_bary(bary[1].wb, bary[1].wc),
            f.project_bary(bary[2].wb, bary[2].wc),
        ];
        let chunk = Chunk {
            id: ChunkId { face: fi as u8, depth: 0, path: 0 },
            bary,
            corners,
            level: 0,
        };
        let desc_bytes = pack_chunks(&frames, &[(0u32, chunk)], res);
        // Single static depth-0 chunk: morph = 1 (no geomorph in the viewer).
        let instance_bytes = pack_instances(&[0u32], &[1.0]);
        if let Some(job) = &mut self.job {
            job.bind_mut().stage = Some(ChunkStage {
                desc_bytes,
                realize_count: 1,
                instance_bytes,
                instance_count: 1,
                // The CPU-surface path is the quadtree planet's; the tile viewer
                // stays procedural.
                surface_enabled: 0.0,
                surface_height_scale: 0.0,
                surface_patches: Vec::new(),
                scatter_aux_bytes: Vec::new(),
                scatter_vis_bytes: Vec::new(),
                scatter_vis_count: 0,
                scatter_layer_params: Vec::new(),
            });
        }
    }

    /// Schedule the render-thread run while a stage is pending.
    fn pump(&mut self) {
        let pending = self.job.as_ref().map(|j| j.bind().stage.is_some()).unwrap_or(false);
        if pending {
            if let Some(cb) = self.run_cb.clone() {
                RenderingServer::singleton().call_on_render_thread(&cb);
            }
        }
    }

    /// Wire the GPU-realized textures into the chunk material once they exist.
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
        // Reveal the chunk only once every atlas is bound (see `ensure_built`).
        if self.wired_pos.is_valid()
            && self.wired_verts.is_valid()
            && self.wired_color.is_valid()
            && self.wired_normal.is_valid()
        {
            if let Some(mmi) = self.mmi.as_mut() {
                if !mmi.is_visible() {
                    mmi.set_visible(true);
                }
            }
        }
    }

    /// Drop the job (its `Owned<K>` handles queue the RID frees on the render
    /// thread) and all keep-alives, so the next `ensure_built` rebuilds at the new
    /// resolution.
    fn teardown(&mut self) {
        if let Some(mut mmi) = self.mmi.take() {
            // Hide immediately: `queue_free` is deferred to end-of-frame, but the
            // atlas frees are queued on the render thread now, so a still-visible
            // MMI would draw a freed texture (binding-1 invalid) for a frame on
            // every resolution change.
            mmi.set_visible(false);
            mmi.queue_free();
        }
        self.job = None;
        self.run_cb = None;
        self.multimesh = None;
        self.material = None;
        self._template = None;
        self.wired_pos = Rid::Invalid;
        self.wired_verts = Rid::Invalid;
        self.wired_color = Rid::Invalid;
        self.wired_normal = Rid::Invalid;
        self.built = false;
    }
}

/// Build the production chunk surface material (same shader the planet uses).
fn make_chunk_material(res: u32, tile_res: u32) -> Gd<ShaderMaterial> {
    let shader = godot::tools::load::<Shader>(CHUNK_SHADER);
    let mut mat = ShaderMaterial::new_gd();
    mat.set_shader(&shader);
    mat.set_shader_parameter("attr_w", &(ATTR_TEX_WIDTH as i32).to_variant());
    mat.set_shader_parameter("verts_per_chunk", &(verts_per_chunk(res) as i32).to_variant());
    mat.set_shader_parameter("chunk_res", &(res as i32).to_variant());
    mat.set_shader_parameter("tile_res", &(tile_res as i32).to_variant());
    mat.set_shader_parameter("lod_colors", &false.to_variant());
    mat
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem::offset_of;

    #[test]
    fn tile_viewer_params_layout() {
        // std430: float4 members force 16-byte alignment; the struct array stride
        // must be a multiple of 16 or `params[0]` reads at the wrong offset. These
        // offsets must match `struct TileViewerParams` in TileViewer.slang.
        assert_eq!(std::mem::size_of::<TileViewerParams>(), 176);
        assert_eq!(std::mem::size_of::<TileViewerParams>() % 16, 0);
        assert_eq!(offset_of!(TileViewerParams, a), 0);
        assert_eq!(offset_of!(TileViewerParams, b), 16);
        assert_eq!(offset_of!(TileViewerParams, c), 32);
        assert_eq!(offset_of!(TileViewerParams, sub0), 48);
        assert_eq!(offset_of!(TileViewerParams, sub1), 64);
        assert_eq!(offset_of!(TileViewerParams, sub2), 80);
        assert_eq!(offset_of!(TileViewerParams, width), 96);
        assert_eq!(offset_of!(TileViewerParams, tex_res), 100);
        assert_eq!(offset_of!(TileViewerParams, terrain), 112);
        assert_eq!(std::mem::size_of::<TerrainGpu>(), 56);
    }

    #[test]
    fn params_default_is_zeroed_pod() {
        // Pod/Zeroable must hold (no uninit padding) so the zero-init buffer and
        // bytemuck round-trip are safe.
        let p = TileViewerParams::zeroed();
        assert_eq!(bytemuck::bytes_of(&p).len(), 176);
    }
}
