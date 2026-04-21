use std::time::Instant;

use godot::builtin::Vector3;
use godot::classes::rendering_device::{
    DataFormat, DriverResource, TextureSamples, TextureType, TextureUsageBits,
};
use godot::classes::{RdTextureFormat, RdTextureView, RenderingDevice, RenderingServer};
use godot::obj::{Gd, NewGd, Singleton};
use godot::prelude::Rid;

use crate::compute_utils::{on_render_thread_sync, ComputePipeline, RdSend};

pub const DEFAULT_SNAPSHOT_COLOR_SHADER: &str =
    "res://addons/celestial_sim/shaders/SnapshotPatchNoise.slang";
pub const DEFAULT_SNAPSHOT_NORMAL_SHADER: &str =
    "res://addons/celestial_sim/shaders/SnapshotPatchNormal.slang";

/// Maximum supported LOD levels (LOD0 cubemap + LOD1..LOD_MAX detail patches).
pub const MAX_LOD_LEVELS: usize = 4;

/// Minimum time interval between snapshot regenerations.
const UPDATE_INTERVAL: std::time::Duration = std::time::Duration::from_millis(100);

// ───────────────────────────────────────────────────────────────────
// Tangent-plane utilities
// ───────────────────────────────────────────────────────────────────

/// Builds an orthonormal tangent basis for a point on the unit sphere.
///
/// Returns `(center_dir, tangent_u, tangent_v)` where:
/// - `center_dir` is the normalized input direction (surface normal)
/// - `tangent_u` and `tangent_v` are two orthonormal vectors in the tangent plane
///
/// Uses the "up" vector `(0,1,0)` to derive the basis, falling back to `(0,0,1)`
/// when `dir` is near a pole to avoid degeneracy.
pub fn build_tangent_basis(dir: Vector3) -> (Vector3, Vector3, Vector3) {
    let n = dir.normalized();

    // Choose a reference vector that isn't parallel to n
    let up = if n.y.abs() > 0.99 {
        Vector3::new(0.0, 0.0, 1.0)
    } else {
        Vector3::new(0.0, 1.0, 0.0)
    };

    let tangent_u = up.cross(n).normalized();
    let tangent_v = n.cross(tangent_u).normalized();

    (n, tangent_u, tangent_v)
}

// ───────────────────────────────────────────────────────────────────
// LodLevel metadata
// ───────────────────────────────────────────────────────────────────

/// Metadata for a single LOD patch (LOD1+).
///
/// Each patch is defined by a tangent-plane basis centered on the sphere.
/// No cube-face dependency — the patch is anchored purely to a direction.
#[derive(Debug, Clone)]
pub struct CameraSnapshot {
    /// Center direction on the unit sphere (surface normal at patch center).
    pub center_dir: Vector3,
    /// First tangent-plane basis vector (orthogonal to center_dir).
    pub tangent_u: Vector3,
    /// Second tangent-plane basis vector (orthogonal to center_dir and tangent_u).
    pub tangent_v: Vector3,
    /// Angular half-extent of this patch (radians). LOD1=π/4, LOD2=π/8, etc.
    pub angular_extent: f32,
    /// Color RID on the main RD (front buffer — shader reads from this).
    pub main_texture_rid: Rid,
    /// Color RID on the local RD (front buffer).
    pub local_texture_rid: Rid,
    /// Normal RID on the main RD (front buffer).
    pub normal_main_texture_rid: Rid,
    /// Normal RID on the local RD (front buffer).
    pub normal_local_texture_rid: Rid,
    /// Color RID on the main RD (back buffer — worker writes here).
    pub back_main_texture_rid: Rid,
    /// Color RID on the local RD (back buffer — worker writes here).
    pub back_local_texture_rid: Rid,
    /// Normal RID on the main RD (back buffer).
    pub back_normal_main_texture_rid: Rid,
    /// Normal RID on the local RD (back buffer).
    pub back_normal_local_texture_rid: Rid,
    /// Per-level texture resolution (level 1 uses half of patch_resolution).
    pub resolution: u32,
}

impl CameraSnapshot {
    fn new_empty(lod_index: u32, resolution: u32) -> Self {
        // Each LOD covers half the angular extent of the previous.
        // LOD1 = π/8, LOD2 = π/16, LOD3 = π/32, LOD4 = π/64
        // (starts at half the cubemap's effective coverage)
        let angular_extent = std::f32::consts::FRAC_PI_4 / 2.0f32.powi(lod_index as i32);
        let (center_dir, tangent_u, tangent_v) = build_tangent_basis(Vector3::new(0.0, 0.0, 1.0));
        Self {
            center_dir,
            tangent_u,
            tangent_v,
            angular_extent,
            main_texture_rid: Rid::Invalid,
            local_texture_rid: Rid::Invalid,
            normal_main_texture_rid: Rid::Invalid,
            normal_local_texture_rid: Rid::Invalid,
            back_main_texture_rid: Rid::Invalid,
            back_local_texture_rid: Rid::Invalid,
            back_normal_main_texture_rid: Rid::Invalid,
            back_normal_local_texture_rid: Rid::Invalid,
            resolution,
        }
    }
}

// ───────────────────────────────────────────────────────────────────
// TextureLodChain
// ───────────────────────────────────────────────────────────────────

/// Manages a chain of textures: LOD0 is the cubemap, LOD1+ are 2D detail patches.
pub struct CameraSnapshotTexture {
    /// Number of extra LOD levels beyond the cubemap (0 means cubemap only).
    pub max_lod_levels: u32,
    /// Resolution of each LOD patch texture (square).
    pub patch_resolution: u32,
    /// LOD1+ metadata and texture RIDs.
    pub levels: Vec<CameraSnapshot>,
    /// Compute pipeline for generating color LOD patches.
    pipeline: Option<ComputePipeline>,
    /// Compute pipeline for generating normal LOD patches.
    normal_pipeline: Option<ComputePipeline>,
    /// Shader path for color patch generation.
    color_shader_path: String,
    /// Shader path for normal map generation (None = no normal maps).
    normal_shader_path: Option<String>,
    /// Extra uniform data bound at binding 7 (e.g. serialized TerrainParams).
    extra_params: Option<Vec<u8>>,
    /// Whether to show debug red borders on snapshot patches (binding 8).
    pub show_borders: bool,
    /// Last camera direction (normalized) used for update threshold check.
    last_camera_dir: Option<Vector3>,
    /// Timestamp of the last snapshot regeneration.
    last_update_time: Option<Instant>,
}

impl CameraSnapshotTexture {
    pub fn new(
        max_lod_levels: u32,
        patch_resolution: u32,
        color_shader_path: Option<String>,
        normal_shader_path: Option<String>,
    ) -> Self {
        let levels = (1..=max_lod_levels)
            .map(|i| CameraSnapshot::new_empty(i, patch_resolution))
            .collect();
        Self {
            max_lod_levels,
            patch_resolution,
            levels,
            pipeline: None,
            normal_pipeline: None,
            color_shader_path: color_shader_path
                .unwrap_or_else(|| DEFAULT_SNAPSHOT_COLOR_SHADER.to_string()),
            normal_shader_path,
            extra_params: None,
            show_borders: true,
            last_camera_dir: None,
            last_update_time: None,
        }
    }

    /// Sets extra uniform data that will be bound at binding 7 during patch dispatch.
    /// Pass serialized bytes (e.g. `bytemuck::bytes_of(&terrain_params).to_vec()`).
    pub fn set_extra_params(&mut self, data: Vec<u8>) {
        self.extra_params = Some(data);
    }

    /// Returns the angular half-extent (radians) for a given LOD level index (1-based).
    /// LOD1 = π/8, LOD2 = π/16, LOD3 = π/32, LOD4 = π/64
    pub fn angular_extent_for_level(level: u32) -> f32 {
        std::f32::consts::FRAC_PI_4 / 2.0f32.powi(level as i32)
    }

    /// Determines how many LOD levels should be active based on camera distance.
    /// Closer camera → more levels. Distance is normalized by planet radius.
    pub fn active_levels_for_distance(distance_ratio: f32, max_levels: u32) -> u32 {
        if distance_ratio > 3.0 {
            0
        } else if distance_ratio > 2.0 {
            1.min(max_levels)
        } else if distance_ratio > 1.5 {
            2.min(max_levels)
        } else if distance_ratio > 1.1 {
            3.min(max_levels)
        } else {
            max_levels
        }
    }

    /// Checks whether enough time has elapsed since the last snapshot update.
    pub fn needs_update(&self, _camera_dir: Vector3) -> bool {
        match self.last_update_time {
            None => true,
            Some(t) => t.elapsed() >= UPDATE_INTERVAL,
        }
    }

    /// Updates all LOD patch centers and tangent bases based on the camera's closest
    /// point on the sphere. No cube-face dependency.
    pub fn update_centers(&mut self, camera_pos: Vector3, _radius: f32) {
        let dir = camera_pos.normalized();
        let (center_dir, tangent_u, tangent_v) = build_tangent_basis(dir);

        for level in &mut self.levels {
            level.center_dir = center_dir;
            level.tangent_u = tangent_u;
            level.tangent_v = tangent_v;
        }

        self.last_camera_dir = Some(dir);
    }

    // ─── GPU texture management ──────────────────────────────────

    fn texture_2d_usage_bits() -> TextureUsageBits {
        TextureUsageBits::SAMPLING_BIT
            | TextureUsageBits::STORAGE_BIT
            | TextureUsageBits::CAN_UPDATE_BIT
            | TextureUsageBits::CAN_COPY_FROM_BIT
    }

    /// Allocates shared 2D textures (main RD + local RD) for all LOD levels.
    /// Creates DOUBLE-BUFFERED pairs: front (for rendering) and back (for compute writes).
    pub fn allocate_textures(&mut self, local_rd: &mut Gd<RenderingDevice>) {
        let has_normals = self.normal_shader_path.is_some();
        for level in &mut self.levels {
            let size = level.resolution;
            // Front buffer (main thread renders from this)
            let (main_rid, local_rid) = Self::create_shared_2d_texture(local_rd, size);
            level.main_texture_rid = main_rid;
            level.local_texture_rid = local_rid;
            // Back buffer (worker writes to this)
            let (back_main, back_local) = Self::create_shared_2d_texture(local_rd, size);
            level.back_main_texture_rid = back_main;
            level.back_local_texture_rid = back_local;

            if has_normals {
                let (normal_main_rid, normal_local_rid) =
                    Self::create_shared_2d_texture(local_rd, size);
                level.normal_main_texture_rid = normal_main_rid;
                level.normal_local_texture_rid = normal_local_rid;

                let (back_normal_main, back_normal_local) =
                    Self::create_shared_2d_texture(local_rd, size);
                level.back_normal_main_texture_rid = back_normal_main;
                level.back_normal_local_texture_rid = back_normal_local;
            }
        }
    }

    /// Swaps front and back buffers for all levels. Call on the main thread
    /// after receiving a snapshot-updated MeshResult so that the shader
    /// atomically sees the new texture data AND the new center_dir.
    pub fn swap_buffers(&mut self) {
        for level in &mut self.levels {
            std::mem::swap(
                &mut level.main_texture_rid,
                &mut level.back_main_texture_rid,
            );
            std::mem::swap(
                &mut level.local_texture_rid,
                &mut level.back_local_texture_rid,
            );
            std::mem::swap(
                &mut level.normal_main_texture_rid,
                &mut level.back_normal_main_texture_rid,
            );
            std::mem::swap(
                &mut level.normal_local_texture_rid,
                &mut level.back_normal_local_texture_rid,
            );
        }
    }

    fn create_shared_2d_texture(local_rd: &mut Gd<RenderingDevice>, size: u32) -> (Rid, Rid) {
        let driver_handle = on_render_thread_sync(move || {
            let mut main_rd = RenderingServer::singleton().get_rendering_device().unwrap();

            let mut format = RdTextureFormat::new_gd();
            format.set_texture_type(TextureType::TYPE_2D);
            format.set_format(DataFormat::R8G8B8A8_UNORM);
            format.set_width(size);
            format.set_height(size);
            format.set_depth(1);
            format.set_array_layers(1);
            format.set_mipmaps(1);
            format.set_samples(TextureSamples::SAMPLES_1);
            format.set_usage_bits(Self::texture_2d_usage_bits());

            let view = RdTextureView::new_gd();
            let main_rid = main_rd.texture_create(&format, &view);
            assert!(main_rid.is_valid(), "Failed to create main 2D texture");

            let handle = main_rd.get_driver_resource(DriverResource::TEXTURE, main_rid, 0);
            (main_rid, handle)
        });

        let (main_rid, handle) = driver_handle;

        let local_rd_send = RdSend(local_rd.clone());
        let local_rid = on_render_thread_sync(move || {
            let mut rd = local_rd_send;
            let rid = rd.0.texture_create_from_extension(
                TextureType::TYPE_2D,
                DataFormat::R8G8B8A8_UNORM,
                TextureSamples::SAMPLES_1,
                Self::texture_2d_usage_bits(),
                handle,
                size as u64,
                size as u64,
                1,
                1,
            );
            assert!(
                rid.is_valid(),
                "Failed to create local 2D texture from extension"
            );
            rid
        });

        (main_rid, local_rid)
    }

    // ─── Compute pipeline / dispatch ─────────────────────────────

    pub fn init_pipeline(&mut self, rd: &mut Gd<RenderingDevice>) {
        if self.pipeline.is_none() {
            self.pipeline = Some(ComputePipeline::new(rd, &self.color_shader_path));
        }
        if self.normal_pipeline.is_none() {
            if let Some(ref path) = self.normal_shader_path {
                self.normal_pipeline = Some(ComputePipeline::new(rd, path));
            }
        }
    }

    /// Generates a single LOD patch using the compute shader.
    ///
    /// Shader bindings (tangent-plane model):
    ///   0 — `RWTexture2D<float4>` (IMAGE)
    ///   1 — `tex_size`        (UNIFORM_BUFFER, u32)
    ///   2 — `radius`          (UNIFORM_BUFFER, f32)
    ///   3 — `center_dir`      (UNIFORM_BUFFER, float3 as [f32;3], padded to 16 bytes)
    ///   4 — `tangent_u`       (UNIFORM_BUFFER, float3 as [f32;3], padded to 16 bytes)
    ///   5 — `tangent_v`       (UNIFORM_BUFFER, float3 as [f32;3], padded to 16 bytes)
    ///   6 — `angular_extent`  (UNIFORM_BUFFER, f32)
    ///   7 — `extra_params`    (UNIFORM_BUFFER, optional — e.g. TerrainParams)
    pub fn generate_snapshot_patch(
        &self,
        rd: &mut Gd<RenderingDevice>,
        level: &CameraSnapshot,
        radius: f32,
    ) {
        let pipeline = self
            .pipeline
            .as_ref()
            .expect("CameraSnapshotTexture pipeline not initialised; call init_pipeline first");

        Self::dispatch_patch_shader(
            rd,
            pipeline,
            level.back_local_texture_rid,
            level,
            radius,
            level.resolution,
            &self.extra_params,
            self.show_borders,
        );
    }

    /// Generates a single normal LOD patch using the normal compute shader.
    pub fn generate_normal_snapshot(
        &self,
        rd: &mut Gd<RenderingDevice>,
        level: &CameraSnapshot,
        radius: f32,
    ) {
        let pipeline = self.normal_pipeline.as_ref().expect(
            "CameraSnapshotTexture normal pipeline not initialised; call init_pipeline first",
        );

        Self::dispatch_patch_shader(
            rd,
            pipeline,
            level.back_normal_local_texture_rid,
            level,
            radius,
            level.resolution,
            &self.extra_params,
            self.show_borders,
        );
    }

    /// Common dispatch logic shared by color and normal patch generation.
    fn dispatch_patch_shader(
        rd: &mut Gd<RenderingDevice>,
        pipeline: &ComputePipeline,
        texture_rid: Rid,
        level: &CameraSnapshot,
        radius: f32,
        patch_resolution: u32,
        extra_params: &Option<Vec<u8>>,
        show_borders: bool,
    ) {
        let size = patch_resolution;
        let workgroups_x = (size + 7) / 8;
        let workgroups_y = (size + 7) / 8;

        let center_dir = level.center_dir;
        let tangent_u = level.tangent_u;
        let tangent_v = level.tangent_v;
        let angular_extent = level.angular_extent;
        let shader = pipeline.shader();
        let pipe = pipeline.pipeline();
        let extra_data = extra_params.clone();
        let show_borders_val: u32 = if show_borders { 1 } else { 0 };

        let rd_send = RdSend(rd.clone());

        on_render_thread_sync(move || {
            let mut rd = rd_send;

            let size_bytes = size.to_le_bytes();
            let radius_bytes = radius.to_le_bytes();
            // Pack vec3 as 3 floats (12 bytes), will be padded to 16 by create_uniform_buffer_raw
            let center_dir_bytes = vec3_to_bytes(center_dir);
            let tangent_u_bytes = vec3_to_bytes(tangent_u);
            let tangent_v_bytes = vec3_to_bytes(tangent_v);
            let angular_extent_bytes = angular_extent.to_le_bytes();

            let size_rid = create_uniform_buffer_raw(&mut rd.0, &size_bytes);
            let radius_rid = create_uniform_buffer_raw(&mut rd.0, &radius_bytes);
            let center_dir_rid = create_uniform_buffer_raw(&mut rd.0, &center_dir_bytes);
            let tangent_u_rid = create_uniform_buffer_raw(&mut rd.0, &tangent_u_bytes);
            let tangent_v_rid = create_uniform_buffer_raw(&mut rd.0, &tangent_v_bytes);
            let angular_extent_rid = create_uniform_buffer_raw(&mut rd.0, &angular_extent_bytes);

            use godot::classes::rendering_device::UniformType;
            use godot::classes::RdUniform;
            use godot::obj::NewGd;
            use godot::prelude::Array;

            let mut tex_uniform = RdUniform::new_gd();
            tex_uniform.set_binding(0);
            tex_uniform.set_uniform_type(UniformType::IMAGE);
            tex_uniform.add_id(texture_rid);

            let mut size_uniform = RdUniform::new_gd();
            size_uniform.set_binding(1);
            size_uniform.set_uniform_type(UniformType::UNIFORM_BUFFER);
            size_uniform.add_id(size_rid);

            let mut radius_uniform = RdUniform::new_gd();
            radius_uniform.set_binding(2);
            radius_uniform.set_uniform_type(UniformType::UNIFORM_BUFFER);
            radius_uniform.add_id(radius_rid);

            let mut center_dir_uniform = RdUniform::new_gd();
            center_dir_uniform.set_binding(3);
            center_dir_uniform.set_uniform_type(UniformType::UNIFORM_BUFFER);
            center_dir_uniform.add_id(center_dir_rid);

            let mut tangent_u_uniform = RdUniform::new_gd();
            tangent_u_uniform.set_binding(4);
            tangent_u_uniform.set_uniform_type(UniformType::UNIFORM_BUFFER);
            tangent_u_uniform.add_id(tangent_u_rid);

            let mut tangent_v_uniform = RdUniform::new_gd();
            tangent_v_uniform.set_binding(5);
            tangent_v_uniform.set_uniform_type(UniformType::UNIFORM_BUFFER);
            tangent_v_uniform.add_id(tangent_v_rid);

            let mut angular_extent_uniform = RdUniform::new_gd();
            angular_extent_uniform.set_binding(6);
            angular_extent_uniform.set_uniform_type(UniformType::UNIFORM_BUFFER);
            angular_extent_uniform.add_id(angular_extent_rid);

            let mut uniforms = Array::new();
            uniforms.push(&tex_uniform);
            uniforms.push(&size_uniform);
            uniforms.push(&radius_uniform);
            uniforms.push(&center_dir_uniform);
            uniforms.push(&tangent_u_uniform);
            uniforms.push(&tangent_v_uniform);
            uniforms.push(&angular_extent_uniform);

            // Binding 7: extra params (optional — e.g. TerrainParams)
            let extra_rid = if let Some(ref data) = extra_data {
                let rid = create_uniform_buffer_raw(&mut rd.0, data);
                let mut extra_uniform = RdUniform::new_gd();
                extra_uniform.set_binding(7);
                extra_uniform.set_uniform_type(UniformType::UNIFORM_BUFFER);
                extra_uniform.add_id(rid);
                uniforms.push(&extra_uniform);
                Some(rid)
            } else {
                None
            };

            // Binding 8: show_border flag
            let show_border_rid =
                create_uniform_buffer_raw(&mut rd.0, &show_borders_val.to_le_bytes());
            let mut show_border_uniform = RdUniform::new_gd();
            show_border_uniform.set_binding(8);
            show_border_uniform.set_uniform_type(UniformType::UNIFORM_BUFFER);
            show_border_uniform.add_id(show_border_rid);
            uniforms.push(&show_border_uniform);

            let uniform_set = rd.0.uniform_set_create(&uniforms, shader, 0);
            assert!(
                uniform_set.is_valid(),
                "Failed to create snapshot patch uniform set"
            );

            let compute_list = rd.0.compute_list_begin();
            rd.0.compute_list_bind_compute_pipeline(compute_list, pipe);
            rd.0.compute_list_bind_uniform_set(compute_list, uniform_set, 0);
            rd.0.compute_list_dispatch(compute_list, workgroups_x, workgroups_y, 1);
            rd.0.compute_list_end();
            rd.0.submit();
            rd.0.sync();

            rd.0.free_rid(uniform_set);
            rd.0.free_rid(size_rid);
            rd.0.free_rid(radius_rid);
            rd.0.free_rid(center_dir_rid);
            rd.0.free_rid(tangent_u_rid);
            rd.0.free_rid(tangent_v_rid);
            rd.0.free_rid(angular_extent_rid);
            if let Some(rid) = extra_rid {
                rd.0.free_rid(rid);
            }
            rd.0.free_rid(show_border_rid);
        });
    }

    /// Generates all active LOD patches into BACK buffers (color and optionally normal).
    pub fn generate_all(&self, rd: &mut Gd<RenderingDevice>, radius: f32) {
        for level in &self.levels {
            if level.back_local_texture_rid.is_valid() {
                self.generate_snapshot_patch(rd, level, radius);
            }
            if self.normal_pipeline.is_some() && level.back_normal_local_texture_rid.is_valid() {
                self.generate_normal_snapshot(rd, level, radius);
            }
        }
    }

    /// Force-regenerate all snapshot textures immediately, bypassing the time
    /// throttle.  Call when texture parameters change (e.g. terrain params
    /// modified in the inspector).
    pub fn force_regenerate(&mut self, rd: &mut Gd<RenderingDevice>, radius: f32) {
        self.generate_all(rd, radius);
        self.last_update_time = Some(Instant::now());
    }

    /// Full update: every UPDATE_INTERVAL (100ms), move patches to the current
    /// camera direction and regenerate textures. Between updates the patches
    /// stay fixed on the sphere surface like "snapshots from the past".
    /// Returns true if textures were regenerated.
    pub fn update(
        &mut self,
        rd: &mut Gd<RenderingDevice>,
        camera_pos: Vector3,
        radius: f32,
    ) -> bool {
        let dir = camera_pos.normalized();

        if !self.needs_update(dir) {
            return false;
        }

        // Move patches to current camera direction and regenerate
        self.update_centers(camera_pos, radius);
        self.generate_all(rd, radius);
        self.last_update_time = Some(Instant::now());
        true
    }

    // ─── Cleanup ─────────────────────────────────────────────────

    pub fn dispose_local(&mut self, local_rd: &mut Gd<RenderingDevice>) {
        for level in &mut self.levels {
            // Front buffers
            if level.local_texture_rid.is_valid() {
                crate::compute_utils::free_rid_on_render_thread(local_rd, level.local_texture_rid);
                level.local_texture_rid = Rid::Invalid;
            }
            if level.main_texture_rid.is_valid() {
                let old_main = level.main_texture_rid;
                on_render_thread_sync(move || {
                    let mut main_rd = RenderingServer::singleton().get_rendering_device().unwrap();
                    main_rd.free_rid(old_main);
                });
                level.main_texture_rid = Rid::Invalid;
            }
            if level.normal_local_texture_rid.is_valid() {
                crate::compute_utils::free_rid_on_render_thread(
                    local_rd,
                    level.normal_local_texture_rid,
                );
                level.normal_local_texture_rid = Rid::Invalid;
            }
            if level.normal_main_texture_rid.is_valid() {
                let old_normal_main = level.normal_main_texture_rid;
                on_render_thread_sync(move || {
                    let mut main_rd = RenderingServer::singleton().get_rendering_device().unwrap();
                    main_rd.free_rid(old_normal_main);
                });
                level.normal_main_texture_rid = Rid::Invalid;
            }
            // Back buffers
            if level.back_local_texture_rid.is_valid() {
                crate::compute_utils::free_rid_on_render_thread(
                    local_rd,
                    level.back_local_texture_rid,
                );
                level.back_local_texture_rid = Rid::Invalid;
            }
            if level.back_main_texture_rid.is_valid() {
                let old_back = level.back_main_texture_rid;
                on_render_thread_sync(move || {
                    let mut main_rd = RenderingServer::singleton().get_rendering_device().unwrap();
                    main_rd.free_rid(old_back);
                });
                level.back_main_texture_rid = Rid::Invalid;
            }
            if level.back_normal_local_texture_rid.is_valid() {
                crate::compute_utils::free_rid_on_render_thread(
                    local_rd,
                    level.back_normal_local_texture_rid,
                );
                level.back_normal_local_texture_rid = Rid::Invalid;
            }
            if level.back_normal_main_texture_rid.is_valid() {
                let old_back_normal = level.back_normal_main_texture_rid;
                on_render_thread_sync(move || {
                    let mut main_rd = RenderingServer::singleton().get_rendering_device().unwrap();
                    main_rd.free_rid(old_back_normal);
                });
                level.back_normal_main_texture_rid = Rid::Invalid;
            }
        }
        if let Some(ref mut pipeline) = self.pipeline {
            pipeline.dispose_direct(local_rd);
        }
        self.pipeline = None;
        if let Some(ref mut pipeline) = self.normal_pipeline {
            pipeline.dispose_direct(local_rd);
        }
        self.normal_pipeline = None;
    }

    pub fn dispose_direct(
        &mut self,
        main_rd: &mut Gd<RenderingDevice>,
        local_rd: &mut Gd<RenderingDevice>,
    ) {
        for level in &mut self.levels {
            // Front buffers
            if level.local_texture_rid.is_valid() {
                crate::compute_utils::free_rid_on_render_thread(local_rd, level.local_texture_rid);
                level.local_texture_rid = Rid::Invalid;
            }
            if level.main_texture_rid.is_valid() {
                crate::compute_utils::free_rid_on_render_thread(main_rd, level.main_texture_rid);
                level.main_texture_rid = Rid::Invalid;
            }
            if level.normal_local_texture_rid.is_valid() {
                crate::compute_utils::free_rid_on_render_thread(
                    local_rd,
                    level.normal_local_texture_rid,
                );
                level.normal_local_texture_rid = Rid::Invalid;
            }
            if level.normal_main_texture_rid.is_valid() {
                crate::compute_utils::free_rid_on_render_thread(
                    main_rd,
                    level.normal_main_texture_rid,
                );
                level.normal_main_texture_rid = Rid::Invalid;
            }
            // Back buffers
            if level.back_local_texture_rid.is_valid() {
                crate::compute_utils::free_rid_on_render_thread(
                    local_rd,
                    level.back_local_texture_rid,
                );
                level.back_local_texture_rid = Rid::Invalid;
            }
            if level.back_main_texture_rid.is_valid() {
                crate::compute_utils::free_rid_on_render_thread(
                    main_rd,
                    level.back_main_texture_rid,
                );
                level.back_main_texture_rid = Rid::Invalid;
            }
            if level.back_normal_local_texture_rid.is_valid() {
                crate::compute_utils::free_rid_on_render_thread(
                    local_rd,
                    level.back_normal_local_texture_rid,
                );
                level.back_normal_local_texture_rid = Rid::Invalid;
            }
            if level.back_normal_main_texture_rid.is_valid() {
                crate::compute_utils::free_rid_on_render_thread(
                    main_rd,
                    level.back_normal_main_texture_rid,
                );
                level.back_normal_main_texture_rid = Rid::Invalid;
            }
        }
        if let Some(ref mut pipeline) = self.pipeline {
            pipeline.dispose_direct(local_rd);
        }
        self.pipeline = None;
        if let Some(ref mut pipeline) = self.normal_pipeline {
            pipeline.dispose_direct(local_rd);
        }
        self.normal_pipeline = None;
    }

    /// Returns true if any color LOD texture has been allocated.
    pub fn has_textures(&self) -> bool {
        self.levels.iter().any(|l| l.local_texture_rid.is_valid())
    }

    /// Returns true if any normal LOD texture has been allocated.
    pub fn has_normal_textures(&self) -> bool {
        self.levels
            .iter()
            .any(|l| l.normal_local_texture_rid.is_valid())
    }

    /// Returns the main-RD color RID for a given LOD level (0-based index into levels,
    /// i.e. index 0 = LOD1).
    pub fn main_texture_rid(&self, index: usize) -> Rid {
        self.levels
            .get(index)
            .map_or(Rid::Invalid, |l| l.main_texture_rid)
    }

    /// Returns the main-RD normal RID for a given LOD level (0-based index into levels,
    /// i.e. index 0 = LOD1).
    pub fn normal_main_texture_rid(&self, index: usize) -> Rid {
        self.levels
            .get(index)
            .map_or(Rid::Invalid, |l| l.normal_main_texture_rid)
    }
}

/// Serializes a Vector3 to 12 bytes (3 × f32 little-endian).
fn vec3_to_bytes(v: Vector3) -> [u8; 12] {
    let mut buf = [0u8; 12];
    buf[0..4].copy_from_slice(&v.x.to_le_bytes());
    buf[4..8].copy_from_slice(&v.y.to_le_bytes());
    buf[8..12].copy_from_slice(&v.z.to_le_bytes());
    buf
}

/// Creates a uniform buffer directly on the given RD.
fn create_uniform_buffer_raw(rd: &mut Gd<RenderingDevice>, data: &[u8]) -> Rid {
    use godot::prelude::PackedByteArray;

    let padded_size = 16.max((data.len() + 15) / 16 * 16);
    let mut padded = vec![0u8; padded_size];
    padded[..data.len()].copy_from_slice(data);

    let mut pba = PackedByteArray::new();
    pba.extend(padded.iter().copied());
    let rid = rd
        .uniform_buffer_create_ex(padded_size as u32)
        .data(&pba)
        .done();
    assert!(rid.is_valid(), "Failed to create uniform buffer");
    rid
}

// ─────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_snapshot_chain_default() {
        let chain = CameraSnapshotTexture::new(0, 512, None, None);
        assert_eq!(chain.levels.len(), 0);
        assert_eq!(chain.max_lod_levels, 0);
    }

    #[test]
    fn test_snapshot_chain_with_levels() {
        let chain = CameraSnapshotTexture::new(4, 512, None, None);
        assert_eq!(chain.levels.len(), 4);
        assert_eq!(chain.max_lod_levels, 4);
    }

    #[test]
    fn test_angular_extent_for_level() {
        use std::f32::consts::FRAC_PI_4;
        // LOD1 = π/4, LOD2 = π/8, LOD3 = π/16, LOD4 = π/32
        // LOD1 = π/8, LOD2 = π/16, LOD3 = π/32, LOD4 = π/64
        assert!(
            (CameraSnapshotTexture::angular_extent_for_level(1) - FRAC_PI_4 / 2.0).abs() < 1e-6
        );
        assert!(
            (CameraSnapshotTexture::angular_extent_for_level(2) - FRAC_PI_4 / 4.0).abs() < 1e-6
        );
        assert!(
            (CameraSnapshotTexture::angular_extent_for_level(3) - FRAC_PI_4 / 8.0).abs() < 1e-6
        );
        assert!(
            (CameraSnapshotTexture::angular_extent_for_level(4) - FRAC_PI_4 / 16.0).abs() < 1e-6
        );
    }

    #[test]
    fn test_lod_level_extents_match_new_empty() {
        let chain = CameraSnapshotTexture::new(4, 512, None, None);
        for (i, level) in chain.levels.iter().enumerate() {
            let expected = CameraSnapshotTexture::angular_extent_for_level((i + 1) as u32);
            assert!(
                (level.angular_extent - expected).abs() < 1e-6,
                "Level {} extent mismatch: {} vs {}",
                i + 1,
                level.angular_extent,
                expected
            );
        }
    }

    #[test]
    fn test_build_tangent_basis_orthonormal() {
        let dir = Vector3::new(1.0, 0.5, -0.3).normalized();
        let (n, u, v) = build_tangent_basis(dir);

        // n should equal normalized dir
        assert!((n - dir).length() < 1e-5);
        // All unit length
        assert!((n.length() - 1.0).abs() < 1e-5);
        assert!((u.length() - 1.0).abs() < 1e-5);
        assert!((v.length() - 1.0).abs() < 1e-5);
        // Mutually perpendicular
        assert!(n.dot(u).abs() < 1e-5, "n·u = {}", n.dot(u));
        assert!(n.dot(v).abs() < 1e-5, "n·v = {}", n.dot(v));
        assert!(u.dot(v).abs() < 1e-5, "u·v = {}", u.dot(v));
    }

    #[test]
    fn test_build_tangent_basis_various_directions() {
        let dirs = [
            Vector3::new(1.0, 0.0, 0.0),   // +X axis
            Vector3::new(-1.0, 0.0, 0.0),  // -X axis
            Vector3::new(0.0, 1.0, 0.0),   // +Y pole
            Vector3::new(0.0, -1.0, 0.0),  // -Y pole
            Vector3::new(0.0, 0.0, 1.0),   // +Z axis
            Vector3::new(0.0, 0.0, -1.0),  // -Z axis
            Vector3::new(1.0, 1.0, 1.0),   // diagonal
            Vector3::new(-0.5, 0.8, -0.3), // arbitrary
        ];
        for dir in dirs {
            let dir = dir.normalized();
            let (n, u, v) = build_tangent_basis(dir);

            assert!((n.length() - 1.0).abs() < 1e-4, "dir={:?}: n not unit", dir);
            assert!((u.length() - 1.0).abs() < 1e-4, "dir={:?}: u not unit", dir);
            assert!((v.length() - 1.0).abs() < 1e-4, "dir={:?}: v not unit", dir);
            assert!(n.dot(u).abs() < 1e-4, "dir={:?}: n·u = {}", dir, n.dot(u));
            assert!(n.dot(v).abs() < 1e-4, "dir={:?}: n·v = {}", dir, n.dot(v));
            assert!(u.dot(v).abs() < 1e-4, "dir={:?}: u·v = {}", dir, u.dot(v));
        }
    }

    #[test]
    fn test_update_centers_sets_tangent_basis() {
        let mut chain = CameraSnapshotTexture::new(2, 512, None, None);
        chain.update_centers(Vector3::new(3.0, 0.0, 0.0), 1.0);

        for level in &chain.levels {
            // center_dir should point in +X direction
            assert!(
                (level.center_dir - Vector3::new(1.0, 0.0, 0.0)).length() < 1e-4,
                "center_dir = {:?}",
                level.center_dir
            );
            // Tangent basis should be orthonormal
            assert!(level.center_dir.dot(level.tangent_u).abs() < 1e-4);
            assert!(level.center_dir.dot(level.tangent_v).abs() < 1e-4);
            assert!(level.tangent_u.dot(level.tangent_v).abs() < 1e-4);
        }
    }

    #[test]
    fn test_update_centers_at_pole() {
        let mut chain = CameraSnapshotTexture::new(2, 512, None, None);
        chain.update_centers(Vector3::new(0.0, 5.0, 0.0), 1.0);

        for level in &chain.levels {
            // center_dir should point in +Y direction
            assert!(
                (level.center_dir - Vector3::new(0.0, 1.0, 0.0)).length() < 1e-4,
                "center_dir = {:?}",
                level.center_dir
            );
            // Basis should still be orthonormal even at pole
            assert!(level.center_dir.dot(level.tangent_u).abs() < 1e-4);
            assert!(level.center_dir.dot(level.tangent_v).abs() < 1e-4);
            assert!(level.tangent_u.dot(level.tangent_v).abs() < 1e-4);
        }
    }

    #[test]
    fn test_needs_update_no_previous() {
        let chain = CameraSnapshotTexture::new(4, 512, None, None);
        assert!(chain.needs_update(Vector3::new(1.0, 0.0, 0.0)));
    }

    #[test]
    fn test_needs_update_small_move() {
        let mut chain = CameraSnapshotTexture::new(4, 512, None, None);
        // Just updated — should not need update yet
        chain.last_camera_dir = Some(Vector3::new(1.0, 0.0, 0.0));
        chain.last_update_time = Some(Instant::now());
        assert!(!chain.needs_update(Vector3::new(1.0, 0.001, 0.0).normalized()));
    }

    #[test]
    fn test_needs_update_large_move() {
        let mut chain = CameraSnapshotTexture::new(4, 512, None, None);
        // Old update time — should need update
        chain.last_camera_dir = Some(Vector3::new(1.0, 0.0, 0.0));
        chain.last_update_time = Some(Instant::now() - std::time::Duration::from_secs(10));
        assert!(chain.needs_update(Vector3::new(0.0, 1.0, 0.0)));
    }

    #[test]
    fn test_active_levels_far() {
        assert_eq!(CameraSnapshotTexture::active_levels_for_distance(5.0, 4), 0);
    }

    #[test]
    fn test_active_levels_medium() {
        assert_eq!(CameraSnapshotTexture::active_levels_for_distance(2.5, 4), 1);
    }

    #[test]
    fn test_active_levels_close() {
        assert_eq!(CameraSnapshotTexture::active_levels_for_distance(1.2, 4), 3);
    }

    #[test]
    fn test_active_levels_very_close() {
        assert_eq!(
            CameraSnapshotTexture::active_levels_for_distance(1.05, 4),
            4
        );
    }

    #[test]
    fn test_active_levels_capped() {
        assert_eq!(
            CameraSnapshotTexture::active_levels_for_distance(1.05, 2),
            2
        );
    }

    #[test]
    fn test_dispose_clears_rids() {
        let chain = CameraSnapshotTexture::new(3, 512, None, None);
        for level in &chain.levels {
            assert!(!level.main_texture_rid.is_valid());
            assert!(!level.local_texture_rid.is_valid());
            assert!(!level.normal_main_texture_rid.is_valid());
            assert!(!level.normal_local_texture_rid.is_valid());
        }
        assert!(!chain.has_textures());
        assert!(!chain.has_normal_textures());
    }

    #[test]
    fn test_vec3_to_bytes_roundtrip() {
        let v = Vector3::new(1.5, -2.3, 0.7);
        let bytes = vec3_to_bytes(v);
        let x = f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        let y = f32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
        let z = f32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]);
        assert!((x - 1.5).abs() < 1e-6);
        assert!((y - (-2.3)).abs() < 1e-6);
        assert!((z - 0.7).abs() < 1e-6);
    }

    #[test]
    fn test_normal_shader_path_stored() {
        let chain =
            CameraSnapshotTexture::new(2, 256, None, Some("res://test_normal.slang".to_string()));
        assert_eq!(
            chain.normal_shader_path.as_deref(),
            Some("res://test_normal.slang")
        );
        assert!(chain.normal_pipeline.is_none());
    }

    #[test]
    fn test_no_normal_shader_path() {
        let chain = CameraSnapshotTexture::new(2, 256, None, None);
        assert!(chain.normal_shader_path.is_none());
        assert!(chain.normal_pipeline.is_none());
    }

    #[test]
    fn test_normal_main_texture_rid_default() {
        let chain = CameraSnapshotTexture::new(2, 256, None, None);
        assert!(!chain.normal_main_texture_rid(0).is_valid());
        assert!(!chain.normal_main_texture_rid(1).is_valid());
        assert!(!chain.normal_main_texture_rid(99).is_valid());
    }
}
