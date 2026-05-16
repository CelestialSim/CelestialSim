//! Runtime scatter-layer dispatcher (triangle-indexed, blue-noise dithered).
//!
//! Owns a `ComputePipeline` for `ScatterPlacement.slang` plus three GPU
//! buffers: the growable transform output, a 4-byte atomic counter, and a
//! once-uploaded blue-noise volume (R8 packed 4-bytes-per-uint).
//!
//! Per dispatch:
//!   - clear the counter to 0
//!   - bind everything (10 buffers, see binding map below)
//!   - run `ceil(state.n_tris / 64)` workgroups
//!   - read back the counter (4 bytes)
//!   - read back the first `count * 12` floats from the transform buffer
//!
//! The counter feeds `MultiMesh.set_instance_count(...)` so every visible
//! instance corresponds to a real spawn (no zero-pad slots).
//!
//! Shader binding layout (keep in sync with `ScatterPlacement.slang`):
//!   0: `ConstantBuffer<ScatterParams>`
//!   1: `ConstantBuffer<TerrainParams>`
//!   2: `RWStructuredBuffer<float>` out_transforms
//!   3: `RWStructuredBuffer<uint>`  out_counter
//!   4: `StructuredBuffer<float4>`  v_pos          (CesState)
//!   5: `StructuredBuffer<int4>`    t_abc          (CesState)
//!   6: `StructuredBuffer<int>`     t_lv           (CesState)
//!   7: `StructuredBuffer<int>`     t_deactivated  (CesState)
//!   8: `ConstantBuffer<uint>`      n_tris         (CesState's u_n_tris)
//!   9: `StructuredBuffer<uint>`    blue_noise     (packed bytes)
//!  10: `StructuredBuffer<uint2>`   t_tri_id       (CesState — stable per-tri u64)
//!  11: `RWStructuredBuffer<uint2>` out_tri_ids    (parallel to out_buffer; u64 per slot)

use crate::buffer_info::BufferInfo;
use crate::compute_utils;
use crate::compute_utils::ComputePipeline;
use crate::state::CesState;
use crate::texture_gen::TerrainParams;
use godot::classes::RenderingDevice;
use godot::obj::Gd;

pub const DEFAULT_SCATTER_SHADER_PATH: &str =
    "res://addons/celestial_sim/shaders/ScatterPlacement.slang";

/// Floats per instance in the MultiMesh 3D layout (mat3x4, row-major).
const FLOATS_PER_INSTANCE: u32 = 12;

/// Mirrors `ScatterParams` in `ScatterPlacement.slang`. 64 bytes, 16-aligned.
///
/// `variant_id` / `variant_count` drive the multi-variant filter: when
/// `variant_count > 1`, the shader hashes each triangle's centroid direction
/// (`hash3(dir * 137.0 + seed).x`) and only emits triangles whose hashed
/// bucket equals `variant_id`. Hashing the centroid direction (instead of
/// `tri_idx`) keeps the assignment stable across LOD compaction — see
/// `ScatterPlacement.slang` for the rationale. Default: `variant_id = 0`,
/// `variant_count = 1` (filter is a no-op for the single-mesh case).
#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ScatterParams {
    pub planet_radius: f32,
    pub seed: u32,
    pub subdivision_level: i32,
    pub noise_strength: f32,

    pub height_min: f32,
    pub height_max: f32,
    pub albedo_tolerance: f32,
    pub noise_dim: u32,

    pub albedo_target: [f32; 3],
    pub _pad: f32,

    pub variant_id: u32,
    pub variant_count: u32,
    pub _pad1: u32,
    pub _pad2: u32,
}

impl Default for ScatterParams {
    fn default() -> Self {
        Self {
            planet_radius: 1.0,
            seed: 0,
            subdivision_level: 5,
            noise_strength: 1.0,

            height_min: 0.0,
            height_max: 1.0,
            // Tolerance >= sqrt(3) disables the albedo similarity check.
            albedo_tolerance: 2.0,
            noise_dim: 32,

            albedo_target: [0.25, 0.4, 0.15],
            _pad: 0.0,

            variant_id: 0,
            variant_count: 1,
            _pad1: 0,
            _pad2: 0,
        }
    }
}

/// Owns the scatter compute pipeline and its GPU buffers.
pub struct CesScatterRuntime {
    pub(crate) pipeline: Option<ComputePipeline>,
    out_buffer: Option<BufferInfo>,
    /// Parallel to `out_buffer`. One `u64` per slot; the shader writes the
    /// source triangle's stable `tri_id` here so the CPU can diff
    /// (added/removed) across LOD iterations. Allocated only when
    /// `uses_incremental` is true.
    out_tri_ids_buffer: Option<BufferInfo>,
    counter_buffer: Option<BufferInfo>,
    /// Blue-noise R8 volume packed 4 bytes per `uint`. Uploaded once on first
    /// dispatch from the bytes stored in `noise_bytes`.
    noise_buffer: Option<BufferInfo>,
    capacity_instances: u32,
    /// Debug/perf toggle used to test whether oversized output buffers are the
    /// main readback cost. When enabled, every dispatch frees and reallocates
    /// the output buffer to `next_capacity(n_tris)` before running.
    force_shrink_each_dispatch: bool,
    params: ScatterParams,
    shader_path: String,
    /// CPU-side blue-noise bytes, set via `set_blue_noise` before the first
    /// dispatch. Empty until the texture layer has loaded it.
    noise_bytes: Vec<u8>,
    /// When true, the runtime allocates `out_tri_ids_buffer`, reads it back
    /// alongside the transform buffer, and emits per-variant
    /// (added, removed) diffs against `previous`. When false, it behaves
    /// exactly like before (flat-buffer output, no diffing) and the main
    /// thread uses MultiMesh.set_buffer.
    uses_incremental: bool,
    /// Sorted-by-tri_id (tri_id, transform) snapshot from the previous
    /// dispatch, one entry per variant. Used to compute added/removed sets.
    previous: Vec<Vec<(u64, [f32; 12])>>,
}

impl CesScatterRuntime {
    pub fn new(shader_path: &str) -> Self {
        Self {
            pipeline: None,
            out_buffer: None,
            out_tri_ids_buffer: None,
            counter_buffer: None,
            noise_buffer: None,
            capacity_instances: 0,
            force_shrink_each_dispatch: false,
            params: ScatterParams::default(),
            shader_path: shader_path.to_string(),
            noise_bytes: Vec::new(),
            uses_incremental: false,
            previous: Vec::new(),
        }
    }

    /// Enables (or disables) the incremental tri_id readback + diff path.
    /// When called with `true`, `previous` is sized to `variant_count` empty
    /// snapshots so the first dispatch reports every spawn as "added".
    pub fn set_incremental(&mut self, enabled: bool, variant_count: usize) {
        self.uses_incremental = enabled;
        if enabled {
            self.previous.resize_with(variant_count, Vec::new);
        } else {
            self.previous.clear();
        }
    }

    pub fn uses_incremental(&self) -> bool {
        self.uses_incremental
    }

    pub fn with_default_shader() -> Self {
        Self::new(DEFAULT_SCATTER_SHADER_PATH)
    }

    pub fn init_pipeline(&mut self, rd: &mut Gd<RenderingDevice>) {
        if self.pipeline.is_none() {
            self.pipeline = Some(ComputePipeline::new(rd, &self.shader_path));
        }
    }

    pub fn set_params(&mut self, params: ScatterParams) {
        self.params = params;
    }

    pub fn params(&self) -> &ScatterParams {
        &self.params
    }

    pub fn has_pipeline(&self) -> bool {
        self.pipeline.is_some()
    }

    /// Stores the blue-noise bytes for upload on the next dispatch. Bytes
    /// must equal `noise_dim^3` in length.
    pub fn set_blue_noise(&mut self, bytes: Vec<u8>, noise_dim: u32) {
        self.noise_bytes = bytes;
        self.params.noise_dim = noise_dim;
        // Force re-upload by dropping the existing buffer; the next dispatch
        // will recreate it.
        self.noise_buffer = None;
    }

    /// Number of floats the next `readback_compact` returns when called with
    /// `count = capacity_instances`.
    pub fn capacity_instances(&self) -> u32 {
        self.capacity_instances
    }

    pub fn set_force_shrink_each_dispatch(&mut self, value: bool) {
        self.force_shrink_each_dispatch = value;
    }

    /// Ensures the output transform buffer holds at least `n_tris` instances'
    /// worth of storage (so worst-case all-pass dispatch doesn't overflow).
    pub fn ensure_capacity(&mut self, rd: &mut Gd<RenderingDevice>, n_tris: u32) {
        let Some(new_capacity) = requested_output_capacity(
            self.out_buffer.is_some(),
            self.capacity_instances,
            n_tris,
            self.force_shrink_each_dispatch,
        ) else {
            return;
        };

        let byte_size = new_capacity
            .saturating_mul(FLOATS_PER_INSTANCE)
            .saturating_mul(4);

        if let Some(old) = self.out_buffer.take() {
            compute_utils::free_rid_on_render_thread(rd, old.rid);
        }

        let buf = compute_utils::create_empty_storage_buffer(rd, byte_size);
        self.out_buffer = Some(buf);
        self.capacity_instances = new_capacity;

        // Keep the parallel tri_ids buffer in lockstep with `out_buffer` when
        // incremental mode is enabled. 8 bytes per instance (u64).
        if self.uses_incremental {
            if let Some(old) = self.out_tri_ids_buffer.take() {
                compute_utils::free_rid_on_render_thread(rd, old.rid);
            }
            let tri_ids_bytes = new_capacity.saturating_mul(8);
            let tri_ids_buf = compute_utils::create_empty_storage_buffer(rd, tri_ids_bytes);
            self.out_tri_ids_buffer = Some(tri_ids_buf);
        }
    }

    /// Lazily creates the 4-byte atomic counter buffer.
    fn ensure_counter(&mut self, rd: &mut Gd<RenderingDevice>) {
        if self.counter_buffer.is_none() {
            self.counter_buffer = Some(compute_utils::create_empty_storage_buffer(rd, 4));
        }
    }

    /// Lazily uploads the blue-noise bytes as a packed `StructuredBuffer<uint>`.
    fn ensure_noise(&mut self, rd: &mut Gd<RenderingDevice>) {
        if self.noise_buffer.is_some() || self.noise_bytes.is_empty() {
            return;
        }
        // Pad to a multiple of 4 bytes so we can re-interpret as `uint`.
        let mut padded = self.noise_bytes.clone();
        while padded.len() % 4 != 0 {
            padded.push(0);
        }
        let packed: &[u32] = bytemuck::cast_slice(&padded);
        let buf = compute_utils::create_storage_buffer(rd, packed);
        self.noise_buffer = Some(buf);
    }

    /// Dispatches one thread per triangle in `state`. Counter is cleared to 0
    /// before dispatch. After dispatch, call `readback_count` then
    /// `readback_compact` to retrieve the live transforms.
    pub fn dispatch(
        &mut self,
        rd: &mut Gd<RenderingDevice>,
        terrain_params: &TerrainParams,
        state: &CesState,
    ) {
        self.ensure_counter(rd);
        self.ensure_noise(rd);
        self.ensure_capacity(rd, state.n_tris);

        let pipeline = self
            .pipeline
            .as_ref()
            .expect("CesScatterRuntime pipeline not initialised");
        let out_buf = self
            .out_buffer
            .as_ref()
            .expect("CesScatterRuntime out_buffer not allocated");
        let counter_buf = self
            .counter_buffer
            .as_ref()
            .expect("CesScatterRuntime counter_buffer not allocated");

        // Clear the counter.
        compute_utils::buffer_clear_on_render_thread(rd, counter_buf.rid, 0, 4);

        // Bind ephemerals.
        let scatter_params_buf = compute_utils::create_uniform_buffer(rd, &self.params);
        let terrain_params_buf = compute_utils::create_uniform_buffer(rd, terrain_params);

        // The blue-noise buffer might be missing (no noise uploaded yet); use a
        // tiny dummy so the shader still runs (it will sample 0 → no spawns).
        let dummy_noise: BufferInfo;
        let noise_buf = match &self.noise_buffer {
            Some(b) => b,
            None => {
                dummy_noise = compute_utils::create_empty_storage_buffer(rd, 16);
                &dummy_noise
            }
        };

        // When incremental mode is off we still must bind something to slots
        // 10/11 because the shader unconditionally references them. Use the
        // counter buffer as a harmless 4-byte placeholder for binding 10 (the
        // shader will read garbage, but the result is discarded — uses_incremental
        // gates whether tri_ids are read back) and a tiny dummy for slot 11.
        let dummy_tri_ids: BufferInfo;
        let out_tri_ids_buf: &BufferInfo = match &self.out_tri_ids_buffer {
            Some(b) => b,
            None => {
                dummy_tri_ids = compute_utils::create_empty_storage_buffer(rd, 16);
                &dummy_tri_ids
            }
        };

        let buffers: Vec<&BufferInfo> = vec![
            &scatter_params_buf,  // 0
            &terrain_params_buf,  // 1
            out_buf,              // 2
            counter_buf,          // 3
            &state.v_pos,         // 4
            &state.t_abc,         // 5
            &state.t_lv,          // 6
            &state.t_deactivated, // 7
            &state.u_n_tris,      // 8
            noise_buf,            // 9
            &state.t_tri_id,      // 10
            out_tri_ids_buf,      // 11
        ];

        let workgroup_count = state.n_tris.div_ceil(64).max(1);
        pipeline.dispatch(rd, &buffers, workgroup_count, "scatter_dispatch");

        // Free ephemeral uniforms.
        compute_utils::free_rid_on_render_thread(rd, scatter_params_buf.rid);
        compute_utils::free_rid_on_render_thread(rd, terrain_params_buf.rid);
        if self.noise_buffer.is_none() {
            compute_utils::free_rid_on_render_thread(rd, noise_buf.rid);
        }
        if self.out_tri_ids_buffer.is_none() {
            compute_utils::free_rid_on_render_thread(rd, out_tri_ids_buf.rid);
        }
    }

    /// Reads back the atomic counter as the live spawn count.
    pub fn readback_count(&self, rd: &mut Gd<RenderingDevice>) -> u32 {
        let buf = self
            .counter_buffer
            .as_ref()
            .expect("counter buffer missing");
        let v: Vec<u32> = compute_utils::convert_buffer_to_vec(rd, buf);
        v.first().copied().unwrap_or(0)
    }

    /// Reads back `count * 12` floats from the transform buffer. Caller must
    /// know `count` from a prior `readback_count` call.
    pub fn readback_compact(&self, rd: &mut Gd<RenderingDevice>, count: u32) -> Vec<f32> {
        // Godot's `buffer_get_data_ex(...).size_bytes(0)` reads the *whole*
        // buffer (size_bytes=0 is the API default meaning "to end"), not zero
        // bytes. Guard count=0 explicitly to avoid returning the full buffer
        // capacity as garbage transforms.
        if count == 0 {
            return Vec::new();
        }
        let out_buf = self.out_buffer.as_ref().expect("out_buffer not allocated");
        let filled_floats = count.saturating_mul(FLOATS_PER_INSTANCE);
        let filled_bytes = filled_floats.saturating_mul(4);

        let view = BufferInfo {
            rid: out_buf.rid,
            max_size: out_buf.max_size,
            filled_size: filled_bytes.min(out_buf.max_size),
            buffer_type: out_buf.buffer_type,
        };
        compute_utils::convert_buffer_to_vec::<f32>(rd, &view)
    }

    /// Reads back `count` u64s from the parallel tri_ids buffer. Returns an
    /// empty Vec when incremental mode is off or no buffer is allocated.
    pub fn readback_tri_ids(&self, rd: &mut Gd<RenderingDevice>, count: u32) -> Vec<u64> {
        // See note in `readback_compact`: size_bytes(0) means "to end".
        if count == 0 {
            return Vec::new();
        }
        let Some(buf) = self.out_tri_ids_buffer.as_ref() else {
            return Vec::new();
        };
        let filled_bytes = (count as u64).saturating_mul(8) as u32;
        let view = BufferInfo {
            rid: buf.rid,
            max_size: buf.max_size,
            filled_size: filled_bytes.min(buf.max_size),
            buffer_type: buf.buffer_type,
        };
        compute_utils::convert_buffer_to_vec::<u64>(rd, &view)
    }

    /// Computes the (added, removed) diff for `variant_idx` between the
    /// previous snapshot and the current dispatch's readbacks, and replaces
    /// the snapshot. `transforms` is the flat float buffer (`count * 12`
    /// floats), `tri_ids` has `count` entries. Callers must ensure
    /// `tri_ids.len() == transforms.len() / 12`.
    pub fn swap_previous_and_diff(
        &mut self,
        variant_idx: usize,
        tri_ids: Vec<u64>,
        transforms: &[f32],
    ) -> (Vec<(u64, [f32; 12])>, Vec<u64>) {
        debug_assert_eq!(tri_ids.len() * 12, transforms.len());
        let mut curr: Vec<(u64, [f32; 12])> = tri_ids
            .into_iter()
            .enumerate()
            .map(|(i, id)| {
                let base = i * 12;
                let mut t = [0.0f32; 12];
                t.copy_from_slice(&transforms[base..base + 12]);
                (id, t)
            })
            .collect();
        curr.sort_unstable_by_key(|p| p.0);

        if self.previous.len() <= variant_idx {
            self.previous.resize_with(variant_idx + 1, Vec::new);
        }
        let prev = &self.previous[variant_idx];
        let diff = two_pointer_diff(prev, &curr);
        self.previous[variant_idx] = curr;
        diff
    }

    /// Frees pipeline + all owned buffers. Mirrors the layer-runtime pattern.
    pub fn dispose_direct(&mut self, rd: &mut Gd<RenderingDevice>) {
        if let Some(ref mut pipeline) = self.pipeline {
            pipeline.dispose_direct(rd);
        }
        self.pipeline = None;
        for slot in [
            &mut self.out_buffer,
            &mut self.out_tri_ids_buffer,
            &mut self.counter_buffer,
            &mut self.noise_buffer,
        ] {
            if let Some(buf) = slot.take() {
                if buf.rid.is_valid() {
                    rd.free_rid(buf.rid);
                }
            }
        }
        self.capacity_instances = 0;
    }
}

fn requested_output_capacity(
    has_buffer: bool,
    current_capacity: u32,
    required: u32,
    force_shrink_each_dispatch: bool,
) -> Option<u32> {
    let desired = next_capacity(required);
    if !has_buffer {
        return Some(desired);
    }
    if force_shrink_each_dispatch {
        return Some(desired);
    }
    if current_capacity >= required {
        None
    } else {
        Some(desired)
    }
}

/// Two-pointer merge over two **sorted-by-key** slices of (tri_id, transform)
/// pairs. Classifies every key as added (in `curr`, not in `prev`), removed
/// (in `prev`, not in `curr`), or stable (drop entirely — same tri_id +
/// same scatter params → identical transform, deterministic from triangle
/// geometry; no need to compare transform values).
pub fn two_pointer_diff(
    prev: &[(u64, [f32; 12])],
    curr: &[(u64, [f32; 12])],
) -> (Vec<(u64, [f32; 12])>, Vec<u64>) {
    let mut added: Vec<(u64, [f32; 12])> = Vec::new();
    let mut removed: Vec<u64> = Vec::new();
    let (mut i, mut j) = (0usize, 0usize);
    while i < prev.len() && j < curr.len() {
        let a = prev[i].0;
        let b = curr[j].0;
        if a == b {
            i += 1;
            j += 1;
        } else if a < b {
            removed.push(a);
            i += 1;
        } else {
            added.push(curr[j]);
            j += 1;
        }
    }
    while i < prev.len() {
        removed.push(prev[i].0);
        i += 1;
    }
    while j < curr.len() {
        added.push(curr[j]);
        j += 1;
    }
    (added, removed)
}

/// Capacity round-up to a power of two with a 64 floor.
pub(crate) fn next_capacity(required: u32) -> u32 {
    const MIN: u32 = 64;
    let target = required.max(MIN);
    if target.is_power_of_two() {
        target
    } else {
        target.checked_next_power_of_two().unwrap_or(u32::MAX)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn read_u32(bytes: &[u8], start: usize) -> u32 {
        u32::from_ne_bytes(bytes[start..start + 4].try_into().unwrap())
    }

    fn read_f32(bytes: &[u8], start: usize) -> f32 {
        f32::from_ne_bytes(bytes[start..start + 4].try_into().unwrap())
    }

    fn read_i32(bytes: &[u8], start: usize) -> i32 {
        i32::from_ne_bytes(bytes[start..start + 4].try_into().unwrap())
    }

    #[test]
    fn scatter_params_pod_size_matches_slang_struct() {
        let params = ScatterParams {
            planet_radius: 60.0,
            seed: 7,
            subdivision_level: 5,
            noise_strength: 0.8,
            height_min: 0.3,
            height_max: 0.6,
            albedo_tolerance: 0.2,
            noise_dim: 32,
            albedo_target: [0.2, 0.6, 0.2],
            _pad: 0.0,
            variant_id: 2,
            variant_count: 3,
            _pad1: 0,
            _pad2: 0,
        };
        let bytes = bytemuck::bytes_of(&params);
        assert_eq!(std::mem::size_of::<ScatterParams>(), 64);
        assert_eq!(std::mem::align_of::<ScatterParams>(), 16);
        assert_eq!(bytes.len(), 64);

        assert_eq!(read_f32(bytes, 0), params.planet_radius);
        assert_eq!(read_u32(bytes, 4), params.seed);
        assert_eq!(read_i32(bytes, 8), params.subdivision_level);
        assert_eq!(read_f32(bytes, 12), params.noise_strength);

        assert_eq!(read_f32(bytes, 16), params.height_min);
        assert_eq!(read_f32(bytes, 20), params.height_max);
        assert_eq!(read_f32(bytes, 24), params.albedo_tolerance);
        assert_eq!(read_u32(bytes, 28), params.noise_dim);

        assert_eq!(read_f32(bytes, 32), params.albedo_target[0]);
        assert_eq!(read_f32(bytes, 36), params.albedo_target[1]);
        assert_eq!(read_f32(bytes, 40), params.albedo_target[2]);
        assert_eq!(read_f32(bytes, 44), params._pad);

        assert_eq!(read_u32(bytes, 48), params.variant_id);
        assert_eq!(read_u32(bytes, 52), params.variant_count);
        assert_eq!(read_u32(bytes, 56), params._pad1);
        assert_eq!(read_u32(bytes, 60), params._pad2);
    }

    #[test]
    fn scatter_params_default_variant_filter_is_passthrough() {
        let p = ScatterParams::default();
        assert_eq!(p.variant_id, 0);
        assert_eq!(p.variant_count, 1);
    }

    #[test]
    fn scatter_params_default_values_sane() {
        let p = ScatterParams::default();
        assert_eq!(p.subdivision_level, 5);
        assert_eq!(p.noise_dim, 32);
        assert_eq!(p.noise_strength, 1.0);
        assert_eq!(p.albedo_tolerance, 2.0);
    }

    #[test]
    fn next_capacity_grows_monotonically() {
        let mut prev = 0;
        for n in [0u32, 1, 64, 65, 128, 256, 1024, 20480, 81920] {
            let c = next_capacity(n);
            assert!(c >= n.max(64));
            assert!(c.is_power_of_two());
            assert!(c >= prev);
            prev = c;
        }
    }

    #[test]
    fn test_scatter_runtime_new_has_no_pipeline() {
        let rt = CesScatterRuntime::with_default_shader();
        assert!(!rt.has_pipeline());
        assert_eq!(rt.capacity_instances(), 0);
    }

    fn mk(id: u64, marker: f32) -> (u64, [f32; 12]) {
        let mut t = [0.0; 12];
        t[3] = marker;
        (id, t)
    }

    #[test]
    fn two_pointer_diff_empty_prev_yields_all_added() {
        let curr = vec![mk(1, 1.0), mk(2, 2.0)];
        let (added, removed) = super::two_pointer_diff(&[], &curr);
        assert_eq!(added.len(), 2);
        assert_eq!(added[0].0, 1);
        assert_eq!(added[1].0, 2);
        assert!(removed.is_empty());
    }

    #[test]
    fn two_pointer_diff_empty_curr_yields_all_removed() {
        let prev = vec![mk(1, 1.0), mk(2, 2.0)];
        let (added, removed) = super::two_pointer_diff(&prev, &[]);
        assert!(added.is_empty());
        assert_eq!(removed, vec![1u64, 2]);
    }

    #[test]
    fn two_pointer_diff_identical_yields_no_changes() {
        let prev = vec![mk(1, 1.0), mk(2, 2.0), mk(7, 9.0)];
        let curr = prev.clone();
        let (added, removed) = super::two_pointer_diff(&prev, &curr);
        assert!(added.is_empty());
        assert!(removed.is_empty());
    }

    #[test]
    fn two_pointer_diff_partial_overlap_classifies_correctly() {
        let prev = vec![mk(1, 1.0), mk(2, 2.0), mk(3, 3.0)];
        let curr = vec![mk(2, 2.0), mk(3, 3.0), mk(4, 4.0)];
        let (added, removed) = super::two_pointer_diff(&prev, &curr);
        assert_eq!(added.len(), 1);
        assert_eq!(added[0].0, 4);
        assert_eq!(removed, vec![1u64]);
    }

    #[test]
    fn two_pointer_diff_handles_interleaved_keys() {
        let prev = vec![mk(1, 1.0), mk(3, 3.0), mk(5, 5.0), mk(7, 7.0)];
        let curr = vec![mk(2, 2.0), mk(3, 3.0), mk(6, 6.0), mk(7, 7.0)];
        let (added, removed) = super::two_pointer_diff(&prev, &curr);
        let added_ids: Vec<u64> = added.iter().map(|p| p.0).collect();
        assert_eq!(added_ids, vec![2u64, 6]);
        assert_eq!(removed, vec![1u64, 5]);
    }

    #[test]
    fn force_shrink_toggle_resizes_buffer_when_n_tris_drops() {
        let current_capacity = next_capacity(200);
        let reduced_tris = 80;
        assert_eq!(
            requested_output_capacity(true, current_capacity, reduced_tris, false),
            None
        );
        assert_eq!(
            requested_output_capacity(true, current_capacity, reduced_tris, true),
            Some(next_capacity(reduced_tris))
        );
    }
}
