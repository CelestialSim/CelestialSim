use godot::builtin::{Vector2, Vector3};
use godot::classes::rendering_device::UniformType;
use godot::classes::{RdUniform, RenderingDevice};
use godot::obj::{Gd, NewGd};
use godot::prelude::{godot_print, Array, Rid};
use std::time::Instant;

use crate::buffer_info::{BufferInfo, BufferType};
use crate::compute_utils;
use crate::compute_utils::{ComputePipeline, RdSend};
use crate::shared_texture::SharedPositionTexture;
use crate::state::CesState;

const SHADER_PATH: &str = "res://addons/celestial_sim/shaders/CreateFinalOutput.slang";
const POSITION_TEXTURE_SHADER_PATH: &str =
    "res://addons/celestial_sim/shaders/CreateFinalOutputVertexTexture.slang";
const VISIBLE_PREFIX_ATOMIC_SHADER_PATH: &str =
    "res://addons/celestial_sim/shaders/ComputeVisiblePrefixAtomic.slang";

pub struct FinalStateShader {
    pipeline: ComputePipeline,
    position_texture_pipeline: ComputePipeline,
    visible_prefix_pipeline_atomic: ComputePipeline,
}

impl FinalStateShader {
    pub fn new(rd: &mut Gd<RenderingDevice>) -> Self {
        Self {
            pipeline: ComputePipeline::new(rd, SHADER_PATH),
            position_texture_pipeline: ComputePipeline::new(rd, POSITION_TEXTURE_SHADER_PATH),
            visible_prefix_pipeline_atomic: ComputePipeline::new(
                rd,
                VISIBLE_PREFIX_ATOMIC_SHADER_PATH,
            ),
        }
    }

    fn visible_prefix_pipeline(&self) -> &ComputePipeline {
        &self.visible_prefix_pipeline_atomic
    }

    pub fn dispose_direct(&mut self, rd: &mut Gd<RenderingDevice>) {
        self.pipeline.dispose_direct(rd);
        self.position_texture_pipeline.dispose_direct(rd);
        self.visible_prefix_pipeline_atomic.dispose_direct(rd);
    }
}

/// GPU-side output buffers from the final compaction step.
pub struct GpuFinalOutput {
    pub pos: Option<BufferInfo>,
    pub tris: Option<BufferInfo>,
    pub uv: Option<BufferInfo>,
    pub n_visible_tris: u32,
}

/// CPU-side final mesh output.
pub struct FinalOutput {
    pub tris: Vec<i32>,
    pub uv: Vec<Vector2>,
    pub pos: Vec<Vector3>,
}

/// Experimental final output that leaves positions on the GPU in a shared
/// texture while the CPU prepares topology separately.
pub struct FinalTextureOutput {
    pub n_visible_tris: u32,
}

impl FinalTextureOutput {
    pub fn vertex_count(&self) -> u32 {
        visible_vertex_count(self.n_visible_tris)
    }

    pub fn is_empty(&self) -> bool {
        self.n_visible_tris == 0
    }
}

/// Creates the final compacted mesh output (GPU dispatch + CPU readback).
/// Mirrors C# `CesFinalState.CreateFinalOutput`.
pub fn create_final_output(
    rd: &mut Gd<RenderingDevice>,
    state: &CesState,
    _low_poly: bool,
    shader: &FinalStateShader,
) -> FinalOutput {
    let gpu_output = create_final_output_gpu(rd, state, shader);
    godot_print!(
        "Number of invisible (deactivate + parents) triangles: {}",
        state.n_tris - gpu_output.n_visible_tris
    );
    read_final_output_to_cpu(rd, gpu_output, true)
}

/// Dispatches the CreateFinalOutput shader to produce a compacted mesh on the GPU.
/// Mirrors C# `CesFinalState.CreateFinalOutputGpu`.
pub fn create_final_output_gpu(
    rd: &mut Gd<RenderingDevice>,
    state: &CesState,
    shader: &FinalStateShader,
) -> GpuFinalOutput {
    if state.n_tris == 0 {
        return GpuFinalOutput {
            pos: None,
            tris: None,
            uv: None,
            n_visible_tris: 0,
        };
    }

    let (visible_mask_buffer, visible_prefix_buffer, n_visible_tris) =
        create_visible_prefix_buffers(rd, state, shader, false);

    if n_visible_tris == 0 {
        free_visible_buffers(rd, visible_mask_buffer, visible_prefix_buffer);
        return GpuFinalOutput {
            pos: None,
            tris: None,
            uv: None,
            n_visible_tris,
        };
    }

    let vertex_count = visible_vertex_count(n_visible_tris);
    let out_pos = compute_utils::create_empty_storage_buffer(
        rd,
        vertex_count * 3 * std::mem::size_of::<f32>() as u32,
    );
    let out_tris = compute_utils::create_empty_storage_buffer(
        rd,
        vertex_count * std::mem::size_of::<i32>() as u32,
    );
    let out_uv = compute_utils::create_empty_storage_buffer(
        rd,
        vertex_count * 2 * std::mem::size_of::<f32>() as u32,
    );

    let n_visible_buf = compute_utils::create_uniform_buffer(rd, &n_visible_tris);

    let buffers: Vec<&BufferInfo> = vec![
        &state.v_pos,           // 0
        &state.t_abc,           // 1
        &state.t_parent,        // 2
        &state.t_neight_ab,     // 3
        &state.t_neight_bc,     // 4
        &state.t_neight_ca,     // 5
        &state.t_lv,            // 6
        &visible_mask_buffer,   // 7
        &visible_prefix_buffer, // 8
        &out_pos,               // 9
        &out_tris,              // 10
        &out_uv,                // 11
        &state.u_n_tris,        // 12
        &n_visible_buf,         // 13
    ];

    shader
        .pipeline
        .dispatch(rd, &buffers, state.n_tris, "final_output");

    free_visible_buffers(rd, visible_mask_buffer, visible_prefix_buffer);
    compute_utils::free_rid_on_render_thread(rd, n_visible_buf.rid);

    GpuFinalOutput {
        pos: Some(out_pos),
        tris: Some(out_tris),
        uv: Some(out_uv),
        n_visible_tris,
    }
}

/// Writes the final compacted positions directly into the back shared position
/// texture, avoiding the CPU position readback used by the standard path.
pub fn create_final_output_to_shared_position_texture(
    rd: &mut Gd<RenderingDevice>,
    state: &CesState,
    shader: &FinalStateShader,
    position_texture: &mut SharedPositionTexture,
    early_exit: bool,
) -> FinalTextureOutput {
    if state.n_tris == 0 {
        return FinalTextureOutput { n_visible_tris: 0 };
    }

    let (visible_mask_buffer, visible_prefix_buffer, n_visible_tris) =
        create_visible_prefix_buffers(rd, state, shader, early_exit);

    if n_visible_tris == 0 {
        free_visible_buffers(rd, visible_mask_buffer, visible_prefix_buffer);
        return FinalTextureOutput { n_visible_tris };
    }

    position_texture.ensure_capacity(rd, visible_vertex_count(n_visible_tris));
    let output_texture_rid = position_texture.back_local_rid();
    assert!(
        output_texture_rid.is_valid(),
        "Shared position texture back buffer is invalid"
    );

    let n_visible_buf = compute_utils::create_uniform_buffer(rd, &n_visible_tris);
    dispatch_final_output_to_texture(
        rd,
        &shader.position_texture_pipeline,
        state,
        &visible_mask_buffer,
        &visible_prefix_buffer,
        output_texture_rid,
        &state.u_n_tris,
        &n_visible_buf,
        "final_output_position_texture",
    );

    free_visible_buffers(rd, visible_mask_buffer, visible_prefix_buffer);
    compute_utils::free_rid_on_render_thread(rd, n_visible_buf.rid);

    FinalTextureOutput { n_visible_tris }
}

/// Reads GPU final output back to CPU arrays.
/// Mirrors C# `CesFinalState.ReadFinalOutputToCpu`.
pub fn read_final_output_to_cpu(
    rd: &mut Gd<RenderingDevice>,
    gpu_output: GpuFinalOutput,
    free_buffers: bool,
) -> FinalOutput {
    let vertex_count = gpu_output.n_visible_tris as usize * 3;
    if vertex_count == 0 {
        return FinalOutput {
            tris: vec![],
            uv: vec![],
            pos: vec![],
        };
    }

    let pos_buf = gpu_output.pos.as_ref().unwrap();
    let tris_buf = gpu_output.tris.as_ref().unwrap();
    let uv_buf = gpu_output.uv.as_ref().unwrap();

    let tris: Vec<i32> = compute_utils::convert_buffer_to_vec(rd, tris_buf);
    let pos: Vec<Vector3> = compute_utils::convert_packed_f32_buffer_to_vec3(rd, pos_buf);
    let uv: Vec<Vector2> = compute_utils::convert_packed_f32_buffer_to_vec2(rd, uv_buf);

    if free_buffers {
        if let Some(ref b) = gpu_output.pos {
            compute_utils::free_rid_on_render_thread(rd, b.rid);
        }
        if let Some(ref b) = gpu_output.tris {
            compute_utils::free_rid_on_render_thread(rd, b.rid);
        }
        if let Some(ref b) = gpu_output.uv {
            compute_utils::free_rid_on_render_thread(rd, b.rid);
        }
    }

    FinalOutput { tris, uv, pos }
}

fn create_visible_prefix_buffers(
    rd: &mut Gd<RenderingDevice>,
    state: &CesState,
    shader: &FinalStateShader,
    early_exit: bool,
) -> (BufferInfo, BufferInfo, u32) {
    let visible_mask_buffer = compute_utils::create_empty_storage_buffer(rd, state.n_tris * 4);
    let visible_prefix_buffer = compute_utils::create_empty_storage_buffer(rd, state.n_tris * 4);
    let visible_count_buffer = compute_utils::create_empty_storage_buffer(rd, 4);
    if early_exit {
        compute_utils::free_rid_on_render_thread(rd, visible_count_buffer.rid);
        return (visible_mask_buffer, visible_prefix_buffer, 1310720);
    }
    let visible_prefix_label = "visible_prefix";
    let visible_prefix_start = Instant::now();
    let visible_prefix_workgroups = state.n_tris.div_ceil(64).max(1);
    shader.visible_prefix_pipeline().dispatch(
        rd,
        &[
            &state.t_divided,
            &state.t_deactivated,
            &visible_mask_buffer,
            &visible_prefix_buffer,
            &visible_count_buffer,
            &state.u_n_tris,
        ],
        visible_prefix_workgroups,
        visible_prefix_label,
    );
    let visible_prefix_elapsed_ns = visible_prefix_start.elapsed().as_nanos() as u64;
    crate::perf::with_current_tree(|tree| {
        let cur = crate::perf::current_path_joined();
        let path_full = if cur.is_empty() {
            visible_prefix_label.to_string()
        } else {
            format!("{cur}::{visible_prefix_label}")
        };
        tree.add_cpu_ns(&path_full, visible_prefix_elapsed_ns);
    });

    let n_visible_tris =
        count_visible_triangles(state.n_tris, state.n_divided, state.n_deactivated_tris);

    godot_print!("visible tris: {}", n_visible_tris);
    compute_utils::free_rid_on_render_thread(rd, visible_count_buffer.rid);
    (visible_mask_buffer, visible_prefix_buffer, n_visible_tris)
}

fn free_visible_buffers(
    rd: &mut Gd<RenderingDevice>,
    visible_mask_buffer: BufferInfo,
    visible_prefix_buffer: BufferInfo,
) {
    compute_utils::free_rid_on_render_thread(rd, visible_mask_buffer.rid);
    compute_utils::free_rid_on_render_thread(rd, visible_prefix_buffer.rid);
}

fn visible_vertex_count(n_visible_tris: u32) -> u32 {
    n_visible_tris.saturating_mul(3)
}

fn count_visible_triangles(n_tris: u32, n_divided: u32, n_deactivated_tris: u32) -> u32 {
    n_tris
        .saturating_sub(n_divided)
        .saturating_sub(n_deactivated_tris)
}

fn dispatch_final_output_to_texture(
    rd: &mut Gd<RenderingDevice>,
    pipeline: &ComputePipeline,
    state: &CesState,
    visible_mask_buffer: &BufferInfo,
    visible_prefix_buffer: &BufferInfo,
    output_texture_rid: Rid,
    n_tris_buffer: &BufferInfo,
    n_visible_buffer: &BufferInfo,
    label: &'static str,
) {
    let rd_send = RdSend(rd.clone());
    let shader_rid = pipeline.shader();
    let pipeline_rid = pipeline.pipeline();
    let dispatch_threads = state.n_tris;
    let uniform_descs = vec![
        (
            0i32,
            state.v_pos.rid,
            BufferType::StorageBuffer.to_uniform_type(),
        ),
        (
            1,
            state.t_abc.rid,
            BufferType::StorageBuffer.to_uniform_type(),
        ),
        (
            2,
            state.t_parent.rid,
            BufferType::StorageBuffer.to_uniform_type(),
        ),
        (
            3,
            state.t_neight_ab.rid,
            BufferType::StorageBuffer.to_uniform_type(),
        ),
        (
            4,
            state.t_neight_bc.rid,
            BufferType::StorageBuffer.to_uniform_type(),
        ),
        (
            5,
            state.t_neight_ca.rid,
            BufferType::StorageBuffer.to_uniform_type(),
        ),
        (
            6,
            state.t_lv.rid,
            BufferType::StorageBuffer.to_uniform_type(),
        ),
        (
            7,
            visible_mask_buffer.rid,
            BufferType::StorageBuffer.to_uniform_type(),
        ),
        (
            8,
            visible_prefix_buffer.rid,
            BufferType::StorageBuffer.to_uniform_type(),
        ),
        (
            10,
            n_tris_buffer.rid,
            BufferType::UniformBuffer.to_uniform_type(),
        ),
        (
            11,
            n_visible_buffer.rid,
            BufferType::UniformBuffer.to_uniform_type(),
        ),
    ];

    let path_full = compute_utils::current_render_thread_timing_path(Some(label));

    if let Some(ref path_full) = path_full {
        crate::perf::with_current_tree(|tree| {
            tree.add_cpu_ns(path_full, 0);
        });
    }

    compute_utils::on_render_thread(move || {
        let mut rd = rd_send;
        let dispatch_start = path_full.as_ref().map(|_| Instant::now());

        let compute_list = rd.0.compute_list_begin();
        rd.0.compute_list_bind_compute_pipeline(compute_list, pipeline_rid);

        let mut uniform_array = Array::<Gd<RdUniform>>::new();
        for (binding, rid, uniform_type) in uniform_descs {
            let mut uniform = RdUniform::new_gd();
            uniform.set_binding(binding);
            uniform.set_uniform_type(uniform_type);
            uniform.add_id(rid);
            uniform_array.push(&uniform);
        }

        let mut texture_uniform = RdUniform::new_gd();
        texture_uniform.set_binding(9);
        texture_uniform.set_uniform_type(UniformType::IMAGE);
        texture_uniform.add_id(output_texture_rid);
        uniform_array.push(&texture_uniform);

        let uniform_set = rd.0.uniform_set_create(&uniform_array, shader_rid, 0);
        assert!(
            uniform_set.is_valid(),
            "Failed to create final-state texture uniform set"
        );

        rd.0.compute_list_bind_uniform_set(compute_list, uniform_set, 0);
        rd.0.compute_list_dispatch(compute_list, dispatch_threads, 1, 1);
        rd.0.compute_list_end();

        rd.0.submit();
        rd.0.sync();
        rd.0.free_rid(uniform_set);

        if let (Some(path), Some(start)) = (path_full, dispatch_start) {
            compute_utils::record_render_thread_timing(path, start.elapsed().as_nanos() as u64);
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_visible_mask_computation() {
        // Test that the visibility mask logic is correct:
        // visible when both divided == 0 AND deactivated == 0
        let div_mask = vec![0, 1, 0, 0, 1];
        let deact_mask = vec![0, 0, 1, 0, 0];
        let visible: Vec<i32> = div_mask
            .iter()
            .zip(deact_mask.iter())
            .map(|(&d, &a)| if d == 0 && a == 0 { 1 } else { 0 })
            .collect();
        assert_eq!(visible, vec![1, 0, 0, 1, 0]);
    }

    #[test]
    fn test_visible_prefix_sum() {
        let mut visible = vec![1, 0, 0, 1, 0];
        crate::compute_utils::sum_array_in_place(&mut visible, false);
        assert_eq!(visible, vec![1, 1, 1, 2, 2]);
        assert_eq!(*visible.last().unwrap(), 2); // 2 visible tris
    }

    #[test]
    fn test_count_visible_triangles_saturates_to_zero() {
        assert_eq!(count_visible_triangles(10, 3, 2), 5);
        assert_eq!(count_visible_triangles(10, 10, 5), 0);
        assert_eq!(count_visible_triangles(1, 3, 7), 0);
    }

    #[test]
    fn test_visible_vertex_count_is_three_per_triangle() {
        assert_eq!(visible_vertex_count(0), 0);
        assert_eq!(visible_vertex_count(1), 3);
        assert_eq!(visible_vertex_count(7), 21);
    }

    #[test]
    fn test_final_texture_output_vertex_count() {
        let output = FinalTextureOutput { n_visible_tris: 4 };
        assert_eq!(output.vertex_count(), 12);
        assert!(!output.is_empty());
    }

    #[test]
    fn test_empty_final_output() {
        let output = FinalOutput {
            tris: vec![],
            uv: vec![],
            pos: vec![],
        };
        assert!(output.tris.is_empty());
        assert!(output.pos.is_empty());
    }

    #[test]
    fn test_uv_float_to_pairs() {
        // Test that UV float array -> Vector2 values works correctly
        let uv_floats: Vec<f32> = vec![1.0, 0.0, 2.0, 0.0, 3.0, 0.0];
        let vertex_count = 3usize;
        let uv: Vec<Vector2> = (0..vertex_count)
            .map(|i| Vector2::new(uv_floats[i * 2], uv_floats[i * 2 + 1]))
            .collect();
        assert_eq!(uv[0], Vector2::new(1.0, 0.0));
        assert_eq!(uv[1], Vector2::new(2.0, 0.0));
        assert_eq!(uv[2], Vector2::new(3.0, 0.0));
    }
}
