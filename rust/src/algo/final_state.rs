use godot::builtin::{Vector2, Vector3};
use godot::classes::RenderingDevice;
use godot::obj::Gd;
use godot::prelude::godot_print;

use crate::buffer_info::BufferInfo;
use crate::compute_utils;
use crate::compute_utils::ComputePipeline;
use crate::state::CesState;

const SHADER_PATH: &str = "res://addons/celestial_sim/shaders/CreateFinalOutput.slang";
const VISIBLE_PREFIX_SHADER_PATH: &str =
    "res://addons/celestial_sim/shaders/ComputeVisiblePrefixSum.slang";

pub struct FinalStateShader {
    pipeline: ComputePipeline,
    visible_prefix_pipeline: ComputePipeline,
}

impl FinalStateShader {
    pub fn new(rd: &mut Gd<RenderingDevice>) -> Self {
        Self {
            pipeline: ComputePipeline::new(rd, SHADER_PATH),
            visible_prefix_pipeline: ComputePipeline::new(rd, VISIBLE_PREFIX_SHADER_PATH),
        }
    }

    pub fn dispose_direct(&mut self, rd: &mut Gd<RenderingDevice>) {
        self.pipeline.dispose_direct(rd);
        self.visible_prefix_pipeline.dispose_direct(rd);
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

    // Step 1: Compute visible mask + prefix sum in one dispatch
    let visible_mask_buffer = compute_utils::create_empty_storage_buffer(rd, state.n_tris * 4);
    let visible_prefix_buffer = compute_utils::create_empty_storage_buffer(rd, state.n_tris * 4);
    shader.visible_prefix_pipeline.dispatch(
        rd,
        &[
            &state.t_divided,
            &state.t_deactivated,
            &visible_mask_buffer,
            &visible_prefix_buffer,
            &state.u_n_tris,
        ],
        1,
    );

    let n_visible_tris = state.n_tris - state.n_divided - state.n_deactivated_tris;

    if n_visible_tris == 0 {
        compute_utils::free_rid_on_render_thread(rd, visible_mask_buffer.rid);
        compute_utils::free_rid_on_render_thread(rd, visible_prefix_buffer.rid);
        return GpuFinalOutput {
            pos: None,
            tris: None,
            uv: None,
            n_visible_tris,
        };
    }

    let vertex_count = n_visible_tris * 3;
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

    shader.pipeline.dispatch(rd, &buffers, state.n_tris);

    compute_utils::free_rid_on_render_thread(rd, visible_mask_buffer.rid);
    compute_utils::free_rid_on_render_thread(rd, visible_prefix_buffer.rid);
    compute_utils::free_rid_on_render_thread(rd, n_visible_buf.rid);

    GpuFinalOutput {
        pos: Some(out_pos),
        tris: Some(out_tris),
        uv: Some(out_uv),
        n_visible_tris,
    }
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
