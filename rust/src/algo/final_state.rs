use godot::builtin::Vector3;
use godot::classes::RenderingDevice;
use godot::prelude::godot_print;

use crate::buffer_info::BufferInfo;
use crate::compute_utils;
use crate::state::CesState;

const SHADER_PATH: &str = "res://addons/celestial_sim_rust/shaders/CreateFinalOutput.slang";

/// GPU-side output buffers from the final compaction step.
pub struct GpuFinalOutput {
    pub pos: Option<BufferInfo>,
    pub tris: Option<BufferInfo>,
    pub color: Option<BufferInfo>,
    pub n_visible_tris: u32,
}

/// CPU-side final mesh output.
pub struct FinalOutput {
    pub tris: Vec<i32>,
    pub color: Vec<[f32; 2]>,
    pub normals: Vec<Vector3>,
    pub pos: Vec<Vector3>,
}

/// Creates the final compacted mesh output (GPU dispatch + CPU readback).
/// Mirrors C# `CesFinalState.CreateFinalOutput`.
pub fn create_final_output(
    rd: &mut RenderingDevice,
    state: &CesState,
    _low_poly: bool,
) -> FinalOutput {
    let gpu_output = create_final_output_gpu(rd, state);
    godot_print!(
        "Number of invisible (deactivate + parents) triangles: {}",
        state.n_tris - gpu_output.n_visible_tris
    );
    read_final_output_to_cpu(rd, gpu_output, true)
}

/// Dispatches the CreateFinalOutput shader to produce a compacted mesh on the GPU.
/// Mirrors C# `CesFinalState.CreateFinalOutputGpu`.
pub fn create_final_output_gpu(rd: &mut RenderingDevice, state: &CesState) -> GpuFinalOutput {
    let div_mask = state.get_divided_mask(rd);
    let deactivated_mask = state.get_t_deactivated_mask(rd);

    let visible_mask: Vec<i32> = div_mask
        .iter()
        .zip(deactivated_mask.iter())
        .map(|(&d, &a)| if d == 0 && a == 0 { 1 } else { 0 })
        .collect();

    let visible_mask_buffer = compute_utils::create_storage_buffer(rd, &visible_mask);

    let mut prefix = visible_mask.clone();
    compute_utils::sum_array_in_place(&mut prefix, false);

    let n_visible_tris = if prefix.is_empty() {
        0u32
    } else {
        prefix[prefix.len() - 1] as u32
    };

    if n_visible_tris == 0 || state.n_tris == 0 {
        rd.free_rid(visible_mask_buffer.rid);
        return GpuFinalOutput {
            pos: None,
            tris: None,
            color: None,
            n_visible_tris,
        };
    }

    let visible_prefix_buffer = compute_utils::create_storage_buffer(rd, &prefix);

    let vertex_count = n_visible_tris * 3;
    let out_pos = compute_utils::create_empty_storage_buffer(
        rd,
        vertex_count * 4 * std::mem::size_of::<f32>() as u32,
    );
    let out_tris = compute_utils::create_empty_storage_buffer(
        rd,
        vertex_count * std::mem::size_of::<i32>() as u32,
    );
    let out_color = compute_utils::create_empty_storage_buffer(
        rd,
        vertex_count * 2 * std::mem::size_of::<f32>() as u32,
    );

    let n_tris_buf = compute_utils::create_uniform_buffer(rd, &state.n_tris);
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
        &out_color,             // 11
        &n_tris_buf,            // 12
        &n_visible_buf,         // 13
    ];

    compute_utils::dispatch_shader(rd, SHADER_PATH, &buffers, state.n_tris);

    rd.free_rid(visible_mask_buffer.rid);
    rd.free_rid(visible_prefix_buffer.rid);
    rd.free_rid(n_tris_buf.rid);
    rd.free_rid(n_visible_buf.rid);

    GpuFinalOutput {
        pos: Some(out_pos),
        tris: Some(out_tris),
        color: Some(out_color),
        n_visible_tris,
    }
}

/// Reads GPU final output back to CPU arrays.
/// Mirrors C# `CesFinalState.ReadFinalOutputToCpu`.
pub fn read_final_output_to_cpu(
    rd: &mut RenderingDevice,
    gpu_output: GpuFinalOutput,
    free_buffers: bool,
) -> FinalOutput {
    let vertex_count = gpu_output.n_visible_tris as usize * 3;
    if vertex_count == 0 {
        return FinalOutput {
            tris: vec![],
            color: vec![],
            normals: vec![],
            pos: vec![],
        };
    }

    let pos_buf = gpu_output.pos.as_ref().unwrap();
    let tris_buf = gpu_output.tris.as_ref().unwrap();
    let color_buf = gpu_output.color.as_ref().unwrap();

    let pos = compute_utils::convert_v4_buffer_to_vec3(rd, pos_buf);
    let tris: Vec<i32> = compute_utils::convert_buffer_to_vec(rd, tris_buf);
    let color_floats: Vec<f32> = compute_utils::convert_buffer_to_vec(rd, color_buf);

    let color: Vec<[f32; 2]> = (0..vertex_count)
        .map(|i| [color_floats[i * 2], color_floats[i * 2 + 1]])
        .collect();

    let normals = vec![Vector3::ZERO; pos.len()];

    if free_buffers {
        if let Some(ref b) = gpu_output.pos {
            rd.free_rid(b.rid);
        }
        if let Some(ref b) = gpu_output.tris {
            rd.free_rid(b.rid);
        }
        if let Some(ref b) = gpu_output.color {
            rd.free_rid(b.rid);
        }
    }

    FinalOutput {
        tris,
        color,
        normals,
        pos,
    }
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
            color: vec![],
            normals: vec![],
            pos: vec![],
        };
        assert!(output.tris.is_empty());
        assert!(output.pos.is_empty());
    }

    #[test]
    fn test_color_float_to_pairs() {
        // Test that color float array -> [f32; 2] pairs works correctly
        let color_floats: Vec<f32> = vec![1.0, 0.0, 2.0, 0.0, 3.0, 0.0];
        let vertex_count = 3usize;
        let color: Vec<[f32; 2]> = (0..vertex_count)
            .map(|i| [color_floats[i * 2], color_floats[i * 2 + 1]])
            .collect();
        assert_eq!(color[0], [1.0, 0.0]);
        assert_eq!(color[1], [2.0, 0.0]);
        assert_eq!(color[2], [3.0, 0.0]);
    }
}
