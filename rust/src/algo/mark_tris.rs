use godot::builtin::Vector3;
use godot::classes::RenderingDevice;

use crate::buffer_info::BufferInfo;
use crate::compute_utils;
use crate::state::CesState;

const SHADER_PATH: &str = "res://addons/celestial_sim_rust/shaders/MarkTrisToDivide.slang";

/// Dispatches the MarkTrisToDivide shader to flag triangles whose screen-space
/// area exceeds `max_tri_size`. Mirrors C# `CesMarkTrisToDivide.FlagLargeTrisToDivide`.
pub fn flag_large_tris_to_divide(
    rd: &mut RenderingDevice,
    state: &CesState,
    camera_pos: Vector3,
    max_divs: u32,
    radius: f32,
    max_tri_size: f32,
) {
    // Temporary storage buffer for per-triangle sizes
    let tris_size = compute_utils::create_storage_buffer(rd, &vec![0.0f32; state.n_tris as usize]);

    // Uniform buffers for shader parameters
    let camera_pos_arr: [f32; 3] = [camera_pos.x, camera_pos.y, camera_pos.z];
    let camera_pos_buf = compute_utils::create_uniform_buffer(rd, &camera_pos_arr);
    let max_divs_buf = compute_utils::create_uniform_buffer(rd, &max_divs);
    let radius_buf = compute_utils::create_uniform_buffer(rd, &radius);
    let max_tri_size_buf = compute_utils::create_uniform_buffer(rd, &max_tri_size);

    let buffers: Vec<&BufferInfo> = vec![
        &state.v_pos,              // 0
        &state.t_abc,              // 1
        &state.t_lv,               // 2
        &state.t_divided,          // 3
        &state.t_to_divide_mask,   // 4
        &camera_pos_buf,           // 5
        &max_divs_buf,             // 6
        &radius_buf,               // 7
        &max_tri_size_buf,         // 8
        &tris_size,                // 9
        &state.t_neight_ab,        // 10
        &state.t_neight_bc,        // 11
        &state.t_neight_ca,        // 12
        &state.t_deactivated,      // 13
        &state.t_to_merge_mask,    // 14
        &state.t_parent,           // 15
    ];

    compute_utils::dispatch_shader(rd, SHADER_PATH, &buffers, state.n_tris);

    // Free temporary buffers
    rd.free_rid(tris_size.rid);
    rd.free_rid(camera_pos_buf.rid);
    rd.free_rid(max_divs_buf.rid);
    rd.free_rid(radius_buf.rid);
    rd.free_rid(max_tri_size_buf.rid);
}
