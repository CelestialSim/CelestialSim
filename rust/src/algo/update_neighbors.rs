use godot::classes::RenderingDevice;
use godot::obj::Gd;

use crate::buffer_info::BufferInfo;
use crate::compute_utils;
use crate::state::CesState;

const SHADER_PATH: &str = "res://addons/celestial_sim_rust/shaders/UpdateNeighbors.slang";

/// Dispatches the UpdateNeighbors shader. Mirrors C# `CesUpdateNeighbors.UpdateNeighbors()`.
pub fn update_neighbors(rd: &mut Gd<RenderingDevice>, state: &CesState) {
    let n_tris_buf = compute_utils::create_uniform_buffer(rd, &state.n_tris);

    let buffers: Vec<&BufferInfo> = vec![
        &state.v_pos,          // 0
        &state.t_abc,          // 1
        &state.t_neight_ab,    // 2
        &state.t_neight_bc,    // 3
        &state.t_neight_ca,    // 4
        &state.t_parent,       // 5
        &state.t_divided,      // 6
        &state.t_lv,           // 7
        &state.t_a_t,          // 8
        &state.t_b_t,          // 9
        &state.t_c_t,          // 10
        &state.t_center_t,     // 11
        &n_tris_buf,           // 12
    ];

    compute_utils::dispatch_shader(rd, SHADER_PATH, &buffers, state.n_tris);

    compute_utils::free_rid_on_render_thread(rd, n_tris_buf.rid);
}
