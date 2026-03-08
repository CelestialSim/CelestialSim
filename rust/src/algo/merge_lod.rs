use godot::classes::RenderingDevice;

use crate::buffer_info::BufferInfo;
use crate::compute_utils;
use crate::state::CesState;

const SHADER_PATH: &str = "res://addons/celestial_sim_rust/shaders/MergeLOD.slang";

/// Extracts indices where the mask value is non-zero.
fn extract_merge_indices(mask: &[i32]) -> Vec<u32> {
    mask.iter()
        .enumerate()
        .filter(|(_, &v)| v != 0)
        .map(|(i, _)| i as u32)
        .collect()
}

/// Performs triangle merging. Mirrors C# `CesMergeLOD.MakeMerge()`.
///
/// Returns the number of triangles merged (0 if nothing to merge).
pub fn make_merge(rd: &mut RenderingDevice, state: &mut CesState) -> u32 {
    let merge_mask = state.get_t_to_merge_mask(rd);
    let idxs_to_merge = extract_merge_indices(&merge_mask);

    let n_tris_to_merge = idxs_to_merge.len() as u32;
    if n_tris_to_merge == 0 {
        return 0;
    }

    let indices_to_merge_buf = compute_utils::create_storage_buffer(rd, &idxs_to_merge);
    let tris_output_buf =
        compute_utils::create_storage_buffer(rd, &vec![0.0f32; n_tris_to_merge as usize]);
    let n_tris_to_merge_buf = compute_utils::create_uniform_buffer(rd, &n_tris_to_merge);

    let buffers: Vec<&BufferInfo> = vec![
        &state.t_abc,              // 0
        &state.t_divided,          // 1
        &n_tris_to_merge_buf,      // 2
        &state.t_neight_ab,        // 3
        &state.t_neight_bc,        // 4
        &state.t_neight_ca,        // 5
        &state.t_a_t,              // 6
        &state.t_b_t,              // 7
        &state.t_c_t,              // 8
        &state.t_center_t,         // 9
        &indices_to_merge_buf,     // 10
        &state.t_to_merge_mask,    // 11
        &tris_output_buf,          // 12
        &state.t_deactivated,      // 13
        &state.t_lv,               // 14
        &state.v_update_mask,      // 15
        &state.v_pos,              // 16
        &state.t_parent,           // 17
    ];

    compute_utils::dispatch_shader(rd, SHADER_PATH, &buffers, n_tris_to_merge);

    rd.free_rid(indices_to_merge_buf.rid);
    rd.free_rid(tris_output_buf.rid);
    rd.free_rid(n_tris_to_merge_buf.rid);

    state.n_deactivated_tris += n_tris_to_merge;

    n_tris_to_merge
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_merge_indices_extraction() {
        let mask = vec![0i32, 1, 0, 1, 0];
        let indices = extract_merge_indices(&mask);
        assert_eq!(indices, vec![1, 3]);
    }

    #[test]
    fn test_merge_indices_extraction_empty() {
        let mask = vec![0i32, 0, 0, 0];
        let indices = extract_merge_indices(&mask);
        assert!(indices.is_empty());
    }

    #[test]
    fn test_merge_indices_extraction_all_set() {
        let mask = vec![1i32, 2, 3];
        let indices = extract_merge_indices(&mask);
        assert_eq!(indices, vec![0, 1, 2]);
    }
}
