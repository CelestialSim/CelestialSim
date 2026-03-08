use godot::classes::RenderingDevice;

use crate::buffer_info::BufferInfo;
use crate::compute_utils;
use crate::state::CesState;

const COMPACT_TRIS_SHADER: &str = "res://addons/celestial_sim_rust/shaders/CompactTris.slang";
const MARK_ACTIVE_VERTS_SHADER: &str =
    "res://addons/celestial_sim_rust/shaders/MarkActiveVertices.slang";
const COMPACT_VERTS_SHADER: &str =
    "res://addons/celestial_sim_rust/shaders/CompactVertices.slang";
const REMAP_TRI_VERTS_SHADER: &str =
    "res://addons/celestial_sim_rust/shaders/RemapTriangleVertices.slang";

/// Compacts deactivated triangles and unused vertices from the mesh buffers.
/// Mirrors C# `CesCompactBuffers.Compact()`.
///
/// Returns 0 if nothing was compacted, otherwise returns the count of compacted items.
pub fn compact(rd: &mut RenderingDevice, state: &mut CesState) -> u32 {
    // --- Phase A: Compact Triangles ---
    let mut deactivated: Vec<i32> = compute_utils::convert_buffer_to_vec(rd, &state.t_deactivated);

    // Prefix sum inverted: count active (non-deactivated) triangles
    compute_utils::sum_array_in_place(&mut deactivated, true);

    let active_count = if deactivated.is_empty() {
        0u32
    } else {
        deactivated[deactivated.len() - 1] as u32
    };

    if active_count == 0 || active_count == state.n_tris {
        return 0;
    }

    let active_prefix_buf = compute_utils::create_storage_buffer(rd, &deactivated);

    // Create 15 destination buffers for compacted triangle data
    let tri_elem_size = std::mem::size_of::<i32>() as u32;
    let dst_byte_size = tri_elem_size * active_count;
    let t_abc_dst = compute_utils::create_empty_storage_buffer(rd, 4 * tri_elem_size * active_count);
    let t_divided_dst = compute_utils::create_empty_storage_buffer(rd, dst_byte_size);
    let t_lv_dst = compute_utils::create_empty_storage_buffer(rd, dst_byte_size);
    let t_to_div_dst = compute_utils::create_empty_storage_buffer(rd, dst_byte_size);
    let t_to_merge_dst = compute_utils::create_empty_storage_buffer(rd, dst_byte_size);
    let t_ico_idx_dst = compute_utils::create_empty_storage_buffer(rd, dst_byte_size);
    let t_neigh_ab_dst = compute_utils::create_empty_storage_buffer(rd, dst_byte_size);
    let t_neigh_bc_dst = compute_utils::create_empty_storage_buffer(rd, dst_byte_size);
    let t_neigh_ca_dst = compute_utils::create_empty_storage_buffer(rd, dst_byte_size);
    let t_a_t_dst = compute_utils::create_empty_storage_buffer(rd, dst_byte_size);
    let t_b_t_dst = compute_utils::create_empty_storage_buffer(rd, dst_byte_size);
    let t_c_t_dst = compute_utils::create_empty_storage_buffer(rd, dst_byte_size);
    let t_center_t_dst = compute_utils::create_empty_storage_buffer(rd, dst_byte_size);
    let t_parent_dst = compute_utils::create_empty_storage_buffer(rd, dst_byte_size);
    let t_deactivated_dst = compute_utils::create_empty_storage_buffer(rd, dst_byte_size);

    let n_tris_total_buf = compute_utils::create_uniform_buffer(rd, &state.n_tris);
    let active_count_buf = compute_utils::create_uniform_buffer(rd, &active_count);

    let buffers: Vec<&BufferInfo> = vec![
        &state.t_deactivated,   // 0
        &active_prefix_buf,     // 1
        &state.t_abc,           // 2
        &state.t_divided,       // 3
        &state.t_lv,            // 4
        &state.t_to_divide_mask, // 5
        &state.t_to_merge_mask, // 6
        &state.t_ico_idx,       // 7
        &state.t_neight_ab,     // 8
        &state.t_neight_bc,     // 9
        &state.t_neight_ca,     // 10
        &state.t_a_t,           // 11
        &state.t_b_t,           // 12
        &state.t_c_t,           // 13
        &state.t_center_t,      // 14
        &state.t_parent,        // 15
        &t_abc_dst,             // 16
        &t_divided_dst,         // 17
        &t_lv_dst,              // 18
        &t_to_div_dst,          // 19
        &t_to_merge_dst,        // 20
        &t_ico_idx_dst,         // 21
        &t_neigh_ab_dst,        // 22
        &t_neigh_bc_dst,        // 23
        &t_neigh_ca_dst,        // 24
        &t_a_t_dst,             // 25
        &t_b_t_dst,             // 26
        &t_c_t_dst,             // 27
        &t_center_t_dst,        // 28
        &t_parent_dst,          // 29
        &t_deactivated_dst,     // 30
        &n_tris_total_buf,      // 31
        &active_count_buf,      // 32
    ];

    compute_utils::dispatch_shader(rd, COMPACT_TRIS_SHADER, &buffers, state.n_tris);

    // Swap old buffers with compacted ones, free old
    let old_bufs = [
        std::mem::replace(&mut state.t_abc, t_abc_dst),
        std::mem::replace(&mut state.t_divided, t_divided_dst),
        std::mem::replace(&mut state.t_lv, t_lv_dst),
        std::mem::replace(&mut state.t_to_divide_mask, t_to_div_dst),
        std::mem::replace(&mut state.t_to_merge_mask, t_to_merge_dst),
        std::mem::replace(&mut state.t_ico_idx, t_ico_idx_dst),
        std::mem::replace(&mut state.t_neight_ab, t_neigh_ab_dst),
        std::mem::replace(&mut state.t_neight_bc, t_neigh_bc_dst),
        std::mem::replace(&mut state.t_neight_ca, t_neigh_ca_dst),
        std::mem::replace(&mut state.t_a_t, t_a_t_dst),
        std::mem::replace(&mut state.t_b_t, t_b_t_dst),
        std::mem::replace(&mut state.t_c_t, t_c_t_dst),
        std::mem::replace(&mut state.t_center_t, t_center_t_dst),
        std::mem::replace(&mut state.t_parent, t_parent_dst),
        std::mem::replace(&mut state.t_deactivated, t_deactivated_dst),
    ];
    for buf in &old_bufs {
        rd.free_rid(buf.rid);
    }

    rd.free_rid(active_prefix_buf.rid);
    rd.free_rid(n_tris_total_buf.rid);
    rd.free_rid(active_count_buf.rid);

    state.n_tris = active_count;
    state.n_deactivated_tris = 0;

    // --- Phase B: Compact Vertices ---
    let vert_active_mask_buf = compute_utils::create_empty_storage_buffer(
        rd,
        std::mem::size_of::<i32>() as u32 * state.n_verts,
    );
    rd.buffer_clear(vert_active_mask_buf.rid, 0, vert_active_mask_buf.filled_size);

    let n_tris_uniform = compute_utils::create_uniform_buffer(rd, &state.n_tris);
    let n_verts_uniform = compute_utils::create_uniform_buffer(rd, &state.n_verts);

    let mark_buffers: Vec<&BufferInfo> = vec![
        &state.t_abc,            // 0
        &state.t_deactivated,    // 1
        &vert_active_mask_buf,   // 2
        &n_tris_uniform,         // 3
        &n_verts_uniform,        // 4
    ];

    compute_utils::dispatch_shader(rd, MARK_ACTIVE_VERTS_SHADER, &mark_buffers, state.n_tris);

    rd.free_rid(n_tris_uniform.rid);
    rd.free_rid(n_verts_uniform.rid);

    let mut vert_mask: Vec<i32> = compute_utils::convert_buffer_to_vec(rd, &vert_active_mask_buf);
    compute_utils::sum_array_in_place(&mut vert_mask, false);

    let active_verts_count = if vert_mask.is_empty() {
        0u32
    } else {
        vert_mask[vert_mask.len() - 1] as u32
    };

    if active_verts_count > 0 && active_verts_count != state.n_verts {
        // Pad prefix array to at least 4 elements (GPU alignment)
        while vert_mask.len() < 4 {
            vert_mask.push(*vert_mask.last().unwrap_or(&0));
        }

        let vert_prefix_buf = compute_utils::create_storage_buffer(rd, &vert_mask);

        let v_pos_dst = compute_utils::create_empty_storage_buffer(
            rd,
            4 * std::mem::size_of::<f32>() as u32 * active_verts_count,
        );
        let v_update_mask_dst = compute_utils::create_empty_storage_buffer(
            rd,
            std::mem::size_of::<i32>() as u32 * active_verts_count,
        );

        let n_verts_total_buf = compute_utils::create_uniform_buffer(rd, &state.n_verts);
        let active_verts_count_buf = compute_utils::create_uniform_buffer(rd, &active_verts_count);

        let compact_vert_buffers: Vec<&BufferInfo> = vec![
            &vert_active_mask_buf,    // 0
            &vert_prefix_buf,         // 1
            &state.v_pos,             // 2
            &state.v_update_mask,     // 3
            &v_pos_dst,               // 4
            &v_update_mask_dst,       // 5
            &n_verts_total_buf,       // 6
            &active_verts_count_buf,  // 7
        ];

        compute_utils::dispatch_shader(
            rd,
            COMPACT_VERTS_SHADER,
            &compact_vert_buffers,
            state.n_verts,
        );

        // Swap vertex buffers
        let old_v_pos = std::mem::replace(&mut state.v_pos, v_pos_dst);
        let old_v_mask = std::mem::replace(&mut state.v_update_mask, v_update_mask_dst);
        rd.free_rid(old_v_pos.rid);
        rd.free_rid(old_v_mask.rid);

        state.n_verts = active_verts_count;

        // --- Phase C: Remap Triangle Vertex Indices ---
        let t_abc_remapped = compute_utils::create_empty_storage_buffer(
            rd,
            4 * std::mem::size_of::<i32>() as u32 * state.n_tris,
        );

        let n_tris_remap_buf = compute_utils::create_uniform_buffer(rd, &state.n_tris);
        let active_verts_remap_buf =
            compute_utils::create_uniform_buffer(rd, &active_verts_count);

        let remap_buffers: Vec<&BufferInfo> = vec![
            &state.t_abc,              // 0
            &vert_prefix_buf,          // 1
            &t_abc_remapped,           // 2
            &n_tris_remap_buf,         // 3
            &active_verts_remap_buf,   // 4
        ];

        compute_utils::dispatch_shader(
            rd,
            REMAP_TRI_VERTS_SHADER,
            &remap_buffers,
            state.n_tris,
        );

        let old_t_abc = std::mem::replace(&mut state.t_abc, t_abc_remapped);
        rd.free_rid(old_t_abc.rid);

        rd.free_rid(n_tris_remap_buf.rid);
        rd.free_rid(active_verts_remap_buf.rid);
        rd.free_rid(vert_prefix_buf.rid);
        rd.free_rid(n_verts_total_buf.rid);
        rd.free_rid(active_verts_count_buf.rid);
    }

    rd.free_rid(vert_active_mask_buf.rid);

    active_count
}

#[cfg(test)]
mod tests {
    use crate::compute_utils::sum_array_in_place;

    #[test]
    fn test_prefix_sum_for_compaction() {
        // invert=true: count where value is 0 (active triangles)
        // [0,1,0,0,1] → active at indices 0,2,3 → prefix = [1,1,2,3,3]
        let mut arr = vec![0i32, 1, 0, 0, 1];
        sum_array_in_place(&mut arr, true);
        assert_eq!(arr, vec![1, 1, 2, 3, 3]);
    }

    #[test]
    fn test_prefix_sum_active_count() {
        // active_count is the last element of the inverted prefix sum
        let mut arr = vec![0i32, 1, 0, 0, 1];
        sum_array_in_place(&mut arr, true);
        let active_count = *arr.last().unwrap() as u32;
        assert_eq!(active_count, 3);
    }

    #[test]
    fn test_no_compact_when_all_active() {
        // When no triangles are deactivated, active_count == n_tris → return 0
        let mut arr = vec![0i32, 0, 0, 0, 0];
        sum_array_in_place(&mut arr, true);
        let active_count = *arr.last().unwrap() as u32;
        let n_tris: u32 = 5;
        // The function would return 0 because active_count == n_tris
        assert_eq!(active_count, n_tris);
    }

    #[test]
    fn test_prefix_sum_all_deactivated() {
        // All deactivated: active_count == 0 → return 0
        let mut arr = vec![1i32, 1, 1, 1];
        sum_array_in_place(&mut arr, true);
        let active_count = *arr.last().unwrap() as u32;
        assert_eq!(active_count, 0);
    }

    #[test]
    fn test_prefix_sum_vertex_compaction() {
        // Non-inverted prefix sum for vertex mask
        let mut arr = vec![1i32, 0, 1, 1, 0];
        sum_array_in_place(&mut arr, false);
        assert_eq!(arr, vec![1, 1, 2, 3, 3]);
    }
}
