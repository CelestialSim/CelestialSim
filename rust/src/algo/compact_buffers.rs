use godot::classes::RenderingDevice;
use godot::obj::Gd;

use crate::buffer_info::BufferInfo;
use crate::compute_utils;
use crate::compute_utils::ComputePipeline;
use crate::state::CesState;

const COMPACT_TRIS_SHADER: &str = "res://addons/celestial_sim/shaders/CompactTris.slang";
const MARK_ACTIVE_VERTS_SHADER: &str =
    "res://addons/celestial_sim/shaders/MarkActiveVertices.slang";
const COMPACT_VERTS_SHADER: &str = "res://addons/celestial_sim/shaders/CompactVertices.slang";
const REMAP_TRI_VERTS_SHADER: &str =
    "res://addons/celestial_sim/shaders/RemapTriangleVertices.slang";

pub struct CompactShaders {
    compact_tris: ComputePipeline,
    mark_active_verts: ComputePipeline,
    compact_verts: ComputePipeline,
    remap_tri_verts: ComputePipeline,
}

impl CompactShaders {
    pub fn new(rd: &mut Gd<RenderingDevice>) -> Self {
        Self {
            compact_tris: ComputePipeline::new(rd, COMPACT_TRIS_SHADER),
            mark_active_verts: ComputePipeline::new(rd, MARK_ACTIVE_VERTS_SHADER),
            compact_verts: ComputePipeline::new(rd, COMPACT_VERTS_SHADER),
            remap_tri_verts: ComputePipeline::new(rd, REMAP_TRI_VERTS_SHADER),
        }
    }

    pub fn dispose_direct(&mut self, rd: &mut Gd<RenderingDevice>) {
        self.compact_tris.dispose_direct(rd);
        self.mark_active_verts.dispose_direct(rd);
        self.compact_verts.dispose_direct(rd);
        self.remap_tri_verts.dispose_direct(rd);
    }

    /// Compacts deactivated triangles and unused vertices from the mesh buffers.
    /// Mirrors C# `CesCompactBuffers.Compact()`.
    ///
    /// Returns 0 if nothing was compacted, otherwise returns the count of compacted items.
    pub fn compact(&self, rd: &mut Gd<RenderingDevice>, state: &mut CesState) -> u32 {
        // --- Phase A: Compact Triangles ---
        let mut deactivated: Vec<i32> =
            compute_utils::convert_buffer_to_vec(rd, &state.t_deactivated);

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
        let t_abc_dst =
            compute_utils::create_empty_storage_buffer(rd, 4 * tri_elem_size * active_count);
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
        let t_tri_id_dst = compute_utils::create_empty_storage_buffer(
            rd,
            std::mem::size_of::<u64>() as u32 * active_count,
        );

        let active_count_buf = compute_utils::create_uniform_buffer(rd, &active_count);

        let buffers: Vec<&BufferInfo> = vec![
            &state.t_deactivated,    // 0
            &active_prefix_buf,      // 1
            &state.t_abc,            // 2
            &state.t_divided,        // 3
            &state.t_lv,             // 4
            &state.t_to_divide_mask, // 5
            &state.t_to_merge_mask,  // 6
            &state.t_ico_idx,        // 7
            &state.t_neight_ab,      // 8
            &state.t_neight_bc,      // 9
            &state.t_neight_ca,      // 10
            &state.t_a_t,            // 11
            &state.t_b_t,            // 12
            &state.t_c_t,            // 13
            &state.t_center_t,       // 14
            &state.t_parent,         // 15
            &t_abc_dst,              // 16
            &t_divided_dst,          // 17
            &t_lv_dst,               // 18
            &t_to_div_dst,           // 19
            &t_to_merge_dst,         // 20
            &t_ico_idx_dst,          // 21
            &t_neigh_ab_dst,         // 22
            &t_neigh_bc_dst,         // 23
            &t_neigh_ca_dst,         // 24
            &t_a_t_dst,              // 25
            &t_b_t_dst,              // 26
            &t_c_t_dst,              // 27
            &t_center_t_dst,         // 28
            &t_parent_dst,           // 29
            &t_deactivated_dst,      // 30
            &state.u_n_tris,         // 31
            &active_count_buf,       // 32
            &state.t_tri_id,         // 33 (src)
            &t_tri_id_dst,           // 34 (dst)
        ];

        self.compact_tris
            .dispatch(rd, &buffers, state.n_tris, "compact_tris");

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
            std::mem::replace(&mut state.t_tri_id, t_tri_id_dst),
        ];
        for buf in &old_bufs {
            compute_utils::free_rid_on_render_thread(rd, buf.rid);
        }

        compute_utils::free_rid_on_render_thread(rd, active_prefix_buf.rid);
        compute_utils::free_rid_on_render_thread(rd, active_count_buf.rid);

        state.n_tris = active_count;
        state.n_deactivated_tris = 0;

        // Sync n_tris uniform buffer after update (needed by Phase B)
        state.sync_n_tris_buffer(rd);

        // --- Phase B: Compact Vertices ---
        let vert_active_mask_buf = compute_utils::create_empty_storage_buffer(
            rd,
            std::mem::size_of::<i32>() as u32 * state.n_verts,
        );
        compute_utils::buffer_clear_on_render_thread(
            rd,
            vert_active_mask_buf.rid,
            0,
            vert_active_mask_buf.filled_size,
        );

        let mark_buffers: Vec<&BufferInfo> = vec![
            &state.t_abc,          // 0
            &state.t_deactivated,  // 1
            &vert_active_mask_buf, // 2
            &state.u_n_tris,       // 3
            &state.u_n_verts,      // 4
        ];

        self.mark_active_verts
            .dispatch(rd, &mark_buffers, state.n_tris, "mark_active_verts");

        let mut vert_mask: Vec<i32> =
            compute_utils::convert_buffer_to_vec(rd, &vert_active_mask_buf);
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

            let active_verts_count_buf =
                compute_utils::create_uniform_buffer(rd, &active_verts_count);

            let compact_vert_buffers: Vec<&BufferInfo> = vec![
                &vert_active_mask_buf,   // 0
                &vert_prefix_buf,        // 1
                &state.v_pos,            // 2
                &state.v_update_mask,    // 3
                &v_pos_dst,              // 4
                &v_update_mask_dst,      // 5
                &state.u_n_verts,        // 6
                &active_verts_count_buf, // 7
            ];

            self.compact_verts
                .dispatch(rd, &compact_vert_buffers, state.n_verts, "compact_verts");

            // Swap vertex buffers
            let old_v_pos = std::mem::replace(&mut state.v_pos, v_pos_dst);
            let old_v_mask = std::mem::replace(&mut state.v_update_mask, v_update_mask_dst);
            compute_utils::free_rid_on_render_thread(rd, old_v_pos.rid);
            compute_utils::free_rid_on_render_thread(rd, old_v_mask.rid);

            state.n_verts = active_verts_count;

            // Sync n_verts uniform buffer after update (needed by Phase C)
            state.sync_n_verts_buffer(rd);

            // --- Phase C: Remap Triangle Vertex Indices ---
            let t_abc_remapped = compute_utils::create_empty_storage_buffer(
                rd,
                4 * std::mem::size_of::<i32>() as u32 * state.n_tris,
            );

            let active_verts_remap_buf =
                compute_utils::create_uniform_buffer(rd, &active_verts_count);

            let remap_buffers: Vec<&BufferInfo> = vec![
                &state.t_abc,            // 0
                &vert_prefix_buf,        // 1
                &t_abc_remapped,         // 2
                &state.u_n_tris,         // 3
                &active_verts_remap_buf, // 4
            ];

            self.remap_tri_verts
                .dispatch(rd, &remap_buffers, state.n_tris, "remap_tri_verts");

            let old_t_abc = std::mem::replace(&mut state.t_abc, t_abc_remapped);
            compute_utils::free_rid_on_render_thread(rd, old_t_abc.rid);

            compute_utils::free_rid_on_render_thread(rd, active_verts_remap_buf.rid);
            compute_utils::free_rid_on_render_thread(rd, vert_prefix_buf.rid);
            compute_utils::free_rid_on_render_thread(rd, active_verts_count_buf.rid);
        }

        compute_utils::free_rid_on_render_thread(rd, vert_active_mask_buf.rid);

        active_count
    }
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
