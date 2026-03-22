use godot::classes::RenderingDevice;
use godot::obj::Gd;

use crate::compute_utils;
use crate::state::CesState;

/// 12 icosphere vertices as float4 (x, y, z, w=0).
pub const VERTICES: [[f32; 4]; 12] = [
    [-0.5257, 0.0000, 0.8507, 0.0],
    [0.5257, 0.0000, 0.8507, 0.0],
    [-0.5257, 0.0000, -0.8507, 0.0],
    [0.5257, 0.0000, -0.8507, 0.0],
    [0.0000, 0.8507, 0.5257, 0.0],
    [0.0000, 0.8507, -0.5257, 0.0],
    [0.0000, -0.8507, 0.5257, 0.0],
    [0.0000, -0.8507, -0.5257, 0.0],
    [0.8507, 0.5257, 0.0000, 0.0],
    [-0.8507, 0.5257, 0.0000, 0.0],
    [0.8507, -0.5257, 0.0000, 0.0],
    [-0.8507, -0.5257, 0.0000, 0.0],
];

/// 20 icosphere triangles as int4 (a, b, c, w=0).
pub const TRIANGLES: [[i32; 4]; 20] = [
    [0, 4, 1, 0],
    [0, 9, 4, 0],
    [9, 5, 4, 0],
    [4, 5, 8, 0],
    [4, 8, 1, 0],
    [8, 10, 1, 0],
    [8, 3, 10, 0],
    [5, 3, 8, 0],
    [5, 2, 3, 0],
    [2, 7, 3, 0],
    [7, 10, 3, 0],
    [7, 6, 10, 0],
    [7, 11, 6, 0],
    [11, 0, 6, 0],
    [0, 1, 6, 0],
    [6, 1, 10, 0],
    [9, 0, 11, 0],
    [9, 11, 2, 0],
    [9, 2, 5, 0],
    [7, 2, 11, 0],
];

/// Neighbor along edge AB for each of the 20 triangles.
pub const NEIGHT_AB: [i32; 20] = [
    1, 16, 18, 2, 3, 6, 7, 8, 18, 19, 11, 12, 19, 16, 0, 14, 1, 16, 17, 9,
];

/// Neighbor along edge BC for each of the 20 triangles.
pub const NEIGHT_BC: [i32; 20] = [
    4, 2, 3, 7, 5, 15, 10, 6, 9, 10, 6, 15, 13, 14, 15, 5, 13, 19, 8, 17,
];

/// Neighbor along edge CA for each of the 20 triangles.
pub const NEIGHT_CA: [i32; 20] = [
    14, 0, 1, 4, 0, 4, 5, 3, 7, 8, 9, 10, 11, 12, 13, 11, 17, 18, 2, 12,
];

/// Creates the initial icosphere CesState with all 17 GPU buffers.
pub fn create_core_state(rd: &mut Gd<RenderingDevice>) -> CesState {
    let n_tris: u32 = 20;
    let n_verts: u32 = 12;

    // Flatten vertex data for GPU (already in float4 layout)
    let v_pos_flat: Vec<f32> = VERTICES.iter().flat_map(|v| v.iter().copied()).collect();

    // Flatten triangle data for GPU (already in int4 layout)
    let t_abc_flat: Vec<i32> = TRIANGLES.iter().flat_map(|t| t.iter().copied()).collect();

    // Level: all zero
    let t_lv = vec![0i32; n_tris as usize];

    // Children pointers: all zero (no children yet)
    let t_a_t = vec![0i32; n_tris as usize];
    let t_b_t = vec![0i32; n_tris as usize];
    let t_c_t = vec![0i32; n_tris as usize];
    let t_center_t = vec![0i32; n_tris as usize];

    // Parent: -1 for root triangles
    let t_parent = vec![-1i32; n_tris as usize];

    // Identity mapping for icosphere index
    let t_ico_idx: Vec<i32> = (0..n_tris as i32).collect();

    // Masks: all zero
    let t_divided = vec![0i32; n_tris as usize];
    let t_deactivated = vec![0i32; n_tris as usize];
    let t_to_divide_mask = vec![0i32; n_tris as usize];
    let t_to_merge_mask = vec![0u32; n_tris as usize];

    // All vertices need initial update
    let v_update_mask = vec![1i32; n_verts as usize];

    CesState {
        n_tris,
        n_verts,
        n_deactivated_tris: 0,
        n_divided: 0,
        start_idx: 0,
        v_pos: compute_utils::create_storage_buffer(rd, &v_pos_flat),
        t_abc: compute_utils::create_storage_buffer(rd, &t_abc_flat),
        t_lv: compute_utils::create_storage_buffer(rd, &t_lv),
        t_divided: compute_utils::create_storage_buffer(rd, &t_divided),
        t_deactivated: compute_utils::create_storage_buffer(rd, &t_deactivated),
        t_neight_ab: compute_utils::create_storage_buffer(rd, &NEIGHT_AB),
        t_neight_bc: compute_utils::create_storage_buffer(rd, &NEIGHT_BC),
        t_neight_ca: compute_utils::create_storage_buffer(rd, &NEIGHT_CA),
        t_ico_idx: compute_utils::create_storage_buffer(rd, &t_ico_idx),
        t_a_t: compute_utils::create_storage_buffer(rd, &t_a_t),
        t_b_t: compute_utils::create_storage_buffer(rd, &t_b_t),
        t_c_t: compute_utils::create_storage_buffer(rd, &t_c_t),
        t_center_t: compute_utils::create_storage_buffer(rd, &t_center_t),
        t_parent: compute_utils::create_storage_buffer(rd, &t_parent),
        v_update_mask: compute_utils::create_storage_buffer(rd, &v_update_mask),
        t_to_divide_mask: compute_utils::create_storage_buffer(rd, &t_to_divide_mask),
        t_to_merge_mask: compute_utils::create_storage_buffer(rd, &t_to_merge_mask),
        u_n_tris: compute_utils::create_uniform_buffer(rd, &n_tris),
        u_n_verts: compute_utils::create_uniform_buffer(rd, &n_verts),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_icosphere_vertex_count() {
        assert_eq!(VERTICES.len(), 12);
        // Each vertex has 4 components (x, y, z, w)
        for v in &VERTICES {
            assert_eq!(v.len(), 4);
        }
    }

    #[test]
    fn test_icosphere_triangle_count() {
        assert_eq!(TRIANGLES.len(), 20);
        // Each triangle has 4 components (a, b, c, w)
        for t in &TRIANGLES {
            assert_eq!(t.len(), 4);
        }
    }

    #[test]
    fn test_icosphere_vertices_on_unit_sphere() {
        for (i, v) in VERTICES.iter().enumerate() {
            let magnitude = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
            assert!(
                (magnitude - 1.0).abs() < 1e-3,
                "Vertex {} has magnitude {}, expected ~1.0",
                i,
                magnitude
            );
        }
    }

    #[test]
    fn test_icosphere_neighbor_consistency() {
        // For every triangle i, if neight_ab[i] == j, then j must have i as one of
        // its neighbors (ab, bc, or ca). Same check for bc and ca edges.
        let neighbor_arrays = [&NEIGHT_AB, &NEIGHT_BC, &NEIGHT_CA];

        for (edge_name, neighbors) in ["ab", "bc", "ca"].iter().zip(neighbor_arrays.iter()) {
            for (i, &j) in neighbors.iter().enumerate() {
                let j = j as usize;
                let j_has_i = NEIGHT_AB[j] == i as i32
                    || NEIGHT_BC[j] == i as i32
                    || NEIGHT_CA[j] == i as i32;
                assert!(
                    j_has_i,
                    "Triangle {} has neighbor {} via edge {}, but triangle {} does not list {} as a neighbor",
                    i, j, edge_name, j, i
                );
            }
        }
    }

    #[test]
    fn test_icosphere_triangle_indices_valid() {
        // All vertex indices in triangles must be in range [0, 12)
        for (i, t) in TRIANGLES.iter().enumerate() {
            for &idx in &t[..3] {
                assert!(
                    idx >= 0 && idx < 12,
                    "Triangle {} has out-of-range vertex index {}",
                    i,
                    idx
                );
            }
            assert_eq!(t[3], 0, "Triangle {} w component should be 0", i);
        }
    }

    #[test]
    fn test_icosphere_vertex_w_zero() {
        for (i, v) in VERTICES.iter().enumerate() {
            assert_eq!(v[3], 0.0, "Vertex {} w component should be 0.0", i);
        }
    }

    #[test]
    fn test_neighbor_arrays_length() {
        assert_eq!(NEIGHT_AB.len(), 20);
        assert_eq!(NEIGHT_BC.len(), 20);
        assert_eq!(NEIGHT_CA.len(), 20);
    }

    #[test]
    fn test_neighbor_indices_in_range() {
        for &n in NEIGHT_AB
            .iter()
            .chain(NEIGHT_BC.iter())
            .chain(NEIGHT_CA.iter())
        {
            assert!(
                n >= 0 && n < 20,
                "Neighbor index {} out of range [0, 20)",
                n
            );
        }
    }

    #[test]
    fn test_each_triangle_has_three_distinct_neighbors() {
        // Each triangle should have 3 distinct neighbors (one per edge)
        for i in 0..20 {
            let neighbors = [NEIGHT_AB[i], NEIGHT_BC[i], NEIGHT_CA[i]];
            assert_ne!(
                neighbors[0], neighbors[1],
                "Triangle {} has duplicate neighbors",
                i
            );
            assert_ne!(
                neighbors[1], neighbors[2],
                "Triangle {} has duplicate neighbors",
                i
            );
            assert_ne!(
                neighbors[0], neighbors[2],
                "Triangle {} has duplicate neighbors",
                i
            );
            // No triangle is its own neighbor
            for &n in &neighbors {
                assert_ne!(n, i as i32, "Triangle {} lists itself as a neighbor", i);
            }
        }
    }
}
