use godot::classes::RenderingDevice;
use godot::obj::Gd;

use crate::buffer_info::BufferInfo;
use crate::compute_utils;
use crate::compute_utils::ComputePipeline;
use crate::state::{CesState, Triangle};

const SHADER_PATH: &str = "res://addons/celestial_sim/shaders/DivideLOD.slang";

pub struct DivShader {
    pipeline: ComputePipeline,
}

impl DivShader {
    pub fn new(rd: &mut Gd<RenderingDevice>) -> Self {
        Self {
            pipeline: ComputePipeline::new(rd, SHADER_PATH),
        }
    }

    pub fn dispose_direct(&mut self, rd: &mut Gd<RenderingDevice>) {
        self.pipeline.dispose_direct(rd);
    }
}

/// Returns `[min(a,b), max(a,b)]`.
fn sort_pair(a: i32, b: i32) -> [i32; 2] {
    if a < b {
        [a, b]
    } else {
        [b, a]
    }
}

/// CPU-side edge deduplication for precise normals mode.
///
/// For each triangle marked for division, assigns new vertex indices for edge
/// midpoints. When two adjacent triangles are both being divided, the shared
/// edge reuses the same midpoint vertex index.
///
/// Returns the modified `tabc` array (with child triangles appended) and the
/// number of new vertices created.
fn compute_new_indices(rd: &mut Gd<RenderingDevice>, state: &CesState) -> (Vec<Triangle>, u32) {
    let to_div_mask = state.get_t_to_divide_mask(rd);
    let mut tabc = state.get_t_abc(rd);
    let start_vindex = state.n_verts as i32;
    let start_tindex = state.n_tris as usize;

    let neight_ab: Vec<i32> = compute_utils::convert_buffer_to_vec(rd, &state.t_neight_ab);
    let neight_bc: Vec<i32> = compute_utils::convert_buffer_to_vec(rd, &state.t_neight_bc);
    let neight_ca: Vec<i32> = compute_utils::convert_buffer_to_vec(rd, &state.t_neight_ca);

    let n = to_div_mask.len();
    let mut local_div = vec![false; n];
    // Flat array simulating [n*4][3]: index = (tri_idx * 4 + edge) * 1, stored as [tri_idx][edge]
    // We use a Vec of [i32; 3] indexed by triangle index.
    let mut added_idxs = vec![[0i32; 3]; n];

    let mut vindex = start_vindex;
    let mut tdiv_index: usize = 0;

    for i in 0..n {
        if to_div_mask[i] == 0 {
            continue;
        }

        let a = tabc[i].a;
        let b = tabc[i].b;
        let c = tabc[i].c;

        // Process edge AB
        let new_vertex_ab = {
            let neigh = neight_ab[i] as usize;
            if neigh < n && local_div[neigh] {
                let nab = sort_pair(tabc[neigh].a, tabc[neigh].b);
                let nbc = sort_pair(tabc[neigh].b, tabc[neigh].c);
                let ab = sort_pair(a, b);
                if nab == ab {
                    added_idxs[neigh][0]
                } else if nbc == ab {
                    added_idxs[neigh][1]
                } else {
                    added_idxs[neigh][2]
                }
            } else {
                let v = vindex;
                vindex += 1;
                v
            }
        };

        // Process edge BC
        let new_vertex_bc = {
            let neigh = neight_bc[i] as usize;
            if neigh < n && local_div[neigh] {
                let nab = sort_pair(tabc[neigh].a, tabc[neigh].b);
                let nbc = sort_pair(tabc[neigh].b, tabc[neigh].c);
                let bc = sort_pair(b, c);
                if nab == bc {
                    added_idxs[neigh][0]
                } else if nbc == bc {
                    added_idxs[neigh][1]
                } else {
                    added_idxs[neigh][2]
                }
            } else {
                let v = vindex;
                vindex += 1;
                v
            }
        };

        // Process edge CA
        let new_vertex_ca = {
            let neigh = neight_ca[i] as usize;
            if neigh < n && local_div[neigh] {
                let nab = sort_pair(tabc[neigh].a, tabc[neigh].b);
                let nbc = sort_pair(tabc[neigh].b, tabc[neigh].c);
                let ca = sort_pair(c, a);
                if nab == ca {
                    added_idxs[neigh][0]
                } else if nbc == ca {
                    added_idxs[neigh][1]
                } else {
                    added_idxs[neigh][2]
                }
            } else {
                let v = vindex;
                vindex += 1;
                v
            }
        };

        local_div[i] = true;

        // Write 4 child triangles
        let base = start_tindex + tdiv_index * 4;

        // Ensure tabc is large enough
        while tabc.len() <= base + 3 {
            tabc.push(Triangle {
                a: 0,
                b: 0,
                c: 0,
                w: 0,
            });
        }

        // child 0: (a, midAB, midCA)
        tabc[base] = Triangle {
            a,
            b: new_vertex_ab,
            c: new_vertex_ca,
            w: 0,
        };
        // child 1: (midAB, b, midBC)
        tabc[base + 1] = Triangle {
            a: new_vertex_ab,
            b,
            c: new_vertex_bc,
            w: 0,
        };
        // child 2: (midCA, midBC, c)
        tabc[base + 2] = Triangle {
            a: new_vertex_ca,
            b: new_vertex_bc,
            c,
            w: 0,
        };
        // child 3: (midBC, midCA, midAB) — center triangle
        tabc[base + 3] = Triangle {
            a: new_vertex_bc,
            b: new_vertex_ca,
            c: new_vertex_ab,
            w: 0,
        };

        // Store midpoint indices
        added_idxs[i] = [new_vertex_ab, new_vertex_bc, new_vertex_ca];

        tdiv_index += 1;
    }

    let num_new_verts = (vindex - start_vindex) as u32;
    (tabc, num_new_verts)
}

impl DivShader {
    /// Performs triangle subdivision. Mirrors C# `CesDivLOD.MakeDiv()`.
    ///
    /// Returns the number of triangles added (0 if nothing to divide).
    pub fn make_div(
        &self,
        rd: &mut Gd<RenderingDevice>,
        state: &mut CesState,
        precise_normals: bool,
    ) -> u32 {
        let remove_repeated_verts: u32 = if precise_normals { 1 } else { 0 };

        let to_div_mask = state.get_t_to_divide_mask(rd);
        let divided_mask = state.get_divided_mask(rd);

        let mut indices_to_div: Vec<u32> = Vec::new();
        for i in 0..to_div_mask.len() {
            if to_div_mask[i] != 0 && divided_mask[i] == 0 {
                indices_to_div.push(i as u32);
            }
        }

        let n_tris_to_div = indices_to_div.len() as u32;
        let n_tris_added = 4 * n_tris_to_div;
        let mut n_verts_added = 3 * n_tris_to_div;

        if n_tris_added == 0 {
            return 0;
        }

        let indices_to_div_buf = compute_utils::create_storage_buffer(rd, &indices_to_div);

        // Extend t_abc and optionally compute new indices on CPU
        state
            .t_abc
            .extend_buffer(rd, 4 * std::mem::size_of::<i32>() as u32 * n_tris_added);
        if precise_normals {
            let (new_tabc, deduped_verts) = compute_new_indices(rd, state);
            n_verts_added = deduped_verts;
            // Replace t_abc buffer with the CPU-computed one
            compute_utils::free_rid_on_render_thread(rd, state.t_abc.rid);
            state.t_abc = compute_utils::create_storage_buffer(rd, &new_tabc);
        }

        // Record old counts before updating
        let old_n_tris = state.n_tris;
        let old_n_verts = state.n_verts;

        state.n_tris += n_tris_added;
        state.n_verts += n_verts_added;

        // Extend vertex buffers
        state
            .v_pos
            .extend_buffer(rd, 4 * std::mem::size_of::<f32>() as u32 * n_verts_added);
        state
            .v_update_mask
            .extend_buffer(rd, std::mem::size_of::<i32>() as u32 * n_verts_added);

        // Extend triangle buffers
        let tri_extend = std::mem::size_of::<i32>() as u32 * n_tris_added;
        state.t_lv.extend_buffer(rd, tri_extend);
        state.t_divided.extend_buffer(rd, tri_extend);
        state.t_deactivated.extend_buffer(rd, tri_extend);
        state.t_to_divide_mask.extend_buffer(rd, tri_extend);
        state.t_to_merge_mask.extend_buffer(rd, tri_extend);
        state.t_ico_idx.extend_buffer(rd, tri_extend);
        state.t_neight_ab.extend_buffer(rd, tri_extend);
        state.t_neight_bc.extend_buffer(rd, tri_extend);
        state.t_neight_ca.extend_buffer(rd, tri_extend);
        state.t_a_t.extend_buffer(rd, tri_extend);
        state.t_b_t.extend_buffer(rd, tri_extend);
        state.t_c_t.extend_buffer(rd, tri_extend);
        state.t_center_t.extend_buffer(rd, tri_extend);
        state.t_parent.extend_buffer(rd, tri_extend);

        // Create uniform buffers for shader parameters
        let old_n_tris_buf = compute_utils::create_uniform_buffer(rd, &old_n_tris);
        let old_n_verts_buf = compute_utils::create_uniform_buffer(rd, &old_n_verts);
        let n_tris_to_div_buf = compute_utils::create_uniform_buffer(rd, &n_tris_to_div);
        let n_verts_added_buf = compute_utils::create_uniform_buffer(rd, &n_verts_added);
        let remove_repeated_buf = compute_utils::create_uniform_buffer(rd, &remove_repeated_verts);

        let buffers: Vec<&BufferInfo> = vec![
            &state.v_pos,            // 0
            &state.t_abc,            // 1
            &state.t_divided,        // 2
            &state.t_to_divide_mask, // 3
            &old_n_tris_buf,         // 4
            &old_n_verts_buf,        // 5
            &n_tris_to_div_buf,      // 6
            &n_verts_added_buf,      // 7
            &state.v_update_mask,    // 8
            &state.t_ico_idx,        // 9
            &state.t_neight_ab,      // 10
            &state.t_neight_bc,      // 11
            &state.t_neight_ca,      // 12
            &state.t_a_t,            // 13
            &state.t_b_t,            // 14
            &state.t_c_t,            // 15
            &state.t_center_t,       // 16
            &state.t_parent,         // 17
            &remove_repeated_buf,    // 18
            &state.t_lv,             // 19
            &indices_to_div_buf,     // 20
        ];

        self.pipeline.dispatch(rd, &buffers, n_tris_to_div);

        // Free temporary buffers
        compute_utils::free_rid_on_render_thread(rd, indices_to_div_buf.rid);
        compute_utils::free_rid_on_render_thread(rd, old_n_tris_buf.rid);
        compute_utils::free_rid_on_render_thread(rd, old_n_verts_buf.rid);
        compute_utils::free_rid_on_render_thread(rd, n_tris_to_div_buf.rid);
        compute_utils::free_rid_on_render_thread(rd, n_verts_added_buf.rid);
        compute_utils::free_rid_on_render_thread(rd, remove_repeated_buf.rid);

        n_tris_added
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sort_pair() {
        assert_eq!(sort_pair(5, 3), [3, 5]);
        assert_eq!(sort_pair(1, 7), [1, 7]);
        assert_eq!(sort_pair(4, 4), [4, 4]);
        assert_eq!(sort_pair(-2, 3), [-2, 3]);
        assert_eq!(sort_pair(10, -5), [-5, 10]);
    }

    #[test]
    fn test_compute_new_indices_logic() {
        // Test the CPU vertex deduplication logic without GPU.
        // We simulate the core algorithm inline since compute_new_indices
        // requires a RenderingDevice. The logic is identical.

        // Setup: 2 adjacent triangles sharing edge (1,2), both marked for division.
        // Triangle 0: vertices (0, 1, 2), neighbor AB = 1
        // Triangle 1: vertices (1, 3, 2), neighbor CA = 0 (edge CA of tri1 = (2,1) matches AB of tri0 = (0,1)... wait)
        // Let's be precise:
        // Tri 0: a=0, b=1, c=2   edges: AB=(0,1), BC=(1,2), CA=(2,0)
        // Tri 1: a=1, b=3, c=2   edges: AB=(1,3), BC=(3,2), CA=(2,1)
        // Shared edge is (1,2) which is BC of tri0 and CA of tri1.

        let to_div_mask = vec![1i32, 1];
        let tabc = vec![
            Triangle {
                a: 0,
                b: 1,
                c: 2,
                w: 0,
            },
            Triangle {
                a: 1,
                b: 3,
                c: 2,
                w: 0,
            },
        ];
        let n_verts: u32 = 4;
        let n_tris: u32 = 2;

        // Neighbor arrays:
        // Tri 0: neight_ab=some_other(-1), neight_bc=1, neight_ca=some_other(-1)
        // Tri 1: neight_ab=some_other(-1), neight_bc=some_other(-1), neight_ca=0
        let neight_ab = vec![-1i32, -1];
        let neight_bc = vec![1i32, -1];
        let neight_ca = vec![-1i32, 0];

        let start_vindex = n_verts as i32;
        let start_tindex = n_tris as usize;

        let n = to_div_mask.len();
        let mut local_div = vec![false; n];
        let mut added_idxs = vec![[0i32; 3]; n];
        let mut tabc = tabc;

        let mut vindex = start_vindex;
        let mut tdiv_index: usize = 0;

        for i in 0..n {
            if to_div_mask[i] == 0 {
                continue;
            }

            let a = tabc[i].a;
            let b = tabc[i].b;
            let c = tabc[i].c;

            // Process edge AB
            let new_vertex_ab = {
                let neigh = neight_ab[i];
                if neigh >= 0 && (neigh as usize) < n && local_div[neigh as usize] {
                    let ni = neigh as usize;
                    let nab = sort_pair(tabc[ni].a, tabc[ni].b);
                    let nbc = sort_pair(tabc[ni].b, tabc[ni].c);
                    let ab = sort_pair(a, b);
                    if nab == ab {
                        added_idxs[ni][0]
                    } else if nbc == ab {
                        added_idxs[ni][1]
                    } else {
                        added_idxs[ni][2]
                    }
                } else {
                    let v = vindex;
                    vindex += 1;
                    v
                }
            };

            // Process edge BC
            let new_vertex_bc = {
                let neigh = neight_bc[i];
                if neigh >= 0 && (neigh as usize) < n && local_div[neigh as usize] {
                    let ni = neigh as usize;
                    let nab = sort_pair(tabc[ni].a, tabc[ni].b);
                    let nbc = sort_pair(tabc[ni].b, tabc[ni].c);
                    let bc = sort_pair(b, c);
                    if nab == bc {
                        added_idxs[ni][0]
                    } else if nbc == bc {
                        added_idxs[ni][1]
                    } else {
                        added_idxs[ni][2]
                    }
                } else {
                    let v = vindex;
                    vindex += 1;
                    v
                }
            };

            // Process edge CA
            let new_vertex_ca = {
                let neigh = neight_ca[i];
                if neigh >= 0 && (neigh as usize) < n && local_div[neigh as usize] {
                    let ni = neigh as usize;
                    let nab = sort_pair(tabc[ni].a, tabc[ni].b);
                    let nbc = sort_pair(tabc[ni].b, tabc[ni].c);
                    let ca = sort_pair(c, a);
                    if nab == ca {
                        added_idxs[ni][0]
                    } else if nbc == ca {
                        added_idxs[ni][1]
                    } else {
                        added_idxs[ni][2]
                    }
                } else {
                    let v = vindex;
                    vindex += 1;
                    v
                }
            };

            local_div[i] = true;

            let base = start_tindex + tdiv_index * 4;
            while tabc.len() <= base + 3 {
                tabc.push(Triangle {
                    a: 0,
                    b: 0,
                    c: 0,
                    w: 0,
                });
            }

            tabc[base] = Triangle {
                a,
                b: new_vertex_ab,
                c: new_vertex_ca,
                w: 0,
            };
            tabc[base + 1] = Triangle {
                a: new_vertex_ab,
                b,
                c: new_vertex_bc,
                w: 0,
            };
            tabc[base + 2] = Triangle {
                a: new_vertex_ca,
                b: new_vertex_bc,
                c,
                w: 0,
            };
            tabc[base + 3] = Triangle {
                a: new_vertex_bc,
                b: new_vertex_ca,
                c: new_vertex_ab,
                w: 0,
            };

            added_idxs[i] = [new_vertex_ab, new_vertex_bc, new_vertex_ca];
            tdiv_index += 1;
        }

        let num_new_verts = (vindex - start_vindex) as u32;

        // Without deduplication, 2 triangles * 3 edges = 6 new vertices.
        // With deduplication of the shared edge (1,2), we expect 5 new vertices.
        assert_eq!(
            num_new_verts, 5,
            "Expected 5 new vertices (one shared edge deduped)"
        );

        // Verify we produced 8 child triangles (4 per parent)
        assert_eq!(tdiv_index, 2);
        assert!(tabc.len() >= start_tindex + 8);

        // Verify the shared midpoint: tri0's BC midpoint should equal tri1's CA midpoint
        let tri0_bc_mid = added_idxs[0][1]; // BC of tri 0
        let tri1_ca_mid = added_idxs[1][2]; // CA of tri 1
        assert_eq!(
            tri0_bc_mid, tri1_ca_mid,
            "Shared edge (1,2) should produce the same midpoint vertex"
        );
    }
}
