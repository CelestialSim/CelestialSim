//! Reference mesh for the per-chunk triangular grid (Phase 2, CEL-62).
//!
//! # Canonical vertex order
//!
//! Vertices are enumerated in **row-major order** by `(i, j)` where:
//! - `i` = row index (the `wb` axis), `0 ≤ i ≤ res`
//! - `j` = column index (the `wc` axis), `0 ≤ j ≤ res − i`
//!
//! The linear index `L` for vertex at `(i, j)` is:
//! ```text
//! L(i, j) = i*(2*res + 3 − i)/2 + j
//! ```
//! (Equivalently: `L(i, j) = i*(res+1) − i*(i−1)/2 + j`.)
//!
//! Row `i` starts at `L_row(i) = i*(2*res + 3 − i)/2`, with `res+1−i` vertices.
//!
//! Special corner vertices:
//! - `L = 0`                   → `(i,j) = (0,0)` → corner A: `wa=1, wb=0, wc=0`
//! - `L = res`                 → `(i,j) = (0,res)` → corner C: `wa=0, wb=0, wc=1`
//! - `L = verts_per_chunk(res)−1` → `(i,j) = (res,0)` → corner B: `wa=0, wb=1, wc=0`
//!
//! `ChunkRealize.slang` (Task 7) must enumerate local vertices in this same order
//! and decode `L` using the same formula.
//!
//! # Vertex storage convention
//!
//! Each vertex is stored as `[wa, wb, wc]` — the full barycentric triplet (`wa+wb+wc=1`):
//! ```text
//! wa = (res − i − j) as f32 / res as f32
//! wb = i as f32 / res as f32
//! wc = j as f32 / res as f32
//! ```
//!
//! # Triangle winding
//!
//! Triangles are output in two interleaved passes for each cell `(i, j)` with `i+j < res`:
//! 1. **Upward triangle**: `L(i,j)`, `L(i,j+1)`, `L(i+1,j)` — always present.
//! 2. **Downward triangle** (only when `i+j+1 < res`): `L(i+1,j)`, `L(i,j+1)`, `L(i+1,j+1)`.
//!
//! Both share the same (uniform) winding, ordered so the OUTWARD surface is front
//! under `render_mode cull_back` in `terrain_chunk.gdshader`.
//!
//! # Perimeter skirts (Phase 3 crack fix)
//!
//! Adjacent chunks at different quadtree depths tessellate their shared edge at
//! different resolutions → T-junctions where heights differ → cracks. To hide them
//! we append a **skirt**: a ring of extra vertices at the same edge positions as the
//! interior perimeter, but pushed **radially inward** (toward the planet centre) by a
//! per-chunk depth, connected to the interior edge by triangles. A crack then shows
//! skirt terrain instead of a hole.
//!
//! ## Skirt vertex numbering (mirrored EXACTLY in `ChunkRealize.slang`)
//!
//! Interior verts occupy `L ∈ [0, interior_verts_per_chunk(res))`. Skirt verts are
//! appended at `L = interior + s`, `s ∈ [0, 3*(res+1))`, with
//! `edge = s / (res+1)` and `t = s % (res+1)` (the along-edge position `0..=res`):
//! - `edge 0` (A→B): interior lattice point `(i,j) = (t, 0)`
//! - `edge 1` (B→C): `(i,j) = (res−t, t)`
//! - `edge 2` (C→A): `(i,j) = (0, res−t)`
//!
//! A skirt vert stores the SAME barycentric as its interior edge vert (so its TEX_UV
//! samples the same atlas texel and it shades like the edge); the realize shader drops
//! its *position* inward. Corners are duplicated (each of the 3 edges owns its own
//! ring), so skirt vertices are always DISTINCT pool entries from the interior edge
//! verts they mirror.
//!
//! ## Skirt triangles
//!
//! For each edge and each of the `res` segments `k` (interior verts `E_k`, `E_{k+1}`;
//! skirt verts `S_k`, `S_{k+1}`) two triangles `T1 = (E_k, E_{k+1}, S_{k+1})`,
//! `T2 = (E_k, S_{k+1}, S_k)`. The interior triangle adjacent to every perimeter
//! segment traverses the boundary edge as `E_{k+1} → E_k`; the skirt traverses it the
//! opposite way (`E_k → E_{k+1}`), so the skirt is consistently oriented with the
//! interior — i.e. the same front/back class under `cull_back`.

use crate::chunk_descriptors::{interior_verts_per_chunk, verts_per_chunk};

/// Compute the linear vertex index for lattice point `(i, j)` at resolution `res`.
///
/// Formula: `L(i, j) = i*(2*res + 3 − i)/2 + j`
///
/// Implemented in u64 to avoid u32 overflow for large `res`.
fn vertex_index(res: u32, i: u32, j: u32) -> i32 {
    let i = i as u64;
    let j = j as u64;
    let res = res as u64;
    // i*(2*res + 3 - i) is always even:
    //   i even → even * anything = even
    //   i odd  → (2*res+3-i) = even+3-odd = odd+odd = even (wait: 2*res is even, 3 is odd,
    //            so 2*res+3 is odd; odd - odd = even) → odd * even = even ✓
    let row_start = i * (2 * res + 3 - i) / 2;
    (row_start + j) as i32
}

/// Map a skirt `(edge, t)` position to its mirrored interior lattice point `(i, j)`.
///
/// The three edges traverse the perimeter `A → B → C → A`, each with positions
/// `t ∈ [0, res]`:
/// - `edge 0` (A→B): `(t, 0)`   — `wc = 0`
/// - `edge 1` (B→C): `(res−t, t)` — `wa = 0`
/// - `edge 2` (C→A): `(0, res−t)` — `wb = 0`
///
/// Must match the `ChunkRealize.slang` skirt decode exactly.
fn skirt_edge_ij(res: u32, edge: u32, t: u32) -> (u32, u32) {
    match edge {
        0 => (t, 0),
        1 => (res - t, t),
        _ => (0, res - t),
    }
}

/// Generate the triangular grid vertices and triangle indices for a chunk at resolution `res`.
///
/// Returns `(vertices, indices)` where:
/// - `vertices`: exactly `verts_per_chunk(res)` entries — the `interior_verts_per_chunk(res)`
///   interior grid verts in canonical `(i,j)` row-major order FIRST, then the
///   `3*(res+1)` perimeter skirt verts (see the module docs). Each is `[wa, wb, wc]`
///   with `wa+wb+wc=1`; a skirt vert carries the same barycentric as the interior edge
///   vert it mirrors.
/// - `indices`: exactly `(res*res + 6*res)*3` values — the `res*res` interior triangles
///   FIRST, then the `6*res` skirt triangles, all with **uniform winding** (outward =
///   front) so the surface shader can use backface culling (`render_mode cull_back`).
///
/// # Panics
/// Panics if `res == 0`.
pub fn chunk_grid(res: u32) -> (Vec<[f32; 3]>, Vec<i32>) {
    assert!(res > 0, "chunk_grid: res must be > 0");

    let nv = verts_per_chunk(res) as usize;
    let interior_nv = interior_verts_per_chunk(res) as usize;
    let nt = (res * res + 6 * res) as usize;

    // --- Vertices: interior grid in row-major (i, j) order, then skirt ring ---
    let mut verts: Vec<[f32; 3]> = Vec::with_capacity(nv);
    for i in 0..=res {
        for j in 0..=(res - i) {
            let wa = (res - i - j) as f32 / res as f32;
            let wb = i as f32 / res as f32;
            let wc = j as f32 / res as f32;
            verts.push([wa, wb, wc]);
        }
    }
    debug_assert_eq!(verts.len(), interior_nv, "interior vertex count mismatch");
    // Skirt verts: edge 0/1/2, each t = 0..=res. Same bary as the mirrored interior
    // edge vert (the realize shader drops the *position* radially inward).
    for edge in 0..3u32 {
        for t in 0..=res {
            let (i, j) = skirt_edge_ij(res, edge, t);
            let wa = (res - i - j) as f32 / res as f32;
            let wb = i as f32 / res as f32;
            let wc = j as f32 / res as f32;
            verts.push([wa, wb, wc]);
        }
    }
    debug_assert_eq!(verts.len(), nv, "vertex count mismatch");

    // --- Interior triangle indices ---
    // For each cell (i, j) with i + j < res (uniform winding, outward = front):
    //   Upward  triangle: L(i,j), L(i,j+1), L(i+1,j)
    //   Downward triangle (when i+j+1 < res): L(i+1,j), L(i,j+1), L(i+1,j+1)
    let mut indices: Vec<i32> = Vec::with_capacity(nt * 3);
    // Uniform winding so backface culling works; the order is chosen so the
    // OUTWARD-facing surface is front under the conventional `cull_back` (verified
    // on-screen — the opposite order renders the planet inside-out).
    for i in 0..res {
        for j in 0..(res - i) {
            // Upward triangle (always present for i+j < res)
            indices.push(vertex_index(res, i, j));
            indices.push(vertex_index(res, i, j + 1));
            indices.push(vertex_index(res, i + 1, j));

            // Downward triangle (only when i+j+1 < res), same winding sense.
            if i + j + 1 < res {
                indices.push(vertex_index(res, i + 1, j));
                indices.push(vertex_index(res, i, j + 1));
                indices.push(vertex_index(res, i + 1, j + 1));
            }
        }
    }

    // --- Skirt triangle indices ---
    // Per edge, per segment k: interior verts E_k/E_{k+1}, skirt verts S_k/S_{k+1}.
    // T1 = (E_k, E_{k+1}, S_{k+1}), T2 = (E_k, S_{k+1}, S_k). The interior triangle
    // adjacent to each perimeter segment traverses the boundary edge E_{k+1}→E_k, so
    // the skirt traverses E_k→E_{k+1} — opposite, i.e. consistently oriented (same
    // front/back class) with the interior.
    let skirt_base = interior_nv as i32;
    let per_edge = (res + 1) as i32;
    for edge in 0..3u32 {
        for k in 0..res {
            let (ei, ej) = skirt_edge_ij(res, edge, k);
            let (e1i, e1j) = skirt_edge_ij(res, edge, k + 1);
            let e_k = vertex_index(res, ei, ej);
            let e_k1 = vertex_index(res, e1i, e1j);
            let s_k = skirt_base + edge as i32 * per_edge + k as i32;
            let s_k1 = skirt_base + edge as i32 * per_edge + (k + 1) as i32;

            // T1: E_k, E_{k+1}, S_{k+1}
            indices.push(e_k);
            indices.push(e_k1);
            indices.push(s_k1);
            // T2: E_k, S_{k+1}, S_k
            indices.push(e_k);
            indices.push(s_k1);
            indices.push(s_k);
        }
    }
    debug_assert_eq!(indices.len(), nt * 3, "index count mismatch");

    (verts, indices)
}

/// Build a Godot `ArrayMesh` reference chunk at resolution `res`.
///
/// The mesh contains `verts_per_chunk(res)` vertices (barycentric coords stored as
/// `Vector3(wa, wb, wc)`) and `res*res` triangles, using the canonical `(i,j)` vertex
/// order from [`chunk_grid`]. This mesh is the template for all chunk MultiMesh instances;
/// the vertex shader reads actual 3-D positions from the vertex-pool texture via
/// `INSTANCE_CUSTOM.r` (the slot index).
///
/// Mirrors the `reference_triangle` idiom in `crates/celestialsim/src/faces.rs`.
pub fn reference_chunk_mesh(
    res: u32,
    material: &godot::obj::Gd<godot::classes::Material>,
) -> godot::obj::Gd<godot::classes::ArrayMesh> {
    use godot::classes::mesh::{ArrayType, PrimitiveType};
    use godot::classes::ArrayMesh;
    use godot::prelude::*;

    let (verts_bary, indices_raw) = chunk_grid(res);

    let packed_verts: PackedVector3Array = verts_bary
        .iter()
        .map(|&[wa, wb, wc]| Vector3::new(wa, wb, wc))
        .collect();

    let packed_normals: PackedVector3Array =
        std::iter::repeat(Vector3::new(0.0, 0.0, 1.0))
            .take(verts_bary.len())
            .collect();

    // Phase 4: store each vertex's chunk-local barycentric (wb, wc) in UV so the
    // fragment shader gets an interpolated per-pixel (u, v) to index the detail
    // atlas (tx = u*tile_res, ty = v*tile_res). u==wb axis, v==wc axis — matching
    // ChunkTileBake.slang's texel→bary reconstruction.
    let packed_uvs: PackedVector2Array = verts_bary
        .iter()
        .map(|&[_wa, wb, wc]| Vector2::new(wb, wc))
        .collect();

    let packed_indices: PackedInt32Array = indices_raw.iter().copied().collect();

    let mut arrays = VarArray::new();
    arrays.resize(ArrayType::MAX.ord() as usize, &Variant::nil());
    arrays.set(ArrayType::VERTEX.ord() as usize, &packed_verts.to_variant());
    arrays.set(ArrayType::NORMAL.ord() as usize, &packed_normals.to_variant());
    arrays.set(ArrayType::TEX_UV.ord() as usize, &packed_uvs.to_variant());
    arrays.set(ArrayType::INDEX.ord() as usize, &packed_indices.to_variant());

    let mut mesh = ArrayMesh::new_gd();
    mesh.add_surface_from_arrays(PrimitiveType::TRIANGLES, &arrays);
    mesh.surface_set_material(0, material);
    mesh
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chunk_descriptors::verts_per_chunk;

    // ── Step 2 (failing tests written before implementation) ─────────────────

    /// Vertex and index counts must match the formulas exactly (interior + skirt).
    #[test]
    fn chunk_grid_vertex_and_index_counts() {
        for res in [1u32, 2, 4, 8, 16] {
            let (verts, indices) = chunk_grid(res);
            assert_eq!(
                verts.len() as u32,
                verts_per_chunk(res),
                "res={res}: wrong vertex count (expected {})",
                verts_per_chunk(res)
            );
            // res*res interior triangles + 6*res skirt triangles (2 per segment ×
            // res segments × 3 edges), each 3 indices.
            let want = (res * res + 6 * res) * 3;
            assert_eq!(
                indices.len() as u32,
                want,
                "res={res}: wrong index count (expected {want})"
            );
        }
    }

    /// L=0 must be corner A: [wa=1, wb=0, wc=0].
    #[test]
    fn chunk_grid_vertex_0_is_corner_a() {
        for res in [1u32, 2, 4, 8] {
            let (verts, _) = chunk_grid(res);
            assert_eq!(
                verts[0],
                [1.0_f32, 0.0, 0.0],
                "res={res}: vertex 0 must be corner A (wa=1)"
            );
        }
    }

    /// The canonical last INTERIOR vertex (L = interior_verts_per_chunk−1) is
    /// (i,j)=(res,0) = corner B: [wa=0,wb=1,wc=0]. (The pool's true last vertex is now
    /// a skirt vert; the interior block must keep its old ordering.)
    #[test]
    fn chunk_grid_last_interior_vertex_is_corner_b() {
        for res in [1u32, 2, 4, 8] {
            let (verts, _) = chunk_grid(res);
            let last_interior = verts[interior_verts_per_chunk(res) as usize - 1];
            assert_eq!(
                last_interior,
                [0.0_f32, 1.0, 0.0],
                "res={res}: last interior vertex must be corner B (wb=1)"
            );
        }
    }

    /// Spot-check specific L↔(i,j) mappings that Task 7's shader must mirror.
    #[test]
    fn chunk_grid_canonical_l_mapping() {
        let res = 4u32;
        let (verts, _) = chunk_grid(res);

        // L=0: (0,0) → corner A
        assert_eq!(verts[0], [1.0, 0.0, 0.0], "L=0 must be corner A");

        // L=res: (0,res) → corner C [wa=0, wb=0, wc=1]
        assert_eq!(verts[res as usize], [0.0, 0.0, 1.0], "L=res must be corner C");

        // L=res+1: (1,0) → [wa=(res-1)/res, wb=1/res, wc=0]
        let expected_l_1_0 = [(res - 1) as f32 / res as f32, 1.0 / res as f32, 0.0];
        assert_eq!(
            verts[(res + 1) as usize],
            expected_l_1_0,
            "L=res+1 must be (1,0)"
        );

        // L = vertex_index(4, 2, 1) = 2*(2*4+3-2)/2 + 1 = 2*9/2 + 1 = 9 + 1 = 10
        // → (2,1): wa=(4-2-1)/4=0.25, wb=2/4=0.5, wc=1/4=0.25
        let l_2_1 = vertex_index(res, 2, 1) as usize;
        assert_eq!(l_2_1, 10, "L(2,1) should be 10 for res=4");
        assert_eq!(
            verts[l_2_1],
            [0.25_f32, 0.5, 0.25],
            "L(2,1) should be [0.25, 0.5, 0.25]"
        );

        // L = vertex_index(4, 3, 0) = 3*(2*4+3-3)/2 + 0 = 3*8/2 = 12
        // → (3,0): wa=(4-3)/4=0.25, wb=3/4=0.75, wc=0
        let l_3_0 = vertex_index(res, 3, 0) as usize;
        assert_eq!(l_3_0, 12, "L(3,0) should be 12 for res=4");
        assert_eq!(
            verts[l_3_0],
            [0.25_f32, 0.75, 0.0],
            "L(3,0) should be [0.25, 0.75, 0.0]"
        );
    }

    /// All index values must be in `0..vertex_count`.
    #[test]
    fn chunk_grid_indices_in_range() {
        for res in [1u32, 2, 4, 8, 16] {
            let (verts, indices) = chunk_grid(res);
            let nv = verts.len() as i32;
            for &idx in &indices {
                assert!(
                    idx >= 0 && idx < nv,
                    "res={res}: index {idx} out of range [0, {nv})"
                );
            }
        }
    }

    /// Every triangle must have 3 distinct vertex indices (no degenerate triangles).
    #[test]
    fn chunk_grid_triangles_have_distinct_vertices() {
        for res in [1u32, 2, 4, 8] {
            let (_, indices) = chunk_grid(res);
            for (t, tri) in indices.chunks(3).enumerate() {
                assert!(
                    tri[0] != tri[1] && tri[1] != tri[2] && tri[0] != tri[2],
                    "res={res}: triangle {t} is degenerate: {:?}",
                    tri
                );
            }
        }
    }

    /// Barycentric coordinates must satisfy wa+wb+wc=1 and all be in [0,1].
    #[test]
    fn chunk_grid_barycentric_coords_valid() {
        for res in [1u32, 2, 4, 8] {
            let (verts, _) = chunk_grid(res);
            for (l, &[wa, wb, wc]) in verts.iter().enumerate() {
                let sum = wa + wb + wc;
                assert!(
                    (sum - 1.0).abs() < 1e-5,
                    "res={res}, L={l}: wa+wb+wc={sum} ≠ 1"
                );
                assert!(wa >= 0.0 && wb >= 0.0 && wc >= 0.0,
                    "res={res}, L={l}: negative coord [{wa},{wb},{wc}]");
            }
        }
    }

    /// Winding is UNIFORM across every INTERIOR triangle (upward and downward) — the
    /// invariant that lets the surface shader use `render_mode cull_back` instead
    /// of `cull_disabled`. A regression to mixed winding would reintroduce the
    /// ~half-the-mesh-culled bug. (The concrete sign is negative here; the
    /// outward-vs-inward facing is verified on-screen, not by this test.)
    ///
    /// Skirt triangles are degenerate in the (wb, wc) plane (a skirt vert shares its
    /// edge vert's barycentric), so this 2-D test only covers the interior block;
    /// [`chunk_grid_consistent_orientation`] checks the skirt's winding combinatorially.
    #[test]
    fn chunk_grid_uniform_winding() {
        let res = 4u32;
        let (verts, indices) = chunk_grid(res);

        // 2-D signed area in the (wb, wc) plane.  Positive = CCW, negative = CW.
        let signed_area = |a: usize, b: usize, c: usize| -> f32 {
            let (wb_a, wc_a) = (verts[a][1], verts[a][2]);
            let (wb_b, wc_b) = (verts[b][1], verts[b][2]);
            let (wb_c, wc_c) = (verts[c][1], verts[c][2]);
            (wb_b - wb_a) * (wc_c - wc_a) - (wb_c - wb_a) * (wc_b - wc_a)
        };

        // Walk the interior triangles (first res*res of the index buffer), not the
        // formula, so the test guards the real winding the GPU sees. All must share
        // the same sign (uniform winding).
        let interior_tris = (res * res) as usize;
        for t in indices[..interior_tris * 3].chunks_exact(3) {
            let (a, b, c) = (t[0] as usize, t[1] as usize, t[2] as usize);
            let area = signed_area(a, b, c);
            assert!(area < 0.0, "interior triangle {t:?} must share the uniform winding sign, got {area}");
        }
    }

    /// The whole mesh (interior + skirt) is a CONSISTENTLY ORIENTED triangle mesh:
    /// every undirected edge shared by two triangles is traversed in OPPOSITE
    /// directions by them. This is exactly the property `cull_back` relies on, and —
    /// unlike the 2-D area test — it holds for the skirt walls (which are degenerate
    /// in barycentric space). It also guarantees the skirt is in the SAME orientation
    /// class as the interior (their shared perimeter edges cancel), so if the interior
    /// renders front-facing the skirt does too.
    #[test]
    fn chunk_grid_consistent_orientation() {
        use std::collections::HashMap;
        for res in [1u32, 2, 4, 8, 16] {
            let (_, indices) = chunk_grid(res);
            // undirected edge {min,max} -> count of each directed orientation.
            // +1 for a->b with a<b, -1 for a->b with a>b. A 2-triangle edge must sum to 0.
            let mut edges: HashMap<(i32, i32), i32> = HashMap::new();
            let mut uses: HashMap<(i32, i32), u32> = HashMap::new();
            let mut bump = |a: i32, b: i32, edges: &mut HashMap<(i32, i32), i32>, uses: &mut HashMap<(i32, i32), u32>| {
                let key = (a.min(b), a.max(b));
                *uses.entry(key).or_insert(0) += 1;
                *edges.entry(key).or_insert(0) += if a < b { 1 } else { -1 };
            };
            for t in indices.chunks_exact(3) {
                bump(t[0], t[1], &mut edges, &mut uses);
                bump(t[1], t[2], &mut edges, &mut uses);
                bump(t[2], t[0], &mut edges, &mut uses);
            }
            for (key, &n) in &uses {
                assert!(n <= 2, "res={res}: edge {key:?} used {n}× (non-manifold)");
                if n == 2 {
                    assert_eq!(
                        edges[key], 0,
                        "res={res}: edge {key:?} used by 2 triangles with the SAME orientation \
                         (inconsistent winding)"
                    );
                }
            }
        }
    }

    /// The interior block (`L < interior_verts_per_chunk`) is byte-identical to the
    /// pre-skirt grid: same verts, same order. The realize `L→(i,j)` decode and the
    /// GPU readback reference depend on this.
    #[test]
    fn chunk_grid_interior_block_unchanged() {
        for res in [1u32, 2, 4, 8, 16] {
            let (verts, _) = chunk_grid(res);
            let interior = interior_verts_per_chunk(res) as usize;
            // Regenerate the interior independently and compare.
            let mut expected: Vec<[f32; 3]> = Vec::new();
            for i in 0..=res {
                for j in 0..=(res - i) {
                    expected.push([
                        (res - i - j) as f32 / res as f32,
                        i as f32 / res as f32,
                        j as f32 / res as f32,
                    ]);
                }
            }
            assert_eq!(&verts[..interior], &expected[..], "res={res}: interior block changed");
        }
    }

    /// Each skirt vert carries the barycentric of its mirrored interior edge vert,
    /// and its (edge, t) decode matches `ChunkRealize.slang`. Edge `e` position `t`
    /// must lie on edge `e` (the off-edge coordinate is 0).
    #[test]
    fn chunk_grid_skirt_verts_have_edge_bary() {
        for res in [1u32, 2, 4, 8] {
            let (verts, _) = chunk_grid(res);
            let base = interior_verts_per_chunk(res) as usize;
            let per_edge = (res + 1) as usize;
            for edge in 0..3u32 {
                for t in 0..=res {
                    let l = base + edge as usize * per_edge + t as usize;
                    let [wa, wb, wc] = verts[l];
                    let (i, j) = super::skirt_edge_ij(res, edge, t);
                    let exp = [
                        (res - i - j) as f32 / res as f32,
                        i as f32 / res as f32,
                        j as f32 / res as f32,
                    ];
                    assert_eq!(verts[l], exp, "res={res} edge={edge} t={t}: skirt bary mismatch");
                    // Off-edge coordinate is zero: edge0→wc, edge1→wa, edge2→wb.
                    match edge {
                        0 => assert_eq!(wc, 0.0, "edge0 must have wc=0"),
                        1 => assert_eq!(wa, 0.0, "edge1 must have wa=0"),
                        _ => assert_eq!(wb, 0.0, "edge2 must have wb=0"),
                    }
                }
            }
        }
    }
}
