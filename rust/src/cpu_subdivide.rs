#[allow(unused_imports)]
use crate::initial_state::{TRIANGLES, VERTICES};

/// CPU-only icosphere subdivision (1→4 split) WITHOUT edge deduplication.
///
/// Matches the GPU shader behavior (DivideLOD.slang with preciseNormals=false):
/// each triangle gets 3 new midpoint vertices (no sharing with neighbors).
/// After subdivision, vertices are normalized to the unit sphere surface.
pub fn cpu_subdivide_once(
    vertices: &[[f32; 4]],
    triangles: &[[i32; 4]],
) -> (Vec<[f32; 4]>, Vec<[i32; 4]>) {
    let n = triangles.len();
    let n_verts = vertices.len() as i32;

    // New child triangles
    let mut new_triangles: Vec<[i32; 4]> = Vec::with_capacity(n * 4);
    // New midpoint positions: 3 per triangle
    let mut new_vertex_positions: Vec<[f32; 4]> = Vec::with_capacity(n * 3);

    for div_index in 0..n {
        let [a, b, c, _] = triangles[div_index];

        // Shader logic: v_idx_start = nVerts + 3 * divTrisIndex
        let v_idx_start = n_verts + 3 * div_index as i32;
        let mid_ab = v_idx_start;
        let mid_bc = v_idx_start + 1;
        let mid_ca = v_idx_start + 2;

        // Compute midpoint positions: (v[a] + v[b]) / 2
        let va = vertices[a as usize];
        let vb = vertices[b as usize];
        let vc = vertices[c as usize];

        new_vertex_positions.push([
            (va[0] + vb[0]) * 0.5,
            (va[1] + vb[1]) * 0.5,
            (va[2] + vb[2]) * 0.5,
            0.0,
        ]);
        new_vertex_positions.push([
            (vb[0] + vc[0]) * 0.5,
            (vb[1] + vc[1]) * 0.5,
            (vb[2] + vc[2]) * 0.5,
            0.0,
        ]);
        new_vertex_positions.push([
            (vc[0] + va[0]) * 0.5,
            (vc[1] + va[1]) * 0.5,
            (vc[2] + va[2]) * 0.5,
            0.0,
        ]);

        // 4 child triangles (matches shader add_triangles):
        // child 0: (a, midAB, midCA)
        new_triangles.push([a, mid_ab, mid_ca, 0]);
        // child 1: (midAB, b, midBC)
        new_triangles.push([mid_ab, b, mid_bc, 0]);
        // child 2: (midCA, midBC, c)
        new_triangles.push([mid_ca, mid_bc, c, 0]);
        // child 3: (midBC, midCA, midAB) — center triangle
        new_triangles.push([mid_bc, mid_ca, mid_ab, 0]);
    }

    // Build final vertex array: original + new midpoints
    let mut out_vertices: Vec<[f32; 4]> = vertices.to_vec();
    out_vertices.extend(new_vertex_positions);

    // Normalize all vertices to unit sphere
    for v in out_vertices.iter_mut() {
        let mag = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
        if mag > 0.0 {
            v[0] /= mag;
            v[1] /= mag;
            v[2] /= mag;
        }
        v[3] = 0.0;
    }

    (out_vertices, new_triangles)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_subdivide_counts() {
        let (verts, tris) = cpu_subdivide_once(&VERTICES, &TRIANGLES);
        assert_eq!(tris.len(), 80, "Expected 80 triangles after 1 subdivision");
        assert_eq!(
            verts.len(),
            72,
            "Expected 72 vertices after 1 subdivision (12 + 20*3 midpoints, no dedup)"
        );
    }

    #[test]
    fn test_cpu_subdivide_vertices_on_sphere() {
        let (verts, _) = cpu_subdivide_once(&VERTICES, &TRIANGLES);
        for (i, v) in verts.iter().enumerate() {
            let mag = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
            assert!(
                (mag - 1.0).abs() < 1e-5,
                "Vertex {} has magnitude {}, expected ~1.0",
                i,
                mag,
            );
        }
    }

    #[test]
    fn test_cpu_subdivide_no_degenerate_triangles() {
        let (_, tris) = cpu_subdivide_once(&VERTICES, &TRIANGLES);
        for (i, t) in tris.iter().enumerate() {
            assert!(
                t[0] != t[1] && t[1] != t[2] && t[0] != t[2],
                "Triangle {} has duplicate vertex indices: [{}, {}, {}]",
                i,
                t[0],
                t[1],
                t[2],
            );
        }
    }

    /// C# reference vertex positions after 1 subdivision with preciseNormals=false,
    /// then sphere normalization (radius=1.0).
    /// Extracted from C# CesDivLOD + SphereTerrain GPU shader output.
    #[rustfmt::skip]
    const CSHARP_REF_VERTICES: [[f32; 4]; 72] = [
        [-0.525686502, 0.0, 0.850678265, 0.0],       // 0
        [0.525686502, 0.0, 0.850678265, 0.0],        // 1
        [-0.525686502, 0.0, -0.850678265, 0.0],      // 2
        [0.525686502, 0.0, -0.850678265, 0.0],       // 3
        [0.0, 0.850678265, 0.525686502, 0.0],        // 4
        [0.0, 0.850678265, -0.525686502, 0.0],       // 5
        [0.0, -0.850678265, 0.525686502, 0.0],       // 6
        [0.0, -0.850678265, -0.525686502, 0.0],      // 7
        [0.850678265, 0.525686502, 0.0, 0.0],        // 8
        [-0.850678265, 0.525686502, 0.0, 0.0],       // 9
        [0.850678265, -0.525686502, 0.0, 0.0],       // 10
        [-0.850678265, -0.525686502, 0.0, 0.0],      // 11
        [-0.30899331, 0.500020206, 0.809013486, 0.0], // 12
        [0.30899331, 0.500020206, 0.809013486, 0.0],  // 13
        [0.0, 0.0, 0.99999994, 0.0],                  // 14
        [-0.809013486, 0.30899331, 0.500020206, 0.0],  // 15
        [-0.500020206, 0.809013486, 0.30899331, 0.0],  // 16
        [-0.30899331, 0.500020206, 0.809013486, 0.0],  // 17
        [-0.500020206, 0.809013486, -0.30899331, 0.0], // 18
        [0.0, 0.99999994, 0.0, 0.0],                   // 19
        [-0.500020206, 0.809013486, 0.30899331, 0.0],  // 20
        [0.0, 0.99999994, 0.0, 0.0],                   // 21
        [0.500020206, 0.809013486, -0.30899331, 0.0],  // 22
        [0.500020206, 0.809013486, 0.30899331, 0.0],   // 23
        [0.500020206, 0.809013486, 0.30899331, 0.0],   // 24
        [0.809013486, 0.30899331, 0.500020206, 0.0],   // 25
        [0.30899331, 0.500020206, 0.809013486, 0.0],   // 26
        [0.99999994, 0.0, 0.0, 0.0],                   // 27
        [0.809013486, -0.30899331, 0.500020206, 0.0],  // 28
        [0.809013486, 0.30899331, 0.500020206, 0.0],   // 29
        [0.809013486, 0.30899331, -0.500020206, 0.0],  // 30
        [0.809013486, -0.30899331, -0.500020206, 0.0], // 31
        [0.99999994, 0.0, 0.0, 0.0],                   // 32
        [0.30899331, 0.500020206, -0.809013486, 0.0],  // 33
        [0.809013486, 0.30899331, -0.500020206, 0.0],  // 34
        [0.500020206, 0.809013486, -0.30899331, 0.0],  // 35
        [-0.30899331, 0.500020206, -0.809013486, 0.0], // 36
        [0.0, 0.0, -0.99999994, 0.0],                  // 37
        [0.30899331, 0.500020206, -0.809013486, 0.0],  // 38
        [-0.30899331, -0.500020206, -0.809013486, 0.0],// 39
        [0.30899331, -0.500020206, -0.809013486, 0.0], // 40
        [0.0, 0.0, -0.99999994, 0.0],                  // 41
        [0.500020206, -0.809013486, -0.30899331, 0.0], // 42
        [0.809013486, -0.30899331, -0.500020206, 0.0], // 43
        [0.30899331, -0.500020206, -0.809013486, 0.0], // 44
        [0.0, -0.99999994, 0.0, 0.0],                  // 45
        [0.500020206, -0.809013486, 0.30899331, 0.0],  // 46
        [0.500020206, -0.809013486, -0.30899331, 0.0], // 47
        [-0.500020206, -0.809013486, -0.30899331, 0.0],// 48
        [-0.500020206, -0.809013486, 0.30899331, 0.0], // 49
        [0.0, -0.99999994, 0.0, 0.0],                  // 50
        [-0.809013486, -0.30899331, 0.500020206, 0.0], // 51
        [-0.30899331, -0.500020206, 0.809013486, 0.0], // 52
        [-0.500020206, -0.809013486, 0.30899331, 0.0], // 53
        [0.0, 0.0, 0.99999994, 0.0],                   // 54
        [0.30899331, -0.500020206, 0.809013486, 0.0],  // 55
        [-0.30899331, -0.500020206, 0.809013486, 0.0], // 56
        [0.30899331, -0.500020206, 0.809013486, 0.0],  // 57
        [0.809013486, -0.30899331, 0.500020206, 0.0],  // 58
        [0.500020206, -0.809013486, 0.30899331, 0.0],  // 59
        [-0.809013486, 0.30899331, 0.500020206, 0.0],  // 60
        [-0.809013486, -0.30899331, 0.500020206, 0.0], // 61
        [-0.99999994, 0.0, 0.0, 0.0],                  // 62
        [-0.99999994, 0.0, 0.0, 0.0],                  // 63
        [-0.809013486, -0.30899331, -0.500020206, 0.0],// 64
        [-0.809013486, 0.30899331, -0.500020206, 0.0], // 65
        [-0.809013486, 0.30899331, -0.500020206, 0.0], // 66
        [-0.30899331, 0.500020206, -0.809013486, 0.0], // 67
        [-0.500020206, 0.809013486, -0.30899331, 0.0], // 68
        [-0.30899331, -0.500020206, -0.809013486, 0.0],// 69
        [-0.809013486, -0.30899331, -0.500020206, 0.0],// 70
        [-0.500020206, -0.809013486, -0.30899331, 0.0],// 71
    ];

    /// C# reference child triangle indices after 1 subdivision.
    /// These are the 80 child triangles (indices 20-99 in the full buffer,
    /// but stored here as 0-79 since we only output children).
    #[rustfmt::skip]
    const CSHARP_REF_TRIANGLES: [[i32; 4]; 80] = [
        [0, 12, 14, 0],   // 0 (was T20)
        [12, 4, 13, 0],   // 1
        [14, 13, 1, 0],   // 2
        [13, 14, 12, 0],  // 3
        [0, 15, 17, 0],   // 4
        [15, 9, 16, 0],   // 5
        [17, 16, 4, 0],   // 6
        [16, 17, 15, 0],  // 7
        [9, 18, 20, 0],   // 8
        [18, 5, 19, 0],   // 9
        [20, 19, 4, 0],   // 10
        [19, 20, 18, 0],  // 11
        [4, 21, 23, 0],   // 12
        [21, 5, 22, 0],   // 13
        [23, 22, 8, 0],   // 14
        [22, 23, 21, 0],  // 15
        [4, 24, 26, 0],   // 16
        [24, 8, 25, 0],   // 17
        [26, 25, 1, 0],   // 18
        [25, 26, 24, 0],  // 19
        [8, 27, 29, 0],   // 20
        [27, 10, 28, 0],  // 21
        [29, 28, 1, 0],   // 22
        [28, 29, 27, 0],  // 23
        [8, 30, 32, 0],   // 24
        [30, 3, 31, 0],   // 25
        [32, 31, 10, 0],  // 26
        [31, 32, 30, 0],  // 27
        [5, 33, 35, 0],   // 28
        [33, 3, 34, 0],   // 29
        [35, 34, 8, 0],   // 30
        [34, 35, 33, 0],  // 31
        [5, 36, 38, 0],   // 32
        [36, 2, 37, 0],   // 33
        [38, 37, 3, 0],   // 34
        [37, 38, 36, 0],  // 35
        [2, 39, 41, 0],   // 36
        [39, 7, 40, 0],   // 37
        [41, 40, 3, 0],   // 38
        [40, 41, 39, 0],  // 39
        [7, 42, 44, 0],   // 40
        [42, 10, 43, 0],  // 41
        [44, 43, 3, 0],   // 42
        [43, 44, 42, 0],  // 43
        [7, 45, 47, 0],   // 44
        [45, 6, 46, 0],   // 45
        [47, 46, 10, 0],  // 46
        [46, 47, 45, 0],  // 47
        [7, 48, 50, 0],   // 48
        [48, 11, 49, 0],  // 49
        [50, 49, 6, 0],   // 50
        [49, 50, 48, 0],  // 51
        [11, 51, 53, 0],  // 52
        [51, 0, 52, 0],   // 53
        [53, 52, 6, 0],   // 54
        [52, 53, 51, 0],  // 55
        [0, 54, 56, 0],   // 56
        [54, 1, 55, 0],   // 57
        [56, 55, 6, 0],   // 58
        [55, 56, 54, 0],  // 59
        [6, 57, 59, 0],   // 60
        [57, 1, 58, 0],   // 61
        [59, 58, 10, 0],  // 62
        [58, 59, 57, 0],  // 63
        [9, 60, 62, 0],   // 64
        [60, 0, 61, 0],   // 65
        [62, 61, 11, 0],  // 66
        [61, 62, 60, 0],  // 67
        [9, 63, 65, 0],   // 68
        [63, 11, 64, 0],  // 69
        [65, 64, 2, 0],   // 70
        [64, 65, 63, 0],  // 71
        [9, 66, 68, 0],   // 72
        [66, 2, 67, 0],   // 73
        [68, 67, 5, 0],   // 74
        [67, 68, 66, 0],  // 75
        [7, 69, 71, 0],   // 76
        [69, 2, 70, 0],   // 77
        [71, 70, 11, 0],  // 78
        [70, 71, 69, 0],  // 79
    ];

    /// Verify that the CPU Rust subdivision matches C# reference vertex positions.
    #[test]
    fn test_cpu_subdivide_matches_csharp_vertices() {
        let (verts, _) = cpu_subdivide_once(&VERTICES, &TRIANGLES);
        assert_eq!(verts.len(), CSHARP_REF_VERTICES.len());

        for (i, (rust_v, csharp_v)) in verts.iter().zip(CSHARP_REF_VERTICES.iter()).enumerate() {
            for comp in 0..3 {
                let diff = (rust_v[comp] - csharp_v[comp]).abs();
                assert!(
                    diff < 1e-4,
                    "Vertex {} component {} mismatch: rust={} csharp={} diff={}",
                    i,
                    comp,
                    rust_v[comp],
                    csharp_v[comp],
                    diff,
                );
            }
        }
    }

    /// Verify that the CPU Rust subdivision matches C# reference triangle indices.
    #[test]
    fn test_cpu_subdivide_matches_csharp_triangles() {
        let (_, tris) = cpu_subdivide_once(&VERTICES, &TRIANGLES);
        assert_eq!(tris.len(), CSHARP_REF_TRIANGLES.len());

        for (i, (rust_t, csharp_t)) in tris.iter().zip(CSHARP_REF_TRIANGLES.iter()).enumerate() {
            assert_eq!(
                rust_t, csharp_t,
                "Triangle {} mismatch: rust={:?} csharp={:?}",
                i, rust_t, csharp_t,
            );
        }
    }
}
