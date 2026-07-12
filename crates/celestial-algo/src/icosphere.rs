//! CPU icosphere generation with per-vertex subdivision-level metadata.
//!
//! A base icosahedron (12 vertices, 20 faces) is subdivided `subdivisions`
//! times using 1->4 midpoint splitting. Edge midpoints are shared between
//! adjacent faces via a cache so the resulting mesh is watertight.

use godot::builtin::Vector3;
use std::collections::HashMap;

/// Raw (un-normalised) golden-ratio icosahedron vertices. `T == (1+sqrt(5))/2`.
pub const T: f32 = 1.618_034_f32;
pub const BASE_VERTICES: [[f32; 3]; 12] = [
    [-1.0, T, 0.0],
    [1.0, T, 0.0],
    [-1.0, -T, 0.0],
    [1.0, -T, 0.0],
    [0.0, -1.0, T],
    [0.0, 1.0, T],
    [0.0, -1.0, -T],
    [0.0, 1.0, -T],
    [T, 0.0, -1.0],
    [T, 0.0, 1.0],
    [-T, 0.0, -1.0],
    [-T, 0.0, 1.0],
];

/// The 20 base faces as ordered base-vertex triples. Winding is consistent
/// (outward CCW); `topology` relies on the ordering to resolve seam neighbours.
pub const BASE_FACES: [[u32; 3]; 20] = [
    [0, 11, 5],
    [0, 5, 1],
    [0, 1, 7],
    [0, 7, 10],
    [0, 10, 11],
    [1, 5, 9],
    [5, 11, 4],
    [11, 10, 2],
    [10, 7, 6],
    [7, 1, 8],
    [3, 9, 4],
    [3, 4, 2],
    [3, 2, 6],
    [3, 6, 8],
    [3, 8, 9],
    [4, 9, 5],
    [2, 4, 11],
    [6, 2, 10],
    [8, 6, 7],
    [9, 8, 1],
];

/// Unit-sphere position of base vertex `v`.
pub fn base_vertex(v: u32) -> Vector3 {
    let [x, y, z] = BASE_VERTICES[v as usize];
    Vector3::new(x, y, z).normalized()
}

/// A single vertex of the generated celestial mesh.
#[derive(Clone, Copy, Debug)]
pub struct CelVertex {
    /// Position on the sphere surface, already scaled by the requested radius.
    pub position: Vector3,
    /// Subdivision level at which this vertex was created.
    /// `0` is a base-icosahedron vertex; `n` is a midpoint introduced on the
    /// n-th subdivision pass.
    pub level: u32,
    /// Global index of this vertex within [`CelMesh::vertices`]. Assigned in
    /// creation order, so it is stable for a given `(radius, subdivisions)`.
    pub global_index: u32,
}

/// CPU-generated icosphere plus the data needed to visualise subdivision levels.
pub struct CelMesh {
    pub vertices: Vec<CelVertex>,
    /// Flat triangle index list (3 consecutive entries per triangle) into
    /// [`CelMesh::vertices`].
    pub indices: Vec<u32>,
    /// Highest subdivision level present (== `subdivisions`).
    pub max_level: u32,
}

impl CelMesh {
    /// Generate an icosphere of the given `radius`, subdivided `subdivisions`
    /// times. `subdivisions == 0` yields the raw icosahedron.
    pub fn generate(radius: f32, subdivisions: u32) -> Self {
        // Golden-ratio icosahedron vertices (unit sphere after normalisation).
        let mut positions: Vec<Vector3> = (0..12).map(base_vertex).collect();

        // Level of each vertex; the 12 base vertices are level 0.
        let mut levels: Vec<u32> = vec![0; positions.len()];

        let mut faces: Vec<[u32; 3]> = BASE_FACES.to_vec();

        for level in 1..=subdivisions {
            // Cache of created midpoints, keyed by the ordered endpoint pair.
            let mut midpoints: HashMap<(u32, u32), u32> = HashMap::new();
            let mut next_faces: Vec<[u32; 3]> = Vec::with_capacity(faces.len() * 4);

            for &[a, b, c] in &faces {
                let ab = midpoint(a, b, level, &mut positions, &mut levels, &mut midpoints);
                let bc = midpoint(b, c, level, &mut positions, &mut levels, &mut midpoints);
                let ca = midpoint(c, a, level, &mut positions, &mut levels, &mut midpoints);

                next_faces.push([a, ab, ca]);
                next_faces.push([b, bc, ab]);
                next_faces.push([c, ca, bc]);
                next_faces.push([ab, bc, ca]);
            }

            faces = next_faces;
        }

        // Assemble the vertex array, scaling unit positions by `radius` and
        // recording the global index of each vertex.
        let vertices: Vec<CelVertex> = positions
            .iter()
            .zip(levels.iter())
            .enumerate()
            .map(|(i, (&pos, &level))| CelVertex {
                position: pos * radius,
                level,
                global_index: i as u32,
            })
            .collect();

        let indices: Vec<u32> = faces.into_iter().flatten().collect();

        CelMesh {
            vertices,
            indices,
            max_level: subdivisions,
        }
    }
}

/// Return the index of the (shared) midpoint between vertices `a` and `b`,
/// creating it on the unit sphere if it does not exist yet.
fn midpoint(
    a: u32,
    b: u32,
    level: u32,
    positions: &mut Vec<Vector3>,
    levels: &mut Vec<u32>,
    cache: &mut HashMap<(u32, u32), u32>,
) -> u32 {
    let key = if a < b { (a, b) } else { (b, a) };
    if let Some(&idx) = cache.get(&key) {
        return idx;
    }

    let mid = ((positions[a as usize] + positions[b as usize]) * 0.5).normalized();
    let idx = positions.len() as u32;
    positions.push(mid);
    levels.push(level);
    cache.insert(key, idx);
    idx
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn icosahedron_base_counts() {
        let mesh = CelMesh::generate(1.0, 0);
        assert_eq!(mesh.vertices.len(), 12);
        assert_eq!(mesh.indices.len(), 20 * 3);
        assert!(mesh.vertices.iter().all(|v| v.level == 0));
    }

    #[test]
    fn three_subdivisions_counts() {
        let mesh = CelMesh::generate(1.0, 3);
        // V = 10 * 4^n + 2, F = 20 * 4^n
        assert_eq!(mesh.vertices.len(), 10 * 64 + 2); // 642
        assert_eq!(mesh.indices.len() / 3, 20 * 64); // 1280 triangles
        assert_eq!(mesh.max_level, 3);
    }

    #[test]
    fn global_index_matches_position() {
        let mesh = CelMesh::generate(2.0, 2);
        for (i, v) in mesh.vertices.iter().enumerate() {
            assert_eq!(v.global_index as usize, i);
        }
    }

    #[test]
    fn radius_is_applied() {
        let mesh = CelMesh::generate(5.0, 1);
        for v in &mesh.vertices {
            assert!((v.position.length() - 5.0).abs() < 1e-4);
        }
    }
}
