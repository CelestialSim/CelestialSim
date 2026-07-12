//! Phase 1 debug visualization of the CPU chunked-quadtree selection: turn a
//! `Vec<Chunk>` into a per-LOD-coloured `ArrayMesh` (throwaway CPU mesh — Phase 2
//! replaces this with GPU realize through the `nodes` pipeline). The pure helpers
//! (`lod_color`, `build_debug_mesh`) return plain Rust data so they unit-test
//! without a running Godot engine; the Godot node converts them to engine-backed
//! `Packed*Array`/`Color` and wires them to a `MeshInstance3D` each frame.

use celestial_algo::quadtree::Chunk;
use godot::builtin::Vector3;

/// Distinct linear-RGB colour per LOD level. Same hue spin (`level * 0.137`) and
/// HSV→RGB construction as the existing `terrain_surface` `layer_color`, so the
/// debug palette matches the rest of the project. Returns plain `[f32;3]` (no
/// engine-backed `Color`) so it is testable headlessly.
pub fn lod_color(level: u8) -> [f32; 3] {
    let hue = ((level as f32) * 0.137).fract();
    let h6 = hue * 6.0;
    let c = 0.95 * 0.8;
    let x = c * (1.0 - ((h6 % 2.0) - 1.0).abs());
    let m = 0.95 - c;
    let (r, g, b) = if h6 < 1.0 {
        (c, x, 0.0)
    } else if h6 < 2.0 {
        (x, c, 0.0)
    } else if h6 < 3.0 {
        (0.0, c, x)
    } else if h6 < 4.0 {
        (0.0, x, c)
    } else if h6 < 5.0 {
        (x, 0.0, c)
    } else {
        (c, 0.0, x)
    };
    [r + m, g + m, b + m]
}

/// CPU vertex arrays for one debug frame. Plain `Vec`s so the builder is
/// headlessly testable; the node converts to `Packed*`.
pub struct DebugMeshData {
    pub positions: Vec<Vector3>,
    pub colors: Vec<[f32; 3]>,
    pub indices: Vec<i32>,
}

/// Tessellate one chunk's triangle into `res`×`res` sub-triangles, renormalised to
/// the sphere. This is the (CPU, throwaway) *realize* of a chunk — Phase 2 does it
/// on the GPU. `res` 1 => the chunk itself (one triangle).
fn tessellate(corners: [Vector3; 3], radius: f32, res: u32) -> Vec<[Vector3; 3]> {
    let n = res.max(1);
    let fnn = n as f32;
    let pt = |i: u32, j: u32| {
        let (wa, wb, wc) = ((n - i - j) as f32 / fnn, i as f32 / fnn, j as f32 / fnn);
        let p = corners[0] * wa + corners[1] * wb + corners[2] * wc;
        if radius > 0.0 {
            p.normalized() * radius
        } else {
            p
        }
    };
    let mut tris = Vec::with_capacity((n * n) as usize);
    for i in 0..n {
        for j in 0..(n - i) {
            tris.push([pt(i, j), pt(i + 1, j), pt(i, j + 1)]);
            if i + j + 1 < n {
                tris.push([pt(i + 1, j), pt(i + 1, j + 1), pt(i, j + 1)]);
            }
        }
    }
    tris
}

/// Realize the selected chunks into a per-LOD-coloured triangle soup. Each chunk
/// becomes `chunk_res`×`chunk_res` triangles all carrying the chunk's LOD colour.
pub fn build_debug_mesh(chunks: &[Chunk], radius: f32, chunk_res: u32) -> DebugMeshData {
    let mut positions = Vec::new();
    let mut colors = Vec::new();
    let mut indices = Vec::new();
    for c in chunks {
        let col = lod_color(c.level);
        for tri in tessellate(c.corners, radius, chunk_res) {
            let base = positions.len() as i32;
            for v in tri {
                positions.push(v);
                colors.push(col);
            }
            indices.push(base);
            indices.push(base + 1);
            indices.push(base + 2);
        }
    }
    DebugMeshData { positions, colors, indices }
}

// ---------------------------------------------------------------------------
// Godot node: selects the cut each frame and draws it. Verified visually
// (screenshot), not unit-tested — it touches the live engine.
// ---------------------------------------------------------------------------

use std::time::Instant;

use celestial_algo::quadtree::{base_face_frames, select_chunks};
use godot::classes::base_material_3d::{CullMode, Flags, ShadingMode};
use godot::classes::mesh::{ArrayType, PrimitiveType};
use godot::classes::viewport::DebugDraw;
use godot::classes::{ArrayMesh, INode3D, MeshInstance3D, Node3D, StandardMaterial3D};
use godot::prelude::*;

/// Phase 1 debug node: each frame, select the chunked-quadtree cut on the CPU and
/// draw it as a per-LOD-coloured mesh. Additive — does not touch the clipmap.
#[derive(GodotClass)]
#[class(base = Node3D, tool, init, internal)]
pub struct CelestialQuadtreeDebug {
    base: Base<Node3D>,
    #[export]
    #[init(val = 1000.0)]
    radius: f32,
    #[export]
    #[init(val = 0.02)]
    screen_error: f32,
    #[export]
    #[init(val = 16)]
    chunk_res: i64,
    #[export]
    #[init(val = 16)]
    max_depth: i64,
    #[export]
    wireframe: bool,
    last_select_ms: f64,
    last_realize_ms: f64,
    chunk_count: i64,
    mesh_instance: Option<Gd<MeshInstance3D>>,
}

#[godot_api]
impl INode3D for CelestialQuadtreeDebug {
    fn ready(&mut self) {
        let mut mi = MeshInstance3D::new_alloc();
        let mut mat = StandardMaterial3D::new_gd();
        mat.set_shading_mode(ShadingMode::UNSHADED);
        mat.set_flag(Flags::ALBEDO_FROM_VERTEX_COLOR, true);
        mat.set_cull_mode(CullMode::DISABLED);
        mi.set_material_override(&mat);
        self.mesh_instance = Some(mi.clone());
        self.base_mut().add_child(&mi);
    }

    fn process(&mut self, _delta: f64) {
        let Some(camera) = self
            .base()
            .get_viewport()
            .and_then(|vp| vp.get_camera_3d())
            .map(|c| c.get_global_position())
        else {
            return;
        };

        let frames = base_face_frames(self.radius);
        let t0 = Instant::now();
        let chunks =
            select_chunks(&frames, camera, self.screen_error, self.chunk_res as u32, self.max_depth as u8, None);
        self.last_select_ms = t0.elapsed().as_secs_f64() * 1000.0;
        self.chunk_count = chunks.len() as i64;

        let t1 = Instant::now();
        let data = build_debug_mesh(&chunks, self.radius, self.chunk_res as u32);
        self.last_realize_ms = t1.elapsed().as_secs_f64() * 1000.0;
        let positions: PackedVector3Array = data.positions.iter().copied().collect();
        let colors: PackedColorArray =
            data.colors.iter().map(|c| Color::from_rgba(c[0], c[1], c[2], 1.0)).collect();
        let indices: PackedInt32Array = data.indices.iter().copied().collect();

        let mut arrays = VarArray::new();
        arrays.resize(ArrayType::MAX.ord() as usize, &Variant::nil());
        arrays.set(ArrayType::VERTEX.ord() as usize, &positions.to_variant());
        arrays.set(ArrayType::COLOR.ord() as usize, &colors.to_variant());
        arrays.set(ArrayType::INDEX.ord() as usize, &indices.to_variant());

        let mut mesh = ArrayMesh::new_gd();
        if !indices.is_empty() {
            mesh.add_surface_from_arrays(PrimitiveType::TRIANGLES, &arrays);
        }
        if let Some(mi) = self.mesh_instance.as_mut() {
            mi.set_mesh(&mesh);
        }

        if let Some(mut vp) = self.base().get_viewport() {
            vp.set_debug_draw(if self.wireframe { DebugDraw::WIREFRAME } else { DebugDraw::DISABLED });
        }
    }
}

#[godot_api]
impl CelestialQuadtreeDebug {
    #[func]
    fn last_select_ms(&self) -> f64 {
        self.last_select_ms
    }

    #[func]
    fn last_realize_ms(&self) -> f64 {
        self.last_realize_ms
    }

    #[func]
    fn chunk_count(&self) -> i64 {
        self.chunk_count
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use celestial_algo::quadtree::{base_face_frames, select_chunks};

    #[test]
    fn lod_color_differs_per_level_and_is_deterministic() {
        assert_ne!(lod_color(0), lod_color(1));
        assert_ne!(lod_color(1), lod_color(2));
        assert_eq!(lod_color(3), lod_color(3));
    }

    #[test]
    fn build_debug_mesh_tessellates_each_chunk() {
        let frames = base_face_frames(1000.0);
        let chunks = select_chunks(&frames, Vector3::new(0.0, 0.0, 1100.0), 0.1, 16, 4, None);
        // res 1 => one triangle per chunk.
        let mesh1 = build_debug_mesh(&chunks, 1000.0, 1);
        assert_eq!(mesh1.positions.len(), chunks.len() * 3);
        let c0 = lod_color(chunks[0].level);
        assert_eq!(mesh1.colors[0], c0);
        assert_eq!(mesh1.colors[2], c0);
        // res 2 => 4 triangles (12 verts) per chunk.
        let mesh2 = build_debug_mesh(&chunks, 1000.0, 2);
        assert_eq!(mesh2.positions.len(), chunks.len() * 4 * 3);
        assert_eq!(mesh2.indices.len(), chunks.len() * 4 * 3);
    }
}
