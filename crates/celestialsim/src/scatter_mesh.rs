//! Built-in procedural grass blade (CEL-73): the default scatter mesh when a
//! `CesScatterLayer` has no mesh assigned. A tapered, quadratically-bent blade
//! with a dark→light-green vertex-colour gradient, so a dense field reads as
//! grass with zero committed assets.
//!
//! The look follows the hexaquo "full-geometry grass" approach, baking into the
//! static mesh the two normal tricks that make flat blade geometry shade like a
//! rounded 3-D blade (no custom shader needed):
//! - **Rounded width normals** — the left/right edge normals fan outward (as if
//!   the cross-section were a cylinder), so interpolation across the blade width
//!   sweeps the normal and the blade catches a soft highlight down its length.
//! - **Tip-up normals** — the forward normal lerps toward vertical from base to
//!   tip, so tips face the sky and pick up a specular glint.
//!
//! What a static mesh can't carry (would need a dedicated grass shader): wind
//! sway, view-space thickening, and per-patch colour/scale noise. Per-instance
//! scale jitter (0.8–1.2) is applied by `ScatterPlace.slang` instead.

use godot::classes::base_material_3d::{CullMode, Flags};
use godot::classes::mesh::{ArrayType, PrimitiveType};
use godot::classes::{ArrayMesh, StandardMaterial3D};
use godot::prelude::*;

/// Cross-section rows base→tip: `(height, half-width, forward bend)`. Height ≈ 1
/// world unit (the placement scale sizes it). Bend is quadratic in height.
const ROWS: [(f32, f32, f32); 5] = [
    (0.00, 0.050, 0.000),
    (0.35, 0.046, 0.020),
    (0.62, 0.038, 0.065),
    (0.84, 0.026, 0.130),
    (1.00, 0.000, 0.220),
];

/// Base→tip vertex colours (dark root → light tip; doubles as base AO).
const COLORS: [Color; 5] = [
    Color::from_rgb(0.05, 0.20, 0.03),
    Color::from_rgb(0.08, 0.28, 0.04),
    Color::from_rgb(0.14, 0.40, 0.06),
    Color::from_rgb(0.24, 0.52, 0.09),
    Color::from_rgb(0.38, 0.62, 0.13),
];

/// How far the edge normals fan outward from the blade face (radians). ~40°
/// gives a clearly rounded cross-section without the edges facing sideways.
const ROUND_ANGLE: f32 = 0.7;

/// Build the blade `ArrayMesh` with rounded + tip-up baked normals and its own
/// double-sided vertex-colour material.
pub fn grass_blade_mesh() -> Gd<ArrayMesh> {
    let mut verts = PackedVector3Array::new();
    let mut normals = PackedVector3Array::new();
    let mut colors = PackedColorArray::new();

    let (sin_a, cos_a) = ROUND_ANGLE.sin_cos();
    let n_rows = ROWS.len();
    for (row, &(y, hw, bend)) in ROWS.iter().enumerate() {
        let t = row as f32 / (n_rows - 1) as f32;
        // Face normal tilts from +Z (base) toward +Y (tip) so tips face the sky.
        let forward = (Vector3::new(0.0, 0.0, 1.0).lerp(Vector3::new(0.0, 1.0, 0.0), t)).normalized();
        let tangent = Vector3::new(1.0, 0.0, 0.0);
        // Edge normals fan ±ROUND_ANGLE around the face normal (cylinder fake).
        let left_n = (forward * cos_a - tangent * sin_a).normalized();
        let right_n = (forward * cos_a + tangent * sin_a).normalized();
        if row + 1 < n_rows {
            verts.push(Vector3::new(-hw, y, bend));
            verts.push(Vector3::new(hw, y, bend));
            normals.push(left_n);
            normals.push(right_n);
            colors.push(COLORS[row]);
            colors.push(COLORS[row]);
        } else {
            verts.push(Vector3::new(0.0, y, bend)); // tip
            normals.push(forward);
            colors.push(COLORS[row]);
        }
    }

    // Two tris per quad band between consecutive rows; the top band fans to the
    // single tip vertex. Row r's pair = (2r, 2r+1); tip = 2*(n_rows-1).
    let mut indices: Vec<i32> = Vec::new();
    for row in 0..n_rows - 1 {
        let a = (2 * row) as i32; // left
        let b = a + 1; // right
        if row + 2 < n_rows {
            let c = (2 * (row + 1)) as i32; // next left
            let d = c + 1; // next right
            indices.extend_from_slice(&[a, c, b, b, c, d]);
        } else {
            let tip = (2 * (n_rows - 1)) as i32;
            indices.extend_from_slice(&[a, tip, b]);
        }
    }
    let indices: PackedInt32Array = indices.into_iter().collect();

    let mut arrays = VarArray::new();
    arrays.resize(ArrayType::MAX.ord() as usize, &Variant::nil());
    arrays.set(ArrayType::VERTEX.ord() as usize, &verts.to_variant());
    arrays.set(ArrayType::NORMAL.ord() as usize, &normals.to_variant());
    arrays.set(ArrayType::COLOR.ord() as usize, &colors.to_variant());
    arrays.set(ArrayType::INDEX.ord() as usize, &indices.to_variant());

    let mut material = StandardMaterial3D::new_gd();
    material.set_flag(Flags::ALBEDO_FROM_VERTEX_COLOR, true);
    material.set_cull_mode(CullMode::DISABLED);
    material.set_roughness(0.85);
    material.set_specular(0.3);

    let mut mesh = ArrayMesh::new_gd();
    mesh.add_surface_from_arrays(PrimitiveType::TRIANGLES, &arrays);
    mesh.surface_set_material(0, &material.upcast::<godot::classes::Material>());
    mesh
}
