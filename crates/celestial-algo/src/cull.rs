//! Pure view-frustum culling helpers (CEL-67). Kept free of Godot scene types
//! so the geometry is unit-testable; the scene glue lives in `planet.rs`.

use godot::builtin::{Plane, Vector3};

/// True when the axis-aligned box given by its 8 world-space `corners` is fully
/// outside the view frustum: every corner sits outside the same frustum plane.
///
/// Frustum planes point OUTWARD — the `Camera3D::get_frustum` convention — so a
/// point is INSIDE when `plane.distance_to(point) <= 0` for every plane, and the
/// box is fully outside when all 8 corners are strictly outside (`> margin`) the
/// SAME plane. `margin` (world units) keeps a box drawn a little past the screen
/// edge so faces don't pop during fast camera rotation.
///
/// Conservative: only culls when provably invisible, so a box straddling the
/// frustum boundary is kept (never falsely culled), which is what keeps the
/// rendered mesh crack-free at the screen edge.
pub fn aabb_outside_frustum(corners: &[Vector3; 8], planes: &[Plane], margin: f32) -> bool {
    planes
        .iter()
        .any(|pl| corners.iter().all(|&c| pl.distance_to(c) > margin))
}

#[cfg(test)]
mod tests {
    use super::*;
    use godot::builtin::{Plane, Vector3};

    /// Six OUTWARD-pointing planes bounding the axis-aligned cube [-1,1]^3,
    /// matching Godot's `Camera3D::get_frustum` convention.
    ///
    /// `distance_to(p) = normal.dot(p) - d`. For the `x <= 1` face the outward
    /// normal is `+x` and the boundary passes through `x = 1`, so `x - d = 0` at
    /// `x = 1` gives `d = 1`. For the `x >= -1` face the outward normal is `-x`
    /// and the boundary is `x = -1`, so `-x - d = 0` at `x = -1` also gives
    /// `d = 1`. By symmetry every outward plane uses `d = 1.0`, making
    /// `distance_to` negative inside the cube and positive outside.
    fn box_frustum() -> Vec<Plane> {
        vec![
            Plane::new(Vector3::new(1.0, 0.0, 0.0), 1.0),  // x <=  1
            Plane::new(Vector3::new(-1.0, 0.0, 0.0), 1.0), // x >= -1
            Plane::new(Vector3::new(0.0, 1.0, 0.0), 1.0),  // y <=  1
            Plane::new(Vector3::new(0.0, -1.0, 0.0), 1.0), // y >= -1
            Plane::new(Vector3::new(0.0, 0.0, 1.0), 1.0),  // z <=  1
            Plane::new(Vector3::new(0.0, 0.0, -1.0), 1.0), // z >= -1
        ]
    }
    fn corners(center: Vector3, half: f32) -> [Vector3; 8] {
        let mut c = [Vector3::ZERO; 8];
        let mut k = 0;
        for sx in [-1.0, 1.0] {
            for sy in [-1.0, 1.0] {
                for sz in [-1.0, 1.0] {
                    c[k] = center + Vector3::new(sx, sy, sz) * half;
                    k += 1;
                }
            }
        }
        c
    }
    #[test]
    fn inside_box_is_not_culled() {
        assert!(!aabb_outside_frustum(
            &corners(Vector3::ZERO, 0.5),
            &box_frustum(),
            0.0
        ));
    }
    #[test]
    fn far_outside_box_is_culled() {
        assert!(aabb_outside_frustum(
            &corners(Vector3::new(5.0, 0.0, 0.0), 0.5),
            &box_frustum(),
            0.0
        ));
    }
    #[test]
    fn straddling_box_is_kept() {
        // Crosses the +x plane; conservative test must NOT cull it.
        assert!(!aabb_outside_frustum(
            &corners(Vector3::new(1.0, 0.0, 0.0), 0.5),
            &box_frustum(),
            0.0
        ));
    }
}
