//! Pure helpers for the `CesMeshSourceGltf` feature.
//!
//! These functions are deliberately Godot-free so they can be unit-tested
//! without a Godot runtime. Variant assignment is computed on the GPU from
//! the triangle centroid direction (see `ScatterPlacement.slang`) — there's
//! no CPU mirror because the spatial hash uses float ops that don't
//! reproduce bit-identically across CPU and GPU.

/// Plain TRS data lifted out of a glTF node, with no Godot dependency
/// so the bake logic stays unit-testable in pure Rust.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NodeTrs {
    pub translation: [f32; 3], // x, y, z
    pub rotation: [f32; 4],    // quaternion x, y, z, w
    pub scale: [f32; 3],       // x, y, z
}

impl NodeTrs {
    pub const IDENTITY: NodeTrs = NodeTrs {
        translation: [0.0, 0.0, 0.0],
        rotation: [0.0, 0.0, 0.0, 1.0],
        scale: [1.0, 1.0, 1.0],
    };
}

/// Returns the index of the first `enabled = true` entry in `picks`, or
/// `None` if all are disabled. Used by the scatter runtime to pick which
/// glTF child to scatter when only single-mesh mode is supported (Phase 3).
pub fn first_enabled_pick(picks: &[bool]) -> Option<usize> {
    picks.iter().position(|&e| e)
}

/// Returns the indices of all `enabled = true` entries in `picks`,
/// preserving order. Empty vec if all disabled.
///
/// Phase 4 uses this to enumerate every variant a `CesMeshSourceGltf`
/// produces, so a single scatter layer can spawn one MultiMesh per pick.
pub fn all_enabled_pick_indices(picks: &[bool]) -> Vec<usize> {
    picks
        .iter()
        .enumerate()
        .filter_map(|(i, &e)| if e { Some(i) } else { None })
        .collect()
}

/// Strips the X and Z components of a node's translation when
/// `bake_xz_translation` is false (the default). Rotation, scale, and Y
/// translation are always preserved.
///
/// Used to bake artist-side per-node TRS into geometry so multiple meshes
/// from one glTF can be scattered without their preview-row layout leaking
/// into the placement.
pub fn bake_node_transform(trs: NodeTrs, bake_xz_translation: bool) -> NodeTrs {
    if bake_xz_translation {
        trs
    } else {
        NodeTrs {
            translation: [0.0, trs.translation[1], 0.0],
            rotation: trs.rotation,
            scale: trs.scale,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bake_identity_returns_identity_when_xz_disabled() {
        let out = bake_node_transform(NodeTrs::IDENTITY, false);
        assert_eq!(out, NodeTrs::IDENTITY);
    }

    #[test]
    fn test_bake_strips_xz_translation_when_disabled() {
        let input = NodeTrs {
            translation: [1.5, 2.5, -3.5],
            rotation: [0.0, 0.0, 0.0, 1.0],
            scale: [1.0, 1.0, 1.0],
        };
        let out = bake_node_transform(input, false);
        assert_eq!(out.translation, [0.0, 2.5, 0.0]);
    }

    #[test]
    fn test_bake_preserves_xz_when_enabled() {
        let input = NodeTrs {
            translation: [1.5, 2.5, -3.5],
            rotation: [0.0, 0.0, 0.0, 1.0],
            scale: [1.0, 1.0, 1.0],
        };
        let out = bake_node_transform(input, true);
        assert_eq!(out.translation, [1.5, 2.5, -3.5]);
    }

    #[test]
    fn test_first_enabled_pick_empty_returns_none() {
        assert_eq!(first_enabled_pick(&[]), None);
    }

    #[test]
    fn test_first_enabled_pick_all_disabled_returns_none() {
        assert_eq!(first_enabled_pick(&[false, false, false]), None);
    }

    #[test]
    fn test_first_enabled_pick_picks_first_enabled() {
        assert_eq!(first_enabled_pick(&[false, true, false, true]), Some(1));
    }

    #[test]
    fn test_all_enabled_pick_indices_empty_returns_empty() {
        let out = all_enabled_pick_indices(&[]);
        assert!(out.is_empty());
    }

    #[test]
    fn test_all_enabled_pick_indices_all_disabled_returns_empty() {
        let out = all_enabled_pick_indices(&[false, false, false, false]);
        assert!(out.is_empty());
    }

    #[test]
    fn test_all_enabled_pick_indices_returns_all_enabled_in_order() {
        let out = all_enabled_pick_indices(&[false, true, false, true, true]);
        assert_eq!(out, vec![1, 3, 4]);
    }

    #[test]
    fn test_all_enabled_pick_indices_preserves_relative_order() {
        let out = all_enabled_pick_indices(&[false, true, false, true, true]);
        // Indices must come back in strictly ascending order.
        for w in out.windows(2) {
            assert!(w[0] < w[1], "expected ascending order, got {:?}", out);
        }
    }

    #[test]
    fn test_bake_always_preserves_rotation_and_scale() {
        let rotation = [0.1f32, 0.2, 0.3, 0.927];
        let scale = [0.679f32, 0.679, 0.679];
        let input = NodeTrs {
            translation: [1.5, 2.5, -3.5],
            rotation,
            scale,
        };
        for bake_xz in [false, true] {
            let out = bake_node_transform(input, bake_xz);
            assert_eq!(out.rotation, rotation, "rotation changed (bake_xz={bake_xz})");
            assert_eq!(out.scale, scale, "scale changed (bake_xz={bake_xz})");
        }
    }
}
