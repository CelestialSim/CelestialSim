use godot::builtin::Vector3;

pub struct VertexTexturePlaceholderMesh {
    pub positions: Vec<Vector3>,
}

pub fn build_placeholder_mesh(vertex_count: u32) -> VertexTexturePlaceholderMesh {
    let vertex_count = usize::try_from(vertex_count).expect("vertex count does not fit in usize");
    VertexTexturePlaceholderMesh {
        positions: vec![Vector3::ZERO; vertex_count],
    }
}

/// Pure cache-hit predicate for the placeholder `ArrayMesh` cache used by the
/// experimental vertex-texture path. Returns true iff `cached_vertex_count` is
/// `Some(n)` with `n == requested_vertex_count`. The mesh itself is held as a
/// `Gd<ArrayMesh>` next to the count on the celestial node, but the matching
/// rule is independent of the engine and tested here.
pub fn placeholder_mesh_cache_hits(
    cached_vertex_count: Option<u32>,
    requested_vertex_count: u32,
) -> bool {
    cached_vertex_count == Some(requested_vertex_count)
}

/// Worker-side decision: should we ship a freshly built placeholder
/// position vec to the main thread this frame? Only when the vertex count
/// differs from the last one we shipped (None = never shipped before).
pub fn should_build_placeholder(last: Option<u32>, current: u32) -> bool {
    last != Some(current)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn placeholder_mesh_can_be_empty() {
        let mesh = build_placeholder_mesh(0);
        assert!(mesh.positions.is_empty());
    }

    #[test]
    fn placeholder_mesh_uses_requested_vertex_count_with_zeroed_attributes() {
        let mesh = build_placeholder_mesh(6);
        assert_eq!(mesh.positions.len(), 6);
        assert!(mesh.positions.iter().all(|p| *p == Vector3::ZERO));
    }

    #[test]
    fn placeholder_mesh_cache_misses_when_empty() {
        assert!(!placeholder_mesh_cache_hits(None, 0));
        assert!(!placeholder_mesh_cache_hits(None, 12));
    }

    #[test]
    fn placeholder_mesh_cache_hits_on_matching_vertex_count() {
        assert!(placeholder_mesh_cache_hits(Some(0), 0));
        assert!(placeholder_mesh_cache_hits(Some(12), 12));
        assert!(placeholder_mesh_cache_hits(Some(123_456), 123_456));
    }

    #[test]
    fn placeholder_mesh_cache_misses_on_different_vertex_count() {
        assert!(!placeholder_mesh_cache_hits(Some(0), 1));
        assert!(!placeholder_mesh_cache_hits(Some(12), 13));
        assert!(!placeholder_mesh_cache_hits(Some(13), 12));
    }

    #[test]
    fn should_build_placeholder_with_no_previous_returns_true() {
        assert!(should_build_placeholder(None, 0));
        assert!(should_build_placeholder(None, 12));
    }

    #[test]
    fn should_build_placeholder_with_changed_count_returns_true() {
        assert!(should_build_placeholder(Some(0), 1));
        assert!(should_build_placeholder(Some(12), 13));
        assert!(should_build_placeholder(Some(13), 12));
    }

    #[test]
    fn should_build_placeholder_with_unchanged_count_returns_false() {
        assert!(!should_build_placeholder(Some(0), 0));
        assert!(!should_build_placeholder(Some(12), 12));
        assert!(!should_build_placeholder(Some(123_456), 123_456));
    }
}
