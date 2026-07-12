//! Chunk pipeline nodes as self-contained units behind the [`ChunkNode`] trait.
//!
//! Mirrors `nodes/mod.rs` for the chunk (grass/foliage) pipeline. Two nodes:
//! `chunk-upload` (CPU → GPU descriptor + instance buffers) and `chunk-realize`
//! (compute dispatch into the chunk multimesh). The graph is assembled from the
//! data-driven registry in [`build_chunk_pipeline`].
//!
//! GPU bodies are stubs; Task 10 fleshes them out.

use celestial_graph::{GraphNode, NodeDesc, NodeId, Graph};

use crate::chunk_pipeline::ChunkCtx;

mod upload;
mod surface_custom;
mod realize;
mod bake;
mod scatter_place;
mod scatter_compact;

pub use upload::ChunkUpload;
pub use surface_custom::ChunkSurfaceCustom;
pub use realize::ChunkRealize;
pub use bake::ChunkBake;
pub use scatter_place::ChunkScatterPlace;
pub use scatter_compact::ChunkScatterCompact;

/// A schedulable chunk operation. Knows its static name and how to record
/// itself; resource reads/writes are owned by the registry, not the node.
pub trait ChunkNode {
    /// Static node name (used for GPU timestamps and logs).
    fn name(&self) -> &'static str;
    /// Hash of the node's own parameters; a change marks the node dirty even
    /// if no upstream resource changed.
    fn params_hash(&self) -> u64 {
        0
    }
    /// Record this node's GPU work. Only called when the node is dirty.
    fn record(&mut self, ctx: &mut ChunkCtx<'_>);
}

/// Bridge the trait objects to the `celestial-graph` executor.
impl<'a> GraphNode<ChunkCtx<'a>> for Box<dyn ChunkNode> {
    fn params_hash(&self) -> u64 {
        (**self).params_hash()
    }
    fn record(&mut self, ctx: &mut ChunkCtx<'a>) {
        (**self).record(ctx)
    }
}

// Resource slots, allocated in this order by `build_chunk_pipeline`. Plain
// indices so the wiring table is testable without a `Graph`/GPU.
const R_CHUNK_DESC: usize = 0;
const R_INSTANCES: usize = 1;
const R_VERTS_TEX: usize = 2;
const R_ATLAS: usize = 3;
/// CEL-73 scatter: the visible `{slot, depth}` gather list (upload → compact).
const R_SCATTER_VIS: usize = 4;
/// CEL-73 scatter: the cached per-slot candidate pool (place → compact).
const R_SCATTER_POOL: usize = 5;
/// CEL-73 scatter: the external per-layer MultiMesh buffers compact writes.
const R_SCATTER_OUT: usize = 6;
/// Custom GPU surface: the per-slot `surface_color/height/normal` buffers the
/// custom node writes and realize/bake read (ordering handle for the node).
const R_SURFACE: usize = 7;
const RESOURCE_COUNT: usize = 8;

/// One row of the chunk pipeline registry: how to build a node plus the
/// resource slots it reads/writes.
struct NodeSpec {
    make: fn() -> Box<dyn ChunkNode>,
    reads: Vec<usize>,
    writes: Vec<usize>,
}

/// Single source of truth for chunk pipeline topology.
fn pipeline_specs() -> Vec<NodeSpec> {
    vec![
        NodeSpec {
            make: || Box::new(ChunkUpload),
            reads: vec![],
            writes: vec![R_CHUNK_DESC, R_INSTANCES, R_SCATTER_VIS],
        },
        // Runs after upload (reads desc) and before realize/bake (writes the
        // surface buffers they read). Earlier registry index => earlier in the
        // stable topo order, same as bake sitting after realize.
        NodeSpec {
            make: || Box::new(ChunkSurfaceCustom),
            reads: vec![R_CHUNK_DESC],
            writes: vec![R_SURFACE],
        },
        NodeSpec {
            make: || Box::new(ChunkRealize),
            reads: vec![R_CHUNK_DESC, R_SURFACE],
            writes: vec![R_VERTS_TEX],
        },
        NodeSpec {
            make: || Box::new(ChunkBake),
            reads: vec![R_CHUNK_DESC, R_SURFACE],
            writes: vec![R_ATLAS],
        },
        NodeSpec {
            make: || Box::new(ChunkScatterPlace),
            // Reads the CPU-surface heightmap too: on the provider route the
            // instances are displaced by the SAME baked elevations as realize.
            reads: vec![R_CHUNK_DESC, R_SURFACE],
            writes: vec![R_SCATTER_POOL],
        },
        NodeSpec {
            make: || Box::new(ChunkScatterCompact),
            reads: vec![R_SCATTER_POOL, R_SCATTER_VIS],
            writes: vec![R_SCATTER_OUT],
        },
    ]
}

/// Node ids the job needs to address after the graph is built.
pub struct ChunkRegistry {
    /// Drives the CPU descriptor + instance upload; marked dirty each frame.
    pub upload: NodeId,
    /// Realize compute dispatch; marked dirty when upload runs.
    pub realize: NodeId,
}

/// Build the chunk graph from the registry. Allocates resources, adds every
/// node in registry order, and returns the ids the job addresses.
pub fn build_chunk_pipeline(graph: &mut Graph<Box<dyn ChunkNode>>) -> ChunkRegistry {
    let resources: Vec<_> = (0..RESOURCE_COUNT).map(|_| graph.add_resource()).collect();
    let mut by_name: std::collections::HashMap<&'static str, NodeId> = Default::default();
    for spec in pipeline_specs() {
        let node = (spec.make)();
        let name = node.name();
        let id = graph.add_node(
            NodeDesc {
                name,
                reads: spec.reads.iter().map(|&s| resources[s]).collect(),
                writes: spec.writes.iter().map(|&s| resources[s]).collect(),
            },
            node,
        );
        by_name.insert(name, id);
    }
    ChunkRegistry {
        upload: by_name["celestial/chunk-upload"],
        realize: by_name["celestial/chunk-realize"],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Lock the chunk pipeline topology: node order/names and per-node
    /// read/write resource slots. Pure data — no GPU.
    #[test]
    fn chunk_registry_topology_is_locked() {
        let specs = pipeline_specs();
        let by_name: std::collections::HashMap<&'static str, &NodeSpec> =
            specs.iter().map(|s| ((s.make)().name(), s)).collect();
        let names: Vec<&'static str> = specs.iter().map(|s| (s.make)().name()).collect();
        assert_eq!(
            names,
            [
                "celestial/chunk-upload",
                "celestial/chunk-surface-custom",
                "celestial/chunk-realize",
                "celestial/chunk-bake",
                "celestial/chunk-scatter-place",
                "celestial/chunk-scatter-compact",
            ]
        );

        let n = |name| by_name[name];
        // chunk-upload: writes desc + instances + the scatter visible list.
        assert_eq!(n("celestial/chunk-upload").reads, Vec::<usize>::new());
        assert_eq!(
            n("celestial/chunk-upload").writes,
            vec![R_CHUNK_DESC, R_INSTANCES, R_SCATTER_VIS]
        );
        // chunk-surface-custom: reads desc, writes the surface buffers realize/
        // bake read. Between upload and realize (earlier registry index).
        assert_eq!(n("celestial/chunk-surface-custom").reads, vec![R_CHUNK_DESC]);
        assert_eq!(n("celestial/chunk-surface-custom").writes, vec![R_SURFACE]);
        // chunk-realize: reads desc + surface, writes verts/tex output.
        assert_eq!(n("celestial/chunk-realize").reads, vec![R_CHUNK_DESC, R_SURFACE]);
        assert_eq!(n("celestial/chunk-realize").writes, vec![R_VERTS_TEX]);
        // chunk-bake: reads desc + surface, writes the colour/normal detail atlas.
        // Runs after realize (later registry index => later in stable topo order).
        assert_eq!(n("celestial/chunk-bake").reads, vec![R_CHUNK_DESC, R_SURFACE]);
        assert_eq!(n("celestial/chunk-bake").writes, vec![R_ATLAS]);
        // scatter-place (CEL-73): reads desc + the CPU-surface heightmap (so
        // instances sit on the provider's terrain), writes the candidate pool.
        assert_eq!(
            n("celestial/chunk-scatter-place").reads,
            vec![R_CHUNK_DESC, R_SURFACE]
        );
        assert_eq!(n("celestial/chunk-scatter-place").writes, vec![R_SCATTER_POOL]);
        // scatter-compact: reads pool + visible list, writes the external
        // multimesh buffers. Last in registry order => runs after place.
        assert_eq!(
            n("celestial/chunk-scatter-compact").reads,
            vec![R_SCATTER_POOL, R_SCATTER_VIS]
        );
        assert_eq!(n("celestial/chunk-scatter-compact").writes, vec![R_SCATTER_OUT]);
    }
}
