//! Engine-agnostic GPU computation graph (CEL-60 M5).
//!
//! Nodes are *operations* (one compute pass each — e.g. "upload descriptors",
//! "fill dispatch args", "realize terrain"), resources are the buffers and
//! textures flowing between them, and [`Graph::execute`] re-records only
//! dirty nodes, in topological order. Per-face (or per-chunk) granularity is
//! data *inside* a node, never graph topology.
//!
//! The crate is engine-free: nodes record through a caller-supplied context
//! type `C` (Godot's `RenderingDevice` in production, a mock in tests), so
//! the scheduling semantics are unit-tested without a GPU.

mod graph;

pub use graph::{ExecReport, Graph, GraphNode, NodeDesc, NodeId, RecordCtx, ResourceId};
