//! The graph: nodes, resources, dirtiness, topological execution.

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ResourceId(pub usize);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct NodeId(pub usize);

/// Static description of a node: its name (used for GPU timestamps and
/// logs) and the resources it reads/writes. Edges are derived from these —
/// a node that reads a resource depends on every node that writes it.
#[derive(Clone, Debug)]
pub struct NodeDesc {
    pub name: &'static str,
    pub reads: Vec<ResourceId>,
    pub writes: Vec<ResourceId>,
}

/// Services the executor offers while a node records — currently GPU
/// timestamp brackets. Implemented by the production adapter and by mocks.
pub trait RecordCtx {
    fn timestamp(&mut self, label: &str);
}

/// A schedulable operation. `C` is the recording context (the production
/// device adapter, or a mock in tests).
pub trait GraphNode<C: RecordCtx + ?Sized> {
    /// Hash of the node's own parameters; a change marks the node dirty
    /// even if no upstream resource changed.
    fn params_hash(&self) -> u64 {
        0
    }
    /// Record this node's GPU work. Only called when the node is dirty.
    fn record(&mut self, ctx: &mut C);
}

struct Entry<N> {
    desc: NodeDesc,
    node: N,
    dirty: bool,
    last_params: Option<u64>,
}

/// What [`Graph::execute`] did: which nodes recorded, which were skipped.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ExecReport {
    pub ran: Vec<NodeId>,
    pub skipped: Vec<NodeId>,
}

#[derive(Default)]
pub struct Graph<N> {
    entries: Vec<Entry<N>>,
    resource_count: usize,
}

impl<N> Graph<N> {
    pub fn new() -> Self {
        Self { entries: Vec::new(), resource_count: 0 }
    }

    pub fn add_resource(&mut self) -> ResourceId {
        self.resource_count += 1;
        ResourceId(self.resource_count - 1)
    }

    /// Nodes start dirty (everything runs on the first execute).
    pub fn add_node(&mut self, desc: NodeDesc, node: N) -> NodeId {
        for r in desc.reads.iter().chain(&desc.writes) {
            assert!(r.0 < self.resource_count, "unknown resource {r:?}");
        }
        self.entries.push(Entry { desc, node, dirty: true, last_params: None });
        NodeId(self.entries.len() - 1)
    }

    pub fn mark_dirty(&mut self, id: NodeId) {
        self.entries[id.0].dirty = true;
    }

    /// Mark every (transitive) reader of `resource` dirty — the external
    /// world changed a resource behind the graph's back (e.g. a CPU upload).
    pub fn invalidate(&mut self, resource: ResourceId) {
        for i in 0..self.entries.len() {
            if self.entries[i].desc.reads.contains(&resource) {
                self.entries[i].dirty = true;
            }
        }
        // Transitivity is handled again at execute time; eagerly propagating
        // here keeps `is_dirty` queries honest between executes.
        self.propagate();
    }

    pub fn is_dirty(&self, id: NodeId) -> bool {
        self.entries[id.0].dirty
    }

    pub fn node(&self, id: NodeId) -> &N {
        &self.entries[id.0].node
    }

    /// Mutable node access for staging per-frame data (does NOT mark dirty —
    /// use `mark_dirty`/`params_hash` for that).
    pub fn node_mut(&mut self, id: NodeId) -> &mut N {
        &mut self.entries[id.0].node
    }

    /// Topological order from write→read edges (Kahn). Panics on cycles.
    fn topo_order(&self) -> Vec<usize> {
        let n = self.entries.len();
        // writer-of-resource → readers edges
        let mut indeg = vec![0usize; n];
        let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];
        for (wi, w) in self.entries.iter().enumerate() {
            for r in &w.desc.writes {
                for (ri, e) in self.entries.iter().enumerate() {
                    if ri != wi && e.desc.reads.contains(r) {
                        adj[wi].push(ri);
                        indeg[ri] += 1;
                    }
                }
            }
        }
        // Stable Kahn: always take the lowest-index ready node, so order is
        // deterministic and respects insertion order among independents.
        let mut ready: std::collections::BTreeSet<usize> =
            (0..n).filter(|&i| indeg[i] == 0).collect();
        let mut order = Vec::with_capacity(n);
        while let Some(&i) = ready.iter().next() {
            ready.remove(&i);
            order.push(i);
            for &j in &adj[i] {
                indeg[j] -= 1;
                if indeg[j] == 0 {
                    ready.insert(j);
                }
            }
        }
        assert_eq!(order.len(), n, "graph has a cycle");
        order
    }

    /// Dirty propagation: a dirty node re-writes its outputs, so every
    /// reader of those outputs becomes dirty too (transitively).
    fn propagate(&mut self) {
        let order = self.topo_order();
        let mut dirty_res = vec![false; self.resource_count];
        for &i in &order {
            if !self.entries[i].dirty {
                let reads_dirty =
                    self.entries[i].desc.reads.iter().any(|r| dirty_res[r.0]);
                if reads_dirty {
                    self.entries[i].dirty = true;
                }
            }
            if self.entries[i].dirty {
                for w in &self.entries[i].desc.writes {
                    dirty_res[w.0] = true;
                }
            }
        }
    }
}

impl<N> Graph<N> {
    /// Run the graph: refresh self-dirtiness from `params_hash`, propagate
    /// along edges, record dirty nodes in topological order (bracketed by
    /// GPU timestamps), clear their dirty flags.
    pub fn execute<C>(&mut self, ctx: &mut C) -> ExecReport
    where
        C: RecordCtx + ?Sized,
        N: GraphNode<C>,
    {
        for e in &mut self.entries {
            let h = e.node.params_hash();
            if e.last_params != Some(h) {
                e.dirty = true;
                e.last_params = Some(h);
            }
        }
        self.propagate();

        let mut report = ExecReport::default();
        for i in self.topo_order() {
            if self.entries[i].dirty {
                let name = self.entries[i].desc.name;
                ctx.timestamp(name);
                self.entries[i].node.record(ctx);
                self.entries[i].dirty = false;
                report.ran.push(NodeId(i));
            } else {
                report.skipped.push(NodeId(i));
            }
        }
        report
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;
    use std::rc::Rc;

    /// Mock context: logs timestamps; nodes log their own runs.
    #[derive(Default)]
    struct Mock {
        timestamps: Vec<&'static str>,
    }
    impl RecordCtx for Mock {
        fn timestamp(&mut self, label: &str) {
            // labels are 'static in tests
            self.timestamps.push(Box::leak(label.to_owned().into_boxed_str()));
        }
    }

    struct TestNode {
        runs: Rc<RefCell<Vec<&'static str>>>,
        name: &'static str,
        params: u64,
    }
    impl GraphNode<Mock> for TestNode {
        fn params_hash(&self) -> u64 {
            self.params
        }
        fn record(&mut self, _ctx: &mut Mock) {
            self.runs.borrow_mut().push(self.name);
        }
    }

    struct Fixture {
        graph: Graph<TestNode>,
        runs: Rc<RefCell<Vec<&'static str>>>,
        r_desc: ResourceId,
        r_args: ResourceId,
        r_mesh: ResourceId,
        upload: NodeId,
        fill: NodeId,
        realize: NodeId,
    }

    /// upload -> (r_desc) -> fill -> (r_args) -> realize; realize also reads
    /// r_desc (a diamond-ish shape matching the terrain pipeline).
    fn pipeline() -> Fixture {
        let mut graph = Graph::new();
        let runs = Rc::new(RefCell::new(Vec::new()));
        let r_desc = graph.add_resource();
        let r_args = graph.add_resource();
        let r_mesh = graph.add_resource();
        let node = |name| TestNode { runs: runs.clone(), name, params: 0 };
        let upload = graph.add_node(
            NodeDesc { name: "upload", reads: vec![], writes: vec![r_desc] },
            node("upload"),
        );
        let fill = graph.add_node(
            NodeDesc { name: "fill", reads: vec![r_desc], writes: vec![r_args] },
            node("fill"),
        );
        let realize = graph.add_node(
            NodeDesc { name: "realize", reads: vec![r_desc, r_args], writes: vec![r_mesh] },
            node("realize"),
        );
        Fixture { graph, runs, r_desc, r_args, r_mesh, upload, fill, realize }
    }

    #[test]
    fn first_execute_runs_everything_in_topo_order() {
        let mut f = pipeline();
        let report = f.graph.execute(&mut Mock::default());
        assert_eq!(*f.runs.borrow(), vec!["upload", "fill", "realize"]);
        assert_eq!(report.ran, vec![f.upload, f.fill, f.realize]);
        assert!(report.skipped.is_empty());
    }

    #[test]
    fn clean_graph_skips_all_nodes() {
        let mut f = pipeline();
        f.graph.execute(&mut Mock::default());
        f.runs.borrow_mut().clear();
        let report = f.graph.execute(&mut Mock::default());
        assert!(f.runs.borrow().is_empty(), "no node may re-run when clean");
        assert!(report.ran.is_empty());
        assert_eq!(report.skipped.len(), 3);
    }

    #[test]
    fn invalidate_dirties_transitive_readers_only() {
        let mut f = pipeline();
        f.graph.execute(&mut Mock::default());
        f.runs.borrow_mut().clear();

        f.graph.invalidate(f.r_desc); // CPU re-uploaded descriptors
        assert!(!f.graph.is_dirty(f.upload), "writers stay clean");
        assert!(f.graph.is_dirty(f.fill));
        assert!(f.graph.is_dirty(f.realize), "transitive reader is dirty");

        let report = f.graph.execute(&mut Mock::default());
        assert_eq!(*f.runs.borrow(), vec!["fill", "realize"]);
        assert_eq!(report.skipped, vec![f.upload]);
    }

    #[test]
    fn downstream_invalidate_never_reruns_upstream() {
        // The original sin this graph exists to prevent: a trees-only change
        // re-rendering the whole planet. A node reading the realize output
        // must re-run alone when only it is dirtied.
        let mut f = pipeline();
        let trees = f.graph.add_node(
            NodeDesc { name: "trees", reads: vec![f.r_mesh], writes: vec![] },
            TestNode { runs: f.runs.clone(), name: "trees", params: 0 },
        );
        f.graph.execute(&mut Mock::default());
        f.runs.borrow_mut().clear();

        f.graph.mark_dirty(trees);
        let report = f.graph.execute(&mut Mock::default());
        assert_eq!(*f.runs.borrow(), vec!["trees"]);
        assert_eq!(report.ran, vec![trees]);
        assert_eq!(report.skipped.len(), 3, "terrain pipeline untouched");
    }

    #[test]
    fn params_change_self_dirties_and_propagates() {
        let mut f = pipeline();
        f.graph.execute(&mut Mock::default());
        f.runs.borrow_mut().clear();

        f.graph.node_mut(f.fill).params = 42;
        f.graph.execute(&mut Mock::default());
        assert_eq!(*f.runs.borrow(), vec!["fill", "realize"]);
    }

    #[test]
    fn diamond_runs_each_node_once() {
        // upload writes r_desc; two consumers write separate resources; a
        // join reads both. Everything dirty -> each node records exactly once.
        let mut graph: Graph<TestNode> = Graph::new();
        let runs = Rc::new(RefCell::new(Vec::new()));
        let r0 = graph.add_resource();
        let ra = graph.add_resource();
        let rb = graph.add_resource();
        let node = |name| TestNode { runs: runs.clone(), name, params: 0 };
        graph.add_node(NodeDesc { name: "src", reads: vec![], writes: vec![r0] }, node("src"));
        graph.add_node(NodeDesc { name: "a", reads: vec![r0], writes: vec![ra] }, node("a"));
        graph.add_node(NodeDesc { name: "b", reads: vec![r0], writes: vec![rb] }, node("b"));
        graph.add_node(NodeDesc { name: "join", reads: vec![ra, rb], writes: vec![] }, node("join"));

        graph.execute(&mut Mock::default());
        let order = runs.borrow().clone();
        assert_eq!(order.len(), 4, "each node exactly once");
        assert_eq!(order[0], "src");
        assert_eq!(order[3], "join");
    }

    #[test]
    fn timestamps_bracket_ran_nodes() {
        let mut f = pipeline();
        let mut mock = Mock::default();
        f.graph.execute(&mut mock);
        assert_eq!(mock.timestamps, vec!["upload", "fill", "realize"]);
    }

    #[test]
    #[should_panic(expected = "cycle")]
    fn cycles_panic() {
        let mut graph: Graph<TestNode> = Graph::new();
        let runs = Rc::new(RefCell::new(Vec::new()));
        let ra = graph.add_resource();
        let rb = graph.add_resource();
        let node = |name| TestNode { runs: runs.clone(), name, params: 0 };
        graph.add_node(NodeDesc { name: "x", reads: vec![rb], writes: vec![ra] }, node("x"));
        graph.add_node(NodeDesc { name: "y", reads: vec![ra], writes: vec![rb] }, node("y"));
        graph.execute(&mut Mock::default());
    }
}
