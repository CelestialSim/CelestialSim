use std::cell::RefCell;
use std::time::Instant;

/// Hierarchical timing node.
pub struct TimingNode {
    pub label: String,
    pub cpu_ns: u64,
    pub gpu_ns: u64,
    pub call_count: u64,
    pub children: Vec<TimingNode>,
}

impl TimingNode {
    fn new(label: impl Into<String>) -> Self {
        Self {
            label: label.into(),
            cpu_ns: 0,
            gpu_ns: 0,
            call_count: 0,
            children: Vec::new(),
        }
    }

    fn child_index(&self, label: &str) -> Option<usize> {
        self.children.iter().position(|c| c.label == label)
    }

    fn ensure_child(&mut self, label: &str) -> usize {
        if let Some(i) = self.child_index(label) {
            i
        } else {
            self.children.push(TimingNode::new(label));
            self.children.len() - 1
        }
    }

    fn at_path_mut(&mut self, path: &[&str]) -> &mut TimingNode {
        let mut node = self;
        for seg in path {
            let idx = node.ensure_child(seg);
            node = &mut node.children[idx];
        }
        node
    }
}

thread_local! {
    static CURRENT_TREE: RefCell<Option<*mut TimingTree>> = const { RefCell::new(None) };
    static CURRENT_PATH: RefCell<Vec<String>> = const { RefCell::new(Vec::new()) };
}

/// Hierarchical CPU + GPU timing tree.
pub struct TimingTree {
    root: TimingNode,
    enabled: bool,
}

impl Default for TimingTree {
    fn default() -> Self {
        Self::new()
    }
}

impl TimingTree {
    pub fn new() -> Self {
        Self {
            root: TimingNode::new("root"),
            enabled: true,
        }
    }

    pub fn disabled() -> Self {
        Self {
            root: TimingNode::new("root"),
            enabled: false,
        }
    }

    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    pub fn root(&self) -> &TimingNode {
        &self.root
    }

    pub fn reset(&mut self) {
        self.root = TimingNode::new("root");
    }

    /// Enter a child scope under the current path.
    /// The returned guard does NOT hold a mutable borrow of the tree, but the
    /// caller MUST keep the tree alive (and not move it) until the guard drops.
    pub fn scope(&mut self, label: &str) -> Scope {
        if !self.enabled {
            return Scope {
                tree: None,
                start: None,
            };
        }
        push_path(label);
        Scope {
            tree: Some(self as *mut _),
            start: Some(Instant::now()),
        }
    }

    pub fn add_gpu_ns(&mut self, path: &str, gpu_ns: u64) {
        let segments: Vec<&str> = path.split("::").filter(|s| !s.is_empty()).collect();
        let node = self.root.at_path_mut(&segments);
        node.gpu_ns += gpu_ns;
    }

    pub fn add_cpu_ns(&mut self, path: &str, cpu_ns: u64) {
        let segments: Vec<&str> = path.split("::").filter(|s| !s.is_empty()).collect();
        let node = self.root.at_path_mut(&segments);
        node.cpu_ns += cpu_ns;
    }

    /// Merge another tree as a subtree under `label` at the current root.
    pub fn merge_subtree(&mut self, label: &str, other: TimingTree) {
        let idx = self.root.ensure_child(label);
        let dst = &mut self.root.children[idx];
        for child in other.root.children {
            merge_node(dst, child);
        }
        dst.cpu_ns += other.root.cpu_ns;
        dst.gpu_ns += other.root.gpu_ns;
        dst.call_count += other.root.call_count;
    }

    pub fn render_ascii(&self) -> String {
        render_tree_ascii(&self.root)
    }
}

fn merge_node(parent: &mut TimingNode, incoming: TimingNode) {
    let idx = parent.ensure_child(&incoming.label);
    let dst = &mut parent.children[idx];
    dst.cpu_ns += incoming.cpu_ns;
    dst.gpu_ns += incoming.gpu_ns;
    dst.call_count += incoming.call_count;
    for child in incoming.children {
        merge_node(dst, child);
    }
}

fn push_path(label: &str) {
    CURRENT_PATH.with(|p| p.borrow_mut().push(label.to_string()));
}

fn pop_path() -> Option<String> {
    CURRENT_PATH.with(|p| p.borrow_mut().pop())
}

/// Returns the current scope path joined with "::". Empty if outside any scope.
pub fn current_path_joined() -> String {
    CURRENT_PATH.with(|p| p.borrow().join("::"))
}

/// Returns the current scope path as a vec of owned segments.
pub fn current_path_segments() -> Vec<String> {
    CURRENT_PATH.with(|p| p.borrow().clone())
}

/// Sets the per-thread "current tree" pointer for the duration of `f`.
/// Used to make `with_current_tree` work without threading the tree through
/// every call site.
pub fn with_thread_tree<R>(tree: &mut TimingTree, f: impl FnOnce() -> R) -> R {
    let _g = install_thread_tree(tree);
    f()
}

/// RAII guard that installs `tree` as the current thread's tree until dropped.
pub struct ThreadTreeGuard {
    prev: Option<*mut TimingTree>,
}

impl Drop for ThreadTreeGuard {
    fn drop(&mut self) {
        CURRENT_TREE.with(|t| *t.borrow_mut() = self.prev);
    }
}

/// Installs `tree` as the current thread's tree until the returned guard is
/// dropped. Safer alternative to `with_thread_tree` when the work that uses the
/// tree spans multiple statements / sequential blocks.
pub fn install_thread_tree(tree: &mut TimingTree) -> ThreadTreeGuard {
    let prev = CURRENT_TREE.with(|t| t.replace(Some(tree as *mut _)));
    ThreadTreeGuard { prev }
}

/// Calls `f` with a mutable reference to the current thread's tree, if any is set.
pub fn with_current_tree<R>(f: impl FnOnce(&mut TimingTree) -> R) -> Option<R> {
    let ptr = CURRENT_TREE.with(|t| *t.borrow());
    ptr.map(|p| {
        // SAFETY: the pointer is only set inside `with_thread_tree`, which holds
        // an exclusive `&mut TimingTree` for the duration of the closure on the
        // same thread; no concurrent access.
        let tree = unsafe { &mut *p };
        f(tree)
    })
}

/// Returns whether the current thread has an enabled tree set.
pub fn current_tree_enabled() -> bool {
    with_current_tree(|t| t.is_enabled()).unwrap_or(false)
}

/// Enter a scope on the current thread's tree (if any).
pub struct ThreadScope {
    enabled: bool,
    start: Option<Instant>,
}

impl ThreadScope {
    pub fn enter(label: &str) -> Self {
        if !current_tree_enabled() {
            return Self {
                enabled: false,
                start: None,
            };
        }
        push_path(label);
        Self {
            enabled: true,
            start: Some(Instant::now()),
        }
    }
}

impl Drop for ThreadScope {
    fn drop(&mut self) {
        if !self.enabled {
            return;
        }
        let elapsed_ns = self
            .start
            .take()
            .map(|s| s.elapsed().as_nanos() as u64)
            .unwrap_or(0);
        let label = pop_path().unwrap_or_default();
        with_current_tree(|tree| {
            let segments: Vec<String> = current_path_segments();
            let mut node = &mut tree.root;
            for seg in &segments {
                let idx = node.ensure_child(seg);
                node = &mut node.children[idx];
            }
            let idx = node.ensure_child(&label);
            let target = &mut node.children[idx];
            target.cpu_ns += elapsed_ns;
            target.call_count += 1;
        });
    }
}

/// RAII scope guard returned by `TimingTree::scope`.
pub struct Scope {
    tree: Option<*mut TimingTree>,
    start: Option<Instant>,
}

impl Drop for Scope {
    fn drop(&mut self) {
        let Some(tree_ptr) = self.tree else {
            return;
        };
        let elapsed_ns = self
            .start
            .take()
            .map(|s| s.elapsed().as_nanos() as u64)
            .unwrap_or(0);
        let label = pop_path().unwrap_or_default();
        // SAFETY: Scope holds an exclusive borrow of the tree via 'a; the
        // pointer is valid for the lifetime of this Scope.
        let tree = unsafe { &mut *tree_ptr };
        let parent_segments: Vec<String> = current_path_segments();
        let mut node = &mut tree.root;
        for seg in &parent_segments {
            let idx = node.ensure_child(seg);
            node = &mut node.children[idx];
        }
        let idx = node.ensure_child(&label);
        let target = &mut node.children[idx];
        target.cpu_ns += elapsed_ns;
        target.call_count += 1;
    }
}

/// Pure helper: pairs ("foo_begin", t1), ("foo_end", t2) into ("foo", t2-t1).
/// Returns (paired, leftover). Pairing is done by matching the closest unmatched
/// `_begin` with each `_end` in input order. Leftovers (unmatched `_begin`)
/// preserve input order.
pub fn merge_gpu_timestamp_pairs(
    timestamps: Vec<(String, u64)>,
) -> (Vec<(String, u64)>, Vec<(String, u64)>) {
    let mut open: Vec<(String, u64)> = Vec::new(); // (path, begin_time)
    let mut paired: Vec<(String, u64)> = Vec::new();

    for (name, time) in timestamps {
        if let Some(path) = name.strip_suffix("_begin") {
            open.push((path.to_string(), time));
        } else if let Some(path) = name.strip_suffix("_end") {
            // Find the most-recent matching open begin.
            if let Some(pos) = open.iter().rposition(|(p, _)| p == path) {
                let (p, begin_t) = open.remove(pos);
                let dur = time.saturating_sub(begin_t);
                paired.push((p, dur));
            }
        }
    }

    let leftover: Vec<(String, u64)> = open
        .into_iter()
        .map(|(p, t)| (format!("{p}_begin"), t))
        .collect();
    (paired, leftover)
}

/// Pure helper: render a TimingNode as an ASCII tree.
pub fn render_tree_ascii(root: &TimingNode) -> String {
    let mut out = String::new();
    let line = format_node_line(root);
    out.push_str(&line);
    out.push('\n');
    let n = root.children.len();
    for (i, child) in root.children.iter().enumerate() {
        render_subtree(child, "", i + 1 == n, &mut out);
    }
    out
}

fn format_node_line(n: &TimingNode) -> String {
    let cpu_ms = n.cpu_ns as f64 / 1_000_000.0;
    let gpu_ms = n.gpu_ns as f64 / 1_000_000.0;
    format!(
        "{:<32} cpu={:>8.3} ms  gpu={:>8.3} ms  ×{}",
        n.label, cpu_ms, gpu_ms, n.call_count
    )
}

fn render_subtree(node: &TimingNode, prefix: &str, is_last: bool, out: &mut String) {
    let connector = if is_last { "└── " } else { "├── " };
    out.push_str(prefix);
    out.push_str(connector);
    out.push_str(&format_node_line(node));
    out.push('\n');
    let child_prefix = format!("{prefix}{}", if is_last { "    " } else { "│   " });
    let n = node.children.len();
    for (i, child) in node.children.iter().enumerate() {
        render_subtree(child, &child_prefix, i + 1 == n, out);
    }
}

/// One CSV row representing a single TimingNode in a frame's tree.
#[derive(Debug, Clone, PartialEq)]
pub struct CsvRow {
    pub frame_id: u64,
    pub phase: String,
    pub depth: u32,
    /// Slash-joined path from the root, e.g. `"root/worker/scatter"`.
    pub path: String,
    pub label: String,
    pub cpu_ms: f64,
    pub gpu_ms: f64,
    pub call_count: u32,
}

/// Header line (no trailing newline) for the CSV log produced by
/// [`render_tree_csv_rows`] / [`format_csv_row`].
pub fn csv_header() -> &'static str {
    "frame_id,phase,depth,path,label,cpu_ms,gpu_ms,call_count"
}

/// Walk the tree depth-first, emitting one [`CsvRow`] per node.
/// `path` is the slash-joined chain of labels from the root down to the node
/// (the root's row has `path == root.label`).
pub fn render_tree_csv_rows(root: &TimingNode, frame_id: u64, phase: &str) -> Vec<CsvRow> {
    let mut out = Vec::new();
    walk_csv(root, 0, root.label.clone(), frame_id, phase, &mut out);
    out
}

fn walk_csv(
    node: &TimingNode,
    depth: u32,
    path: String,
    frame_id: u64,
    phase: &str,
    out: &mut Vec<CsvRow>,
) {
    out.push(CsvRow {
        frame_id,
        phase: phase.to_string(),
        depth,
        path: path.clone(),
        label: node.label.clone(),
        cpu_ms: node.cpu_ns as f64 / 1_000_000.0,
        gpu_ms: node.gpu_ns as f64 / 1_000_000.0,
        call_count: node.call_count as u32,
    });
    for child in &node.children {
        let child_path = format!("{path}/{}", child.label);
        walk_csv(child, depth + 1, child_path, frame_id, phase, out);
    }
}

/// Format a single row as one CSV line (no trailing newline).
/// Text fields containing `,`, `"`, `\n`, or `\r` are wrapped in double quotes
/// with internal `"` escaped as `""`.
pub fn format_csv_row(row: &CsvRow) -> String {
    format!(
        "{},{},{},{},{},{:.6},{:.6},{}",
        row.frame_id,
        csv_escape(&row.phase),
        row.depth,
        csv_escape(&row.path),
        csv_escape(&row.label),
        row.cpu_ms,
        row.gpu_ms,
        row.call_count,
    )
}

fn csv_escape(s: &str) -> String {
    if s.contains(',') || s.contains('"') || s.contains('\n') || s.contains('\r') {
        let escaped = s.replace('"', "\"\"");
        format!("\"{escaped}\"")
    } else {
        s.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread::sleep;
    use std::time::Duration;

    #[test]
    fn timing_tree_scope_guard_accumulates_cpu_ns() {
        let mut tree = TimingTree::new();
        {
            let _g = tree.scope("a");
            sleep(Duration::from_millis(2));
        }
        let root = tree.root();
        assert_eq!(root.children.len(), 1);
        let a = &root.children[0];
        assert_eq!(a.label, "a");
        assert!(a.cpu_ns >= 800_000, "expected >=0.8ms, got {} ns", a.cpu_ns);
        assert_eq!(a.call_count, 1);
    }

    #[test]
    fn timing_tree_nested_scopes_form_tree() {
        let mut tree = TimingTree::new();
        {
            let _a = tree.scope("a");
            {
                let _b = tree.scope("b");
            }
            {
                let _c = tree.scope("c");
            }
        }
        let root = tree.root();
        assert_eq!(root.children.len(), 1);
        let a = &root.children[0];
        assert_eq!(a.label, "a");
        assert_eq!(a.children.len(), 2);
        assert_eq!(a.children[0].label, "b");
        assert_eq!(a.children[1].label, "c");
    }

    #[test]
    fn timing_tree_repeat_scope_aggregates() {
        let mut tree = TimingTree::new();
        {
            let _g = tree.scope("a");
        }
        {
            let _g = tree.scope("a");
        }
        let root = tree.root();
        assert_eq!(root.children.len(), 1);
        let a = &root.children[0];
        assert_eq!(a.call_count, 2);
    }

    #[test]
    fn merge_gpu_timestamp_pairs_basic() {
        let input = vec![
            ("foo_begin".to_string(), 100u64),
            ("foo_end".to_string(), 350),
            ("bar_begin".to_string(), 400),
            ("bar_end".to_string(), 600),
        ];
        let (paired, leftover) = merge_gpu_timestamp_pairs(input);
        assert_eq!(
            paired,
            vec![("foo".to_string(), 250u64), ("bar".to_string(), 200)]
        );
        assert!(leftover.is_empty());
    }

    #[test]
    fn merge_gpu_timestamp_pairs_dangling_begin() {
        let input = vec![("foo_begin".to_string(), 100u64)];
        let (paired, leftover) = merge_gpu_timestamp_pairs(input);
        assert!(paired.is_empty());
        assert_eq!(leftover, vec![("foo_begin".to_string(), 100u64)]);
    }

    #[test]
    fn merge_gpu_timestamp_pairs_interleaved() {
        let input = vec![
            ("foo_begin".to_string(), 100u64),
            ("bar_begin".to_string(), 150),
            ("bar_end".to_string(), 200),
            ("foo_end".to_string(), 300),
        ];
        let (paired, leftover) = merge_gpu_timestamp_pairs(input);
        // bar finishes first, foo finishes second; output order is by end-time.
        assert_eq!(
            paired,
            vec![("bar".to_string(), 50u64), ("foo".to_string(), 200)]
        );
        assert!(leftover.is_empty());
    }

    #[test]
    fn render_tree_ascii_box_drawing() {
        let mut tree = TimingTree::new();
        {
            let _a = tree.scope("a");
            {
                let _b = tree.scope("b");
            }
            {
                let _c = tree.scope("c");
            }
        }
        let s = tree.render_ascii();
        assert!(s.contains("├── "), "missing ├── in:\n{s}");
        assert!(s.contains("└── "), "missing └── in:\n{s}");
        assert!(s.contains("a"));
        assert!(s.contains("b"));
        assert!(s.contains("c"));
    }

    #[test]
    fn render_tree_ascii_columns() {
        let mut tree = TimingTree::new();
        tree.add_cpu_ns("foo", 2_000_000);
        tree.add_gpu_ns("foo", 1_500_000);
        let s = tree.render_ascii();
        assert!(s.contains("2.000"), "expected '2.000' in:\n{s}");
        assert!(s.contains("1.500"), "expected '1.500' in:\n{s}");
    }

    #[test]
    fn merge_subtree_attaches_under_label() {
        let mut a = TimingTree::new();
        {
            let _g = a.scope("outer");
        }
        let mut b = TimingTree::new();
        {
            let _g = b.scope("inner");
        }
        a.merge_subtree("worker", b);
        let root = a.root();
        let labels: Vec<&str> = root.children.iter().map(|c| c.label.as_str()).collect();
        assert!(labels.contains(&"worker"));
        let worker = root.children.iter().find(|c| c.label == "worker").unwrap();
        assert!(worker.children.iter().any(|c| c.label == "inner"));
    }

    fn make_node(label: &str) -> TimingNode {
        TimingNode::new(label)
    }

    #[test]
    fn render_tree_csv_rows_emits_one_row_per_node() {
        // root -> A -> (B, C)
        let mut root = make_node("root");
        let mut a = make_node("A");
        a.children.push(make_node("B"));
        a.children.push(make_node("C"));
        root.children.push(a);
        let rows = render_tree_csv_rows(&root, 0, "phase");
        assert_eq!(rows.len(), 4, "expected 4 rows, got {:?}", rows);
        let labels: Vec<&str> = rows.iter().map(|r| r.label.as_str()).collect();
        assert_eq!(labels, vec!["root", "A", "B", "C"]);
    }

    #[test]
    fn render_tree_csv_rows_columns_match_header() {
        let header = csv_header();
        assert_eq!(header.split(',').count(), 8);

        let row = CsvRow {
            frame_id: 1,
            phase: "p,with,comma".to_string(),
            depth: 2,
            path: "root/\"quoted\"".to_string(),
            label: "label\nnewline".to_string(),
            cpu_ms: 1.5,
            gpu_ms: 2.5,
            call_count: 3,
        };
        let line = format_csv_row(&row);
        // parse: an 8-field CSV (counting field separators that are not inside quotes).
        let mut fields = 1;
        let mut in_quotes = false;
        let bytes = line.as_bytes();
        let mut i = 0;
        while i < bytes.len() {
            let c = bytes[i];
            if c == b'"' {
                if in_quotes && i + 1 < bytes.len() && bytes[i + 1] == b'"' {
                    i += 2;
                    continue;
                }
                in_quotes = !in_quotes;
            } else if c == b',' && !in_quotes {
                fields += 1;
            }
            i += 1;
        }
        assert_eq!(fields, 8, "expected 8 csv fields in: {line}");
    }

    #[test]
    fn render_tree_csv_rows_handles_empty_tree() {
        let tree = TimingTree::new();
        let rows = render_tree_csv_rows(tree.root(), 0, "p");
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].label, "root");
        assert_eq!(rows[0].depth, 0);
    }

    #[test]
    fn render_tree_csv_rows_includes_phase_label_and_frame_id() {
        let mut root = make_node("root");
        let mut a = make_node("A");
        a.children.push(make_node("B"));
        root.children.push(a);
        let rows = render_tree_csv_rows(&root, 42, "PHASE_B");
        assert!(!rows.is_empty());
        for r in &rows {
            assert_eq!(r.frame_id, 42);
            assert_eq!(r.phase, "PHASE_B");
        }
    }

    #[test]
    fn render_tree_csv_rows_path_uses_slash_separator() {
        // root -> A -> B; row for B has path "root/A/B"
        let mut root = make_node("root");
        let mut a = make_node("A");
        a.children.push(make_node("B"));
        root.children.push(a);
        let rows = render_tree_csv_rows(&root, 0, "p");
        let b = rows.iter().find(|r| r.label == "B").expect("B row");
        assert_eq!(b.path, "root/A/B");
        let r = rows.iter().find(|r| r.label == "root").unwrap();
        assert_eq!(r.path, "root");
    }

    #[test]
    fn csv_header_lists_all_expected_columns() {
        assert_eq!(
            csv_header(),
            "frame_id,phase,depth,path,label,cpu_ms,gpu_ms,call_count"
        );
    }

    #[test]
    fn nested_thread_scopes_build_expected_tree() {
        let mut tree = TimingTree::new();
        {
            let _g = install_thread_tree(&mut tree);
            let _outer = ThreadScope::enter("process_body");
            {
                let _mid = ThreadScope::enter("apply_mesh_result");
                {
                    let _inner = ThreadScope::enter("pack_vertices");
                }
            }
        }
        let root = tree.root();
        assert_eq!(root.children.len(), 1);
        let pb = &root.children[0];
        assert_eq!(pb.label, "process_body");
        assert_eq!(pb.children.len(), 1);
        let amr = &pb.children[0];
        assert_eq!(amr.label, "apply_mesh_result");
        assert_eq!(amr.children.len(), 1);
        assert_eq!(amr.children[0].label, "pack_vertices");
    }

    #[test]
    fn apply_mesh_result_tree_emits_main_thread_scope_chain() {
        // Simulate the ordered set of scopes the real apply_mesh_result enters
        // for a typical "has mesh" frame.
        let mut tree = TimingTree::new();
        {
            let _g = install_thread_tree(&mut tree);
            let _root = ThreadScope::enter("process_body");
            let _w = ThreadScope::enter("worker_result_consume");
            let _amr = ThreadScope::enter("apply_mesh_result");
            {
                let _scatter = ThreadScope::enter("apply_scatter_mm");
            }
            {
                let _v = ThreadScope::enter("pack_vertices");
            }
            {
                let _i = ThreadScope::enter("pack_indices");
            }
            {
                let _u = ThreadScope::enter("pack_uvs");
            }
            {
                let _s = ThreadScope::enter("add_surface_from_arrays");
            }
            {
                let _m = ThreadScope::enter("material_setup");
                let _lp = ThreadScope::enter("apply_lod_shader_params");
            }
            {
                let _ib = ThreadScope::enter("instance_set_base_and_free_old");
            }
        }
        let root = tree.root();
        let pb = root
            .children
            .iter()
            .find(|c| c.label == "process_body")
            .unwrap();
        let wrc = pb
            .children
            .iter()
            .find(|c| c.label == "worker_result_consume")
            .unwrap();
        let amr = wrc
            .children
            .iter()
            .find(|c| c.label == "apply_mesh_result")
            .unwrap();
        let labels: Vec<&str> = amr.children.iter().map(|c| c.label.as_str()).collect();
        assert!(labels.contains(&"apply_scatter_mm"), "got: {:?}", labels);
        assert!(labels.contains(&"pack_vertices"));
        assert!(labels.contains(&"pack_indices"));
        assert!(labels.contains(&"pack_uvs"));
        assert!(labels.contains(&"add_surface_from_arrays"));
        assert!(labels.contains(&"material_setup"));
        assert!(labels.contains(&"instance_set_base_and_free_old"));
        // material_setup must contain apply_lod_shader_params nested
        let ms = amr
            .children
            .iter()
            .find(|c| c.label == "material_setup")
            .unwrap();
        assert!(ms
            .children
            .iter()
            .any(|c| c.label == "apply_lod_shader_params"));
    }

    #[test]
    fn nested_calls_via_function_boundaries_preserve_path() {
        // Mirrors the celestial.rs structure: process() installs tree + enters
        // process_body scope, calls process_body() (a fn) which enters
        // worker_result_consume and calls apply_mesh_result() which enters
        // its own scope and inner scatter scope.
        fn apply_mesh() {
            let _scope = ThreadScope::enter("apply_mesh_result");
            {
                let _g = ThreadScope::enter("apply_scatter_mm");
            }
            {
                let _g = ThreadScope::enter("pack_vertices");
            }
        }
        fn process_body_inner() {
            {
                let _g = ThreadScope::enter("worker_result_consume");
                apply_mesh();
            }
        }
        let mut tree = TimingTree::new();
        {
            let _tg = install_thread_tree(&mut tree);
            let _root = ThreadScope::enter("process_body");
            process_body_inner();
        }
        let root = tree.root();
        assert_eq!(root.children.len(), 1, "tree:\n{}", tree.render_ascii());
        let pb = &root.children[0];
        assert_eq!(pb.label, "process_body");
        let wrc = pb
            .children
            .iter()
            .find(|c| c.label == "worker_result_consume")
            .unwrap();
        let amr = wrc
            .children
            .iter()
            .find(|c| c.label == "apply_mesh_result")
            .unwrap();
        let labels: Vec<&str> = amr.children.iter().map(|c| c.label.as_str()).collect();
        assert!(labels.contains(&"apply_scatter_mm"));
        assert!(labels.contains(&"pack_vertices"));
    }

    #[test]
    fn disabled_tree_skips_scope_work() {
        let mut tree = TimingTree::disabled();
        {
            let _g = tree.scope("a");
            sleep(Duration::from_millis(1));
        }
        assert!(tree.root().children.is_empty());
    }
}
