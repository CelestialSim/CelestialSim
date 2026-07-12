//! RAII ownership of Godot `RenderingDevice` RIDs (CEL-91).
//!
//! [`Owned<K>`] is the **one** handle type and carries the **only** `Drop` impl for GPU
//! resources in this crate; [`RidSink`] is the only place a `free_rid` call may live.
//! `K` is a phantom marker ([`Buffer`], [`Texture`], [`Shader`], [`Pipeline`],
//! [`UniformSet`]) — Godot frees every kind through the same `free_rid`, so the marker
//! carries no behaviour, only compile-time separation.
//!
//! # DROP-ORDER CONTRACT (read this before adding fields to a GPU-resource struct)
//!
//! `RenderingDevice` frees *dependents together with their parent*: freeing a shader or a
//! buffer also invalidates the uniform sets and pipelines derived from it. Therefore
//! **uniform sets and pipelines must be freed BEFORE the shaders / buffers / textures they
//! derive from**, or the later `free_rid` hits an already-dead RID.
//!
//! **Nothing in this code enforces that.** It falls out of struct **field declaration
//! order**, because Rust drops fields top-to-bottom. Consumers (e.g. `ChunkGpuResources`)
//! must declare, in this order:
//!
//! ```ignore
//! struct ChunkGpuResources {
//!     // 1. uniform sets
//!     set: RdUniformSet,
//!     // 2. pipelines
//!     pipeline: RdPipeline,
//!     // 3. shaders / buffers / textures (the parents)
//!     shader: RdShader,
//!     buffer: RdBuffer,
//! }
//! ```
//!
//! [`MainDeviceSink`]'s drain is **FIFO**, so the enqueue order produced by that field
//! order is exactly the order the RIDs are handed to `free_rid`.

use std::marker::PhantomData;
use std::sync::atomic::AtomicBool;
#[cfg(not(test))]
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex, Weak};

use godot::builtin::Rid;
#[cfg(not(test))]
use godot::builtin::{Callable, Variant};
use godot::classes::{MultiMesh, RenderingServer};
#[allow(unused_imports)]
use godot::prelude::*;

/// WHO deallocates a RID.
///
/// This trait's implementations are the only legal `free_rid` call sites in the crate.
pub trait RidSink: Send + Sync {
    /// Release `rid` (possibly deferred — see [`MainDeviceSink`]).
    fn free(&self, rid: Rid);
}

/// WHAT kind of RID — a compile-time marker only.
pub trait RdKind {
    /// Human-readable kind name, for diagnostics.
    const LABEL: &'static str;
}

/// Storage-buffer marker.
pub struct Buffer;
/// Texture marker.
pub struct Texture;
/// Compute-shader marker.
pub struct Shader;
/// Compute-pipeline marker.
pub struct Pipeline;
/// Uniform-set marker.
pub struct UniformSet;

impl RdKind for Buffer {
    const LABEL: &'static str = "buffer";
}
impl RdKind for Texture {
    const LABEL: &'static str = "texture";
}
impl RdKind for Shader {
    const LABEL: &'static str = "shader";
}
impl RdKind for Pipeline {
    const LABEL: &'static str = "pipeline";
}
impl RdKind for UniformSet {
    const LABEL: &'static str = "uniform_set";
}

/// The one RAII handle for a `RenderingDevice` RID.
///
/// Dropping it (including by *overwriting* it with a new handle) frees the RID through the
/// sink it was created with. See the module docs for the drop-order contract.
pub struct Owned<K: RdKind> {
    rid: Rid,
    sink: Arc<dyn RidSink>,
    _kind: PhantomData<K>,
}

impl<K: RdKind> Owned<K> {
    /// Take ownership of `rid`, to be released through `sink`.
    pub fn new(rid: Rid, sink: Arc<dyn RidSink>) -> Self {
        Self {
            rid,
            sink,
            _kind: PhantomData,
        }
    }

    /// A `Rid::Invalid` placeholder for a not-yet-built field. Never freed.
    pub fn invalid(sink: Arc<dyn RidSink>) -> Self {
        Self::new(Rid::Invalid, sink)
    }

    /// The owned RID (still owned by `self` — do not free it).
    pub fn rid(&self) -> Rid {
        self.rid
    }

    /// Whether this handle holds a real (non-`Invalid`) RID.
    pub fn is_valid(&self) -> bool {
        self.rid.is_valid()
    }
}

impl<K: RdKind> Drop for Owned<K> {
    fn drop(&mut self) {
        if self.rid.is_valid() {
            self.sink.free(self.rid);
        }
    }
}

impl<K: RdKind> std::fmt::Debug for Owned<K> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Owned<{}>({})", K::LABEL, self.rid)
    }
}

/// Owned storage buffer.
pub type RdBuffer = Owned<Buffer>;
/// Owned texture.
pub type RdTexture = Owned<Texture>;
/// Owned compute shader.
pub type RdShader = Owned<Shader>;
/// Owned compute pipeline.
pub type RdPipeline = Owned<Pipeline>;
/// Owned uniform set.
pub type RdUniformSet = Owned<UniformSet>;

/// Sink for the **main** `RenderingDevice`.
///
/// `free_rid` on the main device is render-thread only, but `Drop` fires on whatever thread
/// the owner happens to die on. So `free` merely enqueues the RID and, the first time the
/// queue goes non-empty, schedules exactly **one** drain on the render thread. The drain
/// releases the queued RIDs in **FIFO** order (see the module drop-order contract).
pub struct MainDeviceSink {
    queue: Mutex<Vec<Rid>>,
    scheduled: AtomicBool,
    /// Self-reference, so `free(&self)` can hand an owning `Arc` to the render-thread
    /// callable (which must be `'static`).
    #[cfg_attr(test, allow(dead_code))]
    me: Weak<MainDeviceSink>,
}

impl MainDeviceSink {
    /// Create a sink. `Arc` because every [`Owned`] handle shares it.
    pub fn new() -> Arc<Self> {
        Arc::new_cyclic(|me| Self {
            queue: Mutex::new(Vec::new()),
            scheduled: AtomicBool::new(false),
            me: me.clone(),
        })
    }

    /// Free every queued RID, FIFO. **Render thread only.**
    #[cfg(not(test))]
    fn drain(&self) {
        self.scheduled.store(false, Ordering::SeqCst);
        let rids: Vec<Rid> = std::mem::take(&mut *self.queue.lock().unwrap());
        if rids.is_empty() {
            return;
        }
        let Some(mut rd) = RenderingServer::singleton().get_rendering_device() else {
            return;
        };
        for rid in rids {
            rd.free_rid(rid);
        }
    }

    /// RIDs queued but not yet drained (tests only — there is no Godot server under
    /// `cargo test`, so nothing is ever scheduled).
    #[cfg(test)]
    pub fn pending(&self) -> Vec<Rid> {
        self.queue.lock().unwrap().clone()
    }
}

impl RidSink for MainDeviceSink {
    #[cfg(not(test))]
    fn free(&self, rid: Rid) {
        {
            let mut q = self.queue.lock().unwrap();
            q.push(rid);
        }
        // Schedule the drain exactly once per pending batch. The callable keeps the sink
        // alive until the render thread has run it.
        if !self.scheduled.swap(true, Ordering::SeqCst) {
            let Some(sink) = self.me.upgrade() else {
                self.scheduled.store(false, Ordering::SeqCst);
                return;
            };
            let callable = Callable::from_sync_fn("ces_gpu_free_drain", move |_args| {
                sink.drain();
                Variant::nil()
            });
            RenderingServer::singleton().call_on_render_thread(&callable);
        }
    }

    #[cfg(test)]
    fn free(&self, rid: Rid) {
        // No Godot server under `cargo test`: only enqueue.
        let _ = &self.scheduled;
        self.queue.lock().unwrap().push(rid);
    }
}

/// Sink for a **local** `RenderingDevice` (the GPU parity tests in `gpu/chunk_gpu_test.rs`).
///
/// Godot frees a local device's resources when the device itself is freed, and the device
/// outlives them — so releasing individual RIDs is unnecessary: this is a no-op.
pub struct LocalDeviceSink;

impl LocalDeviceSink {
    /// Create a shareable no-op sink.
    #[allow(dead_code)]
    pub fn new() -> Arc<Self> {
        Arc::new(Self)
    }
}

impl RidSink for LocalDeviceSink {
    fn free(&self, _rid: Rid) {}
}

/// A `MultiMesh` allocated with `use_indirect`, plus the cleanup Godot forgets.
///
/// Godot never releases an indirect `MultiMesh`'s internal command buffer when the
/// multimesh is freed — an engine bug, reproducible with no CelestialSim code at all
/// (allocate four indirect multimeshes in GDScript, drop them, and Godot reports
/// `4 RIDs of type "StorageBuffer" were leaked` on exit). Since every planet rebuild
/// (a builder re-set, a `tile_res` edit, a scatter-layer change) allocates a fresh set,
/// the leak grows without bound.
///
/// So the multimesh is owned through this handle: on drop it releases the command buffer
/// **before** letting go of the `Gd<MultiMesh>`, which is the free Godot should have done.
/// If a future Godot fixes the bug, this becomes a double free — it will announce itself
/// loudly as a "free of invalid RID" error on the next engine bump, which is the failure
/// mode we want (noisy, not silent).
pub struct IndirectMultiMesh {
    mm: Gd<MultiMesh>,
    sink: Arc<dyn RidSink>,
}

impl IndirectMultiMesh {
    /// Take ownership of an already-`multimesh_allocate_data`'d indirect multimesh.
    pub fn new(mm: Gd<MultiMesh>, sink: Arc<dyn RidSink>) -> Self {
        Self { mm, sink }
    }
}

impl std::ops::Deref for IndirectMultiMesh {
    type Target = Gd<MultiMesh>;
    fn deref(&self) -> &Self::Target {
        &self.mm
    }
}

impl Drop for IndirectMultiMesh {
    fn drop(&mut self) {
        // Resolve the command buffer HERE rather than at construction: the renderer
        // creates it lazily and can recreate it, so the RID is only knowable now — while
        // the multimesh is still alive, which is precisely why this must happen on drop
        // and not after the `Gd` is released.
        let cmd = RenderingServer::singleton().multimesh_get_command_buffer_rd_rid(self.mm.get_rid());
        if cmd.is_valid() {
            self.sink.free(cmd);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use godot::builtin::Rid;
    use std::sync::{Arc, Mutex};

    #[derive(Default)]
    struct SpySink {
        freed: Mutex<Vec<Rid>>,
    }

    impl SpySink {
        fn new() -> Arc<Self> {
            Arc::new(Self::default())
        }
        fn freed(&self) -> Vec<Rid> {
            self.freed.lock().unwrap().clone()
        }
    }

    impl RidSink for SpySink {
        fn free(&self, rid: Rid) {
            self.freed.lock().unwrap().push(rid);
        }
    }

    #[test]
    fn dropping_a_handle_frees_its_rid_through_the_sink() {
        let spy = SpySink::new();
        {
            let _buf: RdBuffer = Owned::new(Rid::new(7), spy.clone());
            assert!(spy.freed().is_empty());
        }
        assert_eq!(spy.freed(), vec![Rid::new(7)]);
    }

    #[test]
    fn an_invalid_rid_is_never_freed() {
        let spy = SpySink::new();
        {
            let h: RdTexture = Owned::invalid(spy.clone());
            assert!(!h.is_valid());
            assert_eq!(h.rid(), Rid::Invalid);
        }
        assert!(spy.freed().is_empty());
    }

    #[test]
    fn field_order_frees_dependents_before_parents() {
        // Fields drop top-to-bottom: the uniform set (a dependent) before its shader.
        #[allow(dead_code)]
        struct Res {
            set: RdUniformSet,
            shader: RdShader,
        }
        let spy = SpySink::new();
        {
            let _r = Res {
                set: Owned::new(Rid::new(1), spy.clone()),
                shader: Owned::new(Rid::new(2), spy.clone()),
            };
        }
        assert_eq!(spy.freed(), vec![Rid::new(1), Rid::new(2)]);
    }

    #[test]
    fn overwriting_a_handle_frees_the_old_rid() {
        let spy = SpySink::new();
        let mut set: RdUniformSet = Owned::new(Rid::new(10), spy.clone());
        assert_eq!(set.rid(), Rid::new(10));
        // Rebinding a live handle drops the old one — the chunk_gpu.rs:598 leak, made
        // impossible.
        set = Owned::new(Rid::new(11), spy.clone());
        assert_eq!(spy.freed(), vec![Rid::new(10)]);
        assert_eq!(set.rid(), Rid::new(11));
        drop(set);
        assert_eq!(spy.freed(), vec![Rid::new(10), Rid::new(11)]);
    }

    #[test]
    fn main_device_sink_queues_instead_of_freeing_inline() {
        let sink = MainDeviceSink::new();
        {
            let _p: RdPipeline = Owned::new(Rid::new(42), sink.clone());
        }
        assert_eq!(sink.pending(), vec![Rid::new(42)]);
    }
}
