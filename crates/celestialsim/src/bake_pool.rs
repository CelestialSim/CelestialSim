//! Background chunk-surface baking (the fast-flight stutter fix), generic over
//! any [`CpuSurfaceProvider`].
//!
//! Baking a chunk's surface costs several ms (a CPU fBm evaluation, …);
//! running it on the main thread capped fast flight to a handful
//! of admissions per frame. This pool moves the work onto worker threads: the
//! planet enqueues bake jobs for chunks it wants to admit, and only admits a
//! chunk into the GPU cache once its surface is READY — ancestor stand-ins cover
//! the ground in the meantime, so delayed admission is invisible.
//!
//! The pool is provider-agnostic: every worker calls
//! [`CpuSurfaceProvider::bake`], and all source-specific state (tile caches,
//! streaming, height scaling) lives behind the provider.

use std::collections::HashSet;
use std::sync::mpsc;
use std::sync::{Arc, Mutex, RwLock};
use std::thread::JoinHandle;

use celestial_algo::clipmap::FaceFrame;
use celestial_algo::quadtree::{Chunk, ChunkId};

use crate::surface::{ChunkSurface, CpuSurfaceProvider};

/// One bake outcome. `surface == None` means the job was CANCELLED (its chunk
/// left the wanted set before a worker got to it) — the in-flight entry is
/// released so the chunk re-queues if it comes back into view.
pub struct BakeResult {
    pub chunk: Chunk,
    pub surface: Option<ChunkSurface>,
    /// The provider generation at the time the bake was REQUESTED. Private:
    /// only the pool constructs results, and only [`BakePool::poll`] compares
    /// it. Stamping the REQUEST (not the bake) means a job queued before a
    /// param edit is rejected even if a worker happens to run it after the
    /// edit — "request time < last parameter update time ⇒ reject".
    req_gen: u64,
}

struct BakeJob {
    frame: FaceFrame,
    chunk: Chunk,
    /// Generation current when this job was requested (see [`BakeResult::req_gen`]).
    req_gen: u64,
}

/// The provider plus the generation it belongs to. A live param edit installs a
/// new provider AND bumps `gen`; workers read both under one guard, so a result
/// is always stamped with exactly the params it was baked from.
struct ProviderSlot {
    gen: u64,
    provider: Arc<dyn CpuSurfaceProvider>,
}

/// Fallback bound for the result channel when the caller passes 0 (see
/// [`BakePool::new`]).
const DEFAULT_RESULT_CAP: usize = 128;

/// Worker pool resampling chunk surfaces off the main thread.
pub struct BakePool {
    /// `Option` so `Drop` can close the channel and stop the workers.
    req_tx: Option<mpsc::Sender<BakeJob>>,
    /// BOUNDED (CEL-91). Each `BakeResult` owns a full `ChunkSurface`
    /// (`12 × tile_res²` bytes = 0.75 MiB at tile_res 256); an unbounded channel
    /// let finished bakes pile up on the heap whenever the main thread drained
    /// slower than the workers produced. With a `sync_channel` the workers BLOCK
    /// on `send` once `cap` results are undrained, which is the backpressure that
    /// actually caps the memory. Shutdown is safe: `Drop` closes the request
    /// channel and the receiver is dropped with the pool, so a blocked `send`
    /// returns `Err` and the worker exits (never deadlocking the join).
    ///
    /// `Option` for exactly that reason: `Drop` must DROP the receiver BEFORE it
    /// joins the workers. A struct field is only dropped after `Drop::drop`
    /// returns, so leaving it in place would keep a `send`-blocked worker blocked
    /// forever and hang the join.
    res_rx: Option<mpsc::Receiver<BakeResult>>,
    /// Chunks queued or being baked (dedup; cleared by [`poll`](Self::poll)).
    in_flight: HashSet<ChunkId>,
    /// Camera-driven cancellation: workers skip chunks not in this set (the
    /// current cut). `None` (initial) means everything is wanted.
    wanted: Arc<RwLock<Option<HashSet<ChunkId>>>>,
    /// Shared, SWAPPABLE provider + its generation. A live param edit installs a
    /// fresh provider here ([`set_provider`](Self::set_provider)); workers read it
    /// per bake, so edits take effect without tearing down + respawning worker
    /// threads (which would block the caller on in-flight bakes).
    provider: Arc<RwLock<ProviderSlot>>,
    /// Main-thread mirror of the current generation. [`poll`](Self::poll) drops
    /// any result stamped with an older one (it was baked from stale params).
    generation: u64,
    workers: Vec<JoinHandle<()>>,
}

impl BakePool {
    /// Spawn `n_workers` baking `tile_res × tile_res` surfaces via `provider`.
    ///
    /// `result_cap` bounds the number of FINISHED-but-undrained results held on
    /// the heap (each one a full `ChunkSurface`); pass the caller's in-flight
    /// bound. 0 means "use [`DEFAULT_RESULT_CAP`]".
    pub fn new(
        provider: Arc<dyn CpuSurfaceProvider>,
        tile_res: u32,
        n_workers: usize,
        result_cap: usize,
    ) -> Self {
        let cap = if result_cap == 0 { DEFAULT_RESULT_CAP } else { result_cap };
        let (req_tx, req_rx) = mpsc::channel::<BakeJob>();
        // Bounded: workers block on `send` rather than growing the heap.
        let (res_tx, res_rx) = mpsc::sync_channel::<BakeResult>(cap);
        let req_rx = Arc::new(Mutex::new(req_rx));
        let wanted: Arc<RwLock<Option<HashSet<ChunkId>>>> = Arc::new(RwLock::new(None));
        let provider: Arc<RwLock<ProviderSlot>> =
            Arc::new(RwLock::new(ProviderSlot { gen: 0, provider }));

        let mut workers = Vec::with_capacity(n_workers.max(1));
        for _ in 0..n_workers.max(1) {
            let req_rx = Arc::clone(&req_rx);
            let res_tx = res_tx.clone();
            let provider = Arc::clone(&provider);
            let wanted = Arc::clone(&wanted);
            workers.push(std::thread::spawn(move || {
                loop {
                    // Hold the queue lock only across the blocking recv.
                    let job = {
                        let guard = match req_rx.lock() {
                            Ok(g) => g,
                            Err(_) => break,
                        };
                        guard.recv()
                    };
                    let job = match job {
                        Ok(j) => j,
                        Err(_) => break, // channel closed
                    };
                    // Read the CURRENT provider (a live edit may have swapped
                    // it). The result is stamped with the job's REQUEST-time
                    // generation, so `poll` drops any bake whose request
                    // predates the latest param edit — regardless of which
                    // provider the worker happened to read here.
                    let prov = match provider.read() {
                        Ok(g) => Arc::clone(&g.provider),
                        Err(_) => break,
                    };
                    // Cancel stale jobs: if the chunk left the wanted set while
                    // queued, release it without baking so fresh work never
                    // waits behind flown-past terrain.
                    let stale = match wanted.read() {
                        Ok(w) => w.as_ref().map(|w| !w.contains(&job.chunk.id)).unwrap_or(false),
                        Err(_) => false,
                    };
                    if stale {
                        let out =
                            BakeResult { chunk: job.chunk, surface: None, req_gen: job.req_gen };
                        if res_tx.send(out).is_err() {
                            break;
                        }
                        continue;
                    }
                    let surface = prov.bake(&job.frame, &job.chunk, tile_res);
                    let out =
                        BakeResult { chunk: job.chunk, surface: Some(surface), req_gen: job.req_gen };
                    if res_tx.send(out).is_err() {
                        break; // receiver gone
                    }
                }
            }));
        }

        BakePool {
            req_tx: Some(req_tx),
            res_rx: Some(res_rx),
            in_flight: HashSet::new(),
            wanted,
            provider,
            generation: 0,
            workers,
        }
    }

    /// Install a fresh provider (e.g. after a live editor param edit) and open a
    /// new GENERATION. Combine with a cache `invalidate_all` + clearing any ready
    /// surfaces so every chunk re-bakes with the new params.
    ///
    /// Bumping the generation is what makes a live edit atomic across the planet.
    /// Without it, the bakes ALREADY RUNNING when the edit lands finish against the
    /// OLD params and are applied anyway, so those chunks keep old-param terrain
    /// (e.g. the old `water_height`) while every other chunk shows the new value —
    /// the planet ends up remembering two different values at once. Releasing
    /// `in_flight` matters just as much: otherwise `request`'s dedup sees those
    /// chunks as still pending and SILENTLY SKIPS their re-bake, so they are never
    /// refreshed at all.
    pub fn set_provider(&mut self, provider: Arc<dyn CpuSurfaceProvider>) {
        if let Ok(mut g) = self.provider.write() {
            g.gen += 1;
            g.provider = provider;
            self.generation = g.gen;
        }
        // Results for these are now stale and will be dropped by `poll`; forget
        // them so the caller's re-request actually queues a fresh bake.
        self.in_flight.clear();
    }

    /// Replace the wanted-chunk set (the current cut) workers use to cancel
    /// stale queued bakes. `None` initial state means everything is wanted.
    pub fn set_wanted(&self, ids: HashSet<ChunkId>) {
        if let Ok(mut w) = self.wanted.write() {
            *w = Some(ids);
        }
    }

    /// Queue a bake for `chunk` unless one is already queued/running. The job
    /// is stamped with the CURRENT generation (its "request time"): if a param
    /// edit lands before the result comes back, [`poll`](Self::poll) rejects it.
    pub fn request(&mut self, frame: &FaceFrame, chunk: &Chunk) {
        if !self.in_flight.insert(chunk.id) {
            return;
        }
        if let Some(tx) = &self.req_tx {
            let job = BakeJob { frame: *frame, chunk: *chunk, req_gen: self.generation };
            if tx.send(job).is_err() {
                self.in_flight.remove(&chunk.id);
            }
        }
    }

    /// The current generation (bumped by every [`set_provider`](Self::set_provider)).
    /// The planet stamps its own per-chunk bookkeeping (ready surfaces) with this
    /// so a surface can also be rejected at RENDER time, not just on arrival.
    pub fn generation(&self) -> u64 {
        self.generation
    }

    /// Is a bake for `id` queued or running?
    pub fn in_flight(&self, id: ChunkId) -> bool {
        self.in_flight.contains(&id)
    }

    /// Number of queued/running bakes (backpressure signal).
    pub fn in_flight_len(&self) -> usize {
        self.in_flight.len()
    }

    /// Drain completed bakes non-blocking.
    ///
    /// Results whose REQUEST predates the current generation (a param edit has
    /// landed since they were queued) are DROPPED, never handed back. The chunk
    /// was already released from `in_flight` by [`set_provider`], so the
    /// caller's re-request re-bakes it with the current params.
    pub fn poll(&mut self) -> Vec<BakeResult> {
        let mut raw = Vec::new();
        if let Some(rx) = self.res_rx.as_ref() {
            while let Ok(r) = rx.try_recv() {
                raw.push(r);
            }
        }
        let mut out = Vec::new();
        for r in raw {
            self.in_flight.remove(&r.chunk.id);
            if r.req_gen != self.generation {
                continue; // requested before the last param edit — discard
            }
            out.push(r);
        }
        out
    }
}

impl Drop for BakePool {
    fn drop(&mut self) {
        self.req_tx = None; // close the request channel; idle workers exit
        // The result channel is BOUNDED, so a worker can be blocked in `send`
        // right now. Dropping the receiver BEFORE the join makes that `send`
        // return `Err` and the worker break out of its loop; leaving it alive
        // (as a plain field would, since fields drop only after this returns)
        // would hang the join forever.
        self.res_rx = None;
        for h in self.workers.drain(..) {
            let _ = h.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::noise_provider::{NoiseParams, NoiseProvider};
    use celestial_algo::quadtree::{base_face_frames, select_chunks};
    use godot::builtin::Vector3;
    use std::time::Duration;

    /// A built-in [`NoiseProvider`] (CPU fBm, no network), for driving the
    /// generic pool with a real provider in tests.
    fn surface_provider() -> Arc<dyn CpuSurfaceProvider> {
        Arc::new(NoiseProvider::new(NoiseParams::default()))
    }

    /// A provider that bakes a CONSTANT height, slowly. `value` identifies which
    /// params produced a surface, so a test can tell a stale bake from a fresh one.
    struct ConstProvider {
        value: f32,
        delay: Duration,
    }

    impl CpuSurfaceProvider for ConstProvider {
        fn bake(&self, _f: &FaceFrame, _c: &Chunk, tile_res: u32) -> ChunkSurface {
            std::thread::sleep(self.delay);
            let n = (tile_res * tile_res) as usize;
            ChunkSurface {
                color: vec![255u8; n * 4],
                height: vec![self.value; n],
                normal: vec![128u8; n * 4],
            }
        }
        fn height_scale(&self) -> f32 {
            1.0
        }
    }

    /// A live param edit swaps the provider. Any bake ALREADY RUNNING was started
    /// against the OLD params, so its result is stale and must never be applied —
    /// otherwise the chunks that happened to be mid-bake keep old-param terrain
    /// while everything else updates (the "only some chunks update" bug).
    ///
    /// Equally, `set_provider` must release those chunks from `in_flight`, or
    /// `request`'s dedup silently SUPPRESSES the re-bake and the chunk is never
    /// refreshed at all.
    #[test]
    fn provider_swap_discards_in_flight_stale_bakes() {
        const STALE: f32 = 1.0;
        const FRESH: f32 = 2.0;

        let mut pool = BakePool::new(
            Arc::new(ConstProvider { value: STALE, delay: Duration::from_millis(300) }),
            16,
            1,
            16,
        );

        let frames = base_face_frames(6371.0);
        let cam = Vector3::new(0.0, 0.0, 6373.0);
        let cut = select_chunks(&frames, cam, 0.05, 20, 6, None);
        let chunk = cut[0];
        let frame = &frames[chunk.id.face as usize];

        // Start a bake, then let the worker actually pick it up and read the OLD
        // provider before we swap.
        pool.request(frame, &chunk);
        std::thread::sleep(Duration::from_millis(60));

        // Live edit: new params.
        pool.set_provider(Arc::new(ConstProvider { value: FRESH, delay: Duration::ZERO }));

        // The in-flight (now stale) bake must not block a re-request.
        assert!(
            !pool.in_flight(chunk.id),
            "set_provider must release in-flight chunks so they can be re-baked"
        );
        pool.request(frame, &chunk);

        // Drain for a while. The STALE surface must NEVER be handed back.
        let mut fresh_seen = false;
        for _ in 0..400 {
            for r in pool.poll() {
                if let Some(s) = r.surface {
                    assert_ne!(
                        s.height[0], STALE,
                        "a bake started with the OLD provider leaked through as a surface"
                    );
                    if s.height[0] == FRESH {
                        fresh_seen = true;
                    }
                }
            }
            if fresh_seen {
                break;
            }
            std::thread::sleep(Duration::from_millis(5));
        }
        assert!(fresh_seen, "the chunk must be re-baked with the NEW provider");
    }

    /// "Request time < last parameter update time ⇒ reject": a job QUEUED
    /// before an edit must be dropped even when a worker only dequeues it
    /// AFTER the swap (and so bakes it with the new provider). One worker,
    /// slow first job: the second job is guaranteed still queued at swap time.
    #[test]
    fn results_requested_before_an_edit_are_rejected_even_if_baked_after() {
        let mut pool = BakePool::new(
            Arc::new(ConstProvider { value: 1.0, delay: Duration::from_millis(200) }),
            16,
            1,
            16,
        );

        let frames = base_face_frames(6371.0);
        let cut = select_chunks(&frames, Vector3::new(0.0, 0.0, 6373.0), 0.05, 20, 6, None);
        let (a, b) = (cut[0], cut[1]);

        pool.request(&frames[a.id.face as usize], &a); // worker picks this up
        pool.request(&frames[b.id.face as usize], &b); // still queued...
        std::thread::sleep(Duration::from_millis(60));
        // ...when the edit lands. `b` will be baked with the NEW provider, but
        // its REQUEST predates the edit — it must still be rejected.
        pool.set_provider(Arc::new(ConstProvider { value: 2.0, delay: Duration::ZERO }));

        let deadline = std::time::Instant::now() + Duration::from_secs(2);
        while std::time::Instant::now() < deadline {
            assert!(
                pool.poll().is_empty(),
                "a result requested before the param edit was handed back"
            );
            std::thread::sleep(Duration::from_millis(5));
        }
    }

    #[test]
    fn pool_bakes_off_thread_and_dedups() {
        let provider = surface_provider();
        let mut pool = BakePool::new(provider, 32, 2, 16);

        let frames = base_face_frames(6371.0);
        let cam = Vector3::new(0.0, 0.0, 6373.0);
        let cut = select_chunks(&frames, cam, 0.05, 20, 6, None);
        let chunk = cut[0];
        let frame = &frames[chunk.id.face as usize];

        pool.request(frame, &chunk);
        pool.request(frame, &chunk); // dedup
        assert_eq!(pool.in_flight_len(), 1);

        let mut results = Vec::new();
        for _ in 0..500 {
            results.extend(pool.poll());
            if !results.is_empty() {
                break;
            }
            std::thread::sleep(Duration::from_millis(2));
        }
        assert_eq!(results.len(), 1, "one deduped bake result");
        let r = &results[0];
        assert_eq!(r.chunk.id, chunk.id);
        // Surface carries the baked CPU-noise height field (finite; may be
        // negative where the seabed dips below sea level).
        let surface = r.surface.as_ref().expect("baked, not cancelled");
        assert!(surface.height[0].is_finite());
        assert!(!pool.in_flight(chunk.id), "in-flight cleared after poll");
    }

    #[test]
    fn stale_jobs_are_cancelled_not_baked() {
        // set_wanted(empty) BEFORE requesting: whenever the worker dequeues the
        // job, the chunk is already unwanted → cancelled outcome, in-flight
        // released, and the chunk is re-requestable.
        let provider = surface_provider();
        let mut pool = BakePool::new(provider, 32, 1, 16);
        let frames = base_face_frames(6371.0);
        let cut = select_chunks(&frames, Vector3::new(0.0, 0.0, 6373.0), 0.05, 20, 6, None);
        let chunk = cut[0];
        let frame = &frames[chunk.id.face as usize];

        pool.set_wanted(HashSet::new()); // nothing is wanted
        pool.request(frame, &chunk);
        let mut results = Vec::new();
        for _ in 0..500 {
            results.extend(pool.poll());
            if !results.is_empty() {
                break;
            }
            std::thread::sleep(Duration::from_millis(2));
        }
        assert_eq!(results.len(), 1);
        assert!(results[0].surface.is_none(), "stale job must be cancelled, not baked");
        assert!(!pool.in_flight(chunk.id), "cancelled job re-requestable");

        // Wanted again → the re-request bakes for real.
        let mut wanted = HashSet::new();
        wanted.insert(chunk.id);
        pool.set_wanted(wanted);
        pool.request(frame, &chunk);
        let mut results = Vec::new();
        for _ in 0..500 {
            results.extend(pool.poll());
            if !results.is_empty() {
                break;
            }
            std::thread::sleep(Duration::from_millis(2));
        }
        assert!(results[0].surface.is_some(), "wanted-again chunk bakes normally");
    }

    /// CEL-91 shutdown safety: the result channel is BOUNDED, so workers can be
    /// blocked in `send` when the pool is dropped. `Drop` must drop the receiver
    /// BEFORE joining, or the join hangs forever. Queue far more work than the
    /// (cap-1) channel can hold, never poll, then drop — the drop must return.
    #[test]
    fn drop_does_not_deadlock_with_workers_blocked_on_a_full_result_channel() {
        let (done_tx, done_rx) = mpsc::channel::<()>();
        let h = std::thread::spawn(move || {
            let mut pool = BakePool::new(
                Arc::new(ConstProvider { value: 1.0, delay: Duration::from_millis(1) }),
                8,
                2,
                1, // capacity 1: workers block on the 2nd undrained result
            );
            let frames = base_face_frames(6371.0);
            let cut = select_chunks(&frames, Vector3::new(0.0, 0.0, 6373.0), 0.05, 20, 6, None);
            for chunk in cut.iter().take(64) {
                pool.request(&frames[chunk.id.face as usize], chunk);
            }
            // Give the workers time to fill the channel and BLOCK in `send`.
            std::thread::sleep(Duration::from_millis(120));
            drop(pool); // must not hang
            let _ = done_tx.send(());
        });
        assert!(
            done_rx.recv_timeout(Duration::from_secs(10)).is_ok(),
            "BakePool::drop deadlocked against a worker blocked on the bounded channel"
        );
        h.join().unwrap();
    }

    /// The bounded channel is backpressure, not data loss: with a small cap the
    /// results still all arrive, one drain at a time.
    #[test]
    fn a_small_result_cap_still_delivers_every_result() {
        let mut pool = BakePool::new(surface_provider(), 8, 2, 1);
        let frames = base_face_frames(6371.0);
        let cut = select_chunks(&frames, Vector3::new(0.0, 0.0, 6373.0), 0.05, 20, 6, None);
        let wanted: HashSet<ChunkId> = cut.iter().take(8).map(|c| c.id).collect();
        pool.set_wanted(wanted);
        for chunk in cut.iter().take(8) {
            pool.request(&frames[chunk.id.face as usize], chunk);
        }

        let mut n = 0;
        for _ in 0..2000 {
            n += pool.poll().len();
            if n == 8 {
                break;
            }
            std::thread::sleep(Duration::from_millis(2));
        }
        assert_eq!(n, 8, "every requested bake must come back despite the cap of 1");
        assert_eq!(pool.in_flight_len(), 0);
    }
}
