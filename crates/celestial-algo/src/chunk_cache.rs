// ChunkCache — pure-CPU slot allocator for GPU terrain chunks.
// Maps ChunkId -> stable cache slot (u32); LRU eviction protects visible chunks.

use std::collections::{BTreeSet, HashMap, HashSet};

use crate::quadtree::{Chunk, ChunkId};

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// What changed during one `update` call.
pub struct CacheDiff {
    /// Chunks that must be realized into the GPU slot (slot, chunk descriptor).
    pub realize: Vec<(u32, Chunk)>,
    /// Slots that were evicted this frame (populated by Task 3).
    pub evicted: Vec<u32>,
    /// One cache slot per DRAWN instance this frame. Every cut chunk is covered:
    /// either by its own slot (resident) or — when it isn't resident yet
    /// (throttled admission, teleport, over-budget) — by the slot of its nearest
    /// **resident ancestor**, drawn as a coarse stand-in so the ground never has a
    /// hole. Ancestor stand-ins are deduped (one instance per stand-in slot).
    pub visible_slots: Vec<u32>,
    /// Parallel to `visible_slots`: `Some(i)` ⇒ this drawn instance is `cut[i]`
    /// (use its geomorph factor); `None` ⇒ an ancestor stand-in (draw at full
    /// detail, morph 1.0). Lets the caller keep per-instance morph aligned with
    /// the drawn set even though some drawn chunks are not in the current cut.
    pub visible_cut_idx: Vec<Option<usize>>,
}

// ---------------------------------------------------------------------------
// Internal bookkeeping
// ---------------------------------------------------------------------------

struct Resident {
    slot: u32,
    last_used: u64,
}

// ---------------------------------------------------------------------------
// ChunkCache
// ---------------------------------------------------------------------------

/// Pure-CPU mapping from `ChunkId` to stable GPU cache slots.
///
/// Slots are allocated from a free list and (eventually, Task 3) reclaimed via
/// LRU eviction when the budget is full.
pub struct ChunkCache {
    budget: u32,
    /// One entry per slot; `None` = free.
    slots: Vec<Option<ChunkId>>,
    /// Resident set: live chunks and where they live.
    resident: HashMap<ChunkId, Resident>,
    /// Stack of free slot indices (popped cheapest first).
    free: Vec<u32>,
    /// LRU index: `(last_used, id)` for every resident chunk, ordered so the
    /// global minimum is the least-recently-used chunk. Eviction pops the front
    /// in O(log n) instead of scanning the whole resident set (the O(misses ×
    /// budget) cliff that tanked FPS once the cache filled).
    lru: BTreeSet<(u64, ChunkId)>,
    /// Monotonically increasing frame counter; bumped at the top of `update`.
    frame: u64,
    /// Chunks that need to be re-realized on the next `update` (set by
    /// `invalidate_all`).  A resident chunk in this set is pushed to
    /// `CacheDiff.realize` — with its existing slot unchanged — and then
    /// removed from the set.
    dirty: HashSet<ChunkId>,
}

impl ChunkCache {
    /// Create a new cache with `budget` slots, all empty.
    pub fn new(budget: u32) -> Self {
        let slots = vec![None; budget as usize];
        // Seed free list in reverse so slot 0 is popped first.
        let free: Vec<u32> = (0..budget).rev().collect();
        Self {
            budget,
            slots,
            resident: HashMap::new(),
            free,
            lru: BTreeSet::new(),
            frame: 0,
            dirty: HashSet::new(),
        }
    }

    /// Process one frame's visible cut with NO per-frame promotion limit.
    pub fn update(&mut self, cut: &[Chunk]) -> CacheDiff {
        self.update_throttled(cut, u32::MAX)
    }

    /// Process one frame's visible cut, promoting at most `max_new` previously
    /// non-resident chunks this frame. Returns what must be realized and which
    /// slots the cut maps to.
    ///
    /// When the free list is exhausted the cache evicts the resident with the
    /// smallest `last_used` that is **not** in the current cut (visible chunks
    /// are always protected).  The freed slot is reused for the incoming chunk
    /// and reported in `CacheDiff.evicted`.
    ///
    /// `max_new` caps how many *new* chunks (cut misses) are admitted per frame,
    /// so a large influx (teleport / fast turn / first fill) spreads its
    /// realize+bake over several frames instead of spiking one. Unadmitted misses
    /// are simply absent from `visible_slots` this frame (not drawn — never drawn
    /// before realized) and reappear in the next cut to be admitted then. Already
    /// resident chunks are never throttled for *drawing* (always touched + drawn), but
    /// their **dirty re-realizes share the `max_new` cap**: a burst of streamed-tile
    /// arrivals dirtying hundreds of resident chunks re-bakes at most `max_new` per
    /// frame (drawn stale meanwhile); the rest stay dirty and drain over the
    /// following frames.
    pub fn update_throttled(&mut self, cut: &[Chunk], max_new: u32) -> CacheDiff {
        self.update_throttled_gated(cut, max_new, &|_| true)
    }

    /// [`update_throttled`](Self::update_throttled) with an **admission gate**:
    /// a non-resident cut chunk is admitted (and a dirty resident re-realized)
    /// only when `admissible(id)` — e.g. "its surface patch finished baking on a
    /// worker thread". Gated-out chunks stay covered by their nearest resident
    /// ancestor, so deferring admission never costs coverage.
    pub fn update_throttled_gated(
        &mut self,
        cut: &[Chunk],
        max_new: u32,
        admissible: &dyn Fn(ChunkId) -> bool,
    ) -> CacheDiff {
        self.frame += 1;
        let frame = self.frame;

        let mut realize = Vec::new();
        let mut evicted = Vec::new();
        let mut visible_slots = Vec::with_capacity(cut.len());
        let mut visible_cut_idx: Vec<Option<usize>> = Vec::with_capacity(cut.len());
        let mut promoted = 0u32;

        // Pass 1 — touch every resident chunk that is in the cut up to `frame`.
        // After this, ALL visible chunks have `last_used == frame` (the max), so
        // the global LRU minimum is guaranteed to be a non-visible chunk — no
        // per-eviction scan/filter needed. Dirty hits re-realize in place.
        for chunk in cut {
            let id = chunk.id;
            if let Some(r) = self.resident.get_mut(&id) {
                self.lru.remove(&(r.last_used, id));
                r.last_used = frame;
                self.lru.insert((frame, id));
                if promoted < max_new && self.dirty.contains(&id) && admissible(id) {
                    self.dirty.remove(&id);
                    realize.push((r.slot, *chunk));
                    promoted += 1;
                }
            }
        }

        // Pass 1.5 — for every cut chunk that is NOT resident, find its
        // resident COVER (descendants when zooming out, else the nearest
        // ancestor when zooming in) and touch every member to `frame`. This
        // both (a) records the stand-ins we'll draw in its place (no hole, no
        // black flash) and (b) protects them from eviction in pass 2 exactly
        // like visible chunks — the `lu < frame` guard below then can't
        // reclaim a slot we're about to draw.
        let mut covers: Vec<(ChunkId, u32)> = Vec::new();
        for chunk in cut {
            if self.resident.contains_key(&chunk.id) {
                continue;
            }
            covers.clear();
            self.resident_cover(chunk.id, &mut covers);
            for &(cid, _) in &covers {
                if let Some(r) = self.resident.get_mut(&cid) {
                    if r.last_used != frame {
                        self.lru.remove(&(r.last_used, cid));
                        r.last_used = frame;
                        self.lru.insert((frame, cid));
                    }
                }
            }
        }

        // Pass 2 — resolve a slot for every cut chunk, in cut order. Every cut
        // chunk contributes exactly one covering slot (its own, or an ancestor
        // stand-in), so `visible_slots` never leaves a hole. Ancestor stand-ins
        // are deduped: a coarse parent standing in for several throttled children
        // is drawn once.
        let mut standin_seen: HashSet<u32> = HashSet::new();
        for (i, chunk) in cut.iter().enumerate() {
            let id = chunk.id;
            if let Some(r) = self.resident.get(&id) {
                visible_slots.push(r.slot); // hit — already touched in pass 1
                visible_cut_idx.push(Some(i));
                continue;
            }
            // Try to admit this new chunk while the frame's new-chunk budget
            // lasts and the caller's gate allows it (e.g. patch baked).
            if promoted < max_new && admissible(id) {
                // Miss — claim a free slot, else evict the global LRU (O(log n)).
                let slot = if let Some(s) = self.free.pop() {
                    Some(s)
                } else {
                    match self.lru.iter().next().copied() {
                        // The minimum has `last_used < frame` => it is NOT in the
                        // current cut and not a protected stand-in ancestor: safe.
                        Some((lu, victim)) if lu < frame => {
                            self.lru.remove(&(lu, victim));
                            let v = self.resident.remove(&victim).unwrap();
                            self.dirty.remove(&victim); // keep dirty ⊆ resident
                            self.slots[v.slot as usize] = None;
                            evicted.push(v.slot);
                            Some(v.slot)
                        }
                        // Every slot holds a chunk placed/protected THIS frame =>
                        // over budget. Fall through to an ancestor stand-in.
                        _ => None,
                    }
                };
                if let Some(slot) = slot {
                    self.slots[slot as usize] = Some(id);
                    self.resident.insert(id, Resident { slot, last_used: frame });
                    self.lru.insert((frame, id));
                    realize.push((slot, *chunk));
                    visible_slots.push(slot);
                    visible_cut_idx.push(Some(i));
                    promoted += 1;
                    continue;
                }
            }
            // Not admitted this frame (throttled / over-budget / teleport):
            // draw its resident COVER as stand-ins so the ground never
            // disappears — resident DESCENDANTS when zooming out (the children
            // that just left the cut keep tiling the parent's area, at the
            // same detail — the "black flash on zoom-out" fix), else the
            // nearest resident ancestor when zooming in (coarse but complete).
            // The chunk itself streams in over the next frames.
            let mut cover: Vec<(ChunkId, u32)> = Vec::new();
            self.resident_cover(id, &mut cover);
            for (_, cslot) in cover {
                if standin_seen.insert(cslot) {
                    visible_slots.push(cslot);
                    visible_cut_idx.push(None); // stand-in — full detail (morph 1)
                }
            }
            // Empty cover (rare: first-ever fill) => nothing to draw for this
            // chunk this frame; it fills in once admitted.
        }

        CacheDiff { realize, evicted, visible_slots, visible_cut_idx }
    }

    /// The cache slot a chunk currently occupies, or `None` if not resident.
    /// (Resident ⇔ realized & drawable; used to align per-instance data with the
    /// `visible_slots` a throttled `update` actually admitted.)
    pub fn slot_of(&self, id: ChunkId) -> Option<u32> {
        self.resident.get(&id).map(|r| r.slot)
    }

    /// Nearest **resident** ancestor of `id`: walk up the quadtree (parent id =
    /// `depth - 1`, `path >> 2`, same face) until a resident chunk is found.
    /// Used to draw a coarse stand-in for a cut chunk that isn't realized yet,
    /// so throttled admission never leaves a hole in the ground.
    fn nearest_resident_ancestor(&self, id: ChunkId) -> Option<(ChunkId, u32)> {
        let mut depth = id.depth;
        let mut path = id.path;
        while depth > 0 {
            depth -= 1;
            path >>= 2;
            let aid = ChunkId { face: id.face, depth, path };
            if let Some(r) = self.resident.get(&aid) {
                return Some((aid, r.slot));
            }
        }
        None
    }

    /// How deep [`resident_cover`](Self::resident_cover) searches for resident
    /// descendants below a missing chunk. Zooming out coarsens one level per
    /// step, so the just-departed children are at `depth + 1`; a small margin
    /// covers fast zooms without letting the walk fan out (4^depth).
    const COVER_DESCENT: u8 = 3;

    /// Resident stand-ins that COVER the area of a non-resident cut chunk.
    /// Preference: resident **descendants** (the finer chunks that tiled this
    /// area a moment ago — same detail, seamless; the zoom-out case), else the
    /// nearest resident **ancestor** (coarse but complete; the zoom-in case).
    /// Descendant coverage may legitimately be partial (a missing subtree stays
    /// uncovered exactly like today's admission gap).
    fn resident_cover(&self, id: ChunkId, out: &mut Vec<(ChunkId, u32)>) {
        let before = out.len();
        self.collect_resident_descendants(id, Self::COVER_DESCENT, out);
        if out.len() == before {
            if let Some(a) = self.nearest_resident_ancestor(id) {
                out.push(a);
            }
        }
    }

    fn collect_resident_descendants(
        &self,
        id: ChunkId,
        remaining: u8,
        out: &mut Vec<(ChunkId, u32)>,
    ) {
        if remaining == 0 {
            return;
        }
        for k in 0..4u64 {
            let child =
                ChunkId { face: id.face, depth: id.depth + 1, path: (id.path << 2) | k };
            if let Some(r) = self.resident.get(&child) {
                out.push((child, r.slot));
            } else {
                self.collect_resident_descendants(child, remaining - 1, out);
            }
        }
    }

    /// Mark a single resident chunk dirty so the next `update`/`update_throttled`
    /// re-realizes it in place (same slot). Use this for targeted re-bakes — e.g.
    /// when a streamed asset (map tile) that exactly this chunk was waiting on
    /// arrives — instead of [`invalidate_all`], which would re-bake the whole
    /// resident set every time a tile lands. No-op if the chunk isn't resident.
    pub fn mark_dirty(&mut self, id: ChunkId) {
        if self.resident.contains_key(&id) {
            self.dirty.insert(id);
        }
    }

    /// Is `id` flagged for re-realize?
    ///
    /// A CPU-surface provider needs this to decide whether to re-bake a chunk's
    /// surface. [`invalidate_all`](Self::invalidate_all) keeps every chunk
    /// RESIDENT and merely marks it dirty, so a "not resident" test alone would
    /// never re-request those chunks — they would re-realize against their stale
    /// surface and keep old-param terrain forever.
    pub fn is_dirty(&self, id: ChunkId) -> bool {
        self.dirty.contains(&id)
    }

    /// Number of chunks currently resident in the cache.
    pub fn resident_count(&self) -> u32 {
        self.resident.len() as u32
    }

    /// Maximum number of slots this cache was created with.
    pub fn budget(&self) -> u32 {
        self.budget
    }

    /// Mark every resident chunk as needing re-realization on the next `update`.
    ///
    /// Call this when terrain parameters change (e.g. height-layer edit) so
    /// every visible chunk is re-shaded from scratch.  Slot assignments are
    /// **preserved**; the next `update` will push dirty residents into
    /// `CacheDiff.realize` with their existing slots, then clear the dirty flag.
    ///
    /// Returns the current resident `(slot, id)` pairs (in unspecified order)
    /// so the caller can enqueue GPU work immediately if desired.
    pub fn invalidate_all(&mut self) -> Vec<(u32, ChunkId)> {
        let pairs: Vec<(u32, ChunkId)> = self
            .resident
            .iter()
            .map(|(id, r)| (r.slot, *id))
            .collect();
        for (_, id) in &pairs {
            self.dirty.insert(*id);
        }
        pairs
    }

    /// Evict every resident chunk that was NOT part of the most recent
    /// `update`'s drawn set — i.e. neither in the cut nor protecting it as a
    /// stand-in (both are touched to the current frame, so "off-screen" is
    /// exactly `last_used < frame`). Returns the freed slots.
    ///
    /// Call this when terrain parameters change: the off-screen residents hold
    /// surfaces baked with the OLD params, and keeping them means a revisited
    /// zone DRAWS that old terrain for the frames its re-bake takes. Evicting
    /// makes a revisit indistinguishable from a first visit — the admission
    /// gate holds the chunk back behind a fresh-baked ancestor stand-in, so
    /// stale data can never reach the screen. On-screen chunks are kept (they
    /// re-bake in place via `invalidate_all`, avoiding a LOD flash mid-edit).
    pub fn evict_offscreen(&mut self) -> Vec<u32> {
        let mut evicted = Vec::new();
        while let Some(&(lu, id)) = self.lru.iter().next() {
            if lu >= self.frame {
                break; // everything from here on was drawn/protected this frame
            }
            self.lru.remove(&(lu, id));
            let r = self.resident.remove(&id).expect("lru entry must be resident");
            self.dirty.remove(&id); // keep dirty ⊆ resident
            self.slots[r.slot as usize] = None;
            self.free.push(r.slot);
            evicted.push(r.slot);
        }
        evicted
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quadtree::{Bary, Chunk, ChunkId};
    use godot::builtin::Vector3;

    /// Minimal test helper: build a Chunk with dummy geometry; the cache only
    /// keys on `ChunkId`.
    fn chunk(face: u8, depth: u8, path: u64) -> Chunk {
        let b = Bary { wb: 0.0, wc: 0.0 };
        Chunk {
            id: ChunkId { face, depth, path },
            bary: [b, b, b],
            corners: [Vector3::ZERO, Vector3::ZERO, Vector3::ZERO],
            level: depth,
        }
    }

    // ----- first-frame realize -----------------------------------------------

    #[test]
    fn throttled_subdivision_covers_children_with_resident_parent() {
        // The "ground disappears while moving" bug: when the cut refines a
        // resident parent into 4 children but the throttle admits them over
        // several frames, the parent must keep covering the area as a stand-in
        // (`visible_cut_idx == None`) until every child is resident.
        let mut cache = ChunkCache::new(64);
        let parent = chunk(0, 3, 0b101010);
        cache.update(&[parent]); // parent resident
        let kids: Vec<Chunk> =
            (0..4).map(|k| chunk(0, 4, (0b101010 << 2) | k)).collect();

        // Frame 1: only 2 children admitted; the parent stands in for the rest.
        let d1 = cache.update_throttled(&kids, 2);
        assert_eq!(d1.realize.len(), 2);
        // Drawn set: 2 children + the parent stand-in (deduped) = 3 instances.
        assert_eq!(d1.visible_slots.len(), 3, "children + parent stand-in");
        let parent_slot = cache.slot_of(parent.id).unwrap();
        assert!(d1.visible_slots.contains(&parent_slot), "parent covers the gap");
        // Morph mapping: the stand-in is flagged None, admitted children Some.
        assert_eq!(
            d1.visible_cut_idx.iter().filter(|i| i.is_none()).count(),
            1,
            "exactly one stand-in instance"
        );

        // Frame 2: remaining children admitted; no stand-in needed.
        let d2 = cache.update_throttled(&kids, 2);
        assert_eq!(d2.realize.len(), 2);
        assert_eq!(d2.visible_slots.len(), 4, "all four children drawn");
        assert!(d2.visible_cut_idx.iter().all(|i| i.is_some()), "no stand-ins");
        assert!(!d2.visible_slots.contains(&parent_slot), "parent retired");
    }

    #[test]
    fn zoom_out_is_covered_by_resident_children_until_parent_admits() {
        // The "black chunks flash when zooming out" bug: the cut coarsens from
        // 4 resident children to their (non-resident, not-yet-baked) parent.
        // Until the parent is admissible, the CHILDREN must keep covering the
        // area as stand-ins — same detail, zero flash.
        let mut cache = ChunkCache::new(64);
        let parent = chunk(0, 3, 0b11);
        let kids: Vec<Chunk> = (0..4).map(|k| chunk(0, 4, (0b11 << 2) | k)).collect();
        cache.update(&kids); // children resident (the zoomed-in state)
        let kid_slots: Vec<u32> =
            kids.iter().map(|c| cache.slot_of(c.id).unwrap()).collect();

        // Zoom out: cut = [parent], but its bake isn't ready (gate = false).
        let d1 = cache.update_throttled_gated(&[parent], 8, &|_| false);
        assert!(d1.realize.is_empty(), "parent not admissible yet");
        assert_eq!(d1.visible_slots.len(), 4, "all four children stand in");
        for s in &kid_slots {
            assert!(d1.visible_slots.contains(s), "child slot {s} covers the parent");
        }
        assert!(d1.visible_cut_idx.iter().all(|i| i.is_none()), "all stand-ins");

        // Bake lands: parent admitted, children retire from the drawn set.
        let d2 = cache.update_throttled_gated(&[parent], 8, &|_| true);
        assert_eq!(d2.realize.len(), 1, "parent realized");
        assert_eq!(d2.visible_slots.len(), 1, "parent alone covers now");
        assert_eq!(d2.visible_cut_idx, vec![Some(0)]);
    }

    #[test]
    fn deep_zoom_out_covers_via_grandchildren() {
        // Fast zoom-out can coarsen more than one level per frame: with only
        // GRANDchildren resident, they must still cover the missing chunk.
        let mut cache = ChunkCache::new(64);
        let top = chunk(0, 2, 0b1);
        let grandkids: Vec<Chunk> = (0..16)
            .map(|k| chunk(0, 4, (0b1 << 4) | k))
            .collect();
        cache.update(&grandkids);

        let d = cache.update_throttled_gated(&[top], 8, &|_| false);
        assert_eq!(d.visible_slots.len(), 16, "all 16 grandchildren stand in");
        assert!(d.visible_cut_idx.iter().all(|i| i.is_none()));
    }

    #[test]
    fn standin_ancestor_is_never_evicted_while_covering() {
        // A stand-in ancestor must be LRU-protected like a visible chunk: under
        // slot pressure the evictor must not reclaim the very slot that is
        // covering unadmitted children this frame.
        let mut cache = ChunkCache::new(6);
        let parent = chunk(0, 3, 7);
        cache.update(&[parent]);
        // Fill remaining 5 slots with unrelated chunks (they leave the cut).
        let others: Vec<Chunk> = (10..15).map(|p| chunk(1, 3, p)).collect();
        let mut both = vec![parent];
        both.extend(others.iter().copied());
        cache.update(&both);

        // New cut: parent's 4 children + 4 more new chunks — heavy pressure,
        // tiny budget. The parent must survive as long as any child is missing.
        let mut cut: Vec<Chunk> = (0..4).map(|k| chunk(0, 4, (7 << 2) | k)).collect();
        cut.extend((20..24).map(|p| chunk(1, 3, p)));
        for _ in 0..8 {
            let d = cache.update_throttled(&cut, 1);
            // Every drawn slot must hold a live resident chunk.
            for &s in &d.visible_slots {
                assert!(
                    cache.slots[s as usize].is_some(),
                    "drawn slot {s} must be resident"
                );
            }
            let all_kids_resident =
                (0..4).all(|k| cache.slot_of(ChunkId { face: 0, depth: 4, path: (7 << 2) | k }).is_some());
            if !all_kids_resident {
                assert!(
                    cache.slot_of(parent.id).is_some(),
                    "parent evicted while still covering unadmitted children"
                );
            }
        }
    }

    #[test]
    fn dirty_rebakes_share_the_realize_throttle() {
        // Regression for the streamed-tile stutter: a burst of tile arrivals
        // dirtying MANY resident chunks must not re-realize them all in one
        // frame — dirty re-bakes share `max_new` and drain over frames, while
        // every resident chunk keeps being DRAWN (stale) meanwhile.
        let mut cache = ChunkCache::new(64);
        let cut: Vec<Chunk> = (0..12).map(|p| chunk(0, 3, p)).collect();
        cache.update(&cut); // all 12 resident, clean

        for c in &cut {
            cache.mark_dirty(c.id); // tile burst: everything waiting re-bakes
        }

        // Frame 1: at most 5 re-realized, ALL 12 still drawn.
        let d1 = cache.update_throttled(&cut, 5);
        assert_eq!(d1.realize.len(), 5, "dirty re-bakes capped by max_new");
        assert_eq!(d1.visible_slots.len(), 12, "throttled-dirty chunks still drawn");

        // Frames 2..: the backlog drains without exceeding the cap.
        let d2 = cache.update_throttled(&cut, 5);
        assert_eq!(d2.realize.len(), 5);
        let d3 = cache.update_throttled(&cut, 5);
        assert_eq!(d3.realize.len(), 2, "remaining dirty drain");
        let d4 = cache.update_throttled(&cut, 5);
        assert!(d4.realize.is_empty(), "backlog fully drained");

        // No chunk was re-realized twice and none was lost.
        let mut all: Vec<u64> = [&d1, &d2, &d3]
            .iter()
            .flat_map(|d| d.realize.iter().map(|(_, c)| c.id.path))
            .collect();
        all.sort_unstable();
        assert_eq!(all, (0..12).collect::<Vec<u64>>());
    }

    #[test]
    fn dirty_and_new_chunks_share_one_budget() {
        // 4 dirty residents + 4 brand-new chunks with max_new = 4: the frame
        // does 4 realizes TOTAL (not 4 + 4), the rest follow next frame.
        let mut cache = ChunkCache::new(64);
        let old: Vec<Chunk> = (0..4).map(|p| chunk(0, 3, p)).collect();
        cache.update(&old);
        for c in &old {
            cache.mark_dirty(c.id);
        }
        let mut cut = old.clone();
        cut.extend((10..14).map(|p| chunk(0, 3, p)));

        let d1 = cache.update_throttled(&cut, 4);
        assert_eq!(d1.realize.len(), 4, "shared budget across dirty + new");
        let d2 = cache.update_throttled(&cut, 4);
        assert_eq!(d2.realize.len(), 4, "remainder next frame");
        let d3 = cache.update_throttled(&cut, 4);
        assert!(d3.realize.is_empty());
    }

    #[test]
    fn throttle_caps_new_promotions_and_fills_over_frames() {
        let mut cache = ChunkCache::new(64);
        let cut: Vec<Chunk> = (0..10).map(|p| chunk(0, 3, p)).collect();

        // Frame 1: admit at most 3 new chunks.
        let d1 = cache.update_throttled(&cut, 3);
        assert_eq!(d1.realize.len(), 3, "max_new=3 admits 3 new chunks");
        assert_eq!(d1.visible_slots.len(), 3, "only admitted chunks are drawable");

        // Frame 2: the 3 stay resident (drawn, not re-realized); 3 more admitted.
        let d2 = cache.update_throttled(&cut, 3);
        assert_eq!(d2.realize.len(), 3, "3 more new chunks admitted");
        assert_eq!(d2.visible_slots.len(), 6, "6 now resident + drawn");

        // Frames 3-4: the rest fill in, then steady state (nothing new).
        let d3 = cache.update_throttled(&cut, 3);
        assert_eq!(d3.visible_slots.len(), 9);
        let d4 = cache.update_throttled(&cut, 3);
        assert_eq!(d4.realize.len(), 1, "last straggler");
        assert_eq!(d4.visible_slots.len(), 10, "all 10 resident");
        let d5 = cache.update_throttled(&cut, 3);
        assert!(d5.realize.is_empty(), "steady state: no re-realize");
        assert_eq!(d5.visible_slots.len(), 10);
    }

    #[test]
    fn first_update_realizes_all_chunks() {
        let mut cache = ChunkCache::new(64);
        let cut = vec![chunk(0, 1, 0), chunk(0, 1, 1), chunk(0, 1, 2)];
        let diff = cache.update(&cut);

        assert_eq!(diff.realize.len(), 3, "all 3 chunks must be realized on first frame");
        assert!(diff.evicted.is_empty(), "no evictions on first frame");
    }

    #[test]
    fn first_update_slots_are_distinct() {
        let mut cache = ChunkCache::new(64);
        let cut = vec![chunk(0, 1, 0), chunk(0, 1, 1), chunk(0, 1, 2), chunk(1, 0, 0)];
        let diff = cache.update(&cut);

        let mut slots: Vec<u32> = diff.realize.iter().map(|(s, _)| *s).collect();
        slots.sort_unstable();
        slots.dedup();
        assert_eq!(slots.len(), 4, "each realized chunk must get a unique slot");
    }

    #[test]
    fn visible_slots_matches_cut_order() {
        let mut cache = ChunkCache::new(64);
        let cut = vec![chunk(0, 1, 0), chunk(0, 1, 1), chunk(0, 1, 2)];
        let diff = cache.update(&cut);

        assert_eq!(diff.visible_slots.len(), 3);
        // visible_slots[i] must be the slot allocated to cut[i]
        let realize_map: std::collections::HashMap<u64, u32> = diff
            .realize
            .iter()
            .map(|(s, c)| (c.id.path, *s))
            .collect();
        for (i, c) in cut.iter().enumerate() {
            assert_eq!(diff.visible_slots[i], realize_map[&c.id.path]);
        }
    }

    // ----- cache hit / idempotence -------------------------------------------

    #[test]
    fn second_identical_update_produces_no_realize() {
        let mut cache = ChunkCache::new(64);
        let cut = vec![chunk(0, 1, 0), chunk(0, 1, 1), chunk(0, 1, 2)];
        cache.update(&cut);
        let diff2 = cache.update(&cut);

        assert!(diff2.realize.is_empty(), "cache hit: no re-realize on identical second update");
        assert!(diff2.evicted.is_empty());
    }

    #[test]
    fn second_update_visible_slots_are_same_as_first() {
        let mut cache = ChunkCache::new(64);
        let cut = vec![chunk(0, 1, 0), chunk(0, 1, 1), chunk(0, 1, 2)];
        let diff1 = cache.update(&cut);
        let diff2 = cache.update(&cut);

        assert_eq!(diff1.visible_slots, diff2.visible_slots,
            "slots must be stable across frames for the same cut");
    }

    // ----- resident_count / budget -------------------------------------------

    #[test]
    fn resident_count_reflects_allocated_slots() {
        let mut cache = ChunkCache::new(64);
        assert_eq!(cache.resident_count(), 0);

        let cut = vec![chunk(0, 1, 0), chunk(0, 1, 1)];
        cache.update(&cut);
        assert_eq!(cache.resident_count(), 2);

        // A second identical update must not change the count.
        cache.update(&cut);
        assert_eq!(cache.resident_count(), 2);
    }

    #[test]
    fn budget_returns_the_value_passed_to_new() {
        let cache = ChunkCache::new(128);
        assert_eq!(cache.budget(), 128);
    }

    // ----- LRU eviction (Task 3) ---------------------------------------------

    /// Budget=3; cuts rotate a new chunk in each frame. After the budget fills,
    /// every new chunk must evict the LRU non-visible resident.
    ///
    /// Frame 1: [A, B, C]   → resident {A(1), B(1), C(1)}, no evictions
    /// Frame 2: [A, B, D]   → A/B hit, D misses → evict C (LRU, not in cut)
    ///                         evicted=[C's slot], realize=[D], resident_count=3
    #[test]
    fn lru_eviction_evicts_oldest_non_visible() {
        let mut cache = ChunkCache::new(3);

        // Frame 1: fill the cache.
        let cut1 = vec![chunk(0, 1, 0), chunk(0, 1, 1), chunk(0, 1, 2)]; // A B C
        let diff1 = cache.update(&cut1);
        assert!(diff1.evicted.is_empty());
        assert_eq!(cache.resident_count(), 3);

        // Remember C's slot — it should be the evicted slot.
        let slot_c = diff1.visible_slots[2];

        // Frame 2: D replaces C in the cut; budget full so eviction required.
        let cut2 = vec![chunk(0, 1, 0), chunk(0, 1, 1), chunk(0, 1, 3)]; // A B D
        let diff2 = cache.update(&cut2);

        // Residency must not exceed budget.
        assert_eq!(cache.resident_count(), 3, "residency must not exceed budget");

        // Exactly one eviction: C (the only non-visible LRU).
        assert_eq!(diff2.evicted.len(), 1, "one eviction expected");
        assert_eq!(diff2.evicted[0], slot_c, "evicted slot is C's old slot");

        // D must be realized into the freed slot.
        assert_eq!(diff2.realize.len(), 1, "D must be realized");
        assert_eq!(diff2.realize[0].0, slot_c, "D reuses C's freed slot");
        assert_eq!(diff2.realize[0].1.id.path, 3, "realized chunk is D");
    }

    /// A chunk that is visible (in the current cut) must never be evicted.
    ///
    /// Budget=2; after filling with [A, B]:
    ///   Frame 2: [A, C] → A is visible → only B can be evicted; assert A stays.
    #[test]
    fn visible_chunk_is_never_evicted() {
        let mut cache = ChunkCache::new(2);

        let cut1 = vec![chunk(0, 1, 0), chunk(0, 1, 1)]; // A B
        let diff1 = cache.update(&cut1);
        let slot_a = diff1.visible_slots[0];
        let slot_b = diff1.visible_slots[1];

        // Frame 2: [A, C] — A is visible, B is the LRU non-visible.
        let cut2 = vec![chunk(0, 1, 0), chunk(0, 1, 2)]; // A C
        let diff2 = cache.update(&cut2);

        // B must be evicted, not A.
        assert_eq!(diff2.evicted.len(), 1);
        assert_eq!(diff2.evicted[0], slot_b, "B must be evicted, not A");
        assert_ne!(diff2.evicted[0], slot_a, "A (visible) must not be evicted");

        // C must be realized into B's old slot.
        assert_eq!(diff2.realize.len(), 1);
        assert_eq!(diff2.realize[0].0, slot_b);

        // A keeps its slot.
        assert_eq!(diff2.visible_slots[0], slot_a, "A's slot is stable");

        assert_eq!(cache.resident_count(), 2);
    }

    /// Multiple evictions in one update: each new chunk evicts the current LRU.
    ///
    /// Budget=3; after [A, B, C] (all last_used=1):
    ///   Frame 2: [D, E, A] — two misses, one hit.
    ///   The cut protects D, E, A.  B and C (last_used=1) are evictable.
    ///   Two slots must be freed; residency stays at 3.
    #[test]
    fn multiple_evictions_in_one_update() {
        let mut cache = ChunkCache::new(3);

        let cut1 = vec![chunk(0, 1, 0), chunk(0, 1, 1), chunk(0, 1, 2)]; // A B C
        cache.update(&cut1);
        assert_eq!(cache.resident_count(), 3);

        // Frame 2: two new chunks + one hit → two evictions needed.
        let cut2 = vec![chunk(0, 1, 3), chunk(0, 1, 4), chunk(0, 1, 0)]; // D E A
        let diff2 = cache.update(&cut2);

        assert_eq!(diff2.evicted.len(), 2, "two evictions required");
        assert_eq!(diff2.realize.len(), 2, "D and E realized");
        assert_eq!(cache.resident_count(), 3, "residency stays at budget");
    }

    /// Regression guard for the O(misses × budget) eviction cliff: one full-cache
    /// update with many misses must stay cheap (O(misses · log n)). The pre-fix
    /// linear scan took ~0.4 s here; the LRU index is ~sub-ms. Generous 100 ms
    /// bound to avoid CI-noise flakiness while still catching an O(n²) regression.
    #[test]
    fn full_cache_eviction_is_not_quadratic() {
        use std::time::Instant;
        let budget = 32768u32;
        let mut cache = ChunkCache::new(budget);
        let fill: Vec<Chunk> = (0..budget as u64).map(|p| chunk_p(p)).collect();
        cache.update(&fill); // fill to budget

        let cut: Vec<Chunk> = (budget as u64..budget as u64 + 1000).map(chunk_p).collect();
        let t = Instant::now();
        let diff = cache.update(&cut); // 1000 misses, all evicting
        let ms = t.elapsed().as_secs_f64() * 1000.0;

        assert_eq!(diff.evicted.len(), 1000, "1000 misses must evict 1000 slots");
        assert!(ms < 100.0, "full-cache update took {ms:.1} ms — eviction is quadratic again");
    }

    /// Over-budget cut (more unique chunks than slots) must NOT panic — it
    /// degrades to fewer drawn chunks. (Pre-fix this hit an `.expect()` panic.)
    #[test]
    fn over_budget_cut_degrades_without_panic() {
        let mut cache = ChunkCache::new(4);
        let cut: Vec<Chunk> = (0..10).map(chunk_p).collect(); // 10 > budget 4
        let diff = cache.update(&cut);
        assert_eq!(cache.resident_count(), 4, "residency capped at budget");
        assert!(diff.visible_slots.len() <= 4, "only budget chunks get slots");
    }

    /// A chunk by path only (helper for the perf/over-budget tests).
    fn chunk_p(path: u64) -> Chunk {
        chunk(0, 10, path)
    }

    // ----- invalidate_all (Task 4) -------------------------------------------

    /// After residency is established, `invalidate_all` must:
    ///   1. Return all resident `(slot, id)` pairs.
    ///   2. Cause the *next* `update(same cut)` to re-realize those chunks WITHOUT
    ///      changing their slot assignments.
    ///   3. A subsequent identical `update` (third call) is a clean cache hit again.
    #[test]
    fn invalidate_all_returns_resident_pairs_and_forces_rerealise() {
        let mut cache = ChunkCache::new(64);
        let cut = vec![chunk(0, 1, 0), chunk(0, 1, 1), chunk(0, 1, 2)];

        // Frame 1: populate the cache.
        let diff1 = cache.update(&cut);
        assert_eq!(diff1.realize.len(), 3, "sanity: 3 chunks realized on frame 1");

        // Record the slots assigned after frame 1.
        let slot0 = diff1.visible_slots[0];
        let slot1 = diff1.visible_slots[1];
        let slot2 = diff1.visible_slots[2];

        // Invalidate the whole cache.
        let mut invalid = cache.invalidate_all();
        // Must return all 3 resident (slot, id) pairs (order not guaranteed).
        assert_eq!(invalid.len(), 3, "invalidate_all must return all 3 resident pairs");
        invalid.sort_by_key(|(s, _)| *s);
        let slots_from_invalid: Vec<u32> = invalid.iter().map(|(s, _)| *s).collect();
        let mut expected_slots = vec![slot0, slot1, slot2];
        expected_slots.sort_unstable();
        assert_eq!(slots_from_invalid, expected_slots, "returned slots must match original assignments");

        // Invalidated chunks stay RESIDENT and are merely marked dirty. A
        // CPU-surface provider must therefore drive its re-bake off `is_dirty`,
        // NOT off "is it resident" — gating on residency alone would never
        // re-bake these, and they would keep old-param terrain forever.
        for c in &cut {
            assert!(cache.slot_of(c.id).is_some(), "invalidate_all must keep chunks resident");
            assert!(cache.is_dirty(c.id), "invalidate_all must mark every chunk dirty");
        }

        // Frame 2: same cut — must re-realize all 3 WITH THE SAME SLOTS.
        let diff2 = cache.update(&cut);
        assert_eq!(diff2.realize.len(), 3, "after invalidate_all, all chunks must be re-realized");
        assert!(diff2.evicted.is_empty(), "invalidation must not evict anything");
        for c in &cut {
            assert!(!cache.is_dirty(c.id), "re-realizing must clear the dirty flag");
        }
        // Slots unchanged.
        assert_eq!(diff2.visible_slots[0], slot0, "slot for chunk 0 must be unchanged");
        assert_eq!(diff2.visible_slots[1], slot1, "slot for chunk 1 must be unchanged");
        assert_eq!(diff2.visible_slots[2], slot2, "slot for chunk 2 must be unchanged");
        // Verify realized entries use the original slots.
        let realize_by_path: std::collections::HashMap<u64, u32> =
            diff2.realize.iter().map(|(s, c)| (c.id.path, *s)).collect();
        assert_eq!(realize_by_path[&0], slot0, "re-realized chunk 0 keeps its slot");
        assert_eq!(realize_by_path[&1], slot1, "re-realized chunk 1 keeps its slot");
        assert_eq!(realize_by_path[&2], slot2, "re-realized chunk 2 keeps its slot");

        // Frame 3: same cut again — now a clean cache hit (no re-realize).
        let diff3 = cache.update(&cut);
        assert!(diff3.realize.is_empty(), "third update must be a clean cache hit");
        assert!(diff3.evicted.is_empty());
        assert_eq!(diff3.visible_slots[0], slot0, "slot stability after re-hit");
    }

    // ----- evict_offscreen (param edit: stale data must never redraw) --------

    /// A param edit evicts the residents that are NOT currently drawn (their
    /// surfaces are stale and a revisit would draw them), keeps the drawn set,
    /// and returns freed slots to the free list for reuse.
    #[test]
    fn evict_offscreen_removes_only_undrawn_residents() {
        let mut cache = ChunkCache::new(64);
        // Frame 1: A B C drawn. Frame 2: only A drawn — B, C now off-screen.
        let (a, b, c) = (chunk(0, 1, 0), chunk(0, 1, 1), chunk(0, 1, 2));
        cache.update(&[a, b, c]);
        cache.update(&[a]);
        assert_eq!(cache.resident_count(), 3, "sanity: B and C linger in the LRU");

        let evicted = cache.evict_offscreen();
        assert_eq!(evicted.len(), 2, "B and C evicted");
        assert!(cache.slot_of(a.id).is_some(), "drawn chunk A kept");
        assert!(cache.slot_of(b.id).is_none(), "off-screen B gone");
        assert!(cache.slot_of(c.id).is_none(), "off-screen C gone");

        // A revisit is a plain miss: B re-realizes like a first visit, into a
        // reclaimed slot, and A is a clean hit.
        let d = cache.update(&[a, b]);
        assert_eq!(d.realize.len(), 1, "revisited B re-realizes from scratch");
        assert_eq!(d.realize[0].1.id, b.id);
        assert!(evicted.contains(&d.realize[0].0), "freed slot reused");
    }

    /// A stand-in ancestor covering not-yet-admitted cut chunks is part of the
    /// drawn set (touched by pass 1.5) and must survive `evict_offscreen` — or
    /// the edit would punch a hole in the ground it is covering.
    #[test]
    fn evict_offscreen_keeps_protected_standin_ancestors() {
        let mut cache = ChunkCache::new(64);
        let parent = chunk(0, 3, 0b101);
        cache.update(&[parent]);
        // Cut refines to the children, but their bakes aren't ready: the
        // parent is not in the cut yet stands in for all four.
        let kids: Vec<Chunk> = (0..4).map(|k| chunk(0, 4, (0b101 << 2) | k)).collect();
        let d = cache.update_throttled_gated(&kids, 8, &|_| false);
        assert!(d.realize.is_empty(), "sanity: children gated out");
        let parent_slot = cache.slot_of(parent.id).unwrap();
        assert!(d.visible_slots.contains(&parent_slot), "sanity: parent covers");

        assert!(cache.evict_offscreen().is_empty(), "covering stand-in kept");
        assert!(cache.slot_of(parent.id).is_some());
    }

    // ----- mark_dirty (targeted re-bake) -------------------------------------

    /// `mark_dirty` on a resident chunk re-realizes ONLY that chunk in place on the
    /// next update (its slot unchanged); a non-resident id is a no-op.
    #[test]
    fn mark_dirty_rerealizes_only_that_chunk() {
        let mut cache = ChunkCache::new(64);
        let cut = vec![chunk(0, 1, 0), chunk(0, 1, 1), chunk(0, 1, 2)];
        let diff1 = cache.update(&cut);
        let slot1 = diff1.visible_slots[1];

        // No-op for a chunk that isn't resident.
        cache.mark_dirty(ChunkId { face: 9, depth: 9, path: 99 });
        // Target the middle chunk only.
        cache.mark_dirty(cut[1].id);

        let diff2 = cache.update(&cut);
        assert_eq!(diff2.realize.len(), 1, "only the marked chunk re-realizes");
        assert_eq!(diff2.realize[0].0, slot1, "it keeps its slot");
        assert_eq!(diff2.realize[0].1.id, cut[1].id, "and it's the one we marked");
        assert!(diff2.evicted.is_empty());

        // Dirty flag is one-shot: a third identical update is a clean hit.
        let diff3 = cache.update(&cut);
        assert!(diff3.realize.is_empty(), "mark_dirty clears after re-realize");
    }

    // ----- partial cut move: slot stability ----------------------------------

    /// Two successive cuts that share most ChunkIds.
    ///
    /// Cut 1: A B C D  (four chunks)
    /// Cut 2: A B C E  (keeps A/B/C, drops D, adds E)
    ///
    /// After update 2:
    ///   - Only E appears in `realize` (one new id).
    ///   - A/B/C keep the exact same slot they had after update 1.
    ///   - D is still resident (eviction is Task 3); count grows by exactly 1.
    #[test]
    fn partial_move_slots_are_stable() {
        let mut cache = ChunkCache::new(64);

        // Cut 1: A=path0, B=path1, C=path2, D=path3
        let cut1 = vec![
            chunk(0, 1, 0), // A
            chunk(0, 1, 1), // B
            chunk(0, 1, 2), // C
            chunk(0, 1, 3), // D
        ];
        let diff1 = cache.update(&cut1);
        assert_eq!(diff1.realize.len(), 4, "all 4 chunks realized on first frame");

        // Record slot assignments for A, B, C from diff1's visible_slots.
        let slot_a = diff1.visible_slots[0];
        let slot_b = diff1.visible_slots[1];
        let slot_c = diff1.visible_slots[2];

        // Cut 2: A B C E — D dropped, E added.
        let cut2 = vec![
            chunk(0, 1, 0), // A — still present
            chunk(0, 1, 1), // B — still present
            chunk(0, 1, 2), // C — still present
            chunk(0, 1, 4), // E — new
        ];
        let diff2 = cache.update(&cut2);

        // Only E should be realized (A/B/C are cache hits).
        assert_eq!(diff2.realize.len(), 1, "only the new chunk E must be realized");
        assert_eq!(diff2.realize[0].1.id.path, 4, "realized chunk is E (path=4)");

        // No evictions yet (Task 3).
        assert!(diff2.evicted.is_empty(), "no evictions before Task 3");

        // A/B/C keep the same slots as after cut 1.
        assert_eq!(diff2.visible_slots[0], slot_a, "A must keep its slot");
        assert_eq!(diff2.visible_slots[1], slot_b, "B must keep its slot");
        assert_eq!(diff2.visible_slots[2], slot_c, "C must keep its slot");

        // D was not evicted, so resident_count grows by 1 (adds E, keeps D).
        assert_eq!(
            cache.resident_count(),
            5,
            "A+B+C+D+E all resident; D left the cut but was not evicted"
        );
    }
}
