// SurfaceCache — bounded, insertion-ordered (FIFO) cache of per-chunk payloads.
//
// Replaces the unbounded `HashMap<ChunkId, ChunkSurface>` that grew to the whole
// quadtree cut (1000-3000 chunks × ~0.75 MiB at tile_res 256 = multiple GB).
// Generic over the payload so this pure-CPU crate never sees the GPU/Godot types.

use std::collections::{HashMap, VecDeque};

use crate::quadtree::ChunkId;

/// A bounded map from `ChunkId` to a payload, evicting in **insertion order**
/// (oldest first) once `capacity` is reached.
///
/// FIFO rather than LRU on purpose: surfaces are consumed shortly after they are
/// produced, so "oldest inserted" is the right victim and the bookkeeping stays
/// O(1) amortized without touching the order on every read.
///
/// The `VecDeque` of ids is kept **exactly** in sync with the map on every path
/// (`insert` / `remove` / `retain` / `clear` / `set_capacity`) — no tombstones —
/// so the eviction victim is always a genuinely-present entry.
pub struct SurfaceCache<T> {
    map: HashMap<ChunkId, T>,
    /// Insertion order, oldest at the front. One entry per key in `map`.
    order: VecDeque<ChunkId>,
    capacity: usize,
}

impl<T> SurfaceCache<T> {
    /// A cache holding at most `capacity` entries (clamped to at least 1 — a
    /// zero-capacity cache would drop every insert and starve the caller).
    pub fn new(capacity: usize) -> Self {
        Self {
            map: HashMap::new(),
            order: VecDeque::new(),
            capacity: capacity.max(1),
        }
    }

    /// Re-cap when the planet's budget changes; evicts down to the new capacity
    /// immediately (oldest first).
    pub fn set_capacity(&mut self, capacity: usize) {
        self.capacity = capacity.max(1);
        while self.map.len() > self.capacity {
            self.evict_oldest();
        }
    }

    /// The current maximum number of entries (always >= 1).
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Insert `value` under `id`, evicting the OLDEST entry if the cache is full.
    /// Returns the evicted value, if any.
    ///
    /// Re-inserting an existing `id` **replaces** its value in place (keeping its
    /// original position in the eviction order) and never evicts anything.
    pub fn insert(&mut self, id: ChunkId, value: T) -> Option<T> {
        if let Some(slot) = self.map.get_mut(&id) {
            *slot = value; // replace in place — order untouched, nothing evicted
            return None;
        }
        let evicted = if self.map.len() >= self.capacity {
            self.evict_oldest()
        } else {
            None
        };
        self.order.push_back(id);
        self.map.insert(id, value);
        evicted
    }

    /// Remove `id`, freeing a slot for reuse. Returns its value, if present.
    pub fn remove(&mut self, id: &ChunkId) -> Option<T> {
        let value = self.map.remove(id)?;
        if let Some(pos) = self.order.iter().position(|k| k == id) {
            self.order.remove(pos);
        }
        Some(value)
    }

    pub fn contains(&self, id: &ChunkId) -> bool {
        self.map.contains_key(id)
    }

    pub fn get(&self, id: &ChunkId) -> Option<&T> {
        self.map.get(id)
    }

    /// Keep only the entries for which `f` returns true (e.g. "still in the cut").
    /// The eviction order is filtered to match, so no dropped id lingers as a
    /// ghost that would make a later `insert` evict nothing.
    pub fn retain(&mut self, mut f: impl FnMut(&ChunkId, &mut T) -> bool) {
        let map = &mut self.map;
        map.retain(|id, v| f(id, v));
        self.order.retain(|id| map.contains_key(id));
    }

    pub fn len(&self) -> usize {
        self.map.len()
    }

    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    pub fn clear(&mut self) {
        self.map.clear();
        self.order.clear();
    }

    /// Pop the oldest inserted entry. `order` is exact (no tombstones), so the
    /// front id is always present in the map.
    fn evict_oldest(&mut self) -> Option<T> {
        let victim = self.order.pop_front()?;
        self.map.remove(&victim)
    }
}

// ---------------------------------------------------------------------------
// Tests (written first — TDD)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quadtree::ChunkId;

    /// Minimal test helper: the cache only keys on `ChunkId`.
    fn id(path: u64) -> ChunkId {
        ChunkId { face: 0, depth: 3, path }
    }

    #[test]
    fn insert_beyond_capacity_evicts_the_oldest() {
        let mut c: SurfaceCache<&str> = SurfaceCache::new(2);
        assert_eq!(c.insert(id(1), "one"), None);
        assert_eq!(c.insert(id(2), "two"), None);
        assert_eq!(c.insert(id(3), "three"), Some("one"), "oldest evicted");
        assert_eq!(c.len(), 2);
        assert!(!c.contains(&id(1)));
        assert!(c.contains(&id(2)));
        assert!(c.contains(&id(3)));
    }

    #[test]
    fn len_never_exceeds_capacity_under_flood() {
        let mut c: SurfaceCache<u64> = SurfaceCache::new(8);
        for p in 0..1000u64 {
            c.insert(id(p), p);
            assert!(c.len() <= 8, "len {} exceeded capacity at {p}", c.len());
        }
        assert_eq!(c.len(), 8);
    }

    #[test]
    fn shrinking_capacity_evicts_down_to_it() {
        let mut c: SurfaceCache<u64> = SurfaceCache::new(10);
        for p in 0..10u64 {
            c.insert(id(p), p);
        }
        assert_eq!(c.len(), 10);
        c.set_capacity(3);
        assert_eq!(c.capacity(), 3);
        assert_eq!(c.len(), 3);
        // The three NEWEST survive.
        assert!(c.contains(&id(7)) && c.contains(&id(8)) && c.contains(&id(9)));
    }

    #[test]
    fn remove_frees_a_slot_for_reuse() {
        let mut c: SurfaceCache<&str> = SurfaceCache::new(1);
        assert_eq!(c.insert(id(1), "one"), None);
        assert_eq!(c.remove(&id(1)), Some("one"));
        assert!(c.is_empty());
        assert_eq!(c.insert(id(2), "two"), None, "removal freed the slot");
        assert_eq!(c.len(), 1);
        assert_eq!(c.get(&id(2)), Some(&"two"));
    }

    #[test]
    fn reinserting_an_existing_id_replaces_without_evicting() {
        let mut c: SurfaceCache<&str> = SurfaceCache::new(2);
        c.insert(id(1), "one");
        c.insert(id(2), "two");
        assert_eq!(c.insert(id(1), "one-v2"), None, "replace must not evict");
        assert_eq!(c.len(), 2);
        assert_eq!(c.get(&id(1)), Some(&"one-v2"));
        assert!(c.contains(&id(2)));
    }

    #[test]
    fn retain_keeps_the_deque_in_sync() {
        let mut c: SurfaceCache<u64> = SurfaceCache::new(4);
        for p in 0..4u64 {
            c.insert(id(p), p);
        }
        // Drop the two oldest (0, 1); keep 2 and 3.
        c.retain(|k, _| k.path >= 2);
        assert_eq!(c.len(), 2);
        assert!(c.contains(&id(2)) && c.contains(&id(3)));

        // Two fresh inserts fit without eviction (retain freed real capacity).
        assert_eq!(c.insert(id(4), 4), None);
        assert_eq!(c.insert(id(5), 5), None);
        assert_eq!(c.len(), 4);

        // Now full: the next insert evicts the OLDEST SURVIVING entry (2), not
        // a stale ghost of 0/1 left behind in the deque.
        assert_eq!(c.insert(id(6), 6), Some(2), "oldest survivor evicted");
        assert_eq!(c.len(), 4);
        assert!(!c.contains(&id(2)));
        assert!(c.contains(&id(3)) && c.contains(&id(4)) && c.contains(&id(5)));
        assert!(c.contains(&id(6)));
    }

    #[test]
    fn zero_capacity_is_clamped_to_one() {
        let mut c: SurfaceCache<u64> = SurfaceCache::new(0);
        assert_eq!(c.capacity(), 1, "a zero-capacity cache would starve the caller");
        assert_eq!(c.insert(id(1), 1), None);
        assert_eq!(c.len(), 1);
        c.set_capacity(0);
        assert_eq!(c.capacity(), 1);
        assert_eq!(c.len(), 1);
    }

    #[test]
    fn clear_empties_both_maps() {
        let mut c: SurfaceCache<u64> = SurfaceCache::new(4);
        for p in 0..4u64 {
            c.insert(id(p), p);
        }
        c.clear();
        assert!(c.is_empty());
        assert_eq!(c.len(), 0);
        // No ghost order entries: 4 fresh inserts must not evict anything.
        for p in 10..14u64 {
            assert_eq!(c.insert(id(p), p), None);
        }
        assert_eq!(c.len(), 4);
    }
}
