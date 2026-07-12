//! Pure-CPU reference for the v5 scatter layer (CEL-73).
//!
//! Candidate positions live on a **stable world lattice**: the level-`L`
//! triangular cells of each icosphere face's quadtree. A candidate is keyed by
//! `(face, level-L cell path, k, seed)` only — never by the hosting chunk's
//! depth, slot, or triangle index — so an object never moves when the chunk
//! quadtree splits or merges (the CEL-69 lesson).
//!
//! This module is the layout/math contract for `shaders/ScatterPlace.slang`:
//! the shader mirrors `wang`/`seed0`/`candidate`/`descend_bary` bit for bit
//! (u32 wrapping ops, same split order as [`quadtree::split_bary`]). The
//! invariant tests here are the CPU proof that placement is LOD-stable.

use crate::quadtree::{split_bary, Bary, ChunkId};

/// Max subdivision span a chunk may host: a chunk at depth `d` hosts lattice
/// cells only when `L - d <= S_MAX`, bounding the per-slot pool region to
/// `4^S_MAX` cells. Shallower chunks host nothing (they are far away, below
/// any sensible `min_lod`).
pub const S_MAX: u32 = 3;

/// Fixed per-slot candidate capacity for a layer: `k_per_cell * 4^S_MAX`.
pub fn capacity(k_per_cell: u32) -> u32 {
    k_per_cell * (1 << (2 * S_MAX))
}

/// Wang hash — must match `ScatterPlace.slang` exactly.
pub fn wang(mut x: u32) -> u32 {
    x = (x ^ 61) ^ (x >> 16);
    x = x.wrapping_mul(9);
    x ^= x >> 4;
    x = x.wrapping_mul(0x27d4_eb2d);
    x ^= x >> 15;
    x
}

/// Map a hashed word to `[0, 1)`.
pub fn u01(x: u32) -> f32 {
    wang(x) as f32 / 4_294_967_296.0
}

/// Per-candidate base seed from the STABLE lattice key `(face, cell path, k, seed)`.
pub fn seed0(face: u8, cell_path: u64, k: u32, seed: u32) -> u32 {
    let lo = cell_path as u32;
    let hi = (cell_path >> 32) as u32;
    wang(wang(wang(face as u32) ^ lo) ^ hi) ^ wang(k ^ seed.wrapping_mul(0x9e37_79b9))
}

/// One candidate in FACE-barycentric space plus its stable per-instance values.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CandidateBary {
    pub wb: f32,
    pub wc: f32,
    /// Uniform keep-value in `[0, 1)`: drawn when `hash01 < density`.
    pub hash01: f32,
    /// Yaw around the radial up-axis, radians in `[0, TAU)`.
    pub yaw: f32,
    /// Uniform scale jitter in `[0.8, 1.2)`.
    pub scale: f32,
}

/// The candidate of lattice cell `cell_path` (corners `cell`, face-bary) index
/// `k`. Uniform in the cell triangle via the sqrt trick.
pub fn candidate(face: u8, cell_path: u64, cell: [Bary; 3], k: u32, seed: u32) -> CandidateBary {
    let s0 = seed0(face, cell_path, k, seed);
    let u1 = u01(s0 ^ 1);
    let u2 = u01(s0 ^ 2);
    let su = u1.sqrt();
    let (w0, w1, w2) = (1.0 - su, su * (1.0 - u2), su * u2);
    CandidateBary {
        wb: cell[0].wb * w0 + cell[1].wb * w1 + cell[2].wb * w2,
        wc: cell[0].wc * w0 + cell[1].wc * w1 + cell[2].wc * w2,
        hash01: u01(s0 ^ 0x68bc_21eb),
        yaw: u01(s0 ^ 3) * std::f32::consts::TAU,
        scale: 0.8 + 0.4 * u01(s0 ^ 4),
    }
}

/// Descend `levels` split steps from `root`, consuming the path bits
/// **root-first**: level `i`'s child index is `(path >> 2*(levels-1-i)) & 3`.
/// Split order matches [`split_bary`] (children 0/1/2 hug the corners, 3 is the
/// inverted centre) so the resulting triangles are bit-identical to the
/// quadtree's own descent.
pub fn descend_bary(root: [Bary; 3], path: u64, levels: u32) -> [Bary; 3] {
    let mut b = root;
    for i in 0..levels {
        let k = ((path >> (2 * (levels - 1 - i))) & 3) as usize;
        b = split_bary(b)[k];
    }
    b
}

/// Which lattice cells a chunk hosts.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum CellRange {
    /// Too shallow (`L - depth > S_MAX`): hosts nothing.
    None,
    /// `depth <= L`: the chunk contains `count = 4^(L-depth)` whole cells; cell
    /// `sub`'s full path is `(chunk.path << 2*(L-depth)) | sub`.
    Subcells { count: u32 },
    /// `depth > L`: the chunk lies inside one cell (its level-`L` ancestor);
    /// it hosts the subset of that cell's candidates that fall inside it.
    Ancestor { path: u64 },
}

/// Classify a chunk against lattice level `lattice_level`.
pub fn cell_range(id: ChunkId, lattice_level: u8) -> CellRange {
    let d = id.depth as i32;
    let l = lattice_level as i32;
    if d <= l {
        let s = (l - d) as u32;
        if s > S_MAX {
            CellRange::None
        } else {
            CellRange::Subcells { count: 1 << (2 * s) }
        }
    } else {
        CellRange::Ancestor { path: id.path >> (2 * (d - l) as u32) }
    }
}

/// Point-in-triangle in face-bary 2D space (`wb`, `wc` as x, y), boundary
/// inclusive. The shader uses the same rule.
pub fn bary_point_in_triangle(p: (f32, f32), tri: [Bary; 3]) -> bool {
    let (v0x, v0y) = (tri[1].wb - tri[0].wb, tri[1].wc - tri[0].wc);
    let (v1x, v1y) = (tri[2].wb - tri[0].wb, tri[2].wc - tri[0].wc);
    let (v2x, v2y) = (p.0 - tri[0].wb, p.1 - tri[0].wc);
    let den = v0x * v1y - v1x * v0y;
    if den.abs() < 1.0e-12 {
        return false;
    }
    let u = (v2x * v1y - v1x * v2y) / den;
    let v = (v0x * v2y - v2x * v0y) / den;
    u >= 0.0 && v >= 0.0 && u + v <= 1.0
}

/// All candidates a chunk (id + face-bary corners) hosts for a layer.
///
/// This is the CPU mirror of one `ScatterPlace` slot region (minus the world
/// projection/displacement): the same `(cell_path, k)` enumeration, jitter and
/// in-chunk filter the shader applies.
pub fn candidates_for_chunk(
    id: ChunkId,
    bary: [Bary; 3],
    k_per_cell: u32,
    lattice_level: u8,
    seed: u32,
) -> Vec<CandidateBary> {
    let mut out = Vec::new();
    match cell_range(id, lattice_level) {
        CellRange::None => {}
        CellRange::Subcells { count } => {
            let levels = (lattice_level - id.depth) as u32;
            for sub in 0..count as u64 {
                let cell = descend_bary(bary, sub, levels);
                let full_path = (id.path << (2 * levels)) | sub;
                for k in 0..k_per_cell {
                    out.push(candidate(id.face, full_path, cell, k, seed));
                }
            }
        }
        CellRange::Ancestor { path } => {
            // The cell triangle: descend the FACE root by the ancestor path.
            let root = [
                Bary { wb: 0.0, wc: 0.0 },
                Bary { wb: 1.0, wc: 0.0 },
                Bary { wb: 0.0, wc: 1.0 },
            ];
            let cell = descend_bary(root, path, lattice_level as u32);
            for k in 0..k_per_cell {
                let c = candidate(id.face, path, cell, k, seed);
                if bary_point_in_triangle((c.wb, c.wc), bary) {
                    out.push(c);
                }
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    const K: u32 = 4;
    const SEED: u32 = 7;

    fn face_root() -> [Bary; 3] {
        [Bary { wb: 0.0, wc: 0.0 }, Bary { wb: 1.0, wc: 0.0 }, Bary { wb: 0.0, wc: 1.0 }]
    }

    /// Children of (id, bary) in split order, path extended low.
    fn children(id: ChunkId, bary: [Bary; 3]) -> [(ChunkId, [Bary; 3]); 4] {
        let kids = split_bary(bary);
        std::array::from_fn(|k| {
            (
                ChunkId { face: id.face, depth: id.depth + 1, path: (id.path << 2) | k as u64 },
                kids[k],
            )
        })
    }

    fn key(c: &CandidateBary) -> (i64, i64, i64) {
        // Quantized identity for set comparison (1e-6 tolerance).
        ((c.wb * 1.0e6).round() as i64, (c.wc * 1.0e6).round() as i64, (c.hash01 * 1.0e6).round() as i64)
    }

    #[test]
    fn capacity_formula() {
        assert_eq!(capacity(1), 64);
        assert_eq!(capacity(4), 256);
    }

    #[test]
    fn shallow_chunk_hosts_nothing() {
        let id = ChunkId { face: 2, depth: 3, path: 0b10_01_11 };
        assert_eq!(cell_range(id, 7), CellRange::None); // L - d = 4 > S_MAX
        assert!(candidates_for_chunk(id, face_root(), K, 7, SEED).is_empty());
        // Exactly at the S_MAX boundary it hosts the full 4^3 cells.
        assert_eq!(cell_range(id, 6), CellRange::Subcells { count: 64 });
    }

    #[test]
    fn candidate_count_matches_cells() {
        let id = ChunkId { face: 0, depth: 5, path: 0b11_00_10_01_11 };
        // d == L: one cell, K candidates.
        assert_eq!(candidates_for_chunk(id, face_root(), K, 5, SEED).len(), K as usize);
        // L = d + 2: 16 cells.
        assert_eq!(candidates_for_chunk(id, face_root(), K, 7, SEED).len(), (16 * K) as usize);
    }

    #[test]
    fn candidates_lie_inside_the_chunk() {
        let id = ChunkId { face: 4, depth: 2, path: 0b01_10 };
        let bary = descend_bary(face_root(), id.path, 2);
        for c in candidates_for_chunk(id, bary, K, 4, SEED) {
            assert!(
                bary_point_in_triangle((c.wb, c.wc), bary),
                "candidate ({}, {}) escaped its chunk",
                c.wb,
                c.wc
            );
        }
    }

    /// The load-bearing invariant: a chunk's candidate set equals the disjoint
    /// union of its 4 children's sets — splitting a chunk never moves, drops,
    /// or duplicates an instance.
    #[test]
    fn stability_parent_equals_children_union() {
        let l = 6u8;
        let parent_id = ChunkId { face: 9, depth: 3, path: 0b10_11_00 };
        let parent_bary = descend_bary(face_root(), parent_id.path, 3);

        let mut parent: Vec<_> = candidates_for_chunk(parent_id, parent_bary, K, l, SEED)
            .iter()
            .map(key)
            .collect();
        let mut kids: Vec<_> = children(parent_id, parent_bary)
            .iter()
            .flat_map(|(id, b)| candidates_for_chunk(*id, *b, K, l, SEED))
            .map(|c| key(&c))
            .collect();

        parent.sort_unstable();
        kids.sort_unstable();
        assert_eq!(parent.len(), (64 * K) as usize); // L-d = 3 → 4^3 cells
        assert_eq!(parent, kids, "children must partition the parent's candidates exactly");
    }

    /// Below the lattice level the descendants of one cell partition that
    /// cell's K candidates: no duplicates, none lost.
    #[test]
    fn deep_chunks_partition_cell() {
        let l = 4u8;
        // The cell IS a chunk at depth L.
        let cell_id = ChunkId { face: 1, depth: l, path: 0b01_11_00_10 };
        let cell_bary = descend_bary(face_root(), cell_id.path, l as u32);
        let mut cell: Vec<_> =
            candidates_for_chunk(cell_id, cell_bary, K, l, SEED).iter().map(key).collect();

        // All 16 grandchildren at depth L+2.
        let mut deep: Vec<_> = children(cell_id, cell_bary)
            .iter()
            .flat_map(|(cid, cb)| children(*cid, *cb))
            .flat_map(|(gid, gb)| candidates_for_chunk(gid, gb, K, l, SEED))
            .map(|c| key(&c))
            .collect();

        cell.sort_unstable();
        deep.sort_unstable();
        assert_eq!(cell.len(), K as usize);
        assert_eq!(cell, deep, "descendants must partition the cell's candidates exactly");
    }

    /// Candidate jitter stays inside its own cell triangle.
    #[test]
    fn candidate_in_cell_triangle() {
        let cell_path = 0b10_01_11_00_01u64;
        let cell = descend_bary(face_root(), cell_path, 5);
        for k in 0..K {
            let c = candidate(3, cell_path, cell, k, SEED);
            assert!(bary_point_in_triangle((c.wb, c.wc), cell));
            assert!((0.0..1.0).contains(&c.hash01));
            assert!((0.8..1.2).contains(&c.scale));
            assert!((0.0..std::f32::consts::TAU).contains(&c.yaw));
        }
    }
}
