//! The async GDScript bake contract (CEL-86): pure helpers + the submission
//! queue a [`crate::builder::CesBuilder`] hands results back through.
//!
//! A GDScript builder that defines `_bake_requested(requests)` is baked
//! ASYNCHRONOUSLY: the planet hands it a batch of chunks each frame and the
//! builder submits surfaces back — from any thread, at any time, any number of
//! times per chunk — via `submit_chunk`. Submissions land in [`SubmitQueue`],
//! which the planet drains on the main thread into the SAME `ready_surfaces`
//! map the Rust [`crate::bake_pool::BakePool`] fills. Admission gating, coarse
//! ancestor stand-ins and per-frame throttling therefore apply unchanged — they
//! never knew where a surface came from.
//!
//! Re-submitting an already-resident chunk marks it dirty and re-realizes it,
//! which is exactly the streaming refinement path (a coarse tile now, a finer
//! one when the download lands) with no extra API.
//!
//! Everything here is pure and unit-tested; the Godot glue lives in
//! `builder.rs` (the `#[func]`s) and `celestial.rs` (drain / request /
//! gate).

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;

use godot::builtin::{Color, Vector3};

use celestial_algo::quadtree::ChunkId;

/// Deepest chunk depth a handle can encode. `ChunkId::path` spends 2 bits per
/// level, so `depth <= 20` fits the 40 path bits below.
pub const MAX_HANDLE_DEPTH: u8 = 20;

/// Widths of the handle bit-fields, low to high: path | depth | face | tag.
const PATH_BITS: u32 = 40;
const DEPTH_BITS: u32 = 5;
const FACE_BITS: u32 = 5;
/// Per-planet tag, so two planets sharing one builder `.tres` cannot have their
/// submissions cross-wired: a planet ignores any handle not bearing its tag.
pub const TAG_BITS: u32 = 12;

const DEPTH_SHIFT: u32 = PATH_BITS;
const FACE_SHIFT: u32 = DEPTH_SHIFT + DEPTH_BITS;
const TAG_SHIFT: u32 = FACE_SHIFT + FACE_BITS;

const PATH_MASK: u64 = (1 << PATH_BITS) - 1;
const DEPTH_MASK: u64 = (1 << DEPTH_BITS) - 1;
const FACE_MASK: u64 = (1 << FACE_BITS) - 1;
/// Mask a raw planet tag down to the bits a handle can carry.
pub const TAG_MASK: u64 = (1 << TAG_BITS) - 1;

/// Pack `(tag, id)` into the opaque `i64` handle GDScript sees. Returns `None`
/// for a chunk too deep to encode (`depth > MAX_HANDLE_DEPTH`) — such a chunk is
/// simply never handed to an async builder.
///
/// Total width is `40 + 5 + 5 + 12 = 62` bits, so the result is always a
/// non-negative `i64` (GDScript ints are signed).
pub fn handle_encode(tag: u16, id: ChunkId) -> Option<i64> {
    if id.depth > MAX_HANDLE_DEPTH || id.face as u64 > FACE_MASK {
        return None;
    }
    if id.path & !PATH_MASK != 0 {
        return None;
    }
    let bits = (id.path & PATH_MASK)
        | ((id.depth as u64 & DEPTH_MASK) << DEPTH_SHIFT)
        | ((id.face as u64 & FACE_MASK) << FACE_SHIFT)
        | ((tag as u64 & TAG_MASK) << TAG_SHIFT);
    Some(bits as i64)
}

/// Unpack a handle, rejecting anything not stamped with `tag` (a submission
/// meant for a different planet) and anything negative or over-wide.
pub fn handle_decode(tag: u16, handle: i64) -> Option<ChunkId> {
    if handle < 0 {
        return None;
    }
    let bits = handle as u64;
    if (bits >> TAG_SHIFT) & TAG_MASK != (tag as u64 & TAG_MASK) {
        return None;
    }
    let depth = ((bits >> DEPTH_SHIFT) & DEPTH_MASK) as u8;
    if depth > MAX_HANDLE_DEPTH {
        return None;
    }
    Some(ChunkId {
        face: ((bits >> FACE_SHIFT) & FACE_MASK) as u8,
        depth,
        path: bits & PATH_MASK,
    })
}

/// The texel-centre world directions of a chunk, row-major, `tile_res²` of them.
///
/// The same mapping the shaders use: barycentric texel centres over the chunk
/// triangle, folded across the diagonal (`u + v > 1`) so the whole square holds
/// valid data, then gnomonically projected onto the unit sphere by normalizing.
/// Pure — safe to call from a worker thread.
pub fn chunk_dirs(corners: [Vector3; 3], tile_res: u32) -> Vec<Vector3> {
    let n = tile_res as usize;
    let res_f = tile_res as f32;
    let (d0, d1, d2) =
        (corners[0].normalized(), corners[1].normalized(), corners[2].normalized());
    let mut dirs = Vec::with_capacity(n * n);
    for ty in 0..n {
        for tx in 0..n {
            let mut u = (tx as f32 + 0.5) / res_f;
            let mut v = (ty as f32 + 0.5) / res_f;
            if u + v > 1.0 {
                let s = u + v;
                u /= s;
                v /= s;
            }
            dirs.push((d0 * (1.0 - u - v) + d1 * u + d2 * v).normalized());
        }
    }
    dirs
}

/// Pack a signed unit component into the rgba8 normal encoding.
fn pack_component(c: f32) -> u8 {
    ((c * 0.5 + 0.5).clamp(0.0, 1.0) * 255.0).round() as u8
}

/// Pack a chunk's rgba8 world normals: each texel uses `over[i]` when the
/// builder supplied one, else a curvature-correct finite difference of the
/// height grid — the same construction `NoiseProvider` uses (the chunk's edge
/// vectors are projected perpendicular to the radial direction before the height
/// gradient is added along it).
///
/// `corners` are the chunk's UNNORMALIZED corners (their length carries the
/// world radius scale); `dirs` is [`chunk_dirs`] for the same chunk.
pub fn pack_normals(
    corners: [Vector3; 3],
    dirs: &[Vector3],
    height: &[f32],
    tile_res: u32,
    over: Option<&[Vector3]>,
) -> Vec<u8> {
    let n = tile_res as usize;
    let res_f = tile_res as f32;
    let ni = n as i32;
    let k = corners[0].length().max(1.0); // world radius scale
    let ex = (corners[1] - corners[0]) / res_f;
    let ey = (corners[2] - corners[0]) / res_f;

    let mut normal = vec![0u8; n * n * 4];
    for ty in 0..n {
        for tx in 0..n {
            let idx = ty * n + tx;
            let dir = dirs[idx];
            let nrm = if let Some(o) = over.and_then(|s| s.get(idx)) {
                o.normalized()
            } else {
                let sample = |x: i32, y: i32| -> f32 {
                    let x = x.clamp(0, ni - 1) as usize;
                    let y = y.clamp(0, ni - 1) as usize;
                    height[y * n + x]
                };
                let (xi, yi) = (tx as i32, ty as i32);
                let exl = ex - dir * dir.dot(ex);
                let eyl = ey - dir * dir.dot(ey);
                let dpx = exl * 2.0 + dir * (k * (sample(xi + 1, yi) - sample(xi - 1, yi)));
                let dpy = eyl * 2.0 + dir * (k * (sample(xi, yi + 1) - sample(xi, yi - 1)));
                let mut nrm = dpx.cross(dpy).normalized();
                if nrm.dot(dir) < 0.0 {
                    nrm = -nrm;
                }
                nrm
            };
            normal[idx * 4] = pack_component(nrm.x);
            normal[idx * 4 + 1] = pack_component(nrm.y);
            normal[idx * 4 + 2] = pack_component(nrm.z);
            normal[idx * 4 + 3] = 255;
        }
    }
    normal
}

/// [`pack_normals`] with no builder-supplied normals: pure finite difference.
pub fn fd_normals(
    corners: [Vector3; 3],
    dirs: &[Vector3],
    height: &[f32],
    tile_res: u32,
) -> Vec<u8> {
    pack_normals(corners, dirs, height, tile_res, None)
}

/// Why a `submit_chunk` payload was rejected.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum SubmitError {
    /// `heights.len() != tile_res²`.
    HeightLen { got: usize, want: usize },
    /// `colors.len() != tile_res²`.
    ColorLen { got: usize, want: usize },
    /// `normals` was non-empty but not `tile_res²` long.
    NormalLen { got: usize, want: usize },
}

impl std::fmt::Display for SubmitError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SubmitError::HeightLen { got, want } => {
                write!(f, "heights has {got} entries, expected tile_res² = {want}")
            }
            SubmitError::ColorLen { got, want } => {
                write!(f, "colors has {got} entries, expected tile_res² = {want}")
            }
            SubmitError::NormalLen { got, want } => {
                write!(f, "normals has {got} entries, expected 0 or tile_res² = {want}")
            }
        }
    }
}

/// A raw `submit_chunk` payload as GDScript handed it over. Held as plain
/// `Vec`s (not `PackedArray`s, which are not `Send`) so the queue can cross
/// threads. Validated at drain time, where `tile_res` is known.
#[derive(Clone, Debug)]
pub struct RawSubmission {
    pub handle: i64,
    pub heights: Vec<f32>,
    pub colors: Vec<Color>,
    pub normals: Vec<Vector3>,
}

/// A validated surface. `normal == None` ⇒ finite-difference it (the planet has
/// the chunk geometry; the builder may not).
#[derive(Clone, Debug, PartialEq)]
pub struct Submission {
    pub handle: i64,
    pub color: Vec<u8>,
    pub height: Vec<f32>,
    pub normal: Option<Vec<u8>>,
}

/// Validate and convert a raw `submit_chunk` payload.
///
/// Non-finite heights (a NaN from a bad division, an inf from a failed fetch)
/// are sanitized to `0.0`: a NaN reaching LOD selection poisons it in a way that
/// is near-impossible to trace back from the symptom.
///
/// An EMPTY `normals` means "finite-difference it for me" and yields
/// `normal: None`; any other length mismatch is an error.
pub fn validate_submission(
    raw: &RawSubmission,
    tile_res: u32,
) -> Result<Submission, SubmitError> {
    let (heights, colors, normals) = (&raw.heights, &raw.colors, &raw.normals);
    let want = (tile_res as usize) * (tile_res as usize);
    if heights.len() != want {
        return Err(SubmitError::HeightLen { got: heights.len(), want });
    }
    if colors.len() != want {
        return Err(SubmitError::ColorLen { got: colors.len(), want });
    }
    if !normals.is_empty() && normals.len() != want {
        return Err(SubmitError::NormalLen { got: normals.len(), want });
    }

    let height: Vec<f32> =
        heights.iter().map(|h| if h.is_finite() { *h } else { 0.0 }).collect();

    let mut color = vec![255u8; want * 4];
    for (i, c) in colors.iter().enumerate() {
        color[i * 4] = (c.r.clamp(0.0, 1.0) * 255.0) as u8;
        color[i * 4 + 1] = (c.g.clamp(0.0, 1.0) * 255.0) as u8;
        color[i * 4 + 2] = (c.b.clamp(0.0, 1.0) * 255.0) as u8;
        color[i * 4 + 3] = 255;
    }

    let normal = if normals.is_empty() {
        None
    } else {
        let mut packed = vec![0u8; want * 4];
        for (i, v) in normals.iter().enumerate() {
            let nrm = v.normalized();
            packed[i * 4] = pack_component(nrm.x);
            packed[i * 4 + 1] = pack_component(nrm.y);
            packed[i * 4 + 2] = pack_component(nrm.z);
            packed[i * 4 + 3] = 255;
        }
        Some(packed)
    };

    Ok(Submission { handle: raw.handle, color, height, normal })
}

/// The mutex-guarded hand-back channel owned by a `CesBuilder`.
///
/// `submit_chunk` pushes from ANY thread; the planet drains on the main thread
/// each frame. The builder owns it, so there is no back-reference to the planet
/// and no reference cycle. A worker that outlives `teardown_job` just pushes
/// into a queue nobody drains — harmless.
#[derive(Default)]
pub struct SubmitQueue {
    items: Mutex<Vec<RawSubmission>>,
    /// A bad payload inside a bake loop would otherwise print thousands of
    /// identical errors per second; report the first and stay quiet after.
    reported_error: AtomicBool,
}

impl SubmitQueue {
    /// Push a raw submission (any thread).
    pub fn push(&self, s: RawSubmission) {
        if let Ok(mut q) = self.items.lock() {
            q.push(s);
        }
    }

    /// Take everything queued (main thread, once per frame).
    pub fn drain(&self) -> Vec<RawSubmission> {
        match self.items.lock() {
            Ok(mut q) => std::mem::take(&mut *q),
            Err(_) => Vec::new(),
        }
    }

    /// Drop every queued submission (a live param edit invalidated them).
    pub fn clear(&self) {
        if let Ok(mut q) = self.items.lock() {
            q.clear();
        }
    }

    /// `true` the FIRST time a payload is rejected, `false` forever after — so
    /// the caller prints one error, not one per frame.
    pub fn should_report_error(&self) -> bool {
        !self.reported_error.swap(true, Ordering::Relaxed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn id(face: u8, depth: u8, path: u64) -> ChunkId {
        ChunkId { face, depth, path }
    }

    #[test]
    fn handle_roundtrips_over_every_face_and_depth() {
        for face in 0..20u8 {
            for depth in 0..=MAX_HANDLE_DEPTH {
                // A full path for this depth: alternating quadrant bits.
                let path = 0xAAAA_AAAA_AAu64 & ((1u64 << (2 * depth as u32)) - 1).max(0);
                let want = id(face, depth, path);
                let h = handle_encode(0x5A5, want).expect("encodable");
                assert!(h >= 0, "handle must be a non-negative GDScript int");
                assert_eq!(handle_decode(0x5A5, h), Some(want));
            }
        }
    }

    #[test]
    fn handle_rejects_a_foreign_planet_tag() {
        let h = handle_encode(7, id(3, 9, 0b1101)).unwrap();
        assert_eq!(handle_decode(7, h), Some(id(3, 9, 0b1101)));
        assert_eq!(handle_decode(8, h), None, "another planet's tag must not decode");
    }

    #[test]
    fn handle_rejects_too_deep_and_negative() {
        assert_eq!(handle_encode(0, id(0, MAX_HANDLE_DEPTH + 1, 0)), None);
        assert_eq!(handle_decode(0, -1), None);
    }

    #[test]
    fn handle_tag_is_masked_not_truncated_into_the_sign_bit() {
        // Widest legal everything: still positive.
        let h = handle_encode(TAG_MASK as u16, id(31, MAX_HANDLE_DEPTH, PATH_MASK)).unwrap();
        assert!(h > 0);
        assert_eq!(handle_decode(TAG_MASK as u16, h).unwrap().face, 31);
    }

    fn raw(handle: i64, heights: &[f32], colors: &[Color], normals: &[Vector3]) -> RawSubmission {
        RawSubmission {
            handle,
            heights: heights.to_vec(),
            colors: colors.to_vec(),
            normals: normals.to_vec(),
        }
    }

    #[test]
    fn validate_rejects_wrong_lengths() {
        let c = vec![Color::from_rgba(1.0, 1.0, 1.0, 1.0); 4];
        assert_eq!(
            validate_submission(&raw(0, &[0.0; 3], &c, &[]), 2),
            Err(SubmitError::HeightLen { got: 3, want: 4 })
        );
        assert_eq!(
            validate_submission(&raw(0, &[0.0; 4], &c[..3], &[]), 2),
            Err(SubmitError::ColorLen { got: 3, want: 4 })
        );
        assert_eq!(
            validate_submission(&raw(0, &[0.0; 4], &c, &[Vector3::UP; 2]), 2),
            Err(SubmitError::NormalLen { got: 2, want: 4 })
        );
    }

    #[test]
    fn validate_sanitizes_non_finite_heights() {
        let c = vec![Color::from_rgba(0.0, 0.0, 0.0, 1.0); 4];
        let h = [1.0, f32::NAN, f32::INFINITY, f32::NEG_INFINITY];
        let s = validate_submission(&raw(0, &h, &c, &[]), 2).unwrap();
        assert_eq!(s.height, vec![1.0, 0.0, 0.0, 0.0]);
        assert!(s.height.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn validate_empty_normals_defers_to_finite_difference() {
        let c = vec![Color::from_rgba(1.0, 0.5, 0.25, 1.0); 4];
        let s = validate_submission(&raw(9, &[0.0; 4], &c, &[]), 2).unwrap();
        assert_eq!(s.handle, 9);
        assert!(s.normal.is_none(), "no normals ⇒ FD at drain time");
        assert_eq!(&s.color[0..4], &[255, 127, 63, 255]);
    }

    #[test]
    fn validate_packs_supplied_normals() {
        let c = vec![Color::from_rgba(0.0, 0.0, 0.0, 1.0); 4];
        // Un-normalized on purpose: the packer must normalize.
        let s =
            validate_submission(&raw(0, &[0.0; 4], &c, &[Vector3::new(0.0, 5.0, 0.0); 4]), 2)
                .unwrap();
        let n = s.normal.expect("packed");
        assert_eq!(&n[0..4], &[128, 255, 128, 255]);
    }

    #[test]
    fn chunk_dirs_are_unit_and_row_major() {
        let corners = [Vector3::new(1.0, 0.0, 0.0), Vector3::new(0.0, 1.0, 0.0), Vector3::new(0.0, 0.0, 1.0)];
        let dirs = chunk_dirs(corners, 8);
        assert_eq!(dirs.len(), 64);
        assert!(dirs.iter().all(|d| (d.length() - 1.0).abs() < 1.0e-5));
    }

    #[test]
    fn fd_normals_of_a_flat_chunk_point_outward() {
        let corners =
            [Vector3::new(1.0, 0.0, 0.0), Vector3::new(0.0, 1.0, 0.0), Vector3::new(0.0, 0.0, 1.0)];
        let dirs = chunk_dirs(corners, 4);
        let n = fd_normals(corners, &dirs, &[0.0; 16], 4);
        for i in 0..16 {
            let v = Vector3::new(
                n[i * 4] as f32 / 255.0 * 2.0 - 1.0,
                n[i * 4 + 1] as f32 / 255.0 * 2.0 - 1.0,
                n[i * 4 + 2] as f32 / 255.0 * 2.0 - 1.0,
            );
            assert!(v.dot(dirs[i]) > 0.0, "normal must face away from the planet centre");
        }
    }

    #[test]
    fn queue_drains_once_and_reports_one_error() {
        let q = SubmitQueue::default();
        q.push(raw(1, &[], &[], &[]));
        q.push(raw(2, &[], &[], &[]));
        let got = q.drain();
        assert_eq!(got.len(), 2);
        assert!(q.drain().is_empty(), "drain takes everything");

        assert!(q.should_report_error(), "first bad payload reports");
        assert!(!q.should_report_error(), "subsequent ones stay quiet");
    }

    #[test]
    fn queue_clear_drops_stale_submissions() {
        let q = SubmitQueue::default();
        q.push(raw(1, &[], &[], &[]));
        q.clear();
        assert!(q.drain().is_empty());
    }

    #[test]
    fn queue_push_is_safe_from_many_threads() {
        let q = std::sync::Arc::new(SubmitQueue::default());
        let hs: Vec<_> = (0..8)
            .map(|t| {
                let q = std::sync::Arc::clone(&q);
                std::thread::spawn(move || {
                    for i in 0..32 {
                        q.push(raw(t * 32 + i, &[], &[], &[]));
                    }
                })
            })
            .collect();
        for h in hs {
            h.join().unwrap();
        }
        assert_eq!(q.drain().len(), 8 * 32);
    }
}
