//! Pure-CPU triangular chunked quadtree over the icosphere base faces (Phase 1).
//! Each face is the root of a quadtree; nodes subdivide 1->4 by edge midpoints.
//! `select_chunks` returns the camera-dependent visible cut. No GPU, no Godot
//! rendering — a pure function of (faces, camera, params), mirroring
//! `clipmap::compute_patches`. Coexists with the clipmap; nothing here replaces it.

use godot::builtin::Vector3;

use crate::clipmap::FaceFrame;
use crate::icosphere::{base_vertex, BASE_FACES};

/// Barycentric coords within a face: weights toward B and C (A = 1 - wb - wc).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Bary {
    pub wb: f32,
    pub wc: f32,
}

impl Bary {
    pub(crate) fn mid(a: Bary, b: Bary) -> Bary {
        Bary { wb: (a.wb + b.wb) * 0.5, wc: (a.wc + b.wc) * 0.5 }
    }
}

/// Stable id / future cache key: base-4 path of child indices from the face root.
/// `Ord` lets the cache key an LRU index by `(last_used, ChunkId)` (deterministic
/// tie-break).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ChunkId {
    pub face: u8,
    pub depth: u8,
    pub path: u64,
}

/// A selected chunk: one triangle on a face at LOD `level` (== depth).
#[derive(Clone, Copy, Debug)]
pub struct Chunk {
    pub id: ChunkId,
    pub bary: [Bary; 3],
    pub corners: [Vector3; 3],
    pub level: u8,
}

/// The 20 base faces as gnomonic/slerp frames at `radius` (reuses the clipmap frame).
pub fn base_face_frames(radius: f32) -> Vec<FaceFrame> {
    BASE_FACES
        .iter()
        .map(|f| FaceFrame {
            a: base_vertex(f[0]) * radius,
            b: base_vertex(f[1]) * radius,
            c: base_vertex(f[2]) * radius,
            radius,
        })
        .collect()
}

/// Cheap **gnomonic** projection: linear-combine the (on-sphere) face corners and
/// renormalise to the sphere — no slerp/`acos`/`sin`, so selection stays cheap.
/// Accurate enough for LOD sizing and the debug view; the crack-exact slerp
/// (`FaceFrame::project_bary`) is reserved for GPU realize in a later phase.
fn project(frame: &FaceFrame, b: Bary) -> Vector3 {
    let wa = 1.0 - b.wb - b.wc;
    let p = frame.a * wa + frame.b * b.wb + frame.c * b.wc;
    if frame.radius > 0.0 {
        p.normalized() * frame.radius
    } else {
        p
    }
}

/// Build a chunk from its barycentric corners (computes world corners).
fn make_chunk(frame: &FaceFrame, id: ChunkId, bary: [Bary; 3]) -> Chunk {
    Chunk {
        id,
        bary,
        corners: [project(frame, bary[0]), project(frame, bary[1]), project(frame, bary[2])],
        level: id.depth,
    }
}

/// Terrain-off **base** world position of chunk sub-vertex `(i, j)` at resolution
/// `res` — the CPU reference that Task 10's GPU readback compares `ChunkRealize`
/// against. Mirrors the shader's math exactly (and `chunk_mesh::chunk_grid`'s
/// vertex order): local barycentric `wa=(res-i-j)/res, wb=i/res, wc=j/res`, then
/// face-barycentric `wa*bary[0] + wb*bary[1] + wc*bary[2]`, then the SAME
/// **gnomonic** projection [`project`] the selection/cull/corner path uses. (It
/// previously used the tri-slerp `FaceFrame::project_bary`, a *different*
/// bary→sphere mapping that diverges from the gnomonic one by up to ~1.5°
/// mid-face — ~170 km at planet radius — so realized geometry landed far from where the
/// cut selected it.) No terrain displacement.
///
/// At the three corners `(0,0)/(res,0)/(0,res)` this returns
/// `chunk.corners[0]/[1]/[2]` respectively.
pub fn chunk_subvertex_base(
    frame: &FaceFrame,
    chunk: &Chunk,
    res: u32,
    i: u32,
    j: u32,
) -> Vector3 {
    let rf = res as f32;
    let wa = (res - i - j) as f32 / rf;
    let wb = i as f32 / rf;
    let wc = j as f32 / rf;
    let wb_f = wa * chunk.bary[0].wb + wb * chunk.bary[1].wb + wc * chunk.bary[2].wb;
    let wc_f = wa * chunk.bary[0].wc + wb * chunk.bary[1].wc + wc * chunk.bary[2].wc;
    project(frame, Bary { wb: wb_f, wc: wc_f })
}

/// The whole face as the depth-0 root chunk (corners A, B, C). (Test helper; the
/// hot path inlines the root barycentric corners in `select_chunks`.)
#[cfg_attr(not(test), allow(dead_code))]
pub fn root_chunk(frame: &FaceFrame, face: u8) -> Chunk {
    let bary = [Bary { wb: 0.0, wc: 0.0 }, Bary { wb: 1.0, wc: 0.0 }, Bary { wb: 0.0, wc: 1.0 }];
    make_chunk(frame, ChunkId { face, depth: 0, path: 0 }, bary)
}

/// Barycentric 1->4 midpoint split (no projection) — the cheap inner step of the
/// descent. Order matches the path code: 0/1/2 hug A/B/C, 3 is the centre.
pub(crate) fn split_bary(b: [Bary; 3]) -> [[Bary; 3]; 4] {
    let [a, bb, c] = b;
    let (mab, mbc, mca) = (Bary::mid(a, bb), Bary::mid(bb, c), Bary::mid(c, a));
    [[a, mab, mca], [mab, bb, mbc], [mca, mbc, c], [mab, mbc, mca]]
}

/// Split a chunk 1->4 by edge midpoints: child 0/1/2 hug corner A/B/C, child 3
/// is the inverted centre. Child index `k` extends the parent path: `path<<2 | k`.
/// (Test/documentation helper — the hot path uses `split_bary` to avoid building
/// `Chunk`s for interior nodes.)
#[cfg_attr(not(test), allow(dead_code))]
fn subdivide(parent: &Chunk, frame: &FaceFrame) -> [Chunk; 4] {
    let [a, b, c] = parent.bary;
    let (mab, mbc, mca) = (Bary::mid(a, b), Bary::mid(b, c), Bary::mid(c, a));
    let child = |k: u64, bary: [Bary; 3]| {
        let id = ChunkId {
            face: parent.id.face,
            depth: parent.id.depth + 1,
            path: (parent.id.path << 2) | k,
        };
        make_chunk(frame, id, bary)
    };
    [
        child(0, [a, mab, mca]),
        child(1, [mab, b, mbc]),
        child(2, [mca, mbc, c]),
        child(3, [mab, mbc, mca]),
    ]
}

/// Descend one quadtree node. The split test is cheap — an **analytic** would-be
/// triangle edge (`face_edge / 2^depth / chunk_res`) compared against the distance
/// to the chunk **centroid** (a *single* projection). Interior nodes therefore
/// cost one projection each; the full 3-corner projection happens only when a
/// node is emitted as a leaf. A chunk is realized as `chunk_res`×`chunk_res`
/// triangles, so this stops `log2(chunk_res)` levels above per-triangle — chunk
/// count (and select cost) is governed by chunk count, not triangle count.
#[allow(clippy::too_many_arguments)]
fn descend(
    frame: &FaceFrame,
    face: u8,
    depth: u8,
    path: u64,
    bary: [Bary; 3],
    camera: Vector3,
    se: f32,
    chunk_res: u32,
    max_depth: u8,
    face_edge: f32,
    cull: Option<f32>,
    surface: Option<&SurfaceFn<'_>>,
    out: &mut Vec<Chunk>,
) {
    // The node's spherical-patch edge length (analytic; halves each level).
    let patch_edge = face_edge / (1u32 << depth.min(31)) as f32;
    let centroid = Bary {
        wb: (bary[0].wb + bary[1].wb + bary[2].wb) / 3.0,
        wc: (bary[0].wc + bary[1].wc + bary[2].wc) / 3.0,
    };
    // Centroid on the sphere surface (|pc| == radius). Reused for cull + LOD.
    let pc = project(frame, centroid);

    // Horizon cull: skip this node AND its subtree when it is *fully* beyond the
    // horizon (occluded by the planet). Curvature-aware angular test: the patch
    // sits in a cone of half-angle `alpha` about its centroid direction; its
    // most-forward point is `alpha` closer (in angle) to the camera direction.
    // Raising that point by the max terrain height `max_disp` and projecting onto
    // the camera ray gives its `dot(P, C)`; if even that stays below the horizon
    // value r² the whole patch is hidden. A linear chord bound would wrongly keep
    // coarse back patches (their chord "reaches" the horizon though the sphere
    // curves every point away) — the angle form culls them correctly. Conservative
    // by `alpha` (a generous patch angular radius), so straddling patches descend
    // and their children are culled instead. `max_disp` is in world units.
    if let Some(max_disp) = cull {
        let r = pc.length();
        let d = camera.length();
        if r > 1.0e-4 && d > r {
            let cos_phi = (pc.dot(camera) / (r * d)).clamp(-1.0, 1.0);
            let phi = cos_phi.acos(); // angle between centroid dir and camera dir
            let alpha = patch_edge / r; // generous angular radius of the patch
            let near = (phi - alpha).max(0.0); // angle of the most-forward point
            if (r + max_disp) * d * near.cos() < r * r {
                return;
            }
        }
    }

    let split = depth < max_depth && {
        let tri_edge = patch_edge / chunk_res as f32;
        // Measure the LOD distance to the DISPLACED surface, not the bare
        // sphere: over elevated terrain the camera can never approach the
        // undisplaced centroid, which would silently cap the reachable depth
        // (and thus the detail) by the local terrain height.
        let pc_surf = match surface {
            Some(f) => pc * (1.0 + f(pc) / pc.length().max(1.0e-6)),
            None => pc,
        };
        let dist = (pc_surf - camera).length().max(1.0e-4);
        tri_edge / dist > se
    };
    if split {
        for (k, kb) in split_bary(bary).into_iter().enumerate() {
            let p = (path << 2) | k as u64;
            descend(
                frame, face, depth + 1, p, kb, camera, se, chunk_res, max_depth, face_edge, cull,
                surface, out,
            );
        }
    } else {
        out.push(make_chunk(frame, ChunkId { face, depth, path }, bary));
    }
}

/// Select the visible cut of the chunked quadtree across all faces. Returns one
/// **chunk descriptor per patch** (not per triangle) — each chunk stands in for a
/// `chunk_res`×`chunk_res` tessellation realized later. Pure function of (faces,
/// camera, params). Unbalanced in Phase 1 (balancing is Phase 3).
/// `cull` enables horizon (back-of-planet) culling: `Some(max_disp)` skips any
/// patch fully beyond the horizon, where `max_disp` is the maximum terrain height
/// above the surface (world units) so tall terrain peeking over the horizon is
/// kept. `None` selects the whole LOD cut (no horizon culling).
pub fn select_chunks(
    faces: &[FaceFrame],
    camera: Vector3,
    screen_error: f32,
    chunk_res: u32,
    max_depth: u8,
    cull: Option<f32>,
) -> Vec<Chunk> {
    select_chunks_displaced(faces, camera, screen_error, chunk_res, max_depth, cull, None)
}

/// Radial surface displacement (world units, positive = outward) at a point on
/// the undisplaced sphere. Used by [`select_chunks_displaced`] to measure the
/// LOD distance to the real (terrain-displaced) surface.
pub type SurfaceFn<'a> = dyn Fn(Vector3) -> f32 + 'a;

/// [`select_chunks`] with an optional radial-displacement sampler. When
/// `surface` is `Some`, each node's LOD distance is measured to the **displaced**
/// centroid `pc · (1 + surface(pc)/|pc|)` instead of the bare sphere. Without it
/// (`None`, or `select_chunks`), a camera standing on terrain of height `h`
/// can never bring `dist` below `h`, capping the reachable depth at
/// `log2(face_edge / (se·h·chunk_res))` regardless of how close it is to the
/// actual ground. The sampler only steers LOD, so a coarse height source is fine.
pub fn select_chunks_displaced(
    faces: &[FaceFrame],
    camera: Vector3,
    screen_error: f32,
    chunk_res: u32,
    max_depth: u8,
    cull: Option<f32>,
    surface: Option<&SurfaceFn<'_>>,
) -> Vec<Chunk> {
    let root = [Bary { wb: 0.0, wc: 0.0 }, Bary { wb: 1.0, wc: 0.0 }, Bary { wb: 0.0, wc: 1.0 }];
    let mut out = Vec::new();
    for (fi, frame) in faces.iter().enumerate() {
        let face_edge = frame.edge_len();
        descend(
            frame, fi as u8, 0, 0, root, camera, screen_error, chunk_res, max_depth, face_edge,
            cull, surface, &mut out,
        );
    }
    out
}

/// Geomorph factor for a selected leaf chunk (Phase 5).
///
/// A leaf at `depth` is valid over the screen-error band `m = tri_edge/dist ∈
/// (se/2, se]`, where `tri_edge = face_edge / 2^depth / chunk_res` and `dist` is
/// the camera distance to the chunk centroid — the SAME quantities [`descend`]
/// tests when it splits (`tri_edge/dist > se`):
/// - the chunk's own split fails at `m ≤ se` (the **near** end), and
/// - its parent's split (`2*tri_edge/dist`) fails at `m ≤ se/2` (the **far** end,
///   where the parent takes over).
///
/// Morph stays at `1` (full detail) over the NEAR part of the band and ramps to
/// `0` only over its far [`MORPH_REGION`] fraction, ending at `m = se/2`:
/// - `1` for `m ≥ se/2·(1 + MORPH_REGION)`: full detail (most of the band);
/// - ramps `1 → 0` as `m` falls through the far `MORPH_REGION` of the band;
/// - `0` at the far end (`m = se/2`): parent resolution, so the handover to the
///   parent is seamless (the surface shader blends the fine grid toward the
///   even/parent sublattice as `morph → 0`).
///
/// Concentrating the morph near the transition (rather than ramping across the
/// whole band) keeps most visible chunks at full detail, so far fewer chunks are
/// mid-morph at once — minimising the texture "swim" of geometry that deforms
/// every frame the camera moves.
///
/// `depth == 0` chunks (the 20 base faces) have no coarser parent to blend with,
/// so they always return `1.0` (no morphing).
pub fn morph_factor(face_edge: f32, depth: u8, chunk_res: u32, dist: f32, se: f32) -> f32 {
    if depth == 0 {
        return 1.0;
    }
    let patch_edge = face_edge / (1u32 << depth.min(31)) as f32;
    let tri_edge = patch_edge / chunk_res.max(1) as f32;
    let m = tri_edge / dist.max(1.0e-4);
    let half = (se * 0.5).max(1.0e-6); // far end of the leaf band (parent handover)
    let region = (MORPH_REGION * half).max(1.0e-6); // morph ramp width, from the far end
    ((m - half) / region).clamp(0.0, 1.0)
}

/// Fraction of a leaf's screen-error band (measured from the far/parent-handover
/// end) over which geomorph ramps `0 → 1`. The near part stays at full detail.
const MORPH_REGION: f32 = 0.4;

#[cfg(test)]
mod tests {
    use super::*;

    fn bary_area(c: [Bary; 3]) -> f32 {
        let (ax, ay) = (c[1].wb - c[0].wb, c[1].wc - c[0].wc);
        let (bx, by) = (c[2].wb - c[0].wb, c[2].wc - c[0].wc);
        0.5 * (ax * by - ay * bx).abs()
    }

    #[test]
    fn chunk_subvertex_base_matches_corners() {
        // At the 3 lattice corners, chunk_subvertex_base must equal the
        // chunk's (gnomonic) world corners — the same mapping selection uses.
        let res = 16u32;
        let frames = base_face_frames(1000.0);
        let frame = &frames[7];
        let root = root_chunk(frame, 7);
        // Use a non-trivial sub-chunk so the corners aren't the face corners.
        let chunk = subdivide(&root, frame)[3]; // inverted centre child
        let corners = [(0u32, 0u32), (res, 0), (0, res)];
        for (k, (i, j)) in corners.iter().enumerate() {
            let got = chunk_subvertex_base(frame, &chunk, res, *i, *j);
            let want = chunk.corners[k];
            assert!(
                (got - want).length() < 1e-3,
                "corner {k} (i={i},j={j}): got {got}, want {want}"
            );
        }
    }

    #[test]
    fn subdivide_splits_parent_into_four_equal_children() {
        let frames = base_face_frames(1.0);
        let root = root_chunk(&frames[0], 0);
        let kids = subdivide(&root, &frames[0]);
        for (k, kid) in kids.iter().enumerate() {
            assert_eq!(kid.id.depth, 1);
            assert_eq!(kid.id.path, (k as u64));
            assert_eq!(kid.level, 1);
        }
        let parent_area = bary_area(root.bary);
        let sum: f32 = kids.iter().map(|c| bary_area(c.bary)).sum();
        assert!((sum - parent_area).abs() < 1e-6, "children must tile parent");
        for kid in &kids {
            assert!((bary_area(kid.bary) - parent_area / 4.0).abs() < 1e-6);
        }
    }

    #[test]
    fn all_corners_lie_on_the_sphere() {
        let r = 1000.0;
        let frames = base_face_frames(r);
        let root = root_chunk(&frames[5], 5);
        for kid in subdivide(&root, &frames[5]) {
            for c in kid.corners {
                assert!((c.length() - r).abs() < 1e-2, "corner off-sphere: {}", c.length());
            }
        }
    }

    #[test]
    fn uniform_when_screen_error_zero_covers_each_face_to_max_depth() {
        // chunk_res 1 + screen_error 0 => split every node to max_depth.
        let frames = base_face_frames(1.0);
        let depth = 3u8;
        let chunks = select_chunks(&frames, Vector3::new(0.0, 0.0, 3.0), 0.0, 1, depth, None);
        assert_eq!(chunks.len(), 20 * 4usize.pow(depth as u32));
        assert!(chunks.iter().all(|c| c.level == depth));
    }

    #[test]
    fn closer_camera_never_selects_fewer_chunks() {
        let frames = base_face_frames(1000.0);
        let se = 0.05;
        let far = select_chunks(&frames, Vector3::new(0.0, 0.0, 8000.0), se, 16, 12, None).len();
        let near = select_chunks(&frames, Vector3::new(0.0, 0.0, 1100.0), se, 16, 12, None).len();
        assert!(near >= far, "near={} should be >= far={}", near, far);
    }

    #[test]
    fn larger_chunk_res_selects_fewer_chunks() {
        // Coarser chunks (more triangles each) => fewer chunk descriptors chosen.
        let frames = base_face_frames(1000.0);
        let cam = Vector3::new(0.0, 0.0, 1100.0);
        let fine = select_chunks(&frames, cam, 0.02, 4, 14, None).len();
        let coarse = select_chunks(&frames, cam, 0.02, 32, 14, None).len();
        assert!(coarse < fine, "coarse={} should be < fine={}", coarse, fine);
    }

    #[test]
    fn selection_is_deterministic() {
        let frames = base_face_frames(1000.0);
        let cam = Vector3::new(300.0, 200.0, 1100.0);
        let a = select_chunks(&frames, cam, 0.04, 16, 10, None);
        let b = select_chunks(&frames, cam, 0.04, 16, 10, None);
        let ids_a: Vec<_> = a.iter().map(|c| c.id).collect();
        let ids_b: Vec<_> = b.iter().map(|c| c.id).collect();
        assert_eq!(ids_a, ids_b);
    }

    #[test]
    fn chunk_count_is_bounded_for_a_close_camera() {
        // The whole point: a close camera selects PATCHES, not triangles. With a
        // realistic chunk_res the cut is a few thousand chunks, not ~100k.
        let frames = base_face_frames(1000.0);
        let chunks = select_chunks(&frames, Vector3::new(0.0, 0.0, 1010.0), 0.02, 16, 16, None);
        assert!(chunks.len() < 20_000, "got {} chunks", chunks.len());
    }

    #[test]
    fn morph_factor_boundaries() {
        // Geomorph ramp (Phase 5): pick a geometry and solve for the distances at
        // the two split boundaries, then assert morph hits 1 / 0.5 / 0 there.
        let face_edge = 1000.0_f32;
        let depth = 3u8;
        let chunk_res = 16u32;
        let se = 0.02_f32;
        let patch_edge = face_edge / (1u32 << depth) as f32;
        let tri_edge = patch_edge / chunk_res as f32;

        // Near end: m = se  (own split just failed) → morph = 1.
        let dist_near = tri_edge / se;
        assert!((morph_factor(face_edge, depth, chunk_res, dist_near, se) - 1.0).abs() < 1e-5);

        // Far end: m = se/2 (parent's split just failed) → morph = 0.
        let dist_far = tri_edge / (se / 2.0);
        assert!(morph_factor(face_edge, depth, chunk_res, dist_far, se).abs() < 1e-5);

        // Morph ramps only over the far MORPH_REGION (0.4) of the band: it hits
        // 0.5 at m = se/2·(1 + 0.4·0.5) = 0.6·se, and is already full detail (1) by
        // the middle of the band (m = 0.8·se > se/2·1.2).
        let dist_half = tri_edge / (0.6 * se);
        assert!((morph_factor(face_edge, depth, chunk_res, dist_half, se) - 0.5).abs() < 1e-5);
        let dist_mid = tri_edge / (0.8 * se);
        assert_eq!(morph_factor(face_edge, depth, chunk_res, dist_mid, se), 1.0);

        // Beyond the band (further than the far end) stays clamped at 0.
        assert_eq!(morph_factor(face_edge, depth, chunk_res, dist_far * 4.0, se), 0.0);
        // Closer than the near end stays clamped at 1.
        assert_eq!(morph_factor(face_edge, depth, chunk_res, dist_near * 0.25, se), 1.0);

        // depth 0 has no parent → always full detail regardless of distance.
        assert_eq!(morph_factor(face_edge, 0, chunk_res, dist_far, se), 1.0);
        assert_eq!(morph_factor(face_edge, 0, chunk_res, dist_near, se), 1.0);
    }

    #[test]
    fn displaced_surface_unlocks_full_depth_over_elevated_terrain() {
        // Planet-scale regression for the "can never see the landmark" bug: a real
        // sits ~120 m up; with height exaggeration 5 the surface is displaced
        // ~0.6 km outward. A camera hovering 50 m above the DISPLACED ground is
        // ~0.65 km from the sphere, so with the undisplaced distance the split
        // `tri_edge/dist > se` stalls several levels short of max_depth. The
        // surface sampler must restore the full depth.
        let r = 6371.0_f32; // km world units
        let frames = base_face_frames(r);
        let (se, chunk_res, max_depth) = (0.05_f32, 20u32, 16u8);
        let disp = 0.6_f32; // km, uniform displaced terrain
        // Camera 50 m above the displaced surface, off +Z.
        let cam = Vector3::new(0.0, 0.0, r + disp + 0.05);

        let deepest = |chunks: &[Chunk]| chunks.iter().map(|c| c.level).max().unwrap();

        let flat = select_chunks(&frames, cam, se, chunk_res, max_depth, None);
        let displaced = select_chunks_displaced(
            &frames,
            cam,
            se,
            chunk_res,
            max_depth,
            None,
            Some(&|_p: Vector3| disp),
        );

        // Sanity: without the sampler the depth really is capped well short.
        assert!(
            deepest(&flat) < max_depth,
            "undisplaced dist should cap depth, got {}",
            deepest(&flat)
        );
        // With the sampler the chunk under the camera reaches max_depth.
        assert_eq!(
            deepest(&displaced),
            max_depth,
            "displaced dist must reach max_depth"
        );
        // And the fix must not blow up the cut size (same order of magnitude —
        // the deeper rings are a handful of extra chunks, not a flood).
        assert!(
            displaced.len() < flat.len() + 200,
            "cut exploded: flat {} displaced {}",
            flat.len(),
            displaced.len()
        );
    }

    #[test]
    fn horizon_cull_removes_the_back_hemisphere() {
        // Camera off +Z at altitude. Culling must (a) shrink the cut and (b)
        // remove the most back-facing chunks — i.e. raise the minimum centroid
        // alignment with the camera direction. (The cull is conservative: patches
        // straddling the horizon are kept, so we test the floor, not every chunk.)
        let r = 1000.0;
        let frames = base_face_frames(r);
        let cam = Vector3::new(0.0, 0.0, 3000.0);
        let full = select_chunks(&frames, cam, 0.02, 16, 10, None);
        let culled = select_chunks(&frames, cam, 0.02, 16, 10, Some(0.0));
        assert!(culled.len() < full.len(), "culled {} !< full {}", culled.len(), full.len());

        let min_align = |chunks: &[Chunk]| {
            chunks
                .iter()
                .map(|c| {
                    let pc = (c.corners[0] + c.corners[1] + c.corners[2]) / 3.0;
                    pc.normalized().dot(cam.normalized())
                })
                .fold(f32::INFINITY, f32::min)
        };
        // Full cut reaches the far side (alignment ≈ −1); culled cut's floor is
        // lifted toward the horizon — the back-facing chunks are gone.
        assert!(min_align(&full) < -0.5, "full should include back chunks, got {}", min_align(&full));
        assert!(
            min_align(&culled) > min_align(&full) + 0.3,
            "cull should remove back-facing chunks: full floor {}, culled floor {}",
            min_align(&full),
            min_align(&culled)
        );
    }

    #[test]
    fn horizon_cull_shrinks_more_near_the_ground() {
        // The horizon is closer when the camera is low, so culling removes a
        // larger fraction near the surface than from far away.
        let r = 1000.0;
        let frames = base_face_frames(r);
        let near = Vector3::new(0.0, 0.0, 1050.0);
        let far = Vector3::new(0.0, 0.0, 9000.0);
        let frac = |cam: Vector3| {
            let full = select_chunks(&frames, cam, 0.02, 16, 12, None).len() as f32;
            let kept = select_chunks(&frames, cam, 0.02, 16, 12, Some(0.0)).len() as f32;
            kept / full
        };
        assert!(
            frac(near) < frac(far),
            "near {} should keep a smaller fraction than far {}",
            frac(near),
            frac(far)
        );
    }
}
