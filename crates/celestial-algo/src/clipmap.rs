//! **Triangular** camera-following clipmap — fills a triangle with nested
//! triangular layers, coarse on the outside and finer toward the **camera**. The
//! unrealized / realized patch split is kept, but every region is a triangle (so
//! the planes tile an icosahedron face exactly) and the fine layers chase the
//! camera instead of sitting at the face centroid.
//!
//! A face is the triangle `A B C`. Layer 0 (coarsest) always covers the *whole*
//! face — that keeps the outer boundary at a uniform resolution shared by every
//! face, so adjacent faces meet crack-free (a watertight, flat, un-normalised
//! icosahedron). Layer 0 leaves a triangular **hole** positioned around the
//! camera's projection onto the face; layer 1 fills it (finer, smaller, also
//! centred on the camera), leaving its own hole; and so on, down to a solid
//! finest triangle right under the camera. So all the detail piles up toward the
//! camera across the whole sphere.
//!
//! Two clearly separated halves, as before:
//!
//! * [`Patch`] — an **unrealized** patch: pure data, no geometry. Its outer
//!   triangle ([`Patch::outer_origin`] / [`Patch::outer_side`]), cell size
//!   ([`Patch::g`]) and the hole the finer layer fills ([`Patch::hole_origin`] /
//!   [`Patch::hole_side`]). Built by [`compute_patches`].
//! * [`RealizedPatch`] — the **realized** patch: the actual triangles. Produced
//!   by [`realize_patch`], a pure function of one descriptor, so patches realize
//!   independently and in parallel ([`realize_all`]).

use std::thread;

use godot::builtin::{Color, Vector3};

/// Hard cap on triangles a single layer's *region* (hole footprint or side band)
/// may realize, so a very close camera (or the editor viewport on scene reload)
/// can never explode the mesh and freeze the editor. `compute_patches` stops
/// deepening a chain once the next layer's region would exceed this. Checked
/// against the actual region — not the whole face — so big planets (whose
/// regions are screen-bounded and small relative to the face) still refine.
const MAX_FACE_TRIS: usize = 150_000;

/// Spherical linear interpolation between two **unit** vectors. `t` in `[0, 1]`.
fn slerp_unit(p: Vector3, q: Vector3, t: f32) -> Vector3 {
    let d = p.dot(q).clamp(-1.0, 1.0);
    let omega = d.acos();
    let so = omega.sin();
    if so.abs() < 1.0e-6 {
        // Nearly identical (or antipodal — won't happen for a triangle edge).
        return (p * (1.0 - t) + q * t).normalized();
    }
    p * (((1.0 - t) * omega).sin() / so) + q * ((t * omega).sin() / so)
}

/// One 2-stage slerp inside a spherical triangle, sweeping from `apex` out to the
/// `c1`-`c2` edge. `w1`/`w2` are the weights toward `c1`/`c2`. The result lies on
/// the unit sphere. Used three ways (one per apex) and averaged in `project_bary`
/// so the mapping is symmetric — no corner is special — which keeps cells uniform.
fn tri_slerp(apex: Vector3, c1: Vector3, c2: Vector3, w1: f32, w2: f32) -> Vector3 {
    let s = w1 + w2; // distance from the apex (0 = apex, 1 = opposite edge)
    if s <= 1.0e-7 {
        return apex;
    }
    let edge = slerp_unit(c1, c2, w2 / s);
    slerp_unit(apex, edge, s)
}

/// World position of a point at finest-cell position `p` along the face edge from
/// world endpoint `a` to `b` (`radius > 0` => the edge is a 2-corner slerp on the
/// sphere, matching `project_bary`; `radius == 0` => linear). `n_fine` = finest
/// cells along the edge.
fn edge_point(a: Vector3, b: Vector3, radius: f32, p: u32, n_fine: u32) -> Vector3 {
    let t = p as f32 / n_fine as f32;
    if radius > 0.0 {
        slerp_unit(a.normalized(), b.normalized(), t) * radius
    } else {
        a + (b - a) * t
    }
}

/// **Canonical adaptive tessellation of a face edge** — the finest-cell positions
/// (`0..=n_fine`, always including both ends) of the vertices that *any* face
/// should place on this edge. A **pure function of the global edge + camera +
/// params**, so two faces sharing the edge derive the *identical* set without
/// talking — the basis for crack-free inter-face boundaries (CEL-66).
///
/// Each base-cell-sized segment (`g0` finest cells) is split in half, recursively,
/// while its on-surface screen size exceeds `screen_error` (the same rule the LOD
/// chains use), down to a single finest cell. `g0` is a power of two, so every
/// split point is lattice-aligned. The split is geometric and deterministic, so
/// endpoint order doesn't matter (the test asserts the world-vertex set is
/// identical when `a`/`b` are swapped).
pub(crate) fn edge_tessellation(
    a: Vector3,
    b: Vector3,
    radius: f32,
    cam: Vector3,
    screen_error: f32,
    n0: u32,
    g0: u32,
) -> Vec<u32> {
    let n_fine = n0 * g0;
    let mut verts = std::collections::BTreeSet::new();
    verts.insert(0u32);
    verts.insert(n_fine);

    // Split segment [lo, hi] (finest positions, hi-lo a power of two) while too big.
    fn split(
        lo: u32,
        hi: u32,
        a: Vector3,
        b: Vector3,
        radius: f32,
        n_fine: u32,
        cam: Vector3,
        se: f32,
        out: &mut std::collections::BTreeSet<u32>,
    ) {
        if hi - lo <= 1 {
            return; // already finest
        }
        let pa = edge_point(a, b, radius, lo, n_fine);
        let pb = edge_point(a, b, radius, hi, n_fine);
        let mid = lo + (hi - lo) / 2;
        let pm = edge_point(a, b, radius, mid, n_fine);
        let dist = (cam - pm).length().max(1.0e-4);
        if (pb - pa).length() / dist > se {
            out.insert(mid);
            split(lo, mid, a, b, radius, n_fine, cam, se, out);
            split(mid, hi, a, b, radius, n_fine, cam, se, out);
        }
    }

    for k in 0..n0 {
        let (lo, hi) = (k * g0, (k + 1) * g0);
        verts.insert(lo);
        split(lo, hi, a, b, radius, n_fine, cam, screen_error, &mut verts);
    }
    verts.into_iter().collect()
}

/// The three world-space corners of a face.
///
/// When `radius > 0` the face is a **gnomonic** chart of a sphere of that radius
/// centred on the origin: the flat triangle `A B C` is the chart, and every point
/// is mapped onto the sphere by central projection from the origin
/// (`p -> p.normalized() * radius`). The camera is likewise projected gnomonically
/// (the ray from the origin through the camera, hitting the face's infinite
/// plane), so adjacent faces agree at their shared edges (the radial direction is
/// shared) — that is what makes the seams line up on a real sphere.
///
/// When `radius == 0` the face stays **flat** (no projection) and the camera is
/// projected orthogonally onto the face plane — the single-triangle demo, where a
/// central projection from the origin would degenerate (the origin lies in the
/// plane).
#[derive(Clone, Copy, Debug)]
pub struct FaceFrame {
    pub a: Vector3,
    pub b: Vector3,
    pub c: Vector3,
    /// Sphere radius for gnomonic projection; `0` => flat (no projection).
    pub radius: f32,
}

impl FaceFrame {
    /// Outward face normal (corners wind CCW seen from outside).
    pub fn normal(&self) -> Vector3 {
        (self.b - self.a).cross(self.c - self.a).normalized()
    }

    /// Edge length (the `A`->`B` edge).
    pub fn edge_len(&self) -> f32 {
        (self.b - self.a).length()
    }

    /// Map a lattice point given by its barycentric weights (`wb` toward `B`, `wc`
    /// toward `C`, the rest toward `A`) onto the rendered surface.
    ///
    /// Flat faces (`radius == 0`) just take the linear combination. Sphere faces
    /// use **spherical (slerp) barycentric interpolation** rather than the simpler
    /// gnomonic (linear-combine-then-normalize): gnomonic crowds cells toward the
    /// face's edges and bloats them at the centre (~1.5x area swing across a face),
    /// which reads as uneven, stretched triangles; slerp spaces them by *equal
    /// angle*, so cells stay uniform.
    ///
    /// Crack-safe across faces: each face edge is a pure two-corner slerp (e.g. a
    /// point on edge `AB` only depends on `A` and `B`), so neighbouring faces
    /// compute their shared edge identically. (Interior points are mildly
    /// apex-biased toward `A`, but that never affects an edge, so no gaps appear.)
    pub fn project_bary(&self, wb: f32, wc: f32) -> Vector3 {
        let wa = 1.0 - wb - wc;
        if self.radius <= 0.0 {
            return self.a * wa + self.b * wb + self.c * wc;
        }
        let (ua, ub, uc) = (self.a.normalized(), self.b.normalized(), self.c.normalized());
        // Symmetric: sweep from each of the three apexes and average. All three
        // agree on every edge (a boundary point reduces to the same edge slerp), so
        // shared face edges stay crack-free; the interior is unbiased.
        let pa = tri_slerp(ua, ub, uc, wb, wc); // apex A
        let pb = tri_slerp(ub, uc, ua, wc, wa); // apex B
        let pc = tri_slerp(uc, ua, ub, wa, wb); // apex C
        (pa + pb + pc).normalized() * self.radius
    }

    /// Project the camera onto this face, as barycentric `(u, v)` where the foot on
    /// the chart plane is `A + u*(B-A) + v*(C-A)`. Gnomonic (central from the
    /// origin) for a sphere face, orthogonal for a flat face. The `(u, v)` may land
    /// outside the triangle — holes are centred on it regardless, so detail tracks
    /// the camera even when the foot is off-face. `None` => camera behind the face.
    fn project_camera(&self, cam: Vector3) -> Option<(f32, f32)> {
        let e1 = self.b - self.a;
        let e2 = self.c - self.a;
        let n = e1.cross(e2); // outward (un-normalised) normal
        let foot = if self.radius > 0.0 {
            // Gnomonic: where the ray origin->camera pierces the face plane.
            let denom = cam.dot(n);
            if denom <= 0.0 {
                return None; // camera behind the face (or parallel)
            }
            let t = self.a.dot(n) / denom;
            if t <= 0.0 {
                return None;
            }
            cam * t
        } else {
            // Orthogonal: drop the camera straight onto the plane along the normal.
            if (cam - self.a).dot(n) <= 0.0 {
                return None; // camera on the inner side
            }
            cam
        };
        let rp = foot - self.a;
        let d00 = e1.dot(e1);
        let d01 = e1.dot(e2);
        let d11 = e2.dot(e2);
        let d20 = rp.dot(e1);
        let d21 = rp.dot(e2);
        let det = d00 * d11 - d01 * d01;
        if det.abs() < 1.0e-12 {
            return None;
        }
        Some(((d11 * d20 - d01 * d21) / det, (d00 * d21 - d01 * d20) / det))
    }

    /// Closest point of the face triangle (in its chart plane) to the chart point
    /// at barycentric `(u, v)` — used when the camera's foot lands off the face.
    /// Returns the nearest edge (`0` = AB, `1` = BC, `2` = CA), the position along
    /// it in `[0, 1]` (edge 0: A->B, edge 1: C->B, edge 2: A->C — matching the
    /// lattice `along` coordinate), and the point's barycentric coords.
    fn closest_edge_point(&self, u: f32, v: f32) -> (u8, f32, (f32, f32)) {
        let p = self.a + (self.b - self.a) * u + (self.c - self.a) * v;
        // (edge, segment start/end, barycentric of start, barycentric step)
        let segs = [
            (0u8, self.a, self.b, (0.0f32, 0.0f32), (1.0f32, 0.0f32)),
            (1u8, self.c, self.b, (0.0, 1.0), (1.0, -1.0)),
            (2u8, self.a, self.c, (0.0, 0.0), (0.0, 1.0)),
        ];
        let mut best = (0u8, 0.0f32, (0.0f32, 0.0f32));
        let mut best_d = f32::MAX;
        for (e, s0, s1, b0, db) in segs {
            let d = s1 - s0;
            let t = ((p - s0).dot(d) / d.dot(d)).clamp(0.0, 1.0);
            let q = s0 + d * t;
            let dist = (p - q).length_squared();
            if dist < best_d {
                best_d = dist;
                best = (e, t, (b0.0 + db.0 * t, b0.1 + db.1 * t));
            }
        }
        best
    }
}

/// A whole-face coarse patch with no hole — used for faces the camera is behind.
fn solid_face(frame: FaceFrame, n_fine: u32, g0: u32) -> Patch {
    Patch {
        layer: 0,
        n_fine,
        g: g0,
        outer_origin: (0, 0),
        outer_side: n_fine,
        hole_origin: (0, 0),
        hole_side: 0,
        frame,
        kind: PatchKind::Hole,
        cam_local: None,
        screen_error: 0.0,
    }
}

/// Does the same-orientation triangle `{(oi,oj), side}` overlap the face triangle
/// `{(0,0), n_fine}` (both in finest cells)? Used to stop the refinement chain
/// once a footprint — the previous layer's hole — falls fully off the face.
fn overlaps_face(oi: i64, oj: i64, side: i64, n_fine: i64) -> bool {
    let lo_i = oi.max(0);
    let lo_j = oj.max(0);
    let hyp = (oi + oj + side).min(n_fine);
    lo_i + lo_j < hyp
}

/// Which shape a [`Patch`] refines.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PatchKind {
    /// Triangular region nested toward the camera's foot (foot **inside** the
    /// face). `outer_*`/`hole_*` are sub-triangles; the annulus between them is
    /// rendered. Used for the single face the camera looks straight down on.
    Hole,
    /// Band from `edge` (`0` = AB/`J=0`, `1` = BC/hypotenuse, `2` = CA/`I=0`),
    /// nested toward that edge (foot **outside** the face). `outer_side`/`hole_side`
    /// are *depths from the edge* and `outer_origin`/`hole_origin` are along-edge
    /// `(start, width)` intervals (all finest cells): each band is bounded both in
    /// depth and along the edge by the screen-error radius, so on a big planet
    /// patches stay screen-sized instead of spanning the whole face edge. Layer 0
    /// still covers the whole face, so coverage stays complete.
    Side(u8),
}

/// An **unrealized** triangular clipmap patch — pure data, no geometry. All
/// coordinates are in *finest* cells (the lattice triangle `{ I, J >= 0,
/// I + J <= n_fine }`), so layers at different cell sizes share one coordinate
/// system.
#[derive(Clone, Copy, Debug)]
pub struct Patch {
    /// Which clipmap layer this is (0 = coarsest / whole face).
    pub layer: u32,
    /// Total cells along the face edge at the *finest* resolution.
    pub n_fine: u32,
    /// This layer's cell size, in finest cells (`2^(num_layers-1-layer)`).
    pub g: u32,
    /// Hole: outer triangle origin (its `A` corner), finest cells (signed).
    /// Side: the band's along-edge `(start, width)` interval.
    pub outer_origin: (i64, i64),
    /// Hole: outer triangle side. Side: outer band depth from the edge. Finest cells.
    pub outer_side: u32,
    /// Hole: hole (next finer layer) origin, finest cells (signed). Side: the hole
    /// band's along-edge `(start, width)`.
    pub hole_origin: (i64, i64),
    /// Hole: hole side. Side: hole band depth. Finest cells. `0` => no hole (finest).
    pub hole_side: u32,
    /// The face this patch lives on (world corners).
    pub frame: FaceFrame,
    /// Triangular hole nested toward a point, or a full-row band from an edge.
    pub kind: PatchKind,
    /// Camera (local space) and raw screen error, so `realize_patch` can derive the
    /// canonical shared-edge tessellation and conform this patch's boundary to it
    /// (CEL-66). Both are global (shared by all faces), so two faces sharing an edge
    /// derive the identical edge vertices. `None` => no inter-face conform.
    pub cam_local: Option<Vector3>,
    pub screen_error: f32,
}

impl Patch {
    /// Upper bound on the triangles [`realize_patch`] may emit, to size the output
    /// buffer; the realized count is usually smaller (clipped to the face / hole).
    pub fn max_triangles(&self) -> usize {
        match self.kind {
            // Exact for the chains `compute_patches` builds: hole footprints
            // are clamped fully on-face and nested, so the realized annulus
            // is outer^2 - hole^2 cells-as-triangles exactly. (A face-sized
            // footprint summed naively as side^2 over-counted the hole
            // interior ~2x per chain and tripped instance-cap checks.)
            PatchKind::Hole => {
                let s = (self.outer_side / self.g) as usize;
                let hs = (self.hole_side / self.g) as usize;
                s * s - hs * hs
            }
            // A band of `depth` rows by `width` columns, two triangles per cell.
            PatchKind::Side(_) => {
                let d = (self.outer_side / self.g) as usize;
                let w = (self.outer_origin.1 / self.g as i64) as usize;
                2 * d * w
            }
        }
    }
}

/// A single realized triangle: three world-space corners and the layer colour.
#[derive(Clone, Copy, Debug)]
pub struct ClipTri {
    pub corners: [Vector3; 3],
    pub color: Color,
}

/// A **realized** patch: the triangles to render.
#[derive(Clone, Debug, Default)]
pub struct RealizedPatch {
    pub tris: Vec<ClipTri>,
}

/// **Compute the patches** for one face: the nested triangular clipmap layers,
/// coarsest first, with each layer's hole positioned around the camera's
/// projection onto the face (or the centroid when `cam_local` is `None`).
///
/// The *number* of layers is chosen by **screen size**, not fixed: layer 0 is
/// always the whole face at the fixed coarse resolution `base_cell` (so every
/// face shares one boundary resolution => watertight), then a finer layer (a
/// hole) is added only while the current layer's cells are bigger on screen than
/// `screen_error` — i.e. while `cell_world / distance_to_camera > screen_error`.
/// `num_layers` is just the maximum depth. So a face far from the camera collapses
/// to a single coarse patch, and detail piles up only where the camera is close.
///
/// Two kinds of chain, picked by where the camera's foot lands on the face:
///
/// * Foot **inside** the face — the one face the camera looks straight down on —
///   gets a [`PatchKind::Hole`] chain (`build_hole_chain`): nested triangles
///   shrinking toward the foot, detail piling up at the centre.
/// * Foot **outside** the face (every other front-facing face) gets a
///   [`PatchKind::Side`] chain (`build_side_chain`) from the edge nearest the
///   camera: bands hugging that edge, finest at the near edge and coarsening
///   inward. Layer 0 still covers the whole face (no coarse leftover); the finer
///   bands are bounded both in depth and *along* the edge by the screen-error
///   radius, so on a big planet they stay screen-sized.
///
/// Both chains size each layer by the same **screen-error** rule against the
/// distance from the camera to the **nearest on-face point on the surface** —
/// never the gnomonic foot, which for glancing faces lands almost at the camera's
/// radius and wildly underestimates (it made side faces refine *deeper* than the
/// hole face). With one metric, cells at a given distance are the same size across
/// the hole face and its side neighbours, and the hole face is always finest.
/// `follow` centres on the camera's foot; otherwise on the face centroid (a hole
/// chain, orientation ignored). `ring` is the minimum band/footprint size in
/// cells; `base_cell` is the world size of the coarsest (layer-0) cell.
///
/// `falloff` (0 = off) coarsens *distant* detail beyond the plain screen-error
/// rule: the effective error grows with the camera's distance to the face's
/// nearest surface point — `err · (1 + falloff · h / radius)` — so faraway
/// terrain spends fewer triangles. Crack-safe across faces because `h` is the
/// distance to the nearest on-face point: two faces sharing an edge measure
/// the same `h` at that edge, so their chains agree there.
///
/// `angle_falloff` (0 = off) coarsens patches that curve *away* from the camera:
/// a patch whose nearest surface point sits at central angle `θ` from the
/// sub-camera direction is tilted by ~`θ`, so its cells project ~`cos θ` smaller
/// on screen and may be that much coarser before crossing the error bound. The
/// effective error is raised by `1/cos θ` (blended by `angle_falloff`, clamped so
/// the grazing limb stays finite) — "the patch's screen size becomes a function
/// of the angle from the planet centre". Crack-safe for the same reason as
/// `falloff`: the angle is taken at the *same* shared near-surface point as `h`,
/// so adjacent faces agree along their shared edge.
pub fn compute_patches(
    frame: FaceFrame,
    num_layers: u32,
    ring: u32,
    base_cell: f32,
    screen_error: f32,
    falloff: f32,
    angle_falloff: f32,
    follow: bool,
    cam_local: Option<Vector3>,
) -> Vec<Patch> {
    let layers = num_layers.max(1);
    let ring = ring.max(2) & !1; // even and >= 2 (keeps holes lattice-aligned)
    let g0 = 1u32 << (layers - 1); // a layer-0 cell, in finest cells

    // Layer-0 (coarsest) cells along the edge — the fixed boundary resolution,
    // identical on congruent faces, so shared edges line up => watertight. The
    // finest lattice is `2^(layers-1)` times denser (the deepest a layer may go).
    let n0 = (frame.edge_len() / base_cell).round().max(1.0) as u32;
    let n_fine = n0 * g0;

    // Camera foot in barycentric coords `(u, v)`: `foot = A + u*(B-A) + v*(C-A)`.
    // With `follow`, the camera's projection onto the face (gnomonic for a sphere).
    // `None` => camera behind the face: leave it a single coarse patch. Without
    // `follow`, the centroid (a centred hole chain, orientation ignored).
    let (u, v) = if follow {
        match cam_local {
            Some(p) => match frame.project_camera(p) {
                Some(uv) => uv,
                None => return vec![solid_face(frame, n_fine, g0)],
            },
            None => (1.0 / 3.0, 1.0 / 3.0),
        }
    } else {
        (1.0 / 3.0, 1.0 / 3.0)
    };

    let inside = u >= 0.0 && v >= 0.0 && (u + v) <= 1.0;
    let mut patches = if inside {
        // The foot is on the face: nest triangular holes toward it.
        let (ti, tj) = (u * n_fine as f32, v * n_fine as f32);
        // Distance to the foot *on the rendered surface* (true altitude), so LOD
        // depends on altitude, not chart distance.
        let target_world = frame.project_bary(u, v);
        let distance = cam_local.map(|p| (p - target_world).length().max(1.0e-4));
        let err = effective_error(&frame, screen_error, falloff, distance);
        let taper = AngleTaper::new(&frame, target_world, cam_local, angle_falloff);
        build_hole_chain(frame, layers, ring, base_cell, err, distance, n_fine, g0, ti, tj, taper)
    } else {
        // The foot is off the face: refine bands from the nearest edge. The
        // distance is to the *nearest on-face point on the surface* — NOT to the
        // gnomonic foot, which for glancing faces lands almost at the camera's
        // radius and underestimates so badly that side faces refined deeper than
        // the hole face. The nearest edge (and the position along it) also says
        // where the bands grow from and where they centre.
        let (edge, along, (uc, vc)) = frame.closest_edge_point(u, v);
        let near_world = frame.project_bary(uc, vc);
        let distance = cam_local.map(|p| (p - near_world).length().max(1.0e-4));
        let tc = along * n_fine as f32;
        let err = effective_error(&frame, screen_error, falloff, distance);
        let taper = AngleTaper::new(&frame, near_world, cam_local, angle_falloff);
        build_side_chain(frame, layers, ring, base_cell, err, distance, n_fine, g0, edge, tc, taper)
    };

    // Stamp the (global) camera + raw screen error so realize can conform this
    // face's boundary to the canonical shared-edge tessellation (CEL-66) — but only
    // on patches that actually reach a face edge. Interior patches stay
    // camera-independent so a tiny camera move doesn't re-dirty them (CEL-60).
    for p in &mut patches {
        if touches_face_edge(p) {
            p.cam_local = cam_local;
            p.screen_error = screen_error;
        }
    }
    patches
}

/// Foreshortening of the screen-error reach (see [`compute_patches`] docs on
/// `angle_falloff`). A face whose centroid sits at central angle `θ` from the
/// sub-camera direction is tilted ~`θ`, so its cells project ~`cos θ` smaller on
/// screen and may be that much coarser before crossing the error bound. We scale
/// **every** layer's reach by the *same* factor `cos θ` (blended by `strength`),
/// evaluated once at the face centroid — a single constant per face, so the nested
/// rings all shrink in proportion and stay strictly ordered (no degenerate /
/// collapsing annuli, which a per-reach factor would cause by shrinking the big
/// coarse rings more than the small fine ones). Faces curving away from the camera
/// thus refine to a shallower depth — fewer triangles where the surface is
/// foreshortened or over the limb. Crack-safe like `falloff`: the factor only
/// changes a face's interior refinement *depth*, never its layer-0 edge resolution
/// (always `base_cell`), which is what neighbouring faces meet along.
#[derive(Clone, Copy)]
struct AngleTaper {
    scale: f32, // constant reach multiplier in (0, 1]; 1 = off
}

impl AngleTaper {
    fn new(frame: &FaceFrame, _near_world: Vector3, cam: Option<Vector3>, strength: f32) -> Self {
        let scale = match cam {
            Some(c) if strength > 0.0 && frame.radius > 0.0 => {
                let centroid = (frame.a + frame.b + frame.c).normalized();
                let cos_t = centroid.dot(c.normalized()).clamp(-1.0, 1.0);
                // scale = 1 - strength*(1 - cos θ); floored so a face never gets a
                // non-positive reach (occluded faces collapse to layer 0 anyway).
                (1.0 - strength * (1.0 - cos_t)).max(0.02)
            }
            _ => 1.0,
        };
        AngleTaper { scale }
    }

    /// Multiply a layer's screen-error reach threshold by this constant. Same for
    /// every layer of the face, so the rings stay proportionally ordered.
    fn reach_scale(&self) -> f32 {
        self.scale
    }
}

/// Distance-biased screen error (see [`compute_patches`] docs on `falloff`).
fn effective_error(frame: &FaceFrame, screen_error: f32, falloff: f32, distance: Option<f32>) -> f32 {
    match distance {
        Some(h) if falloff > 0.0 => {
            let norm = if frame.radius > 0.0 { frame.radius } else { frame.edge_len() };
            screen_error * (1.0 + falloff * h / norm)
        }
        _ => screen_error,
    }
}

/// Conservatively true when no point of this face can be visible from `cam`:
/// the whole face lies beyond the horizon of a shrunken occluder sphere
/// (valleys, 0.85·R) even with surface points raised to peak height (1.35·R).
/// Margins are generous so terrain never pops at the horizon; the caller can
/// then spend a single coarse patch on the face instead of a full chain.
pub fn face_beyond_horizon(frame: &FaceFrame, cam: Vector3) -> bool {
    let radius = frame.radius;
    if radius <= 0.0 {
        return false;
    }
    let cam_len = cam.length();
    if cam_len <= radius * 1.001 {
        return false; // at/under the surface: keep everything
    }
    // Sub-camera point on the face => trivially visible.
    if let Some((u, v)) = frame.project_camera(cam) {
        if u >= 0.0 && v >= 0.0 && u + v <= 1.0 {
            return false;
        }
    }
    let r_occ = radius * 0.85;
    let r_peak = radius * 1.35;
    let max_angle =
        (r_occ / cam_len).clamp(-1.0, 1.0).acos() + (r_occ / r_peak).clamp(-1.0, 1.0).acos();
    let cam_dir = cam / cam_len;
    // Sampled minimum angle over the face (corners + edge midpoints); the
    // occluder margins dwarf the sampling error on icosahedron-sized faces.
    let (a, b, c) = (frame.a.normalized(), frame.b.normalized(), frame.c.normalized());
    let samples = [
        a,
        b,
        c,
        (a + b).normalized(),
        (b + c).normalized(),
        (c + a).normalized(),
    ];
    let cos_best = samples.iter().map(|d| d.dot(cam_dir)).fold(f32::MIN, f32::max);
    cos_best.clamp(-1.0, 1.0).acos() > max_angle
}

/// Nested triangular [`PatchKind::Hole`] layers centred on the foot `(ti, tj)`
/// (finest cells), coarsest first. Each finer layer grows from `ring` cells (far)
/// to filling its parent (near) by the screen-error rule, leaving a hole the next
/// layer fills. Footprints are NOT clamped to the face — `realize_patch` clips
/// them — so a layer can sit partly off the face. Stops when a footprint no longer
/// overlaps the face, the screen-size test is met, or the triangle budget is hit.
fn build_hole_chain(
    frame: FaceFrame,
    layers: u32,
    ring: u32,
    base_cell: f32,
    screen_error: f32,
    distance: Option<f32>,
    n_fine: u32,
    g0: u32,
    ti: f32,
    tj: f32,
    taper: AngleTaper,
) -> Vec<Patch> {
    let nf = n_fine as i64;
    let cell_finest = base_cell / g0 as f32; // world size of a finest cell
    let mut fps: Vec<((i64, i64), i64)> = vec![((0, 0), nf)];
    for l in 1..layers {
        let g = 1i64 << (layers - 1 - l);
        let parent_g = g * 2; // = g_{l-1}, the grid this footprint snaps to
        let (po, pside) = fps[(l - 1) as usize];

        // Footprint side (finest cells): at least `ring` cells, else sized by screen
        // error so it grows from small (far) to filling the parent (near). The
        // parent cell is too coarse within `d_thresh = parent_cell/screen_error`;
        // at height `h` that's an in-plane disk of radius `r = sqrt(d_thresh^2-h^2)`,
        // and the triangle's incircle is sized to cover it.
        let mut want = (ring as i64) * g;
        if let Some(h) = distance {
            let parent_cell = base_cell / (1u32 << (l - 1)) as f32;
            let d_thresh0 = parent_cell / screen_error;
            if h >= d_thresh0 {
                break;
            }
            // Foreshorten the reach by the face's tilt: faces curving away from
            // the camera refine shallower (same constant for every layer).
            let d_thresh = d_thresh0 * taper.reach_scale();
            if h >= d_thresh {
                break; // foreshortened away entirely => stop refining here
            }
            let r = (d_thresh * d_thresh - h * h).sqrt();
            let screen_side = (2.0 * 3f32.sqrt() * r / cell_finest) as i64;
            want = want.max(screen_side);
        }
        let mut side = want.min(pside);
        side -= side.rem_euclid(parent_g);
        if side < parent_g {
            break;
        }

        // Safety budget on the *actual* footprint: CLAMP the side instead of
        // breaking the chain. Breaking silently froze refinement at a coarse
        // layer on big planets (n_fine ~8k at screen_error 0.01 trips this at
        // mid layers, leaving the ground stuck at the last layer's colour);
        // clamping keeps nesting toward the camera and only degrades the
        // band between the clamped footprint edge and the parent's coverage
        // (cells there can reach ~2x the screen target).
        let max_side = ((MAX_FACE_TRIS as f64).sqrt() as i64) * g;
        if side > max_side {
            side = max_side - max_side.rem_euclid(parent_g);
            if side < parent_g {
                break;
            }
        }

        // Centre on the foot (a triangle's centroid is `side/3` from its origin),
        // snapping the origin to the parent grid.
        let snap = |t: f32| (((t - side as f32 / 3.0) / parent_g as f32).round() as i64) * parent_g;
        let (mut oi, mut oj) = (snap(ti), snap(tj));

        // Nest EVERY layer inside its parent (layer 1's parent is the whole
        // face) so annuli tile exactly and nothing is silently uncovered. Hole
        // chains only run when the foot is INSIDE the face, so clamping loses
        // nothing — without it, a face-sized footprint centred on a foot near an
        // edge shifts off the face and the leftover renders at coarse layer 0
        // right next to the camera (the recurring "red patch").
        oi = oi.clamp(po.0, po.0 + pside - side);
        oj = oj.clamp(po.1, po.1 + pside - side);
        let over = (oi + oj) - (po.0 + po.1 + pside - side);
        if over > 0 {
            let trim_x = over.min(oi - po.0);
            oi -= trim_x;
            oj -= over - trim_x;
        }

        if !overlaps_face(oi, oj, side, nf) {
            break;
        }
        fps.push(((oi, oj), side));
    }

    fps.iter()
        .enumerate()
        .map(|(l, &(origin, side))| {
            let (hole_origin, hole_side) =
                fps.get(l + 1).map(|&(o, s)| (o, s as u32)).unwrap_or(((0, 0), 0));
            Patch {
                layer: l as u32,
                n_fine,
                g: 1u32 << (layers - 1 - l as u32),
                outer_origin: origin,
                outer_side: side as u32,
                hole_origin,
                hole_side,
                frame,
                kind: PatchKind::Hole,
                cam_local: None,
                screen_error: 0.0,
            }
        })
        .collect()
}

/// Nested [`PatchKind::Side`] bands from `edge`, coarsest first. Each band is a
/// `(depth, along_start, along_width)` region (finest cells): depth measured from
/// the edge, the along-interval centred on `tc` — the position along the edge
/// nearest the camera. Layer 0 covers the whole face; each finer band is sized by
/// the screen-error rule (the half-disk of chart radius `r` around the near point
/// needs finer cells => band of depth `r`, width `2r`), so the finest cells hug
/// the point on the edge closest to the camera and both dimensions stay
/// screen-bounded — on a big planet a band never spans the whole face edge.
fn build_side_chain(
    frame: FaceFrame,
    layers: u32,
    ring: u32,
    base_cell: f32,
    screen_error: f32,
    distance: Option<f32>,
    n_fine: u32,
    g0: u32,
    edge: u8,
    tc: f32,
    taper: AngleTaper,
) -> Vec<Patch> {
    let nf = n_fine as i64;
    let cell_finest = base_cell / g0 as f32;
    // (depth, along_start, along_width) per layer, finest cells. Layer 0 = face.
    let mut bands: Vec<(i64, i64, i64)> = vec![(nf, 0, nf)];
    for l in 1..layers {
        let g = 1i64 << (layers - 1 - l);
        let parent_g = g * 2;
        let (pdepth, pstart, pwidth) = bands[(l - 1) as usize];

        // Same screen rule as the hole chain, against the nearest on-face point:
        // the parent cell is too coarse within `d_thresh` of the camera; at
        // distance `h` from the near point that is a chart disk of radius `r`
        // around it, whose on-face half is covered by a band of depth `r` and
        // width `2r` centred on the near point.
        let mut want_d = (ring as i64) * g;
        let mut want_w = 2 * (ring as i64) * g;
        if let Some(h) = distance {
            let parent_cell = base_cell / (1u32 << (l - 1)) as f32;
            let d_thresh0 = parent_cell / screen_error;
            if h >= d_thresh0 {
                break;
            }
            // Foreshorten the reach by the face's tilt: a face curving away from
            // the camera refines shallower (same constant for every band).
            let d_thresh = d_thresh0 * taper.reach_scale();
            if h >= d_thresh {
                break;
            }
            let r = (d_thresh * d_thresh - h * h).sqrt();
            // Capped at the face size: bands are clamped to their parent anyway,
            // and an unbounded radius (screen_error -> 0) must not overflow.
            let r_cells = ((r / cell_finest) as i64).min(nf);
            want_d = want_d.max(r_cells);
            want_w = want_w.max(2 * r_cells);
        }
        let mut depth = want_d.min(pdepth);
        depth -= depth.rem_euclid(parent_g);
        if depth < parent_g {
            break;
        }
        let mut width = want_w.min(pwidth);
        width -= width.rem_euclid(parent_g);
        if width < parent_g {
            break;
        }

        // Safety budget on the actual band: clamp depth and width together
        // (see the hole chain's twin check — break would freeze refinement).
        let band_cells = 2 * (depth / g) * (width / g);
        if band_cells as usize > MAX_FACE_TRIS {
            let f = (MAX_FACE_TRIS as f64 / band_cells as f64).sqrt();
            depth = ((depth as f64 * f) as i64 / parent_g) * parent_g;
            width = ((width as f64 * f) as i64 / parent_g) * parent_g;
            if depth < parent_g || width < parent_g {
                break;
            }
        }

        // Centre the band on the near point along the edge, snapped to the parent
        // grid and kept inside the parent's own along-range so bands nest.
        let mut start = (((tc - width as f32 / 2.0) / parent_g as f32).round() as i64) * parent_g;
        start = start.clamp(pstart, pstart + pwidth - width);
        bands.push((depth, start, width));
    }

    bands
        .iter()
        .enumerate()
        .map(|(l, &(depth, start, width))| {
            let (hole_origin, hole_side) = bands
                .get(l + 1)
                .map(|&(d, s, w)| ((s, w), d as u32))
                .unwrap_or(((0, 0), 0));
            Patch {
                layer: l as u32,
                n_fine,
                g: 1u32 << (layers - 1 - l as u32),
                outer_origin: (start, width),
                outer_side: depth as u32,
                hole_origin,
                hole_side,
                frame,
                kind: PatchKind::Side(edge),
                cam_local: None,
                screen_error: 0.0,
            }
        })
        .collect()
}

/// **Realize** one patch into world-space triangles. Pure function of `patch` —
/// safe to run on every patch in parallel. Dispatches by [`PatchKind`]: a hole
/// patch nests toward a point, a side patch lays a full-width band from an edge.
pub fn realize_patch(patch: &Patch, color: Color) -> RealizedPatch {
    let mut realized = match patch.kind {
        PatchKind::Hole => realize_hole(patch, color),
        PatchKind::Side(edge) => realize_side(patch, edge, color),
    };
    conform_to_face_edges(patch, &mut realized.tris);
    realized
}

/// Arc-parameter `t` in `[0, 1]` of world point `p` along the face edge from `a` to
/// `b`, or `None` if `p` is not on that edge. Sphere edges are great-circle arcs
/// (matching `project_bary`); flat edges are straight. The tolerance is relative to
/// the edge length so it scales with the face.
fn arc_param(p: Vector3, a: Vector3, b: Vector3, radius: f32) -> Option<f32> {
    if radius > 0.0 {
        let (ua, ub, up) = (a.normalized(), b.normalized(), p.normalized());
        let n = ua.cross(ub);
        let nl = n.length();
        if nl < 1.0e-6 {
            return None;
        }
        if (up.dot(n / nl)).abs() > 1.0e-4 {
            return None; // off the edge's great circle
        }
        let omega = ua.dot(ub).clamp(-1.0, 1.0).acos();
        let t = ua.dot(up).clamp(-1.0, 1.0).acos() / omega;
        // On-arc only: the range clamp below rejects points past either endpoint.
        (-1.0e-3..=1.0 + 1.0e-3).contains(&t).then(|| t.clamp(0.0, 1.0))
    } else {
        let ab = b - a;
        let len2 = ab.length_squared();
        if len2 < 1.0e-12 {
            return None;
        }
        let t = (p - a).dot(ab) / len2;
        let on_line = (a + ab * t - p).length();
        (on_line < 1.0e-3 * len2.sqrt() && (-1.0e-3..=1.0 + 1.0e-3).contains(&t))
            .then(|| t.clamp(0.0, 1.0))
    }
}

/// Whether [`conform_to_face_edges`] could ever modify this patch: its footprint
/// must reach one of the three face boundary edges (lattice lines `J=0`, `I=0`,
/// `I+J=n_fine`). Only such patches need the camera stamped for the CEL-66 conform;
/// an interior patch's geometry is camera-independent, so leaving its `cam_local`
/// unset keeps it out of the per-frame dirty set (CEL-60 budgeting) — a tiny camera
/// move no longer re-dirties every patch.
///
/// A conservative superset of "conform actually changes something": it returns
/// `true` whenever a boundary edge is reachable even if the canonical tessellation
/// happens to add no vertices. Under-stamping would reopen the seam, so the
/// watertightness test (`hole_side_shared_edge_is_watertight`, which realizes via
/// the gated `compute_patches`) guards the dangerous direction.
fn touches_face_edge(patch: &Patch) -> bool {
    let nf = patch.n_fine as i64;
    match patch.kind {
        PatchKind::Hole => {
            // Outer triangle origin/side in finest cells: it reaches `J=0` when the
            // origin sits at or below that row, `I=0` symmetrically, and the
            // hypotenuse `I+J=n_fine` when origin+side spans out to it.
            let (oi, oj) = patch.outer_origin;
            let s = patch.outer_side as i64;
            oi <= 0 || oj <= 0 || oi + oj + s >= nf
        }
        // A side band hugs a face edge by construction: its depth-0 cells lie on the
        // named edge, and its end columns (`al = 0` / `al = n`) lie on the two
        // perpendicular edges at the face corners. Enumerating which it reaches per
        // edge index is fiddly and a false negative reopens the seam, so always stamp
        // side patches — they are the boundary bands that legitimately conform. The
        // CEL-60 budgeting we protect is the hole chain's interior annuli (below).
        PatchKind::Side(_) => true,
    }
}

/// Conform this patch's triangles that touch a **face boundary edge** to the
/// canonical [`edge_tessellation`] of that edge: split each boundary triangle's
/// edge to insert the canonical vertices missing between its endpoints (fanning the
/// opposite corner). Because both faces sharing an edge derive the *same* canonical
/// vertices — and the patch's own boundary vertices are a subset of them — both
/// faces end up with the identical edge vertex set, so the inter-face seam is
/// watertight (CEL-66). A no-op unless `cam_local` is set.
fn conform_to_face_edges(patch: &Patch, tris: &mut Vec<ClipTri>) {
    let Some(cam) = patch.cam_local else { return };
    if patch.screen_error <= 0.0 {
        return;
    }
    let f = patch.frame;
    let n_fine = patch.n_fine;
    let g0 = patch.g << patch.layer; // layer-0 cell size (finest cells)
    let n0 = n_fine / g0;
    // Canonical vertices on each of the three face edges, as (t, world), sorted.
    let edges = [(f.a, f.b), (f.b, f.c), (f.c, f.a)];
    let canon: Vec<Vec<(f32, Vector3)>> = edges
        .iter()
        .map(|&(a, b)| {
            edge_tessellation(a, b, f.radius, cam, patch.screen_error, n0, g0)
                .into_iter()
                .map(|p| (p as f32 / n_fine as f32, edge_point(a, b, f.radius, p, n_fine)))
                .collect()
        })
        .collect();

    let mut out = Vec::with_capacity(tris.len());
    for t in tris.drain(..) {
        conform_triangle(t, &f, &edges, &canon, &mut out, 0);
    }
    *tris = out;
}

/// Recursively conform one triangle: if one of its edges lies on a face edge and the
/// canonical tessellation has vertices strictly between its endpoints, fan the
/// opposite corner across those vertices and recurse (a triangle at a face corner
/// can touch two boundary edges). Otherwise emit it unchanged.
fn conform_triangle(
    t: ClipTri,
    f: &FaceFrame,
    edges: &[(Vector3, Vector3); 3],
    canon: &[Vec<(f32, Vector3)>],
    out: &mut Vec<ClipTri>,
    depth: u32,
) {
    if depth < 4 {
        let c = t.corners;
        for (i, &(ea, eb)) in edges.iter().enumerate() {
            for k in 0..3 {
                let (ci, cj, ck) = (c[k], c[(k + 1) % 3], c[(k + 2) % 3]);
                let (Some(ti), Some(tj)) = (arc_param(ci, ea, eb, f.radius), arc_param(cj, ea, eb, f.radius)) else {
                    continue;
                };
                let (lo, hi) = (ti.min(tj), ti.max(tj));
                let eps = 1.0e-5;
                let between: Vec<Vector3> = canon[i]
                    .iter()
                    .filter(|&&(t, _)| t > lo + eps && t < hi - eps)
                    .map(|&(_, w)| w)
                    .collect();
                if between.is_empty() {
                    continue;
                }
                // Fan apex ck across [ci, between.., cj] in ci->cj order.
                let mut seq = Vec::with_capacity(between.len() + 2);
                seq.push(ci);
                if ti <= tj {
                    seq.extend(between);
                } else {
                    seq.extend(between.into_iter().rev());
                }
                seq.push(cj);
                for w in seq.windows(2) {
                    conform_triangle(
                        ClipTri { corners: [ck, w[0], w[1]], color: t.color },
                        f, edges, canon, out, depth + 1,
                    );
                }
                return;
            }
        }
    }
    out.push(t);
}

/// Triangulation of one **parent** (coarse) up-triangle's four fine sub-cells when
/// some of its three edges are *stitched* down to the parent resolution to close a
/// t-junction. Each `[usize; 3]` indexes the six lattice points
/// `[P0, P1, P2, M01, M12, M02]` — the three corners then the three edge midpoints
/// (`M01` on `P0`-`P1`, `M12` on `P1`-`P2`, `M02` on `P2`-`P0`). `mask` bit 0 =
/// stitch edge `P0`-`P1` (drop `M01`), bit 1 = stitch `P1`-`P2` (drop `M12`), bit 2
/// = stitch `P2`-`P0` (drop `M02`).
///
/// A stitched edge becomes a single coarse segment (its midpoint vertex is never
/// emitted, so its terrain height is never sampled); unstitched edges keep their
/// midpoint and stay fine, matching the finer interior. Every entry is wound CCW in
/// lattice space (same sense as the regular up-cell) and is non-degenerate. Mask 0
/// is the regular 4-triangle subdivision; mask 7 collapses to the lone coarse
/// triangle. This is the shared primitive mirrored by the GPU `RealizePatch.slang`.
const STITCH_TRIS: [&[[usize; 3]]; 8] = [
    &[[0, 3, 5], [3, 1, 4], [5, 4, 2], [3, 4, 5]], // 0: none (regular subdivision)
    &[[0, 1, 4], [0, 4, 5], [5, 4, 2]],            // 1: drop M01
    &[[0, 3, 5], [3, 1, 2], [3, 2, 5]],            // 2: drop M12
    &[[1, 2, 5], [1, 5, 0]],                       // 3: drop M01, M12
    &[[0, 3, 4], [3, 1, 4], [0, 4, 2]],            // 4: drop M02
    &[[0, 1, 4], [0, 4, 2]],                       // 5: drop M01, M02
    &[[2, 0, 3], [2, 3, 1]],                       // 6: drop M12, M02
    &[[0, 1, 2]],                                  // 7: all coarse (lone triangle)
];

/// Realize a [`PatchKind::Hole`] patch: walk the layer's triangular lattice in
/// **parent (coarse) cells**, skip cells inside the hole, and emit each cell's
/// fine triangles. The three edges of the outer triangle border the next-coarser
/// layer (always exactly 2x), so along them the outermost row is *stitched* (the
/// odd midpoint vertex dropped, [`STITCH_TRIS`]) to stay watertight with that
/// coarser neighbour once terrain height is applied. Layer 0's outer edge is the
/// face edge — shared with a neighbour face at equal resolution — so it is never
/// stitched.
///
/// For hole chains the outer triangle is always fully on-face (the footprint is
/// clamped inside its parent, recursively inside the face), so no face clipping is
/// needed; only the hole is carved out. The inner (hole) edge is rendered at this
/// layer's resolution and is *not* stitched — the finer child inside handles its
/// own outer boundary against this layer.
fn realize_hole(patch: &Patch, color: Color) -> RealizedPatch {
    let f = patch.frame;
    let n = patch.n_fine as f32;
    let g = patch.g as i64;
    let (oi, oj) = patch.outer_origin;
    // Lattice cell (ii, jj) of this layer -> world position. The fine indices give
    // barycentric weights (toward B and C); `project_bary` maps them onto the
    // sphere by equal-angle slerp (or flat for a non-sphere face).
    let vert = |ii: i64, jj: i64| f.project_bary((oi + ii * g) as f32 / n, (oj + jj * g) as f32 / n);

    let nf = patch.n_fine as i64;
    let side = (patch.outer_side as i64) / g; // even: outer_side snaps to the parent grid
    let side_p = side / 2; // parent (coarse) cells along the edge
    let stitch = patch.layer >= 1; // layer 0's outer edge is the face edge (no coarser parent)

    // Hole, expressed in this layer's cells (offset from the outer origin).
    let has_hole = patch.hole_side > 0;
    let hci = (patch.hole_origin.0 - oi) / g;
    let hcj = (patch.hole_origin.1 - oj) / g;
    let hs = (patch.hole_side as i64) / g;
    let in_hole = |ii: i64, jj: i64, slack: i64| {
        has_hole && ii >= hci && jj >= hcj && (ii - hci) + (jj - hcj) <= hs - slack
    };
    // A lattice *vertex* that lies strictly inside the hole is never shaded (the
    // child fills it); the stitch must not reference one. Exactly mirrors the skip
    // in `RealizeVertices.slang::shade_hole_vertex`.
    let hole_skip = |ii: i64, jj: i64| {
        has_hole && ii - hci >= 1 && jj - hcj >= 1 && (ii - hci) + (jj - hcj) <= hs - 1
    };

    let mut tris = Vec::with_capacity(patch.max_triangles());
    let push = |out: &mut Vec<ClipTri>, a: (i64, i64), b: (i64, i64), c: (i64, i64)| {
        out.push(ClipTri { corners: [vert(a.0, a.1), vert(b.0, b.1), vert(c.0, c.1)], color });
    };
    // Regular fine up/down cells, clipped to the face (the footprint may sit partly
    // off the face when `build_hole_chain` pushed the origin negative) and to the
    // outer hypotenuse, with the hole carved out.
    let fine_up = |out: &mut Vec<ClipTri>, ii: i64, jj: i64| {
        let (i_f, j_f) = (oi + ii * g, oj + jj * g);
        if i_f >= 0 && j_f >= 0 && ii + jj <= side - 1 && i_f + j_f + g <= nf && !in_hole(ii, jj, 1) {
            push(out, (ii, jj), (ii + 1, jj), (ii, jj + 1));
        }
    };
    let fine_down = |out: &mut Vec<ClipTri>, ii: i64, jj: i64| {
        let (i_f, j_f) = (oi + ii * g, oj + jj * g);
        if i_f >= 0 && j_f >= 0 && ii + jj <= side - 2 && i_f + j_f + 2 * g <= nf && !in_hole(ii, jj, 2) {
            push(out, (ii + 1, jj), (ii + 1, jj + 1), (ii, jj + 1));
        }
    };

    // Layer 0 never stitches (its outer edge is the face edge) and its `side`
    // (= n_fine / g0) is not parent-aligned, so iterate fine cells directly — the
    // parent-cell walk below assumes an even `side`.
    if !stitch {
        for jj in 0..side {
            for ii in 0..(side - jj) {
                fine_up(&mut tris, ii, jj);
                fine_down(&mut tris, ii, jj);
            }
        }
        return RealizedPatch { tris };
    }

    for jp in 0..side_p {
        for ip in 0..(side_p - jp) {
            let (bi, bj) = (2 * ip, 2 * jp); // this-layer origin of the parent cell
            // Outer-triangle edges this parent up-cell sits on (only when stitching):
            // E0 = bottom (jj=0), E1 = hypotenuse (ip+jp == side_p-1), E2 = left (ii=0).
            let on_e0 = stitch && jp == 0;
            let on_e1 = stitch && ip + jp == side_p - 1;
            let on_e2 = stitch && ip == 0;
            // If any of the six stitch points would land strictly inside the hole
            // (an unshaded vertex), fall back to fine emission (which provably never
            // references a skipped vertex). Usually the hole is nested far inside.
            let hole_here = hole_skip(bi, bj)
                || hole_skip(bi + 2, bj)
                || hole_skip(bi, bj + 2)
                || hole_skip(bi + 1, bj)
                || hole_skip(bi + 1, bj + 1)
                || hole_skip(bi, bj + 1);
            // A clipped parent cell (footprint off the face) borders the face edge,
            // not the coarser parent, so stitch only fully on-face parent cells; the
            // clipped remainder falls back to per-cell face clipping.
            let (i0, j0) = (oi + bi * g, oj + bj * g);
            let clip_here = i0 < 0 || j0 < 0 || i0 + j0 + 2 * g > nf;
            if (on_e0 || on_e1 || on_e2) && !hole_here && !clip_here {
                let mask = on_e0 as usize | (on_e1 as usize) << 1 | (on_e2 as usize) << 2;
                let pts = [
                    (bi, bj),         // P0
                    (bi + 2, bj),     // P1
                    (bi, bj + 2),     // P2
                    (bi + 1, bj),     // M01
                    (bi + 1, bj + 1), // M12
                    (bi, bj + 1),     // M02
                ];
                for t in STITCH_TRIS[mask] {
                    push(&mut tris, pts[t[0]], pts[t[1]], pts[t[2]]);
                }
            } else {
                // Interior (or hole-adjacent) parent up-cell: regular subdivision.
                fine_up(&mut tris, bi, bj);
                fine_up(&mut tris, bi + 1, bj);
                fine_up(&mut tris, bi, bj + 1);
                fine_down(&mut tris, bi, bj);
            }
            // Parent down-cell — always interior to the annulus, never on an outer
            // edge, so always the regular four fine triangles.
            if ip + jp <= side_p - 2 {
                fine_up(&mut tris, bi + 1, bj + 1);
                fine_down(&mut tris, bi + 1, bj);
                fine_down(&mut tris, bi, bj + 1);
                fine_down(&mut tris, bi + 1, bj + 1);
            }
        }
    }
    RealizedPatch { tris }
}

/// Realize a [`PatchKind::Side`] patch: the band from `edge` covering depth
/// `[0, outer_side)` and the along-edge interval `[start, start+width)`, minus
/// the hole band (the next finer layer's region). Cells are addressed as
/// `(depth, along)` and mapped to the lattice per edge; an up/down pair at depth
/// `d` tiles exactly the strip between the lattice lines `d` and `d+1` parallel
/// to the edge, and `along` cuts are straight lattice lines too — so parent and
/// child bands tile with no gap or overlap. `edge`: `0` = AB (`J=0`), `1` = BC
/// (hypotenuse `I+J=n`), `2` = CA (`I=0`).
fn realize_side(patch: &Patch, edge: u8, color: Color) -> RealizedPatch {
    let f = patch.frame;
    let n = patch.n_fine as f32;
    let g = patch.g as i64;
    let np = (patch.n_fine as i64) / g; // cells along the edge at this layer
    let od = (patch.outer_side as i64) / g; // outer band depth, this-layer cells
    let (os, ow) = (patch.outer_origin.0 / g, patch.outer_origin.1 / g);
    let hd = (patch.hole_side as i64) / g; // hole band depth
    let (hs, hw) = (patch.hole_origin.0 / g, patch.hole_origin.1 / g);
    let has_hole = patch.hole_side > 0;
    let in_hole = |d: i64, al: i64| has_hole && d < hd && al >= hs && al < hs + hw;

    let vert = |ii: i64, jj: i64| f.project_bary((ii * g) as f32 / n, (jj * g) as f32 / n);

    let mut tris = Vec::with_capacity(patch.max_triangles());
    for d in 0..od {
        for al in os.max(0)..(os + ow).min(np) {
            if in_hole(d, al) {
                continue;
            }
            // Up triangle of the (d, al) cell, if it's on-face.
            let up = match edge {
                0 => (al + d <= np - 1).then_some((al, d)),
                2 => (al + d <= np - 1).then_some((d, al)),
                _ => (al <= np - 1 - d).then_some((al, np - 1 - d - al)),
            };
            if let Some((ii, jj)) = up {
                tris.push(ClipTri {
                    corners: [vert(ii, jj), vert(ii + 1, jj), vert(ii, jj + 1)],
                    color,
                });
            }
            // Down (interior) triangle — needs one more row of room.
            let down = match edge {
                0 => (al + d <= np - 2).then_some((al, d)),
                2 => (al + d <= np - 2).then_some((d, al)),
                _ => (np - 2 - d >= 0 && al <= np - 2 - d).then_some((al, np - 2 - d - al)),
            };
            if let Some((ii, jj)) = down {
                tris.push(ClipTri {
                    corners: [vert(ii + 1, jj), vert(ii + 1, jj + 1), vert(ii, jj + 1)],
                    color,
                });
            }
        }
    }
    RealizedPatch { tris }
}

/// Realize every patch **independently and in parallel**, one thread per patch.
/// Returns the realized patches in the same order as the input.
pub fn realize_all(patches: &[Patch]) -> Vec<RealizedPatch> {
    thread::scope(|scope| {
        let handles: Vec<_> = patches
            .iter()
            .map(|patch| {
                let color = layer_color(patch.layer);
                scope.spawn(move || realize_patch(patch, color))
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    })
}

/// A distinct, high-contrast hue per clipmap layer so adjacent rings are easy to
/// tell apart. HSV is converted by hand (rather than `Color::from_hsv`, which
/// needs the live engine) so this stays a pure function usable in tests.
pub fn layer_color(layer: u32) -> Color {
    let hue = (layer as f32 * 0.137).fract();
    let (s, v) = (0.8_f32, 0.95_f32);
    let h6 = hue * 6.0;
    let c = v * s;
    let x = c * (1.0 - (h6 % 2.0 - 1.0).abs());
    let m = v - c;
    let (r, g, b) = match h6 as u32 {
        0 => (c, x, 0.0),
        1 => (x, c, 0.0),
        2 => (0.0, c, x),
        3 => (0.0, x, c),
        4 => (x, 0.0, c),
        _ => (c, 0.0, x),
    };
    Color::from_rgba(r + m, g + m, b + m, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    const LAYERS: u32 = 3;
    const RING: u32 = 4;
    const BASE: f32 = 0.25;

    // A face lifted off the origin with its outward normal pointing +Y, so the
    // planet centre (origin) sits below it — like a real icosahedron face.
    fn frame() -> FaceFrame {
        FaceFrame {
            a: Vector3::new(-10.0, 10.0, -6.0),
            b: Vector3::new(0.0, 10.0, 12.0),
            c: Vector3::new(10.0, 10.0, -6.0),
            radius: 0.0,
        }
    }

    /// One icosahedron face as a gnomonic sphere chart of the given radius.
    fn sphere_frame(radius: f32) -> FaceFrame {
        let f = crate::icosphere::BASE_FACES[0];
        FaceFrame {
            a: crate::icosphere::base_vertex(f[0]) * radius,
            b: crate::icosphere::base_vertex(f[1]) * radius,
            c: crate::icosphere::base_vertex(f[2]) * radius,
            radius,
        }
    }

    // ---- canonical edge tessellation (CEL-66) ----

    /// The canonical edge tessellation is a pure function of the global edge: both
    /// endpoint orderings must yield the SAME set of world vertices (mirrored
    /// finest positions), it must include both ends, and stay lattice-aligned.
    #[test]
    fn edge_tessellation_is_orientation_independent() {
        let r = 100.0;
        let a = crate::icosphere::base_vertex(0) * r;
        let b = crate::icosphere::base_vertex(1) * r;
        let cam = Vector3::new(0.51, 147.655, -39.09);
        let (se, n0, g0) = (0.03f32, 26u32, 1u32 << 7); // layers 8
        let n_fine = n0 * g0;

        let ab = edge_tessellation(a, b, r, cam, se, n0, g0);
        let ba = edge_tessellation(b, a, r, cam, se, n0, g0);

        assert_eq!(ab.first(), Some(&0));
        assert_eq!(ab.last(), Some(&n_fine));
        assert!(ab.windows(2).all(|w| w[1] > w[0]), "strictly increasing");
        // Each split point is lattice-aligned within a base cell (multiple of a
        // power-of-two divisor of g0) — trivially true since we only ever halve.

        // Orientation independence: position p from A is position (n_fine - p) from
        // B; the two sets must be mirror images (same world vertices).
        let ba_mirrored: std::collections::BTreeSet<u32> = ba.iter().map(|&p| n_fine - p).collect();
        let ab_set: std::collections::BTreeSet<u32> = ab.iter().copied().collect();
        assert_eq!(ab_set, ba_mirrored, "edge tessellation must not depend on endpoint order");

        // Determinism.
        assert_eq!(ab, edge_tessellation(a, b, r, cam, se, n0, g0));
        // Adaptivity: a near-surface off-centre camera refines below base_cell.
        assert!(ab.len() > n0 as usize + 1, "expected refinement beyond base cells");
    }

    // ---- inter-face shared-edge watertightness (CEL-66) ----

    fn face_frame(face: usize, radius: f32) -> FaceFrame {
        let f = crate::icosphere::BASE_FACES[face];
        FaceFrame {
            a: crate::icosphere::base_vertex(f[0]) * radius,
            b: crate::icosphere::base_vertex(f[1]) * radius,
            c: crate::icosphere::base_vertex(f[2]) * radius,
            radius,
        }
    }

    /// The two global icosphere vertex indices shared by faces `i` and `j`, if they
    /// share an edge (exactly two vertices in common); else `None`.
    fn shared_edge(i: usize, j: usize) -> Option<(u32, u32)> {
        let fi = crate::icosphere::BASE_FACES[i];
        let fj = crate::icosphere::BASE_FACES[j];
        let common: Vec<u32> = fi.iter().copied().filter(|v| fj.contains(v)).collect();
        if common.len() == 2 {
            // Order by global index so both faces parameterise the edge the same way.
            Some((common[0].min(common[1]), common[0].max(common[1])))
        } else {
            None
        }
    }

    /// The set of normalized arc-parameters `t` at which a face's realized patches
    /// place a vertex on the global edge `(va, vb)` (quantised). `t` runs from the
    /// lower-index endpoint, so two faces sharing the edge produce comparable sets.
    fn edge_vertex_params(patches: &[Patch], radius: f32, va: u32, vb: u32) -> std::collections::BTreeSet<i64> {
        let ua = crate::icosphere::base_vertex(va).normalized();
        let ub = crate::icosphere::base_vertex(vb).normalized();
        let n = ua.cross(ub).normalized();
        let omega = ua.dot(ub).clamp(-1.0, 1.0).acos();
        let mut set = std::collections::BTreeSet::new();
        for p in patches {
            for t in realize_patch(p, Color::WHITE).tris {
                for c in t.corners {
                    let up = c.normalized();
                    // On the edge's great circle (in the ua,ub plane)…
                    if up.dot(n).abs() > 1.0e-4 {
                        continue;
                    }
                    // …and within the arc (positive barycentric on both ends).
                    let da = up.dot(ua).clamp(-1.0, 1.0);
                    let db = up.dot(ub).clamp(-1.0, 1.0);
                    let cos_w = omega.cos();
                    if da < cos_w - 1.0e-4 || db < cos_w - 1.0e-4 {
                        continue;
                    }
                    let t = da.acos() / omega; // 0 at va, 1 at vb
                    set.insert((t * 1.0e5).round() as i64);
                    let _ = radius;
                }
            }
        }
        set
    }

    /// CEL-66 bug confirmation (TDD red): a hole face (camera foot) and an
    /// edge-sharing side neighbour must place the SAME vertices on their shared
    /// edge, or terrain height opens a crack there. Pre-fix the side refines that
    /// edge (its near edge) finer than the hole face's coarse boundary, so the sets
    /// differ and this fails; the fix makes both derive the same edge tessellation.
    #[test]
    fn hole_side_shared_edge_is_watertight() {
        let r = 100.0;
        // An OFF-CENTRE near-surface camera (the v3 saved camera) — the foot lands
        // off the hole face's centre, so the hole face's footprints don't reach its
        // far edges (coarse there) while the side neighbours refine those shared
        // edges fine. That asymmetry is the crack the user observed on HQ; a centred
        // camera happens to be watertight (symmetric footprints reach all edges).
        let cam = Vector3::new(0.51, 147.655, -39.09);
        let (layers, ring, base, se) = (8u32, 4u32, 4.0f32, 0.03f32);
        let chains: Vec<Vec<Patch>> = (0..20)
            .map(|f| compute_patches(face_frame(f, r), layers, ring, base, se, 0.0, 0.0, true, Some(cam)))
            .collect();
        let kind = |ps: &[Patch]| ps.first().map(|p| p.kind);

        // Find the (front-facing, refined) hole face with an edge-sharing side
        // neighbour — the visible seam the user observed, not a back-face artifact.
        let front_refined = |f: usize| {
            !face_beyond_horizon(&face_frame(f, r), cam) && chains[f].len() >= 2
        };
        let mut tested = 0;
        for h in 0..20 {
            if kind(&chains[h]) != Some(PatchKind::Hole) || !front_refined(h) {
                continue;
            }
            for s in 0..20 {
                if s == h
                    || !matches!(kind(&chains[s]), Some(PatchKind::Side(_)))
                    || !front_refined(s)
                {
                    continue;
                }
                let Some((va, vb)) = shared_edge(h, s) else { continue };
                let hole_set = edge_vertex_params(&chains[h], r, va, vb);
                let side_set = edge_vertex_params(&chains[s], r, va, vb);
                assert!(!hole_set.is_empty() && !side_set.is_empty(), "no edge verts found");
                assert_eq!(
                    hole_set, side_set,
                    "hole face {h} and side face {s} disagree on shared edge ({va},{vb}): \
                     hole has {} verts, side has {} (extra side verts crack under height)",
                    hole_set.len(), side_set.len(),
                );
                tested += 1;
            }
        }
        assert!(tested > 0, "no hole↔side adjacency found for this camera; pick another");
    }

    /// CEL-66 × CEL-60: `compute_patches` stamps the camera only on patches that
    /// reach a face edge, so interior patches stay out of the per-frame dirty set.
    /// This is only safe if every *un-stamped* patch is genuinely a conform no-op —
    /// otherwise the seam reopens. Verify the gate directly: stamping a camera onto a
    /// patch `touches_face_edge` rejects must leave its realized triangles unchanged.
    #[test]
    fn unstamped_patches_are_conform_noops() {
        let r = 100.0;
        let se = 0.03f32;
        let cams = [
            Vector3::new(0.51, 147.655, -39.09),
            Vector3::new(40.0, 90.0, 30.0),
            Vector3::new(-20.0, 120.0, 80.0),
            Vector3::new(110.0, 10.0, -15.0),
        ];
        let mut checked = 0;
        for cam in cams {
            for f in 0..20 {
                for p in compute_patches(face_frame(f, r), 8, 4, 4.0, se, 0.0, 0.0, true, Some(cam)) {
                    if touches_face_edge(&p) {
                        continue;
                    }
                    // Force the camera on and confirm conform adds nothing.
                    let mut pc = p;
                    pc.cam_local = Some(cam);
                    pc.screen_error = se;
                    let mut tris = match pc.kind {
                        PatchKind::Hole => realize_hole(&pc, Color::WHITE),
                        PatchKind::Side(e) => realize_side(&pc, e, Color::WHITE),
                    }
                    .tris;
                    let before = tris.len();
                    conform_to_face_edges(&pc, &mut tris);
                    assert_eq!(
                        before,
                        tris.len(),
                        "conform modified an un-stamped patch (face {f}, kind {:?}) — predicate \
                         under-stamps and the seam would reopen",
                        pc.kind,
                    );
                    checked += 1;
                }
            }
        }
        assert!(checked > 0, "no interior patches exercised; predicate may be over-stamping");
    }

    /// Realized triangles stay close to equilateral on the sphere (longest edge /
    /// shortest edge per triangle). The symmetric-slerp mapping keeps cell sizes
    /// uniform across each face, so this ratio stays modest everywhere — a big
    /// ratio would mean stretched/sheared cells (a real bug).
    #[test]
    fn measure_triangle_stretch() {
        let r = 20.0;
        // Camera near the surface, off-centre (like the demo).
        let cam = Vector3::new(4.6, 21.0, 10.9);
        let mut worst = 1.0f32;
        let mut sum = 0.0f32;
        let mut count = 0u32;
        for face in crate::icosphere::BASE_FACES.iter() {
            let f = FaceFrame {
                a: crate::icosphere::base_vertex(face[0]) * r,
                b: crate::icosphere::base_vertex(face[1]) * r,
                c: crate::icosphere::base_vertex(face[2]) * r,
                radius: r,
            };
            let ps = compute_patches(f, 8, 4, 4.0, 0.03, 0.0, 0.0, true, Some(cam));
            for p in &ps {
                for t in realize_patch(p, Color::WHITE).tris {
                    let e = [
                        (t.corners[1] - t.corners[0]).length(),
                        (t.corners[2] - t.corners[1]).length(),
                        (t.corners[0] - t.corners[2]).length(),
                    ];
                    let lo = e[0].min(e[1]).min(e[2]);
                    let hi = e[0].max(e[1]).max(e[2]);
                    if lo > 1e-6 {
                        let ratio = hi / lo;
                        worst = worst.max(ratio);
                        sum += ratio;
                        count += 1;
                    }
                }
            }
        }
        let mean = sum / count as f32;
        println!("triangle stretch (whole sphere): worst={worst:.3} mean={mean:.3} over {count} tris");
        // The t-junction stitch (see STITCH_TRIS) emits transition triangles that
        // span a coarse parent edge — intentionally elongated (one edge up to ~2x a
        // cell), so the per-triangle `worst` ratio reaches ~2x. They are a thin
        // boundary row, so the *mean* still guards that the slerp mapping keeps the
        // bulk of cells near-equilateral (a mapping bug would drift the mean too).
        assert!(worst < 3.0, "triangles too stretched: worst ratio {worst}");
        assert!(mean < 1.2, "triangles drifting from equilateral: mean ratio {mean}");
    }

    /// On a sphere chart, realized vertices land on the sphere and the camera's
    /// gnomonic foot is radial (the foot direction equals the camera direction).
    #[test]
    fn gnomonic_projects_onto_sphere() {
        let r = 20.0;
        let f = sphere_frame(r);
        // Camera out along the face centroid direction, above the surface.
        let centroid_dir = ((f.a + f.b + f.c) / 3.0).normalized();
        let cam = centroid_dir * (r * 1.5);
        let ps = compute_patches(f, 4, 4, r * 0.2, 0.1, 0.0, 0.0, true, Some(cam));
        assert!(ps.len() >= 2, "a close camera should refine the sphere face");
        for p in &ps {
            for t in realize_patch(p, Color::WHITE).tris {
                for c in t.corners {
                    assert!((c.length() - r).abs() < 1e-3, "vertex off sphere: {}", c.length());
                }
            }
        }
    }

    /// Camera above the face, slightly off-centre.
    fn cam() -> Option<Vector3> {
        Some(Vector3::new(2.0, 40.0, 1.0))
    }

    /// Realized triangle count never exceeds the buffer hint, and something gets
    /// drawn. (It's no longer exact: patches are clipped to the face.)
    #[test]
    fn realized_within_capacity() {
        let ps = compute_patches(frame(), LAYERS, RING, BASE, 0.0, 0.0, 0.0, true, cam());
        let mut total = 0;
        for p in &ps {
            let r = realize_patch(p, Color::WHITE).tris.len();
            assert!(r <= p.max_triangles(), "realize {r} exceeded cap {}", p.max_triangles());
            total += r;
        }
        assert!(total > 0, "expected some triangles");
    }

    /// A camera whose foot lands off a face edge refines that face with a
    /// **side** chain (full-width bands from the nearest edge), so the whole face
    /// stays covered — no coarse leftover on the far side. This is the case the
    /// two-kind redesign fixes.
    #[test]
    fn off_face_center_uses_side_bands() {
        // Foot at barycentric (~0.4, ~-0.05): just past the AB (v=0) edge.
        let ps = compute_patches(frame(), 2, 4, 2.0, 0.0, 0.0, 0.0, true, Some(Vector3::new(-7.0, 40.0, 1.2)));
        assert!(ps.len() >= 2, "off-face foot should still refine the face");
        // Every layer is a side band from the AB edge (edge 0), and layer 0 covers
        // the whole face (full-width), so nothing is left coarse on the far side.
        for p in &ps {
            assert_eq!(p.kind, PatchKind::Side(0), "off-face foot => side bands");
        }
        // The bands tile with no gap — layer 0 covers the whole face, the finest
        // reaches the near edge, and each hole region equals the next band's region.
        // That is what guarantees the whole face is covered.
        assert_eq!(ps[0].outer_side, ps[0].n_fine, "layer 0 spans to the far edge");
        assert_eq!(ps[0].outer_origin, (0, ps[0].n_fine as i64), "layer 0 spans the full edge");
        assert_eq!(ps.last().unwrap().hole_side, 0, "finest band reaches the near edge");
        for w in ps.windows(2) {
            assert_eq!(w[0].hole_side, w[1].outer_side, "hole depth == next band's depth");
            assert_eq!(w[0].hole_origin, w[1].outer_origin, "hole interval == next band's");
        }
        // Whichever layer holds the full face region fills it exactly.
        let full = ps
            .iter()
            .find(|p| p.outer_side == p.n_fine && p.hole_side == 0)
            .unwrap();
        let np = (full.n_fine / full.g) as usize;
        assert_eq!(realize_patch(full, Color::WHITE).tris.len(), np * np, "full band fills the face");
    }

    /// With a camera hovering low over one face, that face takes the hole chain
    /// and must be the finest on the whole sphere — no side face may refine deeper.
    /// (The old bug: side chains measured distance to the gnomonic foot, which for
    /// glancing faces lands almost at the camera's radius, so they over-refined.)
    #[test]
    fn side_faces_never_finer_than_hole_face() {
        let r = 20.0;
        let f0 = sphere_frame(r);
        let cam = ((f0.a + f0.b + f0.c) / 3.0).normalized() * (r + 2.0);
        let mut hole_min_g = 0u32;
        let mut side_min_g = u32::MAX;
        for (fi, face) in crate::icosphere::BASE_FACES.iter().enumerate() {
            let f = FaceFrame {
                a: crate::icosphere::base_vertex(face[0]) * r,
                b: crate::icosphere::base_vertex(face[1]) * r,
                c: crate::icosphere::base_vertex(face[2]) * r,
                radius: r,
            };
            let ps = compute_patches(f, 8, 4, 4.0, 0.03, 0.0, 0.0, true, Some(cam));
            let min_g = ps.iter().map(|p| p.g).min().unwrap();
            if fi == 0 {
                assert!(ps.iter().all(|p| p.kind == PatchKind::Hole), "camera face uses holes");
                assert!(ps.len() > 1, "camera face must refine");
                hole_min_g = min_g;
            } else if ps.len() > 1 {
                assert!(matches!(ps[0].kind, PatchKind::Side(_)), "other faces use side bands");
                side_min_g = side_min_g.min(min_g);
            }
        }
        assert!(
            hole_min_g <= side_min_g,
            "hole face must be finest: hole g={hole_min_g}, finest side g={side_min_g}",
        );
    }

    /// On a big face the finer side bands stay screen-bounded in BOTH dimensions —
    /// depth *and* along the edge — instead of spanning the whole face edge.
    #[test]
    fn side_bands_bounded_on_big_face() {
        let s = 20.0;
        let base = frame();
        let f = FaceFrame { a: base.a * s, b: base.b * s, c: base.c * s, radius: 0.0 };
        // Camera low over a point just outside the AB edge, 40% of the way along.
        let on_edge = f.a + (f.b - f.a) * 0.4;
        let away = (on_edge - f.c).normalized(); // in-plane, points off the AB edge
        let cam = on_edge + away * 6.0 + f.normal() * 3.0;
        let ps = compute_patches(f, 8, 4, 2.0, 0.05, 0.0, 0.0, true, Some(cam));
        assert!(ps.len() >= 2, "close camera should refine the big face");
        let fine = ps.last().unwrap();
        assert_eq!(fine.kind, PatchKind::Side(0), "foot past AB => bands from AB");
        let nf = fine.n_fine as i64;
        assert!(
            fine.outer_origin.1 < nf / 4,
            "band width should be screen-bounded, not face-wide: {} of {nf}",
            fine.outer_origin.1,
        );
        assert!(
            (fine.outer_side as i64) < nf / 4,
            "band depth should be screen-bounded: {} of {nf}",
            fine.outer_side,
        );
        // And the band sits around the camera's spot along the edge (~40%).
        let centre = fine.outer_origin.0 + fine.outer_origin.1 / 2;
        let frac = centre as f32 / nf as f32;
        assert!((frac - 0.4).abs() < 0.1, "band should centre near the camera: {frac}");
    }

    /// A side chain grades from a shallow finest band at the near edge to a
    /// full-depth coarse layer 0, when the screen-error rule bites.
    #[test]
    fn side_bands_grade_with_distance() {
        // Foot off the AB edge; screen_error tuned so the finer band is shallower.
        let ps = compute_patches(frame(), 4, 2, 2.0, 0.05, 0.0, 0.0, true, Some(Vector3::new(-7.0, 40.0, 1.2)));
        assert!(ps.len() >= 2, "should produce at least two graded bands");
        assert!(ps.iter().all(|p| matches!(p.kind, PatchKind::Side(_))), "all side bands");
        assert_eq!(ps[0].outer_side, ps[0].n_fine, "layer 0 still spans the full face");
        for w in ps.windows(2) {
            assert!(w[1].outer_side <= w[0].outer_side, "bands shrink toward the edge");
            assert_eq!(w[0].hole_side, w[1].outer_side, "hole == next band's depth");
        }
    }

    /// Layer 0 always covers the whole face (watertight boundary); finer layers
    /// shrink and the last is solid.
    #[test]
    fn coarsest_covers_whole_face() {
        let ps = compute_patches(frame(), LAYERS, RING, BASE, 0.0, 0.0, 0.0, true, cam());
        assert_eq!(ps[0].outer_origin, (0, 0));
        assert_eq!(ps[0].outer_side, ps[0].n_fine);
        assert_eq!(ps.last().unwrap().hole_side, 0, "finest layer is solid");
        for w in ps.windows(2) {
            assert_eq!(w[1].g * 2, w[0].g, "cell size halves inward");
        }
    }

    /// Each layer's hole equals the next layer's outer triangle (no gap/overlap).
    /// Finer footprints (layer >= 1) stay nested inside their parent; layer 0's
    /// hole may poke off the face (that's the partial-coverage case), so it's
    /// exempt from the containment check.
    #[test]
    fn holes_nest_inside_parents() {
        let ps = compute_patches(frame(), LAYERS, RING, BASE, 0.0, 0.0, 0.0, true, cam());
        for w in ps.windows(2) {
            let (coarse, fine) = (&w[0], &w[1]);
            assert_eq!(coarse.hole_origin, fine.outer_origin, "hole == finer footprint");
            assert_eq!(coarse.hole_side, fine.outer_side, "hole side == finer side");
            if coarse.layer >= 1 {
                let (ox, oy, m) =
                    (coarse.outer_origin.0, coarse.outer_origin.1, coarse.outer_side as i64);
                let (hx, hy, hm) =
                    (coarse.hole_origin.0, coarse.hole_origin.1, coarse.hole_side as i64);
                assert!(hx >= ox && hy >= oy, "hole origin inside parent");
                assert!(hx + hy + hm <= ox + oy + m, "hole hypotenuse inside parent");
            }
        }
    }

    /// A stitched [`PatchKind::Hole`] patch drops every odd midpoint vertex on its
    /// three outer edges (so the boundary matches the coarser neighbour and is
    /// watertight once terrain height is applied), keeps every interior vertex, and
    /// emits no degenerate triangles. Built by hand as a finest, hole-free layer-1
    /// patch — all three outer edges border the coarser layer 0, so all three stitch.
    #[test]
    fn hole_patch_stitches_outer_boundary() {
        let f = frame();
        let n_fine = 8u32;
        let g = 2i64; // parent_g = 4
        let patch = Patch {
            layer: 1,
            n_fine,
            g: g as u32,
            outer_origin: (0, 0),
            outer_side: 8, // side = 4 this-layer cells, side_p = 2 parent cells
            hole_origin: (0, 0),
            hole_side: 0, // finest, solid (no hole)
            frame: f,
            kind: PatchKind::Hole,
            cam_local: None,
            screen_error: 0.0,
        };
        let tris = realize_patch(&patch, Color::WHITE).tris;

        let vert = |ii: i64, jj: i64| {
            f.project_bary((ii * g) as f32 / n_fine as f32, (jj * g) as f32 / n_fine as f32)
        };
        let same = |a: Vector3, b: Vector3| (a - b).length() < 1e-4;
        let used = |p: Vector3| tris.iter().any(|t| t.corners.iter().any(|&c| same(c, p)));

        // Odd-position vertices on the three outer edges must be GONE.
        let dropped = [(1, 0), (3, 0), (0, 1), (0, 3), (3, 1), (1, 3)];
        for &(ii, jj) in &dropped {
            assert!(!used(vert(ii, jj)), "stitched-away midpoint ({ii},{jj}) still emitted");
        }
        // Even (parent-aligned) corners on the outer edges must remain present.
        for &(ii, jj) in &[(0, 0), (2, 0), (4, 0), (0, 2), (0, 4), (2, 2)] {
            assert!(used(vert(ii, jj)), "parent-aligned vertex ({ii},{jj}) missing");
        }
        // No degenerate triangles.
        for t in &tris {
            let a = (t.corners[1] - t.corners[0]).cross(t.corners[2] - t.corners[0]).length();
            assert!(a > 1e-6, "degenerate triangle: {:?}", t.corners);
        }
        // Reduced count: 3 corner parent up-cells (2 tris each) + 1 interior parent
        // down-cell (4 tris) = 10, vs 16 for an unstitched side-4 solid triangle.
        assert_eq!(tris.len(), 10, "expected stitched count 10, got {}", tris.len());
    }

    // Camera at perpendicular height `h` over barycentric `(u, v)` of the face.
    fn cam_over(u: f32, v: f32, h: f32) -> Option<Vector3> {
        let f = frame();
        let foot = f.a + (f.b - f.a) * u + (f.c - f.a) * v;
        Some(foot + f.normal() * h)
    }

    /// The detail follows the camera: moving the camera over a different part of
    /// the face moves the finest footprint. (High `screen_error` + a very close
    /// camera keeps the footprints small enough to move.)
    #[test]
    fn detail_follows_camera() {
        let a = compute_patches(frame(), 3, 4, 2.0, 2.0, 0.0, 0.0, true, cam_over(0.2, 0.3, 0.5));
        let b = compute_patches(frame(), 3, 4, 2.0, 2.0, 0.0, 0.0, true, cam_over(0.6, 0.3, 0.5));
        let finest = |ps: &[Patch]| ps.last().unwrap().outer_origin;
        assert_ne!(finest(&a), finest(&b), "finest layer must track the camera");
        // Centroid fallback (no camera) sits interior.
        let centred = compute_patches(frame(), 3, 4, 2.0, 0.0, 0.0, 0.0, true, None);
        assert!(centred.last().unwrap().outer_origin.0 > 0, "centroid fallback is interior");
    }

    /// Detail grows as the camera nears: closer adds more layers, and the finest
    /// footprint grows to (nearly) fill the face.
    #[test]
    fn footprint_grows_when_close() {
        let close = compute_patches(frame(), 8, 2, 2.0, 0.02, 0.0, 0.0, true, cam_over(1.0 / 3.0, 1.0 / 3.0, 1.5));
        let far = compute_patches(frame(), 8, 2, 2.0, 0.02, 0.0, 0.0, true, cam_over(1.0 / 3.0, 1.0 / 3.0, 8.0));
        assert!(
            close.len() > far.len(),
            "closer camera should add layers: {} vs {}",
            close.len(),
            far.len(),
        );
        // Layer 1's footprint should grow to nearly cover the whole face when close.
        assert!(
            close[1].outer_side as f32 >= 0.8 * close[0].n_fine as f32,
            "near footprint should nearly fill the face: {} of {}",
            close[1].outer_side,
            close[0].n_fine,
        );
    }

    /// Screen-size LOD: a near camera produces more layers than a far one, and a
    /// distant camera collapses the face to a single coarse patch.
    #[test]
    fn depth_follows_distance() {
        // Coarse cell of 2.0 so the screen-size test actually bites at these
        // distances. `screen_error` = 0.1.
        let near = compute_patches(frame(), 4, RING, 2.0, 0.1, 0.0, 0.0, true, Some(Vector3::new(0.0, 11.0, 0.0)));
        let far = compute_patches(frame(), 4, RING, 2.0, 0.1, 0.0, 0.0, true, Some(Vector3::new(0.0, 130.0, 0.0)));
        assert!(near.len() > far.len(), "near {} should refine deeper than far {}", near.len(), far.len());
        assert_eq!(far.len(), 1, "a far face is a single coarse patch");
        // Even with screen-size depth, layer 0 still covers the whole face.
        assert_eq!(near[0].outer_side, near[0].n_fine);
        assert_eq!(far[0].outer_side, far[0].n_fine);
    }

    /// The finest triangle's *screen size* (cell world size / camera height) should
    /// stay roughly constant as the camera approaches — that's the whole point of
    /// screen-space LOD. Prints the relationship and asserts it stays bounded.
    #[test]
    fn screen_size_stays_bounded() {
        let se = 0.1f32;
        // `layers` and the height range are kept in the regime where the per-face
        // triangle budget (`MAX_FACE_TRIS`) does not bite, so screen error alone
        // decides depth. (Closer than this, the budget caps depth on purpose and
        // the finest cell stops shrinking — that is expected, not a failure.)
        let layers = 8u32;
        let base = 2.0f32;
        let cell_finest = base / (1u32 << (layers - 1)) as f32;
        let centroid = (frame().a + frame().b + frame().c) / 3.0;
        for &h in &[16.0f32, 8.0, 4.0, 2.0, 1.0] {
            let ps = compute_patches(frame(), layers, 4, base, se, 0.0, 0.0, true, Some(centroid + frame().normal() * h));
            let min_g = ps.iter().map(|p| p.g).min().unwrap();
            let finest_world = min_g as f32 * cell_finest;
            let screen = finest_world / h;
            println!("h={h:>6} finest_cell={finest_world:.5} screen={screen:.4} (target {se})");
            assert!(
                screen > se * 0.4 && screen < se * 1.1,
                "screen size {screen} drifted from target {se} at h={h}",
            );
        }
    }

    /// Guard for the face-culling feature (CEL-67): the sub-camera face is kept
    /// and the most-opposite face is culled, on a small sphere chart.
    #[test]
    fn horizon_culls_antipode_keeps_subcam() {
        let r = 20.0;
        let cam = ((sphere_frame(r).a + sphere_frame(r).b + sphere_frame(r).c) / 3.0)
            .normalized() * (r + 2.0);
        // Sub-camera face (face 0) must NOT be beyond horizon.
        assert!(!face_beyond_horizon(&sphere_frame(r), cam));
        // The face whose centroid points most opposite the camera must be culled.
        let cam_dir = cam.normalized();
        let mut worst = (0usize, f32::MAX);
        for (i, f) in crate::icosphere::BASE_FACES.iter().enumerate() {
            let cdir = ((crate::icosphere::base_vertex(f[0])
                + crate::icosphere::base_vertex(f[1])
                + crate::icosphere::base_vertex(f[2])) / 3.0).normalized();
            let d = cdir.dot(cam_dir);
            if d < worst.1 { worst = (i, d); }
        }
        let f = crate::icosphere::BASE_FACES[worst.0];
        let antipode = FaceFrame {
            a: crate::icosphere::base_vertex(f[0]) * r,
            b: crate::icosphere::base_vertex(f[1]) * r,
            c: crate::icosphere::base_vertex(f[2]) * r,
            radius: r,
        };
        assert!(face_beyond_horizon(&antipode, cam), "antipodal face must be culled");
    }

    /// Horizon culling is conservative: the face under the camera is never
    /// culled; a face on the opposite side of the planet always is.
    #[test]
    fn horizon_cull_is_conservative() {
        let r = 1000.0f32;
        let mk = |i: usize| {
            let f = crate::icosphere::BASE_FACES[i];
            FaceFrame {
                a: crate::icosphere::base_vertex(f[0]) * r,
                b: crate::icosphere::base_vertex(f[1]) * r,
                c: crate::icosphere::base_vertex(f[2]) * r,
                radius: r,
            }
        };
        let near = mk(5);
        let centre = near.project_bary(1.0 / 3.0, 1.0 / 3.0);
        let cam = centre + centre.normalized() * 30.0;
        assert!(!face_beyond_horizon(&near, cam), "face under the camera culled");

        let cam_dir = cam.normalized();
        let (far_i, _) = (0..20)
            .map(|i| {
                let f = mk(i);
                let c = (f.a + f.b + f.c).normalized();
                (i, c.dot(cam_dir))
            })
            .min_by(|a, b| a.1.total_cmp(&b.1))
            .unwrap();
        assert!(face_beyond_horizon(&mk(far_i), cam), "antipodal face not culled");

        // From high orbit every face can peek over the horizon margins' cone.
        let high = cam_dir * (r * 20.0);
        assert!(!face_beyond_horizon(&mk(far_i % 20), high) || face_beyond_horizon(&mk(far_i), high));
    }

    /// At planet scale (n_fine ~8k) with a fine screen error, mid-layer
    /// footprints exceed `MAX_FACE_TRIS`. The budget must CLAMP the footprint
    /// (refinement keeps nesting toward the camera, slightly degraded in the
    /// clamped band), never break the chain — breaking froze the ground at a
    /// coarse layer ("the patch stays blue", CEL-60 ground bug).
    #[test]
    fn budget_clamps_instead_of_truncating() {
        let r = 1000.0f32;
        let f = crate::icosphere::BASE_FACES[5];
        let frame = FaceFrame {
            a: crate::icosphere::base_vertex(f[0]) * r,
            b: crate::icosphere::base_vertex(f[1]) * r,
            c: crate::icosphere::base_vertex(f[2]) * r,
            radius: r,
        };
        // Camera 30 above the face centre on the rendered surface.
        let centre = frame.project_bary(1.0 / 3.0, 1.0 / 3.0);
        let cam = centre + centre.normalized() * 30.0;
        let (layers, base_cell, se) = (10u32, 64.0f32, 0.01f32);
        let ps = compute_patches(frame, layers, 4, base_cell, se, 0.0, 0.0, true, Some(cam));

        // The screen rule alone reaches layer 8 here (parent cell 0.5 m at
        // 30 m => 0.0167 > se; 0.25 m => 0.0083 < se).
        let deepest = ps.iter().map(|p| p.layer).max().unwrap();
        assert_eq!(deepest, 8, "chain truncated at layer {deepest} — budget broke instead of clamping");

        // And the budget still bounds every region.
        for p in &ps {
            assert!(
                p.max_triangles() <= MAX_FACE_TRIS,
                "layer {} region {} exceeds budget",
                p.layer,
                p.max_triangles()
            );
        }
    }

    /// THE LOD invariant, measured directly: every **visible** rendered cell
    /// (front of the horizon) subtends at most ~screen_error. If any face's
    /// chain breaks early and leaves coarse layer-0 cells near the camera (the
    /// recurring "red patch"), this fails and names the culprit.
    #[test]
    fn visible_cells_meet_screen_error() {
        let r = 100.0;
        let se = 0.03f32;
        // The r100 top-down LOD capture (altitude 60 over a pole, exactly on a
        // face edge) plus the demo-scene camera the bug was reported from.
        let cams = [Vector3::new(0.0, 160.0, 0.0), Vector3::new(0.51, 147.655, -39.09)];
        let mut worst = 0.0f32;
        let mut culprit = String::new();
        for (cam, (fi, face)) in cams.iter().flat_map(|&c| {
            crate::icosphere::BASE_FACES.iter().enumerate().map(move |f| (c, f))
        }) {
            let f = FaceFrame {
                a: crate::icosphere::base_vertex(face[0]) * r,
                b: crate::icosphere::base_vertex(face[1]) * r,
                c: crate::icosphere::base_vertex(face[2]) * r,
                radius: r,
            };
            let ps = compute_patches(f, 10, 4, 4.0, se, 0.0, 0.0, true, Some(cam));
            for p in &ps {
                for t in realize_patch(p, Color::WHITE).tris {
                    let centre = (t.corners[0] + t.corners[1] + t.corners[2]) / 3.0;
                    // Beyond-horizon cells are invisible: camera below the
                    // tangent plane at the cell.
                    if (cam - centre).dot(centre.normalized()) <= 0.0 {
                        continue;
                    }
                    let dist = (cam - centre).length();
                    let emax = (t.corners[1] - t.corners[0])
                        .length()
                        .max((t.corners[2] - t.corners[1]).length())
                        .max((t.corners[0] - t.corners[2]).length());
                    let screen = emax / dist;
                    if screen > worst {
                        worst = screen;
                        culprit = format!(
                            "face {fi} layer {} kind {:?} g {} dist {dist:.1} edge {emax:.2}",
                            p.layer, p.kind, p.g,
                        );
                    }
                }
            }
        }
        println!("worst visible screen size: {worst:.4} (target {se}) from {culprit}");
        // Transition triangles from the t-junction stitch (STITCH_TRIS) span a
        // coarse parent edge — up to 2x the layer's cell — so a stitched boundary
        // row legitimately reaches ~2x the budget on that one edge (it matches the
        // next-coarser neighbour by construction). The 2x cap still catches gross
        // under-tessellation (a whole layer rendered too coarse).
        assert!(
            worst <= se * 2.0,
            "visible cell breaks the screen-error budget: {worst:.4} > {se} * 2.0 ({culprit})",
        );
    }

    /// `angle_falloff` (cos-θ foreshortening) reduces the whole-sphere triangle
    /// count by coarsening faces that curve away from the camera, while leaving
    /// the sub-camera face (θ ≈ 0) essentially untouched and preserving the
    /// crack-safety invariants every face still relies on (layer 0 spans the whole
    /// face; the chain stays a strictly nested, ordered clipmap — no collapsed
    /// rings). This is the behaviour the per-face constant taper guarantees.
    #[test]
    fn angle_falloff_coarsens_tilted_faces() {
        let r = 1000.0f32;
        let mk = |i: usize| {
            let f = crate::icosphere::BASE_FACES[i];
            FaceFrame {
                a: crate::icosphere::base_vertex(f[0]) * r,
                b: crate::icosphere::base_vertex(f[1]) * r,
                c: crate::icosphere::base_vertex(f[2]) * r,
                radius: r,
            }
        };
        // Camera 40 over face 0's centroid (the demo's near-surface regime).
        let cam = mk(0).project_bary(1.0 / 3.0, 1.0 / 3.0);
        let cam = cam + cam.normalized() * 40.0;
        let tris = |af: f32| -> usize {
            (0..20)
                .flat_map(|i| compute_patches(mk(i), 10, 4, 64.0, 0.01, 1.0, af, true, Some(cam)))
                .map(|p| realize_patch(&p, Color::WHITE).tris.len())
                .sum()
        };
        let base = tris(0.0);
        let tapered = tris(1.0);
        assert!(tapered < base, "angle_falloff must cut triangles: {tapered} vs {base}");
        assert!(
            (tapered as f64) < 0.95 * base as f64,
            "expected a clear reduction, got {tapered} of {base}",
        );

        // The sub-camera face (θ ≈ 0) keeps its depth: foreshortening must not
        // strip detail from the terrain right under the camera.
        let depth = |af: f32| compute_patches(mk(0), 10, 4, 64.0, 0.01, 1.0, af, true, Some(cam))
            .iter().map(|p| p.layer).max().unwrap();
        assert_eq!(depth(0.0), depth(1.0), "sub-camera face depth must be unchanged");

        // Crack-safety invariants on every face's tapered chain: layer 0 still
        // covers the whole face, and the chain stays strictly nested (cell size
        // halves inward — no two layers collapsed to the same footprint).
        for i in 0..20 {
            let ps = compute_patches(mk(i), 10, 4, 64.0, 0.01, 1.0, 1.0, true, Some(cam));
            assert_eq!(ps[0].outer_side, ps[0].n_fine, "face {i}: layer 0 spans the face");
            for w in ps.windows(2) {
                assert_eq!(w[1].g * 2, w[0].g, "face {i}: cell size must halve inward (no collapse)");
            }
        }
    }

    /// Realizing in parallel yields the same triangle counts as serial.
    #[test]
    fn parallel_matches_serial() {
        let ps = compute_patches(frame(), LAYERS, RING, BASE, 0.0, 0.0, 0.0, true, cam());
        for (p, r) in ps.iter().zip(realize_all(&ps)) {
            assert_eq!(realize_patch(p, layer_color(p.layer)).tris.len(), r.tris.len());
        }
    }
}
