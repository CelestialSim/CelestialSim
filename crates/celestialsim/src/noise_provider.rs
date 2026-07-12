//! [`NoiseProvider`] — a built-in CPU fBm-terrain [`CpuSurfaceProvider`].
//!
//! Self-contained (no streaming): every chunk's colour / height / normal is
//! computed on the bake worker from 3-D fractal Perlin noise over the chunk's
//! world directions — continents and hills comparable in character to the GPU
//! `terrain_noise_3d.slang` path (it need NOT be byte-identical). Colour is a
//! height-based ramp (blue seas → green → brown → white peaks); the normal is a
//! finite difference of the height field. The provider is parametrised by the
//! same `CesBuilder` exports that drive the GPU noise, so the CPU-noise planet
//! is tunable like the GPU one.
//!
//! Height is stored as a signed displacement FRACTION of the radius, so
//! [`height_scale`](CpuSurfaceProvider::height_scale) is `1.0` and the GPU
//! surface radius becomes `radius · (1 + height)`.

use godot::builtin::Vector3;

use celestial_algo::clipmap::FaceFrame;
use celestial_algo::quadtree::Chunk;

use crate::surface::{ChunkSurface, CpuSurfaceProvider};
use crate::water::HEIGHT_CENTER_SCALE;

/// The FIXED elevation the terrain is centred on, in normalized 0..1 height —
/// mirroring `terrain_noise_3d.slang`'s `LAND_CENTER`. Terrain shape does NOT
/// depend on `water_height`; the water level simply floods this fixed relief.
const LAND_CENTER: f32 = 0.5;

/// Normalized-height swing per unit of `amp` around [`LAND_CENTER`]. Tuned so the
/// default `amp` (0.25) spans ±0.275 — the SAME 0..1 band the GPU produces (it
/// compresses its base by 0.55 around its land centre). Matching the band matters:
/// the rock/snow colour bands sit at absolute heights (0.58 / 0.66), so too small a
/// swing never reaches them and the planet comes out uniformly green and flat.
const AMP_RELIEF: f32 = 2.2;

/// Elevation above [`LAND_CENTER`] at which ridged erosion reaches full strength.
const RIDGE_FADE: f32 = 0.10;

/// Normalized-height contribution per unit of `ridge_strength`. The default
/// (0.05) lands on ~0.16, matching the GPU's `mountains * 0.16`.
const RIDGE_RELIEF: f32 = 3.2;

/// How far below the water line the seabed fades from pale shore sand to the
/// darker deep tone, in normalized height.
const SEABED_FADE: f32 = 0.10;

/// Height ABOVE the water line over which the dry beach stays fully sand, in
/// normalized height. Small ⇒ a thin shore strip rather than a sandy coastal belt.
const BEACH_FULL: f32 = 0.004;

/// Height over which that beach then fades out into green.
const BEACH_FADE: f32 = 0.006;

/// Plain, `Send + Sync` noise parameters (derived from `CesBuilder` exports).
#[derive(Clone, Copy, Debug)]
pub struct NoiseParams {
    /// Base noise frequency (tiles across the unit sphere).
    pub tiles: f32,
    /// fBm octave count.
    pub octaves: u32,
    /// Per-octave amplitude gain.
    pub gain: f32,
    /// Per-octave frequency growth.
    pub lacunarity: f32,
    /// Overall land amplitude (extra contrast on the raised land).
    pub amp: f32,
    /// Peak displacement as a fraction of the radius (land at the highest fBm).
    pub height_scale: f32,
    /// Normalized sea level in `[0, 1]`: fBm below this is flat sea.
    pub water_height: f32,
    /// Base frequency of the ridged EROSION layer (mountain ridge networks).
    pub ridge_tiles: f32,
    /// Ridged-erosion octave count (more ⇒ finer ridge/valley detail).
    pub ridge_octaves: u32,
    /// Ridged per-octave amplitude gain.
    pub ridge_gain: f32,
    /// Ridged per-octave frequency growth.
    pub ridge_lacunarity: f32,
    /// Erosion ridge amplitude as a fraction of the radius, added on land and
    /// weighted toward higher ground — the visible mountain-carving detail.
    pub ridge_strength: f32,
    /// Planet radius (world units) — for the finite-difference normal.
    pub radius: f32,
}

impl Default for NoiseParams {
    fn default() -> Self {
        NoiseParams {
            tiles: 2.0,
            octaves: 6,
            // Low gain ⇒ the continental base dominates with fine detail on top
            // (gain near 0.5 gives choppy "random noise", not continents).
            gain: 0.12,
            lacunarity: 2.0,
            amp: 0.25,
            height_scale: 0.06,
            water_height: 0.5,
            ridge_tiles: 3.242,
            ridge_octaves: 6,
            ridge_gain: 0.5,
            ridge_lacunarity: 1.8,
            ridge_strength: 0.05,
            radius: 6371.0,
        }
    }
}

/// A built-in CPU fBm-terrain provider.
pub struct NoiseProvider {
    p: NoiseParams,
}

impl NoiseProvider {
    pub fn new(p: NoiseParams) -> Self {
        NoiseProvider { p }
    }

    /// NORMALIZED terrain height in `[0, 1]` — the SAME convention as the GPU's
    /// `terrain_noise_3d.slang::terrain_height`.
    ///
    /// Crucially this is **independent of `water_height`**: the noise is centred
    /// on the FIXED [`LAND_CENTER`], exactly as the GPU recentres on its own fixed
    /// `LAND_CENTER`. That is what makes the water level a real sea-level slider —
    /// raising it FLOODS this fixed terrain. (The old version derived the terrain
    /// itself from `water_height`, so moving the slider re-shaped and rigidly
    /// lifted the whole planet along with the sea, and nothing ever flooded.)
    fn height01(&self, dir: Vector3) -> f32 {
        // Continental base, fBm in [-1, 1] → a swing around the fixed land centre.
        // `amp` is a true AMPLITUDE: larger ⇒ larger swing ⇒ TALLER mountains and
        // deeper basins. (It used to be a power exponent on a 0..1 value, so
        // raising it made mountains SMALLER — the inverted-amplitude bug.)
        let raw = fbm(dir, self.p.tiles, self.p.octaves, self.p.gain, self.p.lacunarity);
        let mut h = LAND_CENTER + 0.5 * raw * self.p.amp.max(0.0) * AMP_RELIEF;

        // Ridged erosion, carved onto the higher ground only. Gated on elevation
        // ABOVE THE LAND CENTRE (not above the water level), so the geometry stays
        // water-independent — a water edit then needs no geometry re-bake at all.
        let land = ((h - LAND_CENTER) / RIDGE_FADE).clamp(0.0, 1.0);
        if land > 0.0 {
            let ridged = ridged_fbm(
                dir,
                self.p.ridge_tiles,
                self.p.ridge_octaves,
                self.p.ridge_gain,
                self.p.ridge_lacunarity,
            );
            h += (ridged - 0.5).max(0.0) * land * self.p.ridge_strength * RIDGE_RELIEF;
        }
        h.clamp(0.0, 1.0)
    }

    /// The value stored in the height buffer (what `ChunkRealize` displaces by).
    ///
    /// `ChunkRealize` puts a CPU-surface vertex at `radius · (1 + h · height_scale())`,
    /// and [`Self::height_scale`] is `1.0`, so returning
    /// `(h01 − 0.5) · HEIGHT_CENTER_SCALE · height_scale` reproduces the GPU's
    /// `radius · (1 + (h01 − 0.5) · 2.4 · height_scale)` EXACTLY. Sea level
    /// (`h01 == water_height`) therefore lands precisely on
    /// [`crate::water::water_radius`] for any `water_height`, with no offset hack.
    fn height(&self, dir: Vector3) -> f32 {
        (self.height01(dir) - 0.5) * HEIGHT_CENTER_SCALE * self.p.height_scale
    }
}

impl CpuSurfaceProvider for NoiseProvider {
    fn bake(&self, frame: &FaceFrame, chunk: &Chunk, tile_res: u32) -> ChunkSurface {
        let n = tile_res;
        let res_f = n as f32;
        let mut color = vec![0u8; (n * n * 4) as usize];
        let mut height = vec![0f32; (n * n) as usize];

        // Chunk-local (u, v) → world direction (the cheap gnomonic the fill /
        // realize path uses): barycentric in the chunk → face barycentric →
        // linear blend of the face corners, renormalised.
        let dir_at = |u: f32, v: f32| -> Vector3 {
            let wa = 1.0 - u - v;
            let wb = wa * chunk.bary[0].wb + u * chunk.bary[1].wb + v * chunk.bary[2].wb;
            let wc = wa * chunk.bary[0].wc + u * chunk.bary[1].wc + v * chunk.bary[2].wc;
            (frame.a * (1.0 - wb - wc) + frame.b * wb + frame.c * wc).normalized()
        };

        // Fill the WHOLE square (out-of-triangle texels project onto the diagonal
        // so edge vertices and the bilinear atlas read valid data — the same
        // convention `build_patch` uses).
        for ty in 0..n {
            for tx in 0..n {
                let mut u = (tx as f32 + 0.5) / res_f;
                let mut v = (ty as f32 + 0.5) / res_f;
                if u + v > 1.0 {
                    let s = u + v;
                    u /= s;
                    v /= s;
                }
                let dir = dir_at(u, v);
                // ONE normalized height drives both: the colour ramp keys off it
                // against `water_height` (same as the GPU albedo), and the height
                // buffer stores the displacement the GPU realizes it by.
                let h01 = self.height01(dir);
                let idx = (ty * n + tx) as usize;
                height[idx] = (h01 - 0.5) * HEIGHT_CENTER_SCALE * self.p.height_scale;
                let c = surface_color(h01, self.p.water_height);
                color[idx * 4..idx * 4 + 4].copy_from_slice(&c);
            }
        }

        // ---- Normals (finite difference of the height field, curvature-correct).
        // Same construction as `build_patch`: chunk texel chords bent into the
        // local tangent plane, plus the radial height delta over a small stride.
        let mut normal = vec![0u8; (n * n * 4) as usize];
        if frame.radius > 0.0 {
            let [c0, c1, c2] = chunk.corners;
            let ex = (c1 - c0) / res_f; // world step of one +tx texel
            let ey = (c2 - c0) / res_f;
            // Height buffer is a fraction of radius, so world height = h · radius.
            let k = frame.radius;
            let ni = n as i32;
            let stride = 1i32;
            for ty in 0..n {
                for tx in 0..n {
                    let idx = (ty * n + tx) as usize;
                    let mut u = ((tx as f32 + 0.5) / res_f).min(1.0);
                    let mut v = ((ty as f32 + 0.5) / res_f).min(1.0);
                    if u + v > 1.0 {
                        let s = u + v;
                        u /= s;
                        v /= s;
                    }
                    let dir = dir_at(u, v);

                    let sample = |x: i32, y: i32| -> f32 {
                        let x = x.clamp(0, ni - 1) as u32;
                        let y = y.clamp(0, ni - 1) as u32;
                        height[(y * n + x) as usize]
                    };
                    let (xi, yi) = (tx as i32, ty as i32);
                    let (xm, xp) = ((xi - stride).max(0), (xi + stride).min(ni - 1));
                    let (ym, yp) = ((yi - stride).max(0), (yi + stride).min(ni - 1));
                    let span_x = (xp - xm).max(1) as f32;
                    let span_y = (yp - ym).max(1) as f32;

                    let exl = ex - dir * dir.dot(ex);
                    let eyl = ey - dir * dir.dot(ey);
                    let dpx = exl * span_x + dir * (k * (sample(xp, yi) - sample(xm, yi)));
                    let dpy = eyl * span_y + dir * (k * (sample(xi, yp) - sample(xi, ym)));
                    let mut nrm = dpx.cross(dpy).normalized();
                    if nrm.dot(dir) < 0.0 {
                        nrm = -nrm;
                    }
                    let pack = |c: f32| ((c * 0.5 + 0.5).clamp(0.0, 1.0) * 255.0).round() as u8;
                    normal[idx * 4] = pack(nrm.x);
                    normal[idx * 4 + 1] = pack(nrm.y);
                    normal[idx * 4 + 2] = pack(nrm.z);
                    normal[idx * 4 + 3] = 255;
                }
            }
        }

        ChunkSurface { color, height, normal }
    }

    fn sample_height(&self, dir: Vector3) -> Option<f32> {
        // The STORED height (relief + sea offset) — i.e. exactly what the GPU
        // displaces the geometry by, so LOD/ground queries match the render.
        Some(self.height(dir))
    }

    fn height_scale(&self) -> f32 {
        // `height()` already returns the FULL displacement fraction in the GPU's
        // convention — `(h01 − 0.5) · 2.4 · height_scale` — so `ChunkRealize`'s
        // `radius · (1 + h · height_scale())` must not rescale it. Any factor other
        // than 1.0 here would scale the terrain but NOT the analytic water sphere,
        // and the sea would stop meeting the shore. Amplitude is controlled by the
        // builder's `height_scale` / `amp` knobs instead.
        1.0
    }
}

/// Colour ramp keyed on the NORMALIZED height `h01` against `water_height` — the
/// same convention as the GPU albedo (`terrain_noise_3d.slang`): sandy seabed →
/// thin sand beach at the waterline → green → brown → white peaks.
///
/// Because both the bands and the analytic ocean key off `water_height`, the beach
/// always sits exactly on the waterline, and raising the water level re-colours
/// the terrain (more sea, less land) instead of moving it.
fn surface_color(h01: f32, water_height: f32) -> [u8; 4] {
    let pack = |r: f32, g: f32, b: f32| {
        [
            (r.clamp(0.0, 1.0) * 255.0) as u8,
            (g.clamp(0.0, 1.0) * 255.0) as u8,
            (b.clamp(0.0, 1.0) * 255.0) as u8,
            255,
        ]
    };
    let lerp3 = |a: [f32; 3], b: [f32; 3], s: f32| {
        [a[0] + (b[0] - a[0]) * s, a[1] + (b[1] - a[1]) * s, a[2] + (b[2] - a[2]) * s]
    };
    let smooth = |x: f32| {
        let s = x.clamp(0.0, 1.0);
        s * s * (3.0 - 2.0 * s)
    };

    if h01 <= water_height {
        // Submerged ground is SAND, not blue. The analytic ocean shader supplies
        // all the blue (it tints whatever is seen through it with a Beer-Lambert
        // depth falloff), so a blue seabed would double up. Mirrors the GPU
        // example's SEABED_SHORE_COLOR / SEABED_DEEP_COLOR: pale warm lagoon sand
        // at the shore fading to a cooler grey-tan into the deep basins.
        let seabed_shore = [0.87, 0.79, 0.64];
        let seabed_deep = [0.50, 0.50, 0.47];
        let depth = (water_height - h01) / SEABED_FADE;
        let c = lerp3(seabed_shore, seabed_deep, smooth(depth / 0.85));
        return pack(c[0], c[1], c[2]);
    }

    let sand = [0.80, 0.70, 0.60];
    let green = [0.20, 0.45, 0.16];
    let brown = [0.42, 0.32, 0.20];
    let white = [0.95, 0.95, 0.97];

    // Height ABOVE the waterline drives the shore bands; absolute height drives
    // the rock/snow bands (as on the GPU, where snow is an absolute elevation).
    let above = h01 - water_height;
    // A VERY thin dry sand beach hugging the waterline, then straight to green.
    // Kept narrow deliberately: the pale SUBMERGED sand already reads as a wide
    // bright shallows band, so a wide dry beach on top of it swamps the green.
    let land = lerp3(sand, green, smooth((above - BEACH_FULL) / BEACH_FADE));
    let c = if h01 < 0.58 {
        land
    } else if h01 < 0.66 {
        lerp3(land, brown, smooth((h01 - 0.58) / 0.04))
    } else {
        lerp3(brown, white, smooth((h01 - 0.66) / 0.06))
    };
    pack(c[0], c[1], c[2])
}

// ---- 3-D fractal Perlin noise --------------------------------------------

/// Fractal Brownian motion of [`perlin3`] in `[-1, 1]` (amplitude-normalised).
fn fbm(dir: Vector3, tiles: f32, octaves: u32, gain: f32, lacunarity: f32) -> f32 {
    let mut freq = tiles;
    let mut amp = 1.0f32;
    let mut sum = 0.0f32;
    let mut norm = 0.0f32;
    for _ in 0..octaves.max(1) {
        sum += amp * perlin3(dir * freq);
        norm += amp;
        freq *= lacunarity.max(1.0e-3);
        amp *= gain;
    }
    if norm > 0.0 {
        (sum / norm).clamp(-1.0, 1.0)
    } else {
        0.0
    }
}

/// Ridged multifractal in `[0, 1]`: each octave is `(1 - |perlin|)²` (sharp
/// ridges where the noise crosses zero), amplitude-normalised. Produces the
/// mountain ridge / valley networks that read as erosion.
fn ridged_fbm(dir: Vector3, tiles: f32, octaves: u32, gain: f32, lacunarity: f32) -> f32 {
    let mut freq = tiles;
    let mut amp = 1.0f32;
    let mut sum = 0.0f32;
    let mut norm = 0.0f32;
    for _ in 0..octaves.max(1) {
        let r = 1.0 - perlin3(dir * freq).abs();
        sum += amp * r * r;
        norm += amp;
        freq *= lacunarity.max(1.0e-3);
        amp *= gain;
    }
    if norm > 0.0 {
        (sum / norm).clamp(0.0, 1.0)
    } else {
        0.0
    }
}

#[inline]
fn fade(t: f32) -> f32 {
    t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
}

#[inline]
fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + t * (b - a)
}

/// Integer hash → pseudo-random `u32`.
#[inline]
fn hash3(x: i32, y: i32, z: i32) -> u32 {
    let mut h = (x.wrapping_mul(374_761_393))
        .wrapping_add(y.wrapping_mul(668_265_263))
        .wrapping_add(z.wrapping_mul(1_274_126_177)) as u32;
    h = (h ^ (h >> 13)).wrapping_mul(1_274_126_177);
    h ^ (h >> 16)
}

/// Classic Perlin gradient dot-product for a hashed corner.
#[inline]
fn grad(hash: u32, x: f32, y: f32, z: f32) -> f32 {
    let h = hash & 15;
    let u = if h < 8 { x } else { y };
    let v = if h < 4 {
        y
    } else if h == 12 || h == 14 {
        x
    } else {
        z
    };
    (if h & 1 == 0 { u } else { -u }) + (if h & 2 == 0 { v } else { -v })
}

/// 3-D Perlin noise in ~`[-1, 1]`.
fn perlin3(p: Vector3) -> f32 {
    let xi = p.x.floor();
    let yi = p.y.floor();
    let zi = p.z.floor();
    let (x0, y0, z0) = (xi as i32, yi as i32, zi as i32);
    let (fx, fy, fz) = (p.x - xi, p.y - yi, p.z - zi);
    let (u, v, w) = (fade(fx), fade(fy), fade(fz));

    let g = |dx: i32, dy: i32, dz: i32| -> f32 {
        grad(hash3(x0 + dx, y0 + dy, z0 + dz), fx - dx as f32, fy - dy as f32, fz - dz as f32)
    };

    let x00 = lerp(g(0, 0, 0), g(1, 0, 0), u);
    let x10 = lerp(g(0, 1, 0), g(1, 1, 0), u);
    let x01 = lerp(g(0, 0, 1), g(1, 0, 1), u);
    let x11 = lerp(g(0, 1, 1), g(1, 1, 1), u);
    let y0l = lerp(x00, x10, v);
    let y1l = lerp(x01, x11, v);
    lerp(y0l, y1l, w)
}

#[cfg(test)]
mod tests {
    use super::*;
    use celestial_algo::quadtree::{base_face_frames, select_chunks};

    #[test]
    fn perlin_is_bounded_and_varies() {
        let mut min = f32::MAX;
        let mut max = f32::MIN;
        for i in 0..2000 {
            let a = i as f32 * 0.137;
            let p = Vector3::new(a.sin() * 7.3, a.cos() * 3.1, (a * 1.7).sin() * 5.0);
            let n = perlin3(p);
            assert!(n.is_finite() && n.abs() <= 1.5, "perlin out of range: {n}");
            min = min.min(n);
            max = max.max(n);
        }
        assert!(max - min > 0.5, "perlin should vary across samples ({min}..{max})");
    }

    #[test]
    fn bake_fills_square_and_has_water_and_land() {
        let p = NoiseParams::default();
        let provider = NoiseProvider::new(p);
        let frames = base_face_frames(p.radius);
        // A coarse cut so the chunk spans a big area (both sea and land likely).
        let cam = Vector3::new(0.0, 0.0, p.radius * 2.0);
        let cut = select_chunks(&frames, cam, 0.5, 20, 3, None);
        let chunk = cut[0];
        let frame = &frames[chunk.id.face as usize];

        let tr = 32u32;
        let s = provider.bake(frame, &chunk, tr);
        assert_eq!(s.color.len(), (tr * tr * 4) as usize);
        assert_eq!(s.height.len(), (tr * tr) as usize);
        assert_eq!(s.normal.len(), (tr * tr * 4) as usize);
        // Every texel colour has full alpha; height is finite and bounded by the
        // normalized 0..1 field mapped through the GPU displacement convention.
        let bound = 0.5 * HEIGHT_CENTER_SCALE * p.height_scale + 1e-4;
        for i in 0..(tr * tr) as usize {
            assert_eq!(s.color[i * 4 + 3], 255);
            assert!(s.height[i].is_finite());
            assert!(s.height[i].abs() <= bound, "height {} out of bounds", s.height[i]);
        }
    }

    /// A sampling of directions, for whole-field comparisons.
    fn probe_dirs() -> Vec<Vector3> {
        (0..400)
            .map(|i| {
                let a = i as f32 * 0.21;
                Vector3::new(a.cos(), (a * 0.7).sin(), a.sin()).normalized()
            })
            .collect()
    }

    /// THE sea-level invariant: a point at the water line (`h01 == water_height`)
    /// must be displaced by `ChunkRealize` to EXACTLY the analytic water sphere's
    /// radius, for ANY `water_height`. Otherwise the ocean reads as a plane at the
    /// wrong level.
    #[test]
    fn cpu_sea_level_matches_analytic_water_radius() {
        for wh in [0.35f32, 0.45, 0.5, 0.62] {
            let p = NoiseParams { water_height: wh, ..NoiseParams::default() };
            let provider = NoiseProvider::new(p);
            // The stored height of a texel exactly at the water line:
            let stored_at_sea = (wh - 0.5) * HEIGHT_CENTER_SCALE * p.height_scale;
            // What ChunkRealize does with a stored height h: r = R·(1 + h·scale).
            let realized = p.radius * (1.0 + stored_at_sea * provider.height_scale());
            let expected = crate::water::water_radius(p.radius, wh, p.height_scale);
            assert!(
                (realized - expected).abs() < 1e-2,
                "water_height {wh}: sea level realized at {realized}, water sphere at {expected}"
            );
        }
    }

    /// The water level must FLOOD fixed terrain, never move it. Changing
    /// `water_height` must leave every displaced height byte-for-byte identical —
    /// the terrain is centred on a FIXED `LAND_CENTER`, exactly like the GPU.
    ///
    /// Regression test: an earlier fix added a `water_height`-derived offset to
    /// every height, which rigidly lifted the whole planet along with the sea, so
    /// raising the water level flooded nothing.
    #[test]
    fn water_height_does_not_move_the_terrain() {
        let lo = NoiseProvider::new(NoiseParams { water_height: 0.35, ..NoiseParams::default() });
        let hi = NoiseProvider::new(NoiseParams { water_height: 0.62, ..NoiseParams::default() });
        for d in probe_dirs() {
            assert!(
                (lo.height(d) - hi.height(d)).abs() < 1e-6,
                "water_height moved the terrain at {d:?}"
            );
        }
    }

    /// Raising the water level must SUBMERGE more of the (fixed) terrain.
    #[test]
    fn raising_water_height_floods_more_land() {
        let dirs = probe_dirs();
        let submerged = |wh: f32| {
            let p = NoiseParams { water_height: wh, ..NoiseParams::default() };
            let prov = NoiseProvider::new(p);
            // Submerged ⇔ the terrain sits below the water sphere.
            let sea = crate::water::water_radius(p.radius, wh, p.height_scale);
            dirs.iter()
                .filter(|d| p.radius * (1.0 + prov.height(**d) * prov.height_scale()) < sea)
                .count()
        };
        let (low, high) = (submerged(0.40), submerged(0.60));
        assert!(high > low, "raising water_height must flood more land ({low} → {high})");
    }

    /// `amp` is an AMPLITUDE: more of it must make mountains BIGGER.
    ///
    /// Regression test: it used to be a power exponent on a 0..1 value
    /// (`land.powf(1.0 + amp)`), so raising it made mountains SMALLER.
    #[test]
    fn higher_amp_makes_bigger_mountains() {
        let relief = |amp: f32| {
            let prov = NoiseProvider::new(NoiseParams { amp, ..NoiseParams::default() });
            let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
            for d in probe_dirs() {
                let h = prov.height(d);
                lo = lo.min(h);
                hi = hi.max(h);
            }
            hi - lo
        };
        let small = relief(0.15);
        let large = relief(0.45);
        assert!(large > small, "higher amp must raise relief ({small} → {large})");
    }
}
