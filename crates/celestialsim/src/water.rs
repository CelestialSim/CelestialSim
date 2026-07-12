//! Analytic water-surface geometry helpers.
//!
//! The water is rendered analytically in a fragment shader (no tessellated
//! surface): a coarse proxy sphere spawns fragments, and the shader
//! ray-intersects the *mathematical* sea-level sphere whose radius is computed
//! here. The level matches the exact radius at which the terrain noise crosses
//! `water_height`, so seas/lakes line up with the terrain's own water coloring.
//!
//! Mirror of the displacement math in `ChunkRealize.slang::displace`:
//! `pos = dir * (radius + (h - 0.5) * 2.4 * radius * height_scale)`. Evaluated
//! at `h = water_height` that is the sea-level radius below. The same formula is
//! used for every builder (noise or custom) so the water level slider means the
//! same thing everywhere — `water_height = 0.5` is the baseline (the noise
//! midpoint AND the custom `h = 0` line), where the sea sits exactly at `radius`.

/// The displacement shader maps a normalized height `h` to a centered offset
/// `(h - 0.5) * HEIGHT_CENTER_SCALE`. Keep in sync with `displace` in
/// `ChunkRealize.slang` (and `ChunkTileBake` / `TileViewer` / `ScatterPlace`).
pub const HEIGHT_CENTER_SCALE: f32 = 2.4;

/// World radius of the analytic sea-level sphere: the terrain surface radius
/// evaluated at `water_height`. `water_height = 0.5` is the baseline — the noise
/// midpoint / custom `h = 0` line — at which the sea sits exactly at `radius`. At
/// the shipped defaults (`CesBuilder::water_height` 0.549, `CesGPUNoiseExample`'s
/// `height_scale` 0.0585) it sits slightly above, at `1.0069 * radius`, so terrain
/// pokes through to form seas/lakes wherever the height rises above water.
/// Lowering the level drains the seas; raising it floods low land.
pub fn water_radius(radius: f32, water_height: f32, height_scale: f32) -> f32 {
    let centered = (water_height - 0.5) * HEIGHT_CENTER_SCALE;
    radius * (1.0 + centered * height_scale)
}

/// Radius of the coarse render-proxy sphere: the analytic water radius inflated
/// by a small margin so the proxy's faceted silhouette always covers the true
/// (smooth) sphere's silhouette — rays that miss the analytic sphere are
/// discarded in-shader, so the proxy only needs to over-cover, never under.
pub const PROXY_MARGIN: f32 = 1.02;

/// Convenience: proxy sphere radius from the analytic water radius.
pub fn proxy_radius(water_radius: f32) -> f32 {
    water_radius * PROXY_MARGIN
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn water_below_the_midpoint_sits_inside_the_surface() {
        // water_height 0.45, height_scale 0.25 => (0.45-0.5)*2.4*0.25 = -0.03.
        // (NOT the default: `CesBuilder::water_height` defaults to 0.5.)
        let r = water_radius(1000.0, 0.45, 0.25);
        assert!((r - 970.0).abs() < 1e-3, "expected 970.0, got {r}");
    }

    #[test]
    fn water_height_one_half_is_mean_surface() {
        // h = 0.5 is the noise midpoint / custom baseline: water sits at radius.
        let r = water_radius(1234.0, 0.5, 0.25);
        assert!((r - 1234.0).abs() < 1e-3, "expected base radius, got {r}");
    }

    #[test]
    fn water_radius_is_monotonic_in_water_height() {
        // Higher water level => larger sea-level sphere.
        let lo = water_radius(1000.0, 0.40, 0.25);
        let mid = water_radius(1000.0, 0.50, 0.25);
        let hi = water_radius(1000.0, 0.60, 0.25);
        assert!(lo < mid && mid < hi, "not monotonic: {lo} {mid} {hi}");
    }

    #[test]
    fn proxy_radius_over_covers() {
        let wr = water_radius(1000.0, 0.45, 0.25);
        assert!(proxy_radius(wr) > wr);
        assert!((proxy_radius(wr) - wr * 1.02).abs() < 1e-3);
    }
}
