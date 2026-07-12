//! CPU → GPU packing of the shared terrain-noise parameters.
//!
//! Both the geometry realize and the surface-detail bake read the same 56-byte
//! `TerrainGpu` block. The CPU splits ownership into two layer structs
//! (`HeightGpu` geometry, `TextureGpu` albedo) so a colour-only edit can route
//! through the graph without re-staging geometry; [`assemble`] packs them back
//! into the single block the shaders read.

/// Terrain noise parameters as the shaders read them (14 floats = 56 bytes,
/// matching `struct TerrainGpu` in the realize/bake shaders — the CONTAINING
/// params structs pad back up to a 16-byte multiple). Octave counts ride as
/// floats and are cast in-shader.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct TerrainGpu {
    pub frequency: f32,
    pub height_octaves: f32,
    pub height_amp: f32,
    pub height_gain: f32,
    pub height_lacunarity: f32,
    pub ridge_tiles: f32,
    pub ridge_octaves: f32,
    pub ridge_gain: f32,
    pub ridge_lacunarity: f32,
    pub ridge_strength: f32,
    pub water_height: f32,
    pub height_scale: f32,
    pub fd_eps: f32,
    pub enabled: f32,
}

/// The height (geometry) layer's params: everything that displaces the surface
/// and is the input to the derived albedo. Owned by the height-layer node; a
/// change re-stages geometry. `enabled` globally toggles terrain displacement.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HeightGpu {
    pub frequency: f32,
    pub height_octaves: f32,
    pub height_amp: f32,
    pub height_gain: f32,
    pub height_lacunarity: f32,
    pub ridge_tiles: f32,
    pub ridge_octaves: f32,
    pub ridge_gain: f32,
    pub ridge_lacunarity: f32,
    pub ridge_strength: f32,
    pub height_scale: f32,
    pub fd_eps: f32,
    pub enabled: f32,
}

impl Default for HeightGpu {
    /// Defaults matching `terrain_noise_3d.slang`'s built-in constants (the
    /// Lague-style continent/ridge composition). Used when no height-layer
    /// resource is assigned.
    fn default() -> Self {
        Self {
            frequency: 1.4,
            height_octaves: 8.0,
            height_amp: 0.35,
            height_gain: 0.396,
            height_lacunarity: 2.0,
            ridge_tiles: 2.4,
            ridge_octaves: 5.0,
            ridge_gain: 0.5,
            ridge_lacunarity: 2.0,
            ridge_strength: 0.0552,
            height_scale: 0.0585,
            fd_eps: 0.0008,
            enabled: 1.0,
        }
    }
}

/// The texture (albedo) layer's params. Most albedo is derived from the height
/// field, so this is currently just the water-colouring threshold — but owning
/// it as a separate layer lets a colour-only change route through the graph
/// without re-staging geometry, and is where future albedo tunables land.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TextureGpu {
    pub water_height: f32,
}

impl Default for TextureGpu {
    /// Default sea level, matching the committed `celestial_v5.tscn` look.
    fn default() -> Self {
        Self { water_height: 0.549 }
    }
}

/// Pack the two layer structs into the single 56-byte `TerrainGpu` the realize
/// and bake shaders read. Field order/positions are unchanged — the split is
/// CPU-side ownership only, so no shader or SPIR-V change is needed.
pub fn assemble(h: &HeightGpu, t: &TextureGpu) -> TerrainGpu {
    TerrainGpu {
        frequency: h.frequency,
        height_octaves: h.height_octaves,
        height_amp: h.height_amp,
        height_gain: h.height_gain,
        height_lacunarity: h.height_lacunarity,
        ridge_tiles: h.ridge_tiles,
        ridge_octaves: h.ridge_octaves,
        ridge_gain: h.ridge_gain,
        ridge_lacunarity: h.ridge_lacunarity,
        ridge_strength: h.ridge_strength,
        water_height: t.water_height,
        height_scale: h.height_scale,
        fd_eps: h.fd_eps,
        enabled: h.enabled,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn assemble_maps_layers_into_terrain_gpu() {
        // Distinct values per field so a mis-mapping is caught.
        let h = HeightGpu {
            frequency: 1.0,
            height_octaves: 2.0,
            height_amp: 3.0,
            height_gain: 4.0,
            height_lacunarity: 5.0,
            ridge_tiles: 6.0,
            ridge_octaves: 7.0,
            ridge_gain: 8.0,
            ridge_lacunarity: 9.0,
            ridge_strength: 12.0,
            height_scale: 13.0,
            fd_eps: 14.0,
            enabled: 1.0,
        };
        let t = TextureGpu { water_height: 99.0 };
        let g = assemble(&h, &t);
        // Geometry fields come from the height layer …
        assert_eq!(g.frequency, 1.0);
        assert_eq!(g.ridge_strength, 12.0);
        assert_eq!(g.height_scale, 13.0);
        assert_eq!(g.fd_eps, 14.0);
        assert_eq!(g.enabled, 1.0);
        // … the albedo field from the texture layer.
        assert_eq!(g.water_height, 99.0);
        // The terrain block the shaders read is 14 floats = 56 bytes.
        assert_eq!(std::mem::size_of::<TerrainGpu>(), 56);
    }

    #[test]
    fn layer_defaults_assemble_to_builtin_terrain() {
        // The no-resource fallback must match the shader's built-in constants
        // so a planet with no layer resources still looks right.
        let g = assemble(&HeightGpu::default(), &TextureGpu::default());
        assert_eq!(g.frequency, 1.4);
        assert_eq!(g.height_octaves, 8.0);
        assert_eq!(g.ridge_tiles, 2.4);
        assert_eq!(g.ridge_octaves, 5.0);
        assert_eq!(g.height_scale, 0.0585);
        assert_eq!(g.water_height, 0.549);
        assert_eq!(g.enabled, 1.0);
    }
}
