//! CPU → GPU std430 packing for the scatter passes (CEL-73).
//!
//! Three buffers per the plan's locked contracts:
//! - [`ScatterParamsGpu`] — one 112-byte params block per layer, shared by
//!   `ScatterPlace.slang` and `ScatterCompact.slang`.
//! - aux ([`pack_scatter_aux`]) — one 16-byte `{path_lo, path_hi, face, pad}`
//!   entry per realize-batch chunk, parallel to the `ChunkGpu` batch (the
//!   descriptor lacks the quadtree path the stable lattice key needs).
//! - vis ([`pack_scatter_vis`]) — one 8-byte `{slot, depth}` `uint2` per
//!   visible instance, the compact pass's gather list.
//!
//! Layouts are locked by tests like `chunk_descriptors::chunk_params_layout`.

use celestial_algo::quadtree::Chunk;

use crate::descriptors::TerrainGpu;

/// Per-layer scatter parameters, byte-identical to `struct ScatterParams` in
/// `ScatterPlace.slang` / `ScatterCompact.slang` (binding 2 / 0).
///
/// std430: a 64-byte scalar header followed by the 56-byte [`TerrainGpu`] at
/// offset 64, plus two tail pads; 128 bytes total (multiple of 16 per the
/// struct-array rule).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ScatterParamsGpu {
    /// Chunks in this realize batch (place dispatch domain).
    pub chunk_count: u32,
    /// Per-slot pool capacity `C = k_per_cell * 4^S_MAX`.
    pub capacity: u32,
    /// Candidates per lattice cell (K).
    pub k_per_cell: u32,
    /// The layer's LOD level (== the stable world lattice level L in
    /// `celestial_algo::scatter`).
    pub lod_level: u32,
    /// Sphere radius.
    pub radius: f32,
    /// LIVE: fraction of candidates drawn (`hash01 < density`).
    pub density: f32,
    /// MultiMesh pool cap (compact clamps its atomic counter to this).
    pub max_instances: u32,
    /// Layer placement seed.
    pub seed: u32,
    /// Visible instances (compact dispatch domain).
    pub vis_count: u32,
    /// LIVE: min normalized terrain height (0..1) an instance may sit at; below
    /// it, nothing scatters (0.45 = the default sea level → no underwater).
    pub min_height: f32,
    /// LIVE: max normalized terrain height (0..1); above it, nothing scatters
    /// (1.0 = no upper limit → keep grass off peaks by lowering this).
    pub max_height: f32,
    /// Per-layer base scale multiplier (place-side: baked into the transform).
    pub base_scale: f32,
    /// CPU-surface route gate (mirrors `ChunkParams.surface_enabled`): non-zero
    /// ⇒ place samples the baked heightmap instead of the procedural noise.
    pub surface_enabled: f32,
    /// CPU-surface displacement scale (`CpuSurfaceProvider::height_scale`) —
    /// must equal `ChunkParams.surface_height_scale` or instances float/sink.
    pub surface_height_scale: f32,
    /// Detail-tile resolution: the per-slot heightmap is `tile_res²` floats.
    pub tile_res: u32,
    pub _pad: u32,
    /// Terrain noise params — displacement must match `ChunkRealize.slang`.
    pub terrain: TerrainGpu,
    /// Tail padding keeping the struct a 16-byte multiple (std430 stride rule).
    pub _pad2: f32,
    pub _pad3: f32,
}

/// Pack one [`ScatterParamsGpu`] into bytes for its per-layer buffer.
#[allow(clippy::too_many_arguments)]
pub fn pack_scatter_params(
    chunk_count: u32,
    capacity: u32,
    k_per_cell: u32,
    lod_level: u32,
    radius: f32,
    density: f32,
    max_instances: u32,
    seed: u32,
    vis_count: u32,
    min_height: f32,
    max_height: f32,
    base_scale: f32,
    surface_enabled: f32,
    surface_height_scale: f32,
    tile_res: u32,
    terrain: &TerrainGpu,
) -> Vec<u8> {
    let p = ScatterParamsGpu {
        chunk_count,
        capacity,
        k_per_cell,
        lod_level,
        radius,
        density,
        max_instances,
        seed,
        vis_count,
        min_height,
        max_height,
        base_scale,
        surface_enabled,
        surface_height_scale,
        tile_res,
        _pad: 0,
        terrain: *terrain,
        _pad2: 0.0,
        _pad3: 0.0,
    };
    bytemuck::bytes_of(&p).to_vec()
}

/// One realize-batch chunk's scatter aux entry (std430, 16 bytes): the chunk's
/// quadtree path split into two words + its face. Parallel to the `ChunkGpu`
/// batch (same index), which already carries slot/level/frame/bary.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ScatterAuxGpu {
    pub path_lo: u32,
    pub path_hi: u32,
    pub face: u32,
    pub _pad: u32,
}

/// Pack the aux buffer for a realize batch (parallel to `pack_chunks` order).
pub fn pack_scatter_aux(realize: &[(u32, Chunk)]) -> Vec<u8> {
    let entries: Vec<ScatterAuxGpu> = realize
        .iter()
        .map(|(_, c)| ScatterAuxGpu {
            path_lo: c.id.path as u32,
            path_hi: (c.id.path >> 32) as u32,
            face: c.id.face as u32,
            _pad: 0,
        })
        .collect();
    bytemuck::cast_slice(&entries).to_vec()
}

/// Pack the compact pass's visible list: one `uint` slot per drawn instance
/// (same order as the instance buffer / `visible_slots`).
pub fn pack_scatter_vis(slots: &[u32]) -> Vec<u8> {
    let mut buf = Vec::with_capacity(slots.len() * 4);
    for s in slots {
        buf.extend_from_slice(&s.to_le_bytes());
    }
    buf
}

#[cfg(test)]
mod tests {
    use super::*;
    use celestial_algo::quadtree::{Bary, Chunk, ChunkId};
    use godot::builtin::Vector3;

    #[test]
    fn scatter_params_layout() {
        use std::mem::offset_of;
        assert_eq!(std::mem::size_of::<ScatterParamsGpu>(), 128);
        assert_eq!(std::mem::size_of::<ScatterParamsGpu>() % 16, 0);
        assert_eq!(offset_of!(ScatterParamsGpu, chunk_count), 0);
        assert_eq!(offset_of!(ScatterParamsGpu, capacity), 4);
        assert_eq!(offset_of!(ScatterParamsGpu, k_per_cell), 8);
        assert_eq!(offset_of!(ScatterParamsGpu, lod_level), 12);
        assert_eq!(offset_of!(ScatterParamsGpu, radius), 16);
        assert_eq!(offset_of!(ScatterParamsGpu, density), 20);
        assert_eq!(offset_of!(ScatterParamsGpu, max_instances), 24);
        assert_eq!(offset_of!(ScatterParamsGpu, seed), 28);
        assert_eq!(offset_of!(ScatterParamsGpu, vis_count), 32);
        assert_eq!(offset_of!(ScatterParamsGpu, min_height), 36);
        assert_eq!(offset_of!(ScatterParamsGpu, max_height), 40);
        assert_eq!(offset_of!(ScatterParamsGpu, base_scale), 44);
        assert_eq!(offset_of!(ScatterParamsGpu, surface_enabled), 48);
        assert_eq!(offset_of!(ScatterParamsGpu, surface_height_scale), 52);
        assert_eq!(offset_of!(ScatterParamsGpu, tile_res), 56);
        assert_eq!(offset_of!(ScatterParamsGpu, terrain), 64);
    }

    #[test]
    fn pack_params_roundtrips() {
        let t = crate::descriptors::assemble(
            &crate::descriptors::HeightGpu::default(),
            &crate::descriptors::TextureGpu::default(),
        );
        let bytes = pack_scatter_params(
            3, 256, 4, 9, 1000.0, 0.5, 100_000, 7, 42, 0.45, 0.9, 2.0, 1.0, 0.18, 64, &t,
        );
        assert_eq!(bytes.len(), 128);
        let p: &ScatterParamsGpu = bytemuck::from_bytes(&bytes);
        assert_eq!(p.chunk_count, 3);
        assert_eq!(p.capacity, 256);
        assert_eq!(p.k_per_cell, 4);
        assert_eq!(p.lod_level, 9);
        assert_eq!(p.radius, 1000.0);
        assert_eq!(p.density, 0.5);
        assert_eq!(p.max_instances, 100_000);
        assert_eq!(p.seed, 7);
        assert_eq!(p.vis_count, 42);
        assert_eq!(p.min_height, 0.45);
        assert_eq!(p.max_height, 0.9);
        assert_eq!(p.base_scale, 2.0);
        assert_eq!(p.surface_enabled, 1.0);
        assert_eq!(p.surface_height_scale, 0.18);
        assert_eq!(p.tile_res, 64);
        assert_eq!(p.terrain, t);
    }

    #[test]
    fn pack_aux_layout() {
        let path: u64 = 0xdead_beef_0000_0003;
        let chunk = Chunk {
            id: ChunkId { face: 5, depth: 2, path },
            bary: [
                Bary { wb: 0.0, wc: 0.0 },
                Bary { wb: 1.0, wc: 0.0 },
                Bary { wb: 0.0, wc: 1.0 },
            ],
            corners: [Vector3::ZERO; 3],
            level: 2,
        };
        let bytes = pack_scatter_aux(&[(9, chunk), (1, chunk)]);
        assert_eq!(bytes.len(), 32); // 16 B per entry
        let entries: &[ScatterAuxGpu] = bytemuck::cast_slice(&bytes);
        assert_eq!(entries[0].path_lo, 0x0000_0003);
        assert_eq!(entries[0].path_hi, 0xdead_beef);
        assert_eq!(entries[0].face, 5);
        assert_eq!(entries[1], entries[0]);
    }

    #[test]
    fn pack_vis_slots() {
        let bytes = pack_scatter_vis(&[7, 3]);
        assert_eq!(bytes.len(), 8);
        let words: &[u32] = bytemuck::cast_slice(&bytes);
        assert_eq!(words, &[7, 3]);
    }
}
