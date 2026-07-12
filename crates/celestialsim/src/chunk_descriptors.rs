//! CPU → GPU packing of chunk descriptors for the Phase 2 chunked quadtree (CEL-62).
//!
//! `ChunkGpu` must match `struct ChunkGpu` in `shaders/ChunkRealize.slang`
//! byte for byte (std430: 96 bytes, 16-byte aligned due to `[f32;4]` members).
//! The size invariant is locked by the `chunk_gpu_layout_is_96_bytes` test.
//!
//! # Instance buffer layout (Godot 3D MultiMesh + custom data)
//!
//! Each instance is 16 `f32` values (64 bytes) in Godot's TRANSFORM_3D + custom_data format:
//! - Floats 0–11: 3×4 transform matrix (3 rows of 4 floats; identity → vertex shader
//!   positions geometry from the vertex-pool texture, not from the instance transform)
//! - **Float 12** (`INSTANCE_CUSTOM.r`): slot index as `f32`; the shader reads
//!   this to index into the per-face vertex pool
//! - **Float 13** (`INSTANCE_CUSTOM.g`): per-chunk geomorph factor (Phase 5):
//!   `1` = full detail, `0` = coarse/parent resolution. The surface vertex shader
//!   blends the realized grid toward its even (parent) sublattice by this factor.
//!   It rides in the per-frame instance buffer (NOT the cached realize descriptor
//!   / atlas), so a moving camera only re-uploads this small buffer — the
//!   realize/bake passes stay fully cached.
//! - Floats 14–15: zero (reserved custom channels b/a)

use celestial_algo::clipmap::FaceFrame;
use celestial_algo::quadtree::Chunk;

use crate::descriptors::TerrainGpu;

/// Realize-batch parameters + embedded terrain, byte-identical to
/// `struct ChunkParams` in `shaders/ChunkRealize.slang` / `ChunkTileBake.slang`
/// (binding 1).
///
/// std430 layout: a 4-`u32` header `{ res, verts_per_chunk, attr_w, chunk_count }`
/// (16 bytes) immediately followed by the 16-float [`TerrainGpu`] (64 bytes) at
/// offset 16. `tile_res` (Phase 4) sits at offset 80, `bump_enable` at 84, and
/// the CPU-surface `surface_enabled`/`surface_height_scale` at 88/92,
/// filling the struct to 96 bytes — a multiple of 16, as the std430 struct-array
/// rule requires. `#[repr(C)]` matches because `u32`/`f32` are 4-aligned and the
/// fields tile the tail with no interior holes (bytemuck `Pod`).
///
/// `ChunkRealize.slang` still declares the original 80-byte `ChunkParams`; it
/// only reads fields at offsets 0..80, so the appended `tile_res`+pad are
/// invisible to it and its committed SPIR-V stays valid (only `params[0]` is
/// read, so the array stride never matters there).
///
/// The `chunk_params_layout` test locks these offsets.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ChunkParams {
    /// Edge resolution (== `chunks[i].res`).
    pub res: u32,
    /// `(res+1)(res+2)/2` — vertices per chunk.
    pub verts_per_chunk: u32,
    /// `verts_tex` width in texels, for the `idx -> (x, y)` wrap.
    pub attr_w: u32,
    /// Number of chunks in this realize batch.
    pub chunk_count: u32,
    /// Terrain noise params (shared with the clipmap path); `enabled == 0` ⇒
    /// pure geometry (no displacement), the readback reference case.
    pub terrain: TerrainGpu,
    /// Phase 4: per-chunk detail-tile resolution (`tile_res × tile_res` texels
    /// baked into the colour/normal atlas). Read by `ChunkTileBake.slang`.
    pub tile_res: u32,
    /// Detail-normal bump enable (1.0 on, 0.0 off): a debug toggle that removes
    /// the high-frequency normal perturbation from the baked normal atlas. Read
    /// by `ChunkTileBake.slang`; invisible to `ChunkRealize.slang` (offset > 80).
    pub bump_enable: f32,
    /// CPU-surface global toggle at offset 88: `1.0` ⇒ the chunk
    /// shaders sample the per-chunk surface color/height storage buffers instead of
    /// the procedural surface; `0.0` ⇒ pixel-identical to the procedural path.
    pub surface_enabled: f32,
    /// CPU-surface displaced-radius factor per meter: the realize shader
    /// multiplies the sampled surface elevation (meters) by this when displacing.
    pub surface_height_scale: f32,
    /// Tail padding keeping the struct a 16-byte multiple (std430 stride rule).
    pub _pad0: f32,
    pub _pad1: f32,
}

/// Pack one [`ChunkParams`] (header + terrain + tile_res + bump + surface) into its
/// buffer. `surface_enabled`/`surface_height_scale` drive the CPU-surface
/// path (pass `0.0, 0.0` when the CPU surface is irrelevant).
pub fn pack_params(
    res: u32,
    verts_per_chunk: u32,
    attr_w: u32,
    chunk_count: u32,
    tile_res: u32,
    bump_enable: f32,
    terrain: &TerrainGpu,
    surface_enabled: f32,
    surface_height_scale: f32,
) -> Vec<u8> {
    let cp = ChunkParams {
        res,
        verts_per_chunk,
        attr_w,
        chunk_count,
        terrain: *terrain,
        tile_res,
        bump_enable,
        surface_enabled,
        surface_height_scale,
        _pad0: 0.0,
        _pad1: 0.0,
    };
    bytemuck::bytes_of(&cp).to_vec()
}

/// One chunk's GPU descriptor (std430, 96 bytes).
///
/// Field order is the shader's layout contract — do not reorder without updating
/// `ChunkRealize.slang`. Every array member is `[f32;4]` or smaller, and the struct
/// is padded to a 16-byte multiple so that an array of `ChunkGpu` has the correct
/// std430 stride (the shader indexes `chunks[i]` directly).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ChunkGpu {
    /// Face frame corner A (xyz) and sphere radius (w); radius=0 → flat terrain.
    pub frame_a: [f32; 4],
    /// Face frame corner B (xyz), w=0 (unused).
    pub frame_b: [f32; 4],
    /// Face frame corner C (xyz), w=0 (unused).
    pub frame_c: [f32; 4],
    /// Barycentric coords of chunk corner 0: (wb, wc); wa = 1 − wb − wc.
    pub bary0: [f32; 2],
    /// Barycentric coords of chunk corner 1.
    pub bary1: [f32; 2],
    /// Barycentric coords of chunk corner 2.
    pub bary2: [f32; 2],
    /// Vertex-pool slot: this chunk owns `verts_per_chunk(res)` contiguous vertices
    /// starting at `slot * verts_per_chunk(res)` in the pool texture.
    pub slot: u32,
    /// LOD level (== quadtree depth from the icosphere face root).
    pub level: u32,
    /// Edge resolution: each chunk edge has `res` segments; vertex count = `verts_per_chunk(res)`.
    pub res: u32,
    /// Explicit std430 padding: rounds the struct from 84 → 96 bytes so that
    /// `size_of::<ChunkGpu>() % 16 == 0` (required by the std430 array-stride rule).
    pub _pad: [u32; 3],
}

/// **Interior** vertex count for one chunk at resolution `res`.
///
/// The triangular grid with `res` edge segments has `(res+1)*(res+2)/2` vertices
/// (sum of rows 1..=res+1). These occupy local indices `L < interior_verts_per_chunk`;
/// the realize `L→(i,j)` decode and the GPU-readback reference both depend on this
/// interior block being byte-identical, so it is split out from the skirt verts.
pub fn interior_verts_per_chunk(res: u32) -> u32 {
    (res + 1) * (res + 2) / 2
}

/// Number of **skirt** vertices appended after the interior grid (Phase 3): one
/// ring vertex per edge position on each of the 3 edges (corners duplicated per
/// edge so each edge owns its own skirt ring), `3*(res+1)`.
pub fn skirt_verts_per_chunk(res: u32) -> u32 {
    3 * (res + 1)
}

/// **Total** vertex count for one chunk at resolution `res` = interior grid +
/// perimeter skirt (Phase 3 crack fix). This is the per-slot stride of the vertex
/// pool, the `verts_tex`/`pos_tex` sizing unit, and the realize dispatch count.
/// The interior block (`L < interior_verts_per_chunk(res)`) is unchanged; skirt
/// verts are appended at `L >= interior_verts_per_chunk(res)`.
pub fn verts_per_chunk(res: u32) -> u32 {
    interior_verts_per_chunk(res) + skirt_verts_per_chunk(res)
}

/// VRAM one resident chunk slot reserves across ALL the GPU pools:
/// geometry `verts_per_chunk(res) × 48 B` (pos_tex rgba32f 16 + verts_tex
/// 2×rgba16f 16 + verts_buf float4 16), the colour + normal detail atlases
/// (`tile_res² × 8 B`), and the CPU-surface colour + height + normal storage
/// buffers (`tile_res² × 12 B` — allocated unconditionally by
/// `ChunkGpu::ensure`). The VRAM budget divides by this; forgetting the surface
/// buffers (the old accounting) made the real allocation ~2× the configured
/// budget at large `tile_res`.
pub fn per_slot_bytes(res: u32, tile_res: u32) -> i64 {
    let tile = tile_res as i64;
    verts_per_chunk(res) as i64 * 48 + tile * tile * (8 + 12)
}

#[cfg(test)]
mod slot_bytes_tests {
    use super::*;

    #[test]
    fn per_slot_bytes_counts_atlases_and_surface_buffers() {
        // tile_res 256: atlases 256²×8 + surface colour/height/normal 256²×12.
        let geom = verts_per_chunk(20) as i64 * 48;
        assert_eq!(per_slot_bytes(20, 256), geom + 65536 * 20);
        // The old accounting (atlases only) under-counted by tile_res²×12.
        assert_eq!(per_slot_bytes(20, 256) - (geom + 65536 * 8), 65536 * 12);
    }
}

impl ChunkGpu {
    /// Pack one chunk into its GPU descriptor.
    ///
    /// `frame` is the icosphere face that owns this chunk; `slot` is the
    /// chunk's assigned vertex-pool slot.
    pub fn from_chunk(frame: &FaceFrame, c: &Chunk, slot: u32, res: u32) -> Self {
        Self {
            frame_a: [frame.a.x, frame.a.y, frame.a.z, frame.radius],
            frame_b: [frame.b.x, frame.b.y, frame.b.z, 0.0],
            frame_c: [frame.c.x, frame.c.y, frame.c.z, 0.0],
            bary0: [c.bary[0].wb, c.bary[0].wc],
            bary1: [c.bary[1].wb, c.bary[1].wc],
            bary2: [c.bary[2].wb, c.bary[2].wc],
            slot,
            level: c.level as u32,
            res,
            _pad: [0; 3],
        }
    }
}

/// Pack all visible chunks into a contiguous `ChunkGpu` byte buffer (one entry per chunk).
///
/// `frames` is the full 20-element icosphere face-frame array indexed by `chunk.id.face`;
/// `realize` is a list of `(slot, Chunk)` pairs as returned by the chunk-cache allocator.
/// Returns `realize.len() * size_of::<ChunkGpu>()` bytes.
pub fn pack_chunks(frames: &[FaceFrame], realize: &[(u32, Chunk)], res: u32) -> Vec<u8> {
    let descs: Vec<ChunkGpu> = realize
        .iter()
        .map(|(slot, chunk)| {
            let frame = &frames[chunk.id.face as usize];
            ChunkGpu::from_chunk(frame, chunk, *slot, res)
        })
        .collect();
    bytemuck::cast_slice(&descs).to_vec()
}

/// Pack per-instance data for the chunk MultiMesh into a byte buffer.
///
/// Each instance is 16 `f32` values (64 bytes) matching Godot's
/// `TRANSFORM_3D + custom_data` multimesh format:
///
/// ```text
/// floats [0..3]   — transform row 0  (identity: 1 0 0 0)
/// floats [4..7]   — transform row 1  (identity: 0 1 0 0)
/// floats [8..11]  — transform row 2  (identity: 0 0 1 0)
/// float  [12]     — INSTANCE_CUSTOM.r = slot as f32    ← slot index here
/// float  [13]     — INSTANCE_CUSTOM.g = geomorph factor ← morph here
/// floats [14..15] — INSTANCE_CUSTOM.b/a = 0
/// ```
///
/// `visible_slots` and `morphs` are parallel arrays (one entry per drawn instance,
/// same order); `morphs[i]` is the geomorph factor for the chunk in slot
/// `visible_slots[i]` (`1` = full detail, `0` = parent resolution). Pass `1.0` for
/// every entry to disable morphing.
///
/// The identity transform is intentional: the vertex shader reads vertex positions
/// from the vertex-pool texture using the slot, so no CPU-side transform is needed.
///
/// # Panics
/// Panics (debug) if `morphs.len() != visible_slots.len()`.
pub fn pack_instances(visible_slots: &[u32], morphs: &[f32]) -> Vec<u8> {
    debug_assert_eq!(
        visible_slots.len(),
        morphs.len(),
        "pack_instances: slots/morphs length mismatch"
    );
    // Identity Transform3D rows as Godot stores them in the multimesh buffer.
    const IDENTITY_TRANSFORM: [f32; 12] = [
        1.0, 0.0, 0.0, 0.0, // row 0: basis-col-0 x, basis-col-1 x, basis-col-2 x, origin.x
        0.0, 1.0, 0.0, 0.0, // row 1: basis-col-0 y, basis-col-1 y, basis-col-2 y, origin.y
        0.0, 0.0, 1.0, 0.0, // row 2: basis-col-0 z, basis-col-1 z, basis-col-2 z, origin.z
    ];

    let mut buf = Vec::with_capacity(visible_slots.len() * 16 * std::mem::size_of::<f32>());
    for (i, &slot) in visible_slots.iter().enumerate() {
        let morph = morphs.get(i).copied().unwrap_or(1.0);
        let floats: [f32; 16] = [
            IDENTITY_TRANSFORM[0],
            IDENTITY_TRANSFORM[1],
            IDENTITY_TRANSFORM[2],
            IDENTITY_TRANSFORM[3],
            IDENTITY_TRANSFORM[4],
            IDENTITY_TRANSFORM[5],
            IDENTITY_TRANSFORM[6],
            IDENTITY_TRANSFORM[7],
            IDENTITY_TRANSFORM[8],
            IDENTITY_TRANSFORM[9],
            IDENTITY_TRANSFORM[10],
            IDENTITY_TRANSFORM[11],
            slot as f32, // float index 12 = INSTANCE_CUSTOM.r
            morph,       // float index 13 = INSTANCE_CUSTOM.g (geomorph factor)
            0.0,         // float index 14 = INSTANCE_CUSTOM.b
            0.0,         // float index 15 = INSTANCE_CUSTOM.a
        ];
        buf.extend_from_slice(bytemuck::bytes_of(&floats));
    }
    buf
}

#[cfg(test)]
mod tests {
    use super::*;
    use celestial_algo::clipmap::FaceFrame;
    use celestial_algo::quadtree::{Bary, Chunk, ChunkId};
    use godot::builtin::Vector3;

    fn test_frame() -> FaceFrame {
        FaceFrame {
            a: Vector3::new(-50.0, 0.0, 28.8),
            b: Vector3::new(50.0, 0.0, 28.8),
            c: Vector3::new(0.0, 0.0, -57.7),
            radius: 100.0,
        }
    }

    fn test_chunk() -> Chunk {
        Chunk {
            id: ChunkId { face: 0, depth: 2, path: 5 },
            bary: [
                Bary { wb: 0.0, wc: 0.0 },
                Bary { wb: 1.0, wc: 0.0 },
                Bary { wb: 0.0, wc: 1.0 },
            ],
            corners: [
                Vector3::ZERO,
                Vector3::new(1.0, 0.0, 0.0),
                Vector3::new(0.0, 1.0, 0.0),
            ],
            level: 2,
        }
    }

    #[test]
    fn chunk_gpu_layout_is_96_bytes() {
        // std430: 16-byte alignment enforced by [f32;4] members; the struct stride
        // must be a multiple of 16 or the shader reads chunks[i] at wrong offsets.
        assert_eq!(std::mem::size_of::<ChunkGpu>(), 96);
        assert_eq!(std::mem::size_of::<ChunkGpu>() % 16, 0);
    }

    #[test]
    fn chunk_params_layout() {
        use std::mem::offset_of;
        // std430 array-stride rule: a StructuredBuffer<ChunkParams> stride must be
        // a multiple of 16, or params[0] reads at the wrong offset.
        assert_eq!(std::mem::size_of::<ChunkParams>(), 96);
        assert_eq!(std::mem::size_of::<ChunkParams>() % 16, 0);
        // Header offsets — must match ChunkRealize.slang / ChunkTileBake.slang.
        assert_eq!(offset_of!(ChunkParams, res), 0);
        assert_eq!(offset_of!(ChunkParams, verts_per_chunk), 4);
        assert_eq!(offset_of!(ChunkParams, attr_w), 8);
        assert_eq!(offset_of!(ChunkParams, chunk_count), 12);
        // TerrainGpu sits immediately after the 16-byte header, no interior pad.
        assert_eq!(offset_of!(ChunkParams, terrain), 16);
        assert_eq!(std::mem::size_of::<TerrainGpu>(), 56);
        // tile_res (Phase 4) follows the 56-byte terrain block at offset 72.
        assert_eq!(offset_of!(ChunkParams, tile_res), 72);
        assert_eq!(offset_of!(ChunkParams, bump_enable), 76);
        assert_eq!(offset_of!(ChunkParams, surface_enabled), 80);
        assert_eq!(offset_of!(ChunkParams, surface_height_scale), 84);
        // Two tail pads keep the struct a 16-byte multiple (88 -> 96).
        assert_eq!(offset_of!(ChunkParams, _pad0), 88);
    }

    #[test]
    fn pack_params_roundtrips_header_and_terrain() {
        let terrain = TerrainGpu {
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
            water_height: 13.0,
            height_scale: 14.0,
            fd_eps: 15.0,
            enabled: 0.0,
        };
        let bytes = pack_params(16, verts_per_chunk(16), 4096, 2, 32, 1.0, &terrain, 1.0, 0.25);
        assert_eq!(bytes.len(), 96);
        let cp: &ChunkParams = bytemuck::from_bytes(&bytes);
        assert_eq!(cp.res, 16);
        assert_eq!(cp.verts_per_chunk, verts_per_chunk(16));
        assert_eq!(cp.attr_w, 4096);
        assert_eq!(cp.chunk_count, 2);
        assert_eq!(cp.terrain, terrain);
        assert_eq!(cp.tile_res, 32);
        assert_eq!(cp.bump_enable, 1.0);
        assert_eq!(cp.surface_enabled, 1.0);
        assert_eq!(cp.surface_height_scale, 0.25);
    }

    #[test]
    fn verts_per_chunk_formula() {
        // Interior is the triangular number (res+1)*(res+2)/2.
        assert_eq!(interior_verts_per_chunk(1), 3); // (2*3)/2 = 3
        assert_eq!(interior_verts_per_chunk(16), 153); // (17*18)/2 = 153
        // Skirt adds 3*(res+1) ring verts (Phase 3).
        assert_eq!(skirt_verts_per_chunk(1), 6); // 3*2
        assert_eq!(skirt_verts_per_chunk(16), 51); // 3*17
        // Total = interior + skirt.
        assert_eq!(verts_per_chunk(1), 9); // 3 + 6
        assert_eq!(verts_per_chunk(16), 204); // 153 + 51
    }

    #[test]
    fn from_chunk_maps_fields_correctly() {
        let frame = test_frame();
        let chunk = test_chunk();
        let g = ChunkGpu::from_chunk(&frame, &chunk, 42, 16);

        // Frame corners — radius goes in frame_a.w, not b/c.
        assert_eq!(g.frame_a, [frame.a.x, frame.a.y, frame.a.z, frame.radius]);
        assert_eq!(g.frame_b, [frame.b.x, frame.b.y, frame.b.z, 0.0]);
        assert_eq!(g.frame_c, [frame.c.x, frame.c.y, frame.c.z, 0.0]);

        // Barycentric corners.
        assert_eq!(g.bary0, [chunk.bary[0].wb, chunk.bary[0].wc]);
        assert_eq!(g.bary1, [chunk.bary[1].wb, chunk.bary[1].wc]);
        assert_eq!(g.bary2, [chunk.bary[2].wb, chunk.bary[2].wc]);

        // Metadata.
        assert_eq!(g.slot, 42);
        assert_eq!(g.level, 2);
        assert_eq!(g.res, 16);

        // Padding must be zeroed (bytemuck Pod requires no uninit bytes).
        assert_eq!(g._pad, [0u32; 3]);
    }

    #[test]
    fn pack_chunks_length_matches_descriptor_size() {
        let frame = test_frame();
        let frames = vec![frame];
        let chunk = test_chunk();
        let realize = vec![(0u32, chunk), (1u32, chunk)];
        let bytes = pack_chunks(&frames, &realize, 16);
        assert_eq!(bytes.len(), realize.len() * std::mem::size_of::<ChunkGpu>());
    }

    #[test]
    fn pack_instances_layout_and_slot_position() {
        // Two slots with distinct morph factors; verify the 16-float / 64-byte blocks.
        let slots = [5u32, 9u32];
        let morphs = [0.25f32, 1.0f32];
        let bytes = pack_instances(&slots, &morphs);

        // 2 instances × 16 f32 × 4 bytes
        assert_eq!(bytes.len(), 2 * 16 * 4);

        let floats: &[f32] = bytemuck::cast_slice(&bytes);

        // Identity transform: rows 0/1/2 of the first instance.
        assert_eq!(&floats[0..4], &[1.0f32, 0.0, 0.0, 0.0]);
        assert_eq!(&floats[4..8], &[0.0f32, 1.0, 0.0, 0.0]);
        assert_eq!(&floats[8..12], &[0.0f32, 0.0, 1.0, 0.0]);

        // Float index 12 = INSTANCE_CUSTOM.r = slot (documented layout contract).
        assert_eq!(floats[12], 5.0, "first instance: slot at float index 12");
        assert_eq!(floats[16 + 12], 9.0, "second instance: slot at float index 12");

        // Float index 13 = INSTANCE_CUSTOM.g = geomorph factor (Phase 5 contract).
        assert_eq!(floats[13], 0.25, "first instance: morph at float index 13");
        assert_eq!(floats[16 + 13], 1.0, "second instance: morph at float index 13");

        // Remaining custom channels (b/a) must be zero.
        assert_eq!(floats[14], 0.0);
        assert_eq!(floats[15], 0.0);
        assert_eq!(floats[16 + 14], 0.0);
        assert_eq!(floats[16 + 15], 0.0);
    }
}
