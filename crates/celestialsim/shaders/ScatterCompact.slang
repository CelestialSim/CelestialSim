// ScatterCompact — CEL-73 scatter draw-list build. One compute thread = one
// cached pool candidate of one VISIBLE chunk. Gathers the visible slots'
// candidates, gates them by the LIVE sliders — `hash01 < density` and the
// normalized-terrain-height band — and atomically appends the surviving
// transforms into the layer's indirect MultiMesh buffer, bumping the indirect
// draw count in the command buffer. Zero CPU readback.
//
// This is the ONLY pass a density/height edit re-runs: placement stays cached
// in the pool (ScatterPlace), terrain realize/bake are untouched.
//
// The invalid-candidate sentinel (hash01 = 2.0) fails the `h < density` gate
// for every density <= 1, so no separate validity check is needed.
//
// counter[0] and cmd[1] are zeroed by the upload node before this dispatch.

struct TerrainGpu {
    float frequency;
    float height_octaves;
    float height_amp;
    float height_gain;
    float height_lacunarity;
    float ridge_tiles;
    float ridge_octaves;
    float ridge_gain;
    float ridge_lacunarity;
    float ridge_strength;
    float water_height;
    float height_scale;
    float fd_eps;
    float enabled;
};

// std430, 128 bytes — `ScatterParamsGpu` in src/scatter_descriptors.rs.
struct ScatterParams {
    uint chunk_count; // unused here (place-side)
    uint capacity;
    uint k_per_cell; // unused here
    uint lod_level;  // unused here
    float radius;    // unused here
    float density;
    uint max_instances;
    uint seed;      // unused here
    uint vis_count; // visible instances (dispatch domain)
    float min_height; // normalized terrain-height gate [min, max]
    float max_height;
    float base_scale;          // unused here (place-side)
    float surface_enabled;     // unused here (place-side)
    float surface_height_scale;// unused here (place-side)
    uint tile_res;             // unused here (place-side)
    uint _pad3;
    TerrainGpu terrain; // unused here
};

[[vk::binding(0, 0)]] StructuredBuffer<ScatterParams> params;
[[vk::binding(1, 0)]] StructuredBuffer<float4> pool;   // 4 float4 / candidate
[[vk::binding(2, 0)]] StructuredBuffer<uint> vis;      // slot / visible chunk
[[vk::binding(3, 0)]] RWStructuredBuffer<uint> counter; // [0] = appended count
[[vk::binding(4, 0)]] RWStructuredBuffer<float> mm_buf; // 12 floats / instance
[[vk::binding(5, 0)]] RWStructuredBuffer<uint> cmd;     // indirect cmd, [1] = count

[shader("compute")]
[numthreads(64, 1, 1)]
void computeMain(uint3 tid: SV_DispatchThreadID) {
    ScatterParams p = params[0];
    uint total = p.vis_count * p.capacity;
    if (tid.x >= total) {
        return;
    }
    uint vi = tid.x / p.capacity;
    uint c = tid.x % p.capacity;
    uint slot = vis[vi];
    uint rec = (slot * p.capacity + c) * 4;
    float4 meta = pool[rec + 3]; // (hash01, normalized terrain height, -, -)
    if (meta.x >= p.density) {
        return; // density gate (also rejects the 2.0 invalid sentinel)
    }
    if (meta.y < p.min_height || meta.y > p.max_height) {
        return; // terrain-height gate (no sea grass / no peak grass)
    }

    uint idx;
    InterlockedAdd(counter[0], 1, idx);
    if (idx >= p.max_instances) {
        return; // pool cap: excess candidates silently dropped
    }
    float4 r0 = pool[rec + 0];
    float4 r1 = pool[rec + 1];
    float4 r2 = pool[rec + 2];
    uint o = idx * 12;
    mm_buf[o + 0] = r0.x;
    mm_buf[o + 1] = r0.y;
    mm_buf[o + 2] = r0.z;
    mm_buf[o + 3] = r0.w;
    mm_buf[o + 4] = r1.x;
    mm_buf[o + 5] = r1.y;
    mm_buf[o + 6] = r1.z;
    mm_buf[o + 7] = r1.w;
    mm_buf[o + 8] = r2.x;
    mm_buf[o + 9] = r2.y;
    mm_buf[o + 10] = r2.z;
    mm_buf[o + 11] = r2.w;
    InterlockedMax(cmd[1], idx + 1);
}
