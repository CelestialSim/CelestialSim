// ChunkTileBake — Phase 4 (CEL-62) per-chunk colour + normal detail tile bake.
//
// One compute thread = one tile texel (tx, ty) of one chunk in the realize
// batch. Each chunk gets a `tile_res x tile_res` square; only the lower-left
// triangle (u + v <= 1) maps onto the chunk's triangular surface — the other
// half is unused. For each texel we reconstruct the chunk-local barycentric
// coordinate, run the EXACT same slerp projection + displacement + albedo as
// ChunkRealize.slang (so the tile agrees with the vertex grid), then ADD a
// high-frequency grass/rock detail layer (scattered green tufts + dark rock
// specks + a matching detail-normal perturbation). Because detail is evaluated
// per tile texel (tile_res, not chunk_res) it is crisper than the per-vertex
// shading the surface used before.
//
// Atlas layout (shared by terrain_chunk.gdshader's fragment fetch):
//   g     = slot*tile_res*tile_res + ty*tile_res + tx
//   coord = (g % attr_w, g / attr_w)        // same wrap as verts_tex / pos_tex
// color_atlas[coord] = rgba8 albedo
// normal_atlas[coord] = rgba8 world normal encoded *0.5+0.5

import terrain_noise_3d;

// std430 — byte-identical to `ChunkGpu` in src/chunk_descriptors.rs (96 bytes).
struct ChunkGpu {
    float4 frame_a; // xyz = face corner A, w = sphere radius (0 => flat face)
    float4 frame_b; // xyz = face corner B
    float4 frame_c; // xyz = face corner C
    float2 bary0;   // chunk corner 0 in FACE-barycentric (wb, wc)
    float2 bary1;   // chunk corner 1
    float2 bary2;   // chunk corner 2
    uint slot;      // vertex-pool / atlas slot
    uint level;     // LOD level (quadtree depth)
    uint res;       // edge resolution
    uint _pad0;
    uint _pad1;
    uint _pad2;
};

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

// std430 — byte-identical to `ChunkParams` in src/chunk_descriptors.rs (96 bytes):
// a 4-uint header, the 64-byte TerrainGpu at offset 16, then tile_res + 3 pad
// uints (rounding 84 -> 96 for the std430 16-byte array-stride rule).
struct ChunkParams {
    uint res;
    uint verts_per_chunk;
    uint attr_w;
    uint chunk_count;
    TerrainGpu terrain;
    uint tile_res;
    // Detail-normal bump enable (1.0 on, 0.0 off) — debug toggle to remove the
    // high-frequency normal perturbation entirely. Sits in the first pad slot.
    float bump_enable;
    // CPU-surface override. surface_enabled != 0 ⇒ take albedo from
    // surface_color and the world normal from surface_height; skip procedural detail.
    // surface_height_scale folds in 1/surface_radius_m and the height exaggeration.
    float surface_enabled;
    float surface_height_scale;
};

[[vk::binding(0, 0)]] StructuredBuffer<ChunkGpu> chunks;
[[vk::binding(1, 0)]] StructuredBuffer<ChunkParams> params;
[[vk::binding(2, 0)]] [format("rgba8")] RWTexture2D<float4> color_atlas;
[[vk::binding(3, 0)]] [format("rgba8")] RWTexture2D<float4> normal_atlas;
// CPU-surface atlases, indexed by linear texel g = slot*tile_res² + ty*tile_res + tx.
// surface_color: one rgba8 uint per texel (little-endian R=byte0). surface_height:
// one elevation in METERS per texel.
[[vk::binding(4, 0)]] StructuredBuffer<uint> surface_color;
[[vk::binding(5, 0)]] StructuredBuffer<float> surface_height;
// CPU-computed world normal (rgba8-packed, *0.5+0.5): curvature-correct and
// finite-differenced at the DEM SOURCE resolution — GPU-side FD at patch
// resolution both flattened per-chunk lighting (constant tangent basis) and
// turned oversampled urban DEM noise into black one-texel pseudo-cliffs.
[[vk::binding(6, 0)]] StructuredBuffer<uint> surface_normal;

TerrainParams to_terrain_params(TerrainGpu t) {
    TerrainParams p;
    p.frequency = t.frequency;
    p.height_octaves = int(t.height_octaves);
    p.height_amp = t.height_amp;
    p.height_gain = t.height_gain;
    p.height_lacunarity = t.height_lacunarity;
    p.ridge_tiles = t.ridge_tiles;
    p.ridge_octaves = int(t.ridge_octaves);
    p.ridge_gain = t.ridge_gain;
    p.ridge_lacunarity = t.ridge_lacunarity;
    p.ridge_strength = t.ridge_strength;
    p.water_height = t.water_height;
    return p;
}

// ---- spherical barycentric projection (identical to ChunkRealize) ----

float3 slerp_unit(float3 p, float3 q, float t) {
    float d = clamp(dot(p, q), -1.0, 1.0);
    float omega = acos(d);
    float so = sin(omega);
    if (abs(so) < 1.0e-6) {
        return normalize(p * (1.0 - t) + q * t);
    }
    return p * (sin((1.0 - t) * omega) / so) + q * (sin(t * omega) / so);
}

float3 tri_slerp(float3 apex, float3 c1, float3 c2, float w1, float w2) {
    float s = w1 + w2;
    if (s <= 1.0e-7) {
        return apex;
    }
    float3 edge = slerp_unit(c1, c2, w2 / s);
    return slerp_unit(apex, edge, s);
}

// GNOMONIC projection: linear-combine the on-sphere corners and renormalise.
// This MUST be the same mapping `celestial_algo::quadtree::project` uses for
// selection/culling/chunk corners (and `build_patch` uses for map texels): the
// old tri-slerp here diverged from it by up to ~1.5 deg mid-face (~170 km at
// planet scale), rendering fine chunks far from where the CPU selected them and
// letting the horizon cull carve a hole around a near-ground camera. Gnomonic
// is also edge-exact across faces (edge points combine only the two shared
// corners), so inter-face boundaries still meet.
float3 project_bary(float3 a, float3 b, float3 c, float radius, float wb, float wc) {
    float wa = 1.0 - wb - wc;
    float3 p = a * wa + b * wb + c * wc;
    if (radius <= 0.0) {
        return p;
    }
    return normalize(p) * radius;
}

float3 displace(float3 dir, float radius, TerrainGpu t, out float2 hm) {
    hm = terrain_height(dir, to_terrain_params(t));
    float centered_height = (hm.x - 0.5) * 2.4;
    return dir * (radius + centered_height * radius * t.height_scale);
}

// ---- high-frequency grass/rock detail (the Phase-4 payload) ----
//
// Evaluated in the unit-direction domain so the speckle density is uniform over
// the sphere; the frequencies are tuned so that even a close-up chunk shows many
// tufts/specks per tile (crisper than the per-vertex shading).
static const float GRASS_FREQ = 2600.0;
static const float ROCK_FREQ = 4200.0;
static const float BUMP_FREQ = 3400.0;
static const float BUMP_STRENGTH = 0.55;

[shader("compute")]
[numthreads(64, 1, 1)]
void computeMain(uint3 tid: SV_DispatchThreadID) {
    ChunkParams cp = params[0];
    uint tr = cp.tile_res;
    uint tile_texels = tr * tr;
    uint total = cp.chunk_count * tile_texels;
    if (tr == 0 || tid.x >= total) {
        return;
    }

    uint ci = tid.x / tile_texels;     // chunk index within the batch
    uint texel = tid.x % tile_texels;  // local tile texel
    uint tx = texel % tr;
    uint ty = texel / tr;

    // Texel-centre chunk-local barycentric. The square's lower-left triangle
    // (u + v <= 1) covers the chunk; the rest is unused — skip it.
    float u = (float(tx) + 0.5) / float(tr); // == wb axis
    float v = (float(ty) + 0.5) / float(tr); // == wc axis
    if (u + v > 1.0) {
        return;
    }
    float wa = 1.0 - u - v;
    float wb = u;
    float wc = v;

    ChunkGpu ch = chunks[ci];
    float radius = ch.frame_a.w;
    TerrainGpu t = cp.terrain;

    // Sub-texel FACE-barycentric, then slerp projection (matches ChunkRealize).
    float2 fbc = wa * ch.bary0 + wb * ch.bary1 + wc * ch.bary2;
    float3 base = project_bary(ch.frame_a.xyz, ch.frame_b.xyz, ch.frame_c.xyz,
                               radius, fbc.x, fbc.y);

    uint g = ch.slot * tile_texels + ty * tr + tx;
    uint attr_w = cp.attr_w;

    // ── CPU-surface branch: provider albedo + provider-derived normal. Overrides ALL
    // procedural albedo / grass / rock / bump detail below. ──
    if (cp.surface_enabled != 0.0 && radius > 0.0) {
        float3 dir = normalize(base);

        // Albedo straight from the baked rgba8 imagery uint (little-endian).
        uint packed = surface_color[g];
        float3 col = float3(float(packed & 0xffu),
                            float((packed >> 8) & 0xffu),
                            float((packed >> 16) & 0xffu)) / 255.0;

        // World normal: unpack the CPU-baked value (see binding 6 note).
        uint pn = surface_normal[g];
        float3 nrm = normalize(float3(float(pn & 0xffu),
                                      float((pn >> 8) & 0xffu),
                                      float((pn >> 16) & 0xffu)) / 255.0 * 2.0 - 1.0);

        uint2 coord = uint2(g % attr_w, g / attr_w);
        color_atlas[coord] = float4(saturate(col), 1.0);
        normal_atlas[coord] = float4(nrm * 0.5 + 0.5, 1.0);
        return;
    }

    float3 color;
    float3 normal;
    if (radius <= 0.0 || t.enabled == 0.0) {
        float3 fn = normalize(cross(ch.frame_b.xyz - ch.frame_a.xyz,
                                    ch.frame_c.xyz - ch.frame_a.xyz));
        normal = radius <= 0.0 ? fn : normalize(base);
        color = float3(1.0); // no builder ⇒ plain white sphere
    } else {
        float3 dir = normalize(base);
        float3 up = abs(dir.y) < 0.99 ? float3(0.0, 1.0, 0.0) : float3(1.0, 0.0, 0.0);
        float3 tg = normalize(cross(up, dir));
        float3 bt = cross(dir, tg);

        float3 dirs[3] = { dir, normalize(dir + tg * t.fd_eps), normalize(dir + bt * t.fd_eps) };
        float3 fd[3];
        float2 hm0 = float2(0.0, 0.0);
        [loop]
        for (uint s = 0; s < 3; s++) {
            float2 hm;
            fd[s] = displace(dirs[s], radius, t, hm);
            if (s == 0) {
                hm0 = hm;
            }
        }
        normal = normalize(cross(fd[1] - fd[0], fd[2] - fd[0]));

        float breakup = terrain_breakup_triplanar(dir);
        color = terrain_albedo_slope(hm0.x, hm0.y, breakup, 1.0, to_terrain_params(t));

        // ── Phase-4 detail: scattered grass tufts + rock specks ──
        // Above-water gate: 0 under/at water level, ramping to 1 just above it.
        // Used to keep ALL high-frequency detail off the smooth water surface.
        float above_water = smoothstep(t.water_height + 0.0, t.water_height + 0.12, hm0.x);

        // How "grassy" this spot is (flat-ish, above water, mid heights) gates
        // the green tufts so detail lands where vegetation would grow.
        float grassiness = saturate(normal.y) *
            above_water *
            (1.0 - smoothstep(0.5, 0.58, hm0.x));

        float gn = noised3d(dir * GRASS_FREQ).x * 0.5 + 0.5;
        float rn = noised3d(dir * ROCK_FREQ + float3(17.3, 4.1, 9.7)).x * 0.5 + 0.5;

        float grass = smoothstep(0.55, 0.85, gn) * grassiness;
        float rock = smoothstep(0.62, 0.9, rn);

        color = lerp(color, float3(0.16, 0.34, 0.10), grass * 0.7);
        color = lerp(color, float3(0.20, 0.18, 0.17), rock * 0.5);

        // Detail-normal perturbation: push the world normal along the tangential
        // gradient of a high-frequency noise so bumps catch light. Gated by
        // `above_water` — on water the normal stays smooth, so the lit surface
        // no longer scatters per-texel specular "dots" across the lake.
        float4 bn = noised3d(dir * BUMP_FREQ);
        float3 grad = bn.yzw;
        float3 tangential = grad - dot(grad, normal) * normal;
        normal = normalize(normal + tangential * (BUMP_STRENGTH * above_water * cp.bump_enable));
    }

    uint2 coord = uint2(g % attr_w, g / attr_w);
    color_atlas[coord] = float4(saturate(color), 1.0);
    normal_atlas[coord] = float4(normal * 0.5 + 0.5, 1.0);
}
