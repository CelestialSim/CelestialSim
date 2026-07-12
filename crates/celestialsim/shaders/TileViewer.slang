// TileViewer — standalone single-tile texture viewer (debug only).
//
// One compute thread = one texel (tx, ty) of a `W×W` output texture. It bakes
// the SAME surface colour as ChunkTileBake.slang (albedo + grass/rock detail)
// over a SQUARE barycentric region of ONE icosphere face, so the user can
// inspect how the per-chunk detail holds up at different zoom levels in
// isolation from the icosphere/clipmap. The viewer re-dispatches the bake every
// time the view (centre / half-extent) changes — no caching.
//
// The projection (`project_bary` slerp), `displace`, finite-difference normal,
// `terrain_albedo_slope`, the grass/rock detail and the detail-normal bump are
// copied VERBATIM from ChunkTileBake.slang so the viewer matches production
// shading. It writes BOTH a colour atlas (binding 1) and a world-normal atlas
// (binding 2) so the debug scene can show flat-albedo (unlit) AND flat+normal
// (lit) modes for the same region the chunk covers.

import terrain_noise_3d;

// std430 — byte-identical to `TerrainGpu` (16 floats, 64 bytes) in descriptors.rs.
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

// std430 — byte-identical to `TileViewerParams` in src/tile_viewer.rs (176 bytes):
//   a/b/c          = FACE corners (xyz world, a.w = sphere radius), 16 B each
//   sub0/sub1/sub2 = the viewed sub-triangle's corners in face barycentric
//                    (wb, wc in .xy) — corner 0 maps to UV (0,0), corner 1 to
//                    (1,0), corner 2 to (0,1)                       (offset 48/64/80)
//   width          = output texture edge in texels                 (offset 96)
//   tex_res        = texture-resolution quantisation grid          (offset 100)
//   terrain        = embedded 64-byte TerrainGpu                   (offset 112)
struct TileViewerParams {
    float4 a;
    float4 b;
    float4 c;
    float4 sub0;
    float4 sub1;
    float4 sub2;
    uint width;
    uint tex_res;
    uint2 _pad;
    TerrainGpu terrain;
};

[[vk::binding(0, 0)]] StructuredBuffer<TileViewerParams> params;
[[vk::binding(1, 0)]] [format("rgba8")] RWTexture2D<float4> out_tex;
// World-space normal atlas (encoded *0.5+0.5), for the flat-texture+normal mode.
[[vk::binding(2, 0)]] [format("rgba8")] RWTexture2D<float4> out_nrm;

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

// ---- spherical barycentric projection (verbatim from ChunkTileBake) ----

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

// ---- high-frequency grass/rock detail (verbatim from ChunkTileBake) ----
static const float GRASS_FREQ = 2600.0;
static const float ROCK_FREQ = 4200.0;
// ---- detail-normal bump (verbatim from ChunkTileBake) ----
static const float BUMP_FREQ = 3400.0;
static const float BUMP_STRENGTH = 0.55;

[shader("compute")]
[numthreads(64, 1, 1)]
void computeMain(uint3 tid: SV_DispatchThreadID) {
    TileViewerParams p = params[0];
    uint w = p.width;
    if (w == 0 || tid.x >= w * w) {
        return;
    }
    uint tx = tid.x % w;
    uint ty = tid.x / w;

    // Texel-centre UV in [0,1]². The displayed mesh is a triangle with corner
    // UVs (0,0),(1,0),(0,1), so only the lower-left half (u+v<=1) is ever
    // sampled; the upper-right half still bakes a valid colour (no discard) so
    // nearest-filtering along the hypotenuse never reads a hole.
    float2 uv = (float2(float(tx), float(ty)) + 0.5) / float(w);
    // Texture-resolution control: quantise UV to a `tex_res × tex_res` grid so
    // the bake shows the surface AT that texture resolution (each block = one
    // sample), independent of the output texels `w` and of camera zoom.
    float tex_res = float(p.tex_res);
    if (tex_res >= 1.0) {
        uv = (floor(uv * tex_res) + 0.5) / tex_res;
    }
    // Map UV across the viewed sub-triangle (barycentric corners sub0/1/2),
    // exactly the region the chunk covers, so lakes/mountains line up.
    float wb = p.sub0.x + uv.x * (p.sub1.x - p.sub0.x) + uv.y * (p.sub2.x - p.sub0.x);
    float wc = p.sub0.y + uv.x * (p.sub1.y - p.sub0.y) + uv.y * (p.sub2.y - p.sub0.y);

    float radius = p.a.w;
    TerrainGpu t = p.terrain;

    float3 base = project_bary(p.a.xyz, p.b.xyz, p.c.xyz, radius, wb, wc);
    // Detail-normal bump enable (1.0 on, 0.0 off) carried in the spare b.w slot.
    float bump_enable = p.b.w;

    float3 color;
    float3 out_normal = normalize(base); // world-space; flat fallback = sphere normal
    if (radius <= 0.0 || t.enabled == 0.0) {
        color = float3(0.35, 0.55, 0.25);
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
        float3 normal = normalize(cross(fd[1] - fd[0], fd[2] - fd[0]));

        float breakup = terrain_breakup_triplanar(dir);
        color = terrain_albedo_slope(hm0.x, hm0.y, breakup, 1.0, to_terrain_params(t));

        // ── grass tufts + rock specks (matches ChunkTileBake) ──
        float grassiness = saturate(normal.y) *
            smoothstep(t.water_height + 0.0, t.water_height + 0.12, hm0.x) *
            (1.0 - smoothstep(0.5, 0.58, hm0.x));

        float gn = noised3d(dir * GRASS_FREQ).x * 0.5 + 0.5;
        float rn = noised3d(dir * ROCK_FREQ + float3(17.3, 4.1, 9.7)).x * 0.5 + 0.5;

        float grass = smoothstep(0.55, 0.85, gn) * grassiness;
        float rock = smoothstep(0.62, 0.9, rn);

        color = lerp(color, float3(0.16, 0.34, 0.10), grass * 0.7);
        color = lerp(color, float3(0.20, 0.18, 0.17), rock * 0.5);

        // Detail-normal bump (matches ChunkTileBake): gated by above-water so the
        // lake stays smooth, and by `bump_enable` for the debug toggle.
        float above_water = smoothstep(t.water_height + 0.0, t.water_height + 0.12, hm0.x);
        float4 bn = noised3d(dir * BUMP_FREQ);
        float3 grad = bn.yzw;
        float3 tangential = grad - dot(grad, normal) * normal;
        normal = normalize(normal + tangential * (BUMP_STRENGTH * above_water * bump_enable));
        out_normal = normal;
    }

    out_tex[uint2(tx, ty)] = float4(saturate(color), 1.0);
    out_nrm[uint2(tx, ty)] = float4(out_normal * 0.5 + 0.5, 1.0);
}
