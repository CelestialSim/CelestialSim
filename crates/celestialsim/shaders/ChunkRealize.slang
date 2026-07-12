// ChunkRealize — Phase 2 (CEL-62) chunk vertex realize. One compute thread =
// one local vertex `L` of one chunk in the realize batch. Each chunk is a
// `res`-segment triangular grid of (res+1)(res+2)/2 vertices; this pass shades
// every vertex (slerp projection + displacement + finite-difference normal +
// albedo) into the shared vertex pool exactly the way RealizeVertices does for
// clipmap patches, so chunk geometry and clipmap geometry read identically.
//
// The local vertex enumeration MUST match crates/celestialsim/src/chunk_mesh.rs:
//   INTERIOR (L < interior_count = (res+1)(res+2)/2):
//     L(i, j) = i*(2*res + 3 - i)/2 + j,  row i has (res + 1 - i) vertices,
//     wa = (res-i-j)/res, wb = i/res, wc = j/res.
//   SKIRT (Phase 3, L >= interior_count): s = L - interior_count, edge = s/(res+1),
//     t = s % (res+1); the mirrored interior edge lattice point is
//       edge 0 (A->B): (i,j) = (t, 0)
//       edge 1 (B->C): (i,j) = (res-t, t)
//       edge 2 (C->A): (i,j) = (0, res-t)
//     A skirt vert is projected + displaced EXACTLY like that interior edge vert,
//     then pushed radially inward by `skirt_depth` so it fills the gap at a
//     T-junction crack instead of showing a hole. `skirt_depth` scales with the
//     chunk's world edge length (computed here from the chunk corners — no extra
//     descriptor field). Its colour/normal/TEX_UV equal the edge vert's, so it
//     shades like the edge.
// The global vertex index in the pool is  slot*verts_per_chunk + L  (verts_per_chunk
// is the TOTAL interior+skirt count) — used as the per-vertex pool slot for the
// RealizeVertices-identical texture writes.

import terrain_noise_3d;

// Skirt depth as a fraction of the chunk's world edge length (Phase 3). Modest, so
// the hidden skirt wall is deep enough to cover LOD-seam height gaps but not so deep
// it reads as a visible cliff at grazing angles.
static const float SKIRT_FACTOR = 0.5;

// std430, 96 bytes — byte-identical to `ChunkGpu` in src/chunk_descriptors.rs.
// Padding is three scalar `uint`s (not a `uint3`): a `uint3` has 16-byte std430
// alignment and would land at offset 96, breaking the 96-byte Rust repr(C).
struct ChunkGpu {
    float4 frame_a; // xyz = face corner A, w = sphere radius (0 => flat face)
    float4 frame_b; // xyz = face corner B
    float4 frame_c; // xyz = face corner C
    float2 bary0;   // chunk corner 0 in FACE-barycentric (wb, wc)
    float2 bary1;   // chunk corner 1
    float2 bary2;   // chunk corner 2
    uint slot;      // vertex-pool slot: owns verts_per_chunk verts at slot*verts_per_chunk
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

// Realize-batch parameters + embedded terrain (one buffer for the whole pass).
// Trailing fields (tile_res..surface_height_scale) match ChunkTileBake's ChunkParams
// and the 96-byte Rust `ChunkParams`; only `params[0]` is read so stride is moot,
// but the struct must reach `surface_*` to drive the DEM displacement below.
struct ChunkParams {
    uint res;             // edge resolution (== chunks[i].res)
    uint verts_per_chunk; // (res+1)(res+2)/2
    uint attr_w;          // verts_tex width (texels), for the idx -> (x, y) wrap
    uint chunk_count;     // number of chunks in this realize batch
    TerrainGpu terrain;
    uint tile_res;        // detail-tile resolution (surface atlas is tile_res² / slot)
    float bump_enable;    // (bake-only) detail-normal bump toggle
    // CPU-surface override: surface_enabled != 0 ⇒ displace vertices by the
    // sampled elevation instead of the procedural noise. surface_height_scale folds
    // in 1/surface_radius_m and the exaggeration — do NOT re-divide.
    float surface_enabled;
    float surface_height_scale;
};

[[vk::binding(0, 0)]] StructuredBuffer<ChunkGpu> chunks;
[[vk::binding(1, 0)]] StructuredBuffer<ChunkParams> params;
[[vk::binding(2, 0)]] [format("rgba16f")] RWTexture2D<float4> verts_tex; // 2 texels / vertex
[[vk::binding(3, 0)]] RWStructuredBuffer<float4> verts;                  // 1 float4 / vertex (pos+height)
// pos_tex (binding 4): 1 rgba32f texel per vertex (xyz = world position, w =
// height). The Godot spatial material can only sample TEXTURES (not the `verts`
// storage buffer), so the integrated chunk shader reads VERTEX from here. Same
// idx -> (x, y) wrap as verts_tex, but indexed by the vertex index directly.
[[vk::binding(4, 0)]] [format("rgba32f")] RWTexture2D<float4> pos_tex;
// CPU-surface heights (binding 5): one elevation in METRES per tile texel, indexed by
// g = slot*tile_res² + ty*tile_res + tx (same layout the bake writes/reads).
[[vk::binding(5, 0)]] StructuredBuffer<float> surface_height;

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

// ---- spherical barycentric projection (identical to RealizeVertices) ----

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

// ---- terrain displacement (matches RealizeVertices / CubemapShade) ----

float3 displace(float3 dir, float radius, TerrainGpu t, out float2 hm) {
    hm = terrain_height(dir, to_terrain_params(t));
    float centered_height = (hm.x - 0.5) * 2.4;
    return dir * (radius + centered_height * radius * t.height_scale);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void computeMain(uint3 tid: SV_DispatchThreadID) {
    ChunkParams cp = params[0];
    uint vpc = cp.verts_per_chunk;
    uint total = cp.chunk_count * vpc;
    if (tid.x >= total) {
        return;
    }

    uint ci = tid.x / vpc; // chunk index within the batch
    uint L = tid.x % vpc;  // local vertex index
    ChunkGpu ch = chunks[ci];

    int res = int(cp.res);
    uint interior_count = uint((res + 1) * (res + 2) / 2);
    bool is_skirt = L >= interior_count;

    int i;
    int j;
    if (!is_skirt) {
        // ---- INTERIOR: decode L -> (i, j): inverse of L(i,j)=i*(2*res+3-i)/2 + j ----
        // Row i has (rows - i) verts with rows = res+1; cumulative row start is
        // i*rows - i*(i-1)/2. Same quadratic trick as RealizeVertices' hole decode.
        int rows = res + 1;
        float fs = float(rows);
        float k = 2.0 * fs + 1.0;
        i = int(floor((k - sqrt(max(k * k - 8.0 * float(L), 0.0))) * 0.5));
        i = clamp(i, 0, rows - 1);
        while (i > 0 && uint(i * rows - (i * (i - 1)) / 2) > L) { i--; }
        while (uint((i + 1) * rows - ((i + 1) * i) / 2) <= L) { i++; }
        j = int(L) - (i * rows - (i * (i - 1)) / 2);
    } else {
        // ---- SKIRT: decode L -> (edge, t) -> mirrored interior edge (i, j) ----
        uint sidx = L - interior_count;
        int per_edge = res + 1;
        int edge = int(sidx) / per_edge;
        int t = int(sidx) % per_edge;
        if (edge == 0) {
            i = t; j = 0;            // edge A->B
        } else if (edge == 1) {
            i = res - t; j = t;      // edge B->C
        } else {
            i = 0; j = res - t;      // edge C->A
        }
    }

    float rf = float(res);
    float wa = float(res - i - j) / rf;
    float wb = float(i) / rf;
    float wc = float(j) / rf;

    // ---- sub-vertex FACE-barycentric, then slerp projection ----
    float2 fbc = wa * ch.bary0 + wb * ch.bary1 + wc * ch.bary2;
    float radius = ch.frame_a.w;
    float3 base = project_bary(ch.frame_a.xyz, ch.frame_b.xyz, ch.frame_c.xyz,
                               radius, fbc.x, fbc.y);

    // ---- shade (matches RealizeVertices::shade_and_store) ----
    TerrainGpu t = cp.terrain;
    float3 pos;
    float3 normal;
    float3 color;
    float height;
    if (cp.surface_enabled != 0.0 && radius > 0.0) {
        // ---- CPU-surface displacement ----
        // The vertex's chunk-local (u, v) == (wb, wc); map to the nearest tile
        // texel and sample the elevation (metres). Position must match the bake's
        // `surface_displaced_texel` formula so per-vertex geometry and per-texel
        // normals agree. The smooth per-vertex normal is `dir`; the crisp
        // DEM-derived normal comes from the baked normal_atlas.
        float3 dir = normalize(base);
        uint tr = cp.tile_res;
        uint tx = min(uint(wb * float(tr)), tr - 1u);
        uint ty = min(uint(wc * float(tr)), tr - 1u);
        uint g = ch.slot * tr * tr + ty * tr + tx;
        // RAW elevation — negative (submerged) values are kept, so the seabed
        // keeps its relief instead of collapsing into a flat disc at `radius`.
        // The analytic ocean sphere hides it; clamping also disagreed with the
        // baked normals, which are finite-differenced from the UNCLAMPED field.
        float meters = surface_height[g];
        pos = dir * radius * (1.0 + meters * cp.surface_height_scale);
        height = meters;
        normal = dir;
        color = float3(1.0); // surface colour comes from the baked color_atlas
    } else if (radius <= 0.0 || t.enabled == 0.0) {
        pos = base;
        // Flat faces have no meaningful sphere normal; use the face normal.
        float3 fn = normalize(cross(ch.frame_b.xyz - ch.frame_a.xyz,
                                    ch.frame_c.xyz - ch.frame_a.xyz));
        normal = radius <= 0.0 ? fn : normalize(base);
        color = float3(1.0); // no builder ⇒ plain white sphere
        height = 0.0;
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
        pos = fd[0];
        height = hm0.x;
        normal = normalize(cross(fd[1] - fd[0], fd[2] - fd[0]));

        float breakup = terrain_breakup_triplanar(dir);
        color = terrain_albedo_slope(hm0.x, hm0.y, breakup, 1.0, to_terrain_params(t));
    }

    // ---- Phase 3 skirt: drop the edge point radially inward by skirt_depth ----
    // The surface point (pos/normal/color/height) was computed IDENTICALLY to the
    // mirrored interior edge vertex above; only the position moves, filling a
    // LOD-seam crack with terrain instead of a hole. The colour/normal/height (and
    // TEX_UV via the mesh) stay equal to the edge so the skirt shades like the edge.
    if (is_skirt) {
        float3 drop_dir = radius > 0.0
            ? normalize(base)
            : normalize(cross(ch.frame_b.xyz - ch.frame_a.xyz,
                              ch.frame_c.xyz - ch.frame_a.xyz));
        // Skirt depth from the chunk's world edge length (corner0 -> corner1).
        float3 wc0 = project_bary(ch.frame_a.xyz, ch.frame_b.xyz, ch.frame_c.xyz,
                                  radius, ch.bary0.x, ch.bary0.y);
        float3 wc1 = project_bary(ch.frame_a.xyz, ch.frame_b.xyz, ch.frame_c.xyz,
                                  radius, ch.bary1.x, ch.bary1.y);
        float chunk_edge = length(wc1 - wc0);
        pos -= drop_dir * (chunk_edge * SKIRT_FACTOR);
    }

    // ---- write into the shared vertex pool (RealizeVertices convention) ----
    uint gv = ch.slot * vpc + L; // global vertex index / pool slot
    verts[gv] = float4(pos, height);

    uint attr_w = cp.attr_w;
    float lay = 0.0; // chunks carry no terrain layer in Phase 2
    uint ti = gv * 2;
    verts_tex[uint2(ti % attr_w, ti / attr_w)] = float4(color, lay);
    uint ti1 = ti + 1;
    verts_tex[uint2(ti1 % attr_w, ti1 / attr_w)] = float4(normal, height);

    // pos_tex: 1 texel per vertex (xyz = world position) so the surface material
    // can sample geometry — it can't read the `verts` storage buffer. Indexed by
    // gv directly (same wrap as verts_tex, stride 1 instead of 2).
    pos_tex[uint2(gv % attr_w, gv / attr_w)] = float4(pos, height);
}
