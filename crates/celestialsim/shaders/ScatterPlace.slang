// ScatterPlace — CEL-73 scatter layer placement. One compute thread = one
// candidate slot `c` of one realize-batch chunk. Runs only when chunks are
// (re)realized; results are CACHED per pool slot, so a moving camera or a
// density/height edit never re-runs this pass.
//
// Candidates live on a STABLE world lattice: level-L cells of the face
// quadtree. A candidate is keyed by (face, level-L cell path, k, seed) ONLY —
// never by the hosting chunk's depth/slot — so instances never move when the
// chunk quadtree splits or merges. The math here mirrors
// crates/celestial-algo/src/scatter.rs bit for bit (wang/seed0/candidate/
// descend order); the invariant tests there are the proof.
//
// Pool record = 4 float4s per candidate:
//   rec0..2 = Godot TRANSFORM_3D rows (row r = (X[r], Y[r], Z[r], origin[r]))
//   rec3.x  = hash01 keep-value in [0,1), or 2.0 = invalid sentinel
// A chunk's FULL capacity region is rewritten every place (invalid slots get
// the sentinel), so slot eviction/reuse needs no clearing pass.

import terrain_noise_3d;

// std430, 96 bytes — byte-identical to `ChunkGpu` in src/chunk_descriptors.rs.
struct ChunkGpu {
    float4 frame_a; // xyz = face corner A, w = sphere radius
    float4 frame_b;
    float4 frame_c;
    float2 bary0; // chunk corner 0 in FACE-barycentric (wb, wc)
    float2 bary1;
    float2 bary2;
    uint slot;
    uint level; // quadtree depth
    uint res;
    uint _pad0;
    uint _pad1;
    uint _pad2;
};

// std430, 16 bytes — `ScatterAuxGpu` in src/scatter_descriptors.rs, parallel
// to the ChunkGpu batch: the quadtree path + face the descriptor lacks.
struct ScatterAux {
    uint path_lo;
    uint path_hi;
    uint face;
    uint _pad;
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

// std430, 128 bytes — `ScatterParamsGpu` in src/scatter_descriptors.rs.
struct ScatterParams {
    uint chunk_count;
    uint capacity;      // C = k_per_cell * 4^S_MAX
    uint k_per_cell;    // K
    uint lod_level;     // L (lattice level)
    float radius;
    float density;      // compact-side gate
    uint max_instances; // compact-side
    uint seed;
    uint vis_count;   // compact-side
    float min_height; // compact-side height gate (normalized 0..1)
    float max_height; // compact-side height gate
    float base_scale; // place-side: baked into the transform
    // CPU-surface (provider) route — mirrors `ChunkParams` in ChunkRealize:
    // when enabled, the instance sits on the SAMPLED heightmap, not the noise.
    float surface_enabled;
    float surface_height_scale;
    uint tile_res;
    uint _pad3;
    TerrainGpu terrain;
};

[[vk::binding(0, 0)]] StructuredBuffer<ChunkGpu> chunks;
[[vk::binding(1, 0)]] StructuredBuffer<ScatterAux> aux;
[[vk::binding(2, 0)]] StructuredBuffer<ScatterParams> params;
[[vk::binding(3, 0)]] RWStructuredBuffer<float4> pool; // 4 float4 / candidate
// Per-slot CPU-surface elevation grid (tile_res² floats / slot), the SAME
// buffer ChunkRealize displaces its vertices with.
[[vk::binding(4, 0)]] StructuredBuffer<float> surface_height;

static const uint S_MAX = 3;
static const float INVALID = 2.0;

// ---- stable hashing (mirrors celestial_algo::scatter exactly) ----

uint wang(uint x) {
    x = (x ^ 61u) ^ (x >> 16);
    x *= 9u;
    x ^= x >> 4;
    x *= 0x27d4eb2du;
    x ^= x >> 15;
    return x;
}

float u01(uint x) {
    return float(wang(x)) * (1.0 / 4294967296.0);
}

uint seed0(uint face, uint2 path, uint k, uint seed) {
    return wang(wang(wang(face) ^ path.x) ^ path.y) ^ wang(k ^ (seed * 0x9e3779b9u));
}

// ---- 64-bit path ops on uint2 (lo, hi) ----

uint2 shl64(uint2 v, uint n) {
    if (n == 0) {
        return v;
    }
    return uint2(v.x << n, (v.y << n) | (v.x >> (32 - n)));
}

uint2 shr64(uint2 v, uint n) {
    if (n == 0) {
        return v;
    }
    if (n >= 32) {
        return uint2(v.y >> (n - 32), 0);
    }
    return uint2((v.x >> n) | (v.y << (32 - n)), v.y >> n);
}

// Child index at `shift` bits from the low end.
uint bits2(uint2 path, uint shift) {
    return shr64(path, shift).x & 3u;
}

// Descend `levels` split steps from tri (A,B,C), consuming path bits
// ROOT-FIRST. Split order matches quadtree::split_bary:
// [a,mab,mca], [mab,b,mbc], [mca,mbc,c], [mab,mbc,mca].
void descend(inout float2 A, inout float2 B, inout float2 C, uint2 path, uint levels) {
    for (uint i = 0; i < levels; i++) {
        uint k = bits2(path, 2 * (levels - 1 - i));
        float2 ab = (A + B) * 0.5;
        float2 bc = (B + C) * 0.5;
        float2 ca = (C + A) * 0.5;
        if (k == 0) {
            B = ab; C = ca;
        } else if (k == 1) {
            A = ab; C = bc;
        } else if (k == 2) {
            A = ca; B = bc;
        } else {
            A = ab; B = bc; C = ca;
        }
    }
}

// Boundary-inclusive point-in-triangle in face-bary 2D (same rule as the CPU).
// `uv` = the point's CHUNK-LOCAL barycentric coords (u along a->b, v along
// a->c) — the SAME (wb, wc) ChunkRealize builds its tile texel index from, so
// the CPU-surface branch can sample the heightmap at the instance's texel.
bool in_triangle_uv(float2 p, float2 a, float2 b, float2 c, out float2 uv) {
    uv = float2(0.0, 0.0);
    float2 v0 = b - a;
    float2 v1 = c - a;
    float2 v2 = p - a;
    float den = v0.x * v1.y - v1.x * v0.y;
    if (abs(den) < 1.0e-12) {
        return false;
    }
    float u = (v2.x * v1.y - v1.x * v2.y) / den;
    float v = (v0.x * v2.y - v2.x * v0.y) / den;
    uv = float2(u, v);
    return u >= 0.0 && v >= 0.0 && u + v <= 1.0;
}

// ---- spherical projection + displacement ----
// GNOMONIC projection: linear-combine the on-sphere corners and renormalise.
// This MUST match `celestial_algo::quadtree::project` (selection/culling/chunk
// corners) and `ChunkRealize.slang`'s `project_bary`, or scattered instances
// land laterally offset from the terrain that hosts them — harmless while
// height is a pure function of direction, but visibly floating/buried with
// sampled height sources (e.g. a CPU-surface provider).

float3 project_bary(float3 a, float3 b, float3 c, float radius, float wb, float wc) {
    float wa = 1.0 - wb - wc;
    float3 p = a * wa + b * wb + c * wc;
    if (radius <= 0.0) {
        return p;
    }
    return normalize(p) * radius;
}

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

float3 displace(float3 dir, float radius, TerrainGpu t, out float2 hm) {
    hm = terrain_height(dir, to_terrain_params(t));
    float centered_height = (hm.x - 0.5) * 2.4;
    return dir * (radius + centered_height * radius * t.height_scale);
}

void write_invalid(uint rec) {
    pool[rec + 3] = float4(INVALID, 0.0, 0.0, 0.0);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void computeMain(uint3 tid: SV_DispatchThreadID) {
    ScatterParams p = params[0];
    uint total = p.chunk_count * p.capacity;
    if (tid.x >= total) {
        return;
    }
    uint ci = tid.x / p.capacity; // chunk index within the batch
    uint c = tid.x % p.capacity;  // candidate index within the slot region
    ChunkGpu ch = chunks[ci];
    ScatterAux ax = aux[ci];
    uint rec = (ch.slot * p.capacity + c) * 4;

    uint d = ch.level;
    uint L = p.lod_level;
    uint2 path = uint2(ax.path_lo, ax.path_hi);

    // ---- resolve this candidate's stable lattice cell ----
    float2 A;
    float2 B;
    float2 C;
    uint2 cell_path;
    uint k;
    bool needs_in_chunk_test = false;
    if (L >= d) {
        uint s = L - d;
        if (s > S_MAX) {
            write_invalid(rec);
            return;
        }
        uint cells = 1u << (2 * s);
        uint cell_sub = c / p.k_per_cell;
        k = c % p.k_per_cell;
        if (cell_sub >= cells) {
            write_invalid(rec);
            return;
        }
        // Cell triangle: descend the CHUNK corners by the sub bits.
        A = ch.bary0; B = ch.bary1; C = ch.bary2;
        descend(A, B, C, uint2(cell_sub, 0), s);
        cell_path = shl64(path, 2 * s);
        cell_path.x |= cell_sub;
    } else {
        // Deep chunk: it lies inside its level-L ancestor cell and hosts the
        // subset of that cell's K candidates that fall inside it.
        if (c >= p.k_per_cell) {
            write_invalid(rec);
            return;
        }
        k = c;
        cell_path = shr64(path, 2 * (d - L));
        // Cell triangle: descend the FACE root by the ancestor path.
        A = float2(0.0, 0.0);
        B = float2(1.0, 0.0);
        C = float2(0.0, 1.0);
        descend(A, B, C, cell_path, L);
        needs_in_chunk_test = true;
    }

    // ---- stable jitter + per-instance values (celestial_algo::candidate) ----
    uint s0 = seed0(ax.face, cell_path, k, p.seed);
    float u1 = u01(s0 ^ 1u);
    float u2 = u01(s0 ^ 2u);
    float su = sqrt(u1);
    float w0 = 1.0 - su;
    float w1 = su * (1.0 - u2);
    float w2 = su * u2;
    float2 fb = A * w0 + B * w1 + C * w2; // face-bary (wb, wc)
    float hash01 = u01(s0 ^ 0x68bc21ebu);
    float yaw = u01(s0 ^ 3u) * 6.28318530718;
    // Per-instance uniform jitter (0.8..1.2) times the layer's base scale.
    float scale = p.base_scale * (0.8 + 0.4 * u01(s0 ^ 4u));

    // Chunk-local (u, v) of the candidate — computed for BOTH paths (the deep
    // chunk one already needs the in-triangle test; the shallow one gets the
    // same coords, which are in-range by construction).
    float2 uv;
    bool inside = in_triangle_uv(fb, ch.bary0, ch.bary1, ch.bary2, uv);
    if (needs_in_chunk_test && !inside) {
        write_invalid(rec);
        return;
    }

    // ---- world position on the displaced terrain ----
    float radius = ch.frame_a.w;
    float3 base = project_bary(ch.frame_a.xyz, ch.frame_b.xyz, ch.frame_c.xyz,
                               radius, fb.x, fb.y);
    float3 pos = base;
    // Normalized terrain height (0..1) at this instance, stored in the pool so
    // the compact pass can gate on min_height/max_height LIVE. 0.5 when terrain
    // is disabled (no meaningful height — passes the default gates).
    float height01 = 0.5;
    TerrainGpu t = p.terrain;
    if (p.surface_enabled != 0.0 && radius > 0.0) {
        // ---- CPU-surface displacement — mirrors ChunkRealize EXACTLY ----
        // Nearest tile texel of the chunk-local (u, v), sampled from the same
        // per-slot elevation grid the terrain vertices were displaced with, so
        // instances sit ON the CPU ground (never floating/buried).
        float3 dir = normalize(base);
        uint tr = p.tile_res;
        uint tx = min(uint(clamp(uv.x, 0.0, 1.0) * float(tr)), tr - 1u);
        uint ty = min(uint(clamp(uv.y, 0.0, 1.0) * float(tr)), tr - 1u);
        uint g = ch.slot * tr * tr + ty * tr + tx;
        float meters = surface_height[g];
        pos = dir * radius * (1.0 + meters * p.surface_height_scale);
        // Re-normalize into the SAME 0..1 height space the procedural route
        // reports, so min_height/max_height keep their meaning across routes:
        // radius·(1 + (h01-0.5)·2.4·height_scale) == radius·(1 + m·surf_scale).
        float denom = 2.4 * t.height_scale;
        height01 = abs(denom) > 1.0e-9
            ? 0.5 + meters * p.surface_height_scale / denom
            : 0.5;
    } else if (radius > 0.0 && t.enabled != 0.0) {
        float3 dir = normalize(base);
        float2 hm;
        pos = displace(dir, radius, t, hm);
        height01 = hm.x;
    }

    // ---- radial-up basis with yaw + uniform scale ----
    float3 up = radius > 0.0 ? normalize(pos) : float3(0.0, 1.0, 0.0);
    float3 upref = abs(up.y) < 0.99 ? float3(0.0, 1.0, 0.0) : float3(1.0, 0.0, 0.0);
    float3 t0 = normalize(cross(upref, up));
    float3 b0 = cross(up, t0);
    float3 xn = t0 * cos(yaw) + b0 * sin(yaw);
    float3 zn = cross(xn, up);
    float3 X = xn * scale;
    float3 Y = up * scale;
    float3 Z = zn * scale;

    // Godot TRANSFORM_3D rows: row r = (X[r], Y[r], Z[r], origin[r]).
    // rec3 = (hash01 keep-value, normalized terrain height, unused, unused).
    pool[rec + 0] = float4(X.x, Y.x, Z.x, pos.x);
    pool[rec + 1] = float4(X.y, Y.y, Z.y, pos.y);
    pool[rec + 2] = float4(X.z, Y.z, Z.z, pos.z);
    pool[rec + 3] = float4(hash01, height01, 0.0, 0.0);
}
