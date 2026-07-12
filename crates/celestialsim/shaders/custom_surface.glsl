#version 450

// ============================================================================
// CelestialSim custom-surface template (runtime-compiled by Godot's
// RenderingDevice). The user's terrain code is spliced in at the user-code
// marker line below by crate::custom_surface::assemble_source (exactly one
// occurrence of that marker may exist in this file).
//
// One invocation = one tile texel of one chunk in the realize batch. We
// reconstruct the texel's world DIRECTION with the SAME face-barycentric ---
// gnomonic mapping the built-in realize/bake shaders use (normalize of the
// linear corner blend --- NEVER slerp), call the user's terrain functions, and
// write the three per-slot surface buffers the realize/bake shaders already
// read when surface_enabled == 1:
//   surface_height[g] = displacement as a FRACTION of radius (realize clamps
//                       to >= 0 and does radius * (1 + h * height_scale))
//   surface_color[g]  = rgba8 albedo, little-endian (R = byte 0)
//   surface_normal[g] = rgba8 world normal, encoded * 0.5 + 0.5
// with g = slot * tile_res^2 + ty * tile_res + tx.
// ============================================================================

layout(local_size_x = 64) in;

// std430 --- byte-identical to ChunkGpu (src/chunk_descriptors.rs, 96 bytes) and
// the chunks binding of ChunkRealize/ChunkTileBake.
struct ChunkGpu {
    vec4 frame_a; // xyz = face corner A, w = sphere radius (0 => flat face)
    vec4 frame_b; // xyz = face corner B
    vec4 frame_c; // xyz = face corner C
    vec2 bary0;   // chunk corner 0 in FACE-barycentric (wb, wc)
    vec2 bary1;   // chunk corner 1
    vec2 bary2;   // chunk corner 2
    uint slot;    // vertex-pool / atlas slot
    uint level;   // LOD level (quadtree depth)
    uint res;     // edge resolution
    uint _pad0;
    uint _pad1;
    uint _pad2;
};

layout(set = 0, binding = 0, std430) readonly buffer Chunks { ChunkGpu chunks[]; };
layout(set = 0, binding = 1, std430) buffer SurfColor  { uint  surface_color[]; };
layout(set = 0, binding = 2, std430) buffer SurfHeight { float surface_height[]; };
layout(set = 0, binding = 3, std430) buffer SurfNormal { uint  surface_normal[]; };
layout(set = 0, binding = 4, std430) readonly buffer Params {
    uint  chunk_count;
    uint  tile_res;
    float water_height;
    float height_scale;
    // Up to 16 generic user params — one per `@export var name: float` on the
    // GDScript builder, spliced in as `#define NAME (P.cels_user[i])` below.
    float cels_user[16];
} P;

// ---- globals the user's terrain functions may read (set per texel in main) ---
float CELS_RADIUS;       // this chunk's sphere radius (world units)
float CELS_WATER_HEIGHT; // normalized sea level 0..1 (layer's water_height)
float CELS_HEIGHT_SCALE; // geometry displacement multiplier (layer's height_scale)

uint cels_pack_rgba8(vec3 c) {
    uvec3 q = uvec3(clamp(c, 0.0, 1.0) * 255.0 + 0.5);
    return q.x | (q.y << 8) | (q.z << 16) | (255u << 24);
}

// Per-param #defines (one per `@export var name: float` on the builder) are
// spliced at this marker, BEFORE the user code that references them.
// __CELS_USER_DEFINES__

// ============================================================================
// USER CODE --- must define:
//   float terrain_height(vec3 dir);          // displacement fraction of radius
//   vec3  terrain_color (vec3 dir, float h);  // albedo 0..1
// and MAY define (guard with #define CELS_CUSTOM_NORMAL):
//   vec3  terrain_normal(vec3 dir, float h);  // world normal
// ============================================================================
// __CELS_USER_CODE__
// ============================================================================

#ifndef CELS_CUSTOM_NORMAL
// Auto normal: finite-difference the user's height field over a small angular
// step, bending the two chunk-tangent chords into the surface (matches the
// built-in procedural bake). Uses the FULL relief so mountains shade strongly
// even when height_scale keeps the silhouette round.
vec3 cels_auto_normal(vec3 dir, float h0) {
    vec3 up = abs(dir.y) < 0.99 ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0);
    vec3 tg = normalize(cross(up, dir));
    vec3 bt = cross(dir, tg);
    const float eps = 0.001;
    vec3 d1 = normalize(dir + tg * eps);
    vec3 d2 = normalize(dir + bt * eps);
    float R = max(CELS_RADIUS, 1.0);
    vec3 p0 = dir * (R * (1.0 + h0));
    vec3 p1 = d1 * (R * (1.0 + terrain_height(d1)));
    vec3 p2 = d2 * (R * (1.0 + terrain_height(d2)));
    vec3 n = normalize(cross(p1 - p0, p2 - p0));
    return dot(n, dir) < 0.0 ? -n : n;
}
#endif

void main() {
    uint gid = gl_GlobalInvocationID.x;
    uint tr = P.tile_res;
    uint tile_texels = tr * tr;
    uint total = P.chunk_count * tile_texels;
    if (tr == 0u || gid >= total) {
        return;
    }

    uint ci = gid / tile_texels;    // chunk index within the batch
    uint texel = gid % tile_texels; // local tile texel
    uint tx = texel % tr;
    uint ty = texel / tr;

    // Texel-centre chunk-local (u, v); the square's lower-left triangle
    // (u + v <= 1) covers the chunk. Out-of-triangle texels fold onto the
    // diagonal so edge vertices / the bilinear atlas read valid data.
    float u = (float(tx) + 0.5) / float(tr);
    float v = (float(ty) + 0.5) / float(tr);
    if (u + v > 1.0) {
        float s = u + v;
        u /= s;
        v /= s;
    }

    ChunkGpu ch = chunks[ci];
    float radius = ch.frame_a.w;

    // chunk-local (u,v) -> face-barycentric -> gnomonic world direction.
    float wa = 1.0 - u - v;
    vec2 fbc = wa * ch.bary0 + u * ch.bary1 + v * ch.bary2;
    vec3 lin = ch.frame_a.xyz * (1.0 - fbc.x - fbc.y)
             + ch.frame_b.xyz * fbc.x
             + ch.frame_c.xyz * fbc.y;
    vec3 dir = normalize(lin);

    CELS_RADIUS = radius;
    CELS_WATER_HEIGHT = P.water_height;
    CELS_HEIGHT_SCALE = P.height_scale;

    float h = terrain_height(dir);
    vec3 col = terrain_color(dir, h);
#ifdef CELS_CUSTOM_NORMAL
    vec3 nrm = normalize(terrain_normal(dir, h));
#else
    vec3 nrm = cels_auto_normal(dir, h);
#endif

    uint g = ch.slot * tile_texels + ty * tr + tx;
    surface_height[g] = h;
    surface_color[g] = cels_pack_rgba8(col);
    surface_normal[g] = cels_pack_rgba8(nrm * 0.5 + 0.5);
}
