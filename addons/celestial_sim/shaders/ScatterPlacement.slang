// ScatterPlacement.slang — triangle-indexed GPU scatter for CelestialSim.
//
// One thread per triangle in the adaptive-LOD `CesState` mesh. Triangles whose
// LOD level matches `subdivision_level` (and that are not deactivated) are
// candidates. Per-thread:
//   1. Compute the triangle center and project it onto the sphere surface.
//   2. Sample the terrain noise inline to derive a continuous biome density.
//   3. Sample the 3D blue-noise volume at the same direction → threshold.
//   4. Accept iff `density > threshold`.
//   5. Atomically claim a slot in `out_transforms` and write a packed mat3x4.
//
// CPU reads `out_counter[0]` and uses it as `MultiMesh.set_instance_count(...)`,
// so every visible instance corresponds to a real spawn — no zero-pad slots.
//
// Binding layout (mirrored in `rust/src/layers/scatter.rs`):
//   0: ConstantBuffer<ScatterParams>
//   1: ConstantBuffer<TerrainParams>
//   2: RWStructuredBuffer<float> out_transforms — 12 floats per instance, mat3x4 row-major
//   3: RWStructuredBuffer<uint>  out_counter    — single u32, atomic-add slot allocator
//   4: StructuredBuffer<float4>  v_pos          — vertex positions (CesState)
//   5: StructuredBuffer<int4>    t_abc          — triangle vertex indices (CesState)
//   6: StructuredBuffer<int>     t_lv           — per-triangle LOD level
//   7: StructuredBuffer<int>     t_deactivated  — non-zero → skip
//   8: ConstantBuffer<uint>      n_tris         — number of valid triangles
//   9: StructuredBuffer<uint>    blue_noise     — packed 4-bytes-per-uint cube; size = noise_dim^3 / 4

import terrain_noise_3d;

static const float SCATTER_PI = 3.14159265358979;

// Slang ScatterParams. Layout MUST match `ScatterParams` in
// `rust/src/layers/scatter.rs` (verified by `scatter_params_pod_size_matches_slang_struct`).
public struct ScatterParams
{
    float planet_radius;
    uint  seed;
    int   subdivision_level;
    float noise_strength;

    float height_min;
    float height_max;
    float albedo_tolerance;
    uint  noise_dim;          // cube side of the blue-noise volume

    float3 albedo_target;
    float  _pad;
};

[[vk::binding(0, 0)]] ConstantBuffer<ScatterParams> scatter_params;
[[vk::binding(1, 0)]] ConstantBuffer<TerrainParams> terrain_params;
[[vk::binding(2, 0)]] RWStructuredBuffer<float>     out_transforms;
[[vk::binding(3, 0)]] RWStructuredBuffer<uint>      out_counter;
[[vk::binding(4, 0)]] StructuredBuffer<float4>      v_pos;
[[vk::binding(5, 0)]] StructuredBuffer<int4>        t_abc;
[[vk::binding(6, 0)]] StructuredBuffer<int>         t_lv;
[[vk::binding(7, 0)]] StructuredBuffer<int>         t_deactivated;
[[vk::binding(8, 0)]] ConstantBuffer<uint>          n_tris;
[[vk::binding(9, 0)]] StructuredBuffer<uint>        blue_noise;

// --- Helpers ---

// Godot MultiMesh 3D layout: 12 floats per instance, row-major mat3x4.
//   [X.x, Y.x, Z.x, ox, X.y, Y.y, Z.y, oy, X.z, Y.z, Z.z, oz]
// where (X, Y, Z) are the basis columns. Mesh-local +Y aligns with `b1`.
void write_transform(uint slot, float3 b0, float3 b1, float3 b2, float3 origin)
{
    uint base = slot * 12u;
    out_transforms[base + 0u]  = b0.x;
    out_transforms[base + 1u]  = b1.x;
    out_transforms[base + 2u]  = b2.x;
    out_transforms[base + 3u]  = origin.x;
    out_transforms[base + 4u]  = b0.y;
    out_transforms[base + 5u]  = b1.y;
    out_transforms[base + 6u]  = b2.y;
    out_transforms[base + 7u]  = origin.y;
    out_transforms[base + 8u]  = b0.z;
    out_transforms[base + 9u]  = b1.z;
    out_transforms[base + 10u] = b2.z;
    out_transforms[base + 11u] = origin.z;
}

// Sample the blue-noise volume by sphere direction. UVW = dir * 0.5 + 0.5
// → integer voxel coords mod noise_dim. Returns a float in [0, 1].
float sample_blue_noise(float3 dir)
{
    uint dim = scatter_params.noise_dim;
    if (dim == 0u) return 0.5;
    float3 uvw = dir * 0.5 + 0.5;
    int3 c = int3(floor(uvw * float(dim))) % int(dim);
    if (c.x < 0) c.x += int(dim);
    if (c.y < 0) c.y += int(dim);
    if (c.z < 0) c.z += int(dim);
    uint flat_idx = uint(c.z) * dim * dim + uint(c.y) * dim + uint(c.x);
    uint packed = blue_noise[flat_idx >> 2u];
    uint byte = (packed >> ((flat_idx & 3u) * 8u)) & 0xFFu;
    return float(byte) * (1.0 / 255.0);
}

// Smooth biome density ∈ [0, 1] from height + albedo similarity. Hard-rejected
// outside [height_min - eps, height_max + eps] window.
float biome_density(float3 dir, float height, float erosion_mask)
{
    const float EPS = 0.05;
    float h_min = scatter_params.height_min;
    float h_max = scatter_params.height_max;
    float d_h = smoothstep(h_min - EPS, h_min + EPS, height)
              * (1.0 - smoothstep(h_max - EPS, h_max + EPS, height));
    if (scatter_params.albedo_tolerance >= 2.0)
    {
        return d_h;
    }
    float breakup = terrain_breakup_triplanar(dir);
    float3 albedo = terrain_albedo_slope(height, erosion_mask, breakup, 1.0, terrain_params);
    float3 diff = albedo - scatter_params.albedo_target;
    float dist = length(diff);
    float d_alb = 1.0 - smoothstep(0.0, scatter_params.albedo_tolerance, dist);
    return d_h * d_alb;
}

// --- Entry ---

[shader("compute")]
[numthreads(64, 1, 1)]
void computeMain(uint3 DTid: SV_DispatchThreadID)
{
    uint tri_idx = DTid.x;
    if (tri_idx >= n_tris) return;

    // Filter to the configured LOD level (and not deactivated). We accept
    // `t_divided != 0` so trees that have been further subdivided for visual
    // detail still render — the parent triangle stays in the buffer until merged.
    if (t_lv[tri_idx] != scatter_params.subdivision_level) return;
    if (t_deactivated[tri_idx] != 0) return;

    int4 abc = t_abc[tri_idx];
    float3 pa = v_pos[abc.x].xyz;
    float3 pb = v_pos[abc.y].xyz;
    float3 pc = v_pos[abc.z].xyz;
    float3 center = (pa + pb + pc) * (1.0 / 3.0);
    float3 dir = normalize(center);

    // Sample terrain noise inline (matches mask layer / planet).
    float2 hm = terrain_height_erosion_triplanar(dir, terrain_params);
    float height = hm.x;
    float erosion = hm.y;

    // Continuous biome density.
    float density = biome_density(dir, height, erosion);
    if (density <= 0.0) return;

    // Blue-noise threshold (scaled by noise_strength).
    float threshold = sample_blue_noise(dir) * scatter_params.noise_strength;
    if (density <= threshold) return;

    // Accepted — claim a slot.
    uint slot;
    InterlockedAdd(out_counter[0], 1u, slot);

    // Displaced world position (matches TerrainHeight.slang formula).
    float displacement = (height - 0.5) * 2.4 * scatter_params.planet_radius * 0.25;
    float3 world_pos = dir * (scatter_params.planet_radius + displacement);

    // Random yaw per triangle.
    float yaw = hash3(float3(float(tri_idx), float(scatter_params.seed), 0.0)).z * SCATTER_PI;
    float cy = cos(yaw);
    float sy = sin(yaw);

    // Build orthonormal basis with `up = dir`. Same convention as before.
    float3 up = dir;
    float3 ref = abs(up.y) < 0.95 ? float3(0.0, 1.0, 0.0) : float3(1.0, 0.0, 0.0);
    float3 tangent = normalize(cross(ref, up));
    float3 bitangent = cross(up, tangent);
    float3 t_rot = tangent * cy + bitangent * sy;
    float3 b_rot = cross(t_rot, up); // right-handed: X × Y = Z

    write_transform(slot, t_rot, up, b_rot, world_pos);
}
