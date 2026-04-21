// Slang Compute Shader — Snapshot Patch Terrain Normal
// Generates a 2D normal texture for a spherical region using terrain noise.
// Reuses the shared tangent-plane utils (bindings 1-6) and the
// terrain_noise_3d module for identical visual output to TerrainNormal.slang.

import terrain_noise_3d;
import snapshot_patch_utils;

// Binding 0: output texture (entry-point specific)
[[vk::binding(0, 0)]]
RWTexture2D<float4> out_patch;

// Binding 7: terrain parameters (noise-specific)
[[vk::binding(7, 0)]]
ConstantBuffer<TerrainParams> terrain_params;

static const float height_scale_factor = 0.25;

/// Displaced position on the sphere for a given direction.
float3 displaced_position(float3 dir)
{
    float h = terrain_height_erosion_triplanar(dir, terrain_params).x;
    float centered_height = (h - 0.5) * 2.4;
    float displacement = centered_height * radius * height_scale_factor;
    return dir * (radius + displacement);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void computeMain(uint3 DTid: SV_DispatchThreadID)
{
    uint x = DTid.x;
    uint y = DTid.y;

    if (x >= tex_size || y >= tex_size)
        return;

    float3 sphere_dir = snapshot_world_direction(uint2(x, y));

    // Compute tangent frame on the sphere surface
    float3 up = abs(sphere_dir.y) < 0.999 ? float3(0, 1, 0) : float3(1, 0, 0);
    float3 tangent1 = normalize(cross(up, sphere_dir));
    float3 tangent2 = cross(sphere_dir, tangent1);

    // Finite-difference step scaled to texel size
    float eps = 1.0 / float(tex_size);

    // Sample displaced positions at center and two offset directions
    float3 p0 = displaced_position(sphere_dir);
    float3 p1 = displaced_position(normalize(sphere_dir + eps * tangent1));
    float3 p2 = displaced_position(normalize(sphere_dir + eps * tangent2));

    // Surface normal via cross product of finite differences
    float3 normal = normalize(cross(p1 - p0, p2 - p0));

    // Encode normal from [-1,1] to [0,1] for RGBA8 storage
    out_patch[uint2(x, y)] = float4(normal * 0.5 + 0.5, 1.0);
}
