// Slang Compute Shader — Snapshot Patch Terrain Color
// Generates a 2D texture for a spherical region using terrain noise.
// Reuses the shared tangent-plane utils (bindings 1-6) and the
// terrain_noise_3d module for identical visual output to TerrainColor.slang.

import terrain_noise_3d;
import snapshot_patch_utils;

// Binding 0: output texture (entry-point specific)
[[vk::binding(0, 0)]]
RWTexture2D<float4> out_patch;

// Binding 7: terrain parameters (noise-specific)
[[vk::binding(7, 0)]]
ConstantBuffer<TerrainParams> terrain_params;

[shader("compute")]
[numthreads(8, 8, 1)]
void computeMain(uint3 DTid: SV_DispatchThreadID)
{
    uint x = DTid.x;
    uint y = DTid.y;

    if (x >= tex_size || y >= tex_size)
        return;

    float3 sphere_dir = snapshot_world_direction(uint2(x, y));

    // Same computation as TerrainColor.slang
    float2 hm = terrain_height_erosion_triplanar(sphere_dir, terrain_params);
    float breakup = terrain_breakup_triplanar(sphere_dir);
    float3 albedo = terrain_albedo(hm.x, hm.y, breakup, terrain_params);

    float4 color = float4(albedo, 1.0);

    // Debug: red border (2-pixel wide band at edges, toggled by show_border)
    if (show_border != 0 && is_border_pixel(uint2(x, y), 2))
        color = float4(1.0, 0.0, 0.0, 1.0);

    out_patch[uint2(x, y)] = color;
}
