// Slang Compute Shader — Cubemap Terrain Color Generator
// Generates cubemap faces by projecting pixels onto a sphere and sampling
// Terrain3D-like height/erosion, then mapping to Terrain3D-like diffuse colors.

import terrain_noise_3d;

[[vk::binding(0, 0)]]
RWTexture2DArray<float4> out_cubemap;

[[vk::binding(1, 0)]]
ConstantBuffer<uint> face_index;

[[vk::binding(2, 0)]]
ConstantBuffer<uint> tex_size;

[[vk::binding(3, 0)]]
ConstantBuffer<float> radius;

[[vk::binding(4, 0)]]
ConstantBuffer<TerrainParams> terrain_params;

// --- Cube face → 3D direction ---

float3 cube_face_direction(uint face, float u, float v)
{
    // Vulkan cubemap convention (OpenGL 4.6 Table 8.18)
    switch (face)
    {
    case 0:
        return float3(1.0, -v, -u); // +X
    case 1:
        return float3(-1.0, -v, u); // -X
    case 2:
        return float3(u, 1.0, v); // +Y
    case 3:
        return float3(u, -1.0, -v); // -Y
    case 4:
        return float3(u, -v, 1.0); // +Z
    case 5:
        return float3(-u, -v, -1.0); // -Z
    default:
        return float3(0, 0, 0);
    }
}

// --- Main entry point ---

[shader("compute")]
[numthreads(8, 8, 1)]
void computeMain(uint3 DTid: SV_DispatchThreadID)
{
    uint x = DTid.x;
    uint y = DTid.y;

    if (x >= tex_size || y >= tex_size)
        return;

    // Map pixel to [-1, +1]
    float u = (float(x) + 0.5) / float(tex_size) * 2.0 - 1.0;
    float v = (float(y) + 0.5) / float(tex_size) * 2.0 - 1.0;

    // Get 3D direction on cube face, project to sphere.
    float3 cube_pt = cube_face_direction(face_index, u, v);
    float3 sphere_dir = normalize(cube_pt);

    // Terrain3D-style triplanar sampling on sphere direction.
    float2 hm = terrain_height_erosion_triplanar(sphere_dir, terrain_params);
    float breakup = terrain_breakup_triplanar(sphere_dir);
    float3 albedo = terrain_albedo(hm.x, hm.y, breakup, terrain_params);

    float4 color = float4(albedo, 1.0);

    out_cubemap[uint3(x, y, face_index)] = color;
}
