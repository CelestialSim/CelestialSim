// Slang Compute Shader — Cubemap Terrain Normal Generator
// Generates cubemap faces with surface normals computed via finite differences
// on the displaced sphere terrain. Normals are encoded as (N * 0.5 + 0.5, 1.0)
// for storage in RGBA8 [0,1] range.

import terrain_noise_3d;

static const float height_scale_factor = 0.25;

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

// --- Displaced position on the sphere for a given direction ---

float3 displaced_position(float3 dir)
{
    float h = terrain_height_erosion_triplanar(dir, terrain_params).x;
    float centered_height = (h - 0.5) * 2.4;
    float displacement = centered_height * radius * height_scale_factor;
    return dir * (radius + displacement);
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
    out_cubemap[uint3(x, y, face_index)] = float4(normal * 0.5 + 0.5, 1.0);
}
