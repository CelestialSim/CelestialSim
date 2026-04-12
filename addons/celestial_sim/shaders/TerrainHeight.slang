// TerrainHeight.slang — Radial height displacement using 3D FBM noise

import terrain_noise_3d;

static const float height_scale_factor = 0.25;

// Vertices buffer (read-write)
[[vk::binding(0, 0)]]
RWStructuredBuffer<float3> vertices;

[[vk::binding(1, 0)]]
StructuredBuffer<int> v_update_mask;

// Uniform buffer containing 'radius'
[[vk::binding(2, 0)]]
ConstantBuffer<float> radius;

[[vk::binding(3, 0)]]
ConstantBuffer<uint> n_verts;

[[vk::binding(4, 0)]]
ConstantBuffer<TerrainParams> terrain_params;

[shader("compute")]
[numthreads(12, 1, 1)]
void computeMain(uint3 DTid: SV_DispatchThreadID)
{
    uint idx = DTid.x;
    if (idx >= n_verts)
        return;

    float3 sphere_dir = normalize(vertices[idx]);
    float2 hm = terrain_height_erosion_triplanar(sphere_dir, terrain_params);

    // Use macro terrain height only for geometry displacement.
    // Fine breakup detail lives in the texture, not the mesh.
    float centered_height = (hm.x - 0.5) * 2.4;
    float displacement = centered_height * radius * height_scale_factor;

    vertices[idx] = sphere_dir * (radius + displacement);
}
