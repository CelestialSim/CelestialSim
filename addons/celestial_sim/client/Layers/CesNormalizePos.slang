// Slang Compute Shader

// Vertices buffer (read-write)
[[vk::binding(0, 0)]]
RWStructuredBuffer<float3> vertices;

[[vk::binding(1, 0)]]
StructuredBuffer<int> v_update_mask;

// // Normals buffer (read-only)
// [[vk::binding(1, 0)]]
// StructuredBuffer<int> indices;

// Uniform buffer containing 'radius'
[[vk::binding(2, 0)]]
ConstantBuffer<float> radius;

[[vk::binding(3, 0)]]
ConstantBuffer<uint> n_verts;

// Main compute shader function
[shader("compute")]
[numthreads(12, 1, 1)]
void computeMain(uint3 DTid: SV_DispatchThreadID)
{
    uint idx = DTid.x;
    if (idx >= n_verts || v_update_mask[idx] == 0) return;
    if (idx >= n_verts) return;

    // Updates position of vertices to make it a sphere
    // vertices[idx] = normalize(vertices[idx]) * radius;
    // if (lv == 0) {
    //     vertices[idx] = normalize(vertices[idx]) * radius * 0.01;
    // }
    // else if (lv == 1) {
    //     vertices[idx] = normalize(vertices[idx]) * radius * 0.5;
    // } else {
    //     vertices[idx] = normalize(vertices[idx]) * radius ;
    // }
    vertices[idx] = normalize(vertices[idx]) * radius;
}
