// Vertices buffer (read-write)
[[vk::binding(0, 0)]]
RWStructuredBuffer<float3> vertices;

[[vk::binding(1, 0)]]
StructuredBuffer<float3> updatedVertices;

[[vk::binding(2, 0)]]
RWStructuredBuffer<int> indicesMask;

[[vk::binding(3, 0)]]
StructuredBuffer<int> updatedIndices;

[[vk::binding(4, 0)]]
StructuredBuffer<int3> t_abc;

[[vk::binding(5, 0)]]
ConstantBuffer<uint> totalTris;


// Main compute shader function
[shader("compute")]
[numthreads(32, 1, 1)]
void computeMain(uint3 DTid : SV_DispatchThreadID)
{
    if (DTid.x >= totalTris ) return;
    uint vidx = updatedIndices[DTid.x];
    indicesMask[vidx] = 0;
    vertices[vidx] = updatedVertices[DTid.x];
}