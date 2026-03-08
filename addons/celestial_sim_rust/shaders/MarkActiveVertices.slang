// Marks vertices that belong to active (non-deactivated) triangles.

[[vk::binding(0, 0)]]
StructuredBuffer<int3> t_abc;

[[vk::binding(1, 0)]]
StructuredBuffer<int> t_deactivated;

[[vk::binding(2, 0)]]
RWStructuredBuffer<int> v_active_mask;

[[vk::binding(3, 0)]]
ConstantBuffer<uint> n_tris;

[[vk::binding(4, 0)]]
ConstantBuffer<uint> n_verts;

[shader("compute")]
[numthreads(64, 1, 1)]
void computeMain(uint3 DTid: SV_DispatchThreadID)
{
    if (DTid.x >= n_tris)
    {
        return;
    }

    uint idx = DTid.x;
    if (t_deactivated[idx] != 0)
    {
        return;
    }

    int3 abc = t_abc[idx];
    if (abc.x >= 0 && abc.x < int(n_verts)) v_active_mask[abc.x] = 1;
    if (abc.y >= 0 && abc.y < int(n_verts)) v_active_mask[abc.y] = 1;
    if (abc.z >= 0 && abc.z < int(n_verts)) v_active_mask[abc.z] = 1;
}
