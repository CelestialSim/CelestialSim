// Remap triangle vertex indices using the compacted vertex prefix buffer.

[[vk::binding(0, 0)]]
StructuredBuffer<int3> t_abc_src;

[[vk::binding(1, 0)]]
StructuredBuffer<int> v_active_prefix;

[[vk::binding(2, 0)]]
RWStructuredBuffer<int3> t_abc_dst;

[[vk::binding(3, 0)]]
ConstantBuffer<uint> n_tris;

[[vk::binding(4, 0)]]
ConstantBuffer<uint> n_verts_active;

int remap_vert(int idx)
{
    if (idx < 0)
    {
        return -1;
    }

    return v_active_prefix[idx] - 1;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void computeMain(uint3 DTid: SV_DispatchThreadID)
{
    if (DTid.x >= n_tris)
    {
        return;
    }

    uint t_idx = DTid.x;
    int3 abc = t_abc_src[t_idx];

    int3 remapped;
    remapped.x = remap_vert(abc.x);
    remapped.y = remap_vert(abc.y);
    remapped.z = remap_vert(abc.z);

    t_abc_dst[t_idx] = remapped;
}
