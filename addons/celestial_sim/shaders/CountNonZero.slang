[[vk::binding(0, 0)]]
StructuredBuffer<uint> mask;

[[vk::binding(1, 0)]]
RWStructuredBuffer<uint> out_count;

[[vk::binding(2, 0)]]
ConstantBuffer<uint> n_tris;

[shader("compute")]
[numthreads(64, 1, 1)]
void computeMain(uint3 DTid: SV_DispatchThreadID)
{
    uint idx = DTid.x;
    if (idx >= n_tris)
    {
        return;
    }

    if (mask[idx] != 0u)
    {
        InterlockedAdd(out_count[0], 1u);
    }
}
