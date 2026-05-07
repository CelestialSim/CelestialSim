// ComputeDivPrefixAtomic — assigns compact 0-based division slots in parallel.
// Triangles marked in `t_to_div` atomically claim an output slot; `div_prefix[idx]`
// stores that exclusive prefix / slot index for each marked triangle.

[[vk::binding(0, 0)]]
StructuredBuffer<int> t_to_div;

[[vk::binding(1, 0)]]
RWStructuredBuffer<int> div_prefix;

[[vk::binding(2, 0)]]
RWStructuredBuffer<uint> div_count;

[[vk::binding(3, 0)]]
ConstantBuffer<uint> n_tris;

[shader("compute")]
[numthreads(64, 1, 1)]
void computeMain(uint3 DTid : SV_DispatchThreadID)
{
    uint idx = DTid.x;
    if (idx >= n_tris)
    {
        return;
    }

    if (t_to_div[idx] == 0)
    {
        div_prefix[idx] = 0;
        return;
    }

    uint slot;
    InterlockedAdd(div_count[0], 1u, slot);
    div_prefix[idx] = int(slot);
}
