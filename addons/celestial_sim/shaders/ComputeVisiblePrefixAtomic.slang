// ComputeVisiblePrefixAtomic — computes visible mask and compaction indices in parallel.
// A triangle is visible if it has NOT been divided and NOT been deactivated.
// Visible triangles atomically claim a compact slot; `visible_prefix[idx]` stores
// `slot + 1` (matching the old inclusive-prefix convention used by the final-output shaders).

[[vk::binding(0, 0)]]
StructuredBuffer<int> t_divided;

[[vk::binding(1, 0)]]
StructuredBuffer<int> t_deactivated;

[[vk::binding(2, 0)]]
RWStructuredBuffer<int> visible_mask;

[[vk::binding(3, 0)]]
RWStructuredBuffer<int> visible_prefix;

[[vk::binding(4, 0)]]
RWStructuredBuffer<uint> visible_count;

[[vk::binding(5, 0)]]
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

    int vis = (t_divided[idx] == 0 && t_deactivated[idx] == 0) ? 1 : 0;
    visible_mask[idx] = vis;

    if (vis == 0)
    {
        visible_prefix[idx] = 0;
        return;
    }

    uint slot;
    InterlockedAdd(visible_count[0], 1u, slot);
    visible_prefix[idx] = int(slot + 1u);
}
