// ComputeVisiblePrefixSum — Computes visible mask AND inclusive prefix sum in one pass.
// A triangle is visible if it has NOT been divided and NOT been deactivated.
// Single-thread sequential pass: computes mask and running sum together.

[[vk::binding(0, 0)]]
StructuredBuffer<int> t_divided;

[[vk::binding(1, 0)]]
StructuredBuffer<int> t_deactivated;

[[vk::binding(2, 0)]]
RWStructuredBuffer<int> visible_mask;

[[vk::binding(3, 0)]]
RWStructuredBuffer<int> visible_prefix;

[[vk::binding(4, 0)]]
ConstantBuffer<uint> n_tris;

[shader("compute")]
[numthreads(1, 1, 1)]
void computeMain(uint3 DTid: SV_DispatchThreadID)
{
    int sum = 0;
    for (uint i = 0; i < n_tris; i++)
    {
        int vis = (t_divided[i] == 0 && t_deactivated[i] == 0) ? 1 : 0;
        visible_mask[i] = vis;
        sum += vis;
        visible_prefix[i] = sum;
    }
}
