// Compacts active vertices into a dense range using a CPU-built prefix sum.

[[vk::binding(0, 0)]]
StructuredBuffer<int> v_active_mask;

[[vk::binding(1, 0)]]
StructuredBuffer<int> v_active_prefix;

[[vk::binding(2, 0)]]
StructuredBuffer<float4> v_pos_src;

[[vk::binding(3, 0)]]
StructuredBuffer<int> v_update_mask_src;

[[vk::binding(4, 0)]]
RWStructuredBuffer<float4> v_pos_dst;

[[vk::binding(5, 0)]]
RWStructuredBuffer<int> v_update_mask_dst;

[[vk::binding(6, 0)]]
ConstantBuffer<uint> n_verts_total;

[[vk::binding(7, 0)]]
ConstantBuffer<uint> n_verts_active;

[shader("compute")]
[numthreads(64, 1, 1)]
void computeMain(uint3 DTid: SV_DispatchThreadID)
{
    if (DTid.x >= n_verts_total)
    {
        return;
    }

    uint src_idx = DTid.x;
    if (v_active_mask[src_idx] == 0)
    {
        return;
    }

    uint dst_idx = (uint)(v_active_prefix[src_idx] - 1);
    if (dst_idx >= n_verts_active)
    {
        return;
    }

    v_pos_dst[dst_idx] = v_pos_src[src_idx];
    v_update_mask_dst[dst_idx] = v_update_mask_src[src_idx];
}
