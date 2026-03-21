// Compacts active triangles into a dense range, remapping parents and children.
// Neighbour indices are remapped when the neighbour is active; inactive ones become -1.

[[vk::binding(0, 0)]]
StructuredBuffer<int> t_deactivated_src;

[[vk::binding(1, 0)]]
StructuredBuffer<int> t_active_prefix;

[[vk::binding(2, 0)]]
StructuredBuffer<int3> t_abc_src;

[[vk::binding(3, 0)]]
StructuredBuffer<int> t_divided_src;

[[vk::binding(4, 0)]]
StructuredBuffer<int> t_lv_src;

[[vk::binding(5, 0)]]
StructuredBuffer<int> t_to_div_src;

[[vk::binding(6, 0)]]
StructuredBuffer<int> t_to_merge_src;

[[vk::binding(7, 0)]]
StructuredBuffer<int> t_ico_idx_src;

[[vk::binding(8, 0)]]
StructuredBuffer<int> t_neigh_ab_src;

[[vk::binding(9, 0)]]
StructuredBuffer<int> t_neigh_bc_src;

[[vk::binding(10, 0)]]
StructuredBuffer<int> t_neigh_ca_src;

[[vk::binding(11, 0)]]
StructuredBuffer<int> t_a_t_src;

[[vk::binding(12, 0)]]
StructuredBuffer<int> t_b_t_src;

[[vk::binding(13, 0)]]
StructuredBuffer<int> t_c_t_src;

[[vk::binding(14, 0)]]
StructuredBuffer<int> t_center_t_src;

[[vk::binding(15, 0)]]
StructuredBuffer<int> t_parent_src;

[[vk::binding(16, 0)]]
RWStructuredBuffer<int3> t_abc_dst;

[[vk::binding(17, 0)]]
RWStructuredBuffer<int> t_divided_dst;

[[vk::binding(18, 0)]]
RWStructuredBuffer<int> t_lv_dst;

[[vk::binding(19, 0)]]
RWStructuredBuffer<int> t_to_div_dst;

[[vk::binding(20, 0)]]
RWStructuredBuffer<int> t_to_merge_dst;

[[vk::binding(21, 0)]]
RWStructuredBuffer<int> t_ico_idx_dst;

[[vk::binding(22, 0)]]
RWStructuredBuffer<int> t_neigh_ab_dst;

[[vk::binding(23, 0)]]
RWStructuredBuffer<int> t_neigh_bc_dst;

[[vk::binding(24, 0)]]
RWStructuredBuffer<int> t_neigh_ca_dst;

[[vk::binding(25, 0)]]
RWStructuredBuffer<int> t_a_t_dst;

[[vk::binding(26, 0)]]
RWStructuredBuffer<int> t_b_t_dst;

[[vk::binding(27, 0)]]
RWStructuredBuffer<int> t_c_t_dst;

[[vk::binding(28, 0)]]
RWStructuredBuffer<int> t_center_t_dst;

[[vk::binding(29, 0)]]
RWStructuredBuffer<int> t_parent_dst;

[[vk::binding(30, 0)]]
RWStructuredBuffer<int> t_deactivated_dst;

[[vk::binding(31, 0)]]
ConstantBuffer<uint> n_tris_total;

[[vk::binding(32, 0)]]
ConstantBuffer<uint> n_tris_active;

// Returns -1 for inactive or out-of-range references; otherwise returns the compacted index.
int remap_if_active(int idx)
{
    if (idx < 0)
    {
        return -1;
    }

    if (idx >= int(n_tris_total))
    {
        return -1;
    }

    if (t_deactivated_src[idx] != 0)
    {
        return -1;
    }

    return t_active_prefix[idx] - 1;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void computeMain(uint3 DTid: SV_DispatchThreadID)
{
    if (DTid.x >= n_tris_total)
    {
        return;
    }

    uint src_idx = DTid.x;

    if (t_deactivated_src[src_idx] != 0)
    {
        return;
    }

    uint dst_idx = (uint)(t_active_prefix[src_idx] - 1);

    if (dst_idx >= n_tris_active)
    {
        return;
    }

    // Copy per-triangle data
    t_abc_dst[dst_idx] = t_abc_src[src_idx];
    t_divided_dst[dst_idx] = t_divided_src[src_idx];
    t_lv_dst[dst_idx] = t_lv_src[src_idx];
    t_to_div_dst[dst_idx] = t_to_div_src[src_idx];
    t_to_merge_dst[dst_idx] = t_to_merge_src[src_idx];
    t_ico_idx_dst[dst_idx] = t_ico_idx_src[src_idx];
    t_deactivated_dst[dst_idx] = 0;

    // Remap hierarchy pointers
    t_parent_dst[dst_idx] = remap_if_active(t_parent_src[src_idx]);
    t_a_t_dst[dst_idx] = remap_if_active(t_a_t_src[src_idx]);
    t_b_t_dst[dst_idx] = remap_if_active(t_b_t_src[src_idx]);
    t_c_t_dst[dst_idx] = remap_if_active(t_c_t_src[src_idx]);
    t_center_t_dst[dst_idx] = remap_if_active(t_center_t_src[src_idx]);

    // Remap neighbours (will be refined by a later pass if needed)
    t_neigh_ab_dst[dst_idx] = remap_if_active(t_neigh_ab_src[src_idx]);
    t_neigh_bc_dst[dst_idx] = remap_if_active(t_neigh_bc_src[src_idx]);
    t_neigh_ca_dst[dst_idx] = remap_if_active(t_neigh_ca_src[src_idx]);
}
