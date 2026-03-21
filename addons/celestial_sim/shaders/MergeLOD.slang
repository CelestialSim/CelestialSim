
// T_ABC buffer (read-write)
[[vk::binding(0, 0)]]
StructuredBuffer<int3> t_abc;

// T_ABC buffer (read-write)
[[vk::binding(1, 0)]]
RWStructuredBuffer<int> t_divided;

[[vk::binding(2, 0)]]
ConstantBuffer<uint> n_tris_to_merge;

[[vk::binding(3, 0)]]
StructuredBuffer<int> t_neigh_ab;

[[vk::binding(4, 0)]]
StructuredBuffer<int> t_neigh_bc;

[[vk::binding(5, 0)]]
StructuredBuffer<int> t_neigh_ca;

// TODO: remove write only for debugging purposes
[[vk::binding(6, 0)]]
StructuredBuffer<int> t_a_t;

// TODO: remove write only for debugging purposes
[[vk::binding(7, 0)]]
StructuredBuffer<int> t_b_t;

// TODO: remove write only for debugging purposes
[[vk::binding(8, 0)]]
StructuredBuffer<int> t_c_t;

// TODO: remove write only for debugging purposes
[[vk::binding(9, 0)]]
StructuredBuffer<int> t_center_t;

[[vk::binding(10, 0)]]
StructuredBuffer<uint> idxs_to_merge;

[[vk::binding(11, 0)]]
RWStructuredBuffer<uint> t_to_merge_mask;

[[vk::binding(12, 0)]]
StructuredBuffer<float> tris_out;

[[vk::binding(13, 0)]]
RWStructuredBuffer<int> t_deactivated;

[[vk::binding(14, 0)]]
StructuredBuffer<uint> t_lv;

[[vk::binding(15, 0)]]
StructuredBuffer<uint> v_to_update_mask;

[[vk::binding(16, 0)]]
StructuredBuffer<float3> v_pos;

[[vk::binding(17, 0)]]
StructuredBuffer<uint> t_parent;

enum Edge
{
    ab,
    bc,
    ca
}

bool divided(uint idx)
{
    return t_divided[idx] == 1 && t_to_merge_mask[idx] == 0;
}

// void updateAllNeights(uint t_to_update_idx, uint this_idx) {
//     updateNeights(t_to_update_idx, t_a_t[this_idx], this_idx);
//     updateNeights(t_to_update_idx, t_b_t[this_idx], this_idx);
//     updateNeights(t_to_update_idx, t_c_t[this_idx], this_idx);
// }

// void updateNeights(uint idx, uint oldNeight, uint newNeight)
// {

//     var abc = t_abc[idx];
//     var parent_verts = t_abc[t_parent[idx]];
//     // replace edge that had oldNeight with newNeight
//     if (t_neigh_ab[idx] == oldNeight)
//     {
//         t_neigh_ab[idx] = newNeight;
//         if (t_a_t[t_parent[idx]] == idx) {
//             var center_pos = (v_pos[parent_verts.x] + v_pos[parent_verts.y]) / 2.0f;

//             v_pos[abc.y] = center_pos; // to debug
//             v_to_update_mask[abc.y] = 0;
//         }
//     }
//     else if (t_neigh_bc[idx] == oldNeight)
//     {
//         t_neigh_bc[idx] = newNeight;
//         if (t_b_t[t_parent[idx]] == idx) {
//             var center_pos = (v_pos[parent_verts.y] + v_pos[parent_verts.z]) / 2.0f;

//             v_pos[abc.z] = center_pos; // to debug
//             v_to_update_mask[abc.z] = 0;
//         }
//     }
//     else if (t_neigh_ca[idx] == oldNeight)
//     {
//         t_neigh_ca[idx] = newNeight;
//         if (t_c_t[t_parent[idx]] == idx) {
//             var center_pos = (v_pos[parent_verts.z] + v_pos[parent_verts.x]) / 2.0f;

//             v_pos[abc.x] = center_pos; // to debug
//             v_to_update_mask[abc.x] = 0;
//         }
//     }
//     // TODO: do not set update mask but just set midpoint to middle of parent vertices
// }

// Merges triangles.
// Accepts a list of parents that will become visible again. The children are removed
[shader("compute")]
[numthreads(2, 1, 1)]
void computeMain(uint3 DTid: SV_DispatchThreadID)
{
    if (DTid.x >= n_tris_to_merge)
        return;
    // DTid.x is starting from zero and increasing for each tris to merge
    uint idx = idxs_to_merge[DTid.x];

    // The neigh of the tris are divided, so we can safely update them
    // if (t_lv[t_a_t[idx]] == t_lv[t_neigh_ab[t_a_t[idx]]])
    // {
    //     updateNeights(t_neigh_ab[idx], t_a_t[idx], idx);
    // }
    // if (t_lv[t_b_t[idx]] == t_lv[t_neigh_bc[t_b_t[idx]]])
    // {
    //     updateNeights(t_neigh_bc[idx], t_b_t[idx], idx);
    // }
    // if (t_lv[t_c_t[idx]] == t_lv[t_neigh_ca[t_c_t[idx]]])
    // {
    //     updateNeights(t_neigh_ca[idx], t_c_t[idx], idx);
    // }
    // TODO: Updating all, less can be updated;
    // TODO: re-enable this
    // updateAllNeights(t_neigh_ab[t_a_t[idx]], idx);
    // updateAllNeights(t_neigh_ab[t_b_t[idx]], idx);
    
    // updateAllNeights(t_neigh_bc[t_b_t[idx]], idx);
    // updateAllNeights(t_neigh_bc[t_c_t[idx]], idx);

    // updateAllNeights(t_neigh_ca[t_c_t[idx]], idx);
    // updateAllNeights(t_neigh_ca[t_a_t[idx]], idx);

    // set divided flag
    t_divided[idx] = 0;
    // Mark tris as merged
    t_to_merge_mask[idx] = 0;
    t_deactivated[t_a_t[idx]] = 1;
    t_deactivated[t_b_t[idx]] = 1;
    t_deactivated[t_c_t[idx]] = 1;
    t_deactivated[t_center_t[idx]] = 1;

    // -----------------------------------------------------------------
    // Debugging purposes only
    // t_neigh_ab[t_a_t[idx]] = 224;
    // t_neigh_bc[t_a_t[idx]] = DTid.x;
    // t_neigh_ca[t_a_t[idx]] = idx;
    // t_divided[t_a_t[idx]] = 1;
    // t_divided[t_b_t[idx]] = 1;
    // t_divided[t_c_t[idx]] = 1;
    // t_divided[t_center_t[idx]] = 1;
    // var t_debug = t_a_t[idx];
    // t_a_t[idx] = 123;
    // t_b_t[idx] = idx;
    // t_c_t[idx] = DTid.x;
    // t_center_t[idx] = DTid.x;
    // -----------------------------------------------------------------
    // set children
    // t_a_t[idx] = -1;
    // t_b_t[idx] = -1;
    // t_c_t[idx] = -1;
    // t_center_t[idx] = -1;

    // setNeightToUpdate(idx, v_idx);
    // TODO: update vertices after
}
