
// T_ABC buffer (read-write)
[[vk::binding(0, 0)]]
StructuredBuffer<int3> t_abc;

// T_ABC buffer (read-write)
[[vk::binding(1, 0)]]
RWStructuredBuffer<int> t_divided;

[[vk::binding(2, 0)]]
ConstantBuffer<uint> nTrisToRemove;

[[vk::binding(3, 0)]]
RWStructuredBuffer<int> t_neigh_ab;

[[vk::binding(4, 0)]]
RWStructuredBuffer<int> t_neigh_bc;

[[vk::binding(5, 0)]]
RWStructuredBuffer<int> t_neigh_ca;

[[vk::binding(6, 0)]]
StructuredBuffer<int> t_a_t;
    
[[vk::binding(7, 0)]]
StructuredBuffer<int> t_b_t;

[[vk::binding(8, 0)]]
StructuredBuffer<int> t_c_t;

[[vk::binding(9, 0)]]
StructuredBuffer<int> t_center_t;

[[vk::binding(10, 0)]]
StructuredBuffer<uint> indicesToRemove;

[[vk::binding(11, 0)]]
StructuredBuffer<uint> t_to_remove;

[[vk::binding(12, 0)]]
RWStructuredBuffer<float> tris_out;

[[vk::binding(13, 0)]]
RWStructuredBuffer<int> t_deactivated;

[[vk::binding(14, 0)]]
StructuredBuffer<uint> t_lv;

[[vk::binding(15, 0)]]
RWStructuredBuffer<uint> v_to_update_mask;

enum Edge
{
    ab,
    bc,
    ca
}

bool divided(uint idx) {
    return t_divided[idx] == 1 && t_to_remove[idx] == 0;
}

void updateNeights(uint idx, uint oldNeight, uint newNeight) {

    var abc = t_abc[idx];
    // replace edge that had oldNeight with newNeight
    if (t_neigh_ab[idx] == oldNeight) {
        t_neigh_ab[idx] = newNeight;
        v_to_update_mask[abc.x] = 1;
        v_to_update_mask[abc.y] = 1;
    }
    else if (t_neigh_bc[idx] == oldNeight) {
        t_neigh_bc[idx] = newNeight;
        v_to_update_mask[abc.y] = 1;
        v_to_update_mask[abc.z] = 1;
    }
    else if (t_neigh_ca[idx] == oldNeight) {
        t_neigh_ca[idx] = newNeight;
        v_to_update_mask[abc.z] = 1;
        v_to_update_mask[abc.x] = 1;
    }
    // TODO: do not set update mask but just set midpoint to middle of parent vertices
}

// Main compute shader function
[shader("compute")]
[numthreads(2, 1, 1)]
void computeMain(uint3 DTid : SV_DispatchThreadID)
{
    if (DTid.x >= nTrisToRemove) return;
    // thread index starting from first to last possible dividable tris
    uint idx = indicesToRemove[DTid.x];
    // index starting from zero and increasing for each divided tris
    // 0 to nTrisToDiv
    uint divTrisIndex = DTid.x;
    tris_out[divTrisIndex] = idx;

    // The children of that is removed are at the same level of the neighs
    if (t_lv[t_a_t[idx]] == t_lv[t_neigh_ab[idx]]) {
        updateNeights(t_neigh_ab[idx], t_a_t[idx], idx);
    }
    if (t_lv[t_b_t[idx]] == t_lv[t_neigh_bc[idx]]) {
        updateNeights(t_neigh_bc[idx], t_b_t[idx], idx);
    }
    if (t_lv[t_c_t[idx]] == t_lv[t_neigh_ca[idx]]) {
        updateNeights(t_neigh_ca[idx], t_c_t[idx], idx);
    }

    // set divided flag
    t_divided[idx] = 0;
    t_deactivated[t_a_t[idx]] = 1;
    t_deactivated[t_b_t[idx]] = 1;
    t_deactivated[t_c_t[idx]] = 1;
    t_deactivated[t_center_t[idx]] = 1;
    // set children
    // t_a_t[idx] = -1;
    // t_b_t[idx] = -1;
    // t_c_t[idx] = -1;
    // t_center_t[idx] = -1;
    
    // setNeightToUpdate(idx, v_idx);
    // TODO: update vertices after
}