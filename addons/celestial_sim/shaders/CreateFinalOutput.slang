[[vk::binding(0, 0)]]
StructuredBuffer<float4> v_pos;

[[vk::binding(1, 0)]]
StructuredBuffer<int4> t_abc;

[[vk::binding(2, 0)]]
StructuredBuffer<int> t_parent;

[[vk::binding(3, 0)]]
StructuredBuffer<int> t_neigh_ab;

[[vk::binding(4, 0)]]
StructuredBuffer<int> t_neigh_bc;

[[vk::binding(5, 0)]]
StructuredBuffer<int> t_neigh_ca;

[[vk::binding(6, 0)]]
StructuredBuffer<int> t_lv;

[[vk::binding(7, 0)]]
StructuredBuffer<int> visible_mask;

[[vk::binding(8, 0)]]
StructuredBuffer<int> visible_prefix;

[[vk::binding(9, 0)]]
RWStructuredBuffer<float4> out_pos;

[[vk::binding(10, 0)]]
RWStructuredBuffer<int> out_tris;

[[vk::binding(11, 0)]]
RWStructuredBuffer<float2> out_color;

[[vk::binding(12, 0)]]
ConstantBuffer<uint> n_tris;

[[vk::binding(13, 0)]]
ConstantBuffer<uint> n_visible_tris;

float4 Midpoint(int v0, int v1)
{
    return (v_pos[v0] + v_pos[v1]) * 0.5f;
}

void StitchEdge(int tri_idx, uint out_idx, int neigh_idx, int parent_v0, int parent_v1)
{
    if (neigh_idx < 0 || (uint)neigh_idx >= n_tris)
    {
        return;
    }

    if (t_lv[neigh_idx] >= t_lv[tri_idx])
    {
        return;
    }

    out_pos[out_idx] = Midpoint(parent_v0, parent_v1);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void computeMain(uint3 DTid : SV_DispatchThreadID)
{
    if (DTid.x >= n_tris)
    {
        return;
    }

    uint idx = DTid.x;
    if (visible_mask[idx] == 0)
    {
        return;
    }

    uint visible_idx = (uint)(visible_prefix[idx] - 1);
    if (visible_idx >= n_visible_tris)
    {
        return;
    }

    uint base_idx = visible_idx * 3u;

    int4 tri = t_abc[idx];
    int parent_idx = t_parent[idx];
    int4 parent_tri = tri;
    if (parent_idx >= 0 && (uint)parent_idx < n_tris)
    {
        parent_tri = t_abc[parent_idx];
    }

    out_pos[base_idx + 0] = v_pos[tri.x];
    out_pos[base_idx + 1] = v_pos[tri.y];
    out_pos[base_idx + 2] = v_pos[tri.z];

    out_tris[base_idx + 0] = (int)(base_idx + 0);
    out_tris[base_idx + 1] = (int)(base_idx + 1);
    out_tris[base_idx + 2] = (int)(base_idx + 2);

    float level = (float)t_lv[idx];
    out_color[base_idx + 0] = float2(level, 0.0f);
    out_color[base_idx + 1] = float2(level, 0.0f);
    out_color[base_idx + 2] = float2(level, 0.0f);

    bool is_a_tris = parent_tri.x == tri.x;
    bool is_b_tris = parent_tri.y == tri.y;
    bool is_c_tris = parent_tri.z == tri.z;

    if (t_lv[idx] < 4)
    {
        if (is_a_tris)
        {
            StitchEdge((int)idx, base_idx + 1, t_neigh_ab[idx], parent_tri.x, parent_tri.y);
            StitchEdge((int)idx, base_idx + 2, t_neigh_ca[idx], parent_tri.z, parent_tri.x);
        }
        else if (is_b_tris)
        {
            StitchEdge((int)idx, base_idx + 0, t_neigh_ab[idx], parent_tri.x, parent_tri.y);
            StitchEdge((int)idx, base_idx + 2, t_neigh_bc[idx], parent_tri.y, parent_tri.z);
        }
        else if (is_c_tris)
        {
            StitchEdge((int)idx, base_idx + 1, t_neigh_bc[idx], parent_tri.y, parent_tri.z);
            StitchEdge((int)idx, base_idx + 0, t_neigh_ca[idx], parent_tri.z, parent_tri.x);
        }
        else
        {
            int tri_idx = (int)idx;
            StitchEdge(tri_idx, base_idx + 0, t_neigh_bc[tri_idx - 1], parent_tri.y, parent_tri.z);
            StitchEdge(tri_idx, base_idx + 1, t_neigh_ca[tri_idx - 3], parent_tri.z, parent_tri.x);
            StitchEdge(tri_idx, base_idx + 2, t_neigh_ab[tri_idx - 2], parent_tri.x, parent_tri.y);
        }
    }
}
