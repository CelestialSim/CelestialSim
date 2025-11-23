// V_POS buffer (read-write)
[[vk::binding(0, 0)]]
RWStructuredBuffer<float3> v_pos;

// T_ABC buffer (read-write)
[[vk::binding(1, 0)]]
RWStructuredBuffer<int3> t_abc;

// T_ABC buffer (read-write)
[[vk::binding(2, 0)]]
RWStructuredBuffer<int> t_divided;

[[vk::binding(3, 0)]]
RWStructuredBuffer<int> t_to_div;

// Uniform buffers (read-only)
[[vk::binding(4, 0)]]
ConstantBuffer<uint> nTris;

[[vk::binding(5, 0)]]
ConstantBuffer<uint> nVerts;

[[vk::binding(6, 0)]]
ConstantBuffer<uint> nTrisAdded;

[[vk::binding(7, 0)]]
ConstantBuffer<uint> nVertsAdded;

// used to make the idx start from zero
[[vk::binding(8, 0)]]
ConstantBuffer<uint> todiv_initial_idx;

// [[vk::binding(9, 0)]]
// StructuredBuffer<int> sum_buffer;

[[vk::binding(9, 0)]]
RWStructuredBuffer<int> v_to_update_mask;

[[vk::binding(10, 0)]]
RWStructuredBuffer<int> t_ico_idx;

[[vk::binding(11, 0)]]
RWStructuredBuffer<int> t_neigh_ab;

[[vk::binding(12, 0)]]
RWStructuredBuffer<int> t_neigh_bc;

[[vk::binding(13, 0)]]
RWStructuredBuffer<int> t_neigh_ca;

[[vk::binding(14, 0)]]
RWStructuredBuffer<int> t_a_t;

[[vk::binding(15, 0)]]
RWStructuredBuffer<int> t_b_t;

[[vk::binding(16, 0)]]
RWStructuredBuffer<int> t_c_t;

[[vk::binding(17, 0)]]
RWStructuredBuffer<int> t_center_t;

[[vk::binding(18, 0)]]
RWStructuredBuffer<int> t_parent;

[[vk::binding(19, 0)]]
ConstantBuffer<uint> removeDupVerts;

[[vk::binding(20, 0)]]
RWStructuredBuffer<int> t_lv;

[[vk::binding(21, 0)]]
RWStructuredBuffer<uint> indicesToDiv;

[[vk::binding(22, 0)]]
RWStructuredBuffer<float> tris_out;

enum Edge
{
    ab,
    bc,
    ca
}

enum Subtris
{
    a,
    b,
    c
}

struct NeightsInfo
{
    // 0 means neight ab faces ba
    // This is used for enight of one core fare to the other
    const Edge[] ab = { Edge.ca, Edge.ab, Edge.ca, Edge.bc, Edge.ca, Edge.ca, Edge.bc, Edge.ca, Edge.bc, Edge.ab, Edge.ca, Edge.ca, Edge.ca, Edge.bc, Edge.ca, Edge.bc, Edge.ab, Edge.ca, Edge.ca, Edge.ab };
    const Edge[] bc = { Edge.ca, Edge.ca, Edge.ab, Edge.ca, Edge.ca, Edge.bc, Edge.bc, Edge.ab, Edge.ca, Edge.ca, Edge.bc, Edge.ca, Edge.ca, Edge.ca, Edge.ab, Edge.bc, Edge.ab, Edge.bc, Edge.ab, Edge.bc };
    const Edge[] ca = { Edge.ab, Edge.ab, Edge.bc, Edge.ab, Edge.bc, Edge.bc, Edge.ab, Edge.bc, Edge.ab, Edge.bc, Edge.bc, Edge.ab, Edge.ab, Edge.bc, Edge.bc, Edge.bc, Edge.ab, Edge.ab, Edge.ab, Edge.ab };

    // const edgesArray[] coreNeightsMap = {
    //     ab_edges, bc_edges, ca_edges
    // };
    // Some elements are instered to simplify the code but are not expected to be used
    const matrix<Subtris, 3, 3> subTrisMap = {
        { Subtris.b, Subtris.a, Subtris.a },
        { Subtris.a, Subtris.c, Subtris.b },
        { Subtris.c, Subtris.a, Subtris.a } 
    };

    // const matrix<Optional<Subtris>, 3, 3> subTrisMapMaybe = {
    //     { Subtris.b, Subtris.a, none },
    //     { none, Subtris.c, Subtris.b },
    //     { Subtris.c, none, Subtris.a }
    // };

}

bool t_divided_or_to_div(uint idx)
{
    return t_to_div[idx] == 1 || t_divided[idx] == 1;
}

bool abNeightDivided(uint idx)
{
    return t_divided_or_to_div(t_neigh_ab[idx]);
}

bool bcNeightDivided(uint idx)
{
    return t_divided_or_to_div(t_neigh_bc[idx]);
}

bool caNeightDivided(uint idx)
{
    return t_divided_or_to_div(t_neigh_ca[idx]);
}

// Adds 3 vertices in the middle of the edges of the triangle
// defined by the vertices abc.
// divTrisIndex is the index of the triangle (0 to nTrisAdded)
// returns the index of the added vertices
int3 add_middle_vertices(uint parent_idx, int3 abc, uint divTrisIndex)
{

    // add vertices
    float3 ab_mid = (v_pos[abc.x] + v_pos[abc.y]) / 2;
    float3 bc_mid = (v_pos[abc.y] + v_pos[abc.z]) / 2;
    float3 ca_mid = (v_pos[abc.z] + v_pos[abc.x]) / 2;
    uint v_idx_start = nVerts + 3 * divTrisIndex;
    int3 v_idx = int3(v_idx_start, v_idx_start + 1, v_idx_start + 2);
    if (removeDupVerts == 1)
    {
        uint t_idx_start = nTris + 4 * divTrisIndex;
        v_idx = t_abc[t_idx_start + 3].zxy;
    }
    v_pos[v_idx.x] = ab_mid;
    v_pos[v_idx.y] = bc_mid;
    v_pos[v_idx.z] = ca_mid;

    return v_idx;
}

void setVerticesToUpdate(uint parent_idx, uint3 v_idx)
{
    // set v_to_update_mask
    if (abNeightDivided(parent_idx))
    {
        v_to_update_mask[v_idx.x] = 1;
    }
    if (bcNeightDivided(parent_idx))
    {
        v_to_update_mask[v_idx.y] = 1;
    }
    if (caNeightDivided(parent_idx))
    {
        v_to_update_mask[v_idx.z] = 1;
    }
}

// Chooses the neighbours based on the given edge, the edge of the neight triangle and what subtriangle it is
// The code below is equivalent to the following switch-case statement
// switch (tuple) {
//     .{ .myEdge = .ab, .neighEdge = .ab, .subTris = .a } => .b,
//     .{ .myEdge = .ab, .neighEdge = .ab, .subTris = .b } => .a,
//     .{ .myEdge = .bc, .neighEdge = .bc, .subTris = .b } => .c,
//     .{ .myEdge = .bc, .neighEdge = .bc, .subTris = .c } => .b,
//     .{ .myEdge = .ca, .neighEdge = .ca, .subTris = .c } => .a,
//     .{ .myEdge = .ca, .neighEdge = .ca, .subTris = .a } => .c,
//     //
//     .{ .myEdge = .ab, .neighEdge = .bc, .subTris = .a } => .c,
//     .{ .myEdge = .ab, .neighEdge = .bc, .subTris = .b } => .b,
//     .{ .myEdge = .bc, .neighEdge = .ca, .subTris = .b } => .a,
//     .{ .myEdge = .bc, .neighEdge = .ca, .subTris = .c } => .c,
//     .{ .myEdge = .ca, .neighEdge = .ab, .subTris = .c } => .b,
//     .{ .myEdge = .ca, .neighEdge = .ab, .subTris = .a } => .a,
//     //
//     .{ .myEdge = .ab, .neighEdge = .ca, .subTris = .a } => .a,
//     .{ .myEdge = .ab, .neighEdge = .ca, .subTris = .b } => .c,
//     .{ .myEdge = .bc, .neighEdge = .ab, .subTris = .b } => .b,
//     .{ .myEdge = .bc, .neighEdge = .ab, .subTris = .c } => .a,
//     .{ .myEdge = .ca, .neighEdge = .bc, .subTris = .c } => .c,
//     .{ .myEdge = .ca, .neighEdge = .bc, .subTris = .a } => .b,
// }
int chooseNeight(uint thisTris, Edge edge, uint parent_idx, Subtris subtris)
{

    var divided = false;
    var neight_idx = 0;

    if (edge == Edge.ab)
    {
        divided = abNeightDivided(parent_idx);
        neight_idx = t_neigh_ab[parent_idx];
    }
    else if (edge == Edge.bc)
    {
        divided = bcNeightDivided(parent_idx);
        neight_idx = t_neigh_bc[parent_idx];
    }
    else
    {
        divided = caNeightDivided(parent_idx);
        neight_idx = t_neigh_ca[parent_idx];
    }


    if (divided)
    {
        // t_center_t[thisTris] =9090;
        // Choose closes subtriangle
        var other = NeightsInfo();
        var otherSubTris = other.subTrisMap[edge][subtris];
        Edge otherNeighEdge = edge;
        var icoIdx = t_ico_idx[thisTris];
        uint[] neighIcoIdxs = { t_ico_idx[t_neigh_ab[neight_idx]], t_ico_idx[t_neigh_bc[neight_idx]], t_ico_idx[t_neigh_ca[neight_idx]] };
        bool differentIcoIdx = icoIdx != neighIcoIdxs[edge];

        if (differentIcoIdx)
        {
            switch (edge)
            {
            case Edge.ab:
                otherNeighEdge = other.ab[t_ico_idx[parent_idx]];
                break;
            case Edge.bc:
                otherNeighEdge = other.bc[t_ico_idx[parent_idx]];
                break;
            case Edge.ca:
                otherNeighEdge = other.ca[t_ico_idx[parent_idx]];
                break;
            }
        }

        // The table above is divided in three parts, the result ar the same offsetted by otherNeighEdge - edge
        var offset = (3 + otherNeighEdge - edge) % 3;
        var neighOrientationOffset = (offset + otherSubTris) % 3;
        const uint[] neighs = { t_a_t[neight_idx], t_b_t[neight_idx], t_c_t[neight_idx] };
        const uint neigh = neighs[neighOrientationOffset];

        // The neighbour also needs to be updated since it this triangle is being divided
        // TODO: check for race conditions
        var neighVerts = t_abc[neigh];
        switch (otherNeighEdge)
        {
        // We only need to update the middle vert of the parent tris,
        // Here both on edge are updated to simplify the code
        case Edge.ab:
            t_neigh_ab[neigh] = thisTris;
            v_to_update_mask[neighVerts.x] = 1;
            v_to_update_mask[neighVerts.y] = 1;
            break;
        case Edge.bc:
            t_neigh_bc[neigh] = thisTris;
            v_to_update_mask[neighVerts.y] = 1;
            v_to_update_mask[neighVerts.z] = 1;
            break;
        case Edge.ca:
            t_neigh_ca[neigh] = thisTris;
            v_to_update_mask[neighVerts.z] = 1;
            v_to_update_mask[neighVerts.x] = 1;
            break;
        }
        // t_neigh_ab[neigh] = edge;
        // t_neigh_bc[neigh] = t_ico_idx[parent_idx];
        // t_neigh_ca[neigh] = (3 + otherNeighEdge + neighOrientationOffset) % 3;
        // t_center_t[neigh] = 123454;
        if (thisTris == 40 && edge == Edge.ab)
        {
            t_a_t[thisTris] = otherNeighEdge;
            t_b_t[thisTris] = icoIdx;
            t_c_t[thisTris] = neighOrientationOffset;
            // Control
            t_center_t[thisTris] = 9999;

        }
        return neigh;
    }
    else
    {
        return neight_idx;
    }
}

// void updateEdgeRecursive(uint thisTris, uint neigh, Edge otherNeighEdge ) {
//     var neighVerts = t_abc[neigh];
//     switch (otherNeighEdge)
//     {
//     // We only need to update the middle vert of the parent tris,
//     // Here both on edge are updated to simplify the code

//     // Todo: here we should split neigh updates since if the tris is to divide the childer
//     // might still need to be created
//     case Edge.ab:
//         if (t_divided[neigh] == 1)
//         {
//             updateEdgeRecursive(thisTris, t_a_t[neigh], otherNeighEdge);
//             updateEdgeRecursive(thisTris, t_b_t[neigh], otherNeighEdge);
//         }
//         else {
//             t_neigh_ab[neigh] = thisTris;
//             v_to_update_mask[neighVerts.x] = 1;
//             v_to_update_mask[neighVerts.y] = 1;
//         }
//         break;
//     case Edge.bc:
//         if (t_divided[neigh] == 1)
//         {
//             updateEdgeRecursive(thisTris, t_b_t[neigh], otherNeighEdge);
//             updateEdgeRecursive(thisTris, t_c_t[neigh], otherNeighEdge);
//         }
//         else {
//             t_neigh_bc[neigh] = thisTris;
//             v_to_update_mask[neighVerts.y] = 1;
//             v_to_update_mask[neighVerts.z] = 1;
//         }
//         break;
//     case Edge.ca:
//         if (t_divided[neigh] == 1)
//         {
//             updateEdgeRecursive(thisTris, t_c_t[neigh], otherNeighEdge);
//             updateEdgeRecursive(thisTris, t_a_t[neigh], otherNeighEdge);
//         }
//         else {
//             t_neigh_ca[neigh] = thisTris;
//             v_to_update_mask[neighVerts.z] = 1;
//             v_to_update_mask[neighVerts.x] = 1;
//         }
//         break;
//     }
// }

uint4 add_triangles(uint parent_idx, int3 abc, int3 v_idx, uint divTrisIndex)
{
    // add triangles
    uint t_idx_start = nTris + 4 * divTrisIndex;
    int4 t_idx = int4(t_idx_start, t_idx_start + 1, t_idx_start + 2, t_idx_start + 3);
    t_abc[t_idx.x] = int3(abc.x, v_idx.x, v_idx.z);
    t_abc[t_idx.y] = int3(v_idx.x, abc.y, v_idx.y);
    t_abc[t_idx.z] = int3(v_idx.z, v_idx.y, abc.z);
    t_abc[t_idx.w] = int3(v_idx.y, v_idx.z, v_idx.x);

    // set t_ico_idx
    t_ico_idx[t_idx.x] = t_ico_idx[parent_idx];
    t_ico_idx[t_idx.y] = t_ico_idx[parent_idx];
    t_ico_idx[t_idx.z] = t_ico_idx[parent_idx];
    t_ico_idx[t_idx.w] = t_ico_idx[parent_idx];

    // set children
    t_a_t[parent_idx] = t_idx.x;
    t_b_t[parent_idx] = t_idx.y;
    t_c_t[parent_idx] = t_idx.z;
    t_center_t[parent_idx] = t_idx.w;
    
    // set parent for all child triangles
    t_parent[t_idx.x] = parent_idx;
    t_parent[t_idx.y] = parent_idx;
    t_parent[t_idx.z] = parent_idx;
    t_parent[t_idx.w] = parent_idx;

    // --- Update neighbours ---
    // Can be disabled for debugging

    // center_tris
    t_neigh_ab[t_idx.w] = t_c_t[parent_idx];
    t_neigh_bc[t_idx.w] = t_a_t[parent_idx];
    t_neigh_ca[t_idx.w] = t_b_t[parent_idx];

    // a_tris
    t_neigh_ab[t_idx.x] = chooseNeight(t_idx.x, Edge.ab, parent_idx, Subtris.a);
    t_neigh_bc[t_idx.x] = t_center_t[parent_idx];
    t_neigh_ca[t_idx.x] = chooseNeight(t_idx.x, Edge.ca, parent_idx, Subtris.a);

    // Update neighs of neights

    // b_tris
    t_neigh_ab[t_idx.y] = chooseNeight(t_idx.y, Edge.ab, parent_idx, Subtris.b);
    t_neigh_bc[t_idx.y] = chooseNeight(t_idx.y, Edge.bc, parent_idx, Subtris.b);
    t_neigh_ca[t_idx.y] = t_center_t[parent_idx];

    // c_tris
    t_neigh_ab[t_idx.z] = t_center_t[parent_idx];
    t_neigh_bc[t_idx.z] = chooseNeight(t_idx.z, Edge.bc, parent_idx, Subtris.c);
    t_neigh_ca[t_idx.z] = chooseNeight(t_idx.z, Edge.ca, parent_idx, Subtris.c);

    return t_idx;
}

// Main compute shader function
[shader("compute")]
[numthreads(2, 1, 1)]
void computeMain(uint3 DTid: SV_DispatchThreadID)
{
    if (DTid.x >= nTrisAdded / 4)
        return;
    // thread index starting from first to last possible dividable tris
    uint idx = indicesToDiv[DTid.x];
    // index starting from zero and increasing for each divided tris
    // 0 to nTrisToDiv
    uint divTrisIndex = DTid.x;
    // todo: remove the idx >= nTris check after confirming issue origin
    tris_out[divTrisIndex] = idx;

    int3 abc = t_abc[idx];

    var v_idx = add_middle_vertices(idx, abc, divTrisIndex);
    setVerticesToUpdate(idx, v_idx);
    // add triangles
    var t_idx = add_triangles(idx, abc, v_idx, divTrisIndex);
    // set divided flag
    t_divided[idx] = 1;

    // set to_div flag
    t_to_div[idx] = 0;

    // set t_lv
    t_lv[t_idx.x] = t_lv[idx] + 1;
    t_lv[t_idx.y] = t_lv[idx] + 1;
    t_lv[t_idx.z] = t_lv[idx] + 1;
    t_lv[t_idx.w] = t_lv[idx] + 1;
}
