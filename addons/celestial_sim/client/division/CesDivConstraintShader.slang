[[vk::binding(0, 0)]]
StructuredBuffer<uint> t_lv;

[[vk::binding(1, 0)]]
StructuredBuffer<int> t_divided;

[[vk::binding(2, 0)]]
RWStructuredBuffer<int> t_to_div;

// Deactivated triangles mask
[[vk::binding(3, 0)]]
StructuredBuffer<int> t_deactivated;

[[vk::binding(4, 0)]]
StructuredBuffer<int> t_neigh_ab;

[[vk::binding(5, 0)]]
StructuredBuffer<int> t_neigh_bc;

[[vk::binding(6, 0)]]
StructuredBuffer<int> t_neigh_ca;

[[vk::binding(7, 0)]]
StructuredBuffer<int> t_parent;

[[vk::binding(8, 0)]]
StructuredBuffer<int> t_center_t;

// Check if triangle can divide based on neighbor levels and division state
bool canTriangleDivide(uint idx, uint canDivLevel)
{
    // Get neighbor indices
    int neighAB = t_neigh_ab[idx];
    int neighBC = t_neigh_bc[idx];
    int neighCA = t_neigh_ca[idx];

    // Check if all neighbors are at the same level
    bool canDivide = t_lv[neighAB] == canDivLevel && t_lv[neighBC] == canDivLevel && t_lv[neighCA] == canDivLevel;

    // Check if this is a center triangle by verifying if parent has this triangle as its center child
    int parent = t_parent[idx];
    bool isCenterTris = (parent >= 0) && (t_center_t[parent] == idx);

    if (isCenterTris)
    {
        // Tris in the center can only divided if neighbors are divided
        canDivide = t_divided[neighAB] == 1 && t_divided[neighBC] == 1 && t_divided[neighCA] == 1;
    }

    return canDivide;
}

// Main compute shader function
[shader("compute")]
[numthreads(2, 1, 1)]
void computeMain(uint3 DTid: SV_DispatchThreadID)
{
    var idx = DTid.x;

    // Only check triangles that are marked for division
    if (t_to_div[idx] == 0)
        return;

    if (t_deactivated[idx] != 0)
    {
        t_to_div[idx] = 0;
        return;
    }

    uint currentLevel = t_lv[idx];
    if (!canTriangleDivide(idx, currentLevel))
    {
        t_to_div[idx] = 0;
        return;
    }

    // TODO: check triangles to remove constraints
    // Now all removal constraints are considered satisfied

    // All constraints passed, triangle can be divided
    // t_to_div[idx] remains 1
}
