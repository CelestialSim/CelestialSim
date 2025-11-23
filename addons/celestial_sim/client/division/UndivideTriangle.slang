// Undivide Triangle Shader
// Marks a triangle as not divided and deactivates its children

// Divided mask buffer (read-write)
[[vk::binding(0, 0)]]
RWStructuredBuffer<int> t_divided;

// Deactivated mask buffer (read-write)
[[vk::binding(1, 0)]]
RWStructuredBuffer<int> t_deactivated;

// Children triangle indices
[[vk::binding(2, 0)]]
StructuredBuffer<int> t_a_t;

[[vk::binding(3, 0)]]
StructuredBuffer<int> t_b_t;

[[vk::binding(4, 0)]]
StructuredBuffer<int> t_c_t;

[[vk::binding(5, 0)]]
StructuredBuffer<int> t_center_t;

// Triangle index to undivide
[[vk::binding(6, 0)]]
ConstantBuffer<int> triangleIndex;

// Triangle level buffer
[[vk::binding(7, 0)]]
StructuredBuffer<int> t_lv;

// Neighbour triangles
[[vk::binding(8, 0)]]
StructuredBuffer<int> t_neigh_ab;

[[vk::binding(9, 0)]]
StructuredBuffer<int> t_neigh_bc;

[[vk::binding(10, 0)]]
StructuredBuffer<int> t_neigh_ca;

[shader("compute")]
[numthreads(1, 1, 1)]
void computeMain(uint3 DTid: SV_DispatchThreadID)
{
    int idx = triangleIndex;

    // Only undivide if the triangle is actually divided
    if (t_divided[idx] != 1)
    {
        return;
    }

    // Get the level of the current triangle
    int currentLevel = t_lv[idx];

    // Get neighbor indices
    int neighAB = t_neigh_ab[idx];
    int neighBC = t_neigh_bc[idx];
    int neighCA = t_neigh_ca[idx];

    // Check if all neighbors are at the same level
    // A neighbor index of -1 means no neighbor (boundary edge)
    bool canUndivide = true;

    if (neighAB != -1 && t_lv[neighAB] != currentLevel)
    {
        canUndivide = false;
    }

    if (neighBC != -1 && t_lv[neighBC] != currentLevel)
    {
        canUndivide = false;
    }

    if (neighCA != -1 && t_lv[neighCA] != currentLevel)
    {
        canUndivide = false;
    }

    // Only undivide if all neighbors are at the same level
    if (canUndivide)
    {
        // Mark the parent triangle as not divided
        t_divided[idx] = 0;

        // Deactivate all child triangles
        int childA = t_a_t[idx];
        int childB = t_b_t[idx];
        int childC = t_c_t[idx];
        int childCenter = t_center_t[idx];

        if (childA > 0)
            t_deactivated[childA] = 1;
        if (childB > 0)
            t_deactivated[childB] = 1;
        if (childC > 0)
            t_deactivated[childC] = 1;
        if (childCenter > 0)
            t_deactivated[childCenter] = 1;
    }
}
