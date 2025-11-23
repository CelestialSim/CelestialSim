// Get Triangle Info Shader
// Retrieves comprehensive information about a specific triangle

// Triangle info struct matching C# CesTriangleInfo
struct TriangleInfo
{
    // Triangle vertex indices
    int vertexA;
    int vertexB;
    int vertexC;

    // Triangle level (subdivision depth)
    int level;

    // Triangle flags
    int isDivided;     // 0 = not divided, 1 = divided
    int isDeactivated; // 0 = active, 1 = deactivated
    int isToDivide;    // 0 = not marked for division, 1 = marked for division

    // Neighbour triangle indices
    int neighbourAB; // Triangle sharing edge AB
    int neighbourBC; // Triangle sharing edge BC
    int neighbourCA; // Triangle sharing edge CA

    // Icosphere index (0-19 for base icosphere)
    int icosphereIndex;

    // Children triangle indices (valid only if isDivided == 1)
    int childA;      // Child triangle at vertex A
    int childB;      // Child triangle at vertex B
    int childC;      // Child triangle at vertex C
    int childCenter; // Central child triangle

    // Parent triangle index (-1 if this is a root triangle)
    int parent;
};

// Triangle indices (ABC)
[[vk::binding(0, 0)]]
StructuredBuffer<int3> t_abc;

// Triangle level buffer
[[vk::binding(1, 0)]]
StructuredBuffer<int> t_lv;

// Divided triangles mask
[[vk::binding(2, 0)]]
StructuredBuffer<int> t_divided;

// Deactivated triangles mask
[[vk::binding(3, 0)]]
StructuredBuffer<int> t_deactivated;

// To divide mask
[[vk::binding(4, 0)]]
StructuredBuffer<int> t_to_div;

// Neighbour triangles
[[vk::binding(5, 0)]]
StructuredBuffer<int> t_neigh_ab;

[[vk::binding(6, 0)]]
StructuredBuffer<int> t_neigh_bc;

[[vk::binding(7, 0)]]
StructuredBuffer<int> t_neigh_ca;

// Icosphere index
[[vk::binding(8, 0)]]
StructuredBuffer<int> t_ico_idx;

// Children triangles
[[vk::binding(9, 0)]]
StructuredBuffer<int> t_a_t;

[[vk::binding(10, 0)]]
StructuredBuffer<int> t_b_t;

[[vk::binding(11, 0)]]
StructuredBuffer<int> t_c_t;

[[vk::binding(12, 0)]]
StructuredBuffer<int> t_center_t;

// Parent triangle
[[vk::binding(13, 0)]]
StructuredBuffer<int> t_parent;

// Input: triangle index to query
[[vk::binding(14, 0)]]
ConstantBuffer<int> triangleIndex;

// Output buffer for triangle info
[[vk::binding(15, 0)]]
RWStructuredBuffer<TriangleInfo> outputInfo;

[shader("compute")]
[numthreads(1, 1, 1)]
void computeMain(uint3 DTid: SV_DispatchThreadID)
{
    int idx = triangleIndex;

    // Create and populate the triangle info struct
    TriangleInfo info;

    // Triangle vertices (indices)
    int3 abc = t_abc[idx];
    info.vertexA = abc.x;
    info.vertexB = abc.y;
    info.vertexC = abc.z;

    // Triangle level
    info.level = t_lv[idx];

    // Flags
    info.isDivided = t_divided[idx];
    info.isDeactivated = t_deactivated[idx];
    info.isToDivide = t_to_div[idx];

    // Neighbours
    info.neighbourAB = t_neigh_ab[idx];
    info.neighbourBC = t_neigh_bc[idx];
    info.neighbourCA = t_neigh_ca[idx];

    // Icosphere index
    info.icosphereIndex = t_ico_idx[idx];

    // Children triangles
    info.childA = t_a_t[idx];
    info.childB = t_b_t[idx];
    info.childC = t_c_t[idx];
    info.childCenter = t_center_t[idx];

    // Parent triangle
    info.parent = t_parent[idx];

    // Write the struct to output buffer
    outputInfo[0] = info;
}
