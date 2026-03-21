// V_POS buffer (for distance calculations)
[[vk::binding(0, 0)]]
StructuredBuffer<float3> v_pos;

// T_ABC buffer
[[vk::binding(1, 0)]]
StructuredBuffer<int3> t_abc;

// Neighbor triangles (read-write)
[[vk::binding(2, 0)]]
RWStructuredBuffer<int> t_neigh_ab;

[[vk::binding(3, 0)]]
RWStructuredBuffer<int> t_neigh_bc;

[[vk::binding(4, 0)]]
RWStructuredBuffer<int> t_neigh_ca;

// Parent triangle
[[vk::binding(5, 0)]]
StructuredBuffer<int> t_parent;

// Divided mask
[[vk::binding(6, 0)]]
StructuredBuffer<int> t_divided;

// Level
[[vk::binding(7, 0)]]
StructuredBuffer<int> t_lv;

// Child triangles
[[vk::binding(8, 0)]]
StructuredBuffer<int> t_a_t;

[[vk::binding(9, 0)]]
StructuredBuffer<int> t_b_t;

[[vk::binding(10, 0)]]
StructuredBuffer<int> t_c_t;

[[vk::binding(11, 0)]]
StructuredBuffer<int> t_center_t;

// Total number of triangles
[[vk::binding(12, 0)]]
ConstantBuffer<uint> nTris;

enum Edge
{
    ab = 0,
    bc = 1,
    ca = 2
}

// Get triangle center
float3 getTriangleCenter(int tris_idx)
{
    int3 abc = t_abc[tris_idx];
    return (v_pos[abc.x] + v_pos[abc.y] + v_pos[abc.z]) / 3.0;
}

// Get parent's neighbor for specific edge
int getParentNeighbor(int parent_idx, Edge edge)
{
    if (edge == Edge.ab)
        return t_neigh_ab[parent_idx];
    else if (edge == Edge.bc)
        return t_neigh_bc[parent_idx];
    else
        return t_neigh_ca[parent_idx];
}

// Update edge neighbor by finding closest child of parent's neighbor
void updateEdgeNeighbor(int tris_idx, int parent_idx, Edge edge)
{
    if (t_lv[tris_idx] >= 4){
        // TODO: This algorithm has some error for high LOD levels
        return;
    }
    // Get parent's neighbor for this edge
    int parent_neigh = getParentNeighbor(parent_idx, edge);
    
    // If parent neighbor is not divided, use it directly
    if (t_divided[parent_neigh] == 0)
    {
        if (edge == Edge.ab)
            t_neigh_ab[tris_idx] = parent_neigh;
        else if (edge == Edge.bc)
            t_neigh_bc[tris_idx] = parent_neigh;
        else
            t_neigh_ca[tris_idx] = parent_neigh;
        return;
    }
    
    // Parent neighbor is divided, find closest child
    int children[3];
    children[0] = t_a_t[parent_neigh];
    children[1] = t_b_t[parent_neigh];
    children[2] = t_c_t[parent_neigh];
    
    // Compute this triangle's center
    float3 this_center = getTriangleCenter(tris_idx);
    
    // Find closest child by center distance
    int closest_child = children[0];
    float min_dist = distance(this_center, getTriangleCenter(children[0]));
    
    for (int i = 1; i < 3; i++)
    {
        float3 child_center = getTriangleCenter(children[i]);
        float dist = distance(this_center, child_center);
        
        if (dist < min_dist)
        {
            min_dist = dist;
            closest_child = children[i];
        }
    }
    
    // Set the neighbor to the closest child
    if (edge == Edge.ab)
        t_neigh_ab[tris_idx] = closest_child;
    else if (edge == Edge.bc)
        t_neigh_bc[tris_idx] = closest_child;
    else
        t_neigh_ca[tris_idx] = closest_child;
}

// Main compute shader function
[shader("compute")]
[numthreads(64, 1, 1)]
void computeMain(uint3 DTid: SV_DispatchThreadID)
{
    if (DTid.x >= nTris || DTid.x < 20)
        return;
    
    int idx = DTid.x;
    
    // Skip level 0 triangles (assume correct neighbors)
    if (t_lv[idx] == 0)
        return;
    
    // Skip divided triangles (only update leaf triangles)
    if (t_divided[idx] == 1)
        return;
    
    // Get parent
    int parent_idx = t_parent[idx];
    
    // Skip if this is the center child triangle
    if (t_center_t[parent_idx] == idx)
    {
        // center_tris
        t_neigh_ab[idx] = t_c_t[parent_idx];
        t_neigh_bc[idx] = t_a_t[parent_idx];
        t_neigh_ca[idx] = t_b_t[parent_idx];
        return;
    }
    if (t_a_t[parent_idx] == idx)
    {
        // center_tris
        updateEdgeNeighbor(idx, parent_idx, Edge.ab);
        t_neigh_bc[idx] = t_center_t[parent_idx];
        updateEdgeNeighbor(idx, parent_idx, Edge.ca);
    }
    if (t_b_t[parent_idx] == idx)
    {
        // center_tris
        updateEdgeNeighbor(idx, parent_idx, Edge.ab);
        updateEdgeNeighbor(idx, parent_idx, Edge.bc);
        t_neigh_ca[idx] = t_center_t[parent_idx];
    }
    if (t_c_t[parent_idx] == idx)
    {
        // center_tris
        t_neigh_ab[idx] = t_center_t[parent_idx];
        updateEdgeNeighbor(idx, parent_idx, Edge.bc);
        updateEdgeNeighbor(idx, parent_idx, Edge.ca);
    }

}
