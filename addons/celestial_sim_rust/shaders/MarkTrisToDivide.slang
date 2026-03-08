// V_POS buffer
[[vk::binding(0, 0)]]
StructuredBuffer<float3> v_pos;

// T_ABC buffer
[[vk::binding(1, 0)]]
StructuredBuffer<int3> t_abc;

[[vk::binding(2, 0)]]
StructuredBuffer<uint> t_lv;

[[vk::binding(3, 0)]]
StructuredBuffer<int> t_divided;

[[vk::binding(4, 0)]]
RWStructuredBuffer<int> t_to_div;

[[vk::binding(5, 0)]]
ConstantBuffer<float3> camera_pos;

[[vk::binding(6, 0)]]
ConstantBuffer<uint> max_divs;

[[vk::binding(7, 0)]]
ConstantBuffer<float> radius;

[[vk::binding(8, 0)]]
ConstantBuffer<float> max_tri_size;

[[vk::binding(9, 0)]]
RWStructuredBuffer<float> tris_size;

// Neighbour triangles
[[vk::binding(10, 0)]]
StructuredBuffer<int> t_neigh_ab;

[[vk::binding(11, 0)]]
StructuredBuffer<int> t_neigh_bc;

[[vk::binding(12, 0)]]
StructuredBuffer<int> t_neigh_ca;

// Deactivated triangles mask
[[vk::binding(13, 0)]]
StructuredBuffer<int> t_deactivated;

// Mask for triangles that should be merged
[[vk::binding(14, 0)]]
RWStructuredBuffer<int> t_to_merge_mask;

[[vk::binding(15, 0)]]
StructuredBuffer<uint> t_parent;

// Camera parameters from viewport settings
static const float fov = 75.0f;         // Field of view in degrees
static const float near_plane = 0.05f;  // Near clipping plane in meters
static const float far_plane = 4000.0f; // Far clipping plane in meters


// Calculate screen-space area of a triangle based on level and distance
// This is a simplified approximation that avoids full vertex projection
float calculateScreenAreaSimplified(int triIdx)
{
    var abc = t_abc[triIdx];
    // Get vertices of tris
    var v_a = v_pos[abc.x];
    var v_b = v_pos[abc.y];
    var v_c = v_pos[abc.z];
    // Get center of tris position
    var v_center = (v_a + v_b + v_c) / 3;
    // Get distance from center to camera
    var dist_from_camera = length(v_center - camera_pos);
    // Tris screen size
    var screen_size = radius / (
            pow(4, t_lv[triIdx]) * dist_from_camera
        );
    return screen_size;
}

// Calculate screen-space area of a triangle based on level and distance
// This is a simplified approximation that avoids full vertex projection
float calculateScreenArea(int triIdx)
{
    // Get triangle level
    uint level = t_lv[triIdx];

    // Get triangle center by averaging vertices
    var abc = t_abc[triIdx];
    var v_a = v_pos[abc.x];
    var v_b = v_pos[abc.y];
    var v_c = v_pos[abc.z];
    var center = (v_a + v_b + v_c) / 3.0f;

    // Distance from camera to triangle center
    float dist = length(center - camera_pos);

    // Base triangle size for an icosphere (20 triangles on a sphere)
    // Area of triangle on unit sphere: 4*pi / 20 ≈ 0.628
    float baseArea = 0.628f * radius * radius;

    // Each subdivision divides a triangle into 4 smaller triangles
    // So area reduces by factor of 4 per level
    float triangleArea = baseArea / pow(4.0f, float(level));

    // Calculate perspective projection parameters
    float fov_rad = fov * float.getPi() / 180.0f;
    float tan_half_fov = tan(fov_rad * 0.5f);

    // Project triangle area to screen space
    // Screen space scaling factor: 1 / (distance * tan(fov/2))^2
    float screenScale = 1.0f / (dist * tan_half_fov);
    float screenArea = triangleArea * screenScale * screenScale;

    return screenArea;
}


// Calculate screen-space area of a triangle
float calculateScreenAreaComplex(int triIdx)
{
    var abc = t_abc[triIdx];

    // Get vertices of triangle
    var v_a = v_pos[abc.x];
    var v_b = v_pos[abc.y];
    var v_c = v_pos[abc.z];

    // Convert world positions to camera-relative positions
    var cam_a = v_a - camera_pos;
    var cam_b = v_b - camera_pos;
    var cam_c = v_c - camera_pos;

    // Get distances from camera
    var dist_a = length(cam_a);
    var dist_b = length(cam_b);
    var dist_c = length(cam_c);

    // Calculate perspective projection parameters
    var fov_rad = fov * 3.14159265359f / 180.0f;
    var tan_half_fov = tan(fov_rad * 0.5f);

    // Project vertices to screen space
    var screen_a = float2(cam_a.x / (dist_a * tan_half_fov), cam_a.y / (dist_a * tan_half_fov));
    var screen_b = float2(cam_b.x / (dist_b * tan_half_fov), cam_b.y / (dist_b * tan_half_fov));
    var screen_c = float2(cam_c.x / (dist_c * tan_half_fov), cam_c.y / (dist_c * tan_half_fov));

    // Calculate triangle area in screen space using 2D cross product
    var ab = screen_b - screen_a;
    var ac = screen_c - screen_a;
    var screen_area = 0.5f * abs(ab.x * ac.y - ab.y * ac.x);

    return screen_area;
}

// Main compute shader function
[shader("compute")]
[numthreads(2, 1, 1)]
void computeMain(uint3 DTid: SV_DispatchThreadID)
{
    var idx = DTid.x;

    // Initialize outputs
    t_to_div[idx] = 0;
    t_to_merge_mask[idx] = 0;

    // Skip if triangle is deactivated
    if (t_deactivated[idx] != 0)
    {
        return;
    }
    // We always merge the parent triangle, not the children
    // Merge if max subdivisions has been reduced by the user
    if (t_lv[idx] > max_divs)
    {

        t_to_merge_mask[t_parent[idx]] = 1;
        return;
    }

    float screenArea = calculateScreenArea(idx);
    if (t_lv[idx] < max_divs)
    {
        // Calculate screen area
        tris_size[idx] = screenArea;

        // Check if triangle is large enough to divide
        t_to_div[idx] = screenArea > max_tri_size;
    }
    // Check if divided triangle is now too small and should be merged
    // TODO: Use a threshold (e.g., 1/4 of max_tri_size) to avoid thrashing
    // if (t_divided[idx] == 1 && t_to_div[idx] == 0)
    // {
    //     t_to_merge_mask[t_parent[idx]] = 1;
    // }

    // Cannot merge core tris
    if (idx < 20)
    {
        return;
    }

    float parentScreenArea = calculateScreenArea(t_parent[idx]);

    // t_divided[idx] == 0 to divide only children
    if (parentScreenArea < (max_tri_size * 0.8f) && t_divided[t_parent[idx]] == 1)
    {
        t_to_merge_mask[t_parent[idx]] = 1;
    }
}
