// Ray-Triangle Intersection Shader
// Checks which triangle (if any) is intersected by a mouse ray

// Vertices buffer
[[vk::binding(0, 0)]]
StructuredBuffer<float3> vertices;

// Triangle indices (ABC)
[[vk::binding(1, 0)]]
StructuredBuffer<int3> t_abc;

// Ray origin
[[vk::binding(2, 0)]]
ConstantBuffer<float3> rayOrigin;

// Ray direction
[[vk::binding(3, 0)]]
ConstantBuffer<float3> rayDirection;

// Output: closest hit triangle index (-1 if no hit) and distance
[[vk::binding(4, 0)]]
RWStructuredBuffer<int> hitTriangleIndex;

[[vk::binding(5, 0)]]
RWStructuredBuffer<float> hitDistance;

// Total number of triangles
[[vk::binding(6, 0)]]
ConstantBuffer<uint> totalTris;

// Divided triangles mask (0 = visible, 1 = divided/hidden)
[[vk::binding(7, 0)]]
StructuredBuffer<int> t_divided;

// Deactivated triangles mask
[[vk::binding(8, 0)]]
StructuredBuffer<int> t_deactivated;

// Ray-Triangle Intersection using Möller-Trumbore algorithm
// Based on: https://iquilezles.org/articles/intersectors/
//
// This method efficiently computes ray-triangle intersection using barycentric coordinates.
// Returns true if the ray intersects the triangle, false otherwise.
//
// Parameters:
//   ro - Ray origin point
//   rd - Ray direction (normalized)
//   v0, v1, v2 - Triangle vertices
//   t - Output parameter: distance along ray to intersection point
//
// The algorithm works by:
// 1. Computing two edge vectors of the triangle (v1v0, v2v0)
// 2. Computing the triangle's normal via cross product
// 3. Solving for barycentric coordinates (u, v) and distance t
// 4. Checking if the intersection point lies within the triangle (u>=0, v>=0, u+v<=1)
// 5. Checking if the intersection is in front of the ray (t>=0)
bool intersectTriangle(float3 ro, float3 rd, float3 v0, float3 v1, float3 v2, out float t)
{
    // Compute triangle edge vectors from v0
    float3 v1v0 = v1 - v0;
    float3 v2v0 = v2 - v0;
    float3 rov0 = ro - v0; // Vector from v0 to ray origin

    // Compute triangle normal (not normalized for efficiency)
    float3 n = cross(v1v0, v2v0);

    // Compute perpendicular vector for barycentric coordinate calculation
    float3 q = cross(rov0, rd);

    // Compute inverse determinant (used for all divisions)
    float d = 1.0 / dot(rd, n);

    // Compute barycentric coordinates u and v
    // These represent the position within the triangle
    float u = d * dot(-q, v2v0); // Weight for vertex v1
    float v = d * dot(q, v1v0);  // Weight for vertex v2
    // Implicit third weight: w = 1 - u - v (weight for vertex v0)

    // Compute distance along ray to intersection point
    t = d * dot(-n, rov0);

    // Check if intersection point is outside triangle bounds
    // u, v must be non-negative and their sum must not exceed 1
    if (u < 0.0 || v < 0.0 || (u + v) > 1.0)
        return false;

    // Check if intersection is behind the ray origin
    if (t < 0.0)
        return false;

    return true;
}

// Simple single-threaded compute shader
// Loops through all triangles sequentially to find the closest intersection
[shader("compute")]
[numthreads(1, 1, 1)]
void computeMain(uint3 DTid: SV_DispatchThreadID)
{
    // Only execute on the first thread
    if (DTid.x != 0)
        return;

    // Initialize output values
    int closestTriIdx = -1;
    float closestDist = 1e30; // Very large number representing infinity

    // Loop through all triangles
    for (uint tidx = 0; tidx < totalTris; tidx++)
    {
        // Skip if triangle is divided or deactivated
        if (t_divided[tidx] == 1 || t_deactivated[tidx] == 1)
            continue;

        // Fetch the three vertices of this triangle
        int3 abc = t_abc[tidx];
        float3 v0 = vertices[abc.x];
        float3 v1 = vertices[abc.y];
        float3 v2 = vertices[abc.z];

        // Test for intersection
        float t;
        if (intersectTriangle(rayOrigin, rayDirection, v0, v1, v2, t))
        {
            // If this is the closest hit so far, update the result
            if (t < closestDist)
            {
                closestDist = t;
                closestTriIdx = (int)tidx;
            }
        }
    }

    // Write the final result
    hitTriangleIndex[0] = closestTriIdx;
    hitDistance[0] = closestDist;
}
