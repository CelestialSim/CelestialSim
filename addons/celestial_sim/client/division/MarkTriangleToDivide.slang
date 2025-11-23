// Mark Triangle To Divide Shader
// Marks a specific triangle to be divided on the next update

// To-divide mask buffer (read-write)
[[vk::binding(0, 0)]]
RWStructuredBuffer<int> t_to_div;

// Triangle index to mark
[[vk::binding(1, 0)]]
ConstantBuffer<int> triangleIndex;

[shader("compute")]
[numthreads(1, 1, 1)]
void computeMain(uint3 DTid: SV_DispatchThreadID)
{
    // Mark the triangle to be divided
    // Constraint validation will be done by CesDivConstraintShader
    int idx = triangleIndex;
    t_to_div[idx] = 1;
}
