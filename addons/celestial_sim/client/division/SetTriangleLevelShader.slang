// Set Triangle Level Shader
// Sets a specific triangle's level to an exact value

// Triangle level buffer (read-write)
[[vk::binding(0, 0)]]
RWStructuredBuffer<int> t_lv;

// Triangle index to modify
[[vk::binding(1, 0)]]
ConstantBuffer<int> triangleIndex;

// New level value to set
[[vk::binding(2, 0)]]
ConstantBuffer<int> newLevel;

[shader("compute")]
[numthreads(1, 1, 1)]
void computeMain(uint3 DTid: SV_DispatchThreadID)
{
    // Set the triangle level to the new value
    int idx = triangleIndex;
    t_lv[idx] = newLevel;
}
