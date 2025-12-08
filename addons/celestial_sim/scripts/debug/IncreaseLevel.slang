// Increase Level Shader
// Increases the level of a specific triangle by 1

// Triangle level buffer
[[vk::binding(0, 0)]]
RWStructuredBuffer<int> t_lv;

// Triangle index to modify
[[vk::binding(1, 0)]]
ConstantBuffer<int> triangleIndex;

// Amount to increase
[[vk::binding(2, 0)]]
ConstantBuffer<int> increaseAmount;

[shader("compute")]
[numthreads(1, 1, 1)]
void computeMain(uint3 DTid: SV_DispatchThreadID)
{
    // Get the values from constant buffers
    int idx = triangleIndex;
    int amount = increaseAmount;

    // Increase the level of the specified triangle
    t_lv[idx] = t_lv[idx] + amount;
}
