// Mark Triangle To Merge Shader
// Marks a specific triangle to be merged on the next update

// To-merge mask buffer (read-write)
[[vk::binding(0, 0)]]
RWStructuredBuffer<int> t_to_merge_mask;

// Triangle index to mark
[[vk::binding(1, 0)]]
ConstantBuffer<int> triangleIndex;

[shader("compute")]
[numthreads(1, 1, 1)]
void computeMain(uint3 DTid: SV_DispatchThreadID)
{
    // Mark the triangle to be merged
    // The actual merge will be done by CesMergeSmallTris shader
    int idx = triangleIndex;
    t_to_merge_mask[idx] = 1;
}
