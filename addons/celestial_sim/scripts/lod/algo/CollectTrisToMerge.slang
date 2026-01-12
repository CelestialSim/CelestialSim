// Collects indices of parent triangles scheduled for merge.

[[vk::binding(0, 0)]]
StructuredBuffer<int> t_to_merge_mask;

[[vk::binding(1, 0)]]
RWStructuredBuffer<uint> output_indices;

[[vk::binding(2, 0)]]
RWStructuredBuffer<uint> global_counter;

[[vk::binding(3, 0)]]
ConstantBuffer<uint> n_tris_total;

[shader("compute")]
[numthreads(64, 1, 1)]
void computeMain(uint3 DTid: SV_DispatchThreadID)
{
    uint idx = DTid.x;
    if (idx >= n_tris_total)
    {
        return;
    }

    if (t_to_merge_mask[idx] == 0)
    {
        return;
    }

    uint write_index;
    InterlockedAdd(global_counter[0], 1, write_index);

    if (write_index >= n_tris_total)
    {
        return;
    }

    output_indices[write_index] = idx;
}
