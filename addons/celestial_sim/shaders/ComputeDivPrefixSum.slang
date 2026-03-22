// ComputeDivPrefixSum — Computes exclusive prefix sum on t_to_div mask.
// Used by DivideLOD to map thread indices to consecutive output slots.

[[vk::binding(0, 0)]]
StructuredBuffer<int> t_to_div;

[[vk::binding(1, 0)]]
RWStructuredBuffer<int> div_prefix;

[[vk::binding(2, 0)]]
ConstantBuffer<uint> n_tris;

[shader("compute")]
[numthreads(1, 1, 1)]
void computeMain(uint3 DTid: SV_DispatchThreadID)
{
    int sum = 0;
    for (uint i = 0; i < n_tris; i++)
    {
        div_prefix[i] = sum;
        if (t_to_div[i] != 0)
        {
            sum++;
        }
    }
}
