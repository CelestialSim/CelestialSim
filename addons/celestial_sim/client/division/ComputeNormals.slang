// Vertices buffer (read-write)
[[vk::binding(0, 0)]]
StructuredBuffer<float3> vertices;

[[vk::binding(1, 0)]]
RWStructuredBuffer<float> v_norm;

[[vk::binding(2, 0)]]
StructuredBuffer<int3> t_abc;

[[vk::binding(3, 0)]]
StructuredBuffer<int> t_divided;

[[vk::binding(4, 0)]]
StructuredBuffer<int> visible_tris_mask_sum;

[[vk::binding(5, 0)]]
ConstantBuffer<uint> totalTris;

[[vk::binding(6, 0)]]
ConstantBuffer<uint> lowPoly;

[[vk::binding(7, 0)]]
RWStructuredBuffer<float> outputVerts; //* 3

[[vk::binding(8, 0)]]
RWStructuredBuffer<float> simValue; //* 2

[[vk::binding(9, 0)]]
StructuredBuffer<int> t_lv;

[[vk::binding(10, 0)]]
RWStructuredBuffer<int> divided_tris;


float3 computeNormals(uint tidx){
    var abc = t_abc[tidx];
    float3 a = vertices[abc.x];
    float3 b = vertices[abc.y];
    float3 c = vertices[abc.z];
    float3 ab = b - a;
    float3 ac = c - a;
    float3 n = - normalize(cross(ab, ac));
    return n;
}

groupshared int simvalueInt[2];

// Main compute shader function
[shader("compute")]
[numthreads(32, 1, 1)]
void computeMain(uint3 DTid : SV_DispatchThreadID)
{   
    uint tidx = DTid.x;
    if (tidx >= totalTris || t_divided[tidx] == 1) return;
    uint outidx = visible_tris_mask_sum[tidx] - 1;
    // Compute triangle normal
    var tidx_norm = computeNormals(tidx);
    var abc = t_abc[tidx];
    // t_norm[tidx * 3] = tidx_norm.x;
    // t_norm[tidx * 3 + 1] = tidx_norm.y;
    // t_norm[tidx * 3 + 2] = tidx_norm.z;
    if (lowPoly == 0) {
        // vertex a
        outputVerts[abc.x * 3] = vertices[abc.x].x;
        outputVerts[abc.x * 3 + 1] = vertices[abc.x].y;
        outputVerts[abc.x * 3 + 2] = vertices[abc.x].z;
        // vertex b
        outputVerts[abc.y * 3] = vertices[abc.y].x;
        outputVerts[abc.y * 3 + 1] = vertices[abc.y].y;
        outputVerts[abc.y * 3 + 2] = vertices[abc.y].z;
        // vertex c
        outputVerts[abc.z * 3] = vertices[abc.z].x;
        outputVerts[abc.z * 3 + 1] = vertices[abc.z].y;
        outputVerts[abc.z * 3 + 2] = vertices[abc.z].z;

        // divided_tris
        divided_tris[outidx * 3] = abc.x;
        divided_tris[outidx * 3 + 1] = abc.y;
        divided_tris[outidx * 3 + 2] = abc.z;

        // test interlocked max
        // simvalueInt[0] = 0;
        // simvalueInt[0] += 1;
        // // InterlockedAdd(simvalueInt[0], 1);
        // simvalueInt[0] = simvalueInt[0] / 10;

        // DeviceMemoryBarrierWithGroupSync();

        // // simValue
        // simValue[abc.x * 2] = simvalueInt[0];
        // simValue[abc.x * 2 + 1] = simvalueInt[0];
        // simValue[abc.y * 2] = simvalueInt[0];
        // simValue[abc.y * 2 + 1] = simvalueInt[0];
        // simValue[abc.z * 2] = simvalueInt[0];
        // simValue[abc.z * 2 + 1] = simvalueInt[0];
        // simValue[abc.x * 2] = max(t_lv[tidx], simValue[abc.x * 2]);
        // simValue[abc.x * 2 + 1] = max(t_lv[tidx], simValue[abc.x * 2 + 1]);
        // simValue[abc.y * 2] = max(t_lv[tidx], simValue[abc.y * 2]);
        // simValue[abc.y * 2 + 1] = max(t_lv[tidx], simValue[abc.y * 2 + 1]);
        // simValue[abc.z * 2] = max(t_lv[tidx], simValue[abc.z * 2]);
        // simValue[abc.z * 2 + 1] = max(t_lv[tidx], simValue[abc.z * 2 + 1]) + simvalueInt[0];
        
        // InterlockedMax(simvalueInt[abc.x * 2],(int)t_lv[tidx]);
        // InterlockedMax(simvalueInt[abc.x * 2 + 1],(int)t_lv[tidx]);
        // InterlockedMax(simvalueInt[abc.y * 2],(int)t_lv[tidx]);
        // InterlockedMax(simvalueInt[abc.y * 2 + 1],(int)t_lv[tidx]);
        // InterlockedMax(simvalueInt[abc.z * 2],(int)t_lv[tidx]);
        // InterlockedMax(simvalueInt[abc.z * 2 + 1],(int)t_lv[tidx]);
        // simValue[abc.x * 2] = t_lv[tidx];
        // simValue[abc.x * 2 + 1] = t_lv[tidx];
        // AllMemoryBarrier();
        // simValue[abc.x * 2] = simvalueInt[abc.x * 2];
        // simValue[abc.x * 2 + 1] = simvalueInt[abc.x * 2 + 1];
        // simValue[abc.y * 2] = simvalueInt[abc.y * 2];
        // simValue[abc.y * 2 + 1] = simvalueInt[abc.y * 2 + 1];
        // simValue[abc.z * 2] = simvalueInt[abc.z * 2];
        // simValue[abc.z * 2 + 1] = simvalueInt[abc.z * 2 + 1];
    } else { // TODO: Only this branch is currently implemented
        // vertex a
        outputVerts[outidx * 3 * 3] = vertices[abc.x].x;
        outputVerts[outidx * 3 * 3 + 1] = vertices[abc.x].y;
        outputVerts[outidx * 3 * 3 + 2] = vertices[abc.x].z;
        // vertex b
        outputVerts[outidx * 3 * 3 + 3] = vertices[abc.y].x;
        outputVerts[outidx * 3 * 3 + 4] = vertices[abc.y].y;
        outputVerts[outidx * 3 * 3 + 5] = vertices[abc.y].z;
        // vertex c
        outputVerts[outidx * 3 * 3 + 6] = vertices[abc.z].x;
        outputVerts[outidx * 3 * 3 + 7] = vertices[abc.z].y;
        outputVerts[outidx * 3 * 3 + 8] = vertices[abc.z].z;
        
        
    }

    // normal
    v_norm[outidx * 3] = tidx_norm.x;
    v_norm[outidx * 3 + 1] = tidx_norm.y;
    v_norm[outidx * 3 + 2] = tidx_norm.z;

    // simValue
    simValue[outidx] = t_lv[tidx];
    // if (lowPoly == 1){
    //     // apply normal to all verts
    //     t_norm[tidx] = tidx_norm;
    // } else {
    //     // in this case vertices can be shared between triangles so normals
    //     // are for each vertex instead of each triangle and must be averaged
    //     //TODO: note that this could not work due to operations not being atomic
    //     t_norm[abc.x] += tidx_norm;
    //     t_norm[abc.y] += tidx_norm;
    //     t_norm[abc.z] += tidx_norm;
    //     AllMemoryBarrierWithGroupSync();
    //     t_norm[abc.x] = normalize(t_norm[abc.x]);
    //     t_norm[abc.y] = normalize(t_norm[abc.y]);
    //     t_norm[abc.z] = normalize(t_norm[abc.z]);
    // }
}