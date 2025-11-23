// Slang Compute Shader

// Define the necessary buffers and constants
[[vk::binding(0, 0)]]
RWStructuredBuffer<float3> v_pos;
[[vk::binding(1, 0)]]
RWStructuredBuffer<int3> t_abc;
[[vk::binding(2, 0)]]
RWStructuredBuffer<float3> t_norm;
[[vk::binding(3, 0)]]
RWStructuredBuffer<int> t_lv;
[[vk::binding(4, 0)]]
RWStructuredBuffer<int2[3]> t_abc_edges;
[[vk::binding(5, 0)]]
RWStructuredBuffer<bool> t_divided;
[[vk::binding(6, 0)]]
RWStructuredBuffer<int> t_neight_ab;
[[vk::binding(7, 0)]]
RWStructuredBuffer<int> t_ab_other_b;
[[vk::binding(8, 0)]]
RWStructuredBuffer<int> t_neight_bc;
[[vk::binding(9, 0)]]
RWStructuredBuffer<int> t_bc_other_c;
[[vk::binding(10, 0)]]
RWStructuredBuffer<int> t_neight_ca;
[[vk::binding(11, 0)]]
RWStructuredBuffer<int> t_ca_other_a;
[[vk::binding(12, 0)]]
RWStructuredBuffer<int> t_a_t;
[[vk::binding(13, 0)]]
RWStructuredBuffer<int> t_b_t;
[[vk::binding(14, 0)]]
RWStructuredBuffer<int> t_c_t;
[[vk::binding(15, 0)]]
RWStructuredBuffer<int> t_center_t;
[[vk::binding(16, 0)]]
RWStructuredBuffer<int> t_parent_t;
[[vk::binding(17, 0)]]
RWStructuredBuffer<int> v_to_update_idx;
[[vk::binding(18, 0)]]
RWStructuredBuffer<float> sim_value;
[[vk::binding(19, 0)]]
RWStructuredBuffer<int> old_state_pointer;

// Constants for dynamic axes and opset version
static const int opset_version = 17; // todo check what happens when using 18

// Function to update height of core vertices
void get_height(RWStructuredBuffer<float3> v_pos, int indices[], float radius)
{
    // Implement the logic to update the height of vertices based on the radius
    // Placeholder logic
    for (uint i = 0; i < indices.getCount(); i++)
    {
        uint idx = indices[i];
        v_pos[idx] = normalize(v_pos[idx]) * radius;
    }
}



// Main compute shader function
[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 DTid : SV_DispatchThreadID)
{
    const uint tlen = 20;
    const uint vlen = 12;
    // Initialize the buffers with the given data
    float3[] initial_v_pos = {
        float3(-0.5257, 0.0000, 0.8507),
        float3(0.5257, 0.0000, 0.8507),
        float3(-0.5257, 0.0000, -0.8507),
        float3(0.5257, 0.0000, -0.8507),
        float3(0.0000, 0.8507, 0.5257),
        float3(0.0000, 0.8507, -0.5257),
        float3(0.0000, -0.8507, 0.5257),
        float3(0.0000, -0.8507, -0.5257),
        float3(0.8507, 0.5257, 0.0000),
        float3(-0.8507, 0.5257, 0.0000),
        float3(0.8507, -0.5257, 0.0000),
        float3(-0.8507, -0.5257, 0.0000)
    };

    for (uint i = 0; i < vlen; i++)
    {
        v_pos[i] = initial_v_pos[i];
    }

    int3[] initial_t_abc = {
        int3(0, 4, 1),
        int3(0, 9, 4),
        int3(9, 5, 4),
        int3(4, 5, 8),
        int3(4, 8, 1),
        int3(8, 10, 1),
        int3(8, 3, 10),
        int3(5, 3, 8),
        int3(5, 2, 3),
        int3(2, 7, 3),
        int3(7, 10, 3),
        int3(7, 6, 10),
        int3(7, 11, 6),
        int3(11, 0, 6),
        int3(0, 1, 6),
        int3(6, 1, 10),
        int3(9, 0, 11),
        int3(9, 11, 2),
        int3(9, 2, 5),
        int3(7, 2, 11)
    };

    float3[] initial_t_norm = {
        float3(-0.0000, 0.35682, 0.93417),
        float3(-0.57735, 0.57735, 0.57735),
        float3(-0.35682, 0.93417, -0.0000),
        float3(0.35682, 0.93417, 0.0000),
        float3(0.57735, 0.57735, 0.57735),
        float3(0.93417, 0.0000, 0.35682),
        float3(0.93417, -0.0000, -0.35682),
        float3(0.57735, 0.57735, -0.57735),
        float3(-0.0000, 0.35682, -0.93417),
        float3(0.0000, -0.35682, -0.93417),
        float3(0.57735, -0.57735, -0.57735),
        float3(0.35682, -0.93417, -0.0000),
        float3(-0.35682, -0.93417, 0.0000),
        float3(-0.57735, -0.57735, 0.57735),
        float3(-0.0000, -0.35682, 0.93417),
        float3(0.57735, -0.57735, 0.57735),
        float3(-0.93417, -0.0000, 0.35682),
        float3(-0.93417, -0.0000, -0.35682),
        float3(-0.57735, 0.57735, -0.57735),
        float3(-0.57735, -0.57735, -0.57735)
    };

    int initial_t_lv[tlen];
    for (int i = 0; i < tlen; i++)
    {
        initial_t_lv[i] = 0;
    }

    t_abc_edges = {
        int2[3](int2(0, 4), int2(1, 4), int2(0, 1)),
        int2[3](int2(0, 9), int2(4, 9), int2(0, 4)),
        int2[3](int2(5, 9), int2(4, 5), int2(4, 9)),
        int2[3](int2(4, 5), int2(5, 8), int2(4, 8)),
        int2[3](int2(4, 8), int2(1, 8), int2(1, 4)),
        int2[3](int2(8, 10), int2(1, 10), int2(1, 8)),
        int2[3](int2(3, 8), int2(3, 10), int2(8, 10)),
        int2[3](int2(3, 5), int2(3, 8), int2(5, 8)),
        int2[3](int2(2, 5), int2(2, 3), int2(3, 5)),
        int2[3](int2(2, 7), int2(3, 7), int2(2, 3)),
        int2[3](int2(7, 10), int2(3, 10), int2(3, 7)),
        int2[3](int2(6, 7), int2(6, 10), int2(7, 10)),
        int2[3](int2(7, 11), int2(6, 11), int2(6, 7)),
        int2[3](int2(0, 11), int2(0, 6), int2(6, 11)),
        int2[3](int2(0, 1), int2(1, 6), int2(0, 6)),
        int2[3](int2(1, 6), int2(1, 10), int2(6, 10)),
        int2[3](int2(0, 9), int2(0, 11), int2(9, 11)),
        int2[3](int2(9, 11), int2(2, 11), int2(2, 9)),
        int2[3](int2(2, 9), int2(2, 5), int2(5, 9)),
        int2[3](int2(2, 7), int2(2, 11), int2(7, 11))
    };

    t_divided = bool[](t_abc.length(), false);
    t_neight_ab = int[](20, 0, 1, 16, 18, 2, 3, 6, 7, 8, 18, 19, 11, 12, 19, 16, 0, 14, 1, 16, 17, 9);
    t_ab_other_b = int[](20, 2, 0, 2, 1, 2, 2, 1, 2, 1, 0, 2, 2, 2, 1, 2, 1, 0, 2, 2, 0);
    t_neight_bc = int[](20, 4, 2, 3, 7, 5, 15, 10, 6, 9, 10, 6, 15, 13, 14, 15, 5, 13, 19, 8, 17);
    t_bc_other_c = int[](20, 2, 2, 0, 2, 2, 1, 1, 0, 2, 2, 1, 2, 2, 2, 0, 1, 0, 1, 0, 1);
    t_neight_ca = int[](20, 14, 0, 1, 4, 0, 4, 5, 3, 7, 8, 9, 10, 11, 12, 13, 11, 17, 18, 2, 12);
    t_ca_other_a = int[](20, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0);
    t_a_t = int[](t_abc.length(), -1);
    t_b_t = int[](t_abc.length(), -1);
    t_c_t = int[](t_abc.length(), -1);
    t_center_t = int[](t_abc.length(), -1);
    t_parent_t = int[](t_abc.length(), -1);

    // Update height of core vertices
    v_to_update_idx = int[](v_pos.length(), 0);
    for (uint i = 0; i < v_pos.length(); i++)
    {
        v_to_update_idx[i] = i;
    }
    get_height(v_pos, v_to_update_idx, 1.0); // Assuming radius is 1.0

    int[] tris_to_check_idx = int[](t_abc.length(), 0);
    for (uint i = 0; i < t_abc.length(); i++)
    {
        tris_to_check_idx[i] = i;
    }
    
    sim_value = float[](t_abc.length(), 0.0);
    old_state_pointer = int[](t_divided.length(), 0);
    for (uint i = 0; i < t_divided.length(); i++)
    {
        old_state_pointer[i] = i;
    }

    // Handle opset version logic
    if (opset_version == 18)
    {
        // Implement specific logic for opset version 18
    }
    else
    {
        // Implement specific logic for opset version 17
    }
}