using Godot;
using static System.Linq.Enumerable;


public class CesCoreState
{
    public static CesState CreateCoreState(RenderingDevice rd)
    {
        var nTris = 20;
        var nVerts = 12;
        float[,] vPos =
        {
            { -0.5257f, 0.0000f, 0.8507f, 0.0f },
            { 0.5257f, 0.0000f, 0.8507f, 0.0f },
            { -0.5257f, 0.0000f, -0.8507f, 0.0f },
            { 0.5257f, 0.0000f, -0.8507f, 0.0f },

            { 0.0000f, 0.8507f, 0.5257f, 0.0f },
            { 0.0000f, 0.8507f, -0.5257f, 0.0f },
            { 0.0000f, -0.8507f, 0.5257f, 0.0f },
            { 0.0000f, -0.8507f, -0.5257f, 0.0f },

            { 0.8507f, 0.5257f, 0.0000f, 0.0f },
            { -0.8507f, 0.5257f, 0.0000f, 0.0f },
            { 0.8507f, -0.5257f, 0.0000f, 0.0f },
            { -0.8507f, -0.5257f, 0.0000f, 0.0f }
        };

        int[,] tAbc =
        {
            { 0, 4, 1, 0 },
            { 0, 9, 4, 0 },
            { 9, 5, 4, 0 },
            { 4, 5, 8, 0 },
            { 4, 8, 1, 0 },
            { 8, 10, 1, 0 },
            { 8, 3, 10, 0 },
            { 5, 3, 8, 0 },
            { 5, 2, 3, 0 },
            { 2, 7, 3, 0 },
            { 7, 10, 3, 0 },
            { 7, 6, 10, 0 },
            { 7, 11, 6, 0 },
            { 11, 0, 6, 0 },
            { 0, 1, 6, 0 },
            { 6, 1, 10, 0 },
            { 9, 0, 11, 0 },
            { 9, 11, 2, 0 },
            { 9, 2, 5, 0 },
            { 7, 2, 11, 0 }
        };

        var t_norm = new float[nTris, 4];
        for (var i = 0; i < nTris; i++)
        {
            var a = tAbc[i, 0];
            var b = tAbc[i, 1];
            var c = tAbc[i, 2];
            var apos = new Vector3(vPos[a, 0], vPos[a, 1], vPos[a, 2]);
            var bpos = new Vector3(vPos[b, 0], vPos[b, 1], vPos[b, 2]);
            var cpos = new Vector3(vPos[c, 0], vPos[c, 1], vPos[c, 2]);
            // normal = (b-a) x (c-a)
            var normal = (bpos - apos).Cross(cpos - apos).Normalized();
        }

        var tLv = new int[nTris];

        int[] tNeightAb = { 1, 16, 18, 2, 3, 6, 7, 8, 18, 19, 11, 12, 19, 16, 0, 14, 1, 16, 17, 9 };
        // int[] tAbOtherB = { 2, 0, 2, 1, 2, 2, 1, 2, 1, 0, 2, 2, 2, 1, 2, 1, 0, 2, 2, 0 };
        int[] tNeightBc = { 4, 2, 3, 7, 5, 15, 10, 6, 9, 10, 6, 15, 13, 14, 15, 5, 13, 19, 8, 17 };
        // int[] tBcOtherC = { 2, 2, 0, 2, 2, 1, 1, 0, 2, 2, 1, 2, 2, 2, 0, 1, 0, 1, 0, 1 };
        int[] tNeightCa = { 14, 0, 1, 4, 0, 4, 5, 3, 7, 8, 9, 10, 11, 12, 13, 11, 17, 18, 2, 12 };
        // int[] tCaOtherA = { 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0 };

        var tAT = new int[nTris];
        var tBT = new int[nTris];
        var tCT = new int[nTris];
        var tCenterT = new int[nTris];
        var tParent = Repeat(-1, nTris).ToArray(); // -1 indicates no parent (root triangles)

        var trisIdx = Range(0, nTris).ToArray();

        var simValue = new float[nTris];

        // Create CesState
        var cs = new CesState(rd)
        {
            // Create buffers for variables
            v_pos = CesComputeUtils.CreateStorageBuffer(rd, vPos),
            t_abc = CesComputeUtils.CreateStorageBuffer(rd, tAbc),
            // t_norm = ComputeUtils.CreateStorageBuffer(rd, t_norm),
            t_lv = CesComputeUtils.CreateStorageBuffer(rd, tLv),
            t_divided = CesComputeUtils.CreateStorageBuffer(rd, Repeat(0, nTris).ToArray()),
            t_deactivated = CesComputeUtils.CreateStorageBuffer(rd, Repeat(0, nTris).ToArray()),
            t_neight_ab = CesComputeUtils.CreateStorageBuffer(rd, tNeightAb),
            t_neight_bc = CesComputeUtils.CreateStorageBuffer(rd, tNeightBc),
            t_neight_ca = CesComputeUtils.CreateStorageBuffer(rd, tNeightCa),
            t_ico_idx = CesComputeUtils.CreateStorageBuffer(rd, trisIdx),
            // cs.t_abc_edges = ComputeUtils.CreateStorageBuffer(rdtAbcEdges);
            t_a_t = CesComputeUtils.CreateStorageBuffer(rd, tAT),
            t_b_t = CesComputeUtils.CreateStorageBuffer(rd, tBT),
            t_c_t = CesComputeUtils.CreateStorageBuffer(rd, tCT),
            t_center_t = CesComputeUtils.CreateStorageBuffer(rd, tCenterT),
            t_parent = CesComputeUtils.CreateStorageBuffer(rd, tParent),
            // t_parent_t = CesComputeUtils.CreateStorageBuffer(rd, tParentT),
            v_update_mask = CesComputeUtils.CreateStorageBuffer(rd, Repeat(1, nVerts).ToArray()),
            t_to_divide_mask = CesComputeUtils.CreateStorageBuffer(rd, Repeat(0, nTris).ToArray()),
            t_to_merge_mask = CesComputeUtils.CreateStorageBuffer(rd, Repeat(0, nTris).ToArray()),
            // sim_value = CesComputeUtils.CreateStorageBuffer(rd, simValue),
            // old_state_pointer = CesComputeUtils.CreateStorageBuffer(rd, oldStatePointer()),
            startIdx = 0,
            nTris = (uint)nTris,
            nVerts = (uint)nVerts
        };

        return cs;
    }
}
