using System.Linq;
using Godot;
using Godot.Collections;


public class CesMergeLOD
{
    private readonly RenderingDevice _rd;
    private readonly string mergeTrisPath = "res://addons/celestial_sim/scripts/lod/algo/MergeLOD.slang";

    public CesMergeLOD(RenderingDevice rd)
    {
        _rd = rd;
    }

    public uint MakeMerge(CesState state)
    {
        // Use the separate t_to_merge_mask instead of inferring from t_to_divide_mask
        var toRemoveMask = state.GetTToMergeMask();
        var idxs_to_merge = new Array<uint>();
        for (uint i = 0; i < toRemoveMask.Length; i++)
        {
            if (toRemoveMask[(int)i] != 0)
            {
                idxs_to_merge.Add(i);
            }
        }

        var n_tris_to_merge = idxs_to_merge.Count;

        if (n_tris_to_merge == 0)
        {
            return 0;
        }
        var deactivated = state.GetTDeactivatedMask().ToArray();
        var indicesToRemoveBuffer = CesComputeUtils.CreateStorageBuffer(_rd, idxs_to_merge.ToArray());
        var trisOutputBuffer = CesComputeUtils.CreateStorageBuffer(_rd, new float[n_tris_to_merge]);
        var bufferInfosRemove = new BufferInfo[]
        {
            state.t_abc,
            state.t_divided,
            CesComputeUtils.CreateUniformBuffer(_rd, n_tris_to_merge),
            state.t_neight_ab,
            state.t_neight_bc,
            state.t_neight_ca,
            state.t_a_t,
            state.t_b_t,
            state.t_c_t,
            state.t_center_t,
            indicesToRemoveBuffer,
            state.t_to_merge_mask,
            trisOutputBuffer,
            state.t_deactivated,
            state.t_lv,
            state.v_update_mask,
            state.v_pos,
            state.t_parent,
        };

        CesComputeUtils.DispatchShader(_rd, mergeTrisPath, bufferInfosRemove, (uint)n_tris_to_merge);
        // var trisOutput = CesComputeUtils.ConvertBufferToArray<float>(_rd, trisOutputBuffer);
        var deactivated1 = state.GetTDeactivatedMask().ToArray();
        return (uint)n_tris_to_merge;
    }
}
