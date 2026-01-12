using System;
using System.Collections.Generic;
using Godot;


public class CesMergeLOD
{
    private readonly RenderingDevice _rd;
    private readonly string mergeTrisPath = "res://addons/celestial_sim/scripts/lod/algo/MergeLOD.slang";
    private readonly string collectTrisPath = "res://addons/celestial_sim/scripts/lod/algo/CollectTrisToMerge.slang";
    private const bool VerifyGpuIndexCollection = false;

    public CesMergeLOD(RenderingDevice rd)
    {
        _rd = rd;
    }

    private static uint[] BuildCpuIndices(CesState state)
    {
        var toMergeMask = state.GetTToMergeMask();

        List<uint> indices = new(toMergeMask.Length);
        for (var i = 0; i < toMergeMask.Length; i++)
        {
            if (toMergeMask[i] != 0)
            {
                indices.Add((uint)i);
            }
        }

        return indices.ToArray();
    }

    private static void ValidateIndices(uint[] cpuIndices, uint[] gpuIndices)
    {
        if (cpuIndices.Length != gpuIndices.Length)
        {
            throw new InvalidOperationException($"GPU merge list mismatch: CPU={cpuIndices.Length}, GPU={gpuIndices.Length}.");
        }

        var sortedCpu = (uint[])cpuIndices.Clone();
        var sortedGpu = (uint[])gpuIndices.Clone();
        Array.Sort(sortedCpu);
        Array.Sort(sortedGpu);

        for (var i = 0; i < sortedCpu.Length; i++)
        {
            if (sortedCpu[i] != sortedGpu[i])
            {
                throw new InvalidOperationException($"GPU merge list mismatch at {i}: CPU={sortedCpu[i]}, GPU={sortedGpu[i]}.");
            }
        }
    }

    private BufferInfo BuildIndicesToMergeBuffer(CesState state, bool captureResult, out uint nTrisToMerge,
        out uint[] gpuSnapshot)
    {
        var outputSize = Math.Max(state.t_to_merge_mask.filledSize, (uint)sizeof(uint));
        var indicesBuffer = CesComputeUtils.CreateEmptyStorageBuffer(_rd, outputSize);
        var counterBuffer = CesComputeUtils.CreateEmptyStorageBuffer(_rd, sizeof(uint));

        var bufferInfos = new BufferInfo[]
        {
            state.t_to_merge_mask,
            indicesBuffer,
            counterBuffer,
            CesComputeUtils.CreateUniformBuffer(_rd, state.nTris)
        };

        if (state.nTris > 0)
        {
            CesComputeUtils.DispatchShader(_rd, collectTrisPath, bufferInfos, state.nTris);
        }

        var countSpan = CesComputeUtils.ConvertBufferToArray<uint>(_rd, counterBuffer);
        nTrisToMerge = countSpan.Length > 0 ? Math.Min(countSpan[0], state.nTris) : 0;
        indicesBuffer.filledSize = nTrisToMerge * sizeof(uint);

        gpuSnapshot = Array.Empty<uint>();
        if (captureResult && nTrisToMerge > 0)
        {
            var gpuSpan = CesComputeUtils.ConvertBufferToArray<uint>(_rd, indicesBuffer);
            gpuSnapshot = gpuSpan[..(int)nTrisToMerge].ToArray();
        }

        return indicesBuffer;
    }

    public uint MakeMerge(CesState state)
    {
        var verifyIndices = VerifyGpuIndexCollection;
        var cpuIndices = verifyIndices ? BuildCpuIndices(state) : Array.Empty<uint>();

        var indicesToRemoveBuffer = BuildIndicesToMergeBuffer(state, verifyIndices, out var nTrisToMerge,
            out var gpuSnapshot);

        if (nTrisToMerge == 0)
        {
            return 0;
        }

        if (verifyIndices)
        {
            ValidateIndices(cpuIndices, gpuSnapshot);
        }

        var trisOutputBuffer = CesComputeUtils.CreateStorageBuffer(_rd, new float[(int)nTrisToMerge]);
        var bufferInfosRemove = new BufferInfo[]
        {
            state.t_abc,
            state.t_divided,
            CesComputeUtils.CreateUniformBuffer(_rd, nTrisToMerge),
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

        CesComputeUtils.DispatchShader(_rd, mergeTrisPath, bufferInfosRemove, nTrisToMerge);

        state.nDeactivatedTris += nTrisToMerge;

        return nTrisToMerge;
    }
}
