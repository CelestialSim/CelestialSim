using System;
using Godot;


public class CesFinalState
{
    private const string CreateFinalOutputPath = "res://addons/celestial_sim/scripts/lod/algo/CreateFinalOutput.slang";

    public static FinalOutput CreateFinalOutput(CesState state, bool lowPoly)
    {
        var gpuOutput = CreateFinalOutputGpu(state, lowPoly);

        GD.Print($"Number of invisible (deactivate + parents) triangles: {state.nTris - gpuOutput.nVisibleTris}");

        return ReadFinalOutputToCpu(state, gpuOutput, true);
    }

    public static GpuFinalOutput CreateFinalOutputGpu(CesState state, bool lowPoly)
    {
        var rd = state.rd;
        var divMask = state.GetDividedMask().ToArray();
        var deactivatedMask = state.GetTDeactivatedMask().ToArray();
        var visibleMask = new int[divMask.Length];

        for (var i = 0; i < divMask.Length; i++)
        {
            visibleMask[i] = (divMask[i] == 0 && deactivatedMask[i] == 0) ? 1 : 0;
        }

        var visibleMaskBuffer = CesComputeUtils.CreateStorageBuffer(rd, visibleMask);

        CesComputeUtils.SumArrayInPlace(visibleMask.AsSpan());
        uint nVisibleTris = visibleMask.Length == 0 ? 0u : (uint)visibleMask[^1];

        if (nVisibleTris == 0 || state.nTris == 0)
        {
            RenderingServer.CallOnRenderThread(Callable.From(() => rd.FreeRid(visibleMaskBuffer.buffer)));
            return new GpuFinalOutput
            {
                nVisibleTris = nVisibleTris
            };
        }

        var visiblePrefixBuffer = CesComputeUtils.CreateStorageBuffer(rd, visibleMask);

        var vertexCount = nVisibleTris * 3u;
        var outPos = CesComputeUtils.CreateEmptyStorageBuffer(rd, vertexCount * 4u * (uint)sizeof(float));
        var outTris = CesComputeUtils.CreateEmptyStorageBuffer(rd, vertexCount * (uint)sizeof(int));
        var outColor = CesComputeUtils.CreateEmptyStorageBuffer(rd, vertexCount * 2u * (uint)sizeof(float));

        var bufferInfos = new BufferInfo[]
        {
            state.v_pos,
            state.t_abc,
            state.t_parent,
            state.t_neight_ab,
            state.t_neight_bc,
            state.t_neight_ca,
            state.t_lv,
            visibleMaskBuffer,
            visiblePrefixBuffer,
            outPos,
            outTris,
            outColor,
            CesComputeUtils.CreateUniformBuffer(rd, state.nTris),
            CesComputeUtils.CreateUniformBuffer(rd, nVisibleTris)
        };

        CesComputeUtils.DispatchShader(rd, CreateFinalOutputPath, bufferInfos, state.nTris);

        RenderingServer.CallOnRenderThread(Callable.From(() => rd.FreeRid(visibleMaskBuffer.buffer)));
        RenderingServer.CallOnRenderThread(Callable.From(() => rd.FreeRid(visiblePrefixBuffer.buffer)));

        return new GpuFinalOutput
        {
            pos = outPos,
            tris = outTris,
            color = outColor,
            nVisibleTris = nVisibleTris
        };
    }

    public static FinalOutput ReadFinalOutputToCpu(CesState state, GpuFinalOutput gpuOutput, bool freeBuffers = true)
    {
        var vertexCount = gpuOutput.nVisibleTris * 3u;
        if (vertexCount == 0)
        {
            return new FinalOutput
            {
                tris = Array.Empty<int>(),
                color = Array.Empty<Vector2>(),
                normals = Array.Empty<Vector3>(),
                pos = Array.Empty<Vector3>()
            };
        }

        var rd = state.rd;
        var pos = CesComputeUtils.ConvertV4BufferToVector3Array(rd, gpuOutput.pos);
        var tris = CesComputeUtils.ConvertBufferToArray<int>(rd, gpuOutput.tris).ToArray();
        var colorFloats = CesComputeUtils.ConvertBufferToArray<float>(rd, gpuOutput.color).ToArray();
        var color = new Vector2[vertexCount];
        for (var i = 0; i < vertexCount; i++)
        {
            var idx = (int)(i * 2);
            color[i] = new Vector2(colorFloats[idx], colorFloats[idx + 1]);
        }

        var normals = new Vector3[pos.Length];

        if (freeBuffers)
        {
            void FreeBuffer(Rid buffer)
            {
                if (buffer.Equals(default))
                {
                    return;
                }

                RenderingServer.CallOnRenderThread(Callable.From(() => rd.FreeRid(buffer)));
            }

            FreeBuffer(gpuOutput.pos.buffer);
            FreeBuffer(gpuOutput.tris.buffer);
            FreeBuffer(gpuOutput.color.buffer);
        }

        return new FinalOutput
        {
            tris = tris,
            color = color,
            normals = normals,
            pos = pos
        };
    }

    public struct GpuFinalOutput
    {
        public BufferInfo pos;
        public BufferInfo tris;
        public BufferInfo color;
        public uint nVisibleTris;
    }

    public struct FinalOutput
    {
        public int[] tris;
        public Vector2[] color;
        public Vector3[] normals;
        public Vector3[] pos;
    }
}

// public class BufferCleaner
// {
//     public List<Rid> buffersToClean = new();
//
//     public void ScheduleClean(Rid buffer)
//     {
//         buffersToClean.Add(buffer);
//     }
//
//     public void CleanBuffers(RenderingDevice rd)
//     {
//         foreach (var buffer in buffersToClean) rd.FreeRid(buffer);
//         buffersToClean.Clear();
//         
//     }
// }
