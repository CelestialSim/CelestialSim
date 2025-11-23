using System;
using System.Runtime.InteropServices;
using ces.Rendering.division;
using Godot;

namespace ces.Rendering;

public class CesState
{
    public CesState core_state;
    public uint nTris; //Counts also invisible tris of previous layers

    public uint nVerts;

    // public Rid old_state_pointer;
    public RenderingDevice rd;
    public uint startIdx; // The first triangle that needs to be updated
    public BufferInfo t_a_t;

    public BufferInfo t_abc; // int[n,4], last column is empty
    public BufferInfo t_b_t;
    public BufferInfo t_c_t;
    public BufferInfo t_center_t;

    public BufferInfo t_parent;

    public BufferInfo t_divided;

    public BufferInfo t_deactivated;

    public BufferInfo t_ico_idx; //icosphere index, 0 to 20 for each element
    public BufferInfo t_lv;

    public BufferInfo t_neight_ab;
    public BufferInfo t_neight_bc;
    public BufferInfo t_neight_ca;

    public BufferInfo t_to_divide_mask;

    public BufferInfo t_to_remove_mask; // Mask for triangles to undivide/remove

    public BufferInfo v_pos; // float4[]

    public BufferInfo v_update_mask; // float[]

    public CesState(RenderingDevice rd)
    {
        this.rd = rd;
    }


    public Rid[] allBuffers()
    {
        return
        [
            t_abc.buffer,
            t_a_t.buffer,
            t_b_t.buffer,
            t_c_t.buffer,
            t_center_t.buffer,
            t_parent.buffer,
            t_divided.buffer,
            t_deactivated.buffer,
            t_ico_idx.buffer,
            t_lv.buffer,
            t_neight_ab.buffer,
            t_neight_bc.buffer,
            t_neight_ca.buffer,
            t_to_divide_mask.buffer,
            t_to_remove_mask.buffer,
            v_pos.buffer,
            v_update_mask.buffer
        ];
    }

    public void Dispose()
    {
        foreach (var buffer in allBuffers())
        {
            RenderingServer.CallOnRenderThread(Callable.From(() => rd.FreeRid(buffer)));
        }
    }

    // public (Vector3[], int[]) GetAddedVerticesPosAndIdExtracted()
    // {
    //     var godotPos = ConvertBufferToVector3Array(v_pos);
    //     var indices = ConvertBufferToArray<int>(v_to_update_idx);
    //     
    //     return (godotPos, indices);
    // }

    // Returns to indices of triangles that can possibly be divided
    public Span<int> GetTToDivideMask()
    {
        var indices = CesComputeUtils.ConvertBufferToArray<int>(rd, t_to_divide_mask);

        return indices;
    }

    // Returns to indices of triangles that should be undivided/removed
    public Span<int> GetTToRemoveMask()
    {
        var indices = CesComputeUtils.ConvertBufferToArray<int>(rd, t_to_remove_mask);

        return indices;
    }

    public int[] ConvertVUpdateMaskToIdx()
    {
        var maskSpan = CesComputeUtils.ConvertBufferToArray<int>(rd, v_update_mask);
        // First pass: Count how many times the value 1 appears
        var count = 0;
        foreach (var t in maskSpan)
            if (t == 1)
                count++;

        // Allocate the result array with the exact size
        var indices = new int[count];

        // Second pass: Collect the indices where maskSpan[i] == 1
        var idx = 0;
        for (var i = 0; i < maskSpan.Length; i++)
            if (maskSpan[i] == 1)
                indices[idx++] = i;

        return indices;
    }

    public (BufferInfo, BufferInfo, uint) GetAddedVerticesPosAndId(int[] indices, BuffersCache cache)
    {
        var idxLen = (uint)indices.Length;
        var pos = CesComputeUtils.ConvertBufferToArray<Vector4>(rd, v_pos);

        var filteredPos = new Vector4[idxLen];
        for (var i = 0; i < idxLen; i++) filteredPos[i] = pos[indices[i]];

        var filteredPosBytes = MemoryMarshal.AsBytes(filteredPos.AsSpan());
        var indexBytes = MemoryMarshal.AsBytes(indices.AsSpan());

        // GD.Print("unrolledFilteredPos len: " + unrolledFilteredPos.Length);
        var posBuffer = cache.GetOrCreateBuffer(rd, "layerPosBuffer", (uint)filteredPosBytes.Length, filteredPosBytes);
        var idxBuffer = cache.GetOrCreateBuffer(rd, "layerIdxBuffer", (uint)indexBytes.Length, indexBytes);

        return (posBuffer, idxBuffer, idxLen);
    }

    public Vector3[] GetPos()
    {
        return CesComputeUtils.ConvertV4BufferToVector3Array(rd, v_pos);
    }

    public Span<int> GetDividedMask()
    {
        return CesComputeUtils.ConvertBufferToArray<int>(rd, t_divided);
    }

    public Span<int> GetTDeactivatedMask()
    {
        return CesComputeUtils.ConvertBufferToArray<int>(rd, t_deactivated);
    }

    // public Vector3[] GetNorm()
    // {
    //     return ComputeUtils.ConvertBufferToVector3Array(rd, t_norm);
    // }
    public int[,] GetABCUnoptimized()
    {
        var tAbcSpan = CesComputeUtils.ConvertBufferToArray<int>(rd, t_abc);
        // Calculate the number of triangles
        var nTris = tAbcSpan.Length / 4;

        // Initialize the result array with the correct size
        var res = new int[nTris, 3];

        // Now, process the data and copy the triangles
        for (var i = 0; i < nTris; i++)
        {
            var srcIndex = i * 4;

            // Copy the first three elements from tAbcSpan to res
            res[i, 0] = tAbcSpan[srcIndex];
            res[i, 1] = tAbcSpan[srcIndex + 1];
            res[i, 2] = tAbcSpan[srcIndex + 2];
        }

        return res;
    }

    public Span<Triangle> GetABCW()
    {
        var tAbcSpan = CesComputeUtils.ConvertBufferToArray<Triangle>(rd, t_abc);

        return tAbcSpan;
    }

    public Span<int> GetLevel()
    {
        // todo: use real sim value
        return CesComputeUtils.ConvertBufferToArray<int>(rd, t_lv);
    }

    // public bool[] GetTUnDivided()
    // {
    //     var divided = t_divided.GetTensorDataAsSpan<bool>().ToArray();
    //     // invert array
    //     for (var i = 0; i < divided.Length; i++)
    //         divided[i] = !divided[i];
    //     return divided;
    // }

    private Vector3 CalculateCenterPoint(Vector3[] vPos, int[,] tAbc, int index)
    {
        return (vPos[tAbc[index, 0]] + vPos[tAbc[index, 1]] + vPos[tAbc[index, 2]]) / 3.0f;
    }

    public Vector3[] GetCenterPoints()
    {
        var pos = GetPos();
        var tAbc = GetTAbc();
        var centerPoints = new Vector3[tAbc.GetLength(0)];
        for (var i = 0; i < tAbc.GetLength(0); i++)
            centerPoints[i] = CalculateCenterPoint(pos, tAbc, i);

        return centerPoints;
    }

    // get t_abc which has shape n * 3
    public int[,] GetTAbc()
    {
        var tAbc = CesComputeUtils.ConvertBufferToArray<int>(rd, t_abc);
        var n = tAbc.Length / 3;
        var res = new int[n, 3];
        for (var i = 0; i < n; i++)
        {
            res[i, 0] = tAbc[i * 3];
            res[i, 1] = tAbc[i * 3 + 1];
            res[i, 2] = tAbc[i * 3 + 2];
        }

        // filter array

        // var filteredRes = new int[filter.Count(f => f), 3];
        // var j = 0;
        // for (var i = 0; i < n; i++)
        // {
        //     if (filter[i])
        //     {
        //         filteredRes[j, 0] = res[i, 0];
        //         filteredRes[j, 1] = res[i, 1];
        //         filteredRes[j, 2] = res[i, 2];
        //         j++;
        //     }
        // }

        return res;
    }

    // This class is defined for convenience since it is easier in compute shader to pass a int4 instead of int3.
    // Using marshalling an int4 can be converted to a triangle and vice versa.
    public struct Triangle
    {
        public int a;
        public int b;
        public int c;
        public int w;
    }


    // //Update sim value with new sim value filtered
    // public void UpdateSimValue(float[] simValue)
    // {
    //     sim_value.Dispose();
    //     sim_value = CreateTensor(simValue, [simValue.Length]);
    // }
    //
    // public void SetPosFromGodot(Vector3[] godotPos)
    // {
    //     // v_pos.Dispose();
    //     var pos = new float[godotPos.Length * 3];
    //     for (var i = 0; i < godotPos.Length; i++)
    //     {
    //         pos[i * 3] = godotPos[i].X;
    //         pos[i * 3 + 1] = godotPos[i].Y;
    //         pos[i * 3 + 2] = godotPos[i].Z;
    //     }
    //     
    //     // Convert float[] to byte[]
    //     v_pos = CreateStorageBuffer(pos);
    // }
    //
    // public void Dispose()
    // {
    //     // Free all buffers
    //     if (rd == null) throw new Exception("Can't dispose CesState without a RenderingDevice");
    //     // rd.FreeRid(t_abc.buffer);
    //     // rd.FreeRid(t_a_t.buffer);
    //     // rd.FreeRid(t_b_t.buffer);
    //     // rd.FreeRid(t_c_t.buffer);
    //     // rd.FreeRid(t_center_t.buffer);
    //     // rd.FreeRid(t_divided.buffer);
    //     // rd.FreeRid(t_ico_idx.buffer);
    //     // rd.FreeRid(t_lv.buffer);
    //     rd.FreeRid(t_neight_ab.buffer);
    //     rd.FreeRid(t_neight_bc.buffer);
    //     rd.FreeRid(t_neight_ca.buffer);
    //     // rd.FreeRid(t_to_divide_mask.buffer);
    //     // rd.FreeRid(v_pos.buffer);
    //     // rd.FreeRid(v_update_mask.buffer);
    //     // Suppress finalization
    //     // GC.SuppressFinalize(this);
    //
    // }

    // ~CesState()
    // {
    //     // Finalizer to clean up resources if Dispose is not called
    //     Dispose();
    // }
}