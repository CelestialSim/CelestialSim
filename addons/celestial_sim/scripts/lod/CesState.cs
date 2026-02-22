using System;
using System.Runtime.InteropServices;
using Godot;

namespace CelestialSim;

public class CesState
{
    public CesState core_state;
    public uint nTris; //Counts also invisible tris of previous layers

    public uint nVerts;

    public uint nDeactivatedTris;

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

    public BufferInfo t_to_merge_mask; // Mask for triangles to merge/remove

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
            t_to_merge_mask.buffer,
            v_pos.buffer,
            v_update_mask.buffer,
        ];
    }

    public void Dispose()
    {
        foreach (var buffer in allBuffers())
        {
            RenderingServer.CallOnRenderThread(Callable.From(() => rd.FreeRid(buffer)));
        }
    }

    // Returns to indices of triangles that can possibly be divided
    public Span<int> GetTToDivideMask()
    {
        var indices = CesComputeUtils.ConvertBufferToArray<int>(rd, t_to_divide_mask);

        return indices;
    }

    // Returns to indices of triangles that should be merged/removed
    public Span<int> GetTToMergeMask()
    {
        var indices = CesComputeUtils.ConvertBufferToArray<int>(rd, t_to_merge_mask);

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

}


public class BufferInfo
{
    public Rid buffer;
    public uint maxSize;
    public uint filledSize;

    public RenderingDevice.UniformType bufferType;
    private readonly RenderingDevice rd;

    public BufferInfo()
    {
        rd = default!;
    }

    public BufferInfo(Rid buffer, uint filledSize, uint maxSize, RenderingDevice rd)
    {
        this.buffer = buffer;
        this.maxSize = maxSize;
        this.filledSize = filledSize;
        bufferType = RenderingDevice.UniformType.StorageBuffer;
        this.rd = rd;
    }

    public BufferInfo(Rid buffer, RenderingDevice rd)
    {
        this.buffer = buffer;
        bufferType = RenderingDevice.UniformType.UniformBuffer;
        this.rd = rd;
    }

    internal RenderingDevice RenderingDevice => rd;

    public RDUniform GetUniformWithBinding(int binding)
    {
        var uniform = new RDUniform
        {
            UniformType = bufferType,
            Binding = binding
        };
        uniform.AddId(buffer);
        return uniform;
    }


    /// <summary>
    /// Extends the buffer by adding the specified number of bytes to its size.
    /// If the current max size can accommodate the extension, only the filledSize is updated.
    /// Otherwise, a new larger buffer is allocated, data is copied, and the old buffer is freed.
    /// </summary>
    /// <param name="bytesToExtend">The number of bytes to add to the buffer's filled size.</param>
    public virtual void ExtendBuffer(uint bytesToExtend)
    {
        var desiredSize = filledSize + bytesToExtend;

        if (maxSize >= desiredSize
            && maxSize <= desiredSize * 2
           ) // if the cached buffer is too big it will impact performance negatively due to transfer time
        {
            // Reuse existing buffer, just update the filled size
            filledSize = desiredSize;
            return;
        }

        // Need to allocate a new larger buffer
        var newBuffer = CesComputeUtils.CreateEmptyStorageBuffer(rd, desiredSize);
        if (buffer.Equals(default)) throw new Exception("Source buffer not valid when extending buffer.");

        // Copy instance members to local variables for lambda capture
        var oldBuffer = buffer;
        var oldFilledSize = filledSize;
        var renderingDevice = rd;

        RenderingServer.CallOnRenderThread(
            Callable.From(() => renderingDevice.BufferCopy(oldBuffer, newBuffer.buffer, 0, 0, oldFilledSize))
        );

        // Free the old buffer
        RenderingServer.CallOnRenderThread(Callable.From(() => renderingDevice.FreeRid(oldBuffer)));

        // Update this instance with the new buffer
        buffer = newBuffer.buffer;
        filledSize = newBuffer.filledSize;
        maxSize = newBuffer.maxSize;
    }

}