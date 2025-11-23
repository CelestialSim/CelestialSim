using System;
using System.Collections.Generic;
using ces.Rendering.division;
using Godot;
using CelestialSim.addons.celestial_sim.client.division;

namespace ces.Rendering;

internal class CesClient
{
    private readonly BuffersCache cache = new();

    // public BufferCleaner bufferCleaner = new();
    public CesCelestial gen;
    public Vector3[] Norm;
    public Vector3[] Pos;

    public Vector2[] SimValue = [];

    public CesState State;
    public int[] Triangles = [];

    public void LayersUpdate(CesState s)
    {
        var layers = gen.Layers;
        for (var i = 0; i < layers.Count; i++)
        {
            var layer = layers[i];
            // Copy state to layer
            if (i == 0)
            {
                layer.SetState(s, gen.Radius);
            }
            else
            {
                layer.SetState(layers[i - 1]);
            }

            // Update layer values
            layer.UpdatePos();

            // Copy values back to state
            // if (i == layers.Count - 1)
            //     layer.SetBackState();
        }
    }

    public virtual void UpdateTriangleGraph(Vector3 camlocal, bool skipAutoDivisionMarking = false)
    {
        if (State == null)
        {
            State = CesCoreState.CreateCoreState(gen.rd);
            State.core_state = CesCoreState.CreateCoreState(gen.rd);
            LayersUpdate(State);
        }
        // State = CesCoreState.CreateCoreState(gen.rd);

        // for each subdivision, get the output and use it as input
        CesDivShader cesDivShader = new(State.rd);
        CesDivConstraintShader constraintShader = new(State.rd);
        CesDivCheckShader divCheckShader = new(State.rd);

        uint nTrisAdded = 0;
        bool firstRun = true;
        while (nTrisAdded > 0 || firstRun)
        {
            firstRun = false;

            // Only flag triangles for division if not skipping (used for manual division)
            if (!skipAutoDivisionMarking)
            {
                divCheckShader.FlagLargeTrisToDivide(State, camlocal, gen.Subdivisions, gen.Radius, gen.TriangleScreenSize);
            }

            // Validate constraints before attempting division (always run before MakeDiv)
            constraintShader.ValidateDivisionConstraints(State, gen.Subdivisions);

            nTrisAdded = cesDivShader.MakeDiv(State, gen.PreciseNormals, cache);
            if (nTrisAdded > 0)
            {
                GD.Print($"Divided {nTrisAdded / 4} triangles");
            }


            // ------ Layers update --------       
            // TODO: Use a different level for each vertex
            LayersUpdate(State);
            // -----------------------------

        }

        // Retriving output
        var finalOutput = CesFinalOutput.CreateFinalOutput(State, gen.LowPolyLook, cache);
        Pos = finalOutput.pos;
        Norm = finalOutput.normals;
        Triangles = finalOutput.tris;
        SimValue = finalOutput.sim;
    }

    public void Dispose()
    {
        // State?.Dispose();
        // cache.Dispose();
        // BufferInfo.DisposeAllBuffers();
        gen.rd.Free();
    }

    // ~CesClient()
    // {
    //     GD.Print("ces state Decionstructor called");
    //     Dispose();
    // }
}

public class BuffersCache
{
    public Dictionary<string, BufferInfo> cache = new();
    public RenderingDevice rd;

    public BufferInfo GetOrCreateBuffer(RenderingDevice rd, string name, uint size, Span<byte> data = default)
    {
        this.rd ??= rd;

        BufferInfo buf = new();
        var bufferInCache = cache.ContainsKey(name);
        var badBufferDimentions = false;
        if (bufferInCache)
        {
            var value = cache[name];
            badBufferDimentions = size > value.maxSize || size < value.maxSize / 2;
            // && value.maxSize <= size * 10  // if the cached buffer is too big it will impact performance negatively due to transfer time
            if (!badBufferDimentions)
            {
                // var offset = size;
                // var clearBytes = value.maxSize - size;
                var updatedLenBuffer = new BufferInfo(value.buffer, size, value.maxSize, rd);
                cache[name] = updatedLenBuffer;
                buf = updatedLenBuffer;
            }
        }

        if (badBufferDimentions)
        {
            // free the old buffer
            var done = new System.Threading.ManualResetEventSlim(false);
            RenderingServer.CallOnRenderThread(Callable.From(() =>
            {
                rd.FreeRid(cache[name].buffer);
                done.Set();
            }));
            done.Wait();
        }
        if (!bufferInCache || badBufferDimentions)
        {
            BufferInfo bufferInfo;
            bufferInfo = data == null ? CesComputeUtils.CreateEmptyStorageBuffer(rd, size) : CesComputeUtils.CreateStorageBuffer(rd, data);
            // bufferInfo.filledSize = size;
            cache[name] = bufferInfo;
            buf = bufferInfo;
        }

        return buf;
    }
}

public struct BufferInfo
{
    public Rid buffer;
    public uint maxSize;
    public uint filledSize;

    public RenderingDevice.UniformType bufferType;
    private readonly RenderingDevice rd;

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
    public void ExtendBuffer(uint bytesToExtend)
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