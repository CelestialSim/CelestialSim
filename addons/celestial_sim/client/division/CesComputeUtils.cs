using System;
using System.Linq;
using System.Runtime.InteropServices;
using Godot;
using Godot.Collections;

namespace ces.Rendering.division;

public class CesComputeUtils
{

    // Extract data from buffers
    public static Vector3[] ConvertV4BufferToVector3Array(RenderingDevice rd, BufferInfo buffer)
    {
        var floatData = ConvertBufferToArray<float>(rd, buffer);
        var numVectors = floatData.Length / 4;
        var positions = new Vector3[numVectors];
        for (var i = 0; i < numVectors; i++)
            positions[i] = new Vector3(floatData[i * 4], floatData[i * 4 + 1], floatData[i * 4 + 2]);
        return positions;
    }

    public static Span<T> ConvertBufferToArray<T>(RenderingDevice rd, Rid buffer, uint sizeBytes) where T : unmanaged
    {
        byte[] byteData = null;
        var done = new System.Threading.ManualResetEventSlim(false);

        RenderingServer.CallOnRenderThread(Callable.From(() =>
        {
            byteData = rd.BufferGetData(buffer, 0, sizeBytes);
            done.Set();
        }));

        done.Wait();

        if (byteData == null || byteData.Length % Marshal.SizeOf<T>() != 0)
            throw new InvalidOperationException("Byte data length is not a multiple of the size of T.");

        return MemoryMarshal.Cast<byte, T>(byteData);
    }

    public static Span<T> ConvertBufferToArray<T>(RenderingDevice rd, BufferInfo bufferinfo)
        where T : unmanaged
    {
        var buffer = bufferinfo.buffer;
        // var bufferSize = (int)(bufferinfo.filledSize / Marshal.SizeOf<T>());
        return ConvertBufferToArray<T>(rd, buffer, bufferinfo.filledSize);
    }

    // Create bufferinfos

    // public static RDUniform AddUniformBuffer(RenderingDevice rd, Vector3 value, int binding)
    // {
    //     var dataBytes = new byte[12];
    //     Buffer.BlockCopy(BitConverter.GetBytes(value.X), 0, dataBytes, 0, 4);
    //     Buffer.BlockCopy(BitConverter.GetBytes(value.Y), 0, dataBytes, 4, 4);
    //     Buffer.BlockCopy(BitConverter.GetBytes(value.Z), 0, dataBytes, 8, 4);
    //     return CreateUniformBuffer(rd, dataBytes, binding);
    // }

    public static BufferInfo CreateUniformBuffer<T>(RenderingDevice rd, T value) where T : unmanaged
    {
        var dataBytes = MemoryMarshal.AsBytes(MemoryMarshal.CreateSpan(ref value, 1)).ToArray();

        // Pad to 16 bytes as required by Godot's uniform buffer specification
        var paddedSize = Math.Max(16, (dataBytes.Length + 15) / 16 * 16); // Round up to nearest multiple of 16
        var paddedDataBytes = new byte[paddedSize];
        Buffer.BlockCopy(dataBytes, 0, paddedDataBytes, 0, dataBytes.Length);

        var buffer = rd.UniformBufferCreate((uint)paddedDataBytes.Length, paddedDataBytes);
        if (buffer.Equals(default)) throw new Exception("Failed to create uniform buffer.");

        return new BufferInfo(buffer, rd);
    }

    // private static RDUniform CreateUniformBuffer(RenderingDevice rd, byte[] dataBytes, int binding)
    // {
    //     var paddedDataBytes = new byte[16];
    //     Buffer.BlockCopy(dataBytes, 0, paddedDataBytes, 0, dataBytes.Length);
    //     var buffer = rd.UniformBufferCreate((uint)paddedDataBytes.Length, paddedDataBytes);
    //     if (buffer.Equals(default)) throw new Exception("Failed to create uniform buffer.");

    //     var uniform = new RDUniform
    //     {
    //         UniformType = RenderingDevice.UniformType.UniformBuffer,
    //         Binding = binding
    //     };
    //     uniform.AddId(buffer);
    //     return uniform;
    // }

    public static BufferInfo CreateStorageBuffer<T>(RenderingDevice rd, T[] data) where T : unmanaged
    {
        // Convert T[] to Span<T> 
        return CreateStorageBuffer(rd, data.AsSpan());
    }


    public static BufferInfo CreateStorageBuffer<T>(RenderingDevice rd, Span<T> data) where T : unmanaged
    {
        // Convert Span<T> to byte[]
        var dataBytes = MemoryMarshal.AsBytes(data);
        return CreateStorageBufferInternal(rd, dataBytes.ToArray(), dataBytes.Length);
    }

    public static BufferInfo CreateStorageBuffer<T>(RenderingDevice rd, T[,] data) where T : unmanaged
    {
        //TODO: This is not needed since we can instead use vector4 and the Triangle struct
        // Convert T[,] to byte[]
        var dataBytes = new byte[data.Length * Marshal.SizeOf<T>()];
        Buffer.BlockCopy(data, 0, dataBytes, 0, dataBytes.Length);
        return CreateStorageBufferInternal(rd, dataBytes, dataBytes.Length);
    }

    public static BufferInfo CreateEmptyStorageBuffer(RenderingDevice rd, uint length)
    {
        var buffer = rd.StorageBufferCreate(length);
        RenderingServer.CallOnRenderThread(Callable.From(() => rd.BufferClear(buffer, 0, length)));
        if (buffer.Equals(default)) throw new Exception("Failed to create storage buffer.");
        var BufferInfo = new BufferInfo(buffer, length, length, rd);
        return BufferInfo;
    }

    private static BufferInfo CreateStorageBufferInternal(RenderingDevice rd, byte[] data, int length)
    {
        // Create buffer
        var buffer = rd.StorageBufferCreate((uint)length, data);
        if (buffer.Equals(default)) throw new Exception("Failed to create storage buffer.");
        var BufferInfo = new BufferInfo(buffer, (uint)length, (uint)length, rd);
        return BufferInfo;
    }

    // Shader utils

    public static Rid LoadShader(RenderingDevice rd, string path)
    {
        // Load and compile compute shader
        var shaderFile = GD.Load<RDShaderFile>(path);
        if (shaderFile == null) throw new Exception("Failed to load shader file.");

        var shaderBytecode = shaderFile.GetSpirV();
        if (shaderBytecode == null) throw new Exception("Failed to get SPIR-V bytecode from GLSL shader.");

        var computeShader = rd.ShaderCreateFromSpirV(shaderBytecode);
        if (computeShader.Equals(default)) throw new Exception("Failed to create shader from SPIR-V bytecode.");
        return computeShader;
    }

    public static void DispatchShader(RenderingDevice rd, string path, BufferInfo[] bplus, uint threads)
    {
        var uniforms = new Array<RDUniform>(Enumerable.Range(0, bplus.Length)
        .Select(i => bplus[i].GetUniformWithBinding(i)).ToArray());

        RenderingServer.CallOnRenderThread(Callable.From(() =>
        {
            var setBackPosShader = LoadShader(rd, path);
            var pipeline = rd.ComputePipelineCreate(setBackPosShader);
            long computeList = rd.ComputeListBegin();
            rd.ComputeListBindComputePipeline(computeList, pipeline);

            var uniformSet = rd.UniformSetCreate(uniforms, setBackPosShader, 0);
            if (uniformSet.Equals(default)) throw new Exception("Failed to create uniform set.");
            rd.ComputeListBindUniformSet(computeList, uniformSet, 0);

            rd.ComputeListDispatch(computeList, threads, 1u, 1u);
            rd.ComputeListEnd();
            rd.Submit();
            rd.Sync();
        }));
    }

    public static void CreateAndBindUniformSet(RenderingDevice rd, Rid computeShader, long computeList,
    Array<RDUniform> uniforms)
    {
        var uniformSet = rd.UniformSetCreate(uniforms, computeShader, 0);
        if (uniformSet.Equals(default)) throw new Exception("Failed to create uniform set.");

        RenderingServer.CallOnRenderThread(Callable.From(() => rd.ComputeListBindUniformSet(computeList, uniformSet, 0)));
    }


    // Given an array like [0,0,1,0,0,1,1,0,1,1]
    // Return              [0,0,1,1,1,2,3,3,4,5]
    public static Span<int> SumArrayInPlace(Span<int> arr, bool invert = false)
    {
        var sum = 0;
        for (var i = 0; i < arr.Length; i++)
        {
            if (invert)
                sum += arr[i] == 0 ? 1 : 0;
            else
                sum += arr[i];
            arr[i] = sum;
        }

        return arr;
    }
}