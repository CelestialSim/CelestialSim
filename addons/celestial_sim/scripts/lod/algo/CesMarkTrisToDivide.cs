using Godot;

namespace CelestialSim;

public class CesMarkTrisToDivide
{
    private readonly RenderingDevice rd;
    private readonly string shaderPath = "res://addons/celestial_sim/scripts/lod/algo/MarkTrisToDivide.slang";

    public CesMarkTrisToDivide(RenderingDevice rd)
    {
        this.rd = rd;
    }

    public void FlagLargeTrisToDivide(CesState cesState, Vector3 cameraPos, uint maxDivs, float radius,
        float maxTriSize)
    {
        var trisSizeBuffer = CesComputeUtils.CreateStorageBuffer(rd, new float[cesState.nTris]);

        // Bind buffers
        var bufferInfos = new BufferInfo[]
        {
            cesState.v_pos,
            cesState.t_abc,
            cesState.t_lv,
            cesState.t_divided,
            cesState.t_to_divide_mask,
            CesComputeUtils.CreateUniformBuffer(rd, cameraPos),
            CesComputeUtils.CreateUniformBuffer(rd, maxDivs),
            CesComputeUtils.CreateUniformBuffer(rd, radius),
            CesComputeUtils.CreateUniformBuffer(rd, maxTriSize),
            trisSizeBuffer,
            cesState.t_neight_ab,
            cesState.t_neight_bc,
            cesState.t_neight_ca,
            cesState.t_deactivated,
            cesState.t_to_merge_mask,
            cesState.t_parent
        };


        // Dispatch compute shader
        var xGroups = cesState.nTris;
        CesComputeUtils.DispatchShader(rd, shaderPath, bufferInfos, xGroups);
        RenderingServer.CallOnRenderThread(Callable.From(() => rd.FreeRid(trisSizeBuffer.buffer)));

    }
}
