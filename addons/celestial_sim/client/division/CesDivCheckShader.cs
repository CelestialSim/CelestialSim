using ces.Rendering;
using ces.Rendering.division;
using Godot;

public class CesDivCheckShader
{
    private readonly Rid computeShader;
    private readonly RenderingDevice rd;
    private readonly string shaderPath = "res://addons/celestial_sim/client/division/CesDivCheckShader.slang";

    public CesDivCheckShader(RenderingDevice rd)
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
            cesState.t_to_merge_mask
        };


        // Dispatch compute shader
        var xGroups = cesState.nTris;
        CesComputeUtils.DispatchShader(rd, shaderPath, bufferInfos, xGroups);

        var toDivideMask = cesState.GetTToDivideMask();
        var divMask = cesState.GetDividedMask();
        var trisSizes = CesComputeUtils.ConvertBufferToArray<float>(rd, trisSizeBuffer);
        // GD.Print("Triangle Sizes:");
        // for (var i = 0; i < cesState.nTris; i++)
        // {
        //     GD.Print($"Tri {i}: Output {trisSizes[i]:F2}, ToDivide: {toDivideMask[i]}, Divided: {divMask[i]}");
        // }
    }
}