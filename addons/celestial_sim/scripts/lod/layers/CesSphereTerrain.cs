using Godot;


// Layers for creating terrain, they have a get height method
[Tool]
public partial class CesSphereTerrain : CesLayer
{
    private Rid computeShader;
    private string shaderPath = "res://addons/celestial_sim/scripts/lod/layers/SphereTerrain.slang";

    private void Init()
    {
    }

    public override void UpdatePos()
    {
        // Bind buffers
        var bufferInfos = new BufferInfo[]
        {
            _cesState.v_pos,
            _cesState.v_update_mask,
            CesComputeUtils.CreateUniformBuffer(rd, _radius),
            CesComputeUtils.CreateUniformBuffer(rd, _cesState.nVerts)
        };

        CesComputeUtils.DispatchShader(rd, shaderPath, bufferInfos, _cesState.nVerts);
    }
}
