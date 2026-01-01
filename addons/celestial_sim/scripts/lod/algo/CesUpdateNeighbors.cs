using Godot;

public class CesUpdateNeighbors
{
    private readonly RenderingDevice _rd;
    private readonly string shaderPath = "res://addons/celestial_sim/scripts/lod/algo/UpdateNeighbors.slang";

    public CesUpdateNeighbors(RenderingDevice rd)
    {
        _rd = rd;
    }

    public void UpdateNeighbors(CesState state)
    {
        var bufferInfos = new BufferInfo[]
        {
            state.v_pos,
            state.t_abc,
            state.t_neight_ab,
            state.t_neight_bc,
            state.t_neight_ca,
            state.t_parent,
            state.t_divided,
            state.t_lv,
            state.t_a_t,
            state.t_b_t,
            state.t_c_t,
            state.t_center_t,
            CesComputeUtils.CreateUniformBuffer(_rd, state.nTris)
        };

        CesComputeUtils.DispatchShader(_rd, shaderPath, bufferInfos, state.nTris);
    }
}
