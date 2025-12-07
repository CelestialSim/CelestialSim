using Godot;


/// <summary>
/// Shader that validates constraints before triangle division.
/// This ensures triangles meet all requirements before MakeDiv is called.
/// Checks include:
/// - Already divided status
/// - Subdivision level limits
/// - Deactivation status
/// - Neighbor level matching (all neighbors must be at same level)
/// - Center triangle constraints (center triangles can only divide if all neighbors are divided)
/// </summary>
public class CesEnforceDivisionConstraints
{
    private readonly RenderingDevice rd;
    private readonly string shaderPath = "res://addons/celestial_sim/scripts/lod/algo/EnforceDivisionConstraints.slang";

    public CesEnforceDivisionConstraints(RenderingDevice rd)
    {
        this.rd = rd;
    }

    /// <summary>
    /// Validates all triangles marked for division and clears the flag for any
    /// that don't meet the necessary constraints.
    /// </summary>
    /// <param name="cesState">Current state containing all triangle data</param>
    /// <param name="maxDivs">Maximum allowed subdivision level</param>
    public void ValidateDivisionConstraints(CesState cesState, uint maxDivs)
    {

        var toDivBefore = cesState.GetTToDivideMask();
        // Bind buffers
        var bufferInfos = new BufferInfo[]
        {
            cesState.t_lv,
            cesState.t_divided,
            cesState.t_to_divide_mask,
            cesState.t_deactivated,
            cesState.t_neight_ab,
            cesState.t_neight_bc,
            cesState.t_neight_ca,
            cesState.t_parent,
            cesState.t_center_t
        };

        var toDivAfter = cesState.GetTToDivideMask();

        // Dispatch compute shader - one thread per triangle
        var xGroups = cesState.nTris;
        CesComputeUtils.DispatchShader(rd, shaderPath, bufferInfos, xGroups);
    }
}
