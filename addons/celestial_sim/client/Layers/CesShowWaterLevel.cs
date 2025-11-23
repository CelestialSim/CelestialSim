using Godot;

namespace CelestialSim.scripts.client.Layers;

// Layers for creating terrain, they have a get height method
[Tool]
public partial class CesShowWaterLevel : CesLayer
{
    public override void UpdatePos()
    {
        var vPos = _cesState.GetPos();
        var tAbc = _cesState.GetTAbc();
        // var oldSimValue = state.sim_value.GetTensorDataAsSpan<double>().ToArray();

        // var simVar = _cesState.GetSimVar();
        // _NewVertices = _planetGenerator.Generate(_NewVertices);
    }
}