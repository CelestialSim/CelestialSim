using Godot;

namespace ces.Rendering.Sims;

[Tool]
public partial class WaterFullSim : CesSimLayer
{
    
    [Export] public bool goDown = true;
    
    [Export]
    public float sourcesPercentage = 0.01f;
    
    WaterFlowSim waterFlowSim = new();
    WaterSourceSim waterSourceSim = new();

    public override void Forward(CesState state, double deltaT)
    {
        waterFlowSim.goDown = goDown;
        // waterSourceSim.sourcesPercentage = sourcesPercentage;
        waterSourceSim.Forward(state, deltaT);
        waterFlowSim.Forward(state, deltaT);
    }

}