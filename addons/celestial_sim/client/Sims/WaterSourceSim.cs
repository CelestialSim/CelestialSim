using System;
using System.Linq;
using Godot;

namespace ces.Rendering.Sims;

[Tool]
public partial class WaterSourceSim : CesSimLayer
{
    private readonly int Seed = new Random().Next();

    [Export] public bool PlaceSources;
    
    [Export]
    public Godot.Collections.Array<Transform3D> Sources = [];


    public override void _Process(double delta)
    {
        if (PlaceSources && Input.IsActionJustPressed("ui_select"))
        {
            GD.Print("Mouse click");
            var treeTransform = CesInteraction.GetPointOnPlanet(celestial);
            if (treeTransform.HasValue)
            {
                Sources.Add(treeTransform.Value);
            }
        }
    }
    
    public void SetSources(double[] sim_value, CesState state)
    {

        var centers = state.GetCenterPoints();
        
        for (var i = 0; i < Sources.Count; i++)
        {
            // find the closest point
            var sourcePos = Sources[i].Origin;
            var index = centers
                .Select((center, idx) => (Distance: center.DistanceSquaredTo(sourcePos), Index: idx))
                .MinBy(tuple => tuple.Distance).Index;
            sim_value[index] = 1;
        }
    }

    public override void Forward(CesState state, double deltaT)
    {
        // var simVal = state.sim_value.GetTensorDataAsSpan<double>().ToArray();
        //
        // SetSources(simVal,state);
        // state.UpdateSimValue(simVal);
    }
}