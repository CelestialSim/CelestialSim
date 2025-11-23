using System.Collections.Generic;
using Godot;

namespace ces.Rendering.Sims;

[Tool]
public partial class ShowLevelSimLayer: CesSimLayer
{
    private double[] ModelForward(
        long[] level
    )
    {
        var simValue = new List<double>(new double[level.Length]);

        for (var i = 0; i < level.Length; i++)
        {
            simValue[i] = level[i];
        }

        return simValue.ToArray();
    }

    public override void Forward(CesState state, double deltaT)
    {

        // var level = state.t_lv.GetTensorDataAsSpan<long>().ToArray();
        //
        // var newSim = ModelForward(level);
        // state.UpdateSimValue(newSim);
    }
}