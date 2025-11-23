using System;
using System.Collections.Generic;
using Godot;

namespace ces.Rendering.Sims;

[Tool]
public partial class WaterFlowSim: CesSimLayer
{
    private readonly int Seed = new Random().Next();

    [Export] public bool goDown = true;

    private double[] ModelForward(
        double deltaT,
        // Assuming all inputs are appropriately defined before this method
        Vector3[] centers,
        bool[] tDivided,
        long[] a_tris,
        long[] b_tris,
        long[] c_tris,
        long[] center_tris,
        long[] tNeightAb,
        long[] tNeightBc,
        long[] tNeightCa,
        double[] oldSimValue
    )
    {
        var spreadSpeed = deltaT;
        var numTriangles = tDivided.GetLength(0);
        var simValue = new List<double>(new double[oldSimValue.Length]);
        // oldSimValue.CopyTo(simValue.ToArray(), 0);

        for (var i = 0; i < numTriangles; i++)
            if (!tDivided[i])
            {
                // Calculate centers and radii
                var abCenter = centers[tNeightAb[i]];
                var bcCenter = centers[tNeightBc[i]];
                var caCenter = centers[tNeightCa[i]];
                var Center = centers[i];
                
                float abCenterRadius;
                float bcCenterRadius;
                float caCenterRadius;
                float CenterRadius;

                if (goDown)
                {
                    // -- Model 1 -- from top to bottom
                    abCenterRadius = abCenter[1];
                    bcCenterRadius = bcCenter[1];
                    caCenterRadius = caCenter[1];
                    CenterRadius = Center[1];
                }
                else
                {
                    // -- Model 2 -- from mountain to valley
                    // var abCenterRadius = abCenter.Length();
                    // var bcCenterRadius = bcCenter.Length();
                    // var caCenterRadius = caCenter.Length();

                    var waterMultiplier = 1f;
                    // -- Model 3 -- from mountain to valley + lakes
                    abCenterRadius = abCenter.Length() + waterMultiplier * (float)oldSimValue[(int)tNeightAb[i]];
                    bcCenterRadius = bcCenter.Length() + waterMultiplier * (float)oldSimValue[(int)tNeightBc[i]];
                    caCenterRadius = caCenter.Length() + waterMultiplier * (float)oldSimValue[(int)tNeightCa[i]];
                    CenterRadius = Center.Length() + waterMultiplier * (float)oldSimValue[i];
                }



                var trisRadius = new[] { abCenterRadius, bcCenterRadius, caCenterRadius, CenterRadius };

                // Find min index
                var minIdx = FindMinIndex(trisRadius);

                if (minIdx < 3)
                {// Water moves and level decreases
                    int lowestNeigh;
                    // Spread water to the lowest neighbor
                    if (minIdx == 0)
                        lowestNeigh = (int)tNeightAb[i];
                    else if (minIdx == 1)
                        lowestNeigh = (int)tNeightBc[i];
                    else
                        lowestNeigh = (int)tNeightCa[i];

                    
                    simValue[lowestNeigh] += oldSimValue[i] * (float)spreadSpeed;
                    simValue[i] += oldSimValue[i] * (1 - (float)spreadSpeed);
                    // if the neigh is divided spread also to its children
                    if (tDivided[lowestNeigh])
                    {
                        simValue[(int)a_tris[lowestNeigh]] += oldSimValue[i] * (float)spreadSpeed;
                        simValue[(int)b_tris[lowestNeigh]] += oldSimValue[i] * (float)spreadSpeed;
                        simValue[(int)c_tris[lowestNeigh]] += oldSimValue[i] * (float)spreadSpeed;
                        simValue[(int)center_tris[lowestNeigh]] += oldSimValue[i] * (float)spreadSpeed;
                    }
                }
                else
                { // Water stays in the same cell
                    simValue[i] += oldSimValue[i];
                }
            }
            else
            {
                //Set to average of children
                simValue[i] = (oldSimValue[(int)a_tris[i]] + oldSimValue[(int)b_tris[i]] + oldSimValue[(int)c_tris[i]] +
                               oldSimValue[(int)center_tris[i]]) / 4.0f;
            }

        return simValue.ToArray();
    }

    private int FindMinIndex(float[] trisRadius)
    {
        var minIdx = 0;
        var minVal = trisRadius[0];
        for (var i = 1; i < trisRadius.Length; i++)
            if (trisRadius[i] < minVal)
            {
                minVal = trisRadius[i];
                minIdx = i;
            }

        return minIdx;
    }
    
    public override void Forward(CesState state, double deltaT)
    {
        // var centers = state.GetCenterPoints();
        // var tDivided = state.t_divided.GetTensorDataAsSpan<bool>().ToArray();
        // var tNeightAb = state.t_neight_ab.GetTensorDataAsSpan<long>().ToArray();
        // var tNeightBc = state.t_neight_bc.GetTensorDataAsSpan<long>().ToArray();
        // var tNeightCa = state.t_neight_ca.GetTensorDataAsSpan<long>().ToArray();
        // var oldSimValue = state.sim_value.GetTensorDataAsSpan<double>().ToArray();
        // var a_tris = state.t_a_t.GetTensorDataAsSpan<long>().ToArray();
        // var b_tris = state.t_b_t.GetTensorDataAsSpan<long>().ToArray();
        // var c_tris = state.t_c_t.GetTensorDataAsSpan<long>().ToArray();
        // var center_tris = state.t_center_t.GetTensorDataAsSpan<long>().ToArray();
        //
        // var newSim = ModelForward(deltaT, centers, tDivided, a_tris, b_tris, c_tris, center_tris, tNeightAb,
        //     tNeightBc, tNeightCa, oldSimValue);
        //
        // state.UpdateSimValue(newSim);
        
    }
}