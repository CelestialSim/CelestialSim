using System;
using System.Collections.Generic;
using Godot;

public class CesRunAlgo
{

    // public BufferCleaner bufferCleaner = new();
    public CesCelestial gen;
    public Vector3[] Norm;
    public Vector3[] Pos;

    public Vector2[] SimValue = [];

    public CesState State;
    public int[] Triangles = [];

    public void LayersUpdate(CesState s)
    {
        var layers = gen.Layers;
        for (var i = 0; i < layers.Count; i++)
        {
            var layer = layers[i];
            // Copy state to layer
            if (i == 0)
            {
                layer.SetState(s, gen.Radius);
            }
            else
            {
                layer.SetState(layers[i - 1]);
            }

            // Update layer values
            layer.UpdatePos();

            // Copy values back to state
            // if (i == layers.Count - 1)
            //     layer.SetBackState();
        }
    }

    public virtual void UpdateTriangleGraph(Vector3 camlocal, bool skipAutoDivisionMarking = false)
    {
        if (State == null)
        {
            State = CesCoreState.CreateCoreState(gen.rd);
            State.core_state = CesCoreState.CreateCoreState(gen.rd);
            LayersUpdate(State);
        }
        // State = CesCoreState.CreateCoreState(gen.rd);

        // for each subdivision, get the output and use it as input
        CesDivLOD cesDivLOD = new(State.rd);
        CesMergeLOD cesMergeLOD = new(State.rd);
        CesMarkTrisToDivide divCheckShader = new(State.rd);

        uint nTrisAdded = 0;
        uint nTrisMerged = 0;
        bool firstRun = true;
        while (nTrisAdded > 0 || nTrisMerged > 0 || firstRun)
        {
            firstRun = false;

            // Only flag triangles for division if not skipping (used for manual division)
            if (!skipAutoDivisionMarking)
            {
                divCheckShader.FlagLargeTrisToDivide(State, camlocal, gen.Subdivisions, gen.Radius, gen.TriangleScreenSize);
            }

            nTrisAdded = cesDivLOD.MakeDiv(State, gen.PreciseNormals);
            if (nTrisAdded > 0)
            {
                GD.Print($"Divided {nTrisAdded / 4} triangles");
            }

            // Merge small triangles (decrease LOD where needed)
            nTrisMerged = cesMergeLOD.MakeMerge(State);
            if (nTrisMerged > 0)
            {
                GD.Print($"Merged {nTrisMerged / 4} triangle(s) (removed {nTrisMerged} child triangles)");
            }

            // TODO: make this more dynamic based on memory and planet size
            if (State.nDeactivatedTris > 100000)
            {
                GD.Print($"Removing free space inside buffers");
                CesCompactBuffers compactor = new(State.rd);
                compactor.Compact(State);
            }

            //TODO: Generate neighs before getting final output state


            // ------ Layers update --------       
            // TODO: Use a different level for each vertex
            LayersUpdate(State);
            // -----------------------------

        }

        // Retriving output
        var finalOutput = CesFinalState.CreateFinalOutput(State, gen.LowPolyLook);
        Pos = finalOutput.pos;
        Norm = finalOutput.normals;
        Triangles = finalOutput.tris;
        SimValue = finalOutput.color;
    }

    public void Dispose()
    {
        // State?.Dispose();
        // cache.Dispose();
        // BufferInfo.DisposeAllBuffers();
        gen.rd.Free();
    }

    // ~Cessrc()
    // {
    //     GD.Print("ces state Decionstructor called");
    //     Dispose();
    // }
}
