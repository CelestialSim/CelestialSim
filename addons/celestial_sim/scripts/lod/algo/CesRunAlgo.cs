using System;
using System.Collections.Generic;
using System.Diagnostics;
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

    private struct AlgoTimingTotals
    {
        public long TotalTicks;
        public long MarkTicks;
        public long DivideTicks;
        public long MergeTicks;
        public long CompactTicks;
        public long NeighborTicks;
        public long LayersTicks;
        public long FinalTicks;
        public int Iterations;

        public static double ToMs(long ticks)
        {
            return ticks * 1000.0 / Stopwatch.Frequency;
        }

        public void PrintSummary()
        {
            var totalMs = ToMs(TotalTicks);
            GD.Print($"AlgoTiming summary: iterations={Iterations} total={totalMs:0.000} ms");
            GD.Print($"AlgoTiming step MarkTrisToDivide: {ToMs(MarkTicks):0.000} ms");
            GD.Print($"AlgoTiming step Divide: {ToMs(DivideTicks):0.000} ms");
            GD.Print($"AlgoTiming step Merge: {ToMs(MergeTicks):0.000} ms");
            GD.Print($"AlgoTiming step Compact: {ToMs(CompactTicks):0.000} ms");
            GD.Print($"AlgoTiming step UpdateNeighbors: {ToMs(NeighborTicks):0.000} ms");
            GD.Print($"AlgoTiming step LayersUpdate: {ToMs(LayersTicks):0.000} ms");
            GD.Print($"AlgoTiming step FinalOutput: {ToMs(FinalTicks):0.000} ms");

            var slowest = (name: "MarkTrisToDivide", ticks: MarkTicks);
            if (DivideTicks > slowest.ticks) slowest = ("Divide", DivideTicks);
            if (MergeTicks > slowest.ticks) slowest = ("Merge", MergeTicks);
            if (CompactTicks > slowest.ticks) slowest = ("Compact", CompactTicks);
            if (NeighborTicks > slowest.ticks) slowest = ("UpdateNeighbors", NeighborTicks);
            if (LayersTicks > slowest.ticks) slowest = ("LayersUpdate", LayersTicks);
            if (FinalTicks > slowest.ticks) slowest = ("FinalOutput", FinalTicks);
            GD.Print($"AlgoTiming slowest: {slowest.name} ({ToMs(slowest.ticks):0.000} ms)");
        }
    }

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
        var timings = new AlgoTimingTotals();
        var totalStart = Stopwatch.GetTimestamp();
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
        CesUpdateNeighbors updateNeighbors = new(State.rd);

        uint nTrisAdded = 0;
        uint nTrisMerged = 0;
        bool firstRun = true;
        while (nTrisAdded > 0 || nTrisMerged > 0 || firstRun)
        {
            firstRun = false;
            timings.Iterations++;

            // Only flag triangles for division if not skipping (used for manual division)
            if (!skipAutoDivisionMarking)
            {
                var markStart = Stopwatch.GetTimestamp();
                divCheckShader.FlagLargeTrisToDivide(State, camlocal, gen.Subdivisions, gen.Radius, gen.TriangleScreenSize);
                timings.MarkTicks += Stopwatch.GetTimestamp() - markStart;
            }

            var divideStart = Stopwatch.GetTimestamp();
            nTrisAdded = cesDivLOD.MakeDiv(State, gen.PreciseNormals);
            timings.DivideTicks += Stopwatch.GetTimestamp() - divideStart;
            if (nTrisAdded > 0)
            {
                GD.Print($"Divided {nTrisAdded / 4} triangles");
            }

            // Merge small triangles (decrease LOD where needed)
            var mergeStart = Stopwatch.GetTimestamp();
            nTrisMerged = cesMergeLOD.MakeMerge(State);
            timings.MergeTicks += Stopwatch.GetTimestamp() - mergeStart;
            if (nTrisMerged > 0)
            {
                GD.Print($"Merged {nTrisMerged / 4} triangle(s) (removed {nTrisMerged} child triangles)");
            }

            // TODO: make this more dynamic based on memory and planet size
            if (State.nDeactivatedTris > 100000)
            {
                GD.Print($"Removing free space inside buffers");
                CesCompactBuffers compactor = new(State.rd);
                var compactStart = Stopwatch.GetTimestamp();
                compactor.Compact(State);
                timings.CompactTicks += Stopwatch.GetTimestamp() - compactStart;
            }

            // Update neighbors after division/merge
            if (nTrisAdded > 0 || nTrisMerged > 0)
            {
                var neighborStart = Stopwatch.GetTimestamp();
                updateNeighbors.UpdateNeighbors(State);
                timings.NeighborTicks += Stopwatch.GetTimestamp() - neighborStart;
            }

            // ------ Layers update --------       
            // TODO: Use a different level for each vertex
            var layersStart = Stopwatch.GetTimestamp();
            LayersUpdate(State);
            timings.LayersTicks += Stopwatch.GetTimestamp() - layersStart;
            // -----------------------------

        }

        // Retriving output
        var finalStart = Stopwatch.GetTimestamp();
        var finalOutput = CesFinalState.CreateFinalOutput(State, gen.LowPolyLook);
        timings.FinalTicks += Stopwatch.GetTimestamp() - finalStart;
        Pos = finalOutput.pos;
        Norm = finalOutput.normals;
        Triangles = finalOutput.tris;
        SimValue = finalOutput.color;

        timings.TotalTicks = Stopwatch.GetTimestamp() - totalStart;
        if (gen.ShowDebugMessages)
        {
            timings.PrintSummary();
        }
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
