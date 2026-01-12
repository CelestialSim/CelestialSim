using System;
using System.Collections.Generic;
using System.Linq;
using Godot;


public class CesDivLOD
{
    private readonly RenderingDevice _rd;
    private readonly string addTrisPath = "res://addons/celestial_sim/scripts/lod/algo/DivideLOD.slang";
    private readonly string collectTrisPath = "res://addons/celestial_sim/scripts/lod/algo/CollectTrisToDivide.slang";
    private const bool VerifyGpuIndexCollection = false;

    public CesDivLOD(RenderingDevice rd)
    {
        _rd = rd;
    }

    private int[] SortArray(int a, int b)
    {
        return a < b ? [a, b] : [b, a];
    }

    private Span<CesState.Triangle> ComputeNewIndices(CesState s, out uint numNewVerts)
    {
        var toDivMask = s.GetTToDivideMask();
        var tabc = s.GetABCW();
        var startVindex = (int)s.nVerts;
        var startTindex = (int)s.nTris;

        var neightAb = CesComputeUtils.ConvertBufferToArray<int>(_rd, s.t_neight_ab);
        var neightBc = CesComputeUtils.ConvertBufferToArray<int>(_rd, s.t_neight_bc);
        var neightCa = CesComputeUtils.ConvertBufferToArray<int>(_rd, s.t_neight_ca);

        // var xyz = new int[lenDivided, 4]; // Stores new indices for midpoints and the original triangle index
        // var edgeDict = new System.Collections.Generic.Dictionary<Edge, int>(); // Maps edges to new vertex indices

        var localDiv = new bool[toDivMask.Length];
        var addedIdxs = new int[toDivMask.Length * 4, 3];

        var vindex = (int)startVindex;
        var tdivIndex = (int)0;
        // For each triangle to divide
        for (var i = 0; i < toDivMask.Length; i++)
        {
            if (toDivMask[i] == 0) continue;
            var idx = i;

            var a = tabc[idx].a;
            var b = tabc[idx].b;
            var c = tabc[idx].c;

            // // Define edges of the triangle
            // var edgeAB = new Edge(a, b);
            // var edgeBC = new Edge(b, c);
            // var edgeCA = new Edge(c, a);

            // Process edge AB
            int newVertexAB;
            if (localDiv[neightAb[idx]])
            {
                var neighSide = neightAb;
                int newVertexId;
                // Edge midpoint already computed
                var nab = SortArray(tabc[neighSide[idx]].a, tabc[neighSide[idx]].b);
                var nbc = SortArray(tabc[neighSide[idx]].b, tabc[neighSide[idx]].c);
                // var nca = SortArray(tabc[neighSide[idx]].c, tabc[neighSide[idx]].a);
                var ab = SortArray(a, b);
                if (nab[0] == ab[0] && nab[1] == ab[1])
                    newVertexId = addedIdxs[neighSide[idx], 0];
                else if (nbc[0] == ab[0] && nbc[1] == ab[1])
                    newVertexId = addedIdxs[neighSide[idx], 1];
                else
                    newVertexId = addedIdxs[neighSide[idx], 2];
                newVertexAB = newVertexId;
            }
            else
            {
                // Compute midpoint, add to vertices, assign new index
                newVertexAB = vindex++;
                // edgeDict[edgeAB] = newVertexAB;
            }

            // Process edge BC
            int newVertexBC;
            if (localDiv[neightBc[idx]])
            {
                var neighSide = neightBc;
                int newVertexId;
                // Edge midpoint already computed
                // Edge midpoint already computed
                var nab = SortArray(tabc[neighSide[idx]].a, tabc[neighSide[idx]].b);
                var nbc = SortArray(tabc[neighSide[idx]].b, tabc[neighSide[idx]].c);
                // var nca = SortArray(tabc[neighSide[idx]].c, tabc[neighSide[idx]].a);
                var bc = SortArray(b, c);
                if (nab[0] == bc[0] && nab[1] == bc[1])
                    newVertexId = addedIdxs[neighSide[idx], 0];
                else if (nbc[0] == bc[0] && nbc[1] == bc[1])
                    newVertexId = addedIdxs[neighSide[idx], 1];
                else
                    newVertexId = addedIdxs[neighSide[idx], 2];
                newVertexBC = newVertexId;
            }
            else
            {
                newVertexBC = vindex++;
                // edgeDict[edgeBC] = newVertexBC;
            }

            // Process edge CA
            int newVertexCA;
            if (localDiv[neightCa[idx]])
            {
                var neighSide = neightCa;
                int newVertexId;
                // Edge midpoint already computed
                // Edge midpoint already computed
                var nab = SortArray(tabc[neighSide[idx]].a, tabc[neighSide[idx]].b);
                var nbc = SortArray(tabc[neighSide[idx]].b, tabc[neighSide[idx]].c);
                // var nca = SortArray(tabc[neighSide[idx]].c, tabc[neighSide[idx]].a);
                var ca = SortArray(c, a);
                if (nab[0] == ca[0] && nab[1] == ca[1])
                    newVertexId = addedIdxs[neighSide[idx], 0];
                else if (nbc[0] == ca[0] && nbc[1] == ca[1])
                    newVertexId = addedIdxs[neighSide[idx], 1];
                else
                    newVertexId = addedIdxs[neighSide[idx], 2];
                newVertexCA = newVertexId;
            }
            else
            {
                newVertexCA = vindex++;
                // edgeDict[edgeCA] = newVertexCA;
            }

            localDiv[idx] = true;

            // Store the new indices
            tabc[startTindex + tdivIndex * 4].a = a;
            tabc[startTindex + tdivIndex * 4].b = newVertexAB;
            tabc[startTindex + tdivIndex * 4].c = newVertexCA;
            //
            tabc[startTindex + tdivIndex * 4 + 1].a = newVertexAB;
            tabc[startTindex + tdivIndex * 4 + 1].b = b;
            tabc[startTindex + tdivIndex * 4 + 1].c = newVertexBC;
            //
            tabc[startTindex + tdivIndex * 4 + 2].a = newVertexCA;
            tabc[startTindex + tdivIndex * 4 + 2].b = newVertexBC;
            tabc[startTindex + tdivIndex * 4 + 2].c = c;
            //
            tabc[startTindex + tdivIndex * 4 + 3].a = newVertexBC;
            tabc[startTindex + tdivIndex * 4 + 3].b = newVertexCA;
            tabc[startTindex + tdivIndex * 4 + 3].c = newVertexAB;
            //store ab vertex
            addedIdxs[idx, 0] = newVertexAB;
            addedIdxs[idx, 1] = newVertexBC;
            addedIdxs[idx, 2] = newVertexCA;

            tdivIndex++;
        }

        numNewVerts = (uint)(vindex - startVindex);
        return tabc;
    }

    private static uint[] BuildCpuIndices(CesState state)
    {
        var toDivMask = state.GetTToDivideMask();
        var dividedMask = state.GetDividedMask();

        if (toDivMask.Length != dividedMask.Length)
            throw new InvalidOperationException("Triangle masks have different lengths.");

        List<uint> indices = new(toDivMask.Length);
        for (var i = 0; i < toDivMask.Length; i++)
        {
            if (toDivMask[i] != 0 && dividedMask[i] == 0)
            {
                indices.Add((uint)i);
            }
        }

        return indices.ToArray();
    }

    private static void ValidateIndices(uint[] cpuIndices, uint[] gpuIndices)
    {
        if (cpuIndices.Length != gpuIndices.Length)
        {
            throw new InvalidOperationException($"GPU divide list mismatch: CPU={cpuIndices.Length}, GPU={gpuIndices.Length}.");
        }

        var sortedCpu = (uint[])cpuIndices.Clone();
        var sortedGpu = (uint[])gpuIndices.Clone();
        Array.Sort(sortedCpu);
        Array.Sort(sortedGpu);

        for (var i = 0; i < sortedCpu.Length; i++)
        {
            if (sortedCpu[i] != sortedGpu[i])
            {
                throw new InvalidOperationException($"GPU divide list mismatch at {i}: CPU={sortedCpu[i]}, GPU={sortedGpu[i]}.");
            }
        }
    }

    private BufferInfo BuildIndicesToDivideBuffer(CesState state, bool captureResult, out uint nTrisToDiv,
        out uint[] gpuSnapshot)
    {
        var outputSize = Math.Max(state.t_to_divide_mask.filledSize, (uint)sizeof(uint));
        // TODO: Move to CesState
        var indicesBuffer = CesComputeUtils.CreateEmptyStorageBuffer(_rd, outputSize);
        var counterBuffer = CesComputeUtils.CreateEmptyStorageBuffer(_rd, sizeof(uint));

        var bufferInfos = new BufferInfo[]
        {
            state.t_to_divide_mask,
            state.t_divided,
            indicesBuffer,
            counterBuffer,
            CesComputeUtils.CreateUniformBuffer(_rd, state.nTris)
        };

        if (state.nTris > 0)
        {
            CesComputeUtils.DispatchShader(_rd, collectTrisPath, bufferInfos, state.nTris);
        }

        var countSpan = CesComputeUtils.ConvertBufferToArray<uint>(_rd, counterBuffer);
        nTrisToDiv = countSpan.Length > 0 ? Math.Min(countSpan[0], state.nTris) : 0;
        indicesBuffer.filledSize = nTrisToDiv * sizeof(uint);

        gpuSnapshot = Array.Empty<uint>();
        if (captureResult && nTrisToDiv > 0)
        {
            var gpuSpan = CesComputeUtils.ConvertBufferToArray<uint>(_rd, indicesBuffer);
            gpuSnapshot = gpuSpan[..(int)nTrisToDiv].ToArray();
        }

        return indicesBuffer;
    }

    public uint MakeDiv(CesState state, bool preciseNormals)
    {
        var removeRepeatedVerts = preciseNormals ? 1u : 0u;
        // Increase size variables
        // var sumArr = CesComputeUtils.SumArrayInPlace(state.GetTToDivideMask());
        // var sumBuffer = CesComputeUtils.CreateStorageBuffer(_rd, sumArr);
        // GD.Print("idxToCheck: " + nTrisToDiv);

        var verifyIndices = VerifyGpuIndexCollection;
        var cpuIndices = verifyIndices ? BuildCpuIndices(state) : Array.Empty<uint>();

        var indicesToDivBuffer = BuildIndicesToDivideBuffer(state, verifyIndices, out var nTrisToDiv,
            out var gpuIndicesSnapshot);

        if (verifyIndices)
        {
            ValidateIndices(cpuIndices, gpuIndicesSnapshot);
        }

        var nTrisAdded = 4 * nTrisToDiv;
        var nVertsAdded = 3 * nTrisToDiv;

        if (nTrisAdded == 0)
        {
            return 0;
        }

        // Compute new indices
        state.t_abc.ExtendBuffer(4 * sizeof(int) * nTrisAdded);
        if (removeRepeatedVerts == 1)
        {
            var newIndices = ComputeNewIndices(state, out nVertsAdded);
            state.t_abc = CesComputeUtils.CreateStorageBuffer(_rd, newIndices);
        }

        state.nTris += nTrisAdded;
        state.nVerts += nVertsAdded;

        // Extend buffers
        state.v_pos.ExtendBuffer(4 * sizeof(float) * nVertsAdded);
        state.v_update_mask.ExtendBuffer(sizeof(int) * nVertsAdded);
        state.t_lv.ExtendBuffer(sizeof(int) * nTrisAdded);
        state.t_divided.ExtendBuffer(sizeof(int) * nTrisAdded);
        state.t_deactivated.ExtendBuffer(sizeof(int) * nTrisAdded);
        state.t_to_divide_mask.ExtendBuffer(sizeof(int) * nTrisAdded);
        state.t_to_merge_mask.ExtendBuffer(sizeof(int) * nTrisAdded);
        state.t_ico_idx.ExtendBuffer(sizeof(int) * nTrisAdded);
        state.t_neight_ab.ExtendBuffer(sizeof(int) * nTrisAdded);
        state.t_neight_bc.ExtendBuffer(sizeof(int) * nTrisAdded);
        state.t_neight_ca.ExtendBuffer(sizeof(int) * nTrisAdded);
        state.t_a_t.ExtendBuffer(sizeof(int) * nTrisAdded);
        state.t_b_t.ExtendBuffer(sizeof(int) * nTrisAdded);
        state.t_c_t.ExtendBuffer(sizeof(int) * nTrisAdded);
        state.t_center_t.ExtendBuffer(sizeof(int) * nTrisAdded);
        state.t_parent.ExtendBuffer(sizeof(int) * nTrisAdded);

        //Add temp test buffer
        // var tempBytes = new byte[state.nTris * sizeof(int)];
        // var emptyBuffer = rd.StorageBufferCreate((uint)tempBytes.Length, tempBytes);
        // Bind buffers
        var bufferInfos = new BufferInfo[]
        {
            state.v_pos,
            state.t_abc,
            state.t_divided,
            state.t_to_divide_mask,
            CesComputeUtils.CreateUniformBuffer(_rd, state.nTris - nTrisAdded),
            CesComputeUtils.CreateUniformBuffer(_rd, state.nVerts - nVertsAdded),
            CesComputeUtils.CreateUniformBuffer(_rd, nTrisToDiv),
            CesComputeUtils.CreateUniformBuffer(_rd, nVertsAdded),
            // sumBuffer,
            state.v_update_mask,
            state.t_ico_idx,
            state.t_neight_ab,
            state.t_neight_bc,
            state.t_neight_ca,
            state.t_a_t,
            state.t_b_t,
            state.t_c_t,
            state.t_center_t,
            state.t_parent,
            CesComputeUtils.CreateUniformBuffer(_rd, removeRepeatedVerts),
            state.t_lv,
            indicesToDivBuffer
        };

        // Dispatch compute shader
        // a check is necessary in the shader to avoid random values filling the
        // elements with the t_to_divide_mask buffer between the old and new triangles
        // happens after 4 divisions

        CesComputeUtils.DispatchShader(_rd, addTrisPath, bufferInfos, (uint)nTrisToDiv);
        return nTrisAdded;
    }
}
