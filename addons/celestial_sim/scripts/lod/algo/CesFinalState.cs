using System;
using System.Collections.Generic;
using System.Linq;
using Godot;


public class CesFinalState
{
    public static FinalOutput CreateFinalOutput(CesState state, bool lowPoly)
    {
        var rd = state.rd;
        var divMask = state.GetDividedMask().ToArray();
        var deactivatedMask = state.GetTDeactivatedMask().ToArray();
        var level = state.GetLevel().ToArray();
        var neighAb = CesComputeUtils.ConvertBufferToArray<int>(rd, state.t_neight_ab).ToArray();
        var neighBc = CesComputeUtils.ConvertBufferToArray<int>(rd, state.t_neight_bc).ToArray();
        var neighCa = CesComputeUtils.ConvertBufferToArray<int>(rd, state.t_neight_ca).ToArray();
        var tAbcArray = CesComputeUtils.ConvertBufferToArray<CesState.Triangle>(rd, state.t_abc).ToArray();
        var tParent = CesComputeUtils.ConvertBufferToArray<int>(rd, state.t_parent).ToArray();

        var visibleTrisIndices = new List<uint>();
        for (uint i = 0; i < divMask.Length; i++)
        {
            if (divMask[(int)i] == 0 && deactivatedMask[(int)i] == 0)
            {
                visibleTrisIndices.Add(i);
            }
        }

        uint nVisibleTris = (uint)visibleTrisIndices.Count;


        GD.Print($"Number of invisible (deactivate + parents) triangles: {state.nTris - nVisibleTris}");

        var v_pos = state.GetPos();
        // TODO: Implement lowpoly == false
        var norm = new Vector3[(int)(nVisibleTris * 3)];

        var color = new Vector2[(nVisibleTris * 3)];

        // Create unique vertices for each triangle (low-poly shading)
        var pos = new Vector3[(int)(nVisibleTris * 3)];
        var tris = new int[(int)(nVisibleTris * 3)];

        Vector3 Midpoint((int v0, int v1) edge)
        {
            return (v_pos[edge.v0] + v_pos[edge.v1]) * 0.5f;
        }

        void StitchEdge(int triIdx, int outIdx0, int neighborIdx, (int v0, int v1) parentEdge)
        {
            if (neighborIdx < 0 || neighborIdx >= level.Length) return;
            if (level[neighborIdx] >= level[triIdx]) return;

            var midpoint = Midpoint(parentEdge);

            pos[outIdx0] = midpoint;
        }

        for (int i = 0; i < visibleTrisIndices.Count; i++)
        {
            var triIdx = (int)visibleTrisIndices[i];
            var triangle = tAbcArray[triIdx];
            var parentIdx = tParent[triIdx];
            var parentTri = parentIdx >= 0 && parentIdx < tAbcArray.Length ? tAbcArray[parentIdx] : triangle;
            uint vertAIdx = (uint)triangle.a;
            uint vertBIdx = (uint)triangle.b;
            uint vertCIdx = (uint)triangle.c;
            var parentAb = (parentTri.a, parentTri.b);
            var parentBc = (parentTri.b, parentTri.c);
            var parentCa = (parentTri.c, parentTri.a);
            bool isATris = parentTri.a == triangle.a;
            bool isBTris = parentTri.b == triangle.b;
            bool isCTris = parentTri.c == triangle.c;

            // Create three unique vertex positions for this triangle
            int baseIdx = i * 3;
            pos[baseIdx + 0] = v_pos[vertAIdx];
            pos[baseIdx + 1] = v_pos[vertBIdx];
            pos[baseIdx + 2] = v_pos[vertCIdx];

            // Indices just reference the vertices sequentially
            tris[baseIdx + 0] = baseIdx + 0;
            tris[baseIdx + 1] = baseIdx + 1;
            tris[baseIdx + 2] = baseIdx + 2;


            color[baseIdx + 0] = new Vector2(level[triIdx], 0);
            color[baseIdx + 1] = new Vector2(level[triIdx], 0);
            color[baseIdx + 2] = new Vector2(level[triIdx], 0);

            // TODO: Neighs are not well structured at high LOD levels, need to fix
            // see also UpdateNeighbors.slang
            if (level[triIdx] < 4)
            {
                if (isATris)
                {
                    StitchEdge(triIdx, baseIdx + 1, neighAb[triIdx], parentAb);
                    StitchEdge(triIdx, baseIdx + 2, neighCa[triIdx], parentCa);
                }
                else if (isBTris)
                {
                    StitchEdge(triIdx, baseIdx + 0, neighAb[triIdx], parentAb);
                    StitchEdge(triIdx, baseIdx + 2, neighBc[triIdx], parentBc);
                }
                else if (isCTris)
                {
                    StitchEdge(triIdx, baseIdx + 1, neighBc[triIdx], parentBc);
                    StitchEdge(triIdx, baseIdx + 0, neighCa[triIdx], parentCa);
                }
                else // Is center tris
                {
                    StitchEdge(triIdx, baseIdx + 0, neighBc[triIdx - 1], parentBc);
                    StitchEdge(triIdx, baseIdx + 1, neighCa[triIdx - 3], parentCa);
                    StitchEdge(triIdx, baseIdx + 2, neighAb[triIdx - 2], parentAb);
                }
            }

        }

        return new FinalOutput
        {
            tris = tris,
            color = color,
            normals = norm,
            pos = pos
        };
    }

    public struct FinalOutput
    {
        public int[] tris;
        public Vector2[] color;
        public Vector3[] normals;
        public Vector3[] pos;
    }
}

// public class BufferCleaner
// {
//     public List<Rid> buffersToClean = new();
//
//     public void ScheduleClean(Rid buffer)
//     {
//         buffersToClean.Add(buffer);
//     }
//
//     public void CleanBuffers(RenderingDevice rd)
//     {
//         foreach (var buffer in buffersToClean) rd.FreeRid(buffer);
//         buffersToClean.Clear();
//         
//     }
// }
