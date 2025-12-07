using System.Collections.Generic;
using System.Linq;
using Godot;


public class CesFinalState
{
    public static FinalOutput CreateFinalOutput(CesState state, bool lowPoly, BuffersCache cache)
    {
        var rd = state.rd;
        var divMask = state.GetDividedMask();
        var deactivatedMask = state.GetTDeactivatedMask();
        var level = state.GetLevel();

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
        // t_abc stores Triangle structs with 4 ints (a, b, c, w), so stride is 4 not 3
        var tAbcArray = CesComputeUtils.ConvertBufferToArray<CesState.Triangle>(rd, state.t_abc);

        for (int i = 0; i < visibleTrisIndices.Count; i++)
        {
            var triIdx = (int)visibleTrisIndices[i];
            var triangle = tAbcArray[triIdx];
            uint vertAIdx = (uint)triangle.a;
            uint vertBIdx = (uint)triangle.b;
            uint vertCIdx = (uint)triangle.c;

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
