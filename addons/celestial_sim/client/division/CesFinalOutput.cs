using System.Linq;
using Godot;

namespace ces.Rendering.division;

public class CesFinalOutput
{
    public static FinalOutput CreateFinalOutput(CesState state, bool lowPoly, BuffersCache cache)
    {
        var rd = state.rd;
        var divMask = state.GetDividedMask();
        var deactivatedMask = state.GetTDeactivatedMask();
        var visibleMask = new int[divMask.Length];
        for (var i = 0; i < divMask.Length; i++)
        {
            visibleMask[i] = (divMask[i] == 0 && deactivatedMask[i] == 0) ? 1 : 0;
        }
        // Here we are making an array of indices to reduce output dim and exclude invisible tris
        // [0,0,0,1,2,2] and gives as an array of size 2
        var visibleTrisMaskSum = CesComputeUtils.SumArrayInPlace(visibleMask, false);
        var numVisibleTris = (uint)visibleTrisMaskSum[^1];
        GD.Print($"Number of invisible (deactivate + parents) triangles: {state.nTris - numVisibleTris}");
        var undividedSumBuffer = CesComputeUtils.CreateStorageBuffer(rd, visibleTrisMaskSum);
        var shaderPath = "res://addons/celestial_sim/client/division/ComputeNormals.slang";
        // var fullLv = state.GetLevel().ToArray();
        // var tris = state.GetDividedTris(divided);
        var visibleTrisBytes = numVisibleTris * 4;
        var v_pos = state.GetPos();
        Vector3[] pos = [];

        Vector3[] norm = [];
        Vector2[] sim = [];
        int[] tris = [];
        var tNormBuf = cache.GetOrCreateBuffer(rd, "t_norm", visibleTrisBytes * 3);
        var dividedTris = cache.GetOrCreateBuffer(rd, "dividedTris", visibleTrisBytes * 3);
        var simValue = cache.GetOrCreateBuffer(rd, "simValue", visibleTrisBytes);
        if (lowPoly) // TODO: Only this branch is currently implemented
        {
            var tripledVertices = cache.GetOrCreateBuffer(rd, "tripledVertices", visibleTrisBytes * 3 * 3);
            var bufferInfos = new BufferInfo[]
            {
                state.v_pos,
                tNormBuf,
                state.t_abc,
                state.t_divided,
                undividedSumBuffer,
                CesComputeUtils.CreateUniformBuffer(rd, state.nTris),
                CesComputeUtils.CreateUniformBuffer(rd, 1), // low poly
                tripledVertices,
                simValue,
                state.t_lv,
                dividedTris,

            };
            CesComputeUtils.DispatchShader(rd, shaderPath, bufferInfos, state.nTris);

            var partNorm = CesComputeUtils.ConvertBufferToArray<Vector3>(rd, tNormBuf);
            norm = new Vector3[(int)(numVisibleTris * 3)];
            for (var i = 0; i < numVisibleTris * 3; i++) norm[i] = partNorm[i / 3];

            var partSim = CesComputeUtils.ConvertBufferToArray<float>(rd, simValue);
            sim = new Vector2[(int)(numVisibleTris * 3)];
            for (var i = 0; i < numVisibleTris * 3; i++) sim[i] = new Vector2(partSim[i / 3], 0);

            pos = CesComputeUtils.ConvertBufferToArray<Vector3>(rd, tripledVertices).ToArray();

            tris = Enumerable.Range(0, (int)(numVisibleTris * 3)).ToArray();
        }
        else
        {
            throw new System.NotImplementedException("Smooth shading branch is not yet implemented.");
            var compressedVertsBytes = state.nVerts * 3 * sizeof(float);
            var compressedVerts = cache.GetOrCreateBuffer(rd, "compressedVerts", compressedVertsBytes);
            var bufferInfos = new BufferInfo[]
            {
                state.v_pos,
                tNormBuf,
                state.t_abc,
                state.t_divided,
                undividedSumBuffer,
                CesComputeUtils.CreateUniformBuffer(rd, state.nTris),
                CesComputeUtils.CreateUniformBuffer(rd, 0), // low poly
                compressedVerts,
                simValue,
                state.t_lv,
                dividedTris
            };
            CesComputeUtils.DispatchShader(rd, shaderPath, bufferInfos, state.nTris);

            norm = new Vector3[state.nVerts];
            sim = new Vector2[state.nVerts];
            var simCount = new float[state.nVerts];
            var tSim = CesComputeUtils.ConvertBufferToArray<float>(rd, simValue);
            var tNorm = CesComputeUtils.ConvertBufferToArray<Vector3>(rd, tNormBuf);
            tris = CesComputeUtils.ConvertBufferToArray<int>(rd, dividedTris).ToArray();
            pos = CesComputeUtils.ConvertBufferToArray<Vector3>(rd, compressedVerts).ToArray();

            for (var i = 0; i < tris.Length; i++)
            {
                var vidx = tris[i];
                norm[vidx] += tNorm[i / 3];
                sim[vidx] += new Vector2(tSim[i / 3], 0);
                simCount[vidx] += 1;
            }

            // Normalize normals
            for (var i = 0; i < state.nVerts; i++)
            {
                norm[i] = norm[i].Normalized();
                sim[i] /= simCount[i];
            }
        }

        return new FinalOutput
        {
            tris = tris,
            sim = sim,
            normals = norm,
            pos = pos
        };
    }

    public struct FinalOutput
    {
        public int[] tris;
        public Vector2[] sim;
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