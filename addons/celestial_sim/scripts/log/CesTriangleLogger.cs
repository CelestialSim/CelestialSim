using Godot;

namespace CelestialSim;

public static class CesTriangleLogger
{

    /// <summary>
    /// Gets comprehensive information about a specific triangle using a compute shader.
    /// Returns vertex indices, level, flags (divided, deactivated, to_divide), 
    /// neighbours, icosphere index, and children triangles.
    /// </summary>
    public static CesTriangleInfo? GetTriangleInfo(CesCelestial celestial, int triangleIndex)
    {
        if (celestial.graphGenerator?.State == null)
        {
            GD.Print("Graph generator or state not initialized");
            return null;
        }

        var state = celestial.graphGenerator.State;

        // Check if triangle index is valid
        if (triangleIndex < 0 || triangleIndex >= state.nTris)
        {
            GD.Print($"Invalid triangle index: {triangleIndex} (valid range: 0-{state.nTris - 1})");
            return null;
        }

        var shaderPath = "res://addons/celestial_sim/scripts/log/GetTriangleInfo.slang";

        // Create output buffer to hold the triangle info struct
        var outputBuffer = CesComputeUtils.CreateStorageBuffer(celestial.rd, new CesTriangleInfo[1]);

        var bufferInfos = new BufferInfo[]
        {
            state.v_pos,                                                               // vertex positions
            state.t_abc,                                                               // triangle indices
            state.t_lv,                                                                // triangle level
            state.t_divided,                                                           // divided mask
            state.t_deactivated,                                                       // deactivated mask
            state.t_to_divide_mask,                                                    // to divide mask
            state.t_neight_ab,                                                         // neighbour AB
            state.t_neight_bc,                                                         // neighbour BC
            state.t_neight_ca,                                                         // neighbour CA
            state.t_ico_idx,                                                           // icosphere index
            state.t_a_t,                                                               // child A triangle
            state.t_b_t,                                                               // child B triangle
            state.t_c_t,                                                               // child C triangle
            state.t_center_t,                                                          // child center triangle
            state.t_parent,                                                            // parent triangle
            CesComputeUtils.CreateUniformBuffer(celestial.rd, triangleIndex),                   // triangle index to query
            outputBuffer                                                               // output buffer
        };

        // Dispatch with only 1 thread since we're querying a single triangle
        CesComputeUtils.DispatchShader(celestial.rd, shaderPath, bufferInfos, 1);

        // Read the result struct directly
        var results = CesComputeUtils.ConvertBufferToArray<CesTriangleInfo>(celestial.rd, outputBuffer);

        // Clean up temporary buffers
        celestial.rd.FreeRid(bufferInfos[16].buffer); // triangle index uniform buffer
        celestial.rd.FreeRid(outputBuffer.buffer);

        return results[0];
    }

}
