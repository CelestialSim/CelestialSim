using Godot;
using Array = Godot.Collections.Array;

namespace CelestialSim;

public class CesDebugUtils
{

    /// <summary>
    /// Gets the ray origin and direction from the global mouse position in local celestial coordinates.
    /// </summary>
    /// <returns>A tuple containing the ray origin and direction in local coordinates, or null if the camera is not available.</returns>
    public static (Vector3 origin, Vector3 direction)? GetMouseRayInLocalCoordinates(CesCelestial celestial)
    {
        Camera3D camera = EditorInterface.Singleton.GetEditorViewport3D().GetCamera3D();

        // Use the camera's forwacelestial.rd, direction (-Z axis in camera's local space)
        var cameraTransform = camera.GlobalTransform;
        var globalOrigin = camera.GlobalPosition;
        var globalDirection = -cameraTransform.Basis.Z; // Camera looks down -Z axis

        // GD.Print($"Global Origin: {globalOrigin}");
        // GD.Print($"Global Direction: {globalDirection}");

        // Transform to local coordinates
        var localOrigin = celestial.posToLocal(globalOrigin);
        var localDirection = celestial.GlobalTransform.Basis.Inverse() * globalDirection;

        return (localOrigin, localDirection);
    }

    /// <summary>
    /// Checks if a ray intersects any triangle using a compute shader.
    /// </summary>
    public static (int triangleIndex, float distance)? CheckRayTriangleIntersection(CesCelestial celestial, Vector3 rayOrigin, Vector3 rayDirection)
    {
        if (celestial.graphGenerator?.State == null)
        {
            GD.Print("Graph generator or state not initialized");
            return null;
        }

        var state = celestial.graphGenerator.State;

        // Debug: Print ray info
        // GD.Print($"Shader Input - Ray Origin: {rayOrigin}, Direction: {rayDirection}");
        // GD.Print($"Total Triangles: {state.nTris}");

        // Debug: Check masks
        var divMask = state.GetDividedMask();
        var deactMask = state.GetTDeactivatedMask();
        int visibleCount = 0;
        for (int i = 0; i < divMask.Length; i++)
        {
            if (divMask[i] == 0 && deactMask[i] == 0)
                visibleCount++;
        }
        GD.Print($"Visible triangles: {visibleCount} out of {state.nTris}");

        var shaderPath = "res://addons/celestial_sim/scripts/debug/RayTriangleIntersection.slang";

        // Create output buffers
        var hitTriangleIndexBuffer = CesComputeUtils.CreateStorageBuffer(celestial.rd, new[] { -1 });
        var hitDistanceBuffer = CesComputeUtils.CreateStorageBuffer(celestial.rd, new[] { 1e30f });

        var bufferInfos = new BufferInfo[]
        {
            state.v_pos,                                                               // vertices
            state.t_abc,                                                               // triangle indices
            CesComputeUtils.CreateUniformBuffer(celestial.rd, rayOrigin),                       // ray origin
            CesComputeUtils.CreateUniformBuffer(celestial.rd, rayDirection),                    // ray direction
            hitTriangleIndexBuffer,                                                    // output hit triangle index
            hitDistanceBuffer,                                                         // output hit distance
            CesComputeUtils.CreateUniformBuffer(celestial.rd, state.nTris),                     // total triangles
            state.t_divided,                                                           // divided mask
            state.t_deactivated                                                        // deactivated mask
        };

        // Dispatch with only 1 thread since we're using a for loop in the shader
        CesComputeUtils.DispatchShader(celestial.rd, shaderPath, bufferInfos, 1);

        // Read results
        var hitIndex = CesComputeUtils.ConvertBufferToArray<int>(celestial.rd, hitTriangleIndexBuffer)[0];
        var hitDist = CesComputeUtils.ConvertBufferToArray<float>(celestial.rd, hitDistanceBuffer)[0];

        // Debug output
        GD.Print($"Shader Output - Hit Index: {hitIndex}, Distance: {hitDist}");

        // Clean up temporary buffers
        celestial.rd.FreeRid(hitTriangleIndexBuffer.buffer);
        celestial.rd.FreeRid(hitDistanceBuffer.buffer);

        if (hitIndex >= 0)
        {
            return (hitIndex, hitDist);
        }

        return null;
    }

    /// <summary>
    /// Increases the level of a specific triangle by the given amount using a compute shader.
    /// </summary>
    public void IncreaseTriangleLevel(CesCelestial celestial, int triangleIndex, int increaseAmount = 2)
    {
        if (celestial.graphGenerator?.State == null)
        {
            GD.Print("Graph generator or state not initialized");
            return;
        }

        var state = celestial.graphGenerator.State;

        var shaderPath = "res://addons/celestial_sim/scripts/debug/IncreaseLevel.slang";

        var bufferInfos = new BufferInfo[]
        {
            state.t_lv,                                                                // triangle level buffer (read-write)
            CesComputeUtils.CreateUniformBuffer(celestial.rd, triangleIndex),                   // triangle index to modify
            CesComputeUtils.CreateUniformBuffer(celestial.rd, increaseAmount)                   // amount to increase
        };

        // Dispatch with only 1 thread since we're modifying a single triangle
        CesComputeUtils.DispatchShader(celestial.rd, shaderPath, bufferInfos, 1);

        // Clean up temporary buffers (uniform buffers for index and amount)
        celestial.rd.FreeRid(bufferInfos[1].buffer);
        celestial.rd.FreeRid(bufferInfos[2].buffer);

        GD.Print($"Triangle {triangleIndex} level increased by {increaseAmount}");
    }

    /// <summary>
    /// Sets the level of a specific triangle to an exact value using a compute shader.
    /// </summary>
    public static void SetTriangleLevel(CesCelestial celestial, int triangleIndex, int newLevel)
    {
        if (celestial.graphGenerator?.State == null)
        {
            GD.Print("Graph generator or state not initialized");
            return;
        }

        var state = celestial.graphGenerator.State;

        var shaderPath = "res://addons/celestial_sim/scripts/debug/SetTriangleLevel.slang";

        var bufferInfos = new BufferInfo[]
        {
            state.t_lv,                                                                // triangle level buffer (read-write)
            CesComputeUtils.CreateUniformBuffer(celestial.rd, triangleIndex),                   // triangle index to modify
            CesComputeUtils.CreateUniformBuffer(celestial.rd, newLevel)                         // new level value
        };

        // Dispatch with only 1 thread since we're modifying a single triangle
        CesComputeUtils.DispatchShader(celestial.rd, shaderPath, bufferInfos, 1);

        // Clean up temporary buffers (uniform buffers for index and level)
        celestial.rd.FreeRid(bufferInfos[1].buffer);
        celestial.rd.FreeRid(bufferInfos[2].buffer);
    }

    /// <summary>
    /// Marks a triangle to be divided on the next mesh update.
    /// </summary>
    public static void DebugMarkTriangleToDivide(CesCelestial celestial, int triangleIndex)
    {
        if (celestial.graphGenerator?.State == null)
        {
            GD.Print("Graph generator or state not initialized");
            return;
        }

        var state = celestial.graphGenerator.State;

        var shaderPath = "res://addons/celestial_sim/scripts/debug/MarkTriangleToDivide.slang";

        var bufferInfos = new BufferInfo[]
        {
            state.t_to_divide_mask,                                                    // to-divide mask buffer (read-write)
            CesComputeUtils.CreateUniformBuffer(celestial.rd, triangleIndex),                   // triangle index to mark
        };

        // Dispatch with only 1 thread
        CesComputeUtils.DispatchShader(celestial.rd, shaderPath, bufferInfos, 1);

        // Clean up temporary buffer
        celestial.rd.FreeRid(bufferInfos[1].buffer);
    }

    /// <summary>
    /// Marks a triangle to be merged on the next mesh update.
    /// The actual merge is performed by CesMergeSmallTris shader during UpdateTriangleGraph.
    /// </summary>
    public static void DebugMarkTriangleToMerge(CesCelestial celestial, int triangleIndex)
    {
        if (celestial.graphGenerator?.State == null)
        {
            GD.Print("Graph generator or state not initialized");
            return;
        }

        var state = celestial.graphGenerator.State;
        var shaderPath = "res://addons/celestial_sim/scripts/debug/MarkTriangleToMerge.slang";


        var bufferInfos = new BufferInfo[]
        {
            state.t_to_merge_mask,                                                    // to-merge mask buffer (read-write)
            CesComputeUtils.CreateUniformBuffer(celestial.rd, triangleIndex),                   // triangle index to mark
        };

        // Dispatch with only 1 thread
        CesComputeUtils.DispatchShader(celestial.rd, shaderPath, bufferInfos, 1);

        // Clean up temporary buffer
        celestial.rd.FreeRid(bufferInfos[1].buffer);
    }

    public static void CheckIntersectionAlgo(CesCelestial celestial)
    {
        var ray = GetMouseRayInLocalCoordinates(celestial);
        if (ray.HasValue)
        {
            var (origin, direction) = ray.Value;

            // Check for triangle intersection
            var hit = CheckRayTriangleIntersection(celestial, origin, direction);
            if (hit.HasValue)
            {
                var (triangleIndex, distance) = hit.Value;

                // Get triangle info to check if already divided
                var triangleInfo = CesTriangleLogger.GetTriangleInfo(celestial, triangleIndex);
                if (triangleInfo.HasValue)
                {
                    if (triangleInfo.Value.IsDivided == 0)
                    {
                        // Mark triangle to be divided
                        DebugMarkTriangleToDivide(celestial, triangleIndex);
                        GD.Print($"Divided 1 triangle (triangle {triangleIndex})");

                        // Apply the division immediately
                        ApplyDebugDivisions(celestial);
                    }
                    else
                    {
                        GD.Print($"Triangle {triangleIndex} is already divided");
                    }
                }
            }
            else
            {
                GD.Print("No triangle hit");
            }
        }
        else
        {
            GD.Print("Camera not available");
        }
    }

    /// <summary>
    /// Applies debug triangle divisions from the DebugTriangleIndicesToDivide array.
    /// Called automatically after the first mesh generation when in debug mode.
    /// Positive indices divide triangles, negative indices merge triangles.
    /// </summary>
    public static void ApplyDebugDivisions(CesCelestial celestial)
    {
        if (celestial.DebugTriangleIndicesToDivide.Count == 0)
            return;

        GD.Print($"Applying {celestial.DebugTriangleIndicesToDivide.Count} debug triangle operations...");

        int dividedCount = 0;
        int mergedCount = 0;

        // Process all triangle indices
        foreach (int value in celestial.DebugTriangleIndicesToDivide)
        {
            if (value >= 0)
            {
                GD.Print($"Dividing triangle {value} in debug mode");

                // Positive index: divide triangle
                DebugMarkTriangleToDivide(celestial, value);
                celestial.ApplyManualDivisions();
                dividedCount++;
            }
            else
            {
                // Negative index: merge triangle
                int triangleIndex = -value;
                GD.Print($"Merging triangle {triangleIndex} in debug mode");

                // Check if triangle is divided before trying to merge
                var triangleInfo = CesTriangleLogger.GetTriangleInfo(celestial, triangleIndex);
                if (triangleInfo.HasValue && triangleInfo.Value.IsDivided == 1)
                {
                    DebugMarkTriangleToMerge(celestial, triangleIndex);
                    celestial.ApplyManualDivisions();
                    mergedCount++;
                }
                else
                {
                    GD.Print($"Triangle {triangleIndex} cannot be merged (not divided or invalid)");
                }
            }
        }


        GD.Print("Manual division/undivision applied");
    }

}
