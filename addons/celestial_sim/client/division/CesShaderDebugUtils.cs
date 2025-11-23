using Godot;

namespace ces.Rendering.division;

public class CesShaderDebugUtils
{
    /// <summary>
    /// Spawns a debug sphere in the 3D scene for visualization purposes.
    /// </summary>
    /// <param name="timeAlive">The duration in seconds that the sphere will remain visible before being automatically removed.</param>
    /// <param name="radius">The radius of the debug sphere.</param>
    /// <param name="color">The color of the debug sphere material.</param>
    /// <param name="position">The world position where the sphere should be spawned.</param>
    public static void SpawnDebugSphere(float timeAlive, float radius, Color color, Vector3 position)
    {
        // Get the main scene tree
        var sceneTree = Engine.GetMainLoop() as SceneTree;
        if (sceneTree == null)
        {
            GD.PrintErr("Failed to get SceneTree for debug sphere spawning");
            return;
        }

        var root = sceneTree.Root;
        if (root == null)
        {
            GD.PrintErr("Failed to get root node for debug sphere spawning");
            return;
        }

        // Create the sphere mesh
        var sphereMesh = new SphereMesh
        {
            Radius = radius,
            Height = radius * 2.0f,
            RadialSegments = 16,
            Rings = 8
        };

        // Create a material with the specified color
        var material = new StandardMaterial3D
        {
            AlbedoColor = color,
            ShadingMode = BaseMaterial3D.ShadingModeEnum.Unshaded
        };

        // Create the MeshInstance3D
        var meshInstance = new MeshInstance3D
        {
            Mesh = sphereMesh,
            Position = position
        };
        meshInstance.SetSurfaceOverrideMaterial(0, material);

        // Add to the scene
        root.CallDeferred(Node.MethodName.AddChild, meshInstance);

        // Schedule removal after timeAlive seconds
        if (timeAlive > 0)
        {
            var timer = sceneTree.CreateTimer(timeAlive);
            timer.Timeout += () =>
            {
                if (GodotObject.IsInstanceValid(meshInstance))
                {
                    meshInstance.QueueFree();
                }
            };
        }
    }
}
