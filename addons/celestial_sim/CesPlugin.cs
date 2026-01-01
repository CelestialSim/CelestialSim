#if TOOLS
using Godot;

[Tool]
public partial class CesPlugin : EditorPlugin
{
    public override void _EnterTree()
    {
        // Initialization of the plugin goes here.
        var celestialScript = GD.Load<Script>("res://addons/celestial_sim/scripts/api/CesCelestial.cs");
        var sphereTerrainScript = GD.Load<Script>("res://addons/celestial_sim/scripts/lod/layers/CesSphereTerrain.cs");
        var icon = GD.Load<Texture2D>("res://addons/celestial_sim/assets/icon.png");
        AddCustomType("CesCelestial", "Node3D", celestialScript, icon);
        AddCustomType("CesSphereTerrainLayer", "Node3D", sphereTerrainScript, icon);
    }

    public override void _ExitTree()
    {
        // Clean-up of the plugin goes here.
        RemoveCustomType("CesCelestial");
        RemoveCustomType("CesSphereTerrainLayer");
    }
}
#endif