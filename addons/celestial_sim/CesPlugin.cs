#if TOOLS
using Godot;

[Tool]
public partial class CesPlugin : EditorPlugin
{
    public override void _EnterTree()
    {
        // Initialization of the plugin goes here.
        var celestialScript = GD.Load<Script>("res://addons/celestial_sim/client/CesCelestial.cs");
        var normalizeLayerScript = GD.Load<Script>("res://addons/celestial_sim/client/Layers/CesNormalizePos.cs");
        var icon = GD.Load<Texture2D>("res://addons/celestial_sim/Assets/icon.png");
        AddCustomType("Celestial", "Node3D", celestialScript, icon);
        AddCustomType("NormalizePosLayer", "Node3D", normalizeLayerScript, icon);
    }

    public override void _ExitTree()
    {
        // Clean-up of the plugin goes here.
        RemoveCustomType("Celestial");
        RemoveCustomType("NormalizePosLayer");
    }
}
#endif