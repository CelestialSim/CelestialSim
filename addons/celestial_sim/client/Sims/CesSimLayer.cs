using Godot;

namespace ces.Rendering.Sims;

[Tool]
public abstract partial class CesSimLayer : Node3D
{
    public CesCelestial celestial;
    public abstract void Forward(CesState state, double deltaT);
}