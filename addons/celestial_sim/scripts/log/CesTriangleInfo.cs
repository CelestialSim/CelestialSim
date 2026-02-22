using System.Runtime.InteropServices;

namespace CelestialSim;

/// <summary>
/// Represents comprehensive information about a triangle in the celestial mesh.
/// This struct matches the output buffer layout of GetTriangleInfo.slang shader.
/// </summary>
[StructLayout(LayoutKind.Sequential)]
public struct CesTriangleInfo
{
    // Triangle vertex indices
    public int VertexA;
    public int VertexB;
    public int VertexC;

    // Triangle level (subdivision depth)
    public int Level;

    // Triangle flags
    public int IsDivided;      // 0 = not divided, 1 = divided
    public int IsDeactivated;  // 0 = active, 1 = deactivated
    public int IsToDivide;     // 0 = not marked for division, 1 = marked for division

    // Neighbour triangle indices
    public int NeighbourAB;    // Triangle sharing edge AB
    public int NeighbourBC;    // Triangle sharing edge BC
    public int NeighbourCA;    // Triangle sharing edge CA

    // Icosphere index (0-19 for base icosphere)
    public int IcosphereIndex;

    // Children triangle indices (valid only if IsDivided == 1)
    public int ChildA;         // Child triangle at vertex A
    public int ChildB;         // Child triangle at vertex B
    public int ChildC;         // Child triangle at vertex C
    public int ChildCenter;    // Central child triangle

    // Parent triangle index (-1 if this is a root triangle)
    public int Parent;

    // Vertex positions
    public Godot.Vector4 PosA;
    public Godot.Vector4 PosB;
    public Godot.Vector4 PosC;

    public override readonly string ToString()
    {
        return $"Triangle Info:\n" +
               $"  Vertices: [{VertexA}, {VertexB}, {VertexC}]\n" +
               $"  Vertex Positions:\n" +
               $"    A[{VertexA}]: {PosA}\n" +
               $"    B[{VertexB}]: {PosB}\n" +
               $"    C[{VertexC}]: {PosC}\n" +
               $"  Level: {Level}\n" +
               $"  IsDivided: {IsDivided}\n" +
               $"  IsDeactivated: {IsDeactivated}\n" +
               $"  IsToDivide: {IsToDivide}\n" +
               $"  Neighbours: AB={NeighbourAB}, BC={NeighbourBC}, CA={NeighbourCA}\n" +
               $"  IcosphereIndex: {IcosphereIndex}\n" +
               $"  Children: A={ChildA}, B={ChildB}, C={ChildC}, Center={ChildCenter}\n" +
               $"  Parent: {Parent}";
    }
}
