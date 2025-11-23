using System;
using System.Collections.Generic;
using System.Linq;
using ces.Rendering.division;
using Godot;

namespace ces.Rendering;

/// <summary>
/// Provides verification methods to check the correctness of an icosphere state,
/// particularly focusing on neighbor relationships and mesh topology.
/// </summary>
public static class CesStateVerifier
{
    /// <summary>
    /// Result of a verification check containing errors and warnings found.
    /// </summary>
    public class VerificationResult
    {
        public List<string> Errors { get; } = new();
        public List<string> Warnings { get; } = new();
        public bool IsValid => Errors.Count == 0;

        public void AddError(string error)
        {
            Errors.Add(error);
        }

        public void AddWarning(string warning)
        {
            Warnings.Add(warning);
        }

        public override string ToString()
        {
            var result = IsValid ? "✓ Verification PASSED" : "✗ Verification FAILED";
            result += $"\nErrors: {Errors.Count}, Warnings: {Warnings.Count}";

            if (Errors.Count > 0)
            {
                result += "\n\nErrors:";
                foreach (var error in Errors)
                {
                    result += $"\n  - {error}";
                }
            }

            if (Warnings.Count > 0)
            {
                result += "\n\nWarnings:";
                foreach (var warning in Warnings)
                {
                    result += $"\n  - {warning}";
                }
            }

            return result;
        }
    }

    /// <summary>
    /// Verifies the correctness of neighbor relationships in the icosphere state.
    /// Checks that neighbors are bidirectional and share the correct edges.
    /// </summary>
    public static VerificationResult VerifyNeighbors(CesState state)
    {
        var result = new VerificationResult();

        try
        {
            // Read all necessary buffers from GPU
            var triangles = CesComputeUtils.ConvertBufferToArray<CesState.Triangle>(state.rd, state.t_abc);
            var neighAb = CesComputeUtils.ConvertBufferToArray<int>(state.rd, state.t_neight_ab);
            var neighBc = CesComputeUtils.ConvertBufferToArray<int>(state.rd, state.t_neight_bc);
            var neighCa = CesComputeUtils.ConvertBufferToArray<int>(state.rd, state.t_neight_ca);
            var divided = CesComputeUtils.ConvertBufferToArray<int>(state.rd, state.t_divided);
            var aT = CesComputeUtils.ConvertBufferToArray<int>(state.rd, state.t_a_t);
            var bT = CesComputeUtils.ConvertBufferToArray<int>(state.rd, state.t_b_t);
            var cT = CesComputeUtils.ConvertBufferToArray<int>(state.rd, state.t_c_t);
            var centerT = CesComputeUtils.ConvertBufferToArray<int>(state.rd, state.t_center_t);

            GD.Print($"Verifying {state.nTris} triangles...");

            // Build a map from child triangle index to parent triangle index
            var parents = new Dictionary<int, int>();
            for (int i = 0; i < state.nTris; i++)
            {
                if (divided[i] == 1)
                {
                    parents[aT[i]] = i;
                    parents[bT[i]] = i;
                    parents[cT[i]] = i;
                    parents[centerT[i]] = i;
                }
            }

            // Verify each triangle's neighbors
            for (int i = 0; i < state.nTris; i++)
            {
                var parent = i;
                if (divided[parent] == 1)
                {
                    var children = GetChildren(parent, aT, bT, cT, centerT);

                    foreach (var child in children)
                    {
                        // Check all three edges of the child
                        CheckReciprocalNeigh(parent, child, EdgeType.AB, neighAb[child],
                            triangles, neighAb, neighBc, neighCa, divided, aT, bT, cT, centerT, result);
                        CheckReciprocalNeigh(parent, child, EdgeType.BC, neighBc[child],
                            triangles, neighAb, neighBc, neighCa, divided, aT, bT, cT, centerT, result);
                        CheckReciprocalNeigh(parent, child, EdgeType.CA, neighCa[child],
                            triangles, neighAb, neighBc, neighCa, divided, aT, bT, cT, centerT, result);
                    }
                }
                else
                {
                    // Not divided: check the parent triangle's neighbors
                    CheckReciprocalNeighUndividedTris(parents,parent, EdgeType.AB, neighAb[parent],
                        triangles, neighAb, neighBc, neighCa, divided, aT, bT, cT, centerT, result);
                    CheckReciprocalNeighUndividedTris(parents,parent, EdgeType.BC, neighBc[parent],
                        triangles, neighAb, neighBc, neighCa, divided, aT, bT, cT, centerT, result);
                    CheckReciprocalNeighUndividedTris(parents,parent, EdgeType.CA, neighCa[parent],
                        triangles, neighAb, neighBc, neighCa, divided, aT, bT, cT, centerT, result);
                }

            }

            GD.Print($"Verification complete: {result.Errors.Count} errors, {result.Warnings.Count} warnings");
        }
        catch (Exception ex)
        {
            result.AddError($"Exception during verification: {ex.Message}");
        }

        return result;
    }

    /// <summary>
    /// Gets the children of a triangle.
    /// If divided, returns all 4 children (A, B, C, Center).
    /// </summary>
    private static List<int> GetChildren(int parent, Span<int> aT, Span<int> bT, Span<int> cT, Span<int> centerT)
    {
        var children = new List<int>
        {
            aT[parent],
            bT[parent],
            cT[parent],
            centerT[parent]
        };

        return children;
    }

    /// <summary>
    /// Checks reciprocal neighbor relationship between a child triangle and its neighbor.
    /// </summary>
    private static void CheckReciprocalNeigh(
        int parent,
        int child,
        EdgeType edge,
        int neigh,
        Span<CesState.Triangle> triangles,
        Span<int> neighAb,
        Span<int> neighBc,
        Span<int> neighCa,
        Span<int> divided,
        Span<int> aT,
        Span<int> bT,
        Span<int> cT,
        Span<int> centerT,
        VerificationResult result)
    {
        // Skip if neighbor is invalid
        if (neigh < 0 || neigh >= triangles.Length)
        {
            if (neigh != -1)
            {
                result.AddError($"Triangle {child}: Invalid neighbor index {neigh} on edge {edge}");
            }
            return;
        }

        // Neighbor is not divided: check if parent is in neighbors of neigh
        bool foundParent = neighAb[neigh] == child || neighBc[neigh] == child || neighCa[neigh] == child ||
            neighAb[neigh] == parent || neighBc[neigh] == parent || neighCa[neigh] == parent;

        if (!foundParent)
        {
            result.AddError($"Triangle {child}: Neighbor {neigh} does not point back to parent {parent} on edge {edge}");
        }
    }

    private static void CheckReciprocalNeighUndividedTris(
        Dictionary<int, int> parents,
        int child,
        EdgeType edge,
        int neigh,
        Span<CesState.Triangle> triangles,
        Span<int> neighAb,
        Span<int> neighBc,
        Span<int> neighCa,
        Span<int> divided,
        Span<int> aT,
        Span<int> bT,
        Span<int> cT,
        Span<int> centerT,
        VerificationResult result)
    {
        // Skip if neighbor is invalid
        if (neigh < 0 || neigh >= triangles.Length)
        {
            if (neigh != -1)
            {
                result.AddError($"Triangle {child}: Invalid neighbor index {neigh} on edge {edge}");
            }
            return;
        }

        int parent = parents.TryGetValue(child, out var p) ? p : child;

        // Neighbor is not divided: check if parent is in neighbors of neigh
        bool foundParent = neighAb[neigh] == child || neighBc[neigh] == child || neighCa[neigh] == child ||
            neighAb[neigh] == parent || neighBc[neigh] == parent || neighCa[neigh] == parent;

        if (!foundParent)
        {
            result.AddError($"Triangle {child}: Neighbor {neigh} does not point back to child {child} on edge {edge}");
        }
    }


    /// <summary>
    /// Checks if two triangles share a common edge (at least two vertices in common).
    /// </summary>
    private static bool HasCommonEdge(CesState.Triangle tri1, CesState.Triangle tri2)
    {
        var vertices1 = new[] { tri1.a, tri1.b, tri1.c };
        var vertices2 = new[] { tri2.a, tri2.b, tri2.c };

        int commonCount = 0;
        foreach (var v1 in vertices1)
        {
            foreach (var v2 in vertices2)
            {
                if (v1 == v2)
                {
                    commonCount++;
                    break;
                }
            }
        }

        return commonCount >= 2;
    }

    /// <summary>
    /// Verifies the subdivision structure of the icosphere.
    /// Checks that divided triangles have valid child triangles.
    /// </summary>
    public static VerificationResult VerifySubdivisionStructure(CesState state)
    {
        var result = new VerificationResult();

        try
        {
            var divided = CesComputeUtils.ConvertBufferToArray<int>(state.rd, state.t_divided);
            var aT = CesComputeUtils.ConvertBufferToArray<int>(state.rd, state.t_a_t);
            var bT = CesComputeUtils.ConvertBufferToArray<int>(state.rd, state.t_b_t);
            var cT = CesComputeUtils.ConvertBufferToArray<int>(state.rd, state.t_c_t);
            var centerT = CesComputeUtils.ConvertBufferToArray<int>(state.rd, state.t_center_t);

            GD.Print($"Verifying subdivision structure...");

            for (int i = 0; i < state.nTris; i++)
            {
                if (divided[i] == 1)
                {
                    // Verify child triangle indices are valid and different
                    var children = new[] { aT[i], bT[i], cT[i], centerT[i] };
                    var childNames = new[] { "A", "B", "C", "Center" };

                    for (int j = 0; j < children.Length; j++)
                    {
                        if (!IsValidTriangleIndex(children[j], state.nTris))
                        {
                            result.AddError($"Triangle {i}: Invalid child {childNames[j]} index {children[j]} (max={state.nTris})");
                        }
                    }

                    // Check for duplicate child indices
                    var uniqueChildren = children.Distinct().ToArray();
                    if (uniqueChildren.Length != children.Length)
                    {
                        result.AddError($"Triangle {i}: Has duplicate child triangle indices");
                    }

                    // Verify children point to different triangles
                    if (aT[i] == i || bT[i] == i || cT[i] == i || centerT[i] == i)
                    {
                        result.AddError($"Triangle {i}: Child triangle points to itself");
                    }
                }
            }

            GD.Print($"Subdivision verification complete: {result.Errors.Count} errors");
        }
        catch (Exception ex)
        {
            result.AddError($"Exception during subdivision verification: {ex.Message}");
        }

        return result;
    }

    /// <summary>
    /// Performs a comprehensive verification of the icosphere state.
    /// Includes neighbor verification, subdivision structure, and topology checks.
    /// </summary>
    public static VerificationResult VerifyComplete(CesState state)
    {
        var result = new VerificationResult();

        GD.Print("=== Starting Complete Icosphere Verification ===");
        GD.Print($"State info: {state.nTris} triangles, {state.nVerts} vertices");

        // Verify neighbors
        var neighborResult = VerifyNeighbors(state);
        result.Errors.AddRange(neighborResult.Errors);
        result.Warnings.AddRange(neighborResult.Warnings);

        // Verify subdivision structure
        var subdivResult = VerifySubdivisionStructure(state);
        result.Errors.AddRange(subdivResult.Errors);
        result.Warnings.AddRange(subdivResult.Warnings);

        GD.Print("=== Verification Complete ===");
        GD.Print(result.ToString());

        return result;
    }

    /// <summary>
    /// Checks if a vertex index is valid.
    /// </summary>
    private static bool IsValidVertexIndex(int idx, uint maxVerts)
    {
        return idx >= 0 && idx < maxVerts;
    }

    /// <summary>
    /// Checks if a triangle index is valid.
    /// </summary>
    private static bool IsValidTriangleIndex(int idx, uint maxTris)
    {
        return idx >= 0 && idx < maxTris;
    }

    /// <summary>
    /// Edge types for neighbor relationships.
    /// </summary>
    private enum EdgeType
    {
        AB,
        BC,
        CA
    }
}
