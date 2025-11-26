# CelestialSim Overview

CelestialSim is a Godot plugin that renders adaptive level-of-detail (LOD) terrain on top of an icosphere base mesh. By dynamically subdividing triangles near the camera while keeping distant regions coarse, it delivers Earth-scale planets without exhausting GPU budget.

## What It Does

- Builds on an icosphere bootstrap (12 vertices, 20 triangles) stored in [`CesState`](xref:ces.Rendering.CesState)
- Tracks triangle neighbors, levels, division flags, and removal masks inside compute-friendly buffers
- Uses Slang compute shaders driven through Godot's `RenderingDevice` to keep CPU usage low while achieving smooth, continuous LOD

## How Shader Dispatch Works

High-level control sits inside lightweight C# wrappers that plug into Godot nodes or tools:

1. `CesDivCheckShader` evaluates every active triangle, measuring screen-space size and storing intermediate values. Its only job is to mark elements in `t_to_divide_mask`.
2. [`CesDivConstraintShader`](xref:CelestialSim.addons.celestial_sim.client.division.CesDivConstraintShader) validates the triangles flagged for subdivision and clears any that would introduce more than a 1-level jump with neighbors.
3. [`CesDivShader`](xref:CelestialSim.addons.celestial_sim.client.division.CesDivShader) performs the actual 1→4 split, creating midpoint vertices, updating neighbor relationships, and maintaining the hierarchy used by [`CesStateVerifier`](xref:ces.Rendering.CesStateVerifier).
4. [`CesComputeUtils`](xref:ces.Rendering.division.CesComputeUtils) wraps common buffer setup plus the `RenderingDevice.Dispatch()` call so each stage can focus on pure shader logic.

Each wrapper loops until no more triangles request subdivision, ensuring consistent detail budgets per frame.

## Main Shader Responsibilities

| Stage | Purpose | Inputs/Outputs |
| --- | --- | --- |
| **CesDivCheckShader.slang** | Projects each triangle to screen space, compares against `max_tri_size`, and populates `t_to_divide_mask`. | Reads vertex positions (`v_pos`) and triangle indices (`t_abc`); writes to `t_to_divide_mask` and a temporary size buffer. |
| **CesDivConstraintShader.slang** | Keeps mesh topology healthy by enforcing a 1-level difference between neighbors before division continues. | Reads neighbor buffers (`t_neigh_*`), division masks, and level data; clears disallowed entries from `t_to_divide_mask`. |
| **CesDivShader.slang** | Creates midpoint vertices, 4 child triangles, and rewires parent/neighbors for validated triangles. | Consumes validated `t_to_divide_mask`; writes to vertex buffers, triangle tables, and parent/child relationships. |
| **ComputeNormals.slang** | Generates per-triangle normals via cross products so both flat (low-poly) and smooth shading paths have fresh data each frame. | Reads final vertex positions and triangle topology; writes to normal buffers for rendering. |

## Adaptive LOD Loop

1. **Initialize** – Load the base icosphere mesh into `CesState`, configure GPU buffers via [`BuffersCache`](xref:ces.Rendering.BuffersCache), and prime any custom simulation layers (e.g., [`CesSimLayer`](xref:ces.Rendering.Sims.CesSimLayer)).
2. **Check** – Run `CesDivCheckShader` to flag triangles whose projected area exceeds `max_tri_size`.
3. **Validate** – Call `CesDivConstraintShader` so that no neighboring triangle differs by more than one LOD level.
4. **Subdivide** – Dispatch `CesDivShader` to split each validated triangle, update neighbor graph data, and enqueue any follow-up work.
5. **Layer Processing** – Optional layers like [`CesNormalizePos`](xref:CelestialSim.scripts.client.Layers.CesNormalizePos) or water simulations modify positions before rendering.
6. **Output** – Execute `ComputeNormals` plus export routines (see [`CesFinalOutput`](xref:ces.Rendering.division.CesFinalOutput)) to feed Godot meshes and materials.

These steps repeat until no additional triangles exceed the subdivision threshold, allowing CelestialSim to react instantly to camera movement while keeping the triangle budget predictable.
