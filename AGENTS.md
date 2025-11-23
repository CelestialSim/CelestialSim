# CelestialSim - Project Overview

## What It Does
CelestialSim is a Godot plugin for rendering planetary/celestial bodies with **adaptive LOD (Level of Detail)** terrain. It dynamically subdivides an icosphere mesh based on camera distance and screen-space triangle size, providing efficient real-time terrain rendering.

## How Shaders Are Called
The system uses **Slang compute shaders** executed via Godot's RenderingDevice API:
- C# wrapper classes (e.g., `CesDivCheckShader`, `CesDivConstraintShader`, `CesDivShader`) dispatch compute shaders
- `CesComputeUtils.DispatchShader()` binds GPU buffers and executes shader code
- Shaders run in a loop until no more triangles need subdivision

## Main Shaders

### 1. **CesDivCheckShader** (`CesDivCheckShader.slang`)
- Calculates screen-space area of each triangle using perspective projection
- Marks triangles for division if they exceed the `max_tri_size` threshold
- Sets `t_to_div` mask for large triangles

### 2. **CesDivConstraintShader** (`CesDivConstraintShader.slang`)
- Validates triangles marked for division
- Checks if each triangle has a difference of max 1 LOD level with neighbors
- Prevents complexity in division algorithm

### 3. **CesDivShader** (`CesDivShader.slang`)
- Performs actual triangle subdivision (1→4 split)
- Creates 3 new midpoint vertices per triangle
- Generates 4 child triangles (corners + center)
- Updates neighbor relationships and parent-child links

### 4. **ComputeNormals** (`ComputeNormals.slang`)
- Calculates triangle normals via cross product
- Prepares final mesh data for rendering
- Handles both flat (low-poly) and smooth shading

## Algorithm Overview

1. **Initialize**: Start with an icosphere base mesh (12 vertices, 20 triangles)

2. **Check Phase**: For each frame:
   - `CesDivCheckShader` evaluates triangle screen size
   - Flags large triangles for subdivision

3. **Validate Phase**:
   - `CesDivConstraintShader` enforces topology rules
   - Ensures mesh remains valid (level transitions)

4. **Subdivide Phase**:
   - `CesDivShader` splits validated triangles
   - Each triangle → 4 child triangles
   - Updates graph structure (neighbors, parents, levels)

5. **Layer Processing**:
   - Apply transformations (e.g., `CesNormalizePos` to sphere surface)
   - Optional terrain layers can modify vertex positions

6. **Output**: Generate final mesh with normals for rendering

The loop continues until no triangles need further subdivision, achieving optimal detail where the camera is looking.



# Development Notes

Build after a feature is implemented or a bug is fixed.
Do not use the task but use the command:

```
dotnet build
```