# Custom GPU terrain (write a `.glsl`)

Drive a planet's terrain from a shader **you** write — no Rust, no fork. You
supply a small `.glsl` with two or three functions; CelestialSim compiles it at
runtime (through Godot's own shader compiler — nothing extra ships) and runs it
on the GPU with no CPU cost. This is a **`CesBuilder` on the `GPU` device** (see the
[builders overview](builders.md)).

## 1. Create the `.glsl` file

In the Godot **FileSystem** dock, **right-click a folder → Create New → TextFile**,
and name it `my_terrain.glsl`. Double-clicking a `.glsl` won't open it for editing —
**right-click the file → Open** to edit it in the script editor. Paste in:

```glsl
// REQUIRED — displacement as a FRACTION of the planet radius.
// 0.0 = sea-level sphere; 0.05 = a bump 5% of the radius tall.
// `dir` is the unit world-space direction on the sphere.
float terrain_height(vec3 dir) {
    float land = 0.5 * sin(dir.x * 5.0) * sin(dir.z * 5.0) + 0.25 * dir.y;
    return max(land, 0.0) * 0.05; // sea floored at 0
}

// REQUIRED — surface albedo (linear RGB, 0..1). `h` is terrain_height(dir).
vec3 terrain_color(vec3 dir, float h) {
    if (h <= 0.0001) {
        return vec3(0.03, 0.15, 0.45);           // sea
    }
    float t = clamp(h / 0.05, 0.0, 1.0);
    return mix(vec3(0.2, 0.45, 0.15), vec3(0.9, 0.9, 0.92), t); // green -> snow
}
```

That's the whole contract. The library handles the rest: the per-texel
world-direction mapping (crack-free across chunk and face seams), the surface
normal (finite-differenced from your `terrain_height`), vertex displacement, and
LOD.

### Optional: your own normal

By default the normal comes from `terrain_height` (using the *full* relief, so
mountains shade strongly even if you keep the silhouette round). To compute it
yourself, add the guard define and the function anywhere in the file:

```glsl
#define CELS_CUSTOM_NORMAL
vec3 terrain_normal(vec3 dir, float h) {
    return dir; // e.g. radial = smooth shading
}
```

### Globals you can read

| name | meaning |
|------|---------|
| `CELS_RADIUS` | this chunk's sphere radius (world units) |
| `CELS_WATER_HEIGHT` | the builder's `water_height` (0..1) |
| `CELS_HEIGHT_SCALE` | the builder's `height_scale` |

Other tunables are plain `const`s you edit in the file (editing it recompiles).

## 2. Assign the builder to your planet

On the `Celestial` node, go to the **Builder** property and click
**Revert value** to clear the default noise builder, then choose **New
CesBuilder** and point **Shader File** at `res://my_terrain.glsl`. A bare
`CesBuilder` is always GPU, so **Shader File** is the only field — your `.glsl`
owns everything else: sea level and displacement come from the `CELS_WATER_HEIGHT`
/ `CELS_HEIGHT_SCALE` constants and the height you return.

Run the scene. Editing the `.glsl` and re-running recompiles.

## Advanced: custom parameters (expose sliders in the inspector)

Instead of hard-coded `const`s, you can drive the shader from **inspector
sliders**. It's three steps: write a builder script, write a shader that reads
the params, then swap your builder in on the planet.

### Create the builder GDScript

In the **FileSystem** dock create a script (e.g. `my_terrain.gd`) that
`extends CesBuilder` and declares one `@export var name: float` per knob. In
`_init`, select the GPU-custom route **and point it at your shader** — that keeps
the builder self-contained (its params and its `.glsl` travel together, so you
never have to wire the shader by hand):

```gdscript
@tool
class_name MyTerrain
extends CesBuilder

@export_range(0.5, 8.0, 0.01) var ridge_sharpness: float = 2.0
@export_range(0.0, 1.0, 0.001) var snow_line: float = 0.8

func _init() -> void:
    device = 0  # BuilderDevice.GPU (0 = GPU, 1 = CPU); builtin_shader stays None
    shader_file =  # add the path to your shader here, you can drag and drop from the FileSystem
    # you will have a path similar to the following
    # shader_file = "res://planets/custom_example.glsl"
```

> `device` is an **int** in GDScript: `0` = GPU, `1` = CPU. The `BuilderDevice`
> enum is Rust-side (it renders as a dropdown in the inspector) and its names
> aren't reachable from GDScript, so write the number.

> **Plain `@export` is enough — live updates just work.** The planet polls your
> builder's `@export` float values each frame and reshades when one changes, so a
> slider drag updates the viewport with no `set(v): … emit_changed()` boilerplate.

### Create a shader that uses the parameters

Write a `.glsl` (as in step 1) at the path you referenced above, that reads each
param by its **UPPERCASE** name — `ridge_sharpness` becomes `RIDGE_SHARPNESS`:

```glsl
float terrain_height(vec3 dir) {
    float r = /* ...compute a 0..1 ridge... */;
    return pow(r, RIDGE_SHARPNESS) * 0.06;
}
vec3 terrain_color(vec3 dir, float h) {
    float snow = step(SNOW_LINE, clamp(h / 0.06, 0.0, 1.0));
    return mix(vec3(0.2, 0.45, 0.15), vec3(0.9, 0.9, 0.95), snow);
}
```

### Swap your builder in on the planet

On the `Celestial` node, go to the **Builder** property and click
**Revert value** to clear the default noise builder, then choose **New →
`MyTerrain`** (the script you created). Because `_init` already set the device and
`shader_file`, that's it — your `@export` knobs appear in the inspector and drive
the shader; drag one and the terrain updates live.

> If `MyTerrain` doesn't appear in the **New** list, the editor hasn't picked up
> the new global class yet — **restart the editor** (or **Project → Reload
> Current Project**), which also reloads the latest build of the extension.

Rules:
- Up to **16** float params are supported (extra ones are ignored).
- Editing a param **value** updates the planet **live** (no recompile).
- **Adding, removing, or renaming** a param changes the generated `#define`
  block, so the shader **recompiles** on the next change.
- Only `float` `@export`s become defines; the base fields (`device`,
  `shader_file`, `layer_name`, `builtin_shader`) are never surfaced.

## Driving it from Rust / GDScript

`CesBuilder` is an ordinary resource. Set `device` (`BuilderDevice::GPU` in Rust,
the int `0` in GDScript) and `shader_file` from either language, or
`extends CesBuilder` and set them in `_init` — the `.glsl` is the interface
either way.

## If the shader has an error

A GLSL compile error is logged (with Godot's message + line) and the planet
falls back to the **white sphere** — it never crashes. Watch for
`[celestial] CustomSurface: GLSL compile error`.

Common gotchas:
- Define **both** `terrain_height` and `terrain_color` (checked before compiling;
  the error names the missing one).
- Return `terrain_height` as a *fraction of radius*, not metres.
- It's spliced into a `#version 450` compute template — don't add your own
  `#version` or `layout`, and avoid backticks / non-ASCII in comments.
