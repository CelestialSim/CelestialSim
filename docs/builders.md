# Terrain builders

A planet's terrain comes from its **builder**. The `Celestial`
node has a single **`builder`** property (one active at a time). The base
**`CesBuilder`** is a *custom* builder, and the two built-in noise terrains are
**subclasses** of it. A new planet **starts with a `CesGPUNoiseExample`** so it isn't
blank; **clear the `Builder` slot** to render a plain **white sphere**. Editing
the builder updates the planet live — no scene reload.

## The paths

| Builder | What it renders | You provide |
|---------|-----------------|-------------|
| **`CesGPUNoiseExample`** (subclass) | Built-in GPU noise (the rich inline shader). The default. | nothing — tune the knobs |
| **`CesCPUNoiseExample`** (subclass) | The same noise baked on CPU worker threads. | nothing — tune the knobs |
| **`CesBuilder`** (GPU) | *Your* terrain as a `.glsl`. | a `shader_file` — [3 GLSL functions](custom_terrain_gpu.md) |
| **`CesBuilder` subclass** (CPU) | *Your* terrain in GDScript. | [3 batched GDScript functions](custom_terrain_cpu_gdscript.md) |
| **`CesBuilder` subclass** (CPU, async) | *Your* terrain, baked off the main thread. | [`_bake_requested` + `submit_chunk`](custom_terrain_cpu_gdscript.md#advanced-async-bake-threads-files-network) |

The two **noise** builders are their own resource types (`CesGPUNoiseExample` /
`CesCPUNoiseExample`) and carry the noise knobs. The base **`CesBuilder`** is for
*custom* terrain. Its `device` is set in code, **not** the inspector: a bare
`CesBuilder` you create with **New** is always **GPU** (it shows only
`shader_file`), and a **CPU** builder is always a subclass that sets
`device = 1` (`BuilderDevice.CPU`) in `_init` (see below) — so you can't pick a
broken CPU-on-a-bare-builder combo.

!!! note
    `device` is an integer in GDScript — **`0` = GPU, `1` = CPU**. The
    `BuilderDevice` enum is defined in Rust and shows as a dropdown in the
    inspector, but its names are not exposed to GDScript, so write the number and
    keep the comment: `device = 1  # BuilderDevice.CPU`.

## Assigning a builder

In the inspector, click the **Builder** property → **New**, then pick
`CesGPUNoiseExample`, `CesCPUNoiseExample`, or `CesBuilder` (for custom terrain). The knobs a
type doesn't use aren't shown.

## What every builder has

`CesBuilder` owns the **[water settings](water.md)** — `water_enabled`,
`water_height` (sea level, default `0.549`) and the wave/colour knobs. They are
native to the base class, so *every* builder carries them: the noise examples and
your own custom one alike. Water is a terrain parameter, not a separate feature.

The two noise builders add their own tuning knobs on top (continent frequency,
octaves, ridge strength, relief…). Those belong to the examples, not to
`CesBuilder` — a custom builder has none of them and ignores them, because your
own `terrain_height` decides everything. Just drag them in the inspector; each one
reshades the planet live.

## Making your own

Three routes, each with a step-by-step tutorial. Pick one and follow it:

- **[Custom GPU terrain — write a `.glsl`](custom_terrain_gpu.md)** — the fast
  path, and the one to reach for by default. Two small functions
  (`terrain_height` / `terrain_color`) on a `CesBuilder`, which is GPU already.
  The tutorial also covers
  [exposing your own inspector sliders](custom_terrain_gpu.md#advanced-custom-parameters-expose-sliders-in-the-inspector).
- **[Custom CPU terrain — write GDScript](custom_terrain_cpu_gdscript.md)** — for
  terrain that needs CPU-side data or logic a shader can't reach (a heightmap
  `Image`, world state). Batched `height` / `color` / `normal` on a `CesBuilder`
  subclass with `device = 1`. It bakes on the main thread, so keep `tile_res`
  moderate.
- **[Custom CPU terrain, async](custom_terrain_cpu_gdscript.md#advanced-async-bake-threads-files-network)** —
  the same, but the planet hands you chunks and never waits, and you call
  `submit_chunk` when each is ready. Use it whenever the bake is slow or has to
  wait on a file or the network. It keeps the bake off the frame — though a burst
  of new chunks can still cost you a frame or two, and a Rust builder is faster
  still than either GDScript route.

## The custom surface functions (same shape, both paths)

Both paths define the same three functions over world-space directions on the
unit sphere — **height**, **color**, and optional **normal**:

- **height** — displacement as a fraction of the planet radius (`0` = sea level;
  the geometry clamps negatives to sea level).
- **color** — surface albedo for the returned height.
- **normal** *(optional)* — world normal; omit it and the library
  finite-differences your height for you.

They differ only in form: **GPU** is per-invocation GLSL (`terrain_height(vec3
dir)`), while **CPU** GDScript is **batched** — called once per chunk with a
`PackedVector3Array`, returning one value per direction.

GPU is fast (runs on the GPU, no CPU cost) — prefer it for procedural terrain.
CPU (GDScript) is for terrain that needs CPU-side data or logic; it bakes on the
main thread (one call per chunk), so keep the planet's `tile_res` moderate — or
go [async](custom_terrain_cpu_gdscript.md#advanced-async-bake-threads-files-network),
where a chunk without a surface yet is simply covered by a coarser ancestor
until you submit it.

## Shipping the addon

The two noise builders are GDScript classes under
`addons/celestialsim/builders/` (`ces_gpu_noise.gd` = `CesGPUNoiseExample`, `ces_cpu_noise.gd` = `CesCPUNoiseExample`). Ship that
folder (and its `.uid` files) at that path — scenes and the default builder
reference the scripts by `res://` path, so a project that moves or omits them
loses those types.

→ [Custom GPU terrain (.glsl)](custom_terrain_gpu.md)
→ [Custom CPU terrain (GDScript)](custom_terrain_cpu_gdscript.md)
