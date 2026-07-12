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
`device = CPU` in `_init` (see below) — so you can't pick a broken
CPU-on-a-bare-builder combo.

## Assigning a builder

In the inspector, click the **Builder** property → **New**, then pick
`CesGPUNoiseExample`, `CesCPUNoiseExample`, or `CesBuilder` (for custom terrain). The knobs a
type doesn't use aren't shown.

## Making your own

- **GPU custom:** create a `CesBuilder` (it's GPU by default) and point
  `shader_file` at your `.glsl` (defines `terrain_height` / `terrain_color`,
  optional `terrain_normal`) — or subclass it to add `@export` params.
- **CPU custom:** write a GDScript that `extends CesBuilder`, sets `device = CPU`
  in `_init`, and defines batched `height` / `color` / `normal`, then assign it.
  You can add your own `@export` parameters on that subclass.
- **CPU custom, async:** the same, but define `_bake_requested` instead of
  `height`/`color`. The planet hands you chunks and never waits; you call
  `submit_chunk` when each is ready. Use this whenever the bake is slow or has to
  wait on a file or the network.

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
