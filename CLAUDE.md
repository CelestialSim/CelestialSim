# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

CelestialSim is a Godot 4.7 plugin that renders planetary bodies with adaptive-LOD
**chunked-quadtree** terrain. The implementation is a Rust **GDExtension** built from the
**`crates/` Cargo workspace** using `godot-rust/gdext`, paired with **Slang** compute
shaders under `crates/celestialsim/shaders/`. The Godot addon `addons/celestial` loads the
compiled cdylib via `addons/celestialsim/celestialsim.gdextension`.

This repo is the **library** plus a minimal committed example scene
(`scenes/celestial_v5.tscn`). Two sibling repos hold the rest of the project (see
CEL-55 three-way restructure):
- **`../CelestialSimDemo`** — the full demo/benchmark scenes (HQ terrain, freefly camera).
- **`../CelestialGrass`** — the vegetation/grass code.

## Build and run

```bash
cargo build -p celestialsim            # debug
cargo build -p celestialsim --release  # release
```

A normal `cargo build` needs **no `slangc`** and ships no slang runtime.

### Slang is dev-only (CEL-61)

Compute SPIR-V is **committed** under `crates/celestialsim/shaders/spirv/*.spv` and baked
into the cdylib via `include_bytes!`. When you edit a `crates/celestialsim/shaders/*.slang`,
regenerate and re-commit the artifacts:

```bash
SLANG_RECOMPILE=1 cargo build -p celestialsim   # rewrites shaders/spirv/*.spv
git add crates/celestialsim/shaders/spirv
```

Only this regenerate path needs `slangc` on `PATH` (or `$SLANGC`). `slangc` output is
byte-stable, so CI runs the regenerate path + `git diff --exit-code` to reject stale
committed SPIR-V.

## Tests

```bash
cargo test                 # whole workspace
cargo test -p celestialsim    # the GDExtension crate only
cargo test test_name       # by name substring
```

`crates/celestial-algo` holds the pure-CPU LOD invariant tests (quadtree selection,
chunk-cache residency — no GPU). The debug-only `chunk_gpu_test` node runs `ChunkRealize`
on a local `RenderingDevice` and checks positions against the CPU reference; it is
excluded from release builds.

## Running scenes from the CLI

`scenes/celestial_v5.tscn` is a committed example: a navigable `Celestial`
planet with a freefly camera + stats HUD (`scripts/quadtree_chunks.gd`). Run it with
`godot --path . scenes/celestial_v5.tscn` (needs a real GPU/display — works from the
toolbx container). The `.agents/skills/` screenshot/start-scene skills wrap the helpers
in `debug/` and route output to `debug/logs/`.

### Debug scenes and scratch go in `debug/`

When you need a throwaway scene, driver script, or repro to reproduce a bug or verify a
fix (e.g. a `.tscn` + `.gd` that drives the planet and quits), put it under **`debug/`**,
never in `scenes/` or `scripts/`. `debug/` is gitignored (`.gitignore`: `/debug/`), so
these stay out of commits and out of the shippable tree. Run them with
`godot --path . debug/<scene>.tscn` (needs a real GPU/display — works from the toolbx
container). Reserve `scenes/` for committed example scenes only. Clean up `debug/`
artifacts (and any `radv_dumps_*` GPU hang dumps) when done.

## Architecture

The terrain is **GPU-realized**: the CPU selects a screen-space-error cut over a fixed
triangular quadtree (constant cost, independent of triangle count), and Slang compute
shaders realize each chunk's geometry and bake its surface-detail atlas straight into an
indirect `MultiMesh` on Godot's main `RenderingDevice` — there is no CPU readback.
Geometry and texture are cached per chunk, so only newly-visible chunks are re-realized.

### The planet node

`crate::celestial::Celestial`
(`crates/celestialsim/src/celestial.rs`) is the Godot-facing `Node3D`. Each frame it
selects a chunk cut on the CPU (`celestial_algo::quadtree::select_chunks` over the base
icosphere faces, with horizon culling), diffs it against the resident set
(`celestial_algo::chunk_cache::ChunkCache`, an LRU with stable pool slots), packs the
newly-visible chunks into descriptors (`crate::chunk_descriptors`), and schedules a job
on the render thread that realizes + bakes only those chunks. Per-instance geomorph
factors ride in the per-frame instance buffer so a moving camera re-uploads only that
buffer (the cached realize/atlas is untouched). New-chunk admission is throttled per
frame (`max_bakes_per_frame`) to amortize bake spikes.

### The GPU graph pipeline

`crate::chunk_pipeline` runs the terrain work as a `celestial-graph`: the operation nodes
`upload` → `realize` → `bake` wired through descriptor/vertex/atlas resources, executed
inside one render-thread job. Dirtiness is data *inside* the recording context (`Ctx`),
never graph topology — `Graph::execute` re-records only dirty nodes in topological order.

- `crates/celestialsim/src/chunk_nodes/` — each operation is a self-contained unit struct
  implementing the `PipelineNode` trait (`upload.rs`, `realize.rs`, `bake.rs`). A node
  knows only its static `name()`; the data-driven registry (`build_pipeline`) is the
  single place that allocates GPU resources and wires each node's reads/writes.
- `crates/celestialsim/src/gpu/` — thin helpers over the main `RenderingDevice`
  (`device.rs`, `chunk_gpu.rs`). Everything here must run on the render thread.
- `crates/celestialsim/src/chunk_descriptors.rs` — CPU→GPU std430 packing of chunk
  descriptors and per-instance data (`ChunkGpu` must match `shaders/ChunkRealize.slang`;
  `pack_params` / `pack_instances`). Pool slots are deterministic so re-realizing only
  the newly-resident chunks is safe.
- `crates/celestialsim/src/descriptors.rs` — the shared 64-byte `TerrainGpu` noise params
  (`HeightGpu` + `TextureGpu` → `assemble`) that both realize and bake read.

### Shaders

`crates/celestialsim/shaders/`:
- `ChunkRealize.slang` — realizes a chunk's vertices (interior + crack-free skirts) into
  the shared vertex pool.
- `ChunkTileBake.slang` — bakes a chunk's per-pixel colour + world-normal detail atlas.
- `terrain_noise_3d.slang` — shared terrain-height noise (used by both).
- `TileViewer.slang` — debug-only single-tile texture viewer bake.

The runtime surface material is `addons/celestialsim/terrain_chunk.gdshader` (a Godot
spatial shader — not compiled to SPIR-V); it samples the pool + atlases and applies the
per-instance geomorph blend in the vertex stage.

### Scatter layers (CEL-73)

`Celestial.scatter_layers` takes `CesScatterLayer` resources
(`crates/celestialsim/src/scatter_layer.rs`): one mesh each (**no mesh = inactive
layer**; for grass assign the committed `addons/celestialsim/grass_blade.tres`,
regenerable via `CesScatterLayer.make_grass_blade_mesh()`; the example oak is
`assets/trees/oak_medium_{branches,leaves}.res` — two identically-seeded
layers, same lattice/K/seed ⇒ aligned transforms; extracted by
`debug/gen_oak.gd`, which also recomputes leaf normals to point outward from
the canopy centre so foliage shades as a soft volume), a `layer_name` (mirrored
into `resource_name` so the inspector array reads as named entries), a LIVE
`enabled` toggle, a **`density`** slider, a **`lod_level`** (the fineness of the
stable lattice = the single density+reach knob: higher = denser and
shorter-range; the layer appears only on terrain chunks at depth `>= lod_level -
3` per the per-slot capacity bound, so reach ≈
`face_edge / (2^(lod_level-3) · chunk_res · screen_error)` metres), a **`scale`**
base multiplier (place-side), and LIVE **`min_height`**/**`max_height`** gates on
normalized terrain height 0..1 (default `min_height = 0.45` = sea level → no
underwater; lower `max_height` to keep vegetation off peaks). Placement runs on
the GPU as two pipeline nodes
after bake: `scatter-place` writes stable-lattice candidates (`{transform,
hash01, terrain-height}`, keyed by `(face, level-L cell path, k, seed)` — never
by chunk depth/slot, so instances survive LOD splits; invariants in
`celestial-algo/src/scatter.rs`, where the lattice level is called `lattice_level
L`) into a per-slot pool for newly-realized chunks
only; `scatter-compact` gathers visible slots, gates by `hash01 < density &&
min_height <= h <= max_height`, and appends into the layer's
indirect MultiMesh (no readback). Density/min_height/max_height edits
restage with `realize_count = 0` — only the compact dispatch re-runs;
`lod_level`/`seed`/`scale` edits re-place via
`ChunkCache::invalidate_all`; layer add/remove or `instances_per_cell`/
`max_instances` changes rebuild the job. Shaders: `ScatterPlace.slang` /
`ScatterCompact.slang` (keep `ScatterParamsGpu` in `scatter_descriptors.rs` in
sync).

### Terrain layers

Terrain params split into a geometry (`HeightGpu`) and an albedo (`TextureGpu`) half in
`crate::descriptors`; both feed the same 64-byte `TerrainGpu` the shaders read. The split
is CPU-side ownership so a colour-only edit need not re-stage geometry. `Default` impls
reproduce the HQ scene's effective terrain when no layer resource is assigned.

### Addon / repo layout

- `addons/celestialsim/` — the shippable addon: `celestialsim.gdextension` (loads
  `res://target/<profile>/libcelestialsim.{so,dll,dylib}`) and `terrain_chunk.gdshader`.
- `crates/` — the Cargo workspace: `celestialsim` (GDExtension), `celestial-algo` (pure CPU
  math), `celestial-graph` (engine-agnostic GPU computation graph).
- `scenes/` / `scripts/` — the committed example scene + its freefly driver.
- `debug/` — wrapper scripts used by the screenshot/benchmark skills (gitignored).

## Conventions specific to this codebase

- **Self-contained pipeline nodes (CEL-65)** — each operation is its own file
  implementing `PipelineNode`; resource wiring lives only in the registry. Keep new nodes
  resource-agnostic and wire them in `build_pipeline`.
- **Render-thread only** — GPU helpers in `gpu/` touch the global `RenderingDevice` and
  must run on the render thread (via `RenderingServer::call_on_render_thread`).
- **Editing a shader** — change the `.slang`, then regenerate committed SPIR-V with
  `SLANG_RECOMPILE=1 cargo build -p celestialsim` and commit `crates/celestialsim/shaders/spirv/`.
  Keep the `ChunkGpu` (`chunk_descriptors.rs`) and `TerrainGpu` (`descriptors.rs`) layouts
  in sync with the shaders.

## Documentation

Rust API docs (published by `.github/workflows/docs.yml`):

```bash
cargo doc -p celestialsim --no-deps --open
```
