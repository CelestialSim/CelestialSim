# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

CelestialSim is a Godot 4.6 plugin that renders planetary bodies with adaptive-LOD terrain. The active implementation is a Rust **GDExtension** (`rust/`) using `godot-rust/gdext`, paired with **Slang** compute shaders living under `addons/celestial_sim/shaders/`. The compiled extension is loaded by `addons/celestial_sim/celestial_sim_rust.gdextension`.

## Build and run

```bash
# Primary build (also the VS Code "build" task)
cd rust && cargo build           # debug
cd rust && cargo build --release # release
```

`rust/build.rs` does two important things on every build:
1. Invokes `slangc` to compile every `*.slang` file under `addons/celestial_sim/shaders/` to SPIR-V into `OUT_DIR` (entry `computeMain`, stage `compute`). If `slangc` is missing, build emits a warning and shader tests are skipped — install [shader-slang](https://github.com/shader-slang/slang) and put `slangc` on `PATH` (or set `SLANGC=/path/to/slangc`).
2. Symlinks (or copies on Windows when symlinks are restricted) the freshly built cdylib from `rust/target/<profile>/` into `addons/celestial_sim/bin/<os>/<profile>/` so Godot picks it up via the `.gdextension` config. **Do not** hand-place binaries into `addons/celestial_sim/bin/` — let `build.rs` manage them.

For Slang import inside the Godot editor, install the [godot-slang](https://github.com/DevPrice/godot-slang) plugin (already vendored under `addons/shader-slang/`).

## Tests

```bash
cd rust && cargo test                                # all tests
cd rust && cargo test --test test_sphere_terrain     # one integration test
cd rust && cargo test test_name_substring            # by name
```

Shader integration tests (`rust/tests/test_*.rs`) execute compiled SPIR-V on a real GPU via `wgpu`. They will silently skip if `slangc` was not available at build time. `rust/tests/shader_test_utils.rs` is the shared harness for loading the SPIR-V produced by `build.rs`.

## Running scenes from the CLI

Use the `godot-start-scene`, `godot-screenshot-scene`, and `compare-csharp-rust` skills under `.agents/skills/` rather than calling Godot directly — they wrap the helpers in `debug/` and route output to `debug/logs/`.

```bash
./debug/run_scene_with_log.sh "$PWD" "res://scenes/celestial_terrain_demo.tscn"
./debug/run_scene_with_screenshot.py "res://scenes/celestial_terrain_demo.tscn" --delay 5
```

The runners resolve Godot from `$GODOT_PATH`, then `godot4`, then `godot`, then a Flatpak install. Logs land in `debug/logs/*.log`; screenshots in `debug/logs/screenshots/`.

## Architecture

### Subdivision pipeline (the core loop)

The LOD algorithm lives in `rust/src/algo/` and is orchestrated by `algo::run_algo::CesRunAlgo` (config: `RunAlgoConfig`). Each iteration of the loop runs a chain of compute shaders against GPU buffers held in `state::CesState`:

1. `mark_tris` — `MarkTrisToDivide.slang` flags triangles whose screen-space size exceeds `triangle_screen_size`.
2. `div_lod` — `DivideLOD.slang` performs 1→4 subdivision for flagged triangles, creating midpoint vertices and child triangles.
3. `merge_lod` — `MergeLOD.slang` collapses triangles whose subdivision is no longer needed.
4. `compact_buffers` — `CompactTris.slang` / `CompactVertices.slang` / prefix-sum shaders compact the sparse vertex/triangle arrays back into dense buffers.
5. `update_neighbors` — `UpdateNeighbors.slang` rebuilds neighbor links and enforces the rule that adjacent triangles differ by at most one LOD level.
6. **Layers** (see below) modify vertex positions / generate per-vertex data.
7. `final_state` — `CreateFinalOutput.slang` + `ComputeNormals` produce the final mesh arrays handed back to Godot via `ArrayMesh`.

The loop continues until no triangles need further subdivision. `initial_state.rs` seeds the buffers from a base icosphere (12 verts, 20 tris). Per-step timing is summarised by `AlgoTimingTotals::print_summary` and printed when `show_debug_messages` is enabled.

### Layers

`rust/src/layers/` defines the `CesLayer` trait and concrete implementations:
- `sphere_terrain` — projects vertices onto the sphere surface (`SphereTerrain.slang`).
- `height_shader_terrain` — runs a user-supplied terrain-height Slang shader (`TerrainHeight.slang`, `terrain_noise_3d.slang`, etc.) to displace vertices.
- `scatter` — places per-instance transforms for grass/trees/etc. via `ScatterPlacement.slang`; results feed a `MultiMesh`.

Layers are configured from Godot via `CesXxxLayerResource` types in `rust/src/layer_resources.rs` and the matching GDScript wrappers under `addons/celestial_sim/layers/*.gd`. The Godot-facing entry node is `CelestialSim` in `rust/src/celestial.rs`, which wires resources into runtimes, dispatches the algo, and pushes results back into Godot meshes/multimeshes.

### Compute dispatch wrapper

`rust/src/compute_utils.rs` is the thin layer around Godot's `RenderingDevice` for binding storage buffers and dispatching compute shaders. `buffer_info.rs` describes buffer layouts; `texture_gen.rs` and `camera_snapshot_texture.rs` handle GPU texture generation (cubemaps, snapshot patches). `compositor.rs` wires the `ces_final_state_compositor.glsl` post-processing pass.

### Scene / addon layout

- `addons/celestial_sim/` — the shippable plugin (gdextension config, Slang shaders, GDScript layer wrappers, compiled binaries under `bin/`).
- `addons/shader-slang/` — third-party Slang import support for Godot.
- `scenes/` — demo / benchmark scenes. `celestial_terrain_demo.tscn` is the main playground.
- `debug/scenes/`, `debug/scripts/` — wrapper scenes/scripts used by the screenshot/benchmark skills.

## Conventions specific to this codebase

- **Pure helpers extracted for testability** — `celestial.rs` factors small pure functions (e.g. `count_enabled_flags`, `zeroed_scatter_results`) out of `Gd<T>`-bound methods specifically so they can be unit-tested without a Godot runtime. Follow this pattern when adding new logic to Godot-bound types.
- **Null-safety on typed Godot arrays** — `non_null_elements` in `celestial.rs` uses `catch_unwind` to skip `<empty>` resource slots in editor-exposed `Array<Gd<T>>`. Use it whenever iterating user-editable resource arrays; raw `arr.get(i)` will panic on nil.
- **Adding a new Slang shader** — drop it in `addons/celestial_sim/shaders/` with a `computeMain` entry point. `build.rs` will pick it up automatically. Reference the SPIR-V from Rust by stem name (`<stem>.spv` in `OUT_DIR`).

## Documentation

Rust API docs (published by `.github/workflows/docs.yml`):

```bash
cargo doc --manifest-path rust/Cargo.toml --no-deps --open
```
