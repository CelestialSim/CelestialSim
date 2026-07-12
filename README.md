# CelestialSim

**GPU-realized procedural planets for Godot.**

Drop a `Celestial` node into a 3D scene and you get a whole planet with adaptive
chunked-quadtree LOD: the CPU picks a screen-space-error cut, the GPU realizes the
geometry and bakes the surface detail. Nothing in the terrain path reads back from the
GPU, so flying toward the surface doesn't cause lag spikes.

### 📖 Everything — install, tutorials, API — is at **<https://celestialsim.github.io/CelestialSim/>**

> **Beta.** Usable today, but pre-1.0: APIs can change between releases. Bug reports and
> feedback are welcome on
> [GitHub](https://github.com/CelestialSim/CelestialSim/issues) or
> [Discord](https://discord.gg/bfCcWkstRB).

## What you get

- **An adaptive-LOD planet node** — one `Celestial` node; detail follows the camera.
- **GPU noise terrain out of the box** — a new planet is never blank.
- **Your own terrain** — write a `.glsl` and it runs on the GPU. No fork, no Rust.
- **A CPU path too** — build terrain in GDScript when you need CPU-side data, synchronously
  or off the main thread. Slower than the GPU path.
- **An analytic ocean** — sea level and water colours live on the builder.
- **GPU scatter objects** — grass, trees and rocks placed and culled on the GPU.

## Install

Download the latest zip from the
[Releases page](https://github.com/CelestialSim/CelestialSim/releases) and copy
`addons/celestialsim/` into your Godot **4.7** project. There is no plugin to enable and
no Slang to install. Prebuilt for Linux, Windows and macOS —
[full instructions](https://celestialsim.github.io/CelestialSim/install/).

## Links

- **Docs** — <https://celestialsim.github.io/CelestialSim/>
- **How it works** — <https://celestialsim.github.io/CelestialSim/architecture/>
- **Web demo of the GPU subdivision** — <https://compute.toys/view/3159>
- **Playable demo builds** — <https://github.com/Calonca/CelestialSimDemo/releases>
- **Discord** — <https://discord.gg/bfCcWkstRB>

## Building from source

```bash
cargo build -p celestialsim            # add --release for an optimized build
```

You do not need `slangc`: the compute SPIR-V is committed and baked into the extension.
For the Rust API reference, run `cargo doc -p celestialsim --no-deps --open`.
