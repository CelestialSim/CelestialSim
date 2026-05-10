# ⚠️ DRAFT — Developer Release Only
> **This repository is a work in progress and under active development.**
>
> - Releases are intended for developers and contributors only.
> - This is *not* production-ready — expect breaking changes, unstable APIs, and incomplete features.
> - If you are not a developer or contributor, please do not use or rely on these releases.
>

# Installation

1. Download the latest `celestial_sim-<version>.zip` from the [Releases page](https://github.com/CelestialSim/CelestialSim/releases).
2. Extract the zip — it contains an `addons/` directory with both `celestial_sim/` and `shader-slang/`.
3. Copy both `addons/celestial_sim/` and `addons/shader-slang/` into your Godot project's `addons/` folder.
4. Open the project in Godot — both plugins will be picked up automatically (the GDExtension is loaded by `addons/celestial_sim/celestial_sim_rust.gdextension`).

The release zip ships prebuilt binaries for Linux x86_64, Windows x86_64, and macOS (universal: x86_64 + arm64). The bundled `shader-slang` is the [DevPrice/godot-slang](https://github.com/DevPrice/godot-slang) plugin pinned to a known-good version — you do **not** need to install it separately from the Asset Library.

## Development

If you are working from a source checkout, the `shader-slang` plugin is already vendored under `addons/shader-slang/`.

# Documentation

The documentation CI now publishes a small landing page plus the Rust API reference generated from `cargo doc`.

To build the Rust docs locally:
1. Run `cargo doc --manifest-path rust/Cargo.toml --no-deps`;
2. Open `rust/target/doc/celestial_sim/index.html`.

To build and open the docs in one step, run `cargo doc --manifest-path rust/Cargo.toml --no-deps --open`.
