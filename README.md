# ⚠️ DRAFT — Developer Release Only
> **This repository is a work in progress and under active development.**
>
> - Releases are intended for developers and contributors only.
> - This is *not* production-ready — expect breaking changes, unstable APIs, and incomplete features.
> - If you are not a developer or contributor, please do not use or rely on these releases.
>

# Installation

1. Download the latest `celestial-<version>.zip` from the [Releases page](https://github.com/CelestialSim/CelestialSim/releases).
2. Extract the zip — it contains an `addons/` directory with `celestialsim/`.
3. Copy `addons/celestialsim/` into your Godot project's `addons/` folder.
4. Open the project in Godot — the GDExtension is loaded by `addons/celestialsim/celestialsim.gdextension`.

The release zip ships prebuilt binaries for Linux x86_64, Windows x86_64, and macOS (universal: x86_64 + arm64). Slang is dev-only (the compiled SPIR-V is baked into the extension), so no Slang plugin is needed at runtime or in the editor.

## Development

Build the extension from the `crates/` Cargo workspace with `cargo build -p celestialsim` (add `--release` for an optimized build).

# Documentation

The documentation CI publishes a small landing page plus the Rust API reference generated from `cargo doc`.

To build the Rust docs locally:
1. Run `cargo doc -p celestialsim --no-deps`;
2. Open `target/doc/celestialsim/index.html`.

To build and open the docs in one step, run `cargo doc -p celestialsim --no-deps --open`.
