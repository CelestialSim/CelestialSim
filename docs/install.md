# Install

!!! warning "Draft documentation"
    Releases are intended for developers and contributors only. Expect breaking changes.

1. Download the latest `celestial_sim-<version>.zip` from the [Releases page](https://github.com/CelestialSim/CelestialSim/releases).
2. Extract the zip — it contains an `addons/` directory with both `celestial_sim/` and `shader-slang/`.
3. Copy both `addons/celestial_sim/` and `addons/shader-slang/` into your Godot project's `addons/` folder.
4. Open the project in Godot — both plugins are picked up automatically (the GDExtension is loaded by `addons/celestial_sim/celestial_sim_rust.gdextension`).

The release zip ships prebuilt binaries for Linux x86_64, Windows x86_64, and macOS (universal: x86_64 + arm64). The bundled `shader-slang` is the [DevPrice/godot-slang](https://github.com/DevPrice/godot-slang) plugin pinned to a known-good version — you do **not** need to install it separately from the Asset Library.
