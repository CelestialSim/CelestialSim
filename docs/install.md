# Install

!!! warning "Draft documentation"
    Releases are intended for developers and contributors only. Expect breaking changes.

1. Download the latest `celestial-<version>.zip` from the [Releases page](https://github.com/CelestialSim/CelestialSim/releases).
2. Extract the zip — it contains an `addons/` directory with both `celestial/` and `celestial_hud/`.
3. Copy both `addons/celestialsim/` and `addons/celestial_hud/` into your Godot project's `addons/` folder.
4. Open the project in Godot — the GDExtension is loaded by `addons/celestialsim/celestialsim.gdextension`, and the editor HUD plugin is enabled from Project Settings.

The release zip ships prebuilt binaries for Linux x86_64, Windows x86_64, and macOS (universal: x86_64 + arm64). Slang is dev-only (the compiled SPIR-V is baked into the extension), so no Slang plugin is needed at runtime or in the editor.
