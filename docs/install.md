# Install

!!! info "Beta"
    CelestialSim is in beta and pre-1.0: APIs can still change between releases. Pin a
    version and check the release notes before upgrading.

## Requirements

- **Godot 4.7.** The extension declares a lower compatibility minimum, but 4.7 is the
  only version developed and tested against.
- A GPU with Vulkan support (the Forward+ or Mobile renderer).

Prebuilt binaries ship for:

| Platform | Binary |
|----------|--------|
| Linux | x86_64 |
| Windows | x86_64 |
| macOS | universal (x86_64 + arm64) |

Android is **experimental** — an APK is exported for the demo, but it is not a supported
target.

## Steps

1. Download the latest `celestial-<version>.zip` from the
   [Releases page](https://github.com/CelestialSim/CelestialSim/releases).
2. Extract the zip. It contains an `addons/` directory with a single `celestialsim/`
   folder inside.
3. Copy `addons/celestialsim/` into your Godot project's `addons/` folder (create
   `addons/` if your project doesn't have one yet).
4. Open the project in Godot.

That's it. The GDExtension is loaded automatically by
`addons/celestialsim/celestialsim.gdextension`.

!!! note "Nothing to enable, nothing else to install"
    There is **no plugin to enable** in Project Settings → Plugins — CelestialSim is a
    GDExtension, not an `EditorPlugin`. You also do **not** need Slang: the compute
    shaders are compiled to SPIR-V ahead of time and baked into the binary.

## Verify it worked

Open (or create) a 3D scene and click **Add Child Node**, then type `Celestial` in the
search box. If the node type shows up, the extension loaded and you're ready to build
[your first planet](getting-started.md).

## Troubleshooting

**`Celestial` doesn't appear in Add Child Node.** Restart the Godot editor. GDExtension
classes are registered when the editor starts, so a project that was already open when
you copied the addon in won't see them.

**Still missing after a restart.** Copy the whole `addons/celestialsim/` folder from the
zip rather than individual files: the `.gdextension` resolves the library from
`addons/celestialsim/bin/`, so an addon missing that folder loads nothing. Godot prints a
load error to the **Output** panel if it cannot find or open the library.
