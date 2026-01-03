# ⚠️ DRAFT — Developer Release Only
> **This repository is a work in progress and under active development.**
>
> - Releases are intended for developers and contributors only.
> - This is *not* production-ready — expect breaking changes, unstable APIs, and incomplete features.
> - If you are not a developer or contributor, please do not use or rely on these releases.
>

# Installation

1. Download the latest release zip from the [Releases page](https://github.com/CelestialSim/CelestialSim/releases)
2. Extract the zip file
3. Copy the `celestial_sim` folder to your project's `addons` folder

## Development

To compile Slang shaders for development across multiple platforms (Windows, macOS, Linux), use the Slang Godot plugin available at https://github.com/DevPrice/godot-slang. Install it from the Godot Asset Library.

# Troubleshooting
The plugin is not working, what should I do ?
- Open the celestial scene and check that the scene image is the one of a planet.
- If this is not the case enable the plugin inside Project/Project Settings/Plugins
- Reopen Godot

# Advance features
### Converting Slang Shaders to c code

```
dotnet build
```

Use 
```
slangc filename.slang -o filename_out.cpp
```

Compile the output C++ file to a shared library using
```
g++ -shared -fPIC -O2 Tests/MultiplyTest_out.cpp -o Tests/libMultiplyTest.so
```

Then the library can be loaded in C#.

To run tests use
godot Tests/scene_name.tscn --quit-after 3