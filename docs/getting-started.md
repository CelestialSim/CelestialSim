# Getting Started

> [!WARNING]
> **Draft documentation:** this documentation is still in progress and missing key pages/sections.

## Prerequisites

- Godot with .NET support
- .NET SDK 8.0+

## Build the plugin

From the repository root:

```bash
dotnet build
```

## Build and preview the docs

From the repository root:

```bash
dotnet tool install -g docfx # or: dotnet tool update -g docfx
~/.dotnet/tools/docfx metadata docs/docfx.json
~/.dotnet/tools/docfx build docs/docfx.json
~/.dotnet/tools/docfx serve docs/docfx/site
```

`serve` starts a local docs server and prints the local URL in the terminal.