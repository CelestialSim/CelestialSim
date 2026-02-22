# Documentation

To build the docs locally:
1. Install/update DocFX (`dotnet tool install -g docfx` or `dotnet tool update -g docfx`);
2. Regenerate API YAML: `~/.dotnet/tools/docfx metadata docs/docfx.json`;
3. Build the site: `~/.dotnet/tools/docfx build docs/docfx.json`;
4. Preview locally: `~/.dotnet/tools/docfx serve docs/docfx/site`.

Inherited members are collapsed using a native DocFX custom template hook in `docs/template/public/main.js`.
