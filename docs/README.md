# Documentation

To build the docs locally:
1. Install `docfx` via `dotnet tools install docfx -g`;
2. Navigate to the `docs/` directory;
3. Build the necessary files with `docfx metadata` and `docfx build`;
4. Serve via `docfx serve docfx/site`; this folder is the configured output in `docfx.json`.
