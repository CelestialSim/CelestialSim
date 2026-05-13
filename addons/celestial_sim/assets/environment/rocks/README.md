# Rock scatter assets

The demo scene (`scenes/celestial_terrain_demo.tscn`) scatters
[`rock_moss_set_02`](https://polyhaven.com/a/rock_moss_set_02) from Poly Haven —
seven photogrammetry-derived rocks sharing one PBR material, licensed
[CC0](https://polyhaven.com/license) (public domain, no attribution required,
commercial use OK).

The asset is **not committed** to the repository. To run the demo locally you
need to download it once, place it under this folder, and produce a decimated
copy that the scene actually references.

## Folder layout

After setup the layout looks like this (none of these files are tracked by git):

```
addons/celestial_sim/assets/environment/rocks/
├── README.md                              ← this file (tracked)
└── rock_moss_set_02_1k/
    ├── rock_moss_set_02_1k.gltf           ← downloaded
    ├── rock_moss_set_02.bin               ← downloaded (shared across resolutions)
    ├── rock_moss_set_02_1k_lo.gltf        ← produced by the decimator
    ├── rock_moss_set_02_1k_lo.bin         ← produced by the decimator
    └── textures/
        ├── rock_moss_set_02_diff_1k.jpg   ← downloaded
        ├── rock_moss_set_02_nor_gl_1k.jpg ← downloaded
        └── rock_moss_set_02_rough_1k.jpg  ← downloaded
```

The demo scene references the `_lo.gltf` (≈ 1,170 triangles per rock). The raw
`_1k.gltf` (≈ 8,000 triangles per rock) is the input to the decimator and is
not referenced by any scene directly, but it must be present for the decimator
to read.

## Download the 1k bundle

Easiest path: click the **gltf 1k** download button on
<https://polyhaven.com/a/rock_moss_set_02> and extract the zip into
`addons/celestial_sim/assets/environment/rocks/rock_moss_set_02_1k/`. Make sure
the unzipped folder contents end up at that path (not nested one level deeper
inside an extra `rock_moss_set_02_1k.gltf/` folder — that's how the website's
zip lays itself out, you may need to flatten one level).