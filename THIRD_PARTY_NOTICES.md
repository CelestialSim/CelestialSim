# Third-party notices

Third-party code and assets bundled in this repository, and the licenses under
which they are used. Full license texts are reproduced at the end of this file.

---

## Terrain noise — `crates/celestialsim/shaders/terrain_noise_3d.slang`

| Component | Author | Source | License |
|---|---|---|---|
| `hash3()`, `noised3d()` — gradient noise with analytic derivatives | Inigo Quilez | https://iquilezles.org/articles/gradientnoise/ | MIT |
| `snoise()` + `mod289`/`permute`/`taylor_inv_sqrt` — 3D simplex noise | Ian McEwan, Ashima Arts; HLSL translation by Keijiro Takahashi | https://github.com/ashima/webgl-noise · https://github.com/keijiro/NoiseShader | MIT — © 2011 Ashima Arts |
| `terrain_height` elevation composition (`fbm_layers`, `ridged_layers`, `smoothed_ridged`, `smooth_max`) | Sebastian Lague | https://github.com/SebLague/Solar-System (`EarthHeight.compute`, `FractalNoise.cginc`, `Math.cginc`) | MIT — © 2020 Sebastian Lague |
| `LAND_POS` / `LAND_COLOR` ramp stop structure (colour values modified in this project) | Daniel Andrino | https://github.com/dandrino/terrain-erosion-3-ways (`util.py`, `_TERRAIN_CMAP`) | MIT — © 2018 Daniel Andrino |

## Water shader — `addons/celestialsim/water_surface.gdshader`

| Component | Author | Source | License |
|---|---|---|---|
| `ray_sphere()`, `blend_rnm()`, `triplanar_normal()`, ocean shading in `fragment()` | Sebastian Lague | https://github.com/SebLague/Solar-System (`Math.cginc`, `Triplanar.cginc`, `OceanEffect.shader`) | MIT — © 2020 Sebastian Lague |

The Reoriented Normal Mapping technique underneath `blend_rnm` is by Colin
Barré-Brisebois & Stephen Hill (self-shadow.com) and Ben Golus; technique
credit only, no code obligation.

## CPU terrain noise — `crates/celestialsim/src/noise_provider.rs`

| Component | Author | Source | License |
|---|---|---|---|
| `fade()`, `grad()`, `perlin3()` | Ken Perlin | "Improving Noise", SIGGRAPH 2002; reference implementation at https://mrl.cs.nyu.edu/~perlin/noise/ | Published algorithm; the reference page states no license. A possible intermediate source, siv::PerlinNoise (Ryo Suzuki), is MIT — its text is reproduced below. |
| `hash3()` integer-hash constants (`374_761_393`, `668_265_263` = `XXH_PRIME32_5/4`) | Yann Collet | https://github.com/Cyan4973/xxHash | BSD-2-Clause (two numeric constants only) |

## Scatter placement hash — `crates/celestial-algo/src/scatter.rs`, `shaders/ScatterPlace.slang`

`wang()` is Thomas Wang's 32-bit integer hash. No known restriction.
The multiplier `0x9e3779b9` (2³²/φ, Knuth) is a mathematical constant.

## Icosphere base mesh — `crates/celestial-algo/src/icosphere.rs`

`BASE_VERTICES` / `BASE_FACES` follow Andreas Kähler's 2009 listing
(http://blog.andreaskahler.com/2009/06/creating-icosphere-mesh-in-code.html).
The vertices are the cyclic permutations of `(0, ±1, ±φ)` — mathematical facts
about the icosahedron. Courtesy credit; no license stated at the source.

## Barycentric coordinates — `crates/celestial-algo/src/clipmap.rs`

`FaceFrame::project_camera` uses the `d00..d21` formulation from Christer
Ericson, *Real-Time Collision Detection* §3.4. De minimis; courtesy credit.

## Example tree assets — `assets/trees/oak_medium_{branches,leaves}.res`

Oak meshes and bundled bark/leaf textures produced with **ez-tree**
(https://github.com/dgreenheck/ez-tree), MIT — © 2024 Daniel Greenheck.
Demo content only; not part of the installable addon.

## Dependencies (not vendored here)

Pulled via Cargo; licenses live with each crate. No GPL/LGPL/AGPL. The full
per-crate list — with license texts for the crates compiled into the shipped
binaries — is the generated `THIRD_PARTY_LICENSES_DEPS.md`
(`python3 tools/gen_dep_licenses.py`; CI regenerates it and fails on drift or
on any copyleft license entering the graph).

- **`godot` / gdext** (godot-rust) — **MPL-2.0**, statically linked into the
  distributed cdylibs.

  **MPL-2.0 §3.2 notice.** The binaries distributed with this project contain
  code covered by the Mozilla Public License 2.0
  (<https://mozilla.org/MPL/2.0/>). The Corresponding Source is the
  `godot-rust/gdext` repository, used unmodified, at the commit pinned in this
  project's `Cargo.lock`: <https://github.com/godot-rust/gdext>. You may
  obtain, modify and redistribute that code under the terms of the MPL.

- **Godot Engine** — MIT. Exported demo builds bundle the engine and carry its
  notice (`Engine.get_license_text()`).
- `bytemuck` (Zlib/Apache-2.0/MIT), `glam`, `libc`, `heck`, `nanoserde`,
  `proc-macro2`, `quote`, `syn` (MIT/Apache-2.0), `venial` (MIT),
  `unicode-ident` (MIT/Apache-2.0 AND Unicode-3.0).
- **Slang** (`slangc`, Apache-2.0 WITH LLVM-exception) — dev-only tool used to
  regenerate the committed SPIR-V; it does not ship, and its output embeds no
  compiler runtime code.

---

# Full license texts

## MIT — Inigo Quilez (gradient noise)

Copyright (c) 2013 Inigo Quilez

Permission is hereby granted, free of charge, to any person obtaining a copy of this software
and associated documentation files (the "Software"), to deal in the Software without
restriction, including without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or
substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

## MIT — Ashima Arts (simplex noise)

Copyright (C) 2011 by Ashima Arts (Simplex noise)
Copyright (C) 2011-2016 by Stefan Gustavson (Classic noise and others)

Permission is hereby granted, free of charge, to any person obtaining a copy of this software
and associated documentation files (the "Software"), to deal in the Software without
restriction, including without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or
substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

## MIT — Sebastian Lague (water shader + terrain elevation composition)

Copyright (c) 2020 Sebastian Lague

Permission is hereby granted, free of charge, to any person obtaining a copy of this software
and associated documentation files (the "Software"), to deal in the Software without
restriction, including without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or
substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

## MIT — Daniel Andrino (terrain colour ramp)

Copyright (c) 2018 Daniel Andrino

Permission is hereby granted, free of charge, to any person obtaining a copy of this software
and associated documentation files (the "Software"), to deal in the Software without
restriction, including without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or
substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

## MIT — Ryo Suzuki (siv::PerlinNoise)

Copyright (c) 2013-2020 Ryo Suzuki <reputeless@gmail.com>

Permission is hereby granted, free of charge, to any person obtaining a copy of this software
and associated documentation files (the "Software"), to deal in the Software without
restriction, including without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or
substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

## MIT — ez-tree (example oak assets)

Copyright (c) 2024 Daniel Greenheck

Permission is hereby granted, free of charge, to any person obtaining a copy of this software
and associated documentation files (the "Software"), to deal in the Software without
restriction, including without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or
substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
