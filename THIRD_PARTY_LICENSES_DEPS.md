# Third-party licenses — Cargo dependencies

<!-- GENERATED FILE - do not edit. Regenerate with:
     python3 tools/gen_dep_licenses.py -->

Licenses of the Rust crates this project depends on, from the full
transitive graph in `Cargo.lock`. **Runtime-linked** crates are compiled
into the distributed binaries (`libcelestialsim.{so,dll,dylib}`); their
license texts are reproduced below, as their terms require for binary
distribution. **Build-time only** crates (proc-macros / codegen) never
enter the binaries and carry no binary-distribution obligation; they are
listed for completeness. Hand-written third-party notices (vendored code,
shaders, assets) live in `THIRD_PARTY_NOTICES.md`.

| Crate | Version | License | Linkage |
|---|---|---|---|
| bytemuck | 1.25.0 | Zlib OR Apache-2.0 OR MIT | **runtime-linked** |
| bytemuck_derive | 1.10.2 | Zlib OR Apache-2.0 OR MIT | build-time only |
| gdextension-api | 0.5.0 | MPL-2.0 | build-time only |
| glam | 0.32.1 | MIT OR Apache-2.0 | **runtime-linked** |
| godot | 0.5.3 | MPL-2.0 | **runtime-linked** |
| godot-bindings | 0.5.3 | MPL-2.0 | build-time only |
| godot-cell | 0.5.3 | MPL-2.0 | **runtime-linked** |
| godot-codegen | 0.5.3 | MPL-2.0 | build-time only |
| godot-core | 0.5.3 | MPL-2.0 | **runtime-linked** |
| godot-ffi | 0.5.3 | MPL-2.0 | **runtime-linked** |
| godot-macros | 0.5.3 | MPL-2.0 | build-time only |
| heck | 0.5.0 | MIT OR Apache-2.0 | build-time only |
| libc | 0.2.186 | MIT OR Apache-2.0 | **runtime-linked** |
| nanoserde | 0.2.1 | MIT OR Apache-2.0 | build-time only |
| nanoserde-derive | 0.2.1 | MIT | build-time only |
| proc-macro2 | 1.0.106 | MIT OR Apache-2.0 | build-time only |
| quote | 1.0.45 | MIT OR Apache-2.0 | build-time only |
| syn | 2.0.117 | MIT OR Apache-2.0 | build-time only |
| unicode-ident | 1.0.24 | (MIT OR Apache-2.0) AND Unicode-3.0 | build-time only |
| venial | 0.6.1 | MIT | build-time only |

---

## MIT — bytemuck 1.25.0

Copyright (c) 2019 Daniel "Lokathor" Gee.

```
MIT License

Copyright (c) 2019 Daniel "Lokathor" Gee.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice (including the next paragraph) shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
```

## MIT — glam 0.32.1

the glam authors

```
Permission is hereby granted, free of charge, to any
person obtaining a copy of this software and associated
documentation files (the "Software"), to deal in the
Software without restriction, including without
limitation the rights to use, copy, modify, merge,
publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software
is furnished to do so, subject to the following
conditions:

The above copyright notice and this permission notice
shall be included in all copies or substantial portions
of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF
ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED
TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT
SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR
IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THE SOFTWARE.
```

## MPL-2.0 — godot, godot-cell, godot-core, godot-ffi (godot-rust/gdext)

These crates are covered by the Mozilla Public License 2.0
(<https://mozilla.org/MPL/2.0/>). Per MPL-2.0 section 3.2, the
Corresponding Source is the `godot-rust/gdext` repository, used
unmodified at the commit pinned in this project's `Cargo.lock`:
<https://github.com/godot-rust/gdext>. You may obtain, modify and
redistribute that code under the terms of the MPL.

## MIT — libc 0.2.186

Copyright (c) The Rust Project Developers

```
Copyright (c) The Rust Project Developers

Permission is hereby granted, free of charge, to any
person obtaining a copy of this software and associated
documentation files (the "Software"), to deal in the
Software without restriction, including without
limitation the rights to use, copy, modify, merge,
publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software
is furnished to do so, subject to the following
conditions:

The above copyright notice and this permission notice
shall be included in all copies or substantial portions
of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF
ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED
TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT
SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR
IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THE SOFTWARE.
```
