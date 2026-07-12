# Custom CPU terrain in GDScript

Compute terrain in **pure GDScript**, on the CPU. Use this when your terrain
needs CPU-side data a shader can't easily reach — a heightmap `Image`, tiles you
downloaded, world state — or when you'd rather write GDScript than GLSL. This is
a **`CesBuilder` on the `CPU` device** (see the [builders overview](builders.md)).

You define the **same functions** as the GPU path — `height`, `color`, and
optionally `normal` — but in GDScript, and **batched**: each is called once per
chunk with a `PackedVector3Array` of texel directions, returning one value per
direction.

> **Performance:** baking runs on the **main thread**, **one call per chunk** (not
> per texel) — the planet hands you a `PackedVector3Array` of all the chunk's
> texel directions and you return a packed array, so there's no per-texel call
> overhead. The loop still runs in GDScript, though, so for heavy math keep the
> planet's `tile_res` moderate, or use the [GPU path](custom_terrain_gpu.md)
> (thousands of GPU threads, far faster — also authored without Rust).
>
> If the bake is slow — or has to **wait** on a file or the network — don't do it
> here: use [Advanced: async bake](#advanced-async-bake-threads-files-network),
> which never blocks the frame.

## 1. Write the builder script

Create a script that **`extends CesBuilder`**, selects the CPU device in `_init`,
and defines the functions. The `@tool` + `class_name` are what make it show up in
the inspector's **New** list:

```gdscript
@tool
class_name MyCpuTerrain
extends CesBuilder

func _init() -> void:
	device = 1 # BuilderDevice.CPU — the planet bakes height/color on the CPU

# REQUIRED — displacement as a FRACTION of the planet radius (0 = sea level),
# one entry per direction. Each `dirs[i]` is a unit world-space direction.
func height(dirs: PackedVector3Array) -> PackedFloat32Array:
	var out := PackedFloat32Array()
	out.resize(dirs.size())
	for i in dirs.size():
		var d := dirs[i]
		out[i] = maxf(0.5 * sin(d.x * 5.0) * sin(d.z * 5.0) + 0.25 * d.y, 0.0) * 0.05
	return out

# REQUIRED — surface albedo, one entry per direction. `heights[i]` is height()[i].
func color(dirs: PackedVector3Array, heights: PackedFloat32Array) -> PackedColorArray:
	var out := PackedColorArray()
	out.resize(dirs.size())
	for i in dirs.size():
		var h := heights[i]
		if h <= 0.0001:
			out[i] = Color(0.03, 0.15, 0.45)                # sea
		else:
			var t := clampf(h / 0.05, 0.0, 1.0)
			out[i] = Color(0.2, 0.45, 0.15).lerp(Color(0.9, 0.9, 0.92), t)
	return out

# OPTIONAL — world normals. Omit this and the library finite-differences the
# height grid for you (relief shading for free).
#func normal(dirs: PackedVector3Array, heights: PackedFloat32Array) -> PackedVector3Array:
#	var out := PackedVector3Array()
#	out.resize(dirs.size())
#	for i in dirs.size():
#		out[i] = dirs[i]   # radial = smooth shading
#	return out
```

With the device set to **CPU** in `_init` the planet calls your batched
`height`/`color`/`normal` once per chunk on the main-thread baker. To expose
tunables as inspector sliders, see
[Advanced: custom parameters](#advanced-custom-parameters) below.

## 2. Assign it to your planet

On the `Celestial` node, find the **Builder** property (it holds the
default noise builder). Click **Revert value** to clear it, then choose **New →
`MyCpuTerrain`** (the script you created). Set the planet's `tile_res` to a
moderate value (e.g. `128`) — lower it if the per-chunk GDScript loop is too slow.

> If `MyCpuTerrain` doesn't appear in the **New** list, the editor hasn't picked
> up the new global class yet — **restart the editor** (or **Project → Reload
> Current Project**), which also reloads the latest build of the extension.

## Advanced: custom parameters

Expose tunables as **inspector sliders** by adding `@export` variables to your
builder script — then just read them **directly** in `height`/`color`/`normal`.
Unlike the GPU path there's no shader and no `#define`: it's the same GDScript, so
you reference each parameter by its normal (lowercase) name.

```gdscript
@tool
class_name MyCpuTerrain
extends CesBuilder

@export_range(0.5, 8.0, 0.01) var ridge_sharpness: float = 2.0
@export_range(0.0, 1.0, 0.001) var snow_line: float = 0.8

func _init() -> void:
	device = 1 # CPU

func height(dirs: PackedVector3Array) -> PackedFloat32Array:
	var out := PackedFloat32Array()
	out.resize(dirs.size())
	for i in dirs.size():
		var d := dirs[i]
		var r := 1.0 - absf(sin(d.x * 5.0) * sin(d.z * 5.0))  # 0..1 ridge
		out[i] = pow(r, ridge_sharpness) * 0.06               # <- your param
	return out

func color(dirs: PackedVector3Array, heights: PackedFloat32Array) -> PackedColorArray:
	var out := PackedColorArray()
	out.resize(dirs.size())
	for i in dirs.size():
		var t := clampf(heights[i] / 0.06, 0.0, 1.0)
		if heights[i] <= 0.0001:
			out[i] = Color(0.03, 0.15, 0.45)                  # sea
		elif t >= snow_line:                                  # <- your param
			out[i] = Color(0.92, 0.94, 0.97)                  # snow
		else:
			out[i] = Color(0.2, 0.45, 0.15)
	return out
```

Dragging a slider updates the terrain **live** — the planet polls the builder's
`@export` float values each frame and re-bakes when one changes, so no
`emit_changed()` is needed. (Editing the *function code* isn't polled; re-run the
scene, or call `emit_changed()` yourself, to pick that up.)

## Advanced: async bake (threads, files, network)

`height`/`color` run on the **main thread**, so anything slow inside them — a
heavy loop, a file read, an HTTP request — stalls the frame. Define
**`_bake_requested`** instead and the planet stops waiting for you: it hands you
the chunks it wants and moves on. You bake them however you like, and call
**`submit_chunk`** when each one is ready.

While a chunk has no surface it simply isn't drawn at full detail — a coarser
ancestor covers that ground — so a slow bake costs detail, never framerate.

Only three names below belong to this library: the planet **calls**
`_bake_requested` (and, optionally, `_base_ready`) on your builder, and you
**call** `submit_chunk` (and `chunk_dirs`) on it. Everything else — `bake_one`,
member variables, helper functions — is ordinary code of yours, named however
you like.

```gdscript
@tool
class_name MyAsyncTerrain
extends CesBuilder

func _init() -> void:
	device = 1 # CPU

# CALLED BY THE PLANET, main thread, once per frame. Take the work and return —
# never block here.
func _bake_requested(requests: Array) -> void:
	for r in requests:
		WorkerThreadPool.add_task(bake_one.bind(r))

# Just a function of yours. Here it runs on a worker thread; it could equally be
# the callback of an HTTP request, or anything else.
func bake_one(r: Dictionary) -> void:
	var tile_res: int = r["tile_res"]
	var n := tile_res * tile_res
	# Per-texel directions are not in the request (they're big); ask for them.
	var dirs: PackedVector3Array = chunk_dirs(r["corners"], tile_res)

	var heights := PackedFloat32Array()
	heights.resize(n)
	var colors := PackedColorArray()
	colors.resize(n)
	for i in n:
		var d := dirs[i]
		var h := maxf(0.5 * sin(d.x * 5.0) * sin(d.z * 5.0) + 0.25 * d.y, 0.0) * 0.05
		heights[i] = h
		colors[i] = Color(0.2, 0.45, 0.15) if h > 0.0001 else Color(0.03, 0.15, 0.45)

	# Safe to call from any thread. Pass a 4th PackedVector3Array of normals to
	# override the automatic finite-difference.
	submit_chunk(r["handle"], heights, colors)
```

### The request

Each entry of `requests` is a `Dictionary`:

| key | type | meaning |
|---|---|---|
| `handle` | `int` | opaque key — pass it back to `submit_chunk` |
| `corners` | `PackedVector3Array` | the chunk's 3 unit-sphere corners |
| `tile_res` | `int` | bake a `tile_res × tile_res` grid |
| `depth` | `int` | the chunk's LOD depth |

A chunk is requested **once** and not asked for again while you owe it. If the
camera flies past before you answer, the planet forgets it — your late
`submit_chunk` is dropped, and the chunk is re-requested if the player returns.
There is no cancel callback to handle.

### Where you may call `submit_chunk`

**Anywhere, at any time, on any thread, as many times as you want.** There is no
designated place for it — it takes a lock, hands the surface over and returns.
Call it straight from `_bake_requested` if the chunk is already cached; call it
from a worker task; call it from an `HTTPRequest`'s `request_completed` signal
minutes later. The planet picks up whatever has arrived on its next frame.

The only thing that matters is the `handle`: whatever you pass must be the one
from the request you're answering. Keep it alongside your in-flight work.

### Submitting again refines a chunk

`submit_chunk` is **idempotent**: call it again for a handle you already
submitted and the planet re-draws that chunk with the new surface. That's the
whole streaming story — return a coarse tile now, and when the real one
downloads, submit the same handle again:

```gdscript
var _http := HTTPRequest.new()   # a Node; add it to the tree from your scene

func _bake_requested(requests: Array) -> void:
	for r in requests:
		submit_chunk(r["handle"], coarse_heights(r), coarse_colors(r))  # something now
		start_download(r)                                               # better later

# Your own signal handler — nothing to do with CelestialSim.
func _on_download_finished(handle: int, body: PackedByteArray) -> void:
	var heights := decode_heights(body)
	var colors := decode_colors(body)
	submit_chunk(handle, heights, colors)   # replaces the coarse version
```

### Hiding the planet until your data arrives

Add **`_base_ready`** and the planet shows its plain fallback until you say your
base data has landed — no half-loaded terrain during the first downloads:

```gdscript
var _base_map_loaded := false

func _base_ready() -> bool:
	return _base_map_loaded
```

### Notes

- `heights` and `colors` must each hold exactly `tile_res²` entries, row-major.
  A wrong length is reported once and the submission dropped; non-finite heights
  are treated as `0`.
- `chunk_dirs(corners, tile_res)` is pure — safe from a worker thread. Call it
  only if you need per-texel directions; a tile-streaming builder usually wants a
  lat/lon box derived from `corners` instead.
- Heights are in **your** unit; `height_scale` (inherited from `CesBuilder`)
  multiplies them into the displacement. For real-world metres on an Earth-sized
  planet, set `height_scale = exaggeration / 6_371_000.0`.
- Don't do slow work inside `_bake_requested` itself — it runs on the main
  thread. Hand it to a thread, a download, or a queue, and return.
- Defining `_bake_requested` replaces `height`/`color`; don't write both.

## Notes

- **Direction mapping** is handled for you: you receive a `PackedVector3Array` of
  unit world directions — one per texel — and return one value per entry, in the
  same order. (Internally the corners are interpolated and renormalized, matching
  the shaders closely over each small chunk.)
- A missing `height` = flat, missing `color` = white, missing `normal` =
  finite-differenced — so partial builders still work. Returning a **shorter**
  array than `dirs.size()` leaves the remaining texels at the default.
- The returned `height` is a fraction; the builder's `height_scale` (inherited
  from `CesBuilder`) multiplies the geometry displacement.

## Which path should I use?

| | GPU (`.glsl`) | CPU (GDScript) | CPU async (GDScript) |
|---|---|---|---|
| You define | `terrain_height` / `terrain_color` | `height` / `color` | `_bake_requested` |
| Language | GLSL | GDScript | GDScript |
| Runs on | thousands of GPU threads | main thread, one batched call per chunk | wherever you put it |
| Best for | procedural terrain | image/data-driven, CPU logic | slow bakes, streaming, network tiles |
| Blocks the frame | no | yes, while it bakes | no |
| tile_res | full (256+) | moderate | full |

See [custom_terrain_gpu.md](custom_terrain_gpu.md) for the GPU path, or the
[builders overview](builders.md).
