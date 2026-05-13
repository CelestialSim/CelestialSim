@tool
class_name CesScatterLayer
extends CesScatterLayerResource
## Triangle-indexed GPU scatter via blue-noise dithering + atomic compaction.
##
## One thread per triangle in the adaptive-LOD mesh; only triangles whose LOD
## level matches `subdivision_level` are candidates. Per triangle: sample the
## terrain noise → continuous biome density, sample the 3D blue-noise volume
## → threshold, accept iff `density > threshold * noise_strength`. Survivors
## are atomically compacted into a packed buffer so every visible MultiMesh
## instance corresponds to a real spawn.

## Optional source for the scattered mesh. When set, takes precedence over
## `mesh` and `material` (which remain for backward compatibility with
## primitive meshes like CapsuleMesh/BoxMesh). For multi-mesh glTF assets,
## use `CesMeshSourceGltf` and toggle individual `picks`.
@export var mesh_source: CesMeshSource:
	set(v):
		mesh_source = v
		emit_changed()

@export var mesh: Mesh:
	set(v):
		mesh = v
		emit_changed()

@export var material: Material:
	set(v):
		material = v
		emit_changed()

## Triangles at this LOD level (and only this level) host scatter candidates.
## Higher = more triangles = more potential spawns. Trees stay anchored as the
## adaptive LOD subdivides FURTHER (children at level+1 don't replace parents),
## but disappear if a region merges back below this level.
@export_range(0, 18) var subdivision_level: int = 5:
	set(v):
		subdivision_level = v
		emit_changed()

## Multiplier applied to the blue-noise threshold. 1.0 = full dithering;
## 0.0 = no rejection (every triangle in biome accepted).
@export_range(0.0, 1.0) var noise_strength: float = 1.0:
	set(v):
		noise_strength = v
		emit_changed()

@export var seed: int = 0:
	set(v):
		seed = v
		emit_changed()

@export var placement_shader_path: String = "res://addons/celestial_sim/shaders/ScatterPlacement.slang":
	set(v):
		placement_shader_path = v
		emit_changed()

@export_group("Biome filter")

## Triangles whose terrain height is outside [height_min, height_max] are rejected.
## The biome is smoothed with a small smoothstep falloff at each edge.
@export_range(0.0, 1.0) var height_min: float = 0.45:
	set(v):
		height_min = v
		emit_changed()

@export_range(0.0, 1.0) var height_max: float = 0.55:
	set(v):
		height_max = v
		emit_changed()

@export var albedo_target: Color = Color(0.25, 0.4, 0.15):
	set(v):
		albedo_target = v
		emit_changed()

## Distance in RGB space from `albedo_target` before the biome density falls
## to zero. Set to 2.0 (≥ √3) to disable the albedo similarity check.
@export_range(0.0, 2.0) var albedo_tolerance: float = 2.0:
	set(v):
		albedo_tolerance = v
		emit_changed()
