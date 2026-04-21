@tool
class_name CesTerrainTextureLayer
extends CesTextureLayerResource
## Generates cubemap terrain texture using a configurable compute shader.

@export var resolution: int = 512:
	set(v):
		resolution = v
		emit_changed()
@export_range(0, 4) var max_snapshot_levels: int = 0:
	set(v):
		max_snapshot_levels = v
		emit_changed()
@export var show_snapshot_borders: bool = true:
	set(v):
		show_snapshot_borders = v
		emit_changed()
@export var snapshot_color_shader: String = "res://addons/celestial_sim/shaders/SnapshotPatchNoise.slang":
	set(v):
		snapshot_color_shader = v
		emit_changed()
@export var snapshot_normal_shader: String = "res://addons/celestial_sim/shaders/SnapshotPatchNormal.slang":
	set(v):
		snapshot_normal_shader = v
		emit_changed()
@export var compute_shader_path: String = "res://addons/celestial_sim/shaders/TerrainColor.slang":
	set(v):
		compute_shader_path = v
		emit_changed()
@export var generate_normal_map: bool = true:
	set(v):
		generate_normal_map = v
		emit_changed()

@export_group("Height Noise")
@export_range(0.1, 20.0) var height_tiles: float = 3.0:
	set(v):
		height_tiles = v
		emit_changed()
@export_range(1, 8) var height_octaves: int = 3:
	set(v):
		height_octaves = v
		emit_changed()
@export_range(0.01, 1.0) var height_amp: float = 0.25:
	set(v):
		height_amp = v
		emit_changed()
@export_range(0.01, 1.0) var height_gain: float = 0.1:
	set(v):
		height_gain = v
		emit_changed()
@export_range(1.0, 4.0) var height_lacunarity: float = 2.0:
	set(v):
		height_lacunarity = v
		emit_changed()

@export_group("Erosion")
@export_range(0.1, 20.0) var erosion_tiles: float = 4.0:
	set(v):
		erosion_tiles = v
		emit_changed()
@export_range(1, 10) var erosion_octaves: int = 5:
	set(v):
		erosion_octaves = v
		emit_changed()
@export_range(0.1, 1.0) var erosion_gain: float = 0.5:
	set(v):
		erosion_gain = v
		emit_changed()
@export_range(1.0, 4.0) var erosion_lacunarity: float = 1.8:
	set(v):
		erosion_lacunarity = v
		emit_changed()
@export_range(0.0, 10.0) var erosion_slope_strength: float = 3.0:
	set(v):
		erosion_slope_strength = v
		emit_changed()
@export_range(0.0, 10.0) var erosion_branch_strength: float = 3.0:
	set(v):
		erosion_branch_strength = v
		emit_changed()
@export_range(0.0, 0.5) var erosion_strength: float = 0.04:
	set(v):
		erosion_strength = v
		emit_changed()

@export_group("Water")
@export_range(0.0, 1.0) var water_height: float = 0.45:
	set(v):
		water_height = v
		emit_changed()
var shader: Shader

func _init() -> void:
	shader = preload("res://addons/celestial_sim/shaders/planet_texture.gdshader")
