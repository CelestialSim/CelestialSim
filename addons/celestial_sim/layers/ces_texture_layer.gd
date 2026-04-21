@tool
class_name CesTextureLayer
extends CesTextureLayerResource
## Generates cubemap noise texture and applies the planet_texture shader.

@export var resolution: int = 512:
	set(v):
		resolution = v
		emit_changed()
@export_range(0, 4) var max_snapshot_levels: int = 4:
	set(v):
		max_snapshot_levels = v
		emit_changed()
@export var show_snapshot_borders: bool = true:
	set(v):
		show_snapshot_borders = v
		emit_changed()
@export var snapshot_color_shader: String = "res://addons/celestial_sim/shaders/SnapshotPatchSimpleNoise.slang":
	set(v):
		snapshot_color_shader = v
		emit_changed()
var shader: Shader

func _init() -> void:
	shader = preload("res://addons/celestial_sim/shaders/planet_texture.gdshader")
