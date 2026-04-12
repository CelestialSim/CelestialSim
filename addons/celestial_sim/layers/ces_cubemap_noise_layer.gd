@tool
class_name CesCubemapNoiseLayer
extends CesTextureLayerResource
## Generates cubemap noise textures for terrain displacement.

@export var resolution: int = 512:
	set(v):
		resolution = v
		emit_changed()
@export var show_debug_cube: bool = false:
	set(v):
		show_debug_cube = v
		emit_changed()
var shader: Shader

func _init() -> void:
	shader = preload("res://addons/celestial_sim/shaders/planet_texture.gdshader")
