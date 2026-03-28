@tool
class_name CesTextureLayer
extends CesTextureLayerResource
## Generates cubemap noise texture and applies the planet_texture shader.

@export var resolution: int = 512
var shader: Shader

func _init() -> void:
	shader = preload("res://addons/celestial_sim/shaders/planet_texture.gdshader")
