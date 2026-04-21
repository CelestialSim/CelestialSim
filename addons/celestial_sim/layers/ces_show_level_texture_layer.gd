@tool
class_name CesShowLevelTextureLayer
extends CesTextureLayerResource
## Applies the show_level shader to visualize LOD levels. No texture computation.

var shader: Shader

func _init() -> void:
	shader = preload("res://addons/celestial_sim/shaders/show_level.gdshader")
