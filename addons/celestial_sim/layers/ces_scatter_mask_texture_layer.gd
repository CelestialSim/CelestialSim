@tool
class_name CesScatterMaskTextureLayer
extends CesTextureLayerResource
## Scatter mask preview layer — renders a B&W overlay showing where objects
## would spawn based on biome color match and the blue-noise threshold.
## No compute pass; reuses the existing planet cubemap / snapshot textures.

## Controls how aggressively the blue-noise threshold suppresses spawns.
## 1.0 = full stochastic dithering; 0.0 = accept all density > 0.
@export var noise_strength: float = 1.0:
	set(v):
		noise_strength = v
		emit_changed()

## Planet surface color that the biome matcher targets.
@export var target_color: Color = Color(0.2, 0.6, 0.2, 1.0):
	set(v):
		target_color = v
		emit_changed()

## Allowed distance in RGB space from target_color before density falls to zero.
@export var color_tolerance: float = 0.3:
	set(v):
		color_tolerance = v
		emit_changed()

var shader: Shader

func _init() -> void:
	shader = preload("res://addons/celestial_sim/shaders/scatter_mask_preview.gdshader")
