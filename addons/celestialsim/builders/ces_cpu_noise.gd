# Built-in noise baked on CPU worker threads (same knobs as CesGPUNoiseExample).
# A thin CesBuilder subclass carrying the noise knobs as its OWN @export vars;
# `_init` selects the CPU built-in route (device = CPU, builtin_shader = Terrain).
# `@tool` so the exports appear/update in the EDITOR (not just at runtime).
#
# No setters needed: the planet polls @export values each frame and reshades on
# change, so dragging a slider updates live without any emit_changed() boilerplate.
@tool
class_name CesCPUNoiseExample
extends CesBuilder

@export_range(0.1, 64.0, 0.001) var frequency: float = 2.0
@export_range(1.0, 12.0, 1.0) var octaves: float = 5.0
@export_range(0.0, 2.0, 0.001) var amp: float = 0.4
@export_range(0.0, 1.0, 0.001) var gain: float = 0.1
@export_range(1.0, 4.0, 0.001) var lacunarity: float = 2.0
@export_range(0.1, 64.0, 0.001) var ridge_tiles: float = 3.242
@export_range(1.0, 12.0, 1.0) var ridge_octaves: float = 6.0
@export_range(0.0, 1.0, 0.001) var ridge_gain: float = 0.5
@export_range(1.0, 4.0, 0.001) var ridge_lacunarity: float = 1.8
@export_range(0.0, 1.0, 0.0001) var ridge_strength: float = 0.05
@export_range(0.0, 1.0, 0.0001) var height_scale: float = 0.25
# water_height (the water level) is now a NATIVE CesBuilder property, inherited by
# every builder — do NOT re-declare it here (that would shadow the native one).

func _init() -> void:
	device = 1         # BuilderDevice.CPU
	builtin_shader = 1 # BuiltinShader.Terrain
