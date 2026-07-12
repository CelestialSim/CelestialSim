# Built-in GPU noise terrain (the rich inline shader). A thin CesBuilder subclass
# that carries the noise knobs as its OWN @export vars; `_init` selects the
# built-in route (device = GPU, builtin_shader = Terrain). The base hides
# `shader_file` for this route; the knobs below show by default.
# `@tool` so the exports appear/update in the EDITOR (not just at runtime).
#
# No setters needed: the planet polls @export values each frame and reshades on
# change, so dragging a slider updates live without any emit_changed() boilerplate.
@tool
class_name CesGPUNoiseExample
extends CesBuilder

@export_range(0.1, 64.0, 0.001) var frequency: float = 1.4
@export_range(1.0, 12.0, 1.0) var octaves: float = 8.0
@export_range(0.0, 2.0, 0.001) var amp: float = 0.35
@export_range(0.0, 1.0, 0.001) var gain: float = 0.396
@export_range(1.0, 4.0, 0.001) var lacunarity: float = 2.0
@export_range(0.1, 64.0, 0.001) var ridge_tiles: float = 2.4
@export_range(1.0, 12.0, 1.0) var ridge_octaves: float = 5.0
@export_range(0.0, 1.0, 0.001) var ridge_gain: float = 0.5
@export_range(1.0, 4.0, 0.001) var ridge_lacunarity: float = 2.0
@export_range(0.0, 1.0, 0.0001) var ridge_strength: float = 0.0552
@export_range(0.0, 1.0, 0.0001) var height_scale: float = 0.0585
@export_range(0.00001, 0.01, 0.00001) var fd_eps: float = 0.0008
# water_height (the water level) is now a NATIVE CesBuilder property, inherited by
# every builder — do NOT re-declare it here (that would shadow the native one).

func _init() -> void:
	device = 0         # BuilderDevice.GPU
	builtin_shader = 1 # BuiltinShader.Terrain
