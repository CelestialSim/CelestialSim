extends Node3D
## Oscillates the planet's debug_snapshot_angle_offset so the snapshot patch
## visibly drifts while the camera stays still. Useful for verifying rendering
## correctness without async lag contributing to the visual.

@export var planet_path: NodePath = "Planet"
@export var oscillation_speed: float = 0.3 # radians per second
@export var oscillation_amplitude: float = 0.5 # radians

var _time: float = 0.0

func _process(delta: float) -> void:
	_time += delta
	var planet = get_node(planet_path)
	if planet:
		var offset = sin(_time * oscillation_speed) * oscillation_amplitude
		planet.set("debug_snapshot_angle_offset", offset)
