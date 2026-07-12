extends Camera3D
## Slowly orbits the origin so the Celestial's adaptive plane can follow.

# The adaptive demo planet has radius 10, so orbit well outside it.
@export var radius: float = 22.0
@export var speed: float = 0.4 # radians / second
@export var height: float = 2.0

var _angle := 0.0

func _process(delta: float) -> void:
	_angle += delta * speed
	global_position = Vector3(sin(_angle) * radius, height, cos(_angle) * radius)
	look_at(Vector3.ZERO, Vector3.UP)
