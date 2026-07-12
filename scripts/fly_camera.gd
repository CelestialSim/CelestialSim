extends Camera3D
## Simple WASD + mouse fly camera. Hold the right mouse button to look; WASD to
## move, E/Q up/down, Shift to boost. Plain yaw/pitch (no planet-relative axes).

@export var speed: float = 12.0
@export var look_sensitivity: float = 0.003
## When > 0, fly speed scales with altitude above a sphere of this radius centred
## on the origin (orbit hops stay fast, street-level approaches stay slow).
## `speed` is the ceiling.
@export var planet_radius: float = 0.0
## Speed floor for the altitude-adaptive mode (world units / s).
@export var min_speed: float = 0.02

var _yaw := 0.0
var _pitch := 0.0
var _looking := false

func _ready() -> void:
	_yaw = rotation.y
	_pitch = rotation.x

func _unhandled_input(event: InputEvent) -> void:
	if event is InputEventMouseButton and event.button_index == MOUSE_BUTTON_RIGHT:
		_looking = event.pressed
		Input.mouse_mode = Input.MOUSE_MODE_CAPTURED if _looking else Input.MOUSE_MODE_VISIBLE
	elif event is InputEventMouseMotion and _looking:
		_yaw -= event.relative.x * look_sensitivity
		_pitch = clamp(_pitch - event.relative.y * look_sensitivity, -1.5, 1.5)
		rotation = Vector3(_pitch, _yaw, 0.0)

func _process(delta: float) -> void:
	var dir := Vector3.ZERO
	if Input.is_key_pressed(KEY_W): dir -= transform.basis.z
	if Input.is_key_pressed(KEY_S): dir += transform.basis.z
	if Input.is_key_pressed(KEY_A): dir -= transform.basis.x
	if Input.is_key_pressed(KEY_D): dir += transform.basis.x
	if Input.is_key_pressed(KEY_E): dir += Vector3.UP
	if Input.is_key_pressed(KEY_Q): dir -= Vector3.UP
	if dir != Vector3.ZERO:
		var boost := 3.0 if Input.is_key_pressed(KEY_SHIFT) else 1.0
		var eff := speed
		if planet_radius > 0.0:
			var altitude := global_position.length() - planet_radius
			eff = clamp(altitude, min_speed, speed)
		global_position += dir.normalized() * eff * boost * delta


## Teleport for UI buttons: place the camera at `pos` aimed at `target`, and
## resync the internal yaw/pitch so the next mouse-look doesn't snap back.
func teleport(pos: Vector3, target: Vector3) -> void:
	look_at_from_position(pos, target, Vector3.UP)
	rotation.z = 0.0
	_yaw = rotation.y
	_pitch = rotation.x
