extends Camera3D
## Simple orbit camera that moves around a target to test adaptive LOD textures.
## Logs LOD-related info periodically.

@export var target: Node3D
@export var orbit_speed: float = 0.3
@export var zoom_speed: float = 2.0
@export var min_distance: float = 1.2
@export var max_distance: float = 10.0
@export var log_interval: float = 1.0
@export var auto_orbit: bool = true ## Orbit automatically on start (toggle with Space)
@export var auto_zoom_in: bool = true ## Also zoom in slowly during auto-orbit

var _distance: float = 5.0
var _angle_h: float = 0.0
var _angle_v: float = 0.0
var _log_timer: float = 0.0
var _auto_orbiting: bool = true

func _ready() -> void:
	_distance = global_position.distance_to(Vector3.ZERO) if target == null else global_position.distance_to(target.global_position)
	_auto_orbiting = auto_orbit
	print("[LOD_TEST] Adaptive texture LOD test scene started")
	print("[LOD_TEST] Controls: WASD=orbit, QE=zoom, Space=toggle auto-orbit")
	if _auto_orbiting:
		print("[LOD_TEST] Auto-orbit enabled")

func _process(delta: float) -> void:
	# Keyboard orbit
	if Input.is_key_pressed(KEY_A):
		_angle_h -= orbit_speed * delta
	if Input.is_key_pressed(KEY_D):
		_angle_h += orbit_speed * delta
	if Input.is_key_pressed(KEY_W):
		_angle_v = clampf(_angle_v + orbit_speed * delta, -PI * 0.45, PI * 0.45)
	if Input.is_key_pressed(KEY_S):
		_angle_v = clampf(_angle_v - orbit_speed * delta, -PI * 0.45, PI * 0.45)
	if Input.is_key_pressed(KEY_Q):
		_distance = clampf(_distance - zoom_speed * delta, min_distance, max_distance)
	if Input.is_key_pressed(KEY_E):
		_distance = clampf(_distance + zoom_speed * delta, min_distance, max_distance)

	# Auto-orbit mode (toggle with spacebar)
	if _auto_orbiting:
		_angle_h += 0.5 * delta
		if auto_zoom_in:
			_distance = clampf(_distance - 0.3 * delta, min_distance, max_distance)

	var target_pos := Vector3.ZERO
	if target != null:
		target_pos = target.global_position

	# Compute orbit position
	var x := _distance * cos(_angle_v) * sin(_angle_h)
	var y := _distance * sin(_angle_v)
	var z := _distance * cos(_angle_v) * cos(_angle_h)

	global_position = target_pos + Vector3(x, y, z)
	look_at(target_pos, Vector3.UP)

	# Periodic logging
	_log_timer += delta
	if _log_timer >= log_interval:
		_log_timer = 0.0
		var dist := global_position.distance_to(target_pos)
		var dir := (target_pos - global_position).normalized()
		print("[LOD_TEST] dist=%.2f angle_h=%.2f angle_v=%.2f dir=(%.2f,%.2f,%.2f)" % [
			dist, _angle_h, _angle_v, dir.x, dir.y, dir.z
		])

func _input(event: InputEvent) -> void:
	if event is InputEventKey:
		var key := event as InputEventKey
		if key.pressed and key.keycode == KEY_SPACE:
			_auto_orbiting = not _auto_orbiting
			print("[LOD_TEST] Auto-orbit %s" % ("ON" if _auto_orbiting else "OFF"))
	if event is InputEventMouseButton:
		var mb := event as InputEventMouseButton
		if mb.pressed:
			if mb.button_index == MOUSE_BUTTON_WHEEL_UP:
				_distance = clampf(_distance - 0.5, min_distance, max_distance)
			elif mb.button_index == MOUSE_BUTTON_WHEEL_DOWN:
				_distance = clampf(_distance + 0.5, min_distance, max_distance)
