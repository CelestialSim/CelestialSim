extends Node3D
# Example driver for Celestial: freefly camera + top-left HUD
# showing selection cost, cache residency, last realize batch, graph executes,
# and the viewport CPU/GPU render time.
#
# Scatter (CEL-73): the scene ships named layers ("Grass" = the committed
# grass_blade.tres, "Oak branches"/"Oak leaves" = assets/trees/*.res extracted
# from the eztree glb). A layer with no mesh renders nothing; the `enabled`
# checkbox toggles a layer live.
# Keys:  V  toggle VSync    Q  cycle terrain quality    TAB  free/capture mouse
# The same VSync/Quality are also the clickable top-left HUD buttons (free the
# mouse with TAB to click them).
# Touch (Android APK): a VirtualJoystick in the bottom-left moves; drag the rest
# of the screen to look; tap the top HUD buttons for VSync/Quality.

@onready var qt: Node3D = $Celestial
@onready var cam: Camera3D = $Camera3D
@onready var label: Label = $Hud/Label

# Terrain quality presets → screen_error (lower = finer/heavier).
const QUALITY_PRESETS := [["Ultra", 0.02], ["High", 0.035], ["Medium", 0.05], ["Low", 0.1]]

var speed := 400.0
var yaw := 0.0
var pitch := 0.0
var _touch := false
var _vp_rid: RID
var _accum := 0.0
var _vsync_btn: Button
var _quality_btn: Button
var _quality_idx := 0

func _ready() -> void:
	# On touch devices (the Android APK) there is no mouse to capture; instead
	# spawn Godot 4.7's built-in VirtualJoystick for movement + drag-to-look.
	# is_touchscreen_available() is unreliable on Android, so also treat any
	# mobile export as touch.
	_touch = DisplayServer.is_touchscreen_available() or OS.has_feature("mobile")
	if not _touch:
		Input.set_mouse_mode(Input.MOUSE_MODE_CAPTURED)
	cam.global_position = Vector3(0, 0, 2500)
	label.position = Vector2(12, 44)
	label.add_theme_color_override("font_color", Color.WHITE)
	label.add_theme_constant_override("outline_size", 6)
	label.add_theme_color_override("font_outline_color", Color(0, 0, 0, 0.8))
	label.add_theme_font_size_override("font_size", 16)
	label.mouse_filter = Control.MOUSE_FILTER_IGNORE
	_vp_rid = get_viewport().get_viewport_rid()
	RenderingServer.viewport_set_measure_render_time(_vp_rid, true)
	_build_buttons()
	if _touch:
		_build_touch_joystick()

# Godot 4.7 VirtualJoystick: renders with default theme styleboxes (no art
# needed) and drives named input actions, so the same forward/backward/left/
# right actions used by WASD also move the camera from touch. Fixed widget in the
# bottom-left, always visible; drag elsewhere on screen looks (_unhandled_input).
func _build_touch_joystick() -> void:
	var vj := VirtualJoystick.new()
	vj.action_up = "forward"
	vj.action_down = "backward"
	vj.action_left = "left"
	vj.action_right = "right"
	vj.joystick_mode = VirtualJoystick.JOYSTICK_FIXED
	vj.visibility_mode = VirtualJoystick.VISIBILITY_ALWAYS  # visible on launch
	# A fixed ~280px square anchored to the bottom-left corner, clear of the HUD.
	vj.set_anchors_preset(Control.PRESET_BOTTOM_LEFT)
	vj.offset_left = 60
	vj.offset_top = -340
	vj.offset_right = 340
	vj.offset_bottom = -60
	$Hud.add_child(vj)

func _build_buttons() -> void:
	var bar := HBoxContainer.new()
	bar.position = Vector2(12, 8)
	bar.add_theme_constant_override("separation", 8)
	$Hud.add_child(bar)

	_vsync_btn = Button.new()
	_vsync_btn.focus_mode = Control.FOCUS_NONE  # don't steal WASD focus
	_vsync_btn.pressed.connect(_toggle_vsync)
	bar.add_child(_vsync_btn)
	_update_vsync_label()

	# Start the quality preset at whatever screen_error the scene shipped with.
	_quality_idx = _nearest_quality(qt.screen_error)
	_quality_btn = Button.new()
	_quality_btn.focus_mode = Control.FOCUS_NONE
	_quality_btn.pressed.connect(_cycle_quality)
	bar.add_child(_quality_btn)
	_update_quality_label()

func _nearest_quality(se: float) -> int:
	var best := 0
	var best_d := INF
	for i in QUALITY_PRESETS.size():
		var d: float = absf(QUALITY_PRESETS[i][1] - se)
		if d < best_d:
			best_d = d
			best = i
	return best

func _toggle_vsync() -> void:
	var on := DisplayServer.window_get_vsync_mode() != DisplayServer.VSYNC_DISABLED
	DisplayServer.window_set_vsync_mode(
		DisplayServer.VSYNC_DISABLED if on else DisplayServer.VSYNC_ENABLED)
	_update_vsync_label()

func _update_vsync_label() -> void:
	var on := DisplayServer.window_get_vsync_mode() != DisplayServer.VSYNC_DISABLED
	_vsync_btn.text = "VSync: %s  (press V)" % ("On" if on else "Off")

func _cycle_quality() -> void:
	_quality_idx = (_quality_idx + 1) % QUALITY_PRESETS.size()
	qt.screen_error = QUALITY_PRESETS[_quality_idx][1]
	_update_quality_label()

func _update_quality_label() -> void:
	_quality_btn.text = "Quality: %s  (press Q)" % QUALITY_PRESETS[_quality_idx][0]

func _unhandled_input(e: InputEvent) -> void:
	if e is InputEventMouseMotion and Input.mouse_mode == Input.MOUSE_MODE_CAPTURED:
		_look(e.relative)
	elif e is InputEventScreenDrag:
		# Touch look: drags on the joystick are consumed by it, so anything
		# reaching here is a look drag (typically the right side of the screen).
		_look(e.relative)
	elif e is InputEventKey and e.pressed:
		match e.keycode:
			KEY_ESCAPE:
				get_tree().quit()
			KEY_TAB:
				# Free the cursor to click the HUD buttons, or recapture to fly.
				Input.set_mouse_mode(
					Input.MOUSE_MODE_VISIBLE if Input.mouse_mode == Input.MOUSE_MODE_CAPTURED
					else Input.MOUSE_MODE_CAPTURED)
			KEY_V:
				_toggle_vsync()
			KEY_Q:
				_cycle_quality()

func _look(rel: Vector2) -> void:
	yaw -= rel.x * 0.003
	pitch = clamp(pitch - rel.y * 0.003, -1.5, 1.5)
	cam.rotation = Vector3(pitch, yaw, 0)

func _process(d: float) -> void:
	# Actions forward/backward/left/right are bound to WASD in project.godot and
	# also driven analogically by the touch VirtualJoystick, so one path serves
	# both. get_vector already clamps magnitude to 1 (no diagonal speed boost).
	var mv := Input.get_vector("left", "right", "backward", "forward")
	var basis := cam.global_transform.basis
	var dir := basis.x * mv.x - basis.z * mv.y
	cam.global_position += dir * speed * d

	_accum += d
	if _accum < 0.25:
		return
	_accum = 0.0
	var fps := Engine.get_frames_per_second()
	var render_cpu := RenderingServer.viewport_get_measured_render_time_cpu(_vp_rid)
	var render_gpu := RenderingServer.viewport_get_measured_render_time_gpu(_vp_rid)
	label.text = "\n".join([
		"FPS %d  (%.2f ms)" % [fps, 1000.0 / maxf(fps, 1.0)],
		"select %.0f us   update %.0f us   resident %d/%d" % [
			qt.select_ms() * 1000.0, qt.update_ms() * 1000.0, qt.resident_count(), qt.effective_budget()],
		"tris %s   realize %d   executes %d" % [
			_fmt_count(qt.triangle_count()), qt.realize_count(), qt.executes()],
		"vram %.2f GiB  (%d slots)  ->  %s pool VRAM" % [
			qt.vram_budget_gib, qt.effective_budget(), _fmt_bytes(qt.pool_vram_bytes())],
		"chunkjob gpu (realize): %s" % qt.gpu_report(),
		"render gpu (draw): %.2f ms   cpu %.2f ms" % [render_gpu, render_cpu],
	])

func _fmt_count(n: int) -> String:
	if n >= 1000000:
		return "%.2fM" % (n / 1000000.0)
	if n >= 1000:
		return "%.1fk" % (n / 1000.0)
	return str(n)

func _fmt_bytes(b: float) -> String:
	var gib := b / (1024.0 * 1024.0 * 1024.0)
	if gib >= 1.0:
		return "%.2f GiB" % gib
	return "%.0f MiB" % (b / (1024.0 * 1024.0))
