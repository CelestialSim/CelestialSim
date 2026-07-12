extends CanvasLayer
## Minimal read-only stats HUD for the GPU-noise example scene. Unlike
## quadtree_chunks.gd this does NOT drive the camera (fly_camera.gd owns it) —
## it only prints selection cost, residency, triangle count and render time.

@export var planet_path: NodePath
@onready var qt: Node3D = get_node_or_null(planet_path)
@onready var label: Label = $Label

var _vp_rid: RID
var _accum := 0.0

func _ready() -> void:
	label.position = Vector2(12, 10)
	label.add_theme_color_override("font_color", Color.WHITE)
	label.add_theme_constant_override("outline_size", 6)
	label.add_theme_color_override("font_outline_color", Color(0, 0, 0, 0.8))
	label.add_theme_font_size_override("font_size", 16)
	label.mouse_filter = Control.MOUSE_FILTER_IGNORE
	_vp_rid = get_viewport().get_viewport_rid()
	RenderingServer.viewport_set_measure_render_time(_vp_rid, true)

func _process(d: float) -> void:
	_accum += d
	if _accum < 0.25 or qt == null:
		return
	_accum = 0.0
	var fps := Engine.get_frames_per_second()
	var render_gpu := RenderingServer.viewport_get_measured_render_time_gpu(_vp_rid)
	label.text = "\n".join([
		"FPS %d  (%.2f ms)" % [fps, 1000.0 / maxf(fps, 1.0)],
		"select %.0f us   resident %d/%d" % [
			qt.select_ms() * 1000.0, qt.resident_count(), qt.effective_budget()],
		"tris %s   realize %d" % [_fmt_count(qt.triangle_count()), qt.realize_count()],
		"render gpu %.2f ms" % render_gpu,
	])

func _fmt_count(n: int) -> String:
	if n >= 1000000:
		return "%.2fM" % (n / 1000000.0)
	if n >= 1000:
		return "%.1fk" % (n / 1000.0)
	return str(n)
