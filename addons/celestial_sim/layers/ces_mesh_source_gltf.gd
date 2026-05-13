@tool
class_name CesMeshSourceGltf
extends CesMeshSource
## Mesh source backed by an imported glTF/glb. Each `MeshInstance3D` child
## of the scene root becomes a `CesGltfChildPick`; enabling multiple picks
## causes the scatter system to spawn them as deterministic variants.

## The imported glTF/glb. On assignment, `picks` is rebuilt from the
## scene root's `MeshInstance3D` children, preserving the enabled flag
## of any pick whose name matches.
@export var gltf: PackedScene:
	set(v):
		gltf = v
		_rebuild_picks()
		emit_changed()

## One entry per `MeshInstance3D` child of `gltf`. Toggle `enabled` per
## row in the inspector to control which children participate in scatter.
@export var picks: Array[CesGltfChildPick] = []:
	set(v):
		picks = v
		emit_changed()

## When true, the X/Z translation of each child node is baked into the
## scattered transform. Default false: artist-side row-layout offsets in
## the source file don't translate scattered instances horizontally.
## Y translation, rotation, and scale are always baked.
@export var bake_xz_translation: bool = false:
	set(v):
		bake_xz_translation = v
		emit_changed()


func _rebuild_picks() -> void:
	var existing_enabled: Dictionary = {}
	for p in picks:
		if p != null:
			existing_enabled[p.node_name] = p.enabled

	var new_names: Array[StringName] = _collect_mesh_child_names()
	var new_picks: Array[CesGltfChildPick] = []
	for name in new_names:
		var pick := CesGltfChildPick.new()
		pick.node_name = name
		pick.enabled = existing_enabled.get(name, true)
		new_picks.append(pick)
	picks = new_picks


func _collect_mesh_child_names() -> Array[StringName]:
	var names: Array[StringName] = []
	if gltf == null:
		return names
	var root := gltf.instantiate()
	if root == null:
		return names
	for child in root.get_children():
		if child is MeshInstance3D:
			names.append(StringName(child.name))
	root.queue_free()
	return names
