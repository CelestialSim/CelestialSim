@tool
class_name CesGltfChildPick
extends Resource
## One selectable child inside a `CesMeshSourceGltf`. The glTF importer
## produces a `PackedScene` whose root has N `MeshInstance3D` children;
## each child becomes one `CesGltfChildPick` with `enabled` controlling
## whether it participates in scatter.

@export var node_name: StringName:
	set(v):
		node_name = v
		emit_changed()

@export var enabled: bool = true:
	set(v):
		enabled = v
		emit_changed()
