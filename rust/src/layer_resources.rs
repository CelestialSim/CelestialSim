use godot::prelude::*;

/// Resource for height/terrain layers (e.g. sphere normalization).
/// Use `Array[CesHeightLayerResource]` on the node for typed "+" button.
#[derive(GodotClass)]
#[class(tool, init, base = Resource)]
pub struct CesHeightLayerResource {
    base: Base<Resource>,

    #[export]
    #[init(val = true)]
    pub enabled: bool,

    #[export]
    #[init(val = GString::new())]
    pub shader_path: GString,
}

/// Base resource for texture layers. GDScript subclasses
/// (CesTextureLayer, CesShowLevelTextureLayer) provide the
/// typed "+" dropdown in the editor.
/// Shader is set by subclasses in _init(), not user-editable.
#[derive(GodotClass)]
#[class(tool, init, base = Resource)]
pub struct CesTextureLayerResource {
    base: Base<Resource>,

    #[export]
    #[init(val = true)]
    pub enabled: bool,
}
