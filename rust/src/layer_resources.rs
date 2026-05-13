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

/// Base resource for scatter layers (GPU object instancing — grass, trees, …).
/// GDScript subclasses (CesScatterLayer) expose per-kind parameters
/// (mesh, density, biome filter) and provide the typed "+" dropdown.
#[derive(GodotClass)]
#[class(tool, init, base = Resource)]
pub struct CesScatterLayerResource {
    base: Base<Resource>,

    #[export]
    #[init(val = true)]
    pub enabled: bool,
}

/// Base resource for a mesh source consumed by `CesScatterLayer`. Concrete
/// behavior (single mesh, glTF child picker, etc.) lives in GDScript
/// subclasses, discriminated in Rust via `script.get_global_name()`.
#[derive(GodotClass)]
#[class(tool, init, base = Resource)]
pub struct CesMeshSourceResource {
    base: Base<Resource>,
}
