pub mod height_shader_terrain;
pub mod sphere_terrain;

use crate::state::CesState;
use crate::texture_gen::TerrainParams;
use godot::classes::RenderingDevice;
use godot::obj::Gd;

/// Trait for terrain transformation layers. Mirrors C# `CesLayer` abstract class.
pub trait CesLayer: Send {
    /// Initialize the layer with state and radius (called on the first layer).
    fn set_state(&mut self, state: &CesState, radius: f32);

    /// Initialize from a previous layer (called on subsequent layers).
    fn set_state_from_layer(&mut self, other: &dyn CesLayer);

    /// Apply this layer's vertex position transformations via compute shader.
    fn update_pos(&self, rd: &mut Gd<RenderingDevice>, state: &CesState);

    /// Get the radius used by this layer.
    fn radius(&self) -> f32;

    /// Initialize the compute pipeline for this layer. Called once when state is created.
    fn init_pipeline(&mut self, rd: &mut Gd<RenderingDevice>);

    /// Dispose GPU resources directly so `Drop` won't touch an already-freed RD.
    /// Must be called before the owning RenderingDevice is freed.
    fn dispose_direct(&mut self, _rd: &mut Gd<RenderingDevice>) {}

    /// Set runtime terrain params. Only meaningful for terrain height layers.
    fn set_terrain_params(&mut self, _params: &TerrainParams) {}
}
