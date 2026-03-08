pub mod sphere_terrain;

use godot::classes::RenderingDevice;
use crate::state::CesState;

/// Trait for terrain transformation layers. Mirrors C# `CesLayer` abstract class.
pub trait CesLayer {
    /// Initialize the layer with state and radius (called on the first layer).
    fn set_state(&mut self, state: &CesState, radius: f32);

    /// Initialize from a previous layer (called on subsequent layers).
    fn set_state_from_layer(&mut self, other: &dyn CesLayer);

    /// Apply this layer's vertex position transformations via compute shader.
    fn update_pos(&self, rd: &mut RenderingDevice, state: &CesState);

    /// Get the radius used by this layer.
    fn radius(&self) -> f32;
}
