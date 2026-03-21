use godot::classes::compositor_effect::EffectCallbackType;
use godot::classes::{CompositorEffect, ICompositorEffect};
use godot::prelude::*;

/// Compositor effect that renders CesState geometry in scene space.
#[derive(GodotClass)]
#[class(tool, base = CompositorEffect)]
pub struct CesFinalStateCompositorRust {
    base: Base<CompositorEffect>,

    #[export]
    target_path: NodePath,

    #[export]
    target_position: Vector3,

    #[export]
    radius: f32,

    #[export]
    ambient_strength: f32,

    #[export]
    sphere_color: Color,
}

#[godot_api]
impl ICompositorEffect for CesFinalStateCompositorRust {
    fn init(base: Base<CompositorEffect>) -> Self {
        let mut s = Self {
            base,
            target_path: NodePath::default(),
            target_position: Vector3::ZERO,
            radius: 1.0,
            ambient_strength: 0.05,
            sphere_color: Color::from_rgb(0.0, 0.0, 0.0),
        };
        s.base_mut()
            .set_effect_callback_type(EffectCallbackType::POST_TRANSPARENT);
        s.base_mut().set_access_resolved_color(true);
        s.base_mut().set_access_resolved_depth(true);
        s
    }

    // TODO: Implement render_callback in Phase 3
}
