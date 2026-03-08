use godot::classes::RenderingDevice;
use godot::obj::Gd;
use crate::buffer_info::BufferInfo;
use crate::compute_utils;
use crate::state::CesState;
use super::CesLayer;

const SHADER_PATH: &str = "res://addons/celestial_sim_rust/shaders/SphereTerrain.slang";

pub struct CesSphereTerrain {
    radius: f32,
}

impl CesSphereTerrain {
    pub fn new() -> Self {
        CesSphereTerrain { radius: 1.0 }
    }
}

impl CesLayer for CesSphereTerrain {
    fn set_state(&mut self, _state: &CesState, radius: f32) {
        self.radius = radius;
    }

    fn set_state_from_layer(&mut self, other: &dyn CesLayer) {
        self.radius = other.radius();
    }

    fn update_pos(&self, rd: &mut Gd<RenderingDevice>, state: &CesState) {
        let radius_buf = compute_utils::create_uniform_buffer(rd, &self.radius);
        let n_verts_buf = compute_utils::create_uniform_buffer(rd, &state.n_verts);

        let buffers: Vec<&BufferInfo> = vec![
            &state.v_pos,         // 0
            &state.v_update_mask, // 1
            &radius_buf,          // 2
            &n_verts_buf,         // 3
        ];

        compute_utils::dispatch_shader(rd, SHADER_PATH, &buffers, state.n_verts);

        compute_utils::free_rid_on_render_thread(rd, radius_buf.rid);
        compute_utils::free_rid_on_render_thread(rd, n_verts_buf.rid);
    }

    fn radius(&self) -> f32 {
        self.radius
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sphere_terrain_default_radius() {
        let st = CesSphereTerrain::new();
        assert_eq!(st.radius(), 1.0);
    }

    #[test]
    fn test_sphere_terrain_set_radius() {
        let mut st = CesSphereTerrain::new();
        st.radius = 42.5;
        assert_eq!(st.radius(), 42.5);
    }
}
