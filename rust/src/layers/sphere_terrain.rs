use super::CesLayer;
use crate::buffer_info::BufferInfo;
use crate::compute_utils;
use crate::compute_utils::ComputePipeline;
use crate::state::CesState;
use godot::classes::RenderingDevice;
use godot::obj::Gd;

const SHADER_PATH: &str = "res://addons/celestial_sim/shaders/SphereTerrain.slang";

pub struct CesSphereTerrain {
    radius: f32,
    pipeline: Option<ComputePipeline>,
}

impl CesSphereTerrain {
    pub fn new() -> Self {
        CesSphereTerrain {
            radius: 1.0,
            pipeline: None,
        }
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
        let pipeline = self
            .pipeline
            .as_ref()
            .expect("CesSphereTerrain pipeline not initialized; call init_pipeline first");
        let radius_buf = compute_utils::create_uniform_buffer(rd, &self.radius);

        let buffers: Vec<&BufferInfo> = vec![
            &state.v_pos,         // 0
            &state.v_update_mask, // 1
            &radius_buf,          // 2
            &state.u_n_verts,     // 3
        ];

        pipeline.dispatch(rd, &buffers, state.n_verts, "sphere_terrain");

        compute_utils::free_rid_on_render_thread(rd, radius_buf.rid);
    }

    fn radius(&self) -> f32 {
        self.radius
    }

    fn init_pipeline(&mut self, rd: &mut Gd<RenderingDevice>) {
        if self.pipeline.is_none() {
            self.pipeline = Some(ComputePipeline::new(rd, SHADER_PATH));
        }
    }

    fn dispose_direct(&mut self, rd: &mut Gd<RenderingDevice>) {
        if let Some(ref mut pipeline) = self.pipeline {
            pipeline.dispose_direct(rd);
        }
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
