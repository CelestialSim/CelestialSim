use super::CesLayer;
use crate::buffer_info::BufferInfo;
use crate::compute_utils;
use crate::compute_utils::ComputePipeline;
use crate::state::CesState;
use crate::texture_gen::TerrainParams;
use godot::classes::RenderingDevice;
use godot::obj::Gd;

const DEFAULT_SHADER_PATH: &str = "res://addons/celestial_sim/shaders/TerrainHeight.slang";

pub struct CesHeightShaderTerrain {
    radius: f32,
    shader_path: String,
    pipeline: Option<ComputePipeline>,
    terrain_params: TerrainParams,
}

impl CesHeightShaderTerrain {
    pub fn new() -> Self {
        Self {
            radius: 1.0,
            shader_path: DEFAULT_SHADER_PATH.to_string(),
            pipeline: None,
            terrain_params: TerrainParams::default(),
        }
    }

    pub fn with_shader_path(shader_path: &str) -> Self {
        Self {
            radius: 1.0,
            shader_path: shader_path.to_string(),
            pipeline: None,
            terrain_params: TerrainParams::default(),
        }
    }
}

impl CesLayer for CesHeightShaderTerrain {
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
            .expect("CesHeightShaderTerrain pipeline not initialized; call init_pipeline first");
        let radius_buf = compute_utils::create_uniform_buffer(rd, &self.radius);
        let params_buf = compute_utils::create_uniform_buffer(rd, &self.terrain_params);

        let buffers: Vec<&BufferInfo> = vec![
            &state.v_pos,         // 0
            &state.v_update_mask, // 1
            &radius_buf,          // 2
            &state.u_n_verts,     // 3
            &params_buf,          // 4
        ];

        pipeline.dispatch(rd, &buffers, state.n_verts);

        compute_utils::free_rid_on_render_thread(rd, radius_buf.rid);
        compute_utils::free_rid_on_render_thread(rd, params_buf.rid);
    }

    fn radius(&self) -> f32 {
        self.radius
    }

    fn init_pipeline(&mut self, rd: &mut Gd<RenderingDevice>) {
        if self.pipeline.is_none() {
            self.pipeline = Some(ComputePipeline::new(rd, &self.shader_path));
        }
    }

    fn dispose_direct(&mut self, rd: &mut Gd<RenderingDevice>) {
        if let Some(ref mut pipeline) = self.pipeline {
            pipeline.dispose_direct(rd);
        }
    }

    fn set_terrain_params(&mut self, params: &TerrainParams) {
        self.terrain_params = *params;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_height_shader_terrain_default() {
        let st = CesHeightShaderTerrain::new();
        assert_eq!(st.radius(), 1.0);
    }

    #[test]
    fn test_height_shader_terrain_custom_path() {
        let st = CesHeightShaderTerrain::with_shader_path("res://custom/MyTerrain.slang");
        assert_eq!(st.shader_path, "res://custom/MyTerrain.slang");
    }

    #[test]
    fn test_height_shader_terrain_set_radius() {
        let mut st = CesHeightShaderTerrain::new();
        st.radius = 42.5;
        assert_eq!(st.radius(), 42.5);
    }
}
