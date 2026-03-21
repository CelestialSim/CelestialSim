use godot::classes::RenderingDevice;
use godot::obj::Gd;

use crate::buffer_info::BufferInfo;
use crate::compute_utils::ComputePipeline;
use crate::state::CesState;

const SHADER_PATH: &str = "res://addons/celestial_sim_rust/shaders/UpdateNeighbors.slang";

pub struct UpdateNeighborsShader {
    pipeline: ComputePipeline,
}

impl UpdateNeighborsShader {
    pub fn new(rd: &mut Gd<RenderingDevice>) -> Self {
        Self {
            pipeline: ComputePipeline::new(rd, SHADER_PATH),
        }
    }

    pub fn dispose_direct(&mut self, rd: &mut Gd<RenderingDevice>) {
        self.pipeline.dispose_direct(rd);
    }

    /// Dispatches the UpdateNeighbors shader. Mirrors C# `CesUpdateNeighbors.UpdateNeighbors()`.
    pub fn update_neighbors(&self, rd: &mut Gd<RenderingDevice>, state: &CesState) {
        let buffers: Vec<&BufferInfo> = vec![
            &state.v_pos,       // 0
            &state.t_abc,       // 1
            &state.t_neight_ab, // 2
            &state.t_neight_bc, // 3
            &state.t_neight_ca, // 4
            &state.t_parent,    // 5
            &state.t_divided,   // 6
            &state.t_lv,        // 7
            &state.t_a_t,       // 8
            &state.t_b_t,       // 9
            &state.t_c_t,       // 10
            &state.t_center_t,  // 11
            &state.u_n_tris,    // 12
        ];

        self.pipeline.dispatch(rd, &buffers, state.n_tris);
    }
}
