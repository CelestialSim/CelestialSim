use godot::classes::RenderingDevice;
use godot::obj::Gd;

use crate::buffer_info::BufferInfo;
use crate::compute_utils::ComputePipeline;
use crate::state::CesState;

const SHADER_PATH: &str = "res://addons/celestial_sim/shaders/MergeLOD.slang";

pub struct MergeShader {
    pipeline: ComputePipeline,
}

impl MergeShader {
    pub fn new(rd: &mut Gd<RenderingDevice>) -> Self {
        Self {
            pipeline: ComputePipeline::new(rd, SHADER_PATH),
        }
    }

    pub fn dispose_direct(&mut self, rd: &mut Gd<RenderingDevice>) {
        self.pipeline.dispose_direct(rd);
    }

    /// Performs triangle merging. Mirrors C# `CesMergeLOD.MakeMerge()`.
    ///
    /// `n_to_merge` is an upper bound from MarkTrisToDivide counters.
    /// Returns the number of triangles merged (0 if nothing to merge).
    pub fn make_merge(
        &self,
        rd: &mut Gd<RenderingDevice>,
        state: &mut CesState,
        n_to_merge: u32,
    ) -> u32 {
        if n_to_merge == 0 {
            return 0;
        }

        let buffers: Vec<&BufferInfo> = vec![
            &state.t_abc,           // 0
            &state.t_divided,       // 1
            &state.u_n_tris,        // 2
            &state.t_neight_ab,     // 3
            &state.t_neight_bc,     // 4
            &state.t_neight_ca,     // 5
            &state.t_a_t,           // 6
            &state.t_b_t,           // 7
            &state.t_c_t,           // 8
            &state.t_center_t,      // 9
            &state.t_to_merge_mask, // 10
            &state.t_deactivated,   // 11
            &state.t_lv,            // 12
            &state.v_update_mask,   // 13
            &state.v_pos,           // 14
            &state.t_parent,        // 15
        ];

        self.pipeline.dispatch(rd, &buffers, state.n_tris);

        n_to_merge
    }
}
