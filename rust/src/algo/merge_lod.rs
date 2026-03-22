use godot::classes::RenderingDevice;
use godot::obj::Gd;

use crate::buffer_info::BufferInfo;
use crate::compute_utils;
use crate::compute_utils::ComputePipeline;
use crate::state::CesState;

const SHADER_PATH: &str = "res://addons/celestial_sim/shaders/MergeLOD.slang";
const COUNT_SHADER_PATH: &str = "res://addons/celestial_sim/shaders/CountNonZero.slang";

pub struct MergeShader {
    pipeline: ComputePipeline,
    count_pipeline: ComputePipeline,
    counter: BufferInfo,
}

impl MergeShader {
    pub fn new(rd: &mut Gd<RenderingDevice>) -> Self {
        Self {
            pipeline: ComputePipeline::new(rd, SHADER_PATH),
            count_pipeline: ComputePipeline::new(rd, COUNT_SHADER_PATH),
            counter: compute_utils::create_storage_buffer(rd, &[0u32]),
        }
    }

    pub fn dispose_direct(&mut self, rd: &mut Gd<RenderingDevice>) {
        self.pipeline.dispose_direct(rd);
        self.count_pipeline.dispose_direct(rd);
        if self.counter.rid.is_valid() {
            rd.free_rid(self.counter.rid);
        }
    }

    fn count_merge_candidates(&self, rd: &mut Gd<RenderingDevice>, state: &CesState) -> u32 {
        // Zero the reusable counter buffer instead of allocating a new one each call
        compute_utils::buffer_clear_on_render_thread(rd, self.counter.rid, 0, 4);
        let buffers: Vec<&BufferInfo> =
            vec![&state.t_to_merge_mask, &self.counter, &state.u_n_tris];
        self.count_pipeline
            .dispatch(rd, &buffers, (state.n_tris + 255) / 256);

        let counts: Vec<u32> = compute_utils::convert_buffer_to_vec(rd, &self.counter);
        counts.first().copied().unwrap_or(0)
    }

    /// Performs triangle merging. Mirrors C# `CesMergeLOD.MakeMerge()`.
    ///
    /// Returns the number of triangles merged (0 if nothing to merge).
    pub fn make_merge(&self, rd: &mut Gd<RenderingDevice>, state: &mut CesState) -> u32 {
        let n_tris_to_merge = self.count_merge_candidates(rd, state);
        if n_tris_to_merge == 0 {
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

        state.n_deactivated_tris += n_tris_to_merge;

        n_tris_to_merge
    }
}
