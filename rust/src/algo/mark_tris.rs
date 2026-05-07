use godot::builtin::Vector3;
use godot::classes::RenderingDevice;
use godot::obj::Gd;

use crate::buffer_info::BufferInfo;
use crate::compute_utils;
use crate::compute_utils::ComputePipeline;
use crate::state::CesState;

const SHADER_PATH: &str = "res://addons/celestial_sim/shaders/MarkTrisToDivide.slang";

/// Counts produced by the MarkTrisToDivide shader via atomic counters.
pub struct MarkTrisCounters {
    pub n_to_divide: u32,
    pub n_to_merge: u32,
    pub n_deactivated: u32,
    pub n_divided: u32,
}

pub struct MarkTrisShader {
    pipeline: ComputePipeline,
    counter: BufferInfo,
}

impl MarkTrisShader {
    pub fn new(rd: &mut Gd<RenderingDevice>) -> Self {
        Self {
            pipeline: ComputePipeline::new(rd, SHADER_PATH),
            counter: compute_utils::create_storage_buffer(rd, &[0u32, 0u32, 0u32, 0u32]),
        }
    }

    pub fn dispose_direct(&mut self, rd: &mut Gd<RenderingDevice>) {
        self.pipeline.dispose_direct(rd);
        if self.counter.rid.is_valid() {
            rd.free_rid(self.counter.rid);
        }
    }

    /// Dispatches the MarkTrisToDivide shader to flag triangles whose screen-space
    /// area exceeds `max_tri_size`. Returns atomic counters for downstream decisions.
    pub fn flag_large_tris_to_divide(
        &self,
        rd: &mut Gd<RenderingDevice>,
        state: &CesState,
        camera_pos: Vector3,
        max_divs: u32,
        radius: f32,
        max_tri_size: f32,
    ) -> MarkTrisCounters {
        // Clear the 4-uint counter buffer (16 bytes)
        compute_utils::buffer_clear_on_render_thread(rd, self.counter.rid, 0, 16);

        // Uniform buffers for shader parameters
        let camera_pos_arr: [f32; 3] = [camera_pos.x, camera_pos.y, camera_pos.z];
        let camera_pos_buf = compute_utils::create_uniform_buffer(rd, &camera_pos_arr);
        let max_divs_buf = compute_utils::create_uniform_buffer(rd, &max_divs);
        let radius_buf = compute_utils::create_uniform_buffer(rd, &radius);
        let max_tri_size_buf = compute_utils::create_uniform_buffer(rd, &max_tri_size);

        let buffers: Vec<&BufferInfo> = vec![
            &state.v_pos,            // 0
            &state.t_abc,            // 1
            &state.t_lv,             // 2
            &state.t_divided,        // 3
            &state.t_to_divide_mask, // 4
            &camera_pos_buf,         // 5
            &max_divs_buf,           // 6
            &radius_buf,             // 7
            &max_tri_size_buf,       // 8
            &state.t_neight_ab,      // 9
            &state.t_neight_bc,      // 10
            &state.t_neight_ca,      // 11
            &state.t_deactivated,    // 12
            &state.t_to_merge_mask,  // 13
            &state.t_parent,         // 14
            &state.u_n_tris,         // 15
            &self.counter,           // 16
        ];

        let workgroups = (state.n_tris + 255) / 256;
        self.pipeline
            .dispatch(rd, &buffers, workgroups, "mark_tris");

        // Read back the 4-uint counter buffer
        let counts: Vec<u32> = compute_utils::convert_buffer_to_vec(rd, &self.counter);

        // Free temporary buffers
        compute_utils::free_rid_on_render_thread(rd, camera_pos_buf.rid);
        compute_utils::free_rid_on_render_thread(rd, max_divs_buf.rid);
        compute_utils::free_rid_on_render_thread(rd, radius_buf.rid);
        compute_utils::free_rid_on_render_thread(rd, max_tri_size_buf.rid);

        MarkTrisCounters {
            n_to_divide: counts[0],
            n_to_merge: counts[1],
            n_deactivated: counts[2],
            n_divided: counts[3],
        }
    }
}
