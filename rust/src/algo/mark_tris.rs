use godot::builtin::Vector3;
use godot::classes::RenderingDevice;
use godot::obj::Gd;

use crate::buffer_info::BufferInfo;
use crate::compute_utils;
use crate::compute_utils::ComputePipeline;
use crate::state::CesState;

const SHADER_PATH: &str = "res://addons/celestial_sim_rust/shaders/MarkTrisToDivide.slang";

pub struct MarkTrisShader {
    pipeline: ComputePipeline,
}

impl MarkTrisShader {
    pub fn new(rd: &mut Gd<RenderingDevice>) -> Self {
        Self {
            pipeline: ComputePipeline::new(rd, SHADER_PATH),
        }
    }

    /// Dispatches the MarkTrisToDivide shader to flag triangles whose screen-space
    /// area exceeds `max_tri_size`. Mirrors C# `CesMarkTrisToDivide.FlagLargeTrisToDivide`.
    pub fn flag_large_tris_to_divide(
        &self,
        rd: &mut Gd<RenderingDevice>,
        state: &CesState,
        camera_pos: Vector3,
        max_divs: u32,
        radius: f32,
        max_tri_size: f32,
    ) {
        // Temporary storage buffer for per-triangle sizes
        let tris_size =
            compute_utils::create_storage_buffer(rd, &vec![0.0f32; state.n_tris as usize]);

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
            &tris_size,              // 9
            &state.t_neight_ab,      // 10
            &state.t_neight_bc,      // 11
            &state.t_neight_ca,      // 12
            &state.t_deactivated,    // 13
            &state.t_to_merge_mask,  // 14
            &state.t_parent,         // 15
        ];

        self.pipeline.dispatch(rd, &buffers, state.n_tris);

        // Free temporary buffers
        compute_utils::free_rid_on_render_thread(rd, tris_size.rid);
        compute_utils::free_rid_on_render_thread(rd, camera_pos_buf.rid);
        compute_utils::free_rid_on_render_thread(rd, max_divs_buf.rid);
        compute_utils::free_rid_on_render_thread(rd, radius_buf.rid);
        compute_utils::free_rid_on_render_thread(rd, max_tri_size_buf.rid);
    }
}
