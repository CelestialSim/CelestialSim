use godot::builtin::Vector3;
use godot::classes::RenderingDevice;
use godot::obj::Gd;
use godot::prelude::Rid;

use crate::buffer_info::BufferInfo;
use crate::compute_utils;

/// GPU-interop triangle with 4 int fields (a, b, c, w). Mirrors the C# `Triangle` struct.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Triangle {
    pub a: i32,
    pub b: i32,
    pub c: i32,
    pub w: i32,
}

/// Holds 17 GPU buffers and metadata for the LOD subdivision algorithm.
/// Mirrors the C# `CesState` class.
pub struct CesState {
    pub n_tris: u32,
    pub n_verts: u32,
    pub n_deactivated_tris: u32,
    pub start_idx: u32,

    // 17 buffer fields
    pub t_abc: BufferInfo,
    pub t_a_t: BufferInfo,
    pub t_b_t: BufferInfo,
    pub t_c_t: BufferInfo,
    pub t_center_t: BufferInfo,
    pub t_parent: BufferInfo,
    pub t_divided: BufferInfo,
    pub t_deactivated: BufferInfo,
    pub t_ico_idx: BufferInfo,
    pub t_lv: BufferInfo,
    pub t_neight_ab: BufferInfo,
    pub t_neight_bc: BufferInfo,
    pub t_neight_ca: BufferInfo,
    pub t_to_divide_mask: BufferInfo,
    pub t_to_merge_mask: BufferInfo,
    pub v_pos: BufferInfo,
    pub v_update_mask: BufferInfo,

    // Reusable uniform buffers for n_tris and n_verts
    pub u_n_tris: BufferInfo,
    pub u_n_verts: BufferInfo,
}

impl CesState {
    /// Returns the RIDs of all 17 GPU buffers.
    pub fn all_buffers(&self) -> Vec<Rid> {
        vec![
            self.t_abc.rid,
            self.t_a_t.rid,
            self.t_b_t.rid,
            self.t_c_t.rid,
            self.t_center_t.rid,
            self.t_parent.rid,
            self.t_divided.rid,
            self.t_deactivated.rid,
            self.t_ico_idx.rid,
            self.t_lv.rid,
            self.t_neight_ab.rid,
            self.t_neight_bc.rid,
            self.t_neight_ca.rid,
            self.t_to_divide_mask.rid,
            self.t_to_merge_mask.rid,
            self.v_pos.rid,
            self.v_update_mask.rid,
            self.u_n_tris.rid,
            self.u_n_verts.rid,
        ]
    }

    /// Frees all 17 GPU buffers. Must be called on the rendering thread
    /// (or wrapped with CallOnRenderThread in Phase 7/8).
    pub fn dispose(&self, rd: &mut Gd<RenderingDevice>) {
        for rid in self.all_buffers() {
            compute_utils::free_rid_on_render_thread(rd, rid);
        }
    }

    /// Frees all GPU buffers directly without deferred dispatch.
    /// Use when the RenderingDevice will be freed immediately after.
    pub fn dispose_direct(&self, rd: &mut Gd<RenderingDevice>) {
        for rid in self.all_buffers() {
            if rid.is_valid() {
                rd.free_rid(rid);
            }
        }
    }

    /// Updates the u_n_tris uniform buffer to match current n_tris value.
    pub fn sync_n_tris_buffer(&self, rd: &mut Gd<RenderingDevice>) {
        compute_utils::update_uniform_buffer(rd, &self.u_n_tris, &self.n_tris);
    }

    /// Updates the u_n_verts uniform buffer to match current n_verts value.
    pub fn sync_n_verts_buffer(&self, rd: &mut Gd<RenderingDevice>) {
        compute_utils::update_uniform_buffer(rd, &self.u_n_verts, &self.n_verts);
    }

    /// Reads the divide mask buffer back to CPU.
    pub fn get_t_to_divide_mask(&self, rd: &mut Gd<RenderingDevice>) -> Vec<i32> {
        compute_utils::convert_buffer_to_vec(rd, &self.t_to_divide_mask)
    }

    /// Reads the merge mask buffer back to CPU.
    pub fn get_t_to_merge_mask(&self, rd: &mut Gd<RenderingDevice>) -> Vec<u32> {
        compute_utils::convert_buffer_to_vec(rd, &self.t_to_merge_mask)
    }

    /// Reads vertex positions as Vec<Vector3> (discards w component).
    pub fn get_pos(&self, rd: &mut Gd<RenderingDevice>) -> Vec<Vector3> {
        compute_utils::convert_v4_buffer_to_vec3(rd, &self.v_pos)
    }

    /// Reads the t_abc buffer as Vec<Triangle>.
    pub fn get_t_abc(&self, rd: &mut Gd<RenderingDevice>) -> Vec<Triangle> {
        compute_utils::convert_buffer_to_vec(rd, &self.t_abc)
    }

    /// Reads the divided mask buffer back to CPU.
    pub fn get_divided_mask(&self, rd: &mut Gd<RenderingDevice>) -> Vec<i32> {
        compute_utils::convert_buffer_to_vec(rd, &self.t_divided)
    }

    /// Reads the deactivated mask buffer back to CPU.
    pub fn get_t_deactivated_mask(&self, rd: &mut Gd<RenderingDevice>) -> Vec<i32> {
        compute_utils::convert_buffer_to_vec(rd, &self.t_deactivated)
    }

    /// Reads the level buffer back to CPU.
    pub fn get_level(&self, rd: &mut Gd<RenderingDevice>) -> Vec<i32> {
        compute_utils::convert_buffer_to_vec(rd, &self.t_lv)
    }

    /// Reads vertex update mask and returns indices where value == 1.
    pub fn convert_v_update_mask_to_idx(&self, rd: &mut Gd<RenderingDevice>) -> Vec<i32> {
        let mask: Vec<i32> = compute_utils::convert_buffer_to_vec(rd, &self.v_update_mask);
        mask.iter()
            .enumerate()
            .filter(|(_, &v)| v == 1)
            .map(|(i, _)| i as i32)
            .collect()
    }

    /// Reads t_abc as an Nx3 array (discarding the w column).
    pub fn get_abc_unoptimized(&self, rd: &mut Gd<RenderingDevice>) -> Vec<[i32; 3]> {
        let flat: Vec<i32> = compute_utils::convert_buffer_to_vec(rd, &self.t_abc);
        let n = flat.len() / 4;
        (0..n)
            .map(|i| [flat[i * 4], flat[i * 4 + 1], flat[i * 4 + 2]])
            .collect()
    }

    /// Reads t_abc as Vec<Triangle> (same as get_t_abc, named for C# compat).
    pub fn get_abc_w(&self, rd: &mut Gd<RenderingDevice>) -> Vec<Triangle> {
        self.get_t_abc(rd)
    }

    /// Computes center points of all triangles.
    pub fn get_center_points(&self, rd: &mut Gd<RenderingDevice>) -> Vec<Vector3> {
        let pos = self.get_pos(rd);
        let abc = self.get_abc_unoptimized(rd);
        abc.iter()
            .map(|tri| {
                let a = pos[tri[0] as usize];
                let b = pos[tri[1] as usize];
                let c = pos[tri[2] as usize];
                (a + b + c) / 3.0
            })
            .collect()
    }
}
