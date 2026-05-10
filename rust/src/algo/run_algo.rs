use godot::builtin::Vector3;
use godot::classes::RenderingDevice;
use godot::obj::Gd;
use godot::prelude::godot_print;

use crate::algo::compact_buffers::CompactShaders;
use crate::algo::div_lod::DivShader;
use crate::algo::final_state::{self, FinalStateShader};
use crate::algo::mark_tris::MarkTrisShader;
use crate::algo::merge_lod::MergeShader;
use crate::algo::update_neighbors::UpdateNeighborsShader;
use crate::initial_state;
use crate::layers::CesLayer;
use crate::perf::ThreadScope;
use crate::shared_texture::SharedPositionTexture;
use crate::state::CesState;

/// Configuration for the subdivision algorithm.
pub struct RunAlgoConfig {
    pub subdivisions: u32,
    pub radius: f32,
    pub triangle_screen_size: f32,
    pub precise_normals: bool,
    pub low_poly_look: bool,
    pub show_debug_messages: bool,
    pub show_debug_lod_histogram: bool,
}

/// Orchestrates the adaptive LOD subdivision loop.
/// Mirrors C# `CesRunAlgo`.
pub struct CesRunAlgo {
    pub state: Option<CesState>,
    mark_tris_shader: Option<MarkTrisShader>,
    update_neighbors_shader: Option<UpdateNeighborsShader>,
    div_shader: Option<DivShader>,
    merge_shader: Option<MergeShader>,
    compact_shaders: Option<CompactShaders>,
    final_state_shader: Option<FinalStateShader>,
}

impl CesRunAlgo {
    pub fn new() -> Self {
        Self {
            state: None,
            mark_tris_shader: None,
            update_neighbors_shader: None,
            div_shader: None,
            merge_shader: None,
            compact_shaders: None,
            final_state_shader: None,
        }
    }

    /// Runs all layers' state initialization and position update.
    fn layers_update(
        rd: &mut Gd<RenderingDevice>,
        state: &CesState,
        layers: &mut [Box<dyn CesLayer>],
        radius: f32,
    ) {
        for i in 0..layers.len() {
            if i == 0 {
                layers[i].set_state(state, radius);
            } else {
                let (prev, current) = layers.split_at_mut(i);
                current[0].set_state_from_layer(&*prev[i - 1]);
            }
            layers[i].update_pos(rd, state);
        }
    }

    fn run_triangle_graph_updates(
        &mut self,
        rd: &mut Gd<RenderingDevice>,
        cam_local: Vector3,
        config: &RunAlgoConfig,
        layers: &mut [Box<dyn CesLayer>],
        skip_auto_division_marking: bool,
    ) {
        let _algo_scope = ThreadScope::enter("update_triangle_graph");

        if self.state.is_none() {
            self.state = Some(initial_state::create_core_state(rd));
            self.mark_tris_shader = Some(MarkTrisShader::new(rd));
            self.update_neighbors_shader = Some(UpdateNeighborsShader::new(rd));
            self.div_shader = Some(DivShader::new(rd));
            self.merge_shader = Some(MergeShader::new(rd));
            self.compact_shaders = Some(CompactShaders::new(rd));
            self.final_state_shader = Some(FinalStateShader::new(rd));
            for layer in layers.iter_mut() {
                layer.init_pipeline(rd);
            }
            Self::layers_update(rd, self.state.as_ref().unwrap(), layers, config.radius);
        }

        let state = self.state.as_mut().unwrap();

        let mut n_tris_added: u32 = 0;
        let mut n_tris_merged: u32 = 0;
        let mut first_run = true;

        while n_tris_added > 0 || n_tris_merged > 0 || first_run {
            first_run = false;

            let n_to_divide;
            let n_to_merge;
            if !skip_auto_division_marking {
                let _g = ThreadScope::enter("mark_tris");
                let mark_counts = self
                    .mark_tris_shader
                    .as_ref()
                    .unwrap()
                    .flag_large_tris_to_divide(
                        rd,
                        state,
                        cam_local,
                        config.subdivisions,
                        config.radius,
                        config.triangle_screen_size,
                    );
                n_to_divide = mark_counts.n_to_divide;
                n_to_merge = mark_counts.n_to_merge;
                state.n_divided = mark_counts.n_divided;
                state.n_deactivated_tris = mark_counts.n_deactivated;
            } else {
                // When marking is skipped (test/manual paths), count from the GPU mask
                let mask = state.get_t_to_divide_mask(rd);
                n_to_divide = mask.iter().filter(|&&x| x != 0).count() as u32;
                // No mark counts available; use n_tris as upper bound so merge is never skipped
                n_to_merge = state.n_tris;
            }

            {
                let _g = ThreadScope::enter("div_lod");
                n_tris_added = self.div_shader.as_ref().unwrap().make_div(
                    rd,
                    state,
                    config.precise_normals,
                    n_to_divide,
                );
            }
            if n_tris_added > 0 {
                state.sync_n_tris_buffer(rd);
                state.sync_n_verts_buffer(rd);
                if config.show_debug_messages {
                    godot_print!("Divided {} triangles", n_tris_added / 4);
                }
            }

            {
                let _g = ThreadScope::enter("merge_lod");
                n_tris_merged = self
                    .merge_shader
                    .as_ref()
                    .unwrap()
                    .make_merge(rd, state, n_to_merge);
            }
            if n_tris_merged > 0 && config.show_debug_messages {
                godot_print!(
                    "Merged {} triangle(s) (removed {} child triangles)",
                    n_tris_merged / 4,
                    n_tris_merged
                );
            }

            if state.n_deactivated_tris > 100_000 {
                if config.show_debug_messages {
                    godot_print!("Removing free space inside buffers");
                }
                let _g = ThreadScope::enter("compact_buffers");
                self.compact_shaders.as_ref().unwrap().compact(rd, state);
            }

            if n_tris_added > 0 || n_tris_merged > 0 {
                let _g = ThreadScope::enter("update_neighbors");
                self.update_neighbors_shader
                    .as_ref()
                    .unwrap()
                    .update_neighbors(rd, state);
            }

            {
                let _g = ThreadScope::enter("layers_update");
                Self::layers_update(rd, state, layers, config.radius);
            }
        }

        // Final compaction pass: when the LOD loop converges after a big
        // level-down (e.g. swapping from level 8 to level 6), it's typical
        // for tens of thousands of deactivated triangles to remain — below
        // the in-loop 100k threshold but enough to inflate `state.n_tris`,
        // which in turn inflates the dispatch size of mark_tris/final_state
        // every subsequent frame. Compacting once after the loop drives
        // `state.n_tris` back down to the active set so steady-state cost
        // matches the fresh-init cost.
        if state.n_deactivated_tris > 0 {
            let _g = ThreadScope::enter("compact_buffers");
            self.compact_shaders.as_ref().unwrap().compact(rd, state);
        }
    }

    /// Runs the main subdivision loop until convergence and reads the final
    /// vertex payload back to the CPU.
    /// Mirrors C# `CesRunAlgo.UpdateTriangleGraph`.
    pub fn update_triangle_graph(
        &mut self,
        rd: &mut Gd<RenderingDevice>,
        cam_local: Vector3,
        config: &RunAlgoConfig,
        layers: &mut [Box<dyn CesLayer>],
        skip_auto_division_marking: bool,
    ) -> final_state::FinalOutput {
        self.run_triangle_graph_updates(rd, cam_local, config, layers, skip_auto_division_marking);

        let final_output = {
            let _g = ThreadScope::enter("final_state");
            final_state::create_final_output(
                rd,
                self.state.as_ref().unwrap(),
                config.low_poly_look,
                self.final_state_shader.as_ref().unwrap(),
            )
        };

        final_output
    }

    /// Runs the main subdivision loop until convergence and writes the final
    /// positions straight into the shared vertex-position texture.
    pub fn update_triangle_graph_to_position_texture(
        &mut self,
        rd: &mut Gd<RenderingDevice>,
        cam_local: Vector3,
        config: &RunAlgoConfig,
        layers: &mut [Box<dyn CesLayer>],
        skip_auto_division_marking: bool,
        position_texture: &mut SharedPositionTexture,
        early_exit: bool
    ) -> final_state::FinalTextureOutput {

        self.run_triangle_graph_updates(rd, cam_local, config, layers, skip_auto_division_marking);
        
        let final_output = {
            let _g = ThreadScope::enter("final_state");
            final_state::create_final_output_to_shared_position_texture(
                rd,
                self.state.as_ref().unwrap(),
                self.final_state_shader.as_ref().unwrap(),
                position_texture,
                early_exit
            )
        };

        final_output
    }

    /// Reads the current visible mesh + level + neighbor topology back from the
    /// GPU and writes it to `path` using the [`crate::algo::dump_format`]
    /// binary layout. Called from a debug/test path; not used in production.
    /// Caller must invoke this after a frame's algo loop has converged
    /// (e.g. at the end of `run_triangle_graph_updates`).
    pub fn dump_visible_state(
        &mut self,
        rd: &mut Gd<RenderingDevice>,
        radius: f32,
        path: &std::path::Path,
    ) -> std::io::Result<()> {
        use crate::algo::dump_format::{encode, DumpHeader, DumpedTri};
        use std::io::{Error, ErrorKind};

        let state = self.state.as_ref().ok_or_else(|| {
            Error::new(
                ErrorKind::Other,
                "CesRunAlgo::dump_visible_state: state is None (algo never ran)",
            )
        })?;
        let shader = self.final_state_shader.as_ref().ok_or_else(|| {
            Error::new(
                ErrorKind::Other,
                "CesRunAlgo::dump_visible_state: final_state_shader is None",
            )
        })?;

        // Run the standard final-output pass to obtain post-stitch positions.
        let final_output = final_state::create_final_output(rd, state, false, shader);
        let n_visible_tris = (final_output.pos.len() / 3) as u32;

        // Read back the per-internal-triangle data we need for adjacency.
        let levels: Vec<i32> = state.get_level(rd);
        let divided: Vec<i32> = state.get_divided_mask(rd);
        let deactivated: Vec<i32> = state.get_t_deactivated_mask(rd);
        let neigh_ab: Vec<i32> = crate::compute_utils::convert_buffer_to_vec(rd, &state.t_neight_ab);
        let neigh_bc: Vec<i32> = crate::compute_utils::convert_buffer_to_vec(rd, &state.t_neight_bc);
        let neigh_ca: Vec<i32> = crate::compute_utils::convert_buffer_to_vec(rd, &state.t_neight_ca);

        let n_tris = state.n_tris as usize;
        // Mirror the visibility logic in ComputeVisiblePrefixAtomic: visible
        // when divided == 0 AND deactivated == 0.
        let visible_mask: Vec<i32> = (0..n_tris)
            .map(|i| {
                let d = *divided.get(i).unwrap_or(&0);
                let a = *deactivated.get(i).unwrap_or(&0);
                if d == 0 && a == 0 {
                    1
                } else {
                    0
                }
            })
            .collect();
        // Inclusive prefix sum so visible_prefix[i] - 1 is the dense visible index.
        let mut visible_prefix: Vec<i32> = Vec::with_capacity(n_tris);
        let mut acc: i32 = 0;
        for &v in &visible_mask {
            acc += v;
            visible_prefix.push(acc);
        }
        let counted_visible = *visible_prefix.last().unwrap_or(&0) as u32;
        if counted_visible != n_visible_tris {
            return Err(Error::new(
                ErrorKind::Other,
                format!(
                    "visible-count mismatch: prefix says {counted_visible}, final_output says {n_visible_tris}"
                ),
            ));
        }

        let resolve_neighbor = |n: i32| -> i32 {
            if n < 0 || (n as usize) >= n_tris {
                return -1;
            }
            if visible_mask[n as usize] == 0 {
                return -1;
            }
            visible_prefix[n as usize] - 1
        };

        let mut tris: Vec<DumpedTri> = Vec::with_capacity(n_visible_tris as usize);
        for i in 0..n_tris {
            if visible_mask[i] == 0 {
                continue;
            }
            let v = (visible_prefix[i] - 1) as usize;
            let p0 = final_output.pos[3 * v];
            let p1 = final_output.pos[3 * v + 1];
            let p2 = final_output.pos[3 * v + 2];
            tris.push(DumpedTri {
                positions: [
                    [p0.x, p0.y, p0.z],
                    [p1.x, p1.y, p1.z],
                    [p2.x, p2.y, p2.z],
                ],
                level: levels[i],
                neighbors: [
                    resolve_neighbor(neigh_ab[i]),
                    resolve_neighbor(neigh_bc[i]),
                    resolve_neighbor(neigh_ca[i]),
                ],
            });
        }

        let header = DumpHeader {
            n_visible_tris,
            radius,
        };
        let bytes = encode(header, &tris);
        std::fs::write(path, &bytes)
    }

    /// Frees all GPU resources held by the state.
    pub fn dispose(&mut self, rd: &mut Gd<RenderingDevice>) {
        if let Some(ref state) = self.state {
            state.dispose(rd);
        }
        self.state = None;
    }

    /// Frees all GPU resources directly (no deferred dispatch).
    /// Use when the RenderingDevice will be freed immediately after.
    pub fn dispose_direct(&mut self, rd: &mut Gd<RenderingDevice>) {
        if let Some(ref state) = self.state {
            state.dispose_direct(rd);
        }
        self.state = None;
        if let Some(ref mut s) = self.mark_tris_shader {
            s.dispose_direct(rd);
        }
        if let Some(ref mut s) = self.update_neighbors_shader {
            s.dispose_direct(rd);
        }
        if let Some(ref mut s) = self.div_shader {
            s.dispose_direct(rd);
        }
        if let Some(ref mut s) = self.merge_shader {
            s.dispose_direct(rd);
        }
        if let Some(ref mut s) = self.compact_shaders {
            s.dispose_direct(rd);
        }
        if let Some(ref mut s) = self.final_state_shader {
            s.dispose_direct(rd);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_run_algo_config_defaults() {
        let config = RunAlgoConfig {
            subdivisions: 3,
            radius: 1.0,
            triangle_screen_size: 0.1,
            precise_normals: false,
            low_poly_look: false,
            show_debug_messages: false,
            show_debug_lod_histogram: false,
        };
        assert_eq!(config.subdivisions, 3);
        assert_eq!(config.radius, 1.0);
    }

    #[test]
    fn test_run_algo_initial_state() {
        let algo = CesRunAlgo::new();
        assert!(algo.state.is_none());
    }
}
