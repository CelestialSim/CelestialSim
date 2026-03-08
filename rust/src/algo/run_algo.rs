use godot::builtin::Vector3;
use godot::classes::RenderingDevice;
use godot::prelude::godot_print;

use crate::algo::compact_buffers;
use crate::algo::div_lod;
use crate::algo::final_state;
use crate::algo::mark_tris;
use crate::algo::merge_lod;
use crate::algo::update_neighbors;
use crate::initial_state;
use crate::layers::CesLayer;
use crate::state::CesState;

/// Configuration for the subdivision algorithm.
pub struct RunAlgoConfig {
    pub subdivisions: u32,
    pub radius: f32,
    pub triangle_screen_size: f32,
    pub precise_normals: bool,
    pub low_poly_look: bool,
    pub show_debug_messages: bool,
}

/// Orchestrates the adaptive LOD subdivision loop.
/// Mirrors C# `CesRunAlgo`.
pub struct CesRunAlgo {
    pub state: Option<CesState>,
    pub pos: Vec<Vector3>,
    pub normals: Vec<Vector3>,
    pub triangles: Vec<i32>,
    pub sim_value: Vec<[f32; 2]>,
}

impl CesRunAlgo {
    pub fn new() -> Self {
        Self {
            state: None,
            pos: vec![],
            normals: vec![],
            triangles: vec![],
            sim_value: vec![],
        }
    }

    /// Runs all layers' state initialization and position update.
    fn layers_update(
        rd: &mut RenderingDevice,
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

    /// Runs the main subdivision loop until convergence.
    /// Mirrors C# `CesRunAlgo.UpdateTriangleGraph`.
    pub fn update_triangle_graph(
        &mut self,
        rd: &mut RenderingDevice,
        cam_local: Vector3,
        config: &RunAlgoConfig,
        layers: &mut [Box<dyn CesLayer>],
        skip_auto_division_marking: bool,
    ) {
        if self.state.is_none() {
            self.state = Some(initial_state::create_core_state(rd));
            Self::layers_update(rd, self.state.as_ref().unwrap(), layers, config.radius);
        }

        let state = self.state.as_mut().unwrap();

        let mut n_tris_added: u32 = 0;
        let mut n_tris_merged: u32 = 0;
        let mut first_run = true;

        while n_tris_added > 0 || n_tris_merged > 0 || first_run {
            first_run = false;

            if !skip_auto_division_marking {
                mark_tris::flag_large_tris_to_divide(
                    rd,
                    state,
                    cam_local,
                    config.subdivisions,
                    config.radius,
                    config.triangle_screen_size,
                );
            }

            n_tris_added = div_lod::make_div(rd, state, config.precise_normals);
            if n_tris_added > 0 && config.show_debug_messages {
                godot_print!("Divided {} triangles", n_tris_added / 4);
            }

            n_tris_merged = merge_lod::make_merge(rd, state);
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
                compact_buffers::compact(rd, state);
            }

            if n_tris_added > 0 || n_tris_merged > 0 {
                update_neighbors::update_neighbors(rd, state);
            }

            Self::layers_update(rd, state, layers, config.radius);
        }

        let final_output = final_state::create_final_output(rd, state, config.low_poly_look);
        self.pos = final_output.pos;
        self.normals = final_output.normals;
        self.triangles = final_output.tris;
        self.sim_value = final_output.color;
    }

    /// Frees all GPU resources held by the state.
    pub fn dispose(&mut self, rd: &mut RenderingDevice) {
        if let Some(ref state) = self.state {
            state.dispose(rd);
        }
        self.state = None;
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
        };
        assert_eq!(config.subdivisions, 3);
        assert_eq!(config.radius, 1.0);
    }

    #[test]
    fn test_run_algo_initial_state() {
        let algo = CesRunAlgo::new();
        assert!(algo.state.is_none());
        assert!(algo.pos.is_empty());
        assert!(algo.normals.is_empty());
        assert!(algo.triangles.is_empty());
        assert!(algo.sim_value.is_empty());
    }
}
