use godot::builtin::Vector3;
use godot::classes::RenderingDevice;
use godot::obj::Gd;
use godot::prelude::godot_print;
use std::time::{Duration, Instant};

use crate::algo::compact_buffers::CompactShaders;
use crate::algo::div_lod::DivShader;
use crate::algo::final_state::{self, FinalStateShader};
use crate::algo::mark_tris::MarkTrisShader;
use crate::algo::merge_lod::MergeShader;
use crate::algo::update_neighbors::UpdateNeighborsShader;
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

struct AlgoTimingTotals {
    total: Duration,
    mark: Duration,
    divide: Duration,
    merge: Duration,
    compact: Duration,
    neighbors: Duration,
    layers: Duration,
    final_output: Duration,
    iterations: u32,
}

impl AlgoTimingTotals {
    fn print_summary(&self) {
        godot_print!(
            "AlgoTiming summary: iterations={} total={:.3} ms",
            self.iterations,
            self.total.as_secs_f64() * 1000.0
        );
        godot_print!(
            "AlgoTiming step MarkTrisToDivide: {:.3} ms",
            self.mark.as_secs_f64() * 1000.0
        );
        godot_print!(
            "AlgoTiming step Divide: {:.3} ms",
            self.divide.as_secs_f64() * 1000.0
        );
        godot_print!(
            "AlgoTiming step Merge: {:.3} ms",
            self.merge.as_secs_f64() * 1000.0
        );
        godot_print!(
            "AlgoTiming step Compact: {:.3} ms",
            self.compact.as_secs_f64() * 1000.0
        );
        godot_print!(
            "AlgoTiming step UpdateNeighbors: {:.3} ms",
            self.neighbors.as_secs_f64() * 1000.0
        );
        godot_print!(
            "AlgoTiming step LayersUpdate: {:.3} ms",
            self.layers.as_secs_f64() * 1000.0
        );
        godot_print!(
            "AlgoTiming step FinalOutput: {:.3} ms",
            self.final_output.as_secs_f64() * 1000.0
        );

        let mut slowest = ("MarkTrisToDivide", self.mark);
        if self.divide > slowest.1 {
            slowest = ("Divide", self.divide);
        }
        if self.merge > slowest.1 {
            slowest = ("Merge", self.merge);
        }
        if self.compact > slowest.1 {
            slowest = ("Compact", self.compact);
        }
        if self.neighbors > slowest.1 {
            slowest = ("UpdateNeighbors", self.neighbors);
        }
        if self.layers > slowest.1 {
            slowest = ("LayersUpdate", self.layers);
        }
        if self.final_output > slowest.1 {
            slowest = ("FinalOutput", self.final_output);
        }

        godot_print!(
            "AlgoTiming slowest: {} ({:.3} ms)",
            slowest.0,
            slowest.1.as_secs_f64() * 1000.0
        );
    }
}

/// Orchestrates the adaptive LOD subdivision loop.
/// Mirrors C# `CesRunAlgo`.
pub struct CesRunAlgo {
    pub state: Option<CesState>,
    pub pos: Vec<Vector3>,
    pub normals: Vec<Vector3>,
    pub triangles: Vec<i32>,
    pub sim_value: Vec<[f32; 2]>,
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
            pos: vec![],
            normals: vec![],
            triangles: vec![],
            sim_value: vec![],
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

    /// Runs the main subdivision loop until convergence.
    /// Mirrors C# `CesRunAlgo.UpdateTriangleGraph`.
    pub fn update_triangle_graph(
        &mut self,
        rd: &mut Gd<RenderingDevice>,
        cam_local: Vector3,
        config: &RunAlgoConfig,
        layers: &mut [Box<dyn CesLayer>],
        skip_auto_division_marking: bool,
    ) {
        let total_start = Instant::now();
        let mut timings = AlgoTimingTotals {
            total: Duration::ZERO,
            mark: Duration::ZERO,
            divide: Duration::ZERO,
            merge: Duration::ZERO,
            compact: Duration::ZERO,
            neighbors: Duration::ZERO,
            layers: Duration::ZERO,
            final_output: Duration::ZERO,
            iterations: 0,
        };

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
            timings.iterations += 1;

            if !skip_auto_division_marking {
                let mark_start = Instant::now();
                self.mark_tris_shader
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
                timings.mark += mark_start.elapsed();
            }

            let divide_start = Instant::now();
            n_tris_added =
                self.div_shader
                    .as_ref()
                    .unwrap()
                    .make_div(rd, state, config.precise_normals);
            timings.divide += divide_start.elapsed();
            if n_tris_added > 0 {
                state.sync_n_tris_buffer(rd);
                state.sync_n_verts_buffer(rd);
                if config.show_debug_messages {
                    godot_print!("Divided {} triangles", n_tris_added / 4);
                }
            }

            let merge_start = Instant::now();
            n_tris_merged = self.merge_shader.as_ref().unwrap().make_merge(rd, state);
            timings.merge += merge_start.elapsed();
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
                let compact_start = Instant::now();
                self.compact_shaders.as_ref().unwrap().compact(rd, state);
                timings.compact += compact_start.elapsed();
            }

            if n_tris_added > 0 || n_tris_merged > 0 {
                let neighbors_start = Instant::now();
                self.update_neighbors_shader
                    .as_ref()
                    .unwrap()
                    .update_neighbors(rd, state);
                timings.neighbors += neighbors_start.elapsed();
            }

            let layers_start = Instant::now();
            Self::layers_update(rd, state, layers, config.radius);
            timings.layers += layers_start.elapsed();
        }

        let final_start = Instant::now();
        let final_output = final_state::create_final_output(
            rd,
            state,
            config.low_poly_look,
            self.final_state_shader.as_ref().unwrap(),
        );
        timings.final_output += final_start.elapsed();
        self.pos = final_output.pos;
        self.normals = final_output.normals;
        self.triangles = final_output.tris;
        self.sim_value = final_output.color;

        timings.total = total_start.elapsed();
        if config.show_debug_messages {
            timings.print_summary();
        }
    }

    /// Frees all GPU resources held by the state.
    pub fn dispose(&mut self, rd: &mut Gd<RenderingDevice>) {
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
