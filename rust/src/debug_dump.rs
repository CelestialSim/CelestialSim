use godot::builtin::Vector3;
use godot::classes::{INode, RenderingServer};
use godot::prelude::*;

use crate::algo::run_algo::{CesRunAlgo, RunAlgoConfig};
use crate::layers::sphere_terrain::CesSphereTerrain;
use crate::layers::CesLayer;

/// Debug node: runs the full CesRunAlgo pipeline with subdivision=1,
/// triangle_screen_size=0, then saves positions and triangles as JSON.
#[derive(GodotClass)]
#[class(base = Node)]
pub struct CesDumpSubdivisionRust {
    base: Base<Node>,
}

#[godot_api]
impl INode for CesDumpSubdivisionRust {
    fn init(base: Base<Node>) -> Self {
        Self { base }
    }

    fn ready(&mut self) {
        godot_print!("[DUMP_RUST] Starting Rust RunAlgo dump...");

        let rs = RenderingServer::singleton();
        let mut rd = rs.create_local_rendering_device().unwrap();

        let config = RunAlgoConfig {
            subdivisions: 1,
            radius: 1.0,
            triangle_screen_size: 0.0,
            precise_normals: false,
            low_poly_look: true,
            show_debug_messages: true,
        };

        let mut algo = CesRunAlgo::new();
        let mut layers: Vec<Box<dyn CesLayer>> = vec![Box::new(CesSphereTerrain::new())];

        // Camera far away; triangle_screen_size=0 forces all subdivisions
        let cam_local = Vector3::new(0.0, 0.0, 100.0);
        algo.update_triangle_graph(&mut rd, cam_local, &config, &mut layers, false);

        godot_print!(
            "[DUMP_RUST] pos={} tris={}",
            algo.pos.len(),
            algo.triangles.len()
        );

        // Build JSON
        let mut json = String::from("{\n  \"vertices\": [\n");
        for (i, v) in algo.pos.iter().enumerate() {
            if i > 0 {
                json.push_str(",\n");
            }
            json.push_str(&format!("    [{}, {}, {}]", v.x, v.y, v.z));
        }
        json.push_str("\n  ],\n  \"triangles\": [");
        for (i, &idx) in algo.triangles.iter().enumerate() {
            if i > 0 {
                json.push(',');
            }
            json.push_str(&format!("{}", idx));
        }
        json.push_str("]\n}\n");

        let out_dir = "debug/output";
        std::fs::create_dir_all(out_dir).unwrap();
        let out_path = format!("{}/rust_mesh.json", out_dir);
        std::fs::write(&out_path, &json).unwrap();
        godot_print!("[DUMP_RUST] Wrote {}", out_path);

        algo.dispose(&mut rd);
        godot_print!("[DUMP_RUST] DONE");
        self.base().get_tree().quit();
    }
}
