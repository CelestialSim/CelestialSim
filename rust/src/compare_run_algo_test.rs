#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    fn workspace_root() -> PathBuf {
        // rust/ is the crate root; workspace root is one level up
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..")
    }

    fn load_mesh_json(path: &std::path::Path) -> (Vec<[f64; 3]>, Vec<i64>) {
        let data = std::fs::read_to_string(path)
            .unwrap_or_else(|e| panic!("Failed to read {}: {}", path.display(), e));
        let val: serde_json::Value = serde_json::from_str(&data)
            .unwrap_or_else(|e| panic!("Failed to parse JSON {}: {}", path.display(), e));

        let vertices: Vec<[f64; 3]> = val["vertices"]
            .as_array()
            .expect("vertices should be an array")
            .iter()
            .map(|v| {
                let arr = v.as_array().expect("vertex should be [x,y,z]");
                [
                    arr[0].as_f64().unwrap(),
                    arr[1].as_f64().unwrap(),
                    arr[2].as_f64().unwrap(),
                ]
            })
            .collect();

        let triangles: Vec<i64> = val["triangles"]
            .as_array()
            .expect("triangles should be an array")
            .iter()
            .map(|v| v.as_i64().unwrap())
            .collect();

        (vertices, triangles)
    }

    #[test]
    fn compare_csharp_rust_run_algo() {
        let root = workspace_root();
        let csharp_path = root.join("debug/output/csharp_mesh.json");
        let rust_path = root.join("debug/output/rust_mesh.json");

        if !csharp_path.exists() || !rust_path.exists() {
            panic!(
                "JSON dumps not found. Run the dump scenes first:\n  \
                 ./debug/run_scene_with_log.sh \"$PWD\" res://debug/scenes/DumpRunAlgoCSharp.tscn\n  \
                 ./debug/run_scene_with_log.sh \"$PWD\" res://debug/scenes/DumpRunAlgoRust.tscn"
            );
        }

        let (cs_verts, cs_tris) = load_mesh_json(&csharp_path);
        let (rs_verts, rs_tris) = load_mesh_json(&rust_path);

        // Compare vertex counts
        assert_eq!(
            cs_verts.len(),
            rs_verts.len(),
            "Vertex count mismatch: C#={} Rust={}",
            cs_verts.len(),
            rs_verts.len()
        );

        // Compare triangle counts
        assert_eq!(
            cs_tris.len(),
            rs_tris.len(),
            "Triangle index count mismatch: C#={} Rust={}",
            cs_tris.len(),
            rs_tris.len()
        );

        // Compare vertices with tolerance
        let tol = 1e-5;
        let mut max_diff: f64 = 0.0;
        for (i, (cv, rv)) in cs_verts.iter().zip(rs_verts.iter()).enumerate() {
            for axis in 0..3 {
                let diff = (cv[axis] - rv[axis]).abs();
                if diff > max_diff {
                    max_diff = diff;
                }
                assert!(
                    diff < tol,
                    "Vertex {} axis {} differs: C#={} Rust={} diff={}",
                    i,
                    axis,
                    cv[axis],
                    rv[axis],
                    diff
                );
            }
        }

        // Compare triangle indices exactly
        for (i, (ct, rt)) in cs_tris.iter().zip(rs_tris.iter()).enumerate() {
            assert_eq!(
                ct, rt,
                "Triangle index {} differs: C#={} Rust={}",
                i, ct, rt
            );
        }

        println!(
            "PASS: {} vertices, {} triangle indices match (max vertex diff: {:.2e})",
            cs_verts.len(),
            cs_tris.len(),
            max_diff
        );
    }
}
