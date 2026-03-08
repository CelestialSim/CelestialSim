use std::path::PathBuf;
use std::process::Command;

/// Integration test: launches the CesRustTestScene in Godot and verifies
/// the Rust GDExtension prints "Hello from Rust GDExtension!" in the log.
#[test]
fn test_rust_plugin_loads_in_godot() {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let project_root = manifest_dir.parent().unwrap();
    let script = project_root.join("debug/run_scene_with_log.sh");

    if !script.exists() {
        eprintln!("Skipping: run_scene_with_log.sh not found");
        return;
    }

    // Build the cdylib first so the .so is up to date
    let build = Command::new("cargo")
        .args(["build", "--manifest-path"])
        .arg(manifest_dir.join("Cargo.toml"))
        .output()
        .expect("failed to run cargo build");
    assert!(build.status.success(), "cargo build failed: {}", String::from_utf8_lossy(&build.stderr));

    let output = Command::new("bash")
        .arg(&script)
        .arg(project_root)
        .arg("res://debug/scenes/CesRustTestScene.tscn")
        .output()
        .expect("failed to run Godot scene");

    let stdout = String::from_utf8_lossy(&output.stdout);

    // Find the log file path from the script output
    let log_path = stdout
        .lines()
        .find(|l| l.starts_with("Saved log:"))
        .and_then(|l| l.strip_prefix("Saved log: "))
        .map(|p| p.trim())
        .expect("Could not find log file path in script output");

    let log_contents = std::fs::read_to_string(log_path)
        .unwrap_or_else(|e| panic!("Failed to read log file {}: {}", log_path, e));

    assert!(
        log_contents.contains("[CelestialSimRust] Hello from Rust GDExtension!"),
        "Log does not contain Rust hello message. Log contents:\n{}",
        log_contents
    );
    assert!(
        log_contents.contains("Rust plugin integration test PASSED"),
        "Log does not contain PASSED message. Log contents:\n{}",
        log_contents
    );

    println!("Godot integration test passed: Rust plugin loaded successfully");
}
