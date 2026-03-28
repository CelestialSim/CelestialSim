use std::path::PathBuf;

fn spv_path(name: &str) -> PathBuf {
    let out_dir = std::env::var("OUT_DIR").expect("OUT_DIR not set — run via `cargo test`");
    PathBuf::from(out_dir).join(format!("{}.spv", name))
}

#[test]
fn test_cubemap_noise_spirv_compiled() {
    let spv_file = spv_path("CubemapNoise");
    if !spv_file.exists() {
        eprintln!(
            "Skipping test_cubemap_noise_spirv_compiled: {} not found (slangc not available)",
            spv_file.display()
        );
        return;
    }
    let bytes = std::fs::read(&spv_file).unwrap();
    assert!(
        bytes.len() > 100,
        "SPIR-V file too small ({} bytes): {}",
        bytes.len(),
        spv_file.display()
    );
}
