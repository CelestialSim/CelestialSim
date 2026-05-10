//! Warms the project-local blue-noise cache used by `celestial.rs`. Run once:
//!   `cargo test --release --test test_blue_noise_warm_project_cache -- --include-ignored --nocapture`
//! After this, `cd <project_root>` and starting Godot loads the cache instantly.

use std::path::PathBuf;
use std::time::Instant;

#[path = "../src/blue_noise.rs"]
mod blue_noise;

fn project_root() -> PathBuf {
    // CARGO_MANIFEST_DIR is `<repo>/rust`; the project root is one level up.
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .to_path_buf()
}

fn cache_path() -> PathBuf {
    let [w, h, d]: [u32; 3] = [32, 32, 32];
    project_root().join(".cache").join(format!(
        "blue_noise_{}x{}x{}_seed{:#x}.bin",
        w,
        h,
        d,
        blue_noise::DEFAULT_SEED
    ))
}

#[test]
#[ignore]
fn warm_project_blue_noise_cache_32_cubed() {
    let path = cache_path();
    println!("Cache path: {}", path.display());
    let t0 = Instant::now();
    let bytes = blue_noise::load_or_generate(&path, [32, 32, 32], blue_noise::DEFAULT_SEED);
    println!(
        "load_or_generate returned {} bytes in {:.2}s",
        bytes.len(),
        t0.elapsed().as_secs_f64()
    );
    assert_eq!(bytes.len(), 32 * 32 * 32);
    assert!(path.exists(), "cache file should exist after generation");
}
