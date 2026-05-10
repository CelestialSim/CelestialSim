//! Pre-generates the project-local 32x32x32 blue-noise volume so it can be
//! committed to the repository. Run from the repo root:
//!
//!   cargo run --release --manifest-path rust/Cargo.toml --bin generate_blue_noise
//!
//! Writes `<repo_root>/.cache/blue_noise_32x32x32_seed0xce1e54.bin` (creating
//! the directory if needed). Re-running is a no-op when the cache already
//! matches the requested size+seed.

use std::path::PathBuf;
use std::time::Instant;

#[path = "../blue_noise.rs"]
mod blue_noise;

const SIZE: [u32; 3] = [32, 32, 32];

fn project_root() -> PathBuf {
    // CARGO_MANIFEST_DIR is `<repo>/rust`; the project root is one level up.
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("CARGO_MANIFEST_DIR has a parent")
        .to_path_buf()
}

fn main() {
    let [w, h, d] = SIZE;
    let seed = blue_noise::DEFAULT_SEED;
    let path = project_root()
        .join(".cache")
        .join(format!("blue_noise_{}x{}x{}_seed{:#x}.bin", w, h, d, seed));

    println!("Output: {}", path.display());
    let t0 = Instant::now();
    let bytes = blue_noise::load_or_generate(&path, SIZE, seed);
    println!(
        "Done in {:.2}s ({} payload bytes)",
        t0.elapsed().as_secs_f64(),
        bytes.len()
    );
    assert!(
        path.exists(),
        "expected cache file to exist after load_or_generate"
    );
}
