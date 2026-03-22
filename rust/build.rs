use std::env;
use std::fs;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let shaders_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap())
        .join("../addons/celestial_sim/shaders");

    println!("cargo::rerun-if-changed={}", shaders_dir.display());

    let slangc = match which_slangc() {
        Some(path) => path,
        None => {
            println!(
                "cargo::warning=slangc not found on PATH; \
                 shader SPIR-V will not be compiled. Shader tests will be skipped."
            );
            return;
        }
    };

    let entries = match fs::read_dir(&shaders_dir) {
        Ok(e) => e,
        Err(e) => {
            println!(
                "cargo::warning=Cannot read shaders directory {}: {}",
                shaders_dir.display(),
                e
            );
            return;
        }
    };

    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("slang") {
            continue;
        }

        let stem = path.file_stem().unwrap().to_str().unwrap();
        let spv_path = out_dir.join(format!("{}.spv", stem));

        println!("cargo::rerun-if-changed={}", path.display());

        let status = Command::new(&slangc)
            .arg(&path)
            .args(["-target", "spirv"])
            .args(["-entry", "computeMain"])
            .args(["-stage", "compute"])
            .arg("-I")
            .arg(&shaders_dir)
            .arg("-o")
            .arg(&spv_path)
            .status();

        match status {
            Ok(s) if s.success() => {
                // println!(
                //     "cargo::warning=Compiled {} -> {}",
                //     path.display(),
                //     spv_path.display()
                // );
            }
            Ok(s) => {
                println!(
                    "cargo::warning=slangc failed for {} (exit code {:?})",
                    path.display(),
                    s.code()
                );
            }
            Err(e) => {
                println!(
                    "cargo::warning=Failed to run slangc for {}: {}",
                    path.display(),
                    e
                );
            }
        }
    }
}

fn which_slangc() -> Option<String> {
    // Check explicit env var first
    if let Ok(path) = env::var("SLANGC") {
        return Some(path);
    }

    // Try to find on PATH
    let output = Command::new("which").arg("slangc").output().ok()?;
    if output.status.success() {
        let path = String::from_utf8(output.stdout).ok()?.trim().to_string();
        if !path.is_empty() {
            return Some(path);
        }
    }

    None
}
