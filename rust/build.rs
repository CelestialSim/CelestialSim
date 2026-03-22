use std::env;
use std::fs;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let shaders_dir = manifest_dir.join("../addons/celestial_sim/shaders");

    println!("cargo::rerun-if-env-changed=PROFILE");
    println!("cargo::rerun-if-env-changed=TARGET");
    ensure_gdextension_lib_alias(&manifest_dir);

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

fn ensure_gdextension_lib_alias(manifest_dir: &std::path::Path) {
    let profile = match env::var("PROFILE") {
        Ok(p) => p,
        Err(_) => return,
    };
    let target = match env::var("TARGET") {
        Ok(t) => t,
        Err(_) => return,
    };

    let (os_dir, lib_name) = if target.contains("windows") {
        ("windows", "celestial_sim.dll")
    } else if target.contains("apple") {
        ("macos", "libcelestial_sim.dylib")
    } else {
        ("linux", "libcelestial_sim.so")
    };

    let project_root = match manifest_dir.parent() {
        Some(p) => p,
        None => return,
    };

    let target_lib = project_root
        .join("rust")
        .join("target")
        .join(&profile)
        .join(lib_name);
    let alias_path = project_root
        .join("addons")
        .join("celestial_sim")
        .join("bin")
        .join(os_dir)
        .join(&profile)
        .join(lib_name);

    if let Some(parent) = alias_path.parent() {
        if let Err(e) = fs::create_dir_all(parent) {
            println!(
                "cargo::warning=Failed to create gdx alias directory {}: {}",
                parent.display(),
                e
            );
            return;
        }
    }

    if let Ok(meta) = fs::symlink_metadata(&alias_path) {
        if meta.is_dir() {
            let _ = fs::remove_dir_all(&alias_path);
        } else {
            let _ = fs::remove_file(&alias_path);
        }
    }

    if let Err(e) = create_symlink(&target_lib, &alias_path) {
        // On platforms where symlink creation is restricted, fallback to copy if source exists.
        if target_lib.exists() {
            match fs::copy(&target_lib, &alias_path) {
                Ok(_) => {
                    println!(
                        "cargo::warning=Symlink unavailable ({}); copied {} -> {}",
                        e,
                        target_lib.display(),
                        alias_path.display()
                    );
                }
                Err(copy_err) => {
                    println!(
                        "cargo::warning=Failed to alias gdextension lib (symlink: {}; copy: {})",
                        e, copy_err
                    );
                }
            }
        } else {
            println!(
                "cargo::warning=Failed to create gdextension symlink {} -> {}: {}",
                alias_path.display(),
                target_lib.display(),
                e
            );
        }
    }
}

#[cfg(unix)]
fn create_symlink(src: &std::path::Path, dst: &std::path::Path) -> std::io::Result<()> {
    std::os::unix::fs::symlink(src, dst)
}

#[cfg(windows)]
fn create_symlink(src: &std::path::Path, dst: &std::path::Path) -> std::io::Result<()> {
    std::os::windows::fs::symlink_file(src, dst)
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
