//! Stages precompiled SPIR-V into `OUT_DIR` so the crate can `include_bytes!`
//! it. SPIR-V is **committed** under `shaders/spirv/` and is the default build
//! input — a normal `cargo build` needs no `slangc`, so `slang` is a dev-only
//! (build-time) tool, not a release dependency.
//!
//! When you edit a `*.slang` shader, regenerate and re-commit the artifacts:
//!
//! ```text
//! SLANG_RECOMPILE=1 cargo build -p celestialsim   # rewrites shaders/spirv/*.spv
//! git add crates/celestialsim/shaders/spirv
//! ```
//!
//! Only the regenerate path needs `slangc` on `PATH` (or `$SLANGC`). CI runs
//! the regenerate path and `git diff --exit-code` to catch stale SPIR-V.

use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let shaders_dir = manifest_dir.join("shaders");
    let spirv_dir = shaders_dir.join("spirv");

    println!("cargo::rerun-if-changed={}", shaders_dir.display());
    println!("cargo::rerun-if-env-changed=SLANG_RECOMPILE");
    println!("cargo::rerun-if-env-changed=SLANGC");

    if env::var_os("SLANG_RECOMPILE").is_some() {
        recompile_committed_spirv(&shaders_dir, &spirv_dir);
    } else {
        warn_if_spirv_stale(&shaders_dir, &spirv_dir);
    }

    stage_committed_spirv(&spirv_dir, &out_dir);
}

/// Warn (don't fail) when an entry-point `*.slang` is newer than its committed
/// `*.spv` — the default build won't recompile, so the dev must regenerate.
fn warn_if_spirv_stale(shaders_dir: &Path, spirv_dir: &Path) {
    let Ok(entries) = fs::read_dir(shaders_dir) else { return };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("slang") {
            continue;
        }
        if !fs::read_to_string(&path).unwrap_or_default().contains("computeMain") {
            continue;
        }
        let stem = path.file_stem().unwrap().to_str().unwrap();
        let spv = spirv_dir.join(format!("{stem}.spv"));
        let newer = |a: &Path, b: &Path| match (a.metadata(), b.metadata()) {
            (Ok(am), Ok(bm)) => match (am.modified(), bm.modified()) {
                (Ok(at), Ok(bt)) => at > bt,
                _ => false,
            },
            _ => false,
        };
        if !spv.exists() || newer(&path, &spv) {
            println!(
                "cargo::warning={} is newer than committed {}; run \
                 `SLANG_RECOMPILE=1 cargo build -p celestialsim` and commit shaders/spirv/",
                path.display(),
                spv.display()
            );
        }
    }
}

/// Copy the committed `shaders/spirv/*.spv` into `OUT_DIR` for `include_bytes!`.
/// This is the default (slangc-free) build path.
fn stage_committed_spirv(spirv_dir: &Path, out_dir: &Path) {
    let entries = fs::read_dir(spirv_dir).unwrap_or_else(|e| {
        panic!(
            "committed SPIR-V dir {} missing ({e}); run `SLANG_RECOMPILE=1 cargo build` \
             with slangc installed to generate it",
            spirv_dir.display()
        )
    });
    let mut staged = 0;
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("spv") {
            continue;
        }
        println!("cargo::rerun-if-changed={}", path.display());
        let file_name = path.file_name().unwrap();
        fs::copy(&path, out_dir.join(file_name))
            .unwrap_or_else(|e| panic!("copy {} -> OUT_DIR failed: {e}", path.display()));
        staged += 1;
    }
    if staged == 0 {
        panic!(
            "no *.spv in {}; run `SLANG_RECOMPILE=1 cargo build` with slangc installed",
            spirv_dir.display()
        );
    }
}

/// Run `slangc` over every entry-point `*.slang` and (re)write the committed
/// `shaders/spirv/*.spv`. Only invoked when `SLANG_RECOMPILE` is set.
fn recompile_committed_spirv(shaders_dir: &Path, spirv_dir: &Path) {
    let slangc = which_slangc().expect(
        "SLANG_RECOMPILE set but slangc not found on PATH or $SLANGC — \
         install shader-slang to regenerate SPIR-V",
    );
    fs::create_dir_all(spirv_dir).expect("create shaders/spirv/ dir");

    for entry in fs::read_dir(shaders_dir).expect("shaders/ dir missing").flatten() {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("slang") {
            continue;
        }
        println!("cargo::rerun-if-changed={}", path.display());
        // Modules (imported by other shaders) have no compute entry point.
        if !fs::read_to_string(&path).unwrap_or_default().contains("computeMain") {
            continue;
        }
        let stem = path.file_stem().unwrap().to_str().unwrap();
        let spv_path = spirv_dir.join(format!("{stem}.spv"));
        let status = Command::new(&slangc)
            .arg(&path)
            .args(["-target", "spirv"])
            .args(["-entry", "computeMain"])
            .args(["-stage", "compute"])
            .arg("-I")
            .arg(shaders_dir)
            .arg("-o")
            .arg(&spv_path)
            .status();
        match status {
            Ok(s) if s.success() => {}
            other => panic!("slangc failed for {}: {:?}", path.display(), other),
        }
    }
}

fn which_slangc() -> Option<String> {
    if let Ok(path) = env::var("SLANGC") {
        return Some(path);
    }
    let output = Command::new(if cfg!(windows) { "where" } else { "which" })
        .arg("slangc")
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let stdout = String::from_utf8(output.stdout).ok()?;
    let path = stdout.lines().next()?.trim().to_string();
    (!path.is_empty()).then_some(path)
}
