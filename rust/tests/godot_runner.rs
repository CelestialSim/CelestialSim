//! Test-side helpers for locating the Godot binary and running a scene
//! headlessly. Mirrors the lookup order in `debug/run_scene_with_log.sh`:
//! `$GODOT_PATH` → `godot4` → `godot` → flatpak `org.godotengine.Godot`.
//! Returns `None` so callers can skip-with-warning when no binary is found,
//! matching the convention used in `test_sphere_terrain.rs` for missing slangc.

use std::path::Path;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

/// Path to the project root, two levels up from the cdylib `Cargo.toml`.
pub fn project_path() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("CARGO_MANIFEST_DIR should have a parent")
        .to_path_buf()
}

/// Returns a `Command` set up to invoke Godot, or `None` if no binary is
/// available. The returned command has no arguments yet — callers add them.
pub fn find_godot_command() -> Option<Command> {
    if let Ok(p) = std::env::var("GODOT_PATH") {
        if !p.is_empty() {
            return Some(Command::new(p));
        }
    }
    for name in ["godot4", "godot"] {
        if which(name).is_some() {
            return Some(Command::new(name));
        }
    }
    if has_flatpak_godot() {
        let mut cmd = Command::new("flatpak");
        cmd.args(["run", "org.godotengine.Godot"]);
        return Some(cmd);
    }
    None
}

fn which(name: &str) -> Option<std::path::PathBuf> {
    let path = std::env::var_os("PATH")?;
    for dir in std::env::split_paths(&path) {
        let candidate = dir.join(name);
        if candidate.is_file() {
            return Some(candidate);
        }
    }
    None
}

fn has_flatpak_godot() -> bool {
    let out = Command::new("flatpak")
        .arg("list")
        .stderr(Stdio::null())
        .output();
    match out {
        Ok(o) if o.status.success() => {
            let s = String::from_utf8_lossy(&o.stdout);
            s.lines()
                .any(|l| l.to_ascii_lowercase().contains("godot"))
        }
        _ => false,
    }
}

/// Runs a Godot scene headlessly with the given env vars and waits up to
/// `timeout`. Returns Ok(()) on a clean exit (or if the process exits non-zero
/// — Godot occasionally exits 1 for unrelated late-shutdown issues; we let the
/// caller decide based on file output). Returns Err on spawn failure or
/// timeout.
pub fn run_scene_headless(
    mut command: Command,
    project_path: &Path,
    scene: &str,
    env: &[(&str, &str)],
    timeout: Duration,
) -> std::io::Result<()> {
    // Note: we deliberately do NOT pass `--headless`. Under `--headless`,
    // Godot's `RenderingServer::create_local_rendering_device()` returns None
    // on this host (no Vulkan in pure software headless mode), and
    // `CesCelestialRust::restart_with_current_layers` panics on the resulting
    // `Option::unwrap`. Without `--headless` Godot falls back through
    // X11 → Wayland and the local RD is created against the real GPU. CI
    // environments without a display server will need a virtual one (e.g.
    // Xvfb) or a `--headless` path that fixes the local-RD issue separately.
    command.arg("--path").arg(project_path).arg(scene);
    for (k, v) in env {
        command.env(k, v);
    }
    // Forward Godot's stdout/stderr through this process so we can see what
    // happened on test failure. Piping without draining can deadlock when the
    // OS pipe buffer fills.
    command.stdout(Stdio::inherit()).stderr(Stdio::inherit());

    let mut child = command.spawn()?;
    let start = Instant::now();
    loop {
        match child.try_wait()? {
            Some(_status) => return Ok(()),
            None => {
                if start.elapsed() >= timeout {
                    let _ = child.kill();
                    let _ = child.wait();
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::TimedOut,
                        format!("Godot scene timed out after {:?}", timeout),
                    ));
                }
                std::thread::sleep(Duration::from_millis(100));
            }
        }
    }
}
