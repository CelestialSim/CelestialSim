//! Triangle-distribution probe (CEL-67 follow-up): bins realized triangles by
//! distance to the camera and reports count + screen-space cell size per band,
//! so we can see where triangles actually land (too many close? too few mid?
//! too many far?). Pure CPU via the same `compute_patches` / `realize_patch`
//! the GPU mirrors — deterministic, no engine.
//!
//! Run: `cargo run -p celestialsim-algo --example tri_hist --release`

use celestial_algo::clipmap::{compute_patches, realize_patch, FaceFrame};
use celestial_algo::icosphere::{base_vertex, BASE_FACES};
use godot::builtin::{Color, Vector3};

// HQ scene params (celestial_v4_hq.tscn) + node defaults.
const RADIUS: f32 = 1000.0;
const NUM_LAYERS: u32 = 10;
const RING: u32 = 4;
const BASE_CELL: f32 = 64.0;
const SCREEN_ERROR: f32 = 0.01;
const FALLOFF: f32 = 1.0;

// A pixel's angular size at FOV 60deg over a 720px-tall viewport (rad/px).
const RAD_PER_PX: f32 = (60.0f32 * std::f32::consts::PI / 180.0) / 720.0;

fn main() {
    // Camera ~40 units above the surface (the bench's near-surface viewpoint).
    let here = Vector3::new(0.0, 0.30, 1.0).normalized();
    let cam = here * (RADIUS + 40.0);

    let frames: Vec<FaceFrame> = BASE_FACES
        .iter()
        .map(|f| FaceFrame {
            a: base_vertex(f[0]) * RADIUS,
            b: base_vertex(f[1]) * RADIUS,
            c: base_vertex(f[2]) * RADIUS,
            radius: RADIUS,
        })
        .collect();

    // Horizon: a surface point (unit dir d) is visible iff it lies within the
    // angular horizon cap of the sub-camera direction, cos(alpha) = R/|cam|.
    let cam_hat = cam.normalized();
    let cos_horizon = RADIUS / cam.length();

    // ACTUAL triangle count vs `angle_falloff` (cos-θ patch foreshortening) — the
    // real pipeline number, not the projection below. Re-runs compute_patches for
    // a few strengths and sums realized triangles.
    println!("--- actual angle_falloff sweep (real compute_patches tri counts) ---");
    let total_for = |af: f32| -> u64 {
        let mut n = 0u64;
        for frame in &frames {
            for patch in compute_patches(*frame, NUM_LAYERS, RING, BASE_CELL, SCREEN_ERROR, FALLOFF, af, true, Some(cam)) {
                n += realize_patch(&patch, Color::WHITE).tris.len() as u64;
            }
        }
        n
    };
    let base_tris = total_for(0.0);
    for af in [0.0f32, 0.25, 0.5, 0.75, 1.0] {
        let t = total_for(af);
        println!("  angle_falloff={af:.2}: {t:>10} tris ({:+.0}%)", 100.0 * (t as f64 / base_tris as f64 - 1.0));
    }

    // Distance bands (world units from the camera).
    let edges = [0.0, 25.0, 50.0, 100.0, 200.0, 400.0, 800.0, 1600.0, f32::INFINITY];
    let nb = edges.len() - 1;
    let mut count = vec![0u64; nb];
    let mut sum_px = vec![0.0f64; nb]; // sum of screen-space longest-edge px
    let mut vis_count = vec![0u64; nb]; // visible (in front of horizon) per band
    let mut total = 0u64;
    let mut visible_total = 0u64;

    let mut face_occ = vec![(0u64, 0u64); frames.len()]; // (total, occluded) per face
    let mut layer_occ = vec![(0u64, 0u64); NUM_LAYERS as usize]; // (total, occluded) per layer
    // Approach (a) projection: under screen_size *= f(angle), a region at angle
    // theta needs ~1/f(theta) bigger cells => f(theta)^2 as many triangles. So
    // the projected triangle count is the sum of f(theta)^2 over current cells
    // (f = max(0,cos theta), zero past 90 deg). cos^1 = gentler density law.
    let mut proj_cos2 = 0.0f64; // density ∝ cos^2 (2D foreshorten both axes)
    let mut proj_cos1 = 0.0f64; // density ∝ cos^1 (one axis)
    for (fi, frame) in frames.iter().enumerate() {
        for patch in compute_patches(*frame, NUM_LAYERS, RING, BASE_CELL, SCREEN_ERROR, FALLOFF, 0.0, true, Some(cam)) {
            for t in realize_patch(&patch, Color::WHITE).tris {
                let c = (t.corners[0] + t.corners[1] + t.corners[2]) / 3.0;
                let dist = (c - cam).length().max(1.0e-4);
                let e = [
                    (t.corners[1] - t.corners[0]).length(),
                    (t.corners[2] - t.corners[1]).length(),
                    (t.corners[0] - t.corners[2]).length(),
                ]
                .into_iter()
                .fold(0.0f32, f32::max);
                let px = (e / dist) / RAD_PER_PX; // screen-space longest edge, pixels
                let b = (0..nb).find(|&i| dist < edges[i + 1]).unwrap_or(nb - 1);
                count[b] += 1;
                sum_px[b] += px as f64;
                total += 1;
                let cos_theta = c.normalized().dot(cam_hat).max(0.0); // 0 past 90 deg
                proj_cos2 += (cos_theta * cos_theta) as f64;
                proj_cos1 += cos_theta as f64;
                let occluded = c.normalized().dot(cam_hat) < cos_horizon;
                if !occluded {
                    vis_count[b] += 1;
                    visible_total += 1;
                }
                face_occ[fi].0 += 1;
                layer_occ[patch.layer as usize].0 += 1;
                if occluded {
                    face_occ[fi].1 += 1;
                    layer_occ[patch.layer as usize].1 += 1;
                }
            }
        }
    }
    println!(
        "--- approach (a) projection: screen_size *= f(angle from planet center) ---"
    );
    println!(
        "  cos^2 law: {:.0} tris ({:.0}% of base) -> -{:.0}%",
        proj_cos2, 100.0 * proj_cos2 / total as f64, 100.0 * (1.0 - proj_cos2 / total as f64)
    );
    println!(
        "  cos^1 law: {:.0} tris ({:.0}% of base) -> -{:.0}%",
        proj_cos1, 100.0 * proj_cos1 / total as f64, 100.0 * (1.0 - proj_cos1 / total as f64)
    );
    println!("--- per-face (top 8 by occluded cells) ---");
    let mut order: Vec<usize> = (0..frames.len()).collect();
    order.sort_by_key(|&i| std::cmp::Reverse(face_occ[i].1));
    for &i in order.iter().take(8) {
        let (tot, occ) = face_occ[i];
        println!("  face {i:2}: {tot:>8} tris, {occ:>8} occluded ({:.0}%)", 100.0 * occ as f64 / tot.max(1) as f64);
    }
    println!("--- per-layer ---");
    for (l, (tot, occ)) in layer_occ.iter().enumerate() {
        if *tot > 0 {
            println!("  layer {l:2}: {tot:>8} tris, {occ:>8} occluded ({:.0}%)", 100.0 * *occ as f64 / (*tot).max(1) as f64);
        }
    }

    // --- Per-CELL angle gate: drop a triangle when its region is beyond the
    // horizon cap + a peak-height margin (so mountains poking over the horizon
    // survive). Cracks at that boundary are themselves occluded => safe. We
    // report the triangle reduction and any VISIBLE triangle lost (pop risk).
    //
    // Peak margin: a surface point's terrain peak (height factor `peak`, e.g.
    // 1.30 => +0.30R) reaches further over the limb. The keep-cone half-angle
    // grows so a peak at the edge stays in: cos(cap) = R/(peak*|cam|)-ish; we
    // sweep `peak` to show the safe-vs-aggressive tradeoff. ---
    println!("\n--- per-cell horizon gate sweep (drop cells beyond horizon + peak margin) ---");
    println!("{:>8} {:>12} {:>9} {:>13} {:>13}", "peak", "tris", "vs base", "occluded%", "visible lost");
    for peak in [1.0f32, 1.10, 1.20, 1.30, 1.45] {
        // A surface point at direction d is kept if it (or its peak) is within
        // the horizon cap. Peak raises the effective camera horizon: a point is
        // visible-with-peak when dot(d,cam_hat) >= R/(peak*|cam|).
        let cap = (RADIUS / (peak * cam.length())).clamp(-1.0, 1.0);
        let mut g_total = 0u64;
        let mut g_occluded = 0u64;
        let mut visible_lost = 0u64;
        for frame in &frames {
            for patch in compute_patches(*frame, NUM_LAYERS, RING, BASE_CELL, SCREEN_ERROR, FALLOFF, 0.0, true, Some(cam)) {
                for t in realize_patch(&patch, Color::WHITE).tris {
                    let ctr = (t.corners[0] + t.corners[1] + t.corners[2]) / 3.0;
                    let d = ctr.normalized().dot(cam_hat);
                    let kept = d >= cap; // within horizon+peak cone
                    let vis = d >= cos_horizon; // geometrically visible (no peak)
                    if !kept {
                        if vis { visible_lost += 1; } // dropped a visible cell (pop)
                        continue; // gated out
                    }
                    g_total += 1;
                    if !vis { g_occluded += 1; }
                }
            }
        }
        println!(
            "{:>7.2} {:>12} {:>8.1}% {:>12.1}% {:>13}",
            peak, g_total,
            100.0 * g_total as f64 / total as f64,
            100.0 * g_occluded as f64 / g_total.max(1) as f64,
            visible_lost,
        );
    }

    let horizon_dist = (cam.length() * cam.length() - RADIUS * RADIUS).sqrt();
    println!(
        "camera |p|={:.0} ({:.0} above surface), horizon at ~{:.0} units, total tris = {total}",
        cam.length(), cam.length() - RADIUS, horizon_dist
    );
    println!(
        "visible (in front of horizon) = {visible_total} ({:.1}%), occluded = {} ({:.1}%)",
        100.0 * visible_total as f64 / total.max(1) as f64,
        total - visible_total,
        100.0 * (total - visible_total) as f64 / total.max(1) as f64,
    );
    println!("{:>10} {:>10} {:>7} {:>12} {:>10} {:>10}", "band(world)", "tris", "%", "cum%", "avg px", "visible%");
    let mut cum = 0u64;
    for i in 0..nb {
        cum += count[i];
        let lo = edges[i];
        let hi = edges[i + 1];
        let band = if hi.is_finite() { format!("{lo:.0}-{hi:.0}") } else { format!("{lo:.0}+") };
        let pct = 100.0 * count[i] as f64 / total.max(1) as f64;
        let cumpct = 100.0 * cum as f64 / total.max(1) as f64;
        let avgpx = if count[i] > 0 { sum_px[i] / count[i] as f64 } else { 0.0 };
        let vispct = if count[i] > 0 { 100.0 * vis_count[i] as f64 / count[i] as f64 } else { 0.0 };
        println!("{band:>10} {:>10} {pct:>6.1}% {cumpct:>11.1}% {avgpx:>10.1} {vispct:>9.0}%", count[i]);
    }
}
