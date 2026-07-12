//! Task 10 verification: run the committed `ChunkRealize.spv` on a real Vulkan
//! `RenderingDevice` and prove the realized vertex positions match the CPU
//! reference `quadtree::chunk_subvertex_base` (terrain OFF), plus a terrain-ON
//! envelope check.
//!
//! Mirrors `conform_gpu_test`: a throwaway `Node` whose `run()` drives the GPU
//! on a **local** `RenderingDevice` (so we own `submit`/`sync` and can read the
//! pool back deterministically — the main device's submit is owned by the
//! renderer). It exercises the real production [`ChunkGpuResources`] compute path
//! (pipeline build, descriptor/params upload, `record_realize`, vertex-pool
//! readback); only the `MultiMesh` (a main-device, render-only concern) is
//! skipped via `new_compute_only`. A scene calls `run()` and prints
//! `CHUNK_GPU_TEST: PASS/FAIL ...`.

use bytemuck::Zeroable;
use godot::classes::{Node, RenderingDevice, RenderingServer};
use godot::prelude::*;

use celestial_algo::clipmap::FaceFrame;
use celestial_algo::quadtree::{base_face_frames, chunk_subvertex_base, Bary, Chunk, ChunkId};

use crate::chunk_descriptors::{pack_chunks, pack_instances, verts_per_chunk};
use crate::descriptors::{assemble, HeightGpu, TerrainGpu, TextureGpu};
use crate::gpu::chunk_gpu::ChunkGpuResources;

#[derive(GodotClass)]
#[class(base = Node, init, internal)]
pub struct ChunkRealizeGpuTest {
    base: Base<Node>,
}

#[godot_api]
impl ChunkRealizeGpuTest {
    /// Run the chunk-realize GPU verification; returns a PASS/FAIL line.
    #[func]
    fn run(&self) -> GString {
        match run_check() {
            Ok(msg) => GString::from(format!("CHUNK_GPU_TEST: PASS — {msg}").as_str()),
            Err(msg) => GString::from(format!("CHUNK_GPU_TEST: FAIL — {msg}").as_str()),
        }
    }

    /// Run the CEL-73 scatter-place GPU verification: dispatch the committed
    /// `ScatterPlace.spv` on a local device and compare every pool record
    /// (validity, hash01, full transform) against the `celestial_algo::scatter`
    /// CPU reference — including the gnomonic projection. PASS/FAIL line.
    #[func]
    fn run_scatter(&self) -> GString {
        match scatter_check() {
            Ok(msg) => GString::from(format!("SCATTER_GPU_TEST: PASS — {msg}").as_str()),
            Err(msg) => GString::from(format!("SCATTER_GPU_TEST: FAIL — {msg}").as_str()),
        }
    }

    /// Debug: bake ONE chunk's detail tile at `tile_res` (terrain ON) and save the
    /// colour + normal atlases as `<out_dir>/bake_color_<tile_res>.png` /
    /// `bake_normal_<tile_res>.png`. Isolates the bake shader's output (noise
    /// quality) from the render pipeline. `out_dir` is a res:// or absolute path.
    #[func]
    fn bake_chunk_png(&self, out_dir: GString, tile_res: i64) -> GString {
        match bake_png(&out_dir.to_string(), tile_res.clamp(8, 2048) as u32) {
            Ok(m) => GString::from(format!("BAKE_PNG: {m}").as_str()),
            Err(e) => GString::from(format!("BAKE_PNG: FAIL — {e}").as_str()),
        }
    }
}

/// Bake one chunk (face root, terrain ON) at `tile_res` and write the colour +
/// normal atlas tiles as PNGs. The atlas is a linear `tile_res*tile_res`-texel
/// buffer wrapped at `attr_w`, so its raw bytes reinterpret directly as a
/// `tile_res × tile_res` RGBA8 image (the unused `u+v>1` half stays as-cleared).
fn bake_png(out_dir: &str, tile_res: u32) -> Result<String, String> {
    use godot::classes::image::Format;
    use godot::classes::{DirAccess, Image, ProjectSettings};

    let frames = base_face_frames(RADIUS);
    let frame = &frames[FACE as usize];
    let root_bary =
        [Bary { wb: 0.0, wc: 0.0 }, Bary { wb: 1.0, wc: 0.0 }, Bary { wb: 0.0, wc: 1.0 }];
    let chunk = Chunk {
        id: ChunkId { face: FACE, depth: 0, path: 0 },
        bary: root_bary,
        corners: [
            frame.project_bary(root_bary[0].wb, root_bary[0].wc),
            frame.project_bary(root_bary[1].wb, root_bary[1].wc),
            frame.project_bary(root_bary[2].wb, root_bary[2].wc),
        ],
        level: 0,
    };
    let realize = vec![(0u32, chunk)];

    let mut rd = RenderingServer::singleton()
        .create_local_rendering_device()
        .ok_or("no local RenderingDevice")?;
    let mut gpu = ChunkGpuResources::new_compute_only(1, RES, tile_res, RADIUS);
    if !gpu.ensure_ready(&mut rd) {
        return Err("ensure_ready failed".into());
    }

    let terrain = assemble(&HeightGpu::default(), &TextureGpu::default()); // terrain ON
    let desc = pack_chunks(&frames, &realize, RES);
    gpu.upload_descs(&mut rd, &desc, 1);
    gpu.upload_params(&mut rd, 1, &terrain);
    let list = rd.compute_list_begin();
    gpu.record_bake(&mut rd, list, tile_res * tile_res);
    rd.compute_list_end();
    rd.submit();
    rd.sync();

    let color = rd.texture_get_data(gpu.color_atlas(), 0);
    let normal = rd.texture_get_data(gpu.normal_atlas(), 0);

    let want = (tile_res * tile_res * 4) as usize;
    if color.len() < want {
        return Err(format!("atlas data {} < {want} (tile_res {tile_res})", color.len()));
    }

    let abs_dir = ProjectSettings::singleton().globalize_path(out_dir).to_string();
    DirAccess::make_dir_recursive_absolute(&GString::from(abs_dir.as_str()));
    let tr = tile_res as i32;
    let save = |name: &str, data: &PackedByteArray| -> Result<(), String> {
        // Reinterpret the first tile_res*tile_res texels as a square image.
        let slice = PackedByteArray::from(&data.to_vec()[..want]);
        let img = Image::create_from_data(tr, tr, false, Format::RGBA8, &slice)
            .ok_or("Image::create_from_data failed")?;
        let path = format!("{abs_dir}/{name}_{tile_res}.png");
        let err = img.save_png(&GString::from(path.as_str()));
        if err != godot::global::Error::OK {
            return Err(format!("save_png {path} -> {err:?}"));
        }
        Ok(())
    };
    save("bake_color", &color)?;
    save("bake_normal", &normal)?;

    Ok(format!("wrote bake_color_{tile_res}.png + bake_normal_{tile_res}.png to {abs_dir}"))
}

const RES: u32 = 8;
const TILE_RES: u32 = 32;
const RADIUS: f32 = 1000.0;
const FACE: u8 = 7;

/// The 1–2 chunk fixture: chunk A is face 7's root triangle; chunk B is its
/// inverted centre child (non-trivial barycentric mix, exercising the
/// `wa*bary0 + wb*bary1 + wc*bary2` interpolation). Slots 0 and 1.
fn fixture() -> (Vec<FaceFrame>, Vec<(u32, Chunk)>) {
    let frames = base_face_frames(RADIUS);
    let frame = &frames[FACE as usize];

    let corner = |b: Bary| frame.project_bary(b.wb, b.wc);

    let root_bary = [Bary { wb: 0.0, wc: 0.0 }, Bary { wb: 1.0, wc: 0.0 }, Bary { wb: 0.0, wc: 1.0 }];
    let chunk_a = Chunk {
        id: ChunkId { face: FACE, depth: 0, path: 0 },
        bary: root_bary,
        corners: [corner(root_bary[0]), corner(root_bary[1]), corner(root_bary[2])],
        level: 0,
    };

    // Inverted centre child: corners are the edge midpoints of the root.
    let mid = |a: Bary, b: Bary| Bary { wb: (a.wb + b.wb) * 0.5, wc: (a.wc + b.wc) * 0.5 };
    let centre_bary = [
        mid(root_bary[0], root_bary[1]),
        mid(root_bary[1], root_bary[2]),
        mid(root_bary[2], root_bary[0]),
    ];
    let chunk_b = Chunk {
        id: ChunkId { face: FACE, depth: 1, path: 3 },
        bary: centre_bary,
        corners: [corner(centre_bary[0]), corner(centre_bary[1]), corner(centre_bary[2])],
        level: 1,
    };

    (frames, vec![(0u32, chunk_a), (1u32, chunk_b)])
}

fn read_f32(data: &PackedByteArray, byte: usize) -> f32 {
    f32::from_le_bytes([
        data.get(byte).unwrap(),
        data.get(byte + 1).unwrap(),
        data.get(byte + 2).unwrap(),
        data.get(byte + 3).unwrap(),
    ])
}

/// Realize the fixture with `terrain` and read back the vertex-pool positions
/// for every chunk: returns `(slot, i, j, gpu_position)` for each pool vertex.
fn realize_positions(
    gpu: &mut ChunkGpuResources,
    rd: &mut Gd<RenderingDevice>,
    frames: &[FaceFrame],
    realize: &[(u32, Chunk)],
    terrain: &TerrainGpu,
) -> Vec<(u32, u32, u32, Vector3)> {
    let vpc = verts_per_chunk(RES);
    let desc_bytes = pack_chunks(frames, realize, RES);
    gpu.upload_descs(rd, &desc_bytes, realize.len() as u32);
    gpu.upload_params(rd, realize.len() as u32, terrain);

    let list = rd.compute_list_begin();
    gpu.record_realize(rd, list, realize.len() as u32 * vpc);
    rd.compute_list_end();
    rd.submit();
    rd.sync();

    let raw = rd.buffer_get_data(gpu.pos_buf());
    let mut out = Vec::new();
    for (slot, _chunk) in realize {
        // Enumerate (i, j) in the canonical chunk_mesh order; L increments 0,1,2…
        let mut l = 0u32;
        for i in 0..=RES {
            for j in 0..=(RES - i) {
                let gv = slot * vpc + l;
                let b = gv as usize * 16;
                let pos = Vector3::new(read_f32(&raw, b), read_f32(&raw, b + 4), read_f32(&raw, b + 8));
                out.push((*slot, i, j, pos));
                l += 1;
            }
        }
    }
    out
}

// ── CEL-73 scatter-place verification ────────────────────────────────────────

const SCATTER_L: u8 = 3; // lattice level
const SCATTER_K: u32 = 2; // candidates per cell
const SCATTER_SEED: u32 = 7;

/// The scatter fixture: the realize pair (root + centre child — Subcells with
/// s=3 and s=2) plus a depth-5 chunk (Ancestor path with an in-chunk filter).
fn scatter_fixture() -> (Vec<FaceFrame>, Vec<(u32, Chunk)>) {
    use celestial_algo::scatter::descend_bary;
    let (frames, mut realize) = fixture();
    let frame = &frames[FACE as usize];
    let root =
        [Bary { wb: 0.0, wc: 0.0 }, Bary { wb: 1.0, wc: 0.0 }, Bary { wb: 0.0, wc: 1.0 }];
    let path: u64 = 0b11_01_00_10_01;
    let bary = descend_bary(root, path, 5);
    let corner = |b: Bary| frame.project_bary(b.wb, b.wc);
    realize.push((
        2u32,
        Chunk {
            id: ChunkId { face: FACE, depth: 5, path },
            bary,
            corners: [corner(bary[0]), corner(bary[1]), corner(bary[2])],
            level: 5,
        },
    ));
    (frames, realize)
}

/// CPU expectation for pool record `c` of a chunk: `Some(candidate)` when the
/// shader must write a valid record, `None` when it must write the sentinel.
/// Mirrors `ScatterPlace.slang`'s per-thread cell resolution exactly.
fn expected_candidate(
    id: ChunkId,
    bary: [Bary; 3],
    c: u32,
) -> Option<celestial_algo::scatter::CandidateBary> {
    use celestial_algo::scatter::{
        bary_point_in_triangle, candidate, cell_range, descend_bary, CellRange,
    };
    let root =
        [Bary { wb: 0.0, wc: 0.0 }, Bary { wb: 1.0, wc: 0.0 }, Bary { wb: 0.0, wc: 1.0 }];
    match cell_range(id, SCATTER_L) {
        CellRange::None => None,
        CellRange::Subcells { count } => {
            let cell_sub = c / SCATTER_K;
            let k = c % SCATTER_K;
            if cell_sub >= count {
                return None;
            }
            let levels = (SCATTER_L - id.depth) as u32;
            let cell = descend_bary(bary, cell_sub as u64, levels);
            let full_path = (id.path << (2 * levels)) | cell_sub as u64;
            Some(candidate(id.face, full_path, cell, k, SCATTER_SEED))
        }
        CellRange::Ancestor { path } => {
            if c >= SCATTER_K {
                return None;
            }
            let cell = descend_bary(root, path, SCATTER_L as u32);
            let cand = candidate(id.face, path, cell, c, SCATTER_SEED);
            bary_point_in_triangle((cand.wb, cand.wc), bary).then_some(cand)
        }
    }
}

/// The CPU reference transform for a candidate: gnomonic projection to the
/// sphere (terrain OFF ⇒ no displacement), radial-up basis with yaw + scale —
/// the exact math of `ScatterPlace.slang`'s tail.
fn expected_rows(
    frame: &FaceFrame,
    cand: &celestial_algo::scatter::CandidateBary,
) -> [f32; 12] {
    let wa = 1.0 - cand.wb - cand.wc;
    let p = frame.a * wa + frame.b * cand.wb + frame.c * cand.wc;
    let pos = p.normalized() * RADIUS;
    let up = pos.normalized();
    let upref =
        if up.y.abs() < 0.99 { Vector3::new(0.0, 1.0, 0.0) } else { Vector3::new(1.0, 0.0, 0.0) };
    let t0 = upref.cross(up).normalized();
    let b0 = up.cross(t0);
    let xn = t0 * cand.yaw.cos() + b0 * cand.yaw.sin();
    let zn = xn.cross(up);
    let (x, y, z) = (xn * cand.scale, up * cand.scale, zn * cand.scale);
    [x.x, y.x, z.x, pos.x, x.y, y.y, z.y, pos.y, x.z, y.z, z.z, pos.z]
}

fn scatter_check() -> Result<String, String> {
    use crate::gpu::chunk_gpu::ScatterConfig;
    use crate::scatter_descriptors::{pack_scatter_aux, pack_scatter_params};

    let (frames, realize) = scatter_fixture();
    let capacity = celestial_algo::scatter::capacity(SCATTER_K);

    let mut rd = RenderingServer::singleton()
        .create_local_rendering_device()
        .ok_or("no local RenderingDevice")?;
    let mut gpu = ChunkGpuResources::new_compute_only(realize.len() as u32, RES, TILE_RES, RADIUS);
    gpu.set_scatter_configs(vec![ScatterConfig {
        mm_rid: Rid::Invalid,
        capacity,
        max_instances: 1024,
    }]);
    if !gpu.ensure_ready(&mut rd) {
        return Err("ensure_ready failed (scatter compute-only build)".into());
    }

    let desc = pack_chunks(&frames, &realize, RES);
    gpu.upload_descs(&mut rd, &desc, realize.len() as u32);
    let terrain_off = TerrainGpu::zeroed(); // enabled == 0 ⇒ pure gnomonic sphere
    let params = pack_scatter_params(
        realize.len() as u32,
        capacity,
        SCATTER_K,
        SCATTER_L as u32,
        RADIUS,
        1.0,  // density (unused by place)
        1024, // max_instances (unused by place)
        SCATTER_SEED,
        0,   // vis_count (unused by place)
        0.0, // min_height (compact-side; place records the transform regardless)
        1.0, // max_height
        1.0, // base_scale (test's expected_rows assumes native scale)
        0.0, // surface_enabled: procedural route (no CPU heightmap)
        0.0, // surface_height_scale
        TILE_RES,
        &terrain_off,
    );
    let aux = pack_scatter_aux(&realize);
    gpu.upload_scatter(&mut rd, &aux, &[], &[params]);

    let list = rd.compute_list_begin();
    gpu.record_scatter_place(&mut rd, list, 0, realize.len() as u32);
    rd.compute_list_end();
    rd.submit();
    rd.sync();

    let raw = rd.buffer_get_data(gpu.scatter_pool_buf(0));

    let frame = &frames[FACE as usize];
    let pos_tol = 1.0e-4 * RADIUS;
    let basis_tol = 1.0e-3f32;
    let mut checked = 0usize;
    let mut valid = 0usize;
    let mut max_pos_err = 0.0f32;
    for (slot, chunk) in &realize {
        for c in 0..capacity {
            let base = ((slot * capacity + c) as usize) * 64;
            let rec: Vec<f32> = (0..16).map(|f| read_f32(&raw, base + f * 4)).collect();
            let expect = expected_candidate(chunk.id, chunk.bary, c);
            checked += 1;
            match expect {
                None => {
                    if rec[12] != 2.0 {
                        return Err(format!(
                            "slot {slot} c {c}: expected sentinel, got hash01 {}",
                            rec[12]
                        ));
                    }
                }
                Some(cand) => {
                    valid += 1;
                    if (rec[12] - cand.hash01).abs() > 1.0e-6 {
                        return Err(format!(
                            "slot {slot} c {c}: hash01 {} != CPU {}",
                            rec[12], cand.hash01
                        ));
                    }
                    let want = expected_rows(frame, &cand);
                    for (f, (&got, &w)) in rec[..12].iter().zip(want.iter()).enumerate() {
                        let tol = if f % 4 == 3 { pos_tol } else { basis_tol * cand.scale };
                        let err = (got - w).abs();
                        if !err.is_finite() || err > tol {
                            return Err(format!(
                                "slot {slot} c {c} row-float {f}: gpu {got} vs cpu {w} (err {err:.6} > tol {tol:.6})"
                            ));
                        }
                        if f % 4 == 3 {
                            max_pos_err = max_pos_err.max(err);
                        }
                    }
                }
            }
        }
    }
    Ok(format!(
        "{valid} valid / {checked} records match the CPU reference (gnomonic, \
         {} chunks incl. depth-5 ancestor path; max origin err {max_pos_err:.5}, tol {pos_tol:.3})",
        realize.len()
    ))
}

fn run_check() -> Result<String, String> {
    let (frames, realize) = fixture();
    let vpc = verts_per_chunk(RES);

    let mut rd = RenderingServer::singleton()
        .create_local_rendering_device()
        .ok_or("no local RenderingDevice")?;

    let mut gpu = ChunkGpuResources::new_compute_only(realize.len() as u32, RES, TILE_RES, RADIUS);
    if !gpu.ensure_ready(&mut rd) {
        return Err("ChunkGpuResources::ensure_ready failed (pipeline/pool build)".into());
    }

    // Also exercise the instance packer (no multimesh in compute-only mode, so
    // this is just a shape/size check that it produces the expected bytes).
    let slots: Vec<u32> = realize.iter().map(|(s, _)| *s).collect();
    let inst = pack_instances(&slots, &vec![1.0f32; slots.len()]);
    if inst.len() != realize.len() * 16 * 4 {
        return Err(format!("pack_instances size {} unexpected", inst.len()));
    }

    // ── Terrain OFF: positions must equal chunk_subvertex_base exactly. ──
    let terrain_off = TerrainGpu::zeroed(); // enabled == 0 → pure geometry
    let off = realize_positions(&mut gpu, &mut rd, &frames, &realize, &terrain_off);

    // Tolerance basis: 1e-4 * scale, scale ≥ radius.
    let tol = 1.0e-4 * frames[FACE as usize].edge_len().max(RADIUS);
    let mut max_err = 0.0f32;
    let mut worst = (0u32, 0u32, 0u32);
    for &(slot, i, j, pos) in &off {
        let chunk = realize.iter().find(|(s, _)| *s == slot).map(|(_, c)| c).unwrap();
        let frame = &frames[chunk.id.face as usize];
        let want = chunk_subvertex_base(frame, chunk, RES, i, j);
        let err = (pos - want).length();
        if !err.is_finite() {
            return Err(format!("slot {slot} (i={i},j={j}): non-finite position {pos}"));
        }
        if err > max_err {
            max_err = err;
            worst = (slot, i, j);
        }
    }
    let total = off.len();
    if max_err > tol {
        return Err(format!(
            "terrain-OFF max error {max_err:.6} > tol {tol:.6} at slot {} (i={},j={}); {total} verts",
            worst.0, worst.1, worst.2
        ));
    }

    // ── Terrain ON: every position within the [0.7r, 1.3r] envelope. ──
    let terrain_on = assemble(&HeightGpu::default(), &TextureGpu::default());
    let on = realize_positions(&mut gpu, &mut rd, &frames, &realize, &terrain_on);
    let (lo, hi) = (RADIUS * 0.7, RADIUS * 1.3);
    let mut env_bad = 0usize;
    let mut env_worst = 0.0f32;
    for &(_, _, _, pos) in &on {
        let l = pos.length();
        if !l.is_finite() || l < lo || l > hi {
            env_bad += 1;
            env_worst = env_worst.max((l - RADIUS).abs());
        }
    }


    if env_bad > 0 {
        return Err(format!(
            "terrain-ON envelope: {env_bad}/{} verts out of [{lo:.0},{hi:.0}] (worst |Δr|={env_worst:.1})",
            on.len()
        ));
    }

    Ok(format!(
        "terrain-OFF max error {max_err:.6} (tol {tol:.6}) over {total} verts \
         ({} chunks × {vpc} verts, res {RES}); terrain-ON {} verts within [{lo:.0},{hi:.0}]",
        realize.len(),
        on.len(),
    ))
}
