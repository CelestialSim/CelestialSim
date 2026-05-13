// SPIR-V integration test for the variant filter in `ScatterPlacement.slang`.
//
// The shader hashes the triangle centroid direction (stable across LOD
// compaction) to assign each triangle to exactly one variant. We dispatch
// the compiled shader N times (one per `variant_id`) against a synthetic
// mesh where each triangle has a unique direction, then assert:
//
//   - sum of per-variant emit counts == N_TRIS  (complete + disjoint partition)
//   - re-running with the same params yields identical counts (determinism)
//   - variant_count = 1 emits every candidate
//   - variant_count > 1 spreads emissions across all variants (no single-bucket)
//
// We don't predict per-triangle bucket assignments in Rust: the shader's
// `hash3` relies on `sin()` whose CPU/GPU implementations diverge enough
// that a Rust mirror wouldn't give bit-identical results.

mod shader_test_utils;

use shader_test_utils::*;
use std::path::PathBuf;

fn spv_path(name: &str) -> PathBuf {
    let out_dir = std::env::var("OUT_DIR").expect("OUT_DIR not set — run via `cargo test`");
    PathBuf::from(out_dir).join(format!("{}.spv", name))
}

// 64 bytes: must match `ScatterParams` in `rust/src/layers/scatter.rs` and
// the Slang struct in `ScatterPlacement.slang`. We hand-roll a layout-equal
// copy here so the test crate doesn't depend on the lib's internal type.
#[repr(C, align(16))]
#[derive(Clone, Copy, Debug)]
#[allow(dead_code)]
struct ScatterParamsTest {
    planet_radius: f32,
    seed: u32,
    subdivision_level: i32,
    noise_strength: f32,

    height_min: f32,
    height_max: f32,
    albedo_tolerance: f32,
    noise_dim: u32,

    albedo_target: [f32; 3],
    _pad: f32,

    variant_id: u32,
    variant_count: u32,
    _pad1: u32,
    _pad2: u32,
}

unsafe impl bytemuck::Zeroable for ScatterParamsTest {}
unsafe impl bytemuck::Pod for ScatterParamsTest {}

// Mirrors `TerrainParams` in `rust/src/texture_gen.rs`.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
#[allow(dead_code)]
struct TerrainParamsTest {
    height_tiles: f32,
    height_octaves: i32,
    height_amp: f32,
    height_gain: f32,
    height_lacunarity: f32,
    erosion_tiles: f32,
    erosion_octaves: i32,
    erosion_gain: f32,
    erosion_lacunarity: f32,
    erosion_slope_strength: f32,
    erosion_branch_strength: f32,
    erosion_strength: f32,
    water_height: f32,
    _pad: [f32; 3],
}

unsafe impl bytemuck::Zeroable for TerrainParamsTest {}
unsafe impl bytemuck::Pod for TerrainParamsTest {}

fn terrain_params_default() -> TerrainParamsTest {
    TerrainParamsTest {
        height_tiles: 3.0,
        height_octaves: 3,
        height_amp: 0.25,
        height_gain: 0.1,
        height_lacunarity: 2.0,
        erosion_tiles: 4.0,
        erosion_octaves: 5,
        erosion_gain: 0.5,
        erosion_lacunarity: 1.8,
        erosion_slope_strength: 3.0,
        erosion_branch_strength: 3.0,
        erosion_strength: 0.04,
        water_height: 0.45,
        _pad: [0.0; 3],
    }
}

const N_TRIS: u32 = 256;
const SUBDIVISION_LEVEL: i32 = 0;
// Keep small: the shader's `hash3(dir * 137.0 + float(seed))` uses `sin()`
// which loses precision above ~1e6, so a huge seed would dominate the dot
// product and collapse every direction into the same hash bucket. Same
// convention as the yaw seed; the demo scenes use values like 8, 20, 42.
const SEED: u32 = 42;

/// Build a synthetic mesh with N_TRIS unique vertex directions distributed
/// on the unit sphere via a Fibonacci spiral. Each triangle references its
/// own vertex three times so the centroid direction equals the vertex
/// direction (post-normalize). This gives the variant filter a different
/// `dir` for every triangle, exercising the spatial hash.
fn build_inputs() -> (Vec<[f32; 4]>, Vec<[i32; 4]>, Vec<i32>, Vec<i32>) {
    let golden_angle = std::f32::consts::PI * (3.0 - (5.0_f32).sqrt());
    let n = N_TRIS as f32;
    let v_pos: Vec<[f32; 4]> = (0..N_TRIS)
        .map(|i| {
            let y = 1.0 - (i as f32 / (n - 1.0)) * 2.0; // [1, -1]
            let r = (1.0 - y * y).max(0.0).sqrt();
            let theta = golden_angle * i as f32;
            [r * theta.cos(), y, r * theta.sin(), 0.0]
        })
        .collect();

    let t_abc: Vec<[i32; 4]> = (0..N_TRIS).map(|i| [i as i32, i as i32, i as i32, 0]).collect();
    let t_lv: Vec<i32> = vec![SUBDIVISION_LEVEL; N_TRIS as usize];
    let t_deactivated: Vec<i32> = vec![0; N_TRIS as usize];

    (v_pos, t_abc, t_lv, t_deactivated)
}

fn make_scatter_params(variant_id: u32, variant_count: u32) -> ScatterParamsTest {
    ScatterParamsTest {
        planet_radius: 1.0,
        seed: SEED,
        subdivision_level: SUBDIVISION_LEVEL,
        // noise_strength = 0 → blue-noise threshold collapses to 0 so
        // `density > threshold` always passes when density > 0.
        noise_strength: 0.0,
        // Wide-open height window so the smoothstep gates evaluate to ~1
        // regardless of the actual sampled height.
        height_min: -1e6,
        height_max: 1e6,
        // >= sqrt(3) disables albedo check.
        albedo_tolerance: 2.0,
        // noise_dim = 0 → sample_blue_noise short-circuits to 0.5, but
        // noise_strength = 0 kills the threshold anyway.
        noise_dim: 0,
        albedo_target: [0.0, 0.0, 0.0],
        _pad: 0.0,
        variant_id,
        variant_count,
        _pad1: 0,
        _pad2: 0,
    }
}

/// Layout for the 10 bindings, mirroring `ScatterPlacement.slang`.
fn binding_layout() -> Vec<wgpu::BindGroupLayoutEntry> {
    use wgpu::*;
    let uniform = |binding: u32| BindGroupLayoutEntry {
        binding,
        visibility: ShaderStages::COMPUTE,
        ty: BindingType::Buffer {
            ty: BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    };
    let storage_ro = |binding: u32| BindGroupLayoutEntry {
        binding,
        visibility: ShaderStages::COMPUTE,
        ty: BindingType::Buffer {
            ty: BufferBindingType::Storage { read_only: true },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    };
    let storage_rw = |binding: u32| BindGroupLayoutEntry {
        binding,
        visibility: ShaderStages::COMPUTE,
        ty: BindingType::Buffer {
            ty: BufferBindingType::Storage { read_only: false },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    };
    vec![
        uniform(0),    // scatter_params
        uniform(1),    // terrain_params
        storage_rw(2), // out_transforms
        storage_rw(3), // out_counter
        storage_ro(4), // v_pos
        storage_ro(5), // t_abc
        storage_ro(6), // t_lv
        storage_ro(7), // t_deactivated
        uniform(8),    // n_tris
        storage_ro(9), // blue_noise
    ]
}

#[test]
fn variant_filter_partitions_triangles_into_disjoint_sets() {
    let spv_file = spv_path("ScatterPlacement");
    if !spv_file.exists() {
        eprintln!(
            "Skipping variant_filter_partitions_triangles_into_disjoint_sets: {} not found (slangc not available)",
            spv_file.display()
        );
        return;
    }

    let Some(counts) = pollster::block_on(run_variants(3)) else {
        eprintln!("Skipping: no GPU device or shader execution failed");
        return;
    };

    let total: u32 = counts.iter().sum();
    assert_eq!(
        total, N_TRIS,
        "sum of per-variant emissions ({total}) != N_TRIS ({N_TRIS}) — partition is incomplete or overlapping. Counts: {counts:?}"
    );

    for (i, &c) in counts.iter().enumerate() {
        assert!(
            c > 0,
            "variant {i} emitted 0 triangles with 256 inputs — hash distribution looks broken. Counts: {counts:?}"
        );
    }

    // Determinism: re-run with the same params, expect identical counts.
    let Some(counts_again) = pollster::block_on(run_variants(3)) else {
        return;
    };
    assert_eq!(
        counts, counts_again,
        "non-deterministic shader output across runs: {counts:?} vs {counts_again:?}"
    );
}

#[test]
fn variant_count_one_emits_all_candidates() {
    let spv_file = spv_path("ScatterPlacement");
    if !spv_file.exists() {
        eprintln!(
            "Skipping variant_count_one_emits_all_candidates: {} not found (slangc not available)",
            spv_file.display()
        );
        return;
    }

    let Some(counts) = pollster::block_on(run_variants(1)) else {
        eprintln!("Skipping: no GPU device or shader execution failed");
        return;
    };
    assert_eq!(counts.len(), 1, "variant_count = 1 should produce one entry");
    assert_eq!(
        counts[0], N_TRIS,
        "variant_count = 1 should emit every candidate, got {}",
        counts[0]
    );
}

/// Dispatch the shader once per `variant_id` in `0..variant_count`, returning
/// the per-variant emit counts read from the atomic counter. Returns `None`
/// when no GPU device is available.
async fn run_variants(variant_count: u32) -> Option<Vec<u32>> {
    let (device, queue) = match create_compute_device().await {
        Some(dq) => dq,
        None => return None,
    };

    let shader_spirv = match std::fs::read(spv_path("ScatterPlacement")) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("Cannot read ScatterPlacement.spv: {e}");
            return None;
        }
    };

    let (v_pos, t_abc, t_lv, t_deactivated) = build_inputs();
    let n_tris = N_TRIS;

    let v_pos_buf = create_buffer_init(
        &device,
        "v_pos",
        bytemuck::cast_slice(&v_pos),
        wgpu::BufferUsages::STORAGE,
    );
    let t_abc_buf = create_buffer_init(
        &device,
        "t_abc",
        bytemuck::cast_slice(&t_abc),
        wgpu::BufferUsages::STORAGE,
    );
    let t_lv_buf = create_buffer_init(
        &device,
        "t_lv",
        bytemuck::cast_slice(&t_lv),
        wgpu::BufferUsages::STORAGE,
    );
    let t_deact_buf = create_buffer_init(
        &device,
        "t_deactivated",
        bytemuck::cast_slice(&t_deactivated),
        wgpu::BufferUsages::STORAGE,
    );
    let n_tris_buf = create_buffer_init(
        &device,
        "n_tris",
        bytemuck::bytes_of(&n_tris),
        wgpu::BufferUsages::UNIFORM,
    );
    let noise_buf = create_buffer_init(
        &device,
        "blue_noise",
        bytemuck::cast_slice(&[0u32; 4]),
        wgpu::BufferUsages::STORAGE,
    );
    let terrain = terrain_params_default();
    let terrain_buf = create_buffer_init(
        &device,
        "terrain_params",
        bytemuck::bytes_of(&terrain),
        wgpu::BufferUsages::UNIFORM,
    );

    let out_size_bytes = (n_tris as u64) * 12 * 4;
    let out_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("out_transforms"),
        size: out_size_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let mut emitted_counts = Vec::with_capacity(variant_count as usize);

    for variant_id in 0..variant_count {
        let params = make_scatter_params(variant_id, variant_count);
        let params_buf = create_buffer_init(
            &device,
            "scatter_params",
            bytemuck::bytes_of(&params),
            wgpu::BufferUsages::UNIFORM,
        );
        let counter_buf = create_buffer_init(
            &device,
            "out_counter",
            bytemuck::bytes_of(&0u32),
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );

        let entries = [
            wgpu::BindGroupEntry {
                binding: 0,
                resource: params_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: terrain_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: out_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: counter_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: v_pos_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: t_abc_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 6,
                resource: t_lv_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 7,
                resource: t_deact_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 8,
                resource: n_tris_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 9,
                resource: noise_buf.as_entire_binding(),
            },
        ];

        let workgroups = (n_tris.div_ceil(64), 1, 1);
        dispatch_compute(
            &device,
            &queue,
            &shader_spirv,
            &binding_layout(),
            &entries,
            workgroups,
        );

        let bytes = read_buffer(&device, &queue, &counter_buf).await;
        let count = u32::from_ne_bytes(bytes[0..4].try_into().unwrap());
        emitted_counts.push(count);
    }

    Some(emitted_counts)
}
