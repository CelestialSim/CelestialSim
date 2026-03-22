mod shader_test_utils;

use bytemuck;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use shader_test_utils::*;
use std::path::PathBuf;

fn spv_path(name: &str) -> PathBuf {
    let out_dir = std::env::var("OUT_DIR").expect("OUT_DIR not set — run via `cargo test`");
    PathBuf::from(out_dir).join(format!("{}.spv", name))
}

fn load_spv(name: &str) -> Vec<u8> {
    let path = spv_path(name);
    std::fs::read(&path).unwrap_or_else(|e| {
        panic!(
            "Cannot read compiled shader {}: {}. Was slangc available during build?",
            path.display(),
            e
        )
    })
}

fn cpu_visible_mask(t_divided: &[i32], t_deactivated: &[i32]) -> Vec<i32> {
    t_divided
        .iter()
        .zip(t_deactivated.iter())
        .map(|(&div, &deact)| if div == 0 && deact == 0 { 1 } else { 0 })
        .collect()
}

fn cpu_prefix_sum(input: &[i32]) -> Vec<i32> {
    let mut result = Vec::with_capacity(input.len());
    let mut sum = 0i32;
    for &v in input {
        sum += v;
        result.push(sum);
    }
    result
}

async fn run_compute_visible_prefix_sum(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    spirv: &[u8],
    t_divided: &[i32],
    t_deactivated: &[i32],
) -> (Vec<i32>, Vec<i32>) {
    let n_tris: u32 = t_divided.len() as u32;

    let divided_buf = create_buffer_init(
        device,
        "t_divided",
        bytemuck::cast_slice(t_divided),
        wgpu::BufferUsages::STORAGE,
    );

    let deactivated_buf = create_buffer_init(
        device,
        "t_deactivated",
        bytemuck::cast_slice(t_deactivated),
        wgpu::BufferUsages::STORAGE,
    );

    let visible_mask_data = vec![0i32; n_tris as usize];
    let visible_mask_buf = create_buffer_init(
        device,
        "visible_mask",
        bytemuck::cast_slice(&visible_mask_data),
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    );

    let visible_prefix_data = vec![0i32; n_tris as usize];
    let visible_prefix_buf = create_buffer_init(
        device,
        "visible_prefix",
        bytemuck::cast_slice(&visible_prefix_data),
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    );

    let n_tris_buf = create_buffer_init(
        device,
        "n_tris",
        bytemuck::bytes_of(&n_tris),
        wgpu::BufferUsages::UNIFORM,
    );

    let layout_entries = [
        wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        },
        wgpu::BindGroupLayoutEntry {
            binding: 1,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        },
        wgpu::BindGroupLayoutEntry {
            binding: 2,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        },
        wgpu::BindGroupLayoutEntry {
            binding: 3,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        },
        wgpu::BindGroupLayoutEntry {
            binding: 4,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        },
    ];

    let bind_group_entries = [
        wgpu::BindGroupEntry {
            binding: 0,
            resource: divided_buf.as_entire_binding(),
        },
        wgpu::BindGroupEntry {
            binding: 1,
            resource: deactivated_buf.as_entire_binding(),
        },
        wgpu::BindGroupEntry {
            binding: 2,
            resource: visible_mask_buf.as_entire_binding(),
        },
        wgpu::BindGroupEntry {
            binding: 3,
            resource: visible_prefix_buf.as_entire_binding(),
        },
        wgpu::BindGroupEntry {
            binding: 4,
            resource: n_tris_buf.as_entire_binding(),
        },
    ];

    dispatch_compute(
        device,
        queue,
        spirv,
        &layout_entries,
        &bind_group_entries,
        (1, 1, 1),
    );

    let mask_bytes = read_buffer(device, queue, &visible_mask_buf).await;
    let mask: &[i32] = bytemuck::cast_slice(&mask_bytes);

    let prefix_bytes = read_buffer(device, queue, &visible_prefix_buf).await;
    let prefix: &[i32] = bytemuck::cast_slice(&prefix_bytes);

    (mask.to_vec(), prefix.to_vec())
}

#[test]
fn test_all_visible() {
    let spv_file = spv_path("ComputeVisiblePrefixSum");
    if !spv_file.exists() {
        eprintln!("Skipping: {} not found", spv_file.display());
        return;
    }
    pollster::block_on(async {
        let (device, queue) = match create_compute_device().await {
            Some(dq) => dq,
            None => {
                eprintln!("Skipping: no GPU available");
                return;
            }
        };
        let spirv = load_spv("ComputeVisiblePrefixSum");
        let n = 1024;
        let t_divided = vec![0i32; n];
        let t_deactivated = vec![0i32; n];
        let (mask, prefix) =
            run_compute_visible_prefix_sum(&device, &queue, &spirv, &t_divided, &t_deactivated)
                .await;
        assert_eq!(mask.len(), n);
        assert_eq!(prefix.len(), n);
        for i in 0..n {
            assert_eq!(mask[i], 1, "Expected mask[{}] = 1, got {}", i, mask[i]);
            assert_eq!(
                prefix[i],
                (i + 1) as i32,
                "Expected prefix[{}] = {}, got {}",
                i,
                i + 1,
                prefix[i]
            );
        }
    });
}

#[test]
fn test_none_visible_divided() {
    let spv_file = spv_path("ComputeVisiblePrefixSum");
    if !spv_file.exists() {
        eprintln!("Skipping: {} not found", spv_file.display());
        return;
    }
    pollster::block_on(async {
        let (device, queue) = match create_compute_device().await {
            Some(dq) => dq,
            None => {
                eprintln!("Skipping: no GPU available");
                return;
            }
        };
        let spirv = load_spv("ComputeVisiblePrefixSum");
        let n = 1024;
        let t_divided = vec![1i32; n];
        let t_deactivated = vec![0i32; n];
        let (mask, prefix) =
            run_compute_visible_prefix_sum(&device, &queue, &spirv, &t_divided, &t_deactivated)
                .await;
        assert_eq!(mask.len(), n);
        assert_eq!(prefix.len(), n);
        for i in 0..n {
            assert_eq!(mask[i], 0, "Expected mask[{}] = 0, got {}", i, mask[i]);
            assert_eq!(
                prefix[i], 0,
                "Expected prefix[{}] = 0, got {}",
                i, prefix[i]
            );
        }
    });
}

#[test]
fn test_none_visible_deactivated() {
    let spv_file = spv_path("ComputeVisiblePrefixSum");
    if !spv_file.exists() {
        eprintln!("Skipping: {} not found", spv_file.display());
        return;
    }
    pollster::block_on(async {
        let (device, queue) = match create_compute_device().await {
            Some(dq) => dq,
            None => {
                eprintln!("Skipping: no GPU available");
                return;
            }
        };
        let spirv = load_spv("ComputeVisiblePrefixSum");
        let n = 1024;
        let t_divided = vec![0i32; n];
        let t_deactivated = vec![1i32; n];
        let (mask, prefix) =
            run_compute_visible_prefix_sum(&device, &queue, &spirv, &t_divided, &t_deactivated)
                .await;
        assert_eq!(mask.len(), n);
        assert_eq!(prefix.len(), n);
        for i in 0..n {
            assert_eq!(mask[i], 0, "Expected mask[{}] = 0, got {}", i, mask[i]);
            assert_eq!(
                prefix[i], 0,
                "Expected prefix[{}] = 0, got {}",
                i, prefix[i]
            );
        }
    });
}

#[test]
fn test_mixed() {
    let spv_file = spv_path("ComputeVisiblePrefixSum");
    if !spv_file.exists() {
        eprintln!("Skipping: {} not found", spv_file.display());
        return;
    }
    pollster::block_on(async {
        let (device, queue) = match create_compute_device().await {
            Some(dq) => dq,
            None => {
                eprintln!("Skipping: no GPU available");
                return;
            }
        };
        let spirv = load_spv("ComputeVisiblePrefixSum");
        let t_divided = vec![0i32, 1, 0, 0, 1];
        let t_deactivated = vec![0i32, 0, 1, 0, 0];
        let expected_mask = vec![1i32, 0, 0, 1, 0];
        let expected_prefix = vec![1i32, 1, 1, 2, 2];
        let (mask, prefix) =
            run_compute_visible_prefix_sum(&device, &queue, &spirv, &t_divided, &t_deactivated)
                .await;
        assert_eq!(
            mask, expected_mask,
            "Mask mismatch: {:?} vs {:?}",
            mask, expected_mask
        );
        assert_eq!(
            prefix, expected_prefix,
            "Prefix mismatch: {:?} vs {:?}",
            prefix, expected_prefix
        );
    });
}

#[test]
fn test_fuzz() {
    let spv_file = spv_path("ComputeVisiblePrefixSum");
    if !spv_file.exists() {
        eprintln!("Skipping: {} not found", spv_file.display());
        return;
    }
    pollster::block_on(async {
        let (device, queue) = match create_compute_device().await {
            Some(dq) => dq,
            None => {
                eprintln!("Skipping: no GPU available");
                return;
            }
        };
        let spirv = load_spv("ComputeVisiblePrefixSum");

        let seed: u64 = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;
        eprintln!("Fuzz seed: {} (reproduce with FUZZ_SEED={})", seed, seed);
        let mut rng = SmallRng::seed_from_u64(
            std::env::var("FUZZ_SEED")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(seed),
        );

        for iteration in 0..20 {
            let size = rng.random_range(1..=50_000usize);
            let t_divided: Vec<i32> = (0..size)
                .map(|_| if rng.random_bool(0.5) { 1 } else { 0 })
                .collect();
            let t_deactivated: Vec<i32> = (0..size)
                .map(|_| if rng.random_bool(0.3) { 1 } else { 0 })
                .collect();
            let expected_mask = cpu_visible_mask(&t_divided, &t_deactivated);
            let expected_prefix = cpu_prefix_sum(&expected_mask);

            let (mask, prefix) =
                run_compute_visible_prefix_sum(&device, &queue, &spirv, &t_divided, &t_deactivated)
                    .await;
            assert_eq!(
                mask, expected_mask,
                "Fuzz iteration {} (size={}): mask mismatch",
                iteration, size
            );
            assert_eq!(
                prefix, expected_prefix,
                "Fuzz iteration {} (size={}): prefix mismatch",
                iteration, size
            );
            eprintln!("  fuzz iteration {} passed (size={})", iteration, size);
        }
    });
}
