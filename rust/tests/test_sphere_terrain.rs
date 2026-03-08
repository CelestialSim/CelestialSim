mod shader_test_utils;

use bytemuck;
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

#[test]
fn test_sphere_terrain_normalizes_to_radius() {
    // Skip if SPIR-V was not compiled (slangc missing)
    let spv_file = spv_path("SphereTerrain");
    if !spv_file.exists() {
        eprintln!(
            "Skipping test_sphere_terrain_normalizes_to_radius: {} not found (slangc not available)",
            spv_file.display()
        );
        return;
    }

    pollster::block_on(run_sphere_terrain_test());
}

async fn run_sphere_terrain_test() {
    let (device, queue) = match create_compute_device().await {
        Some(dq) => dq,
        None => {
            eprintln!("Skipping: no GPU with SPIRV_SHADER_PASSTHROUGH support");
            return;
        }
    };

    let shader_spirv = load_spv("SphereTerrain");

    // Input: two vertices at non-unit positions
    // WGSL array<vec3<f32>> uses 16-byte stride (vec3 aligned to 16 bytes).
    // Each vertex is [x, y, z, padding].
    let vertices_data: Vec<[f32; 4]> = vec![
        [2.0, 0.0, 0.0, 0.0],
        [0.0, 3.0, 0.0, 0.0],
        [0.0, 0.0, 4.0, 0.0],
        [1.0, 1.0, 1.0, 0.0],
    ];
    let n_verts: u32 = vertices_data.len() as u32;
    let radius: f32 = 5.0;

    // v_update_mask — shader checks but current code has mask check commented out, so contents don't matter
    let v_update_mask: Vec<i32> = vec![1; n_verts as usize];

    let vertices_buf = create_buffer_init(
        &device,
        "vertices",
        bytemuck::cast_slice(&vertices_data),
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    );

    let mask_buf = create_buffer_init(
        &device,
        "v_update_mask",
        bytemuck::cast_slice(&v_update_mask),
        wgpu::BufferUsages::STORAGE,
    );

    let radius_buf = create_buffer_init(
        &device,
        "radius",
        bytemuck::bytes_of(&radius),
        wgpu::BufferUsages::UNIFORM,
    );

    let n_verts_buf = create_buffer_init(
        &device,
        "n_verts",
        bytemuck::bytes_of(&n_verts),
        wgpu::BufferUsages::UNIFORM,
    );

    // Layout entries matching SphereTerrain.slang bindings
    let layout_entries = [
        // binding 0: RWStructuredBuffer<float3> vertices
        wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        },
        // binding 1: StructuredBuffer<int> v_update_mask
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
        // binding 2: ConstantBuffer<float> radius
        wgpu::BindGroupLayoutEntry {
            binding: 2,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        },
        // binding 3: ConstantBuffer<uint> n_verts
        wgpu::BindGroupLayoutEntry {
            binding: 3,
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
            resource: vertices_buf.as_entire_binding(),
        },
        wgpu::BindGroupEntry {
            binding: 1,
            resource: mask_buf.as_entire_binding(),
        },
        wgpu::BindGroupEntry {
            binding: 2,
            resource: radius_buf.as_entire_binding(),
        },
        wgpu::BindGroupEntry {
            binding: 3,
            resource: n_verts_buf.as_entire_binding(),
        },
    ];

    // SphereTerrain uses numthreads(12,1,1), 4 vertices -> 1 workgroup is enough
    let workgroups = (1, 1, 1);

    dispatch_compute(
        &device,
        &queue,
        &shader_spirv,
        &layout_entries,
        &bind_group_entries,
        workgroups,
    );

    // Read back — 16-byte stride per vec3 (padded)
    let result_bytes = read_buffer(&device, &queue, &vertices_buf).await;
    let result_verts: &[[f32; 4]] = bytemuck::cast_slice(&result_bytes);

    assert_eq!(result_verts.len(), n_verts as usize);

    for (i, v) in result_verts.iter().enumerate() {
        let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
        assert!(
            (len - radius).abs() < 0.01,
            "Vertex {} has length {}, expected {} (values: {:?})",
            i,
            len,
            radius,
            v
        );

        // Also check direction is preserved
        let orig = &vertices_data[i];
        let orig_len = (orig[0] * orig[0] + orig[1] * orig[1] + orig[2] * orig[2]).sqrt();
        if orig_len > 0.001 {
            let expected_dir = [orig[0] / orig_len, orig[1] / orig_len, orig[2] / orig_len];
            let actual_dir = [v[0] / len, v[1] / len, v[2] / len];
            for d in 0..3 {
                assert!(
                    (expected_dir[d] - actual_dir[d]).abs() < 0.01,
                    "Vertex {} direction mismatch on axis {}: expected {}, got {}",
                    i,
                    d,
                    expected_dir[d],
                    actual_dir[d]
                );
            }
        }
    }

    eprintln!("SphereTerrain test passed: all {} vertices normalized to radius {}", n_verts, radius);
}
