use wgpu::util::DeviceExt;

/// Create a wgpu device and queue suitable for compute shader testing.
/// Returns `None` if no suitable GPU adapter is available.
pub async fn create_compute_device() -> Option<(wgpu::Device, wgpu::Queue)> {
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
        backends: wgpu::Backends::all(),
        ..Default::default()
    });

    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        })
        .await?;

    let result = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: Some("compute-test-device"),
                ..Default::default()
            },
            None,
        )
        .await;

    match result {
        Ok(pair) => Some(pair),
        Err(e) => {
            eprintln!("Failed to create compute device: {}", e);
            None
        }
    }
}

/// Convert SPIR-V bytes to WGSL source using naga.
pub fn spirv_to_wgsl(spirv_bytes: &[u8]) -> String {
    let options = naga::front::spv::Options {
        adjust_coordinate_space: false,
        strict_capabilities: false,
        block_ctx_dump_prefix: None,
    };
    let module =
        naga::front::spv::parse_u8_slice(spirv_bytes, &options).expect("Failed to parse SPIR-V");

    let info = naga::valid::Validator::new(
        naga::valid::ValidationFlags::empty(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .expect("naga validation failed");

    let mut wgsl = String::new();
    let mut writer =
        naga::back::wgsl::Writer::new(&mut wgsl, naga::back::wgsl::WriterFlags::empty());
    writer.write(&module, &info).expect("Failed to write WGSL");
    wgsl
}

/// Create a GPU buffer initialised with the given byte data.
pub fn create_buffer_init(
    device: &wgpu::Device,
    label: &str,
    data: &[u8],
    usage: wgpu::BufferUsages,
) -> wgpu::Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents: data,
        usage,
    })
}

/// Dispatch a compute shader from raw SPIR-V bytes.
/// Internally converts SPIR-V to WGSL via naga for maximum compatibility.
///
/// `bind_group_layout_entries` describes the binding layout.
/// `bind_group_entries` provides the actual buffer bindings.
/// `workgroups` is (x, y, z) dispatch size.
pub fn dispatch_compute(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    shader_spirv: &[u8],
    bind_group_layout_entries: &[wgpu::BindGroupLayoutEntry],
    bind_group_entries: &[wgpu::BindGroupEntry],
    workgroups: (u32, u32, u32),
) {
    let wgsl_source = spirv_to_wgsl(shader_spirv);

    // naga may rename the entry point; find the @compute function name
    let entry_point = find_compute_entry_point(&wgsl_source)
        .expect("No @compute entry point found in generated WGSL");

    let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("compute-shader"),
        source: wgpu::ShaderSource::Wgsl(wgsl_source.into()),
    });

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("compute-bind-group-layout"),
        entries: bind_group_layout_entries,
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("compute-pipeline-layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("compute-pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader_module,
        entry_point: Some(&entry_point),
        compilation_options: Default::default(),
        cache: None,
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("compute-bind-group"),
        layout: &bind_group_layout,
        entries: bind_group_entries,
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("compute-encoder"),
    });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("compute-pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(workgroups.0, workgroups.1, workgroups.2);
    }

    queue.submit(std::iter::once(encoder.finish()));
}

/// Read the contents of a GPU buffer back to the CPU.
pub async fn read_buffer(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    buffer: &wgpu::Buffer,
) -> Vec<u8> {
    let size = buffer.size();
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("staging-read"),
        size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("read-encoder"),
    });
    encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, size);
    queue.submit(std::iter::once(encoder.finish()));

    let slice = staging.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        tx.send(result).unwrap();
    });
    device.poll(wgpu::Maintain::Wait);
    rx.recv().unwrap().expect("Failed to map staging buffer");

    let data = slice.get_mapped_range().to_vec();
    staging.unmap();
    data
}

/// Find the compute entry point name in WGSL source generated by naga.
/// naga wraps the original entry point and creates a `main` function
/// with the `@compute` attribute, possibly on separate lines.
fn find_compute_entry_point(wgsl: &str) -> Option<String> {
    let lines: Vec<&str> = wgsl.lines().collect();
    for (i, line) in lines.iter().enumerate() {
        let trimmed = line.trim();
        if trimmed.contains("@compute") {
            // Check if fn is on the same line
            if let Some(name) = extract_fn_name(trimmed) {
                return Some(name);
            }
            // Check next line(s) for the fn declaration
            for j in (i + 1)..lines.len().min(i + 3) {
                if let Some(name) = extract_fn_name(lines[j].trim()) {
                    return Some(name);
                }
            }
        }
    }
    None
}

fn extract_fn_name(line: &str) -> Option<String> {
    if let Some(fn_pos) = line.find("fn ") {
        let after_fn = &line[fn_pos + 3..];
        if let Some(paren) = after_fn.find('(') {
            return Some(after_fn[..paren].trim().to_string());
        }
    }
    None
}
