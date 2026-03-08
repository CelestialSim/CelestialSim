use godot::builtin::{PackedByteArray, Vector3};
use godot::classes::RdUniform;
use godot::classes::RenderingDevice;
use godot::obj::Gd;
use godot::prelude::load;

use crate::buffer_info::BufferInfo;

/// Creates a GPU storage buffer from typed data.
pub fn create_storage_buffer<T: bytemuck::Pod>(rd: &mut RenderingDevice, data: &[T]) -> BufferInfo {
    let bytes = bytemuck::cast_slice(data);
    create_storage_buffer_from_bytes(rd, bytes)
}

/// Creates a zeroed GPU storage buffer of the given byte length.
pub fn create_empty_storage_buffer(rd: &mut RenderingDevice, byte_length: u32) -> BufferInfo {
    let rid = rd.storage_buffer_create(byte_length);
    assert!(rid.is_valid(), "Failed to create empty storage buffer");
    rd.buffer_clear(rid, 0, byte_length);
    BufferInfo::new_storage(rid, byte_length, byte_length)
}

/// Creates a uniform buffer from a single value, with 16-byte alignment padding.
pub fn create_uniform_buffer<T: bytemuck::Pod>(rd: &mut RenderingDevice, value: &T) -> BufferInfo {
    let data_bytes = bytemuck::bytes_of(value);
    // Pad to 16-byte alignment as required by Godot's uniform buffer spec
    let padded_size = 16.max((data_bytes.len() + 15) / 16 * 16);
    let mut padded = vec![0u8; padded_size];
    padded[..data_bytes.len()].copy_from_slice(data_bytes);

    let mut pba = PackedByteArray::new();
    pba.extend(padded.iter().copied());
    let rid = rd
        .uniform_buffer_create_ex(padded_size as u32)
        .data(&pba)
        .done();
    assert!(rid.is_valid(), "Failed to create uniform buffer");
    BufferInfo::new_uniform(rid)
}

/// Loads a .slang shader resource, gets SPIR-V, creates pipeline, dispatches compute, and syncs.
pub fn dispatch_shader(
    rd: &mut RenderingDevice,
    shader_path: &str,
    buffers: &[&BufferInfo],
    threads: u32,
) {
    // Build uniforms array
    let uniforms: Vec<Gd<RdUniform>> = buffers
        .iter()
        .enumerate()
        .map(|(i, b)| b.get_uniform_with_binding(i as i32))
        .collect();

    // Load shader resource and get SPIR-V
    let mut shader_resource = load::<godot::classes::Resource>(shader_path);
    let spirv = shader_resource
        .call("get_spirv", &[])
        .to::<Gd<godot::classes::RdShaderSpirv>>();
    let compute_shader = rd.shader_create_from_spirv(&spirv);
    assert!(
        compute_shader.is_valid(),
        "Failed to create shader from SPIR-V"
    );

    let pipeline = rd.compute_pipeline_create(compute_shader);
    let compute_list = rd.compute_list_begin();
    rd.compute_list_bind_compute_pipeline(compute_list, pipeline);

    // Create uniform set from the array
    let mut uniform_array = godot::prelude::Array::<Gd<RdUniform>>::new();
    for u in uniforms {
        uniform_array.push(&u);
    }
    let uniform_set = rd.uniform_set_create(&uniform_array, compute_shader, 0);
    assert!(uniform_set.is_valid(), "Failed to create uniform set");
    rd.compute_list_bind_uniform_set(compute_list, uniform_set, 0);

    rd.compute_list_dispatch(compute_list, threads, 1, 1);
    rd.compute_list_end();
    rd.submit();
    rd.sync();
}

/// Reads a GPU buffer back to CPU and reinterprets as a Vec<T>.
pub fn convert_buffer_to_vec<T: bytemuck::Pod>(
    rd: &mut RenderingDevice,
    buffer: &BufferInfo,
) -> Vec<T> {
    let byte_data = rd
        .buffer_get_data_ex(buffer.rid)
        .offset_bytes(0)
        .size_bytes(buffer.filled_size)
        .done();
    let bytes = byte_data.as_slice();
    bytemuck::cast_slice(bytes).to_vec()
}

/// Reads a float4 GPU buffer and converts to Vec<Vector3> (discarding w).
pub fn convert_v4_buffer_to_vec3(rd: &mut RenderingDevice, buffer: &BufferInfo) -> Vec<Vector3> {
    let float_data: Vec<f32> = convert_buffer_to_vec(rd, buffer);
    let num_vectors = float_data.len() / 4;
    let mut result = Vec::with_capacity(num_vectors);
    for i in 0..num_vectors {
        result.push(Vector3::new(
            float_data[i * 4],
            float_data[i * 4 + 1],
            float_data[i * 4 + 2],
        ));
    }
    result
}

/// CPU prefix sum in-place.
///
/// Without invert: `[0,0,1,0,0,1,1,0,1,1]` → `[0,0,1,1,1,2,3,3,4,5]`
/// With invert: counts where the original value is 0 instead of non-zero.
/// `[1,0,1,1,0]` with invert → `[0,1,1,1,2]`
pub fn sum_array_in_place(arr: &mut [i32], invert: bool) {
    let mut sum: i32 = 0;
    for v in arr.iter_mut() {
        if invert {
            sum += if *v == 0 { 1 } else { 0 };
        } else {
            sum += *v;
        }
        *v = sum;
    }
}

// --- internal helper ---

fn create_storage_buffer_from_bytes(rd: &mut RenderingDevice, bytes: &[u8]) -> BufferInfo {
    let len = bytes.len() as u32;
    let mut pba = PackedByteArray::new();
    pba.extend(bytes.iter().copied());
    let rid = rd.storage_buffer_create_ex(len).data(&pba).done();
    assert!(rid.is_valid(), "Failed to create storage buffer");
    BufferInfo::new_storage(rid, len, len)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sum_array_in_place_basic() {
        // C# example: [0,0,1,0,0,1,1,0,1,1] → [0,0,1,1,1,2,3,3,4,5]
        let mut arr = vec![0, 0, 1, 0, 0, 1, 1, 0, 1, 1];
        sum_array_in_place(&mut arr, false);
        assert_eq!(arr, vec![0, 0, 1, 1, 1, 2, 3, 3, 4, 5]);
    }

    #[test]
    fn test_sum_array_in_place_all_ones() {
        let mut arr = vec![1, 1, 1, 1];
        sum_array_in_place(&mut arr, false);
        assert_eq!(arr, vec![1, 2, 3, 4]);
    }

    #[test]
    fn test_sum_array_in_place_all_zeros() {
        let mut arr = vec![0, 0, 0];
        sum_array_in_place(&mut arr, false);
        assert_eq!(arr, vec![0, 0, 0]);
    }

    #[test]
    fn test_sum_array_in_place_empty() {
        let mut arr: Vec<i32> = vec![];
        sum_array_in_place(&mut arr, false);
        assert!(arr.is_empty());
    }

    #[test]
    fn test_sum_array_in_place_from_plan() {
        // Plan example: [1,0,1,1,0] → [0,1,1,2,3]
        // Wait — let's check: with invert=false, prefix sum of [1,0,1,1,0]:
        // sum after each: 1, 1, 2, 3, 3  → that doesn't match the plan.
        // The plan says [1,0,1,1,0] -> [0,1,1,2,3] which looks like it's shifted by 1.
        // Looking at C# code more carefully: it does sum += arr[i] THEN arr[i] = sum.
        // So for [1,0,1,1,0]: 1,1,2,3,3 — inclusive prefix sum.
        // The plan's example might be wrong, let's match C# behavior.
        let mut arr = vec![1, 0, 1, 1, 0];
        sum_array_in_place(&mut arr, false);
        assert_eq!(arr, vec![1, 1, 2, 3, 3]);
    }

    #[test]
    fn test_sum_array_in_place_inverted() {
        // With invert: counts 0s instead. For [1,0,1,1,0]:
        // contributions: 0,1,0,0,1 → prefix sum: 0,1,1,1,2
        let mut arr = vec![1, 0, 1, 1, 0];
        sum_array_in_place(&mut arr, true);
        assert_eq!(arr, vec![0, 1, 1, 1, 2]);
    }

    #[test]
    fn test_sum_array_in_place_inverted_all_zeros() {
        // Invert of all zeros: every 0 contributes 1
        let mut arr = vec![0, 0, 0];
        sum_array_in_place(&mut arr, true);
        assert_eq!(arr, vec![1, 2, 3]);
    }

    #[test]
    fn test_sum_array_in_place_inverted_all_ones() {
        // Invert of all ones: every 1 contributes 0
        let mut arr = vec![1, 1, 1];
        sum_array_in_place(&mut arr, true);
        assert_eq!(arr, vec![0, 0, 0]);
    }
}
