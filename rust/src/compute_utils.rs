use godot::builtin::{Callable, PackedByteArray, Variant, Vector3};
use godot::classes::RdUniform;
use godot::classes::{RenderingDevice, RenderingServer};
use godot::obj::{Gd, NewGd, Singleton};
use godot::prelude::{load, Array, Rid};

use std::sync::Mutex;

use crate::buffer_info::BufferInfo;

// ---------------------------------------------------------------------------
// Render-thread dispatch helpers (mirrors C# CallOnRenderThread pattern)
// ---------------------------------------------------------------------------

/// Newtype to send `Gd<RenderingDevice>` across threads.
/// SAFETY: The wrapped RD is a local RenderingDevice created via
/// `create_local_rendering_device()`. Actual RD method calls only happen on
/// the render thread (via `call_on_render_thread`), so there is no concurrent
/// unsynchronized access.
pub(crate) struct RdSend(pub Gd<RenderingDevice>);
unsafe impl Send for RdSend {}

/// Schedules `f` to run on the render thread (fire-and-forget).
pub fn on_render_thread(f: impl FnOnce() + Send + 'static) {
    let f = Mutex::new(Some(f));
    let callable = Callable::from_sync_fn("rt_fire", move |_args: &[&Variant]| {
        if let Some(f) = f.lock().unwrap().take() {
            f();
        }
    });
    RenderingServer::singleton().call_on_render_thread(&callable);
}

/// Schedules `f` to run on the render thread and blocks until the result is
/// available.  Mirrors C#'s `CallOnRenderThread` + `ManualResetEventSlim`.
pub fn on_render_thread_sync<T: Send + 'static>(f: impl FnOnce() -> T + Send + 'static) -> T {
    let (tx, rx) = std::sync::mpsc::sync_channel(1);
    on_render_thread(move || {
        let result = f();
        let _ = tx.send(result);
    });
    rx.recv().expect("render thread work failed")
}

/// Frees a GPU RID on the render thread (fire-and-forget).
pub fn free_rid_on_render_thread(rd: &mut Gd<RenderingDevice>, rid: Rid) {
    let rd_send = RdSend(rd.clone());
    on_render_thread(move || {
        let mut rd = rd_send; // force whole-struct capture (Rust 2021)
        rd.0.free_rid(rid);
    });
}

/// Clears a GPU buffer on the render thread (fire-and-forget).
pub fn buffer_clear_on_render_thread(
    rd: &mut Gd<RenderingDevice>,
    rid: Rid,
    offset: u32,
    size: u32,
) {
    let rd_send = RdSend(rd.clone());
    on_render_thread(move || {
        let mut rd = rd_send; // force whole-struct capture (Rust 2021)
        rd.0.buffer_clear(rid, offset, size);
    });
}

/// Creates a GPU storage buffer from typed data.
pub fn create_storage_buffer<T: bytemuck::Pod>(
    rd: &mut Gd<RenderingDevice>,
    data: &[T],
) -> BufferInfo {
    let bytes = bytemuck::cast_slice(data);
    create_storage_buffer_from_bytes(rd, bytes)
}

/// Creates a zeroed GPU storage buffer of the given byte length.
pub fn create_empty_storage_buffer(rd: &mut Gd<RenderingDevice>, byte_length: u32) -> BufferInfo {
    let rid = rd.storage_buffer_create(byte_length);
    assert!(rid.is_valid(), "Failed to create empty storage buffer");
    buffer_clear_on_render_thread(rd, rid, 0, byte_length);
    BufferInfo::new_storage(rid, byte_length, byte_length)
}

/// Creates a uniform buffer from a single value, with 16-byte alignment padding.
pub fn create_uniform_buffer<T: bytemuck::Pod>(
    rd: &mut Gd<RenderingDevice>,
    value: &T,
) -> BufferInfo {
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
/// The entire dispatch runs on the render thread (fire-and-forget).
pub fn dispatch_shader(
    rd: &mut Gd<RenderingDevice>,
    shader_path: &str,
    buffers: &[&BufferInfo],
    threads: u32,
) {
    // Extract plain-data info (Send-safe) to build uniforms inside the closure
    let buffer_descs: Vec<(Rid, crate::buffer_info::BufferType)> =
        buffers.iter().map(|b| (b.rid, b.buffer_type)).collect();

    let rd_send = RdSend(rd.clone());
    let shader_path = shader_path.to_string();

    on_render_thread(move || {
        let mut rd = rd_send; // force whole-struct capture (Rust 2021)
        let mut shader_resource = load::<godot::classes::Resource>(&shader_path);
        let spirv = shader_resource
            .call("get_spirv", &[])
            .to::<Gd<godot::classes::RdShaderSpirv>>();
        let compute_shader = rd.0.shader_create_from_spirv(&spirv);
        assert!(
            compute_shader.is_valid(),
            "Failed to create shader from SPIR-V"
        );

        let pipeline = rd.0.compute_pipeline_create(compute_shader);
        let compute_list = rd.0.compute_list_begin();
        rd.0.compute_list_bind_compute_pipeline(compute_list, pipeline);

        let mut uniform_array = Array::<Gd<RdUniform>>::new();
        for (i, (rid, btype)) in buffer_descs.into_iter().enumerate() {
            let mut uniform = RdUniform::new_gd();
            uniform.set_uniform_type(btype.to_uniform_type());
            uniform.set_binding(i as i32);
            uniform.add_id(rid);
            uniform_array.push(&uniform);
        }
        let uniform_set = rd.0.uniform_set_create(&uniform_array, compute_shader, 0);
        assert!(uniform_set.is_valid(), "Failed to create uniform set");
        rd.0.compute_list_bind_uniform_set(compute_list, uniform_set, 0);

        rd.0.compute_list_dispatch(compute_list, threads, 1, 1);
        rd.0.compute_list_end();
        rd.0.submit();
        rd.0.sync();

        // Free transient GPU objects
        rd.0.free_rid(uniform_set);
        rd.0.free_rid(pipeline);
        rd.0.free_rid(compute_shader);
    });
}

/// Reads a GPU buffer back to CPU and reinterprets as a Vec<T>.
/// Blocks until the render thread completes the read.
pub fn convert_buffer_to_vec<T: bytemuck::Pod + Send>(
    rd: &mut Gd<RenderingDevice>,
    buffer: &BufferInfo,
) -> Vec<T> {
    let rd_send = RdSend(rd.clone());
    let rid = buffer.rid;
    let filled_size = buffer.filled_size;

    on_render_thread_sync(move || {
        let mut rd = rd_send; // force whole-struct capture (Rust 2021)
        let byte_data =
            rd.0.buffer_get_data_ex(rid)
                .offset_bytes(0)
                .size_bytes(filled_size)
                .done();
        let bytes = byte_data.as_slice();
        bytemuck::cast_slice::<u8, T>(bytes).to_vec()
    })
}

/// Reads a float4 GPU buffer and converts to Vec<Vector3> (discarding w).
pub fn convert_v4_buffer_to_vec3(
    rd: &mut Gd<RenderingDevice>,
    buffer: &BufferInfo,
) -> Vec<Vector3> {
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

fn create_storage_buffer_from_bytes(rd: &mut Gd<RenderingDevice>, bytes: &[u8]) -> BufferInfo {
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
