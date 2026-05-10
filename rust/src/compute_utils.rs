use godot::builtin::{Callable, PackedByteArray, Variant, Vector2, Vector3};
use godot::classes::RdUniform;
use godot::classes::{RenderingDevice, RenderingServer};
use godot::obj::{Gd, NewGd, Singleton};
use godot::prelude::{load, Array, Rid};

use std::sync::{Mutex, OnceLock};
use std::time::Instant;

use crate::buffer_info::BufferInfo;

// ---------------------------------------------------------------------------
// Render-thread dispatch helpers (mirrors C# CallOnRenderThread pattern)
// ---------------------------------------------------------------------------

static RENDER_THREAD_TIMINGS: OnceLock<Mutex<Vec<(String, u64)>>> = OnceLock::new();

fn render_thread_timings() -> &'static Mutex<Vec<(String, u64)>> {
    RENDER_THREAD_TIMINGS.get_or_init(|| Mutex::new(Vec::new()))
}

fn push_render_thread_timing(path: String, elapsed_ns: u64) {
    render_thread_timings()
        .lock()
        .unwrap()
        .push((path, elapsed_ns));
}

pub fn record_render_thread_timing(path: String, elapsed_ns: u64) {
    push_render_thread_timing(path, elapsed_ns);
}

pub fn drain_render_thread_timings() -> Vec<(String, u64)> {
    let mut guard = render_thread_timings().lock().unwrap();
    std::mem::take(&mut *guard)
}

pub fn current_render_thread_timing_path(label: Option<&str>) -> Option<String> {
    if !crate::perf::current_tree_enabled() {
        return None;
    }
    let cur = crate::perf::current_path_joined();
    match (cur.is_empty(), label) {
        (true, Some(label)) => Some(label.to_string()),
        (false, Some(label)) => Some(format!("{cur}::{label}")),
        (false, None) => Some(cur),
        (true, None) => None,
    }
}

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
    let timing_path = current_render_thread_timing_path(None);
    on_render_thread(move || {
        let start = timing_path.as_ref().map(|_| Instant::now());
        let result = f();
        if let (Some(path), Some(start)) = (timing_path, start) {
            push_render_thread_timing(path, start.elapsed().as_nanos() as u64);
        }
        let _ = tx.send(result);
    });
    rx.recv().expect("render thread work failed")
}

/// Frees a GPU RID on the render thread (fire-and-forget).
pub fn free_rid_on_render_thread(rd: &mut Gd<RenderingDevice>, rid: Rid) {
    let rd_send = RdSend(rd.clone());
    let timing_path = current_render_thread_timing_path(Some("free_rid"));
    on_render_thread(move || {
        let start = timing_path.as_ref().map(|_| Instant::now());
        let mut rd = rd_send; // force whole-struct capture (Rust 2021)
        rd.0.free_rid(rid);
        if let (Some(path), Some(start)) = (timing_path, start) {
            push_render_thread_timing(path, start.elapsed().as_nanos() as u64);
        }
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
    let timing_path = current_render_thread_timing_path(Some("buffer_clear"));
    on_render_thread(move || {
        let start = timing_path.as_ref().map(|_| Instant::now());
        let mut rd = rd_send; // force whole-struct capture (Rust 2021)
        rd.0.buffer_clear(rid, offset, size);
        if let (Some(path), Some(start)) = (timing_path, start) {
            push_render_thread_timing(path, start.elapsed().as_nanos() as u64);
        }
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

/// Updates an existing uniform buffer with new data (16-byte aligned).
/// The buffer must already exist and be large enough.
pub fn update_uniform_buffer<T: bytemuck::Pod>(
    rd: &mut Gd<RenderingDevice>,
    buffer: &BufferInfo,
    value: &T,
) {
    let data_bytes = bytemuck::bytes_of(value);
    let padded_size = 16.max((data_bytes.len() + 15) / 16 * 16);
    let mut padded = vec![0u8; padded_size];
    padded[..data_bytes.len()].copy_from_slice(data_bytes);

    let rid = buffer.rid;
    let rd_send = RdSend(rd.clone());
    let timing_path = current_render_thread_timing_path(Some("buffer_update"));
    on_render_thread(move || {
        let start = timing_path.as_ref().map(|_| Instant::now());
        let mut rd = rd_send;
        let mut pba = PackedByteArray::new();
        pba.extend(padded.iter().copied());
        rd.0.buffer_update(rid, 0, padded_size as u32, &pba);
        if let (Some(path), Some(start)) = (timing_path, start) {
            push_render_thread_timing(path, start.elapsed().as_nanos() as u64);
        }
    });
}

/// Cached compute pipeline. Owns the shader and pipeline RIDs; they are freed on drop.
pub struct ComputePipeline {
    rd: RdSend,
    shader: Rid,
    pipeline: Rid,
}

impl ComputePipeline {
    /// Loads the shader from `shader_path`, creates SPIR-V and pipeline.
    /// Blocks until the render-thread work completes.
    pub fn new(rd: &mut Gd<RenderingDevice>, shader_path: &str) -> Self {
        let rd_send = RdSend(rd.clone());
        let shader_path = shader_path.to_string();

        let (shader, pipeline) = on_render_thread_sync(move || {
            let mut rd = rd_send;
            let mut shader_resource = load::<godot::classes::Resource>(&shader_path);
            let spirv = shader_resource
                .call("get_spirv", &[])
                .to::<Gd<godot::classes::RdShaderSpirv>>();
            let shader = rd.0.shader_create_from_spirv(&spirv);
            assert!(shader.is_valid(), "Failed to create shader from SPIR-V");
            let pipeline = rd.0.compute_pipeline_create(shader);
            assert!(pipeline.is_valid(), "Failed to create compute pipeline");
            (shader, pipeline)
        });

        Self {
            rd: RdSend(rd.clone()),
            shader,
            pipeline,
        }
    }

    /// Returns the raw shader RID (for building custom uniform sets).
    pub fn shader(&self) -> Rid {
        self.shader
    }

    /// Returns the raw pipeline RID.
    pub fn pipeline(&self) -> Rid {
        self.pipeline
    }

    /// Dispatches the cached pipeline. Only the uniform set is created (and freed) per call.
    pub fn dispatch(
        &self,
        rd: &mut Gd<RenderingDevice>,
        buffers: &[&BufferInfo],
        threads: u32,
        label: &'static str,
    ) {
        let buffer_descs: Vec<(Rid, crate::buffer_info::BufferType)> =
            buffers.iter().map(|b| (b.rid, b.buffer_type)).collect();

        let rd_send = RdSend(rd.clone());
        let shader = self.shader;
        let pipeline = self.pipeline;

        // Capture the timing-tree path on the worker thread BEFORE queueing.
        let path_full = current_render_thread_timing_path(Some(label));

        // Track the dispatch on the worker tree for render-thread-time folding later.
        if let Some(ref path_full) = path_full {
            crate::perf::with_current_tree(|tree| {
                tree.add_cpu_ns(path_full, 0);
            });
        }

        on_render_thread(move || {
            let mut rd = rd_send;
            let dispatch_start = if path_full.is_some() {
                Some(Instant::now())
            } else {
                None
            };

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
            let uniform_set = rd.0.uniform_set_create(&uniform_array, shader, 0);
            assert!(uniform_set.is_valid(), "Failed to create uniform set");
            rd.0.compute_list_bind_uniform_set(compute_list, uniform_set, 0);

            rd.0.compute_list_dispatch(compute_list, threads, 1, 1);
            rd.0.compute_list_end();

            rd.0.submit();
            rd.0.sync();

            rd.0.free_rid(uniform_set);

            if let Some(start) = dispatch_start.as_ref() {
                if let Some(path) = path_full {
                    push_render_thread_timing(path, start.elapsed().as_nanos() as u64);
                }
            }
        });
    }

    /// Frees shader and pipeline resources directly, marking them invalid
    /// so `Drop` won't attempt cleanup through the (potentially freed) RD.
    pub fn dispose_direct(&mut self, rd: &mut Gd<RenderingDevice>) {
        if self.pipeline.is_valid() {
            rd.free_rid(self.pipeline);
            self.pipeline = Rid::Invalid;
        }
        if self.shader.is_valid() {
            rd.free_rid(self.shader);
            self.shader = Rid::Invalid;
        }
    }
}

impl Drop for ComputePipeline {
    fn drop(&mut self) {
        if !self.shader.is_valid() && !self.pipeline.is_valid() {
            return;
        }

        let shader = self.shader;
        let pipeline = self.pipeline;

        // Destructors may run while the engine is shutting down during hot-reload.
        // Swallow any Godot-binding panic to avoid abort-on-drop during cleanup.
        let cleanup_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let rd_send = RdSend(self.rd.0.clone());
            on_render_thread(move || {
                let mut rd = rd_send;
                if pipeline.is_valid() {
                    rd.0.free_rid(pipeline);
                }
                if shader.is_valid() {
                    rd.0.free_rid(shader);
                }
            });
        }));

        if cleanup_result.is_err() {
            eprintln!(
                "ComputePipeline::drop: engine unavailable during cleanup; shader/pipeline RIDs may leak"
            );
        }

        self.shader = Rid::Invalid;
        self.pipeline = Rid::Invalid;
    }
}

/// Reads a GPU buffer back to CPU and reinterprets as a Vec<T>.
/// Blocks until the render thread completes the read.
pub fn convert_buffer_to_vec<T: bytemuck::Pod + Send>(
    rd: &mut Gd<RenderingDevice>,
    buffer: &BufferInfo,
) -> Vec<T> {
    let _scope = crate::perf::ThreadScope::enter("readback");
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

/// Reads a single element at the given index from a GPU buffer.
pub fn read_buffer_element<T: bytemuck::Pod + Send>(
    rd: &mut Gd<RenderingDevice>,
    buffer: &BufferInfo,
    index: u32,
) -> T {
    let _scope = crate::perf::ThreadScope::enter("readback_elem");
    let elem_size = std::mem::size_of::<T>() as u32;
    let offset = index * elem_size;
    let rd_send = RdSend(rd.clone());
    let rid = buffer.rid;

    on_render_thread_sync(move || {
        let mut rd = rd_send;
        let byte_data =
            rd.0.buffer_get_data_ex(rid)
                .offset_bytes(offset)
                .size_bytes(elem_size)
                .done();
        let bytes = byte_data.as_slice();
        *bytemuck::from_bytes::<T>(bytes)
    })
}

/// Drains all currently captured GPU timestamps from `rd`. Returns
/// `(name, gpu_time_ns)` tuples in capture order.
pub fn drain_gpu_timestamps(rd: &mut Gd<RenderingDevice>) -> Vec<(String, u64)> {
    let rd_send = RdSend(rd.clone());
    on_render_thread_sync(move || {
        let rd = rd_send;
        let n = rd.0.get_captured_timestamps_count();
        let mut out = Vec::with_capacity(n as usize);
        for i in 0..n {
            let name = rd.0.get_captured_timestamp_name(i).to_string();
            let t = rd.0.get_captured_timestamp_gpu_time(i);
            out.push((name, t));
        }
        out
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

/// Reads a packed float buffer (xyzxyz...) and reinterprets it as Vec<Vector3>.
///
/// This is a forced cast path and assumes Godot `real` is `f32` for Vector3.
pub fn convert_packed_f32_buffer_to_vec3(
    rd: &mut Gd<RenderingDevice>,
    buffer: &BufferInfo,
) -> Vec<Vector3> {
    let float_data: Vec<f32> = convert_buffer_to_vec(rd, buffer);
    let num_vectors = float_data.len() / 3;

    assert_eq!(
        std::mem::size_of::<Vector3>(),
        3 * std::mem::size_of::<f32>()
    );
    assert_eq!(std::mem::align_of::<Vector3>(), std::mem::align_of::<f32>());

    unsafe {
        std::slice::from_raw_parts(float_data.as_ptr() as *const Vector3, num_vectors).to_vec()
    }
}

/// Reads a packed float buffer (uvuv...) and reinterprets it as Vec<Vector2>.
///
/// This is a forced cast path and assumes Godot `real` is `f32` for Vector2.
pub fn convert_packed_f32_buffer_to_vec2(
    rd: &mut Gd<RenderingDevice>,
    buffer: &BufferInfo,
) -> Vec<Vector2> {
    let float_data: Vec<f32> = convert_buffer_to_vec(rd, buffer);
    let num_vectors = float_data.len() / 2;

    assert_eq!(
        std::mem::size_of::<Vector2>(),
        2 * std::mem::size_of::<f32>()
    );
    assert_eq!(std::mem::align_of::<Vector2>(), std::mem::align_of::<f32>());

    unsafe {
        std::slice::from_raw_parts(float_data.as_ptr() as *const Vector2, num_vectors).to_vec()
    }
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
    let pba = PackedByteArray::from(bytes);
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
