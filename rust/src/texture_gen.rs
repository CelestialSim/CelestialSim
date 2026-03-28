use godot::classes::rendering_device::{
    DataFormat, DriverResource, TextureSamples, TextureType, TextureUsageBits, UniformType,
};
use godot::classes::{RdTextureFormat, RdTextureView, RdUniform, RenderingDevice, RenderingServer};
use godot::obj::{Gd, NewGd, Singleton};
use godot::prelude::{Array, PackedByteArray, Rid};

use crate::compute_utils::{on_render_thread_sync, ComputePipeline, RdSend};

const CUBEMAP_NOISE_SHADER: &str = "res://addons/celestial_sim/shaders/CubemapNoise.slang";

/// Creates a uniform buffer directly on the given RD (for use inside render-thread callbacks).
fn create_uniform_buffer_raw(rd: &mut Gd<RenderingDevice>, data: &[u8]) -> Rid {
    let padded_size = 16.max((data.len() + 15) / 16 * 16);
    let mut padded = vec![0u8; padded_size];
    padded[..data.len()].copy_from_slice(data);

    let mut pba = PackedByteArray::new();
    pba.extend(padded.iter().copied());
    let rid = rd
        .uniform_buffer_create_ex(padded_size as u32)
        .data(&pba)
        .done();
    assert!(rid.is_valid(), "Failed to create uniform buffer");
    rid
}

fn cubemap_usage_bits() -> TextureUsageBits {
    TextureUsageBits::SAMPLING_BIT
        | TextureUsageBits::STORAGE_BIT
        | TextureUsageBits::CAN_UPDATE_BIT
        | TextureUsageBits::CAN_COPY_FROM_BIT
}

/// Allocates a shared cubemap texture and dispatches CubemapNoise.slang to fill it.
pub struct CubemapTextureGen {
    main_texture_rid: Rid,
    local_texture_rid: Rid,
    size: u32,
    pipeline: Option<ComputePipeline>,
}

impl CubemapTextureGen {
    pub fn new() -> Self {
        Self {
            main_texture_rid: Rid::Invalid,
            local_texture_rid: Rid::Invalid,
            size: 0,
            pipeline: None,
        }
    }

    /// Allocates a cubemap on the main RD (render thread) and creates a writable
    /// mirror on `local_rd` via `texture_create_from_extension`.
    ///
    /// Returns `(main_rid, local_rid)`.
    pub fn create_shared_cubemap(
        &mut self,
        local_rd: &mut Gd<RenderingDevice>,
        size: u32,
    ) -> (Rid, Rid) {
        // Step 1: create texture + get driver handle on the render thread (main RD).
        let driver_handle = on_render_thread_sync(move || {
            let mut main_rd = RenderingServer::singleton().get_rendering_device().unwrap();

            let mut format = RdTextureFormat::new_gd();
            format.set_texture_type(TextureType::CUBE);
            format.set_format(DataFormat::R8G8B8A8_UNORM);
            format.set_width(size);
            format.set_height(size);
            format.set_depth(1);
            format.set_array_layers(6);
            format.set_mipmaps(1);
            format.set_samples(TextureSamples::SAMPLES_1);
            format.set_usage_bits(cubemap_usage_bits());

            let view = RdTextureView::new_gd();
            let main_rid = main_rd.texture_create(&format, &view);
            assert!(main_rid.is_valid(), "Failed to create main cubemap texture");

            let handle = main_rd.get_driver_resource(DriverResource::TEXTURE, main_rid, 0);
            (main_rid, handle)
        });

        let (main_rid, handle) = driver_handle;

        // Step 2: create the extension on the local RD (also via render thread).
        let local_rd_send = RdSend(local_rd.clone());
        let local_rid = on_render_thread_sync(move || {
            let mut rd = local_rd_send;
            let rid = rd.0.texture_create_from_extension(
                TextureType::CUBE,
                DataFormat::R8G8B8A8_UNORM,
                TextureSamples::SAMPLES_1,
                cubemap_usage_bits(),
                handle,
                size as u64,
                size as u64,
                1,
                6,
            );
            assert!(
                rid.is_valid(),
                "Failed to create local cubemap texture from extension"
            );
            rid
        });

        self.main_texture_rid = main_rid;
        self.local_texture_rid = local_rid;
        self.size = size;

        (main_rid, local_rid)
    }

    /// Loads the CubemapNoise compute shader and creates the pipeline.
    pub fn init_pipeline(&mut self, rd: &mut Gd<RenderingDevice>) {
        if self.pipeline.is_none() {
            self.pipeline = Some(ComputePipeline::new(rd, CUBEMAP_NOISE_SHADER));
        }
    }

    /// Dispatches the noise shader for all 6 cubemap faces.
    ///
    /// Shader bindings:
    ///   0 — `RWTexture2DArray<float4>` (IMAGE)
    ///   1 — `face_index` (UNIFORM_BUFFER, u32)
    ///   2 — `tex_size`   (UNIFORM_BUFFER, u32)
    ///   3 — `radius`     (UNIFORM_BUFFER, f32)
    pub fn generate(&self, rd: &mut Gd<RenderingDevice>, size: u32, radius: f32) {
        let pipeline = self
            .pipeline
            .as_ref()
            .expect("CubemapTextureGen pipeline not initialised; call init_pipeline first");

        let workgroups_x = (size + 7) / 8;
        let workgroups_y = (size + 7) / 8;

        for face in 0u32..6 {
            self.dispatch_face(rd, pipeline, face, size, radius, workgroups_x, workgroups_y);
        }
    }

    /// Dispatches the compute shader for a single cubemap face (synchronous).
    fn dispatch_face(
        &self,
        rd: &mut Gd<RenderingDevice>,
        pipeline: &ComputePipeline,
        face: u32,
        size: u32,
        radius: f32,
        workgroups_x: u32,
        workgroups_y: u32,
    ) {
        let texture_rid = self.local_texture_rid;
        let shader = pipeline.shader();
        let pipe = pipeline.pipeline();

        let rd_send = RdSend(rd.clone());

        on_render_thread_sync(move || {
            let mut rd = rd_send;

            // Create uniform buffers for the scalar parameters.
            let face_bytes = face.to_le_bytes();
            let size_bytes = size.to_le_bytes();
            let radius_bytes = radius.to_le_bytes();

            let face_rid = create_uniform_buffer_raw(&mut rd.0, &face_bytes);
            let size_rid = create_uniform_buffer_raw(&mut rd.0, &size_bytes);
            let radius_rid = create_uniform_buffer_raw(&mut rd.0, &radius_bytes);

            // Binding 0: cubemap image
            let mut tex_uniform = RdUniform::new_gd();
            tex_uniform.set_binding(0);
            tex_uniform.set_uniform_type(UniformType::IMAGE);
            tex_uniform.add_id(texture_rid);

            // Binding 1: face_index
            let mut face_uniform = RdUniform::new_gd();
            face_uniform.set_binding(1);
            face_uniform.set_uniform_type(UniformType::UNIFORM_BUFFER);
            face_uniform.add_id(face_rid);

            // Binding 2: tex_size
            let mut size_uniform = RdUniform::new_gd();
            size_uniform.set_binding(2);
            size_uniform.set_uniform_type(UniformType::UNIFORM_BUFFER);
            size_uniform.add_id(size_rid);

            // Binding 3: radius
            let mut radius_uniform = RdUniform::new_gd();
            radius_uniform.set_binding(3);
            radius_uniform.set_uniform_type(UniformType::UNIFORM_BUFFER);
            radius_uniform.add_id(radius_rid);

            let mut uniforms = Array::new();
            uniforms.push(&tex_uniform);
            uniforms.push(&face_uniform);
            uniforms.push(&size_uniform);
            uniforms.push(&radius_uniform);

            let uniform_set = rd.0.uniform_set_create(&uniforms, shader, 0);
            assert!(
                uniform_set.is_valid(),
                "Failed to create cubemap uniform set"
            );

            let compute_list = rd.0.compute_list_begin();
            rd.0.compute_list_bind_compute_pipeline(compute_list, pipe);
            rd.0.compute_list_bind_uniform_set(compute_list, uniform_set, 0);
            rd.0.compute_list_dispatch(compute_list, workgroups_x, workgroups_y, 1);
            rd.0.compute_list_end();
            rd.0.submit();
            rd.0.sync();

            // Clean up per-face resources.
            rd.0.free_rid(uniform_set);
            rd.0.free_rid(face_rid);
            rd.0.free_rid(size_rid);
            rd.0.free_rid(radius_rid);
        });
    }

    /// Frees GPU resources on both main and local RDs.
    pub fn dispose_direct(
        &mut self,
        main_rd: &mut Gd<RenderingDevice>,
        local_rd: &mut Gd<RenderingDevice>,
    ) {
        if self.local_texture_rid.is_valid() {
            crate::compute_utils::free_rid_on_render_thread(local_rd, self.local_texture_rid);
            self.local_texture_rid = Rid::Invalid;
        }
        if self.main_texture_rid.is_valid() {
            crate::compute_utils::free_rid_on_render_thread(main_rd, self.main_texture_rid);
            self.main_texture_rid = Rid::Invalid;
        }
        if let Some(ref mut pipeline) = self.pipeline {
            pipeline.dispose_direct(local_rd);
        }
        self.pipeline = None;
    }

    /// Frees only the local-RD resources (local texture + pipeline).
    /// Also frees the main-RD texture on the render thread.
    pub fn dispose_local(&mut self, local_rd: &mut Gd<RenderingDevice>) {
        if self.local_texture_rid.is_valid() {
            crate::compute_utils::free_rid_on_render_thread(local_rd, self.local_texture_rid);
            self.local_texture_rid = Rid::Invalid;
        }
        if self.main_texture_rid.is_valid() {
            let old_main = self.main_texture_rid;
            on_render_thread_sync(move || {
                let mut main_rd = RenderingServer::singleton().get_rendering_device().unwrap();
                main_rd.free_rid(old_main);
            });
            self.main_texture_rid = Rid::Invalid;
        }
        if let Some(ref mut pipeline) = self.pipeline {
            pipeline.dispose_direct(local_rd);
        }
        self.pipeline = None;
    }

    /// Frees old textures, creates a new shared cubemap at `new_size`, and
    /// regenerates the noise.  Returns `(new_main_rid, old_main_rid)` so the
    /// caller can defer freeing the old texture until materials are updated.
    ///
    /// Called from the worker thread when the user changes cubemap resolution.
    pub fn resize(
        &mut self,
        rd: &mut Gd<RenderingDevice>,
        new_size: u32,
        radius: f32,
    ) -> (Rid, Rid) {
        // 1. Free old local texture on the local RD (only used for compute, safe to free now).
        if self.local_texture_rid.is_valid() {
            crate::compute_utils::free_rid_on_render_thread(rd, self.local_texture_rid);
            self.local_texture_rid = Rid::Invalid;
        }
        // 2. Remember the old main texture (caller will free it after materials update).
        let old_main = self.main_texture_rid;
        self.main_texture_rid = Rid::Invalid;
        // 3. Create new shared cubemap at the requested size.
        self.create_shared_cubemap(rd, new_size);
        // 4. Ensure the pipeline exists (it is size-independent).
        self.init_pipeline(rd);
        // 5. Generate noise into the new texture.
        self.generate(rd, new_size, radius);

        (self.main_texture_rid, old_main)
    }

    /// Returns the main-RD cubemap RID (for sampling in other shaders / materials).
    pub fn main_texture_rid(&self) -> Rid {
        self.main_texture_rid
    }

    /// Returns true if a cubemap texture has been created.
    pub fn has_texture(&self) -> bool {
        self.local_texture_rid.is_valid()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cubemap_texture_gen_default() {
        let gen = CubemapTextureGen::new();
        assert!(!gen.main_texture_rid().is_valid());
        assert_eq!(gen.size, 0);
    }
}
