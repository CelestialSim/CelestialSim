//! Pipeline / buffer / texture creation helpers (render-thread only).
//!
//! Every creator here hands its RID straight to an [`Owned`] handle bound to the caller's
//! [`RidSink`], so the resource is released by `Drop` alone — see `gpu::owned` for the
//! drop-order contract. Nothing in this module frees a RID explicitly.

use std::sync::Arc;

use godot::classes::rendering_device::{
    DataFormat, ShaderLanguage, ShaderStage, TextureUsageBits, UniformType,
};
use godot::classes::{
    RdShaderSource, RdShaderSpirv, RdTextureFormat, RdTextureView, RdUniform, RenderingDevice,
};
use godot::prelude::*;

use super::owned::{Owned, RdBuffer, RdPipeline, RdShader, RdTexture, RidSink};

/// Compute shader + pipeline from SPIR-V bytes produced by `build.rs`.
///
/// On a pipeline-create failure the early return drops the `RdShader`, which frees the
/// shader through `sink` — no manual cleanup.
pub fn compute_pipeline(
    rd: &mut Gd<RenderingDevice>,
    sink: &Arc<dyn RidSink>,
    spirv: &[u8],
    name: &str,
) -> Option<(RdShader, RdPipeline)> {
    if spirv.is_empty() {
        godot_error!("[celestial] {name}: empty SPIR-V — was slangc installed at build time?");
        return None;
    }
    let mut sp = RdShaderSpirv::new_gd();
    sp.set_stage_bytecode(ShaderStage::COMPUTE, &PackedByteArray::from(spirv));
    let shader_rid = rd.shader_create_from_spirv(&sp);
    if !shader_rid.is_valid() {
        godot_error!("[celestial] {name}: shader_create_from_spirv failed");
        return None;
    }
    let shader: RdShader = Owned::new(shader_rid, sink.clone());
    let pipeline_rid = rd.compute_pipeline_create(shader.rid());
    if !pipeline_rid.is_valid() {
        godot_error!("[celestial] {name}: compute_pipeline_create failed");
        return None; // dropping `shader` frees it
    }
    Some((shader, Owned::new(pipeline_rid, sink.clone())))
}

/// Compute shader + pipeline from GLSL **source compiled at runtime** by Godot's
/// own shader compiler (no `slangc`, nothing extra shipped). Used for the
/// user-authored custom-surface terrain (`crate::custom_surface`): a compile
/// error is logged with Godot's message and returns `None` so the caller can
/// fall back to the built-in path instead of crashing.
pub fn compute_pipeline_from_glsl(
    rd: &mut Gd<RenderingDevice>,
    sink: &Arc<dyn RidSink>,
    glsl: &str,
    name: &str,
) -> Option<(RdShader, RdPipeline)> {
    let mut src = RdShaderSource::new_gd();
    src.set_language(ShaderLanguage::GLSL);
    src.set_stage_source(ShaderStage::COMPUTE, glsl);
    let Some(spirv) = rd.shader_compile_spirv_from_source_ex(&src).allow_cache(true).done() else {
        godot_error!("[celestial] {name}: shader_compile_spirv_from_source returned null");
        return None;
    };
    let err = spirv.get_stage_compile_error(ShaderStage::COMPUTE);
    if !err.is_empty() {
        godot_error!("[celestial] {name}: GLSL compile error:\n{err}");
        return None;
    }
    let shader_rid = rd.shader_create_from_spirv(&spirv);
    if !shader_rid.is_valid() {
        godot_error!("[celestial] {name}: shader_create_from_spirv failed");
        return None;
    }
    let shader: RdShader = Owned::new(shader_rid, sink.clone());
    let pipeline_rid = rd.compute_pipeline_create(shader.rid());
    if !pipeline_rid.is_valid() {
        godot_error!("[celestial] {name}: compute_pipeline_create failed");
        return None; // dropping `shader` frees it
    }
    Some((shader, Owned::new(pipeline_rid, sink.clone())))
}

/// Uninitialized storage buffer of `size` bytes.
pub fn storage_buffer_empty(
    rd: &mut Gd<RenderingDevice>,
    sink: &Arc<dyn RidSink>,
    size: u64,
) -> RdBuffer {
    Owned::new(rd.storage_buffer_create_ex(size as u32).done(), sink.clone())
}

/// Storage buffer pre-filled with `bytes`.
pub fn storage_buffer(
    rd: &mut Gd<RenderingDevice>,
    sink: &Arc<dyn RidSink>,
    bytes: &[u8],
) -> RdBuffer {
    let rid = rd
        .storage_buffer_create_ex(bytes.len() as u32)
        .data(&PackedByteArray::from(bytes))
        .done();
    Owned::new(rid, sink.clone())
}

/// RGBA16F storage texture (compute writes, material samples).
pub fn attribute_texture(
    rd: &mut Gd<RenderingDevice>,
    sink: &Arc<dyn RidSink>,
    width: u32,
    height: u32,
) -> RdTexture {
    let mut fmt = RdTextureFormat::new_gd();
    fmt.set_format(DataFormat::R16G16B16A16_SFLOAT);
    fmt.set_width(width);
    fmt.set_height(height);
    fmt.set_usage_bits(
        TextureUsageBits::STORAGE_BIT | TextureUsageBits::SAMPLING_BIT,
    );
    Owned::new(rd.texture_create(&fmt, &RdTextureView::new_gd()), sink.clone())
}

/// RGBA32F storage texture (compute writes world positions, material samples).
/// One texel per vertex: the surface material reads VERTEX from here because a
/// spatial shader cannot sample the `verts` storage buffer the readback uses.
pub fn position_texture(
    rd: &mut Gd<RenderingDevice>,
    sink: &Arc<dyn RidSink>,
    width: u32,
    height: u32,
) -> RdTexture {
    let mut fmt = RdTextureFormat::new_gd();
    fmt.set_format(DataFormat::R32G32B32A32_SFLOAT);
    fmt.set_width(width);
    fmt.set_height(height);
    // CAN_COPY_FROM so debug tooling can read the drawn positions back.
    fmt.set_usage_bits(
        TextureUsageBits::STORAGE_BIT
            | TextureUsageBits::SAMPLING_BIT
            | TextureUsageBits::CAN_COPY_FROM_BIT,
    );
    Owned::new(rd.texture_create(&fmt, &RdTextureView::new_gd()), sink.clone())
}

/// RGBA8-UNORM storage texture (compute writes, material samples). Used for the
/// Phase-4 per-chunk colour/normal detail atlases: `STORAGE_BIT` so the bake
/// compute pass can write it, `SAMPLING_BIT` so `terrain_chunk.gdshader` can
/// `texelFetch` it. Normals are encoded `*0.5+0.5` into [0,1] to fit unorm.
pub fn atlas_texture(
    rd: &mut Gd<RenderingDevice>,
    sink: &Arc<dyn RidSink>,
    width: u32,
    height: u32,
) -> RdTexture {
    let mut fmt = RdTextureFormat::new_gd();
    fmt.set_format(DataFormat::R8G8B8A8_UNORM);
    fmt.set_width(width);
    fmt.set_height(height);
    // CAN_COPY_FROM so the detail tile can be read back / exported for inspection.
    fmt.set_usage_bits(
        TextureUsageBits::STORAGE_BIT
            | TextureUsageBits::SAMPLING_BIT
            | TextureUsageBits::CAN_COPY_FROM_BIT,
    );
    Owned::new(rd.texture_create(&fmt, &RdTextureView::new_gd()), sink.clone())
}

pub fn uniform(utype: UniformType, binding: i32, rid: Rid) -> Gd<RdUniform> {
    let mut u = RdUniform::new_gd();
    u.set_uniform_type(utype);
    u.set_binding(binding);
    u.add_id(rid);
    u
}
