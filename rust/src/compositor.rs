use godot::classes::light_3d::Param as LightParam;
use godot::classes::notify::ObjectNotification;
use godot::classes::rendering_device::{
    CompareOperator, PolygonCullMode, PolygonFrontFace, RenderPrimitive, ShaderStage, UniformType,
};
use godot::classes::{
    CompositorEffect, DirectionalLight3D, Engine, ICompositorEffect, Node, Node3D,
    RdPipelineColorBlendState, RdPipelineColorBlendStateAttachment, RdPipelineDepthStencilState,
    RdPipelineMultisampleState, RdPipelineRasterizationState, RdShaderFile, RdUniform, RenderData,
    RenderSceneBuffersRd, RenderingDevice, RenderingServer, SceneTree,
};
use godot::prelude::*;

use crate::initial_state;
use crate::state::CesState;

const DRAW_SHADER_PATH: &str =
    "res://addons/celestial_sim_rust/shaders/ces_final_state_compositor.glsl";

/// Compositor effect that renders CesState geometry in scene space.
#[derive(GodotClass)]
#[class(tool, base = CompositorEffect)]
pub struct CesFinalStateCompositorRust {
    base: Base<CompositorEffect>,

    #[export]
    target_path: NodePath,

    #[export]
    target_position: Vector3,

    #[export]
    radius: f32,

    #[export]
    ambient_strength: f32,

    #[export]
    sphere_color: Color,

    // GPU resources (populated by construct() on the render thread)
    rd: Option<Gd<RenderingDevice>>,
    draw_shader: Rid,
    vertex_format: i64,
    vertex_storage_buffer: Rid,
    triangle_storage_buffer: Rid,
    structured_vertex_array: Rid,
    structured_uniform_set: Rid,
    structured_vertex_count: u32,
    ces_state: Option<CesState>,

    render_pipeline: Rid,
    framebuffer_format: i64,
}

#[godot_api]
impl ICompositorEffect for CesFinalStateCompositorRust {
    fn init(base: Base<CompositorEffect>) -> Self {
        Self {
            base,
            target_path: NodePath::default(),
            target_position: Vector3::ZERO,
            radius: 1.0,
            ambient_strength: 0.05,
            sphere_color: Color::from_rgb(0.0, 0.0, 0.0),
            rd: None,
            draw_shader: Rid::Invalid,
            vertex_format: 0,
            vertex_storage_buffer: Rid::Invalid,
            triangle_storage_buffer: Rid::Invalid,
            structured_vertex_array: Rid::Invalid,
            structured_uniform_set: Rid::Invalid,
            structured_vertex_count: 0,
            ces_state: None,
            render_pipeline: Rid::Invalid,
            framebuffer_format: -1,
        }
    }

    fn on_notification(&mut self, what: ObjectNotification) {
        if what == ObjectNotification::PREDELETE {
            if let Some(rd) = &mut self.rd {
                // Free dependents before dependencies
                if self.structured_uniform_set.is_valid() {
                    rd.free_rid(self.structured_uniform_set);
                }
                if self.structured_vertex_array.is_valid() {
                    rd.free_rid(self.structured_vertex_array);
                }
                if self.render_pipeline.is_valid() {
                    rd.free_rid(self.render_pipeline);
                }
                // Free all CesState GPU buffers directly (not deferred)
                if let Some(ref state) = self.ces_state {
                    for rid in state.all_buffers() {
                        if rid.is_valid() {
                            rd.free_rid(rid);
                        }
                    }
                }
                if self.draw_shader.is_valid() {
                    rd.free_rid(self.draw_shader);
                }
            }
        }
    }

    fn render_callback(&mut self, _effect_callback_type: i32, render_data: Option<Gd<RenderData>>) {
        // Lazy-init on first render_callback (runs on render thread)
        if self.rd.is_none() {
            godot_print!("[CompositorRust] render_callback: initializing...");
            self.construct();
            if self.rd.is_some() && self.draw_shader.is_valid() {
                godot_print!("[CompositorRust] construct() succeeded, shader valid, vtx_count={}", self.structured_vertex_count);
            } else {
                godot_print!("[CompositorRust] construct() FAILED: rd={} shader_valid={}", self.rd.is_some(), self.draw_shader.is_valid());
            }
        }
        if self.rd.is_none() || !self.draw_shader.is_valid() {
            return;
        }

        let Some(render_data) = render_data else {
            return;
        };

        // Get scene buffers (cast to RD variant)
        let Some(scene_buffers_base) = render_data.get_render_scene_buffers() else {
            return;
        };
        let Ok(scene_buffers) = scene_buffers_base.try_cast::<RenderSceneBuffersRd>() else {
            return;
        };

        let size = scene_buffers.get_internal_size();
        if size.x == 0 || size.y == 0 {
            return;
        }

        let color_image = scene_buffers.get_color_layer(0);
        let depth_image = scene_buffers.get_depth_layer(0);
        if !color_image.is_valid() || !depth_image.is_valid() {
            return;
        }

        let textures: Array<Rid> = array![color_image, depth_image];
        let (framebuffer, fb_format) = {
            let rd = self.rd.as_mut().unwrap();
            let framebuffer = rd.framebuffer_create(&textures);
            if !framebuffer.is_valid() {
                return;
            }
            let fb_format = rd.framebuffer_get_format(framebuffer);
            (framebuffer, fb_format)
        };
        if !self.ensure_pipeline(fb_format) {
            self.rd.as_mut().unwrap().free_rid(framebuffer);
            return;
        }

        let Some(scene_data) = render_data.get_render_scene_data() else {
            self.rd.as_mut().unwrap().free_rid(framebuffer);
            return;
        };

        let cam_transform = scene_data.get_cam_transform();
        let cam_projection = scene_data.get_cam_projection();
        let view_transform = cam_transform.affine_inverse();

        // Resolve target position (optionally follow a target node)
        let mut position = self.target_position;
        if !self.target_path.is_empty() {
            if let Some(tree) = Engine::singleton()
                .get_main_loop()
                .and_then(|ml| ml.try_cast::<SceneTree>().ok())
            {
                if let Some(root) = tree.get_root() {
                    let root_node: Gd<Node> = root.upcast();
                    if let Some(target) = root_node.try_get_node_as::<Node3D>(&self.target_path) {
                        position = target.get_global_transform().origin;
                    }
                }
            }
        }

        let model_transform = Transform3D::new(
            Basis::from_scale(Vector3::new(self.radius, self.radius, self.radius)),
            position,
        );

        // Lighting
        let scene_light = find_directional_light();
        let (light_dir, light_color, light_intensity) = if let Some(light) = &scene_light {
            let dir = -light.get_global_transform().basis.col_c().normalized();
            let color = light.get_color();
            let energy = light.get_param(LightParam::ENERGY);
            (dir, color, energy)
        } else {
            (Vector3::new(0.0, 0.0, 1.0), Color::WHITE, 1.0)
        };

        let light_dir_view = (view_transform.basis * light_dir).normalized();

        // Pack push constants: 60 floats = 240 bytes
        let mut push_data = [0.0f32; 60];
        write_transform_to_array(&model_transform, &mut push_data, 0);
        write_transform_to_array(&view_transform, &mut push_data, 16);
        write_projection_to_array(&cam_projection, &mut push_data, 32);
        push_data[48] = light_dir_view.x;
        push_data[49] = light_dir_view.y;
        push_data[50] = light_dir_view.z;
        push_data[51] = light_intensity;
        push_data[52] = light_color.r;
        push_data[53] = light_color.g;
        push_data[54] = light_color.b;
        push_data[55] = self.ambient_strength;
        push_data[56] = self.sphere_color.r;
        push_data[57] = self.sphere_color.g;
        push_data[58] = self.sphere_color.b;
        push_data[59] = self.sphere_color.a;

        let push_bytes: PackedByteArray = {
            let byte_slice: &[u8] = bytemuck::cast_slice(&push_data);
            PackedByteArray::from(byte_slice)
        };

        // Draw
        let render_pipeline = self.render_pipeline;
        let vertex_array = self.structured_vertex_array;
        let uniform_set = self.structured_uniform_set;
        let rd = self.rd.as_mut().unwrap();

        let draw_list = rd.draw_list_begin(framebuffer);
        if draw_list < 0 {
            rd.free_rid(framebuffer);
            return;
        }

        rd.draw_list_bind_render_pipeline(draw_list, render_pipeline);
        rd.draw_list_bind_vertex_array(draw_list, vertex_array);
        rd.draw_list_bind_uniform_set(draw_list, uniform_set, 0);
        rd.draw_list_set_push_constant(draw_list, &push_bytes, push_bytes.len() as u32);
        rd.draw_list_draw(draw_list, false, 1);
        rd.draw_list_end();

        // Debug: print once to confirm draw succeeds
        if self.framebuffer_format != fb_format || !self.render_pipeline.is_valid() {
            // Pipeline was just (re)created, print confirmation
        }

        rd.free_rid(framebuffer);
    }
}

impl CesFinalStateCompositorRust {
    /// Loads the draw shader, creates CesState GPU buffers, and builds the
    /// uniform set. Must run on the render thread.
    fn construct(&mut self) {
        let Some(mut rd) = RenderingServer::singleton().get_rendering_device() else {
            godot_error!("CesFinalStateCompositorRust: No rendering device available");
            return;
        };

        // Load and compile the draw shader
        let shader_file: Gd<RdShaderFile> = load(DRAW_SHADER_PATH);
        let Some(spirv) = shader_file.get_spirv() else {
            godot_error!("CesFinalStateCompositorRust: Failed to get SPIR-V from shader file");
            return;
        };

        let vertex_err = spirv.get_stage_compile_error(ShaderStage::VERTEX);
        if !vertex_err.is_empty() {
            godot_error!("CesFinalStateCompositorRust: Vertex compile error: {vertex_err}");
            return;
        }
        let fragment_err = spirv.get_stage_compile_error(ShaderStage::FRAGMENT);
        if !fragment_err.is_empty() {
            godot_error!("CesFinalStateCompositorRust: Fragment compile error: {fragment_err}");
            return;
        }

        let draw_shader = rd.shader_create_from_spirv(&spirv);
        if !draw_shader.is_valid() {
            godot_error!("CesFinalStateCompositorRust: Failed to create shader");
            return;
        }

        // Empty vertex format (vertices are fetched from storage buffer in shader)
        let vertex_format = rd.vertex_format_create(&Array::new());

        // Initialize icosphere GPU state
        let ces_state = initial_state::create_core_state(&mut rd);
        let vertex_storage_buffer = ces_state.v_pos.rid;
        let triangle_storage_buffer = ces_state.t_abc.rid;
        let structured_vertex_count = ces_state.n_tris * 3;

        // Vertex array with empty source buffers (data comes from storage buffers)
        let structured_vertex_array =
            rd.vertex_array_create(structured_vertex_count, vertex_format, &Array::new());
        if !structured_vertex_array.is_valid() {
            godot_error!("CesFinalStateCompositorRust: Failed to create vertex array");
            return;
        }

        // Uniform set: binding 0 = vertex positions, binding 1 = triangle indices
        let mut vertex_uniform = RdUniform::new_gd();
        vertex_uniform.set_binding(0);
        vertex_uniform.set_uniform_type(UniformType::STORAGE_BUFFER);
        vertex_uniform.add_id(vertex_storage_buffer);

        let mut tri_uniform = RdUniform::new_gd();
        tri_uniform.set_binding(1);
        tri_uniform.set_uniform_type(UniformType::STORAGE_BUFFER);
        tri_uniform.add_id(triangle_storage_buffer);

        let uniforms: Array<Gd<RdUniform>> = array![&vertex_uniform, &tri_uniform];
        let structured_uniform_set = rd.uniform_set_create(&uniforms, draw_shader, 0);
        if !structured_uniform_set.is_valid() {
            godot_error!("CesFinalStateCompositorRust: Failed to create uniform set");
            return;
        }

        // Store all resources
        self.draw_shader = draw_shader;
        self.vertex_format = vertex_format;
        self.vertex_storage_buffer = vertex_storage_buffer;
        self.triangle_storage_buffer = triangle_storage_buffer;
        self.structured_vertex_count = structured_vertex_count;
        self.structured_vertex_array = structured_vertex_array;
        self.structured_uniform_set = structured_uniform_set;
        self.ces_state = Some(ces_state);
        self.rd = Some(rd);
    }

    /// Creates or recreates the render pipeline when the framebuffer format changes.
    fn ensure_pipeline(&mut self, fb_format: i64) -> bool {
        let Some(rd) = &mut self.rd else { return false };
        if !self.draw_shader.is_valid() {
            return false;
        }
        if self.framebuffer_format == fb_format && self.render_pipeline.is_valid() {
            return true;
        }

        if self.render_pipeline.is_valid() {
            rd.free_rid(self.render_pipeline);
        }
        self.framebuffer_format = fb_format;

        let mut blend = RdPipelineColorBlendState::new_gd();
        let attachment = RdPipelineColorBlendStateAttachment::new_gd();
        let attachments: Array<Gd<RdPipelineColorBlendStateAttachment>> = array![&attachment];
        blend.set_attachments(&attachments);

        let mut depth_state = RdPipelineDepthStencilState::new_gd();
        depth_state.set_enable_depth_test(true);
        depth_state.set_enable_depth_write(true);
        depth_state.set_depth_compare_operator(CompareOperator::GREATER_OR_EQUAL);

        let mut raster_state = RdPipelineRasterizationState::new_gd();
        raster_state.set_cull_mode(PolygonCullMode::BACK);
        raster_state.set_front_face(PolygonFrontFace::CLOCKWISE);

        let multisample_state = RdPipelineMultisampleState::new_gd();

        self.render_pipeline = rd.render_pipeline_create(
            self.draw_shader,
            fb_format,
            self.vertex_format,
            RenderPrimitive::TRIANGLES,
            &raster_state,
            &multisample_state,
            &depth_state,
            &blend,
        );

        self.render_pipeline.is_valid()
    }
}

/// Pack a `Transform3D` into a float array at `offset` (column-major 4x4).
/// Matches C#'s `WriteTransformToArray` which uses `Basis.X/Y/Z` (columns).
fn write_transform_to_array(transform: &Transform3D, array: &mut [f32], offset: usize) {
    let b = &transform.basis;
    // Column 0 (Basis.X in C# = col_a in gdext)
    let col0 = b.col_a();
    array[offset] = col0.x;
    array[offset + 1] = col0.y;
    array[offset + 2] = col0.z;
    array[offset + 3] = 0.0;
    // Column 1 (Basis.Y in C# = col_b in gdext)
    let col1 = b.col_b();
    array[offset + 4] = col1.x;
    array[offset + 5] = col1.y;
    array[offset + 6] = col1.z;
    array[offset + 7] = 0.0;
    // Column 2 (Basis.Z in C# = col_c in gdext)
    let col2 = b.col_c();
    array[offset + 8] = col2.x;
    array[offset + 9] = col2.y;
    array[offset + 10] = col2.z;
    array[offset + 11] = 0.0;
    // Column 3 (origin)
    array[offset + 12] = transform.origin.x;
    array[offset + 13] = transform.origin.y;
    array[offset + 14] = transform.origin.z;
    array[offset + 15] = 1.0;
}

/// Pack a `Projection` into a float array at `offset` (column-major 4x4).
fn write_projection_to_array(projection: &Projection, array: &mut [f32], offset: usize) {
    for col in 0..4 {
        let v = projection.cols[col];
        array[offset + col * 4] = v.x;
        array[offset + col * 4 + 1] = v.y;
        array[offset + col * 4 + 2] = v.z;
        array[offset + col * 4 + 3] = v.w;
    }
}

/// Recursively searches the scene tree for the first `DirectionalLight3D`.
fn find_directional_light() -> Option<Gd<DirectionalLight3D>> {
    let main_loop = Engine::singleton().get_main_loop()?;
    let tree = main_loop.try_cast::<SceneTree>().ok()?;
    let root = tree.get_root()?;
    find_directional_light_recursive(&root.upcast::<Node>())
}

fn find_directional_light_recursive(node: &Gd<Node>) -> Option<Gd<DirectionalLight3D>> {
    if let Ok(light) = node.clone().try_cast::<DirectionalLight3D>() {
        return Some(light);
    }
    let count = node.get_child_count();
    for i in 0..count {
        if let Some(child) = node.get_child(i) {
            if let Some(found) = find_directional_light_recursive(&child) {
                return Some(found);
            }
        }
    }
    None
}
