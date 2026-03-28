use crate::algo::run_algo::{CesRunAlgo, RunAlgoConfig};
use crate::layer_resources::{CesHeightLayerResource, CesTextureLayerResource};
use crate::layers::sphere_terrain::CesSphereTerrain;
use crate::layers::CesLayer;
use crate::texture_gen::CubemapTextureGen;
use godot::builtin::{
    PackedInt32Array, PackedVector2Array, PackedVector3Array, Transform3D, Variant, Vector2,
    Vector3,
};
use godot::classes::mesh::ArrayType;
use godot::classes::mesh::PrimitiveType;
use godot::classes::notify::Node3DNotification;
use godot::classes::{
    ArrayMesh, Camera3D, CollisionShape3D, ConcavePolygonShape3D, Engine, INode3D, Node3D,
    RenderingDevice, RenderingServer, Script, Shader, ShaderMaterial, StaticBody3D,
    TextureCubemapRd,
};
use godot::prelude::*;
use std::thread;
use std::time::{Duration, Instant};

/// Safely iterate a typed array, skipping null/nil entries.
/// This prevents panics when the user has added an array slot
/// but hasn't selected a subclass yet (the slot is &lt;empty&gt;/nil).
fn non_null_elements<T: GodotClass + Inherits<Resource>>(
    arr: &Array<Gd<T>>,
) -> Vec<Gd<T>> {
    let mut result = Vec::new();
    for i in 0..arr.len() {
        // Array::get calls from_variant which panics on nil → Gd<T>.
        // Catch the panic to gracefully skip null slots.
        if let Ok(Some(gd)) =
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| arr.get(i)))
        {
            result.push(gd);
        }
    }
    result
}

/// Check the GDScript class_name of a texture layer's attached script.
fn layer_script_class(layer: &Gd<CesTextureLayerResource>) -> StringName {
    let script_var = layer.get("script");
    if script_var.is_nil() {
        return StringName::default();
    }
    let script = script_var.to::<Gd<Script>>();
    script.get_global_name()
}

/// Compute a fingerprint of height + texture layers for change detection.
fn layers_fingerprint(
    height: &Array<Gd<CesHeightLayerResource>>,
    texture: &Array<Gd<CesTextureLayerResource>>,
) -> String {
    let mut parts = Vec::new();
    parts.push(format!("len:{}:{}", height.len(), texture.len()));
    for l in non_null_elements(height) {
        parts.push(format!("H:{}", l.bind().enabled));
    }
    for l in non_null_elements(texture) {
        let class = layer_script_class(&l);
        let enabled = l.bind().enabled;
        parts.push(format!("T:{}:{}", class, enabled));
    }
    parts.join(",")
}

#[derive(Clone, Copy, PartialEq)]
struct SettingsSnapshot {
    radius: f32,
    subdivisions: u32,
    triangle_screen_size: f32,
    low_poly_look: bool,
    precise_normals: bool,
    generate_collision: bool,
    show_debug_messages: bool,
    seed: i32,
}

struct MeshResult {
    pos: Vec<Vector3>,
    triangles: Vec<i32>,
    uv: Vec<Vector2>,
}

enum TextureCommand {
    Regenerate { size: u32, radius: f32 },
}

struct TextureResult {
    main_texture_rid: Rid,
    old_main_texture_rid: Rid,
}

#[derive(Clone)]
struct WorkerConfig {
    subdivisions: u32,
    radius: f32,
    triangle_screen_size: f32,
    precise_normals: bool,
    low_poly_look: bool,
    show_debug_messages: bool,
}

struct WorkerState {
    rd: Gd<RenderingDevice>,
    graph_generator: Option<CesRunAlgo>,
    layers: Vec<Box<dyn CesLayer>>,
    texture_gen: CubemapTextureGen,
}

struct ScopedTimer {
    label: &'static str,
    start: Instant,
    enabled: bool,
}

impl ScopedTimer {
    fn new(label: &'static str, enabled: bool) -> Self {
        Self {
            label,
            start: Instant::now(),
            enabled,
        }
    }
}

impl Drop for ScopedTimer {
    fn drop(&mut self) {
        if !self.enabled {
            return;
        }

        let elapsed_ms = self.start.elapsed().as_secs_f64() * 1000.0;
        // During hot-reload, Drop can run while Godot bindings are unavailable.
        // Never let timing logs panic inside a destructor.
        if std::panic::catch_unwind(|| {
            godot_print!("{} took {:.3} ms", self.label, elapsed_ms);
        })
        .is_err()
        {
            eprintln!("{} took {:.3} ms", self.label, elapsed_ms);
        }
    }
}

// SAFETY: WorkerState contains a local RenderingDevice created via
// RenderingServer::create_local_rendering_device(). Local RDs are independent
// and do not share state with the main renderer, making cross-thread use safe.
unsafe impl Send for WorkerState {}

#[derive(GodotClass)]
#[class(tool, base = Node3D)]
pub struct CesCelestialRust {
    base: Base<Node3D>,

    #[export]
    gameplay_camera: Option<Gd<Camera3D>>,

    #[export]
    radius: f32,

    #[export]
    subdivisions: u32,

    #[export]
    triangle_screen_size: f32,

    #[export]
    low_poly_look: bool,

    #[export]
    precise_normals: bool,

    #[export]
    generate_collision: bool,

    #[export]
    show_debug_messages: bool,

    #[export]
    show_process_timing: bool,

    #[export]
    simulated_process_delay_ms: u32,

    #[export]
    seed: i32,

    #[export]
    height_layers: Array<Gd<CesHeightLayerResource>>,

    #[export]
    texture_layers: Array<Gd<CesTextureLayerResource>>,

    instance: Rid,
    mesh: Option<Gd<ArrayMesh>>,
    active_shader: Option<Gd<Shader>>,
    cubemap_texture: Option<Gd<TextureCubemapRd>>,
    // Threading fields
    work_tx: Option<std::sync::mpsc::Sender<(Vector3, WorkerConfig)>>,
    result_rx: Option<std::sync::mpsc::Receiver<MeshResult>>,
    texture_cmd_tx: Option<std::sync::mpsc::Sender<TextureCommand>>,
    texture_result_rx: Option<std::sync::mpsc::Receiver<TextureResult>>,
    worker_handle: Option<std::thread::JoinHandle<WorkerState>>,
    gen_mesh_running: bool,
    last_cam_position: Vector3,
    last_obj_transform: Transform3D,
    last_settings: SettingsSnapshot,
    last_layers_fingerprint: String,
    values_updated: bool,
    is_shutting_down: bool,
    cubemap_resolution_active: u32,
}

#[godot_api]
impl INode3D for CesCelestialRust {
    fn init(base: Base<Node3D>) -> Self {
        Self {
            base,
            gameplay_camera: None,
            radius: 1.0,
            subdivisions: 3,
            triangle_screen_size: 0.1,
            low_poly_look: true,
            precise_normals: false,
            generate_collision: false,
            show_debug_messages: false,
            show_process_timing: false,
            simulated_process_delay_ms: 0,
            seed: 0,
            height_layers: Array::new(),
            texture_layers: Array::new(),
            instance: Rid::Invalid,
            mesh: None,
            active_shader: None,
            cubemap_texture: None,
            work_tx: None,
            result_rx: None,
            texture_cmd_tx: None,
            texture_result_rx: None,
            worker_handle: None,
            gen_mesh_running: false,
            last_cam_position: Vector3::ZERO,
            last_obj_transform: Transform3D::IDENTITY,
            last_settings: SettingsSnapshot {
                radius: 1.0,
                subdivisions: 3,
                triangle_screen_size: 0.1,
                low_poly_look: true,
                precise_normals: false,
                generate_collision: false,
                show_debug_messages: false,
                seed: 0,
            },
            values_updated: false,
            is_shutting_down: false,
            cubemap_resolution_active: 0,
            last_layers_fingerprint: String::new(),
        }
    }

    fn enter_tree(&mut self) {
        self.is_shutting_down = false;
        self.gen_mesh_running = false;
        self.values_updated = true;

        let mut rs = RenderingServer::singleton();
        self.instance = rs.instance_create();

        let scenario = self.base().get_world_3d().unwrap().get_scenario();
        rs.instance_set_scenario(self.instance, scenario);

        let mesh = ArrayMesh::new_gd();
        rs.instance_set_base(self.instance, mesh.get_rid());
        self.mesh = Some(mesh);

        // RD is now owned by the worker thread
        let mut rd = rs.create_local_rendering_device().unwrap();

        // Build runtime layers and texture gen from exported arrays
        let mut runtime_layers: Vec<Box<dyn CesLayer>> = Vec::new();
        let mut texture_gen = CubemapTextureGen::new();

        // Height layers
        if self.height_layers.is_empty() {
            // Default: include SphereTerrain when no height layers are configured
            runtime_layers.push(Box::new(CesSphereTerrain::new()));
        } else {
            for layer_gd in non_null_elements(&self.height_layers).into_iter() {
                if !layer_gd.bind().enabled {
                    continue;
                }
                runtime_layers.push(Box::new(CesSphereTerrain::new()));
            }
        }

        // Texture layers — last active layer's shader wins
        self.active_shader = None;
        for layer_gd in non_null_elements(&self.texture_layers).into_iter() {
            if !layer_gd.bind().enabled {
                continue;
            }
            // Every active texture layer can set a shader (defined in GDScript _init)
            let shader_var = layer_gd.get("shader");
            if !shader_var.is_nil() {
                self.active_shader = Some(shader_var.to::<Gd<Shader>>());
            }
            // Cubemap layers also generate a texture
            let script_class = layer_script_class(&layer_gd);
            if script_class == StringName::from("CesTextureLayer") {
                let resolution = layer_gd.get("resolution").to::<u32>();
                let mut gen = CubemapTextureGen::new();
                gen.create_shared_cubemap(&mut rd, resolution);
                let mut tex = TextureCubemapRd::new_gd();
                tex.set_texture_rd_rid(gen.main_texture_rid());
                self.cubemap_texture = Some(tex);
                texture_gen = gen;
            }
        }

        self.spawn_worker(rd, runtime_layers, texture_gen);
        self.cubemap_resolution_active = self.find_cubemap_resolution();
        self.last_layers_fingerprint =
            layers_fingerprint(&self.height_layers, &self.texture_layers);
    }

    fn ready(&mut self) {
        self.add_subnodes();
        self.last_settings = self.current_settings();
        self.values_updated = true;
    }

    fn process(&mut self, _delta: f64) {
        if self.is_shutting_down {
            return;
        }

        // Detect layer array changes (add/remove/reorder/enable toggle)
        let current_fp = layers_fingerprint(&self.height_layers, &self.texture_layers);
        if current_fp != self.last_layers_fingerprint {
            self.restart_with_current_layers();
            return;
        }

        // Check if cubemap resolution changed — send lightweight regenerate command
        if self.check_cubemap_restart_needed() {
            let new_resolution = self.find_cubemap_resolution();
            if let Some(ref tx) = self.texture_cmd_tx {
                let _ = tx.send(TextureCommand::Regenerate {
                    size: new_resolution,
                    radius: self.radius,
                });
            }
            self.cubemap_resolution_active = new_resolution;
            // Force a work dispatch so the worker wakes up and processes the texture command
            self.values_updated = true;
        }

        // Check for completed texture regeneration results
        if let Some(ref rx) = self.texture_result_rx {
            while let Ok(result) = rx.try_recv() {
                let mut tex = TextureCubemapRd::new_gd();
                tex.set_texture_rd_rid(result.main_texture_rid);
                self.cubemap_texture = Some(tex);
                // Free the old main texture now that it's no longer referenced by materials.
                if result.old_main_texture_rid.is_valid() {
                    let old_rid = result.old_main_texture_rid;
                    crate::compute_utils::on_render_thread(move || {
                        let mut main_rd =
                            RenderingServer::singleton().get_rendering_device().unwrap();
                        main_rd.free_rid(old_rid);
                    });
                }
                self.values_updated = true;
            }
        }

        if self.simulated_process_delay_ms > 0 {
            thread::sleep(Duration::from_millis(
                self.simulated_process_delay_ms as u64,
            ));
        }
        let mut show_timer = false;
        // 1. Check for completed results (non-blocking)
        if let Some(ref result_rx) = self.result_rx {
            if let Ok(result) = result_rx.try_recv() {
                self.gen_mesh_running = false;
                self.apply_mesh_result(result);
                show_timer = self.show_process_timing; // Show timing when we get a result
            }
        }

        let _process_timer = ScopedTimer::new("CesCelestialRust::process", show_timer);

        let global_transform = self.base().get_global_transform();
        let mut rs = RenderingServer::singleton();
        rs.instance_set_transform(self.instance, global_transform);

        // 2. Check if we need new work
        let cam = self.get_camera();
        if cam.is_none() {
            return;
        }
        let cam = cam.unwrap();
        let cam_pos = cam.get_global_position();
        let current_settings = self.current_settings();
        let has_changed = global_transform != self.last_obj_transform
            || cam_pos != self.last_cam_position
            || current_settings != self.last_settings
            || self.values_updated;

        if !has_changed || self.gen_mesh_running {
            return;
        }

        // 3. Update tracking state and dispatch work
        self.last_obj_transform = global_transform;
        self.last_cam_position = cam_pos;
        self.last_settings = current_settings;
        self.values_updated = false;

        let cam_local = global_transform.affine_inverse() * cam_pos;
        let config = WorkerConfig {
            subdivisions: self.subdivisions,
            radius: self.radius,
            triangle_screen_size: self.triangle_screen_size,
            precise_normals: self.precise_normals,
            low_poly_look: self.low_poly_look,
            show_debug_messages: self.show_debug_messages,
        };

        if let Some(ref work_tx) = self.work_tx {
            if work_tx.send((cam_local, config)).is_ok() {
                self.gen_mesh_running = true;
            } else {
                godot_print!("[CesCelestialRust] ERROR: Failed to send work to worker thread");
            }
        } else {
            godot_print!("[CesCelestialRust] ERROR: work_tx is None, worker not spawned");
        }
    }

    fn exit_tree(&mut self) {
        self.shutdown_for_reload_or_exit();
    }

    fn on_notification(&mut self, what: Node3DNotification) {
        if what == Node3DNotification::PREDELETE {
            self.shutdown_for_reload_or_exit();
        }
    }
}

impl CesCelestialRust {
    fn shutdown_for_reload_or_exit(&mut self) {
        if self.is_shutting_down {
            return;
        }
        self.is_shutting_down = true;

        // Free visible scene resources first.
        if std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let mut rs = RenderingServer::singleton();
            if self.instance.is_valid() {
                rs.free_rid(self.instance);
                self.instance = Rid::Invalid;
            }
            if let Some(ref mesh) = self.mesh {
                rs.free_rid(mesh.get_rid());
            }
            self.mesh = None;
        }))
        .is_err()
        {
            eprintln!("CesCelestialRust shutdown: failed to free scene RIDs (engine unavailable)");
            self.instance = Rid::Invalid;
            self.mesh = None;
        }

        // Signal worker to stop and stop accepting results.
        drop(self.work_tx.take());
        drop(self.texture_cmd_tx.take());
        self.result_rx = None;
        self.texture_result_rx = None;

        // Always join the worker before unload to avoid executing old code after DLL reload.
        if let Some(handle) = self.worker_handle.take() {
            match handle.join() {
                Ok(mut worker) => {
                    if let Some(ref mut gen) = worker.graph_generator {
                        gen.dispose_direct(&mut worker.rd);
                    }
                    for layer in worker.layers.iter_mut() {
                        layer.dispose_direct(&mut worker.rd);
                    }
                    // Flush any pending fire-and-forget render-thread callbacks
                    // before freeing the local RD they reference.
                    crate::compute_utils::on_render_thread_sync(|| {});
                    worker.texture_gen.dispose_local(&mut worker.rd);
                    worker.rd.free();
                }
                Err(_) => {
                    eprintln!("CesCelestialRust shutdown: worker thread panicked while joining");
                }
            }
        }
    }

    /// Restart the worker with the current layers configuration.
    /// Shuts down the existing worker, then re-runs the layer setup and spawn.
    fn restart_with_current_layers(&mut self) {
        // Shut down existing worker
        self.shutdown_for_reload_or_exit();

        // Re-initialize
        self.is_shutting_down = false;
        self.gen_mesh_running = false;
        self.values_updated = true;

        let mut rs = RenderingServer::singleton();
        self.instance = rs.instance_create();

        let scenario = self.base().get_world_3d().unwrap().get_scenario();
        rs.instance_set_scenario(self.instance, scenario);

        let mesh = ArrayMesh::new_gd();
        rs.instance_set_base(self.instance, mesh.get_rid());
        self.mesh = Some(mesh);

        let mut rd = rs.create_local_rendering_device().unwrap();

        let mut runtime_layers: Vec<Box<dyn CesLayer>> = Vec::new();
        let mut texture_gen = CubemapTextureGen::new();
        if self.height_layers.is_empty() {
            runtime_layers.push(Box::new(CesSphereTerrain::new()));
        } else {
            for layer_gd in non_null_elements(&self.height_layers).into_iter() {
                if !layer_gd.bind().enabled {
                    continue;
                }
                runtime_layers.push(Box::new(CesSphereTerrain::new()));
            }
        }

        self.active_shader = None;
        for layer_gd in non_null_elements(&self.texture_layers).into_iter() {
            if !layer_gd.bind().enabled {
                continue;
            }
            let shader_var = layer_gd.get("shader");
            if !shader_var.is_nil() {
                self.active_shader = Some(shader_var.to::<Gd<Shader>>());
            }
            let script_class = layer_script_class(&layer_gd);
            if script_class == StringName::from("CesTextureLayer") {
                let resolution = layer_gd.get("resolution").to::<u32>();
                let mut gen = CubemapTextureGen::new();
                gen.create_shared_cubemap(&mut rd, resolution);
                let mut tex = TextureCubemapRd::new_gd();
                tex.set_texture_rd_rid(gen.main_texture_rid());
                self.cubemap_texture = Some(tex);
                texture_gen = gen;
            }
        }

        self.spawn_worker(rd, runtime_layers, texture_gen);
        self.cubemap_resolution_active = self.find_cubemap_resolution();
        self.last_layers_fingerprint =
            layers_fingerprint(&self.height_layers, &self.texture_layers);
    }

    fn check_cubemap_restart_needed(&self) -> bool {
        for layer_gd in non_null_elements(&self.texture_layers).into_iter() {
            if !layer_gd.bind().enabled {
                continue;
            }
            if layer_script_class(&layer_gd) == StringName::from("CesTextureLayer") {
                let resolution = layer_gd.get("resolution").to::<u32>();
                return self.cubemap_resolution_active != 0
                    && resolution != self.cubemap_resolution_active;
            }
        }
        false
    }

    fn current_settings(&self) -> SettingsSnapshot {
        SettingsSnapshot {
            radius: self.radius,
            subdivisions: self.subdivisions,
            triangle_screen_size: self.triangle_screen_size,
            low_poly_look: self.low_poly_look,
            precise_normals: self.precise_normals,
            generate_collision: self.generate_collision,
            show_debug_messages: self.show_debug_messages,
            seed: self.seed,
        }
    }

    fn get_camera(&self) -> Option<Gd<Camera3D>> {
        if let Some(ref cam) = self.gameplay_camera {
            return Some(cam.clone());
        }

        if let Some(parent) = self.base().get_parent() {
            if let Some(camera) = parent.try_get_node_as::<Camera3D>("Camera3D") {
                return Some(camera);
            }
        }

        if Engine::singleton().is_editor_hint() {
            if let Some(camera) = self.base().get_viewport().and_then(|vp| vp.get_camera_3d()) {
                return Some(camera);
            }
        }

        self.base().get_viewport().and_then(|vp| vp.get_camera_3d())
    }

    fn apply_mesh_result(&mut self, result: MeshResult) {
        if result.pos.is_empty() || result.triangles.is_empty() {
            return;
        }

        let MeshResult { pos, triangles, uv } = result;
        let triangle_count = triangles.len() / 3;

        let packed_verts = PackedVector3Array::from(pos);
        let packed_indices = PackedInt32Array::from(triangles);
        let packed_uvs: PackedVector2Array = PackedVector2Array::from(uv);

        let mut surface_array = varray![];
        surface_array.resize(ArrayType::MAX.ord() as usize, &Variant::nil());
        surface_array.set(ArrayType::VERTEX.ord() as usize, &packed_verts.to_variant());
        surface_array.set(
            ArrayType::INDEX.ord() as usize,
            &packed_indices.to_variant(),
        );
        surface_array.set(ArrayType::TEX_UV.ord() as usize, &packed_uvs.to_variant());

        let mut new_mesh = ArrayMesh::new_gd();
        new_mesh.add_surface_from_arrays(PrimitiveType::TRIANGLES, &surface_array);

        if let Some(ref shader) = self.active_shader {
            let mut material = ShaderMaterial::new_gd();
            material.set_shader(shader);
            material.set_shader_parameter("radius", &self.radius.to_variant());
            if let Some(ref cubemap) = self.cubemap_texture {
                material.set_shader_parameter("planet_texture", &cubemap.to_variant());
            }
            new_mesh.surface_set_material(0, &material);
        }

        let mut rs = RenderingServer::singleton();
        rs.instance_set_base(self.instance, new_mesh.get_rid());

        if let Some(ref old_mesh) = self.mesh {
            rs.free_rid(old_mesh.get_rid());
        }
        self.mesh = Some(new_mesh);

        if self.show_debug_messages {
            godot_print!("Mesh Triangles: {}", triangle_count);
        }

        if self.generate_collision {
            self.update_collision();
        }
    }

    fn add_subnodes(&mut self) {
        // StaticBody3D with CollisionShape3D
        let mut static_body = StaticBody3D::new_alloc();
        static_body.set_name("StaticBody3D");

        let mut collision_shape = CollisionShape3D::new_alloc();
        collision_shape.set_name("CollisionShape3D");
        let shape = ConcavePolygonShape3D::new_gd();
        collision_shape.set_shape(&shape);

        static_body.add_child(&collision_shape);
        self.base_mut().add_child(&static_body);
    }

    /// Spawns the persistent worker thread. The worker owns the RenderingDevice,
    /// RunAlgo, and layers. It waits for work requests and sends back MeshResults.
    fn spawn_worker(
        &mut self,
        rd: Gd<RenderingDevice>,
        layers: Vec<Box<dyn CesLayer>>,
        texture_gen: CubemapTextureGen,
    ) {
        let (work_tx, work_rx) = std::sync::mpsc::channel::<(Vector3, WorkerConfig)>();
        let (result_tx, result_rx) = std::sync::mpsc::channel::<MeshResult>();
        let (texture_cmd_tx, texture_cmd_rx) = std::sync::mpsc::channel::<TextureCommand>();
        let (texture_result_tx, texture_result_rx) = std::sync::mpsc::channel::<TextureResult>();
        let radius = self.radius;
        // Find cubemap resolution from child node
        let cubemap_size = self.find_cubemap_resolution();

        let mut worker = WorkerState {
            rd,
            graph_generator: None,
            layers,
            texture_gen,
        };

        let handle = std::thread::spawn(move || {
            // Generate cubemap texture once at startup (only if a cubemap node was found)
            if worker.texture_gen.has_texture() {
                worker.texture_gen.init_pipeline(&mut worker.rd);
                worker
                    .texture_gen
                    .generate(&mut worker.rd, cubemap_size, radius);
            }

            while let Ok((cam_local, config)) = work_rx.recv() {
                // Process any pending texture regeneration commands first
                while let Ok(cmd) = texture_cmd_rx.try_recv() {
                    match cmd {
                        TextureCommand::Regenerate { size, radius } => {
                            let (new_rid, old_rid) =
                                worker.texture_gen.resize(&mut worker.rd, size, radius);
                            let _ = texture_result_tx.send(TextureResult {
                                main_texture_rid: new_rid,
                                old_main_texture_rid: old_rid,
                            });
                        }
                    }
                }
                let algo_config = RunAlgoConfig {
                    subdivisions: config.subdivisions,
                    radius: config.radius,
                    triangle_screen_size: config.triangle_screen_size,
                    precise_normals: config.precise_normals,
                    low_poly_look: config.low_poly_look,
                    show_debug_messages: config.show_debug_messages,
                };

                if worker.graph_generator.is_none() {
                    worker.graph_generator = Some(CesRunAlgo::new());
                }

                let gen = worker.graph_generator.as_mut().unwrap();
                let output = gen.update_triangle_graph(
                    &mut worker.rd,
                    cam_local,
                    &algo_config,
                    &mut worker.layers,
                    false,
                );

                if output.pos.is_empty() || output.tris.is_empty() {
                    let _ = result_tx.send(MeshResult {
                        pos: vec![],
                        triangles: vec![],
                        uv: vec![],
                    });
                    continue;
                }

                let result = MeshResult {
                    pos: output.pos,
                    triangles: output.tris,
                    uv: output.uv,
                };

                if result_tx.send(result).is_err() {
                    break; // Main thread dropped the receiver, exit
                }
            }

            // Return worker state for cleanup
            worker
        });

        self.work_tx = Some(work_tx);
        self.result_rx = Some(result_rx);
        self.texture_cmd_tx = Some(texture_cmd_tx);
        self.texture_result_rx = Some(texture_result_rx);
        self.worker_handle = Some(handle);
    }

    fn find_cubemap_resolution(&self) -> u32 {
        for layer_gd in non_null_elements(&self.texture_layers).into_iter() {
            if !layer_gd.bind().enabled {
                continue;
            }
            if layer_script_class(&layer_gd) == StringName::from("CesTextureLayer") {
                return layer_gd.get("resolution").to::<u32>();
            }
        }
        512
    }

    fn update_collision(&mut self) {
        if let Some(mesh) = self.mesh.as_ref() {
            let shape = mesh.create_trimesh_shape();
            if let Some(shape) = shape {
                let mut collision = self
                    .base()
                    .get_node_as::<CollisionShape3D>("StaticBody3D/CollisionShape3D");
                collision.set_shape(&shape);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::algo::run_algo::RunAlgoConfig;

    #[test]
    fn test_default_config() {
        let config = RunAlgoConfig {
            subdivisions: 3,
            radius: 1.0,
            triangle_screen_size: 0.1,
            precise_normals: true,
            low_poly_look: false,
            show_debug_messages: false,
        };
        assert_eq!(config.subdivisions, 3);
        assert!((config.radius - 1.0).abs() < f32::EPSILON);
        assert!((config.triangle_screen_size - 0.1).abs() < f32::EPSILON);
        assert!(config.precise_normals);
        assert!(!config.low_poly_look);
    }

    #[test]
    fn test_mesh_result_is_send() {
        fn assert_send<T: Send>() {}
        assert_send::<super::MeshResult>();
    }

    #[test]
    fn test_worker_config_is_send() {
        fn assert_send<T: Send>() {}
        assert_send::<super::WorkerConfig>();
    }

    #[test]
    fn test_worker_state_is_send() {
        fn assert_send<T: Send>() {}
        assert_send::<super::WorkerState>();
    }
}
