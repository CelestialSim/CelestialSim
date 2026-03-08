use crate::algo::run_algo::{CesRunAlgo, RunAlgoConfig};
use crate::layers::sphere_terrain::CesSphereTerrain;
use crate::layers::CesLayer;
use godot::builtin::{
    PackedInt32Array, PackedVector2Array, PackedVector3Array, Transform3D, Variant, Vector2,
    Vector3,
};
use godot::classes::mesh::ArrayType;
use godot::classes::mesh::PrimitiveType;
use godot::classes::{
    ArrayMesh, Camera3D, CollisionShape3D, ConcavePolygonShape3D, Engine, INode3D, Node3D,
    RenderingDevice, RenderingServer, Shader, ShaderMaterial, StaticBody3D,
};
use godot::prelude::*;

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
    normals: Vec<Vector3>,
    triangles: Vec<i32>,
    sim_value: Vec<[f32; 2]>,
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
    seed: i32,

    #[export]
    shader: Option<Gd<Shader>>,

    instance: Rid,
    mesh: Option<Gd<ArrayMesh>>,
    // Threading fields
    work_tx: Option<std::sync::mpsc::Sender<(Vector3, WorkerConfig)>>,
    result_rx: Option<std::sync::mpsc::Receiver<MeshResult>>,
    worker_handle: Option<std::thread::JoinHandle<WorkerState>>,
    gen_mesh_running: bool,
    last_cam_position: Vector3,
    last_obj_transform: Transform3D,
    last_settings: SettingsSnapshot,
    values_updated: bool,
    is_shutting_down: bool,
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
            seed: 0,
            shader: None,
            instance: Rid::Invalid,
            mesh: None,
            work_tx: None,
            result_rx: None,
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
        }
    }

    fn enter_tree(&mut self) {
        let mut rs = RenderingServer::singleton();
        self.instance = rs.instance_create();

        let scenario = self.base().get_world_3d().unwrap().get_scenario();
        rs.instance_set_scenario(self.instance, scenario);

        let mesh = ArrayMesh::new_gd();
        rs.instance_set_base(self.instance, mesh.get_rid());
        self.mesh = Some(mesh);

        // RD is now owned by the worker thread
        let rd = rs.create_local_rendering_device().unwrap();
        let layers: Vec<Box<dyn CesLayer>> = vec![Box::new(CesSphereTerrain::new())];
        self.spawn_worker(rd, layers);
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

        let global_transform = self.base().get_global_transform();
        let mut rs = RenderingServer::singleton();
        rs.instance_set_transform(self.instance, global_transform);

        // 1. Check for completed results (non-blocking)
        if let Some(ref result_rx) = self.result_rx {
            if let Ok(result) = result_rx.try_recv() {
                self.gen_mesh_running = false;
                self.apply_mesh_result(result);
            }
        }

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
        self.is_shutting_down = true;

        // Drop the sender to signal the worker thread to exit
        drop(self.work_tx.take());
        self.result_rx = None;

        // Join the worker thread with a timeout to recover the RenderingDevice.
        // The worker should be idle on recv() which will return Err immediately
        // once work_tx is dropped.
        if let Some(handle) = self.worker_handle.take() {
            let (done_tx, done_rx) = std::sync::mpsc::channel();
            std::thread::spawn(move || {
                let worker = handle.join();
                let _ = done_tx.send(worker);
            });
            match done_rx.recv_timeout(std::time::Duration::from_secs(2)) {
                Ok(Ok(mut worker)) => {
                    // Free the local RenderingDevice explicitly before engine teardown
                    if let Some(ref mut gen) = worker.graph_generator {
                        gen.dispose(&mut worker.rd);
                    }
                    worker.rd.free();
                }
                _ => {
                    godot_warn!("Worker thread did not shut down in time; GPU resources may leak");
                }
            }
        }

        let mut rs = RenderingServer::singleton();
        if self.instance.is_valid() {
            rs.free_rid(self.instance);
            self.instance = Rid::Invalid;
        }
        if let Some(ref mesh) = self.mesh {
            rs.free_rid(mesh.get_rid());
        }
        self.mesh = None;
    }
}

impl CesCelestialRust {
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

        let mut packed_verts = PackedVector3Array::new();
        for v in result.pos.iter() {
            packed_verts.push(*v);
        }

        let mut packed_normals = PackedVector3Array::new();
        for n in result.normals.iter() {
            packed_normals.push(*n);
        }

        let mut packed_indices = PackedInt32Array::new();
        for idx in result.triangles.iter() {
            packed_indices.push(*idx);
        }

        let mut packed_uvs = PackedVector2Array::new();
        for uv in result.sim_value.iter() {
            packed_uvs.push(Vector2::new(uv[0], uv[1]));
        }

        let mut surface_array = varray![];
        surface_array.resize(ArrayType::MAX.ord() as usize, &Variant::nil());
        surface_array.set(ArrayType::VERTEX.ord() as usize, &packed_verts.to_variant());
        surface_array.set(
            ArrayType::NORMAL.ord() as usize,
            &packed_normals.to_variant(),
        );
        surface_array.set(
            ArrayType::INDEX.ord() as usize,
            &packed_indices.to_variant(),
        );
        surface_array.set(ArrayType::TEX_UV.ord() as usize, &packed_uvs.to_variant());

        let mut new_mesh = ArrayMesh::new_gd();
        new_mesh.add_surface_from_arrays(PrimitiveType::TRIANGLES, &surface_array);

        if let Some(ref shader) = self.shader {
            let mut material = ShaderMaterial::new_gd();
            material.set_shader(shader);
            material.set_shader_parameter("radius", &self.radius.to_variant());
            new_mesh.surface_set_material(0, &material);
        }

        let mut rs = RenderingServer::singleton();
        rs.instance_set_base(self.instance, new_mesh.get_rid());

        if let Some(ref old_mesh) = self.mesh {
            rs.free_rid(old_mesh.get_rid());
        }
        self.mesh = Some(new_mesh);

        if self.show_debug_messages {
            godot_print!("Mesh Triangles: {}", result.triangles.len() / 3);
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
    fn spawn_worker(&mut self, rd: Gd<RenderingDevice>, layers: Vec<Box<dyn CesLayer>>) {
        let (work_tx, work_rx) = std::sync::mpsc::channel::<(Vector3, WorkerConfig)>();
        let (result_tx, result_rx) = std::sync::mpsc::channel::<MeshResult>();

        let mut worker = WorkerState {
            rd,
            graph_generator: None,
            layers,
        };

        let handle = std::thread::spawn(move || {
            while let Ok((cam_local, config)) = work_rx.recv() {
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
                gen.update_triangle_graph(
                    &mut worker.rd,
                    cam_local,
                    &algo_config,
                    &mut worker.layers,
                    false,
                );

                if gen.pos.is_empty() || gen.triangles.is_empty() {
                    let _ = result_tx.send(MeshResult {
                        pos: vec![],
                        normals: vec![],
                        triangles: vec![],
                        sim_value: vec![],
                    });
                    continue;
                }

                let result = MeshResult {
                    pos: gen.pos.clone(),
                    normals: gen.normals.clone(),
                    triangles: gen.triangles.clone(),
                    sim_value: gen.sim_value.clone(),
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
        self.worker_handle = Some(handle);
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
