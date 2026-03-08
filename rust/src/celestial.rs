use godot::builtin::{
    PackedInt32Array, PackedVector2Array, PackedVector3Array, Transform3D, Variant, Vector2,
    Vector3,
};
use godot::classes::mesh::ArrayType;
use godot::classes::mesh::PrimitiveType;
use godot::classes::{
    ArrayMesh, Camera3D, ConcavePolygonShape3D, CollisionShape3D, INode3D, Node3D,
    RenderingDevice, RenderingServer, Shader, ShaderMaterial, StaticBody3D,
};
use godot::prelude::*;

use crate::algo::run_algo::{CesRunAlgo, RunAlgoConfig};
use crate::layers::sphere_terrain::CesSphereTerrain;
use crate::layers::CesLayer;

#[derive(GodotClass)]
#[class(base = Node3D)]
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
    rd: Option<Gd<RenderingDevice>>,
    graph_generator: Option<CesRunAlgo>,
    layers: Vec<Box<dyn CesLayer>>,
    last_cam_position: Vector3,
    last_obj_transform: Transform3D,
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
            precise_normals: true,
            generate_collision: false,
            show_debug_messages: false,
            seed: 0,
            shader: None,
            instance: Rid::Invalid,
            mesh: None,
            rd: None,
            graph_generator: None,
            layers: Vec::new(),
            last_cam_position: Vector3::ZERO,
            last_obj_transform: Transform3D::IDENTITY,
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

        self.rd = Some(rs.create_local_rendering_device().unwrap());
    }

    fn ready(&mut self) {
        self.add_subnodes();
        self.layers = vec![Box::new(CesSphereTerrain::new())];
    }

    fn process(&mut self, _delta: f64) {
        if self.is_shutting_down {
            return;
        }

        let global_transform = self.base().get_global_transform();
        let mut rs = RenderingServer::singleton();
        rs.instance_set_transform(self.instance, global_transform);

        let cam = self.get_camera();
        if cam.is_none() {
            return;
        }
        let cam = cam.unwrap();

        let cam_pos = cam.get_global_position();
        let has_changed = global_transform != self.last_obj_transform
            || cam_pos != self.last_cam_position
            || self.values_updated;

        if !has_changed {
            return;
        }

        self.last_obj_transform = global_transform;
        self.last_cam_position = cam_pos;
        self.values_updated = false;

        let cam_local = global_transform.affine_inverse() * cam_pos;
        self.gen_mesh(cam_local);
    }

    fn exit_tree(&mut self) {
        self.is_shutting_down = true;

        if let Some(ref mut rd) = self.rd {
            if let Some(ref mut gen) = self.graph_generator {
                gen.dispose(rd);
            }
        }
        self.graph_generator = None;

        let mut rs = RenderingServer::singleton();
        if self.instance.is_valid() {
            rs.free_rid(self.instance);
            self.instance = Rid::Invalid;
        }
        if let Some(ref mesh) = self.mesh {
            rs.free_rid(mesh.get_rid());
        }
        self.mesh = None;
        self.rd = None;
        self.layers.clear();
    }
}

impl CesCelestialRust {
    fn get_camera(&self) -> Option<Gd<Camera3D>> {
        if let Some(ref cam) = self.gameplay_camera {
            return Some(cam.clone());
        }
        self.base().get_viewport().and_then(|vp| vp.get_camera_3d())
    }

    fn gen_mesh(&mut self, cam_local: Vector3) {
        let config = RunAlgoConfig {
            subdivisions: self.subdivisions,
            radius: self.radius,
            triangle_screen_size: self.triangle_screen_size,
            precise_normals: self.precise_normals,
            low_poly_look: self.low_poly_look,
            show_debug_messages: self.show_debug_messages,
        };

        if self.graph_generator.is_none() {
            self.graph_generator = Some(CesRunAlgo::new());
        }

        let rd = self.rd.as_mut().unwrap();
        let gen = self.graph_generator.as_mut().unwrap();
        gen.update_triangle_graph(rd, cam_local, &config, &mut self.layers, false);

        if gen.pos.is_empty() || gen.triangles.is_empty() {
            return;
        }

        // Build packed arrays
        let mut packed_verts = PackedVector3Array::new();
        for v in gen.pos.iter() {
            packed_verts.push(*v);
        }

        let mut packed_normals = PackedVector3Array::new();
        for n in gen.normals.iter() {
            packed_normals.push(*n);
        }

        let mut packed_indices = PackedInt32Array::new();
        for idx in gen.triangles.iter() {
            packed_indices.push(*idx);
        }

        let mut packed_uvs = PackedVector2Array::new();
        for uv in gen.sim_value.iter() {
            packed_uvs.push(Vector2::new(uv[0], uv[1]));
        }

        // Build surface array
        let mut surface_array = varray![];
        surface_array.resize(ArrayType::MAX.ord() as usize, &Variant::nil());
        surface_array.set(ArrayType::VERTEX.ord() as usize, &packed_verts.to_variant());
        surface_array.set(ArrayType::NORMAL.ord() as usize, &packed_normals.to_variant());
        surface_array.set(ArrayType::INDEX.ord() as usize, &packed_indices.to_variant());
        surface_array.set(ArrayType::TEX_UV.ord() as usize, &packed_uvs.to_variant());

        let mut new_mesh = ArrayMesh::new_gd();
        new_mesh.add_surface_from_arrays(PrimitiveType::TRIANGLES, &surface_array);

        if let Some(ref shader) = self.shader {
            let mut material = ShaderMaterial::new_gd();
            material.set_shader(shader);
            material.set_shader_parameter("radius", &self.radius.to_variant());
            new_mesh.surface_set_material(0, &material);
        }

        // Swap mesh on rendering server
        let mut rs = RenderingServer::singleton();
        rs.instance_set_base(self.instance, new_mesh.get_rid());

        if let Some(ref old_mesh) = self.mesh {
            rs.free_rid(old_mesh.get_rid());
        }
        self.mesh = Some(new_mesh);

        if self.show_debug_messages {
            godot_print!("Mesh triangles: {}", gen.triangles.len() / 3);
        }

        // Collision
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
}
