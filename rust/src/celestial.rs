use crate::algo::run_algo::{CesRunAlgo, RunAlgoConfig};
use crate::camera_snapshot_texture::CameraSnapshotTexture;
use crate::layer_resources::{
    CesHeightLayerResource, CesScatterLayerResource, CesTextureLayerResource,
};
use crate::layers::height_shader_terrain::CesHeightShaderTerrain;
use crate::layers::scatter::{CesScatterRuntime, ScatterParams, DEFAULT_SCATTER_SHADER_PATH};
use crate::layers::sphere_terrain::CesSphereTerrain;
use crate::layers::CesLayer;
use crate::texture_gen::{CubemapTextureGen, TerrainParams};
use godot::builtin::Callable;
use godot::builtin::{
    Color, PackedFloat32Array, PackedInt32Array, PackedVector2Array, PackedVector3Array,
    Transform3D, Variant, Vector2, Vector3,
};
use godot::classes::mesh::ArrayType;
use godot::classes::mesh::PrimitiveType;
use godot::classes::multi_mesh::TransformFormat;
use godot::classes::notify::Node3DNotification;
use godot::classes::{
    ArrayMesh, Camera3D, CollisionShape3D, ConcavePolygonShape3D, Engine, INode3D, Image,
    ImageTexture3D, Material, Mesh, MultiMesh, MultiMeshInstance3D, Node, Node3D,
    RenderingDevice, RenderingServer, Script, Shader, ShaderMaterial, StaticBody3D, Texture2Drd,
    TextureCubemapRd,
};
use godot::prelude::*;
use std::thread;
use std::time::{Duration, Instant};

const TERRAIN_NORMAL_SHADER: &str = "res://addons/celestial_sim/shaders/TerrainNormal.slang";

/// Safely iterate a typed array, skipping null/nil entries.
/// This prevents panics when the user has added an array slot
/// but hasn't selected a subclass yet (the slot is <empty>/nil).
fn non_null_elements<T: GodotClass + Inherits<Resource>>(arr: &Array<Gd<T>>) -> Vec<Gd<T>> {
    // Deref to AnyArray so that get(i) returns Option<Variant> without calling
    // from_variant — avoiding the panic that gdext fires on nil object variants.
    let any: &godot::builtin::AnyArray = arr;
    let mut result = Vec::new();
    for i in 0..any.len() {
        let Some(variant) = any.get(i) else { continue };
        let Ok(gd) = variant.try_to::<Gd<T>>() else { continue };
        result.push(gd);
    }
    result
}

/// Pure helper: counts how many booleans in the slice are `true`.
/// Extracted so we can unit-test the core logic without a Godot runtime.
fn count_enabled_flags(flags: &[bool]) -> usize {
    flags.iter().filter(|&&x| x).count()
}

/// Build placeholder scatter results sized to the number of enabled layers.
/// This keeps worker outputs index-aligned with runtimes and MultiMesh children
/// even when a given frame generates zero instances.
fn zeroed_scatter_results(slot_count: usize) -> (Vec<Vec<f32>>, Vec<u32>) {
    (vec![Vec::new(); slot_count], vec![0; slot_count])
}

/// Counts the number of enabled scatter layers in a typed array, skipping null slots.
fn count_enabled_scatter_layers(arr: &Array<Gd<CesScatterLayerResource>>) -> usize {
    let flags: Vec<bool> = non_null_elements(arr)
        .iter()
        .map(|l| l.bind().enabled)
        .collect();
    count_enabled_flags(&flags)
}

/// Finalises runtime scatter params with the active planet radius.
fn scatter_params_with_runtime_context(
    mut params: ScatterParams,
    radius: f32,
) -> ScatterParams {
    params.planet_radius = radius;
    params
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct ScatterLayerFieldValues {
    seed: i64,
    subdivision_level: i64,
    noise_strength: f32,
    height_min: f32,
    height_max: f32,
    albedo_target: [f32; 3],
    albedo_tolerance: f32,
}

fn scatter_params_from_layer_fields(fields: ScatterLayerFieldValues) -> ScatterParams {
    let defaults = ScatterParams::default();
    ScatterParams {
        seed: fields.seed as u32,
        subdivision_level: fields.subdivision_level.clamp(0, i32::MAX as i64) as i32,
        noise_strength: fields.noise_strength,
        height_min: fields.height_min,
        height_max: fields.height_max,
        albedo_tolerance: fields.albedo_tolerance,
        albedo_target: fields.albedo_target,
        ..defaults
    }
}

/// Read scatter params from a CesScatterLayer GDScript resource.
/// The active celestial radius is injected separately via
/// `scatter_params_with_runtime_context` when building runtime params.
fn read_scatter_layer_params(layer_gd: &Gd<CesScatterLayerResource>) -> ScatterParams {
    let defaults = ScatterParams::default();
    let seed_var = layer_gd.get("seed");
    let subdivision_level_var = layer_gd.get("subdivision_level");
    let noise_strength_var = layer_gd.get("noise_strength");
    let height_min_var = layer_gd.get("height_min");
    let height_max_var = layer_gd.get("height_max");
    let albedo_target_var = layer_gd.get("albedo_target");
    let albedo_tolerance_var = layer_gd.get("albedo_tolerance");

    let albedo_target = if albedo_target_var.is_nil() {
        defaults.albedo_target
    } else {
        let color = albedo_target_var.to::<Color>();
        [color.r, color.g, color.b]
    };

    scatter_params_from_layer_fields(ScatterLayerFieldValues {
        seed: if seed_var.is_nil() {
            defaults.seed as i64
        } else {
            seed_var.to::<i64>()
        },
        subdivision_level: if subdivision_level_var.is_nil() {
            defaults.subdivision_level as i64
        } else {
            subdivision_level_var.to::<i64>()
        },
        noise_strength: if noise_strength_var.is_nil() {
            defaults.noise_strength
        } else {
            noise_strength_var.to::<f32>()
        },
        height_min: if height_min_var.is_nil() {
            defaults.height_min
        } else {
            height_min_var.to::<f32>()
        },
        height_max: if height_max_var.is_nil() {
            defaults.height_max
        } else {
            height_max_var.to::<f32>()
        },
        albedo_target,
        albedo_tolerance: if albedo_tolerance_var.is_nil() {
            defaults.albedo_tolerance
        } else {
            albedo_tolerance_var.to::<f32>()
        },
    })
}

fn read_enabled_scatter_layer_params(
    arr: &Array<Gd<CesScatterLayerResource>>,
) -> Vec<ScatterParams> {
    non_null_elements(arr)
        .into_iter()
        .filter(|layer_gd| layer_gd.bind().enabled)
        .map(|layer_gd| read_scatter_layer_params(&layer_gd))
        .collect()
}

/// Check the GDScript class_name of a texture layer's attached script.
fn layer_script_class(layer: &Gd<CesTextureLayerResource>) -> StringName {
    let script_var = layer.get("script");
    // Use try_to instead of is_nil() + to(): a null TYPE_OBJECT variant
    // passes is_nil() == false but panics on to::<Gd<T>>().
    if let Ok(script) = script_var.try_to::<Gd<Script>>() {
        script.get_global_name()
    } else {
        StringName::default()
    }
}

/// Formats a single scatter layer's contribution to the structure id.
/// Includes subdivision_level and noise_strength so changes trigger
/// a rebuild via the structure id system.
fn scatter_structure_part(
    instance_id: i64,
    enabled: bool,
    subdivision_level: i32,
    noise_strength: f32,
) -> String {
    format!(
        "S:{}:{}:{}:{}",
        instance_id, enabled, subdivision_level, noise_strength
    )
}

/// Compute a lightweight structural ID of height + texture + scatter layers.
/// Only tracks array lengths, instance IDs, class names, and enabled flags.
/// Property values are NOT included — the `Resource.changed` signal handles that.
fn layers_structure_id(
    height: &Array<Gd<CesHeightLayerResource>>,
    texture: &Array<Gd<CesTextureLayerResource>>,
    scatter: &Array<Gd<CesScatterLayerResource>>,
) -> String {
    let mut parts = Vec::new();
    parts.push(format!(
        "len:{}:{}:{}",
        height.len(),
        texture.len(),
        scatter.len()
    ));
    for l in non_null_elements(height) {
        parts.push(format!("H:{}:{}", l.instance_id(), l.bind().enabled));
    }
    for l in non_null_elements(texture) {
        let class = layer_script_class(&l);
        // Include structural properties so changing them triggers a full restart.
        // Use Display on Variant to avoid panics when the property doesn't exist.
        let resolution = l.get("resolution");
        let gen_normals = l.get("generate_normal_map");
        let shader_path = l.get("compute_shader_path");
        let max_snapshot = l.get("max_snapshot_levels");
        let snapshot_shader = l.get("snapshot_color_shader");
        parts.push(format!(
            "T:{}:{}:{}:{}:{}:{}:{}:{}",
            l.instance_id(),
            class,
            l.bind().enabled,
            resolution,
            gen_normals,
            shader_path,
            max_snapshot,
            snapshot_shader
        ));
    }
    for l in non_null_elements(scatter) {
        let params = read_scatter_layer_params(&l);
        parts.push(scatter_structure_part(
            l.instance_id().to_i64(),
            l.bind().enabled,
            params.subdivision_level,
            params.noise_strength,
        ));
    }
    parts.join(",")
}

/// Default size for the blue-noise volume used by the scatter mask layer.
///
/// 32³ generates in ~20 s on a single CPU core; 64³ would take ~30+ minutes
/// because the void-and-cluster implementation does a full 3D FFT per
/// add/remove iteration. The first call writes a cache file; subsequent calls
/// (same size+seed) load it instantly.
const BLUE_NOISE_SIZE: [u32; 3] = [32, 32, 32];

/// Resolve the on-disk cache path for the project-local blue-noise volume.
fn blue_noise_cache_path() -> std::path::PathBuf {
    let dir = std::env::current_dir()
        .unwrap_or_else(|_| std::path::PathBuf::from("."))
        .join(".cache");
    let [w, h, d] = BLUE_NOISE_SIZE;
    dir.join(format!(
        "blue_noise_{}x{}x{}_seed{:#x}.bin",
        w,
        h,
        d,
        crate::blue_noise::DEFAULT_SEED
    ))
}

/// Load the blue-noise bytes from cache (or generate + cache on first run).
/// Called both by the mask layer (for the `Texture3D` preview) and by the
/// scatter runtime (for the compute-shader storage buffer). Logs progress.
fn load_blue_noise_bytes() -> Vec<u8> {
    use crate::blue_noise::{load_or_generate, DEFAULT_SEED};
    let path = blue_noise_cache_path();
    godot_print!(
        "[CesCelestialRust] Loading blue-noise volume {}x{}x{} from {} (will generate if missing — first generation takes ~20 s)",
        BLUE_NOISE_SIZE[0],
        BLUE_NOISE_SIZE[1],
        BLUE_NOISE_SIZE[2],
        path.display()
    );
    let t0 = std::time::Instant::now();
    let bytes = load_or_generate(&path, BLUE_NOISE_SIZE, DEFAULT_SEED);
    godot_print!(
        "[CesCelestialRust] Blue-noise volume ready in {:.2}s ({} bytes)",
        t0.elapsed().as_secs_f64(),
        bytes.len()
    );
    bytes
}

/// Wrap raw blue-noise bytes in a Godot `ImageTexture3D` for use as a planet
/// shader uniform (mask preview).
fn build_blue_noise_texture_from_bytes(bytes: &[u8]) -> Gd<ImageTexture3D> {
    use godot::classes::image::Format;
    let [w, h, d] = BLUE_NOISE_SIZE;
    let mut images: Array<Gd<Image>> = Array::new();
    for z in 0..d {
        let start = (z * w * h) as usize;
        let end = start + (w * h) as usize;
        let packed = PackedByteArray::from(bytes[start..end].to_vec());
        let mut img = Image::new_gd();
        img.set_data(w as i32, h as i32, false, Format::R8, &packed);
        images.push(&img);
    }
    let mut tex = ImageTexture3D::new_gd();
    let _ = tex.create(Format::R8, w as i32, h as i32, d as i32, false, &images);
    tex
}

/// Read terrain params from a CesTerrainTextureLayer GDScript resource.
fn read_terrain_params(layer: &Gd<CesTextureLayerResource>) -> TerrainParams {
    TerrainParams {
        height_tiles: layer.get("height_tiles").to::<f32>(),
        height_octaves: layer.get("height_octaves").to::<i32>(),
        height_amp: layer.get("height_amp").to::<f32>(),
        height_gain: layer.get("height_gain").to::<f32>(),
        height_lacunarity: layer.get("height_lacunarity").to::<f32>(),
        erosion_tiles: layer.get("erosion_tiles").to::<f32>(),
        erosion_octaves: layer.get("erosion_octaves").to::<i32>(),
        erosion_gain: layer.get("erosion_gain").to::<f32>(),
        erosion_lacunarity: layer.get("erosion_lacunarity").to::<f32>(),
        erosion_slope_strength: layer.get("erosion_slope_strength").to::<f32>(),
        erosion_branch_strength: layer.get("erosion_branch_strength").to::<f32>(),
        erosion_strength: layer.get("erosion_strength").to::<f32>(),
        water_height: layer.get("water_height").to::<f32>(),
        _pad: [0.0; 3],
    }
}

fn read_show_snapshot_borders(
    texture: &Array<Gd<CesTextureLayerResource>>,
    fallback: bool,
) -> bool {
    for layer_gd in non_null_elements(texture).into_iter() {
        if !layer_gd.bind().enabled {
            continue;
        }
        let class = layer_script_class(&layer_gd);
        if class == StringName::from("CesTextureLayer")
            || class == StringName::from("CesTerrainTextureLayer")
        {
            let value = layer_gd.get("show_snapshot_borders");
            if !value.is_nil() {
                return value.to::<bool>();
            }
            return fallback;
        }
    }
    fallback
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
    debug_snapshot_angle_offset: f32,
    show_snapshot_borders: bool,
}

struct MeshResult {
    pos: Vec<Vector3>,
    triangles: Vec<i32>,
    uv: Vec<Vector2>,
    /// Snapshot data (bundled with mesh result). Empty if no snapshots.
    snapshot_main_rids: Vec<Rid>,
    snapshot_normal_main_rids: Vec<Rid>,
    snapshot_level_info: Vec<(Vector3, Vector3, Vector3, f32)>,
    snapshot_updated: bool,
    /// One transform buffer per enabled scatter layer (same order as enabled layers).
    /// Empty when no scatter work was done.
    scatter_buffers: Vec<Vec<f32>>,
    /// Number of transforms written per scatter layer (n_tris * density).
    /// Parallel to `scatter_buffers`.
    scatter_instance_counts: Vec<u32>,
}

enum TextureCommand {
    Regenerate {
        size: u32,
        radius: f32,
        terrain_params: Option<TerrainParams>,
    },
    ParamsChanged {
        terrain_params: TerrainParams,
        radius: f32,
    },
}

struct TextureResult {
    main_texture_rid: Rid,
    old_main_texture_rid: Rid,
    normal_main_texture_rid: Rid,
    normal_old_main_texture_rid: Rid,
}

#[derive(Clone)]
struct WorkerConfig {
    subdivisions: u32,
    radius: f32,
    triangle_screen_size: f32,
    precise_normals: bool,
    low_poly_look: bool,
    show_debug_messages: bool,
    debug_snapshot_angle_offset: f32,
    show_snapshot_borders: bool,
    scatter_params: Vec<ScatterParams>,
}

struct WorkerState {
    rd: Gd<RenderingDevice>,
    graph_generator: Option<CesRunAlgo>,
    layers: Vec<Box<dyn CesLayer>>,
    texture_gen: CubemapTextureGen,
    normal_texture_gen: CubemapTextureGen,
    terrain_params: Option<TerrainParams>,
    snapshot_chain: CameraSnapshotTexture,
    /// Parallel to the enabled scatter layers (same order). One runtime per layer.
    scatter_runtimes: Vec<CesScatterRuntime>,
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
    use_editor_camera: bool,

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

    /// Debug: angular offset (radians) applied to snapshot center direction.
    /// Set non-zero to animate snapshot positions independently of camera.
    #[export]
    debug_snapshot_angle_offset: f32,

    /// Show red debug borders on LOD snapshot patches.
    #[export]
    show_snapshot_borders: bool,

    #[export]
    height_layers: Array<Gd<CesHeightLayerResource>>,

    #[export]
    texture_layers: Array<Gd<CesTextureLayerResource>>,

    #[export]
    scatter_layers: Array<Gd<CesScatterLayerResource>>,

    instance: Rid,
    mesh: Option<Gd<ArrayMesh>>,
    active_shader: Option<Gd<Shader>>,
    cubemap_texture: Option<Gd<TextureCubemapRd>>,
    normal_cubemap_texture: Option<Gd<TextureCubemapRd>>,
    snapshot_textures: Vec<Option<Gd<Texture2Drd>>>,
    snapshot_normal_textures: Vec<Option<Gd<Texture2Drd>>>,
    snapshot_info: Vec<(Vector3, Vector3, Vector3, f32)>,
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
    last_structure_id: String,
    values_updated: bool,
    is_shutting_down: bool,
    cubemap_resolution_active: u32,
    /// Set by `Resource.changed` signal handler; indicates layer properties changed.
    layers_dirty: bool,
    /// True when dirty flag was caused by a structural change (add/remove/reorder layers).
    structural_dirty: bool,
    /// Instance IDs of layers whose "changed" signal we are currently connected to.
    connected_layer_ids: Vec<i64>,
    /// One `MultiMeshInstance3D` child per enabled scatter layer.
    /// `None` for layers without a mesh (index alignment preserved).
    scatter_mmi_nodes: Vec<Option<Gd<MultiMeshInstance3D>>>,
    /// Blue-noise Texture3D built once per active `CesScatterMaskTextureLayer`.
    blue_noise_texture: Option<Gd<ImageTexture3D>>,
    /// Raw R8 blue-noise bytes (length = noise_dim^3). Loaded once and shared
    /// between the mask Texture3D and every scatter runtime's GPU buffer.
    blue_noise_bytes: Vec<u8>,
    /// Scatter mask preview parameters forwarded to the planet shader material.
    scatter_mask_noise_strength: f32,
    scatter_mask_target_color: Color,
    scatter_mask_color_tolerance: f32,
}

#[godot_api]
impl INode3D for CesCelestialRust {
    fn init(base: Base<Node3D>) -> Self {
        Self {
            base,
            gameplay_camera: None,
            use_editor_camera: true,
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
            debug_snapshot_angle_offset: 0.0,
            show_snapshot_borders: true,
            height_layers: Array::new(),
            texture_layers: Array::new(),
            scatter_layers: Array::new(),
            instance: Rid::Invalid,
            mesh: None,
            active_shader: None,
            cubemap_texture: None,
            normal_cubemap_texture: None,
            snapshot_textures: Vec::new(),
            snapshot_normal_textures: Vec::new(),
            snapshot_info: Vec::new(),

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
                debug_snapshot_angle_offset: 0.0,
                show_snapshot_borders: true,
            },
            values_updated: false,
            is_shutting_down: false,
            cubemap_resolution_active: 0,
            last_structure_id: String::new(),
            layers_dirty: false,
            structural_dirty: false,
            connected_layer_ids: Vec::new(),
            scatter_mmi_nodes: Vec::new(),
            blue_noise_texture: None,
            blue_noise_bytes: Vec::new(),
            scatter_mask_noise_strength: 1.0,
            scatter_mask_target_color: Color::from_rgba(0.2, 0.6, 0.2, 1.0),
            scatter_mask_color_tolerance: 0.3,
        }
    }

    fn enter_tree(&mut self) {
        if Engine::singleton().is_editor_hint() {
            self.is_shutting_down = false;
            self.gen_mesh_running = false;
            self.values_updated = true;
            return;
        }

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
        let mut normal_texture_gen = CubemapTextureGen::new();
        let mut terrain_params: Option<TerrainParams> = None;

        // Height layers
        if self.height_layers.is_empty() {
            // Default: include SphereTerrain when no height layers are configured
            runtime_layers.push(Box::new(CesSphereTerrain::new()));
        } else {
            for layer_gd in non_null_elements(&self.height_layers).into_iter() {
                if !layer_gd.bind().enabled {
                    continue;
                }
                let shader_path = layer_gd.bind().shader_path.to_string();
                if shader_path.is_empty() {
                    runtime_layers.push(Box::new(CesSphereTerrain::new()));
                } else {
                    runtime_layers.push(Box::new(CesHeightShaderTerrain::with_shader_path(
                        &shader_path,
                    )));
                }
            }
        }

        // Texture layers — last active layer's shader wins
        self.active_shader = None;
        let mut snapshot_chain = CameraSnapshotTexture::new(0, 512, None, None);
        godot_print!(
            "[CesCelestialRust] Processing {} texture layers",
            self.texture_layers.len()
        );
        for layer_gd in non_null_elements(&self.texture_layers).into_iter() {
            if !layer_gd.bind().enabled {
                godot_print!("[CesCelestialRust] Skipping disabled texture layer");
                continue;
            }
            // Every active texture layer can set a shader (defined in GDScript _init)
            let shader_var = layer_gd.get("shader");
            if let Ok(shader) = shader_var.try_to::<Gd<Shader>>() {
                self.active_shader = Some(shader);
            }
            // Cubemap layers also generate a texture
            let script_class = layer_script_class(&layer_gd);
            godot_print!(
                "[CesCelestialRust] Texture layer class: {}",
                script_class.to_string()
            );
            if script_class == StringName::from("CesTextureLayer") {
                let resolution = layer_gd.get("resolution").to::<u32>();
                let mut gen = CubemapTextureGen::new();
                gen.create_shared_cubemap(&mut rd, resolution);
                let mut tex = TextureCubemapRd::new_gd();
                tex.set_texture_rd_rid(gen.main_texture_rid());
                self.cubemap_texture = Some(tex);
                texture_gen = gen;
                godot_print!(
                    "[CesCelestialRust] Created CesTextureLayer cubemap, resolution={}",
                    resolution
                );
                // LOD chain for CesTextureLayer (no normals)
                let max_lod = layer_gd.get("max_snapshot_levels");
                let max_snapshot_levels = if max_lod.is_nil() {
                    0u32
                } else {
                    max_lod.to::<u32>()
                };
                if max_snapshot_levels > 0 {
                    let color_shader_var = layer_gd.get("snapshot_color_shader");
                    let color_shader = if color_shader_var.is_nil()
                        || color_shader_var.to::<String>().is_empty()
                    {
                        None
                    } else {
                        Some(color_shader_var.to::<String>())
                    };
                    let mut chain = CameraSnapshotTexture::new(
                        max_snapshot_levels,
                        resolution,
                        color_shader,
                        None,
                    );
                    chain.allocate_textures(&mut rd);
                    self.snapshot_textures.clear();
                    self.snapshot_normal_textures.clear();
                    for i in 0..chain.levels.len() {
                        let mut tex2d = Texture2Drd::new_gd();
                        tex2d.set_texture_rd_rid(chain.main_texture_rid(i));
                        self.snapshot_textures.push(Some(tex2d));
                    }
                    snapshot_chain = chain;
                    godot_print!(
                        "[Snapshot] Allocated {} LOD patches (resolution={})",
                        self.snapshot_textures.len(),
                        resolution
                    );
                }
            }
            if script_class == StringName::from("CesTerrainTextureLayer") {
                let resolution = layer_gd.get("resolution").to::<u32>();
                let compute_path = layer_gd.get("compute_shader_path").to::<String>();
                godot_print!(
                    "[CesCelestialRust] Terrain texture compute path: {}, resolution={}",
                    compute_path,
                    resolution
                );
                let mut gen = CubemapTextureGen::with_shader_path(&compute_path);
                gen.create_shared_cubemap(&mut rd, resolution);
                let mut tex = TextureCubemapRd::new_gd();
                tex.set_texture_rd_rid(gen.main_texture_rid());
                self.cubemap_texture = Some(tex);
                texture_gen = gen;
                godot_print!("[CesCelestialRust] Created CesTerrainTextureLayer cubemap");

                // Read terrain params from GDScript layer
                terrain_params = Some(read_terrain_params(&layer_gd));

                // Normal cubemap (conditional on generate_normal_map)
                let gen_normals = layer_gd.get("generate_normal_map").to::<bool>();
                if gen_normals {
                    let mut ngen = CubemapTextureGen::with_shader_path(TERRAIN_NORMAL_SHADER);
                    ngen.create_shared_cubemap(&mut rd, resolution);
                    let mut ntex = TextureCubemapRd::new_gd();
                    ntex.set_texture_rd_rid(ngen.main_texture_rid());
                    self.normal_cubemap_texture = Some(ntex);
                    normal_texture_gen = ngen;
                    godot_print!("[CesCelestialRust] Created normal cubemap");
                }
                // LOD chain for CesTerrainTextureLayer
                let max_lod = layer_gd.get("max_snapshot_levels");
                let max_snapshot_levels = if max_lod.is_nil() {
                    0u32
                } else {
                    max_lod.to::<u32>()
                };
                if max_snapshot_levels > 0 {
                    let color_shader_var = layer_gd.get("snapshot_color_shader");
                    let color_shader = if color_shader_var.is_nil()
                        || color_shader_var.to::<String>().is_empty()
                    {
                        None
                    } else {
                        Some(color_shader_var.to::<String>())
                    };
                    let normal_shader_var = layer_gd.get("snapshot_normal_shader");
                    let normal_shader = if normal_shader_var.is_nil()
                        || normal_shader_var.to::<String>().is_empty()
                    {
                        None
                    } else if layer_gd.get("generate_normal_map").to::<bool>() {
                        Some(normal_shader_var.to::<String>())
                    } else {
                        None
                    };
                    let mut chain = CameraSnapshotTexture::new(
                        max_snapshot_levels,
                        resolution,
                        color_shader,
                        normal_shader,
                    );
                    if let Some(ref tp) = terrain_params {
                        chain.set_extra_params(bytemuck::bytes_of(tp).to_vec());
                    }
                    chain.allocate_textures(&mut rd);
                    self.snapshot_textures.clear();
                    self.snapshot_normal_textures.clear();
                    for i in 0..chain.levels.len() {
                        let mut tex2d = Texture2Drd::new_gd();
                        tex2d.set_texture_rd_rid(chain.main_texture_rid(i));
                        self.snapshot_textures.push(Some(tex2d));
                        if chain.has_normal_textures() {
                            let mut ntex2d = Texture2Drd::new_gd();
                            ntex2d.set_texture_rd_rid(chain.normal_main_texture_rid(i));
                            self.snapshot_normal_textures.push(Some(ntex2d));
                        }
                    }
                    snapshot_chain = chain;
                    godot_print!(
                        "[Snapshot] Allocated {} LOD patches (resolution={})",
                        self.snapshot_textures.len(),
                        resolution
                    );
                }
            }
            if script_class == StringName::from("CesScatterMaskTextureLayer") {
                let noise_v = layer_gd.get("noise_strength");
                let color_v = layer_gd.get("target_color");
                let tol_v = layer_gd.get("color_tolerance");
                self.scatter_mask_noise_strength =
                    if noise_v.is_nil() { 1.0 } else { noise_v.to::<f32>() };
                self.scatter_mask_target_color = if color_v.is_nil() {
                    Color::from_rgba(0.2, 0.6, 0.2, 1.0)
                } else {
                    color_v.to::<Color>()
                };
                self.scatter_mask_color_tolerance =
                    if tol_v.is_nil() { 0.3 } else { tol_v.to::<f32>() };
                if self.blue_noise_bytes.is_empty() {
                    self.blue_noise_bytes = load_blue_noise_bytes();
                }
                if self.blue_noise_texture.is_none() {
                    self.blue_noise_texture =
                        Some(build_blue_noise_texture_from_bytes(&self.blue_noise_bytes));
                    godot_print!("[CesCelestialRust] Built blue-noise Texture3D for scatter mask");
                }
            }
        }

        let scatter_runtimes = self.build_scatter_runtimes();

        self.spawn_worker(
            rd,
            runtime_layers,
            texture_gen,
            normal_texture_gen,
            terrain_params,
            snapshot_chain,
            scatter_runtimes,
        );
        self.cubemap_resolution_active = self.find_cubemap_resolution();
        self.last_structure_id = layers_structure_id(&self.height_layers, &self.texture_layers, &self.scatter_layers);
        self.connect_layer_signals();
        self.build_scatter_mmi_children();
    }

    fn ready(&mut self) {
        self.add_subnodes();
        self.last_settings = self.current_settings();
        self.values_updated = true;
    }

    fn process(&mut self, _delta: f64) {
        let _process_timer = ScopedTimer::new("CesCelestialRust::process", self.show_process_timing);
        let should_run_preview = self.should_run_editor_preview();
        if !should_run_preview {
            if self.instance.is_valid() || self.work_tx.is_some() || self.worker_handle.is_some() {
                self.shutdown_for_reload_or_exit();
            }
            return;
        }

        if !self.instance.is_valid() || self.work_tx.is_none() || self.worker_handle.is_none() {
            self.restart_with_current_layers();
            return;
        }

        if self.is_shutting_down {
            return;
        }

        // Detect structural changes (layer added/removed/reordered/enabled toggle)
        let current_sid = layers_structure_id(&self.height_layers, &self.texture_layers, &self.scatter_layers);
        if current_sid != self.last_structure_id {
            self.last_structure_id = current_sid;
            self.disconnect_layer_signals();
            self.connect_layer_signals();
            self.layers_dirty = true;
            self.structural_dirty = true;
        }

        // Handle dirty flag immediately — the worker thread is the natural gate
        if self.layers_dirty {
            self.layers_dirty = false;
            // Property-only changes on the scatter mask layer (noise_strength,
            // target_color, color_tolerance) only need a uniform re-bind on the
            // existing planet material. Re-read them and re-apply to the live
            // material so slider drags are reflected immediately without a
            // full restart.
            self.refresh_scatter_mask_params();
            self.reapply_planet_shader_params();
            if self.structural_dirty {
                // Structural change: full restart (layer add/remove/enable, resolution, normal toggle)
                self.structural_dirty = false;
                let t0 = Instant::now();
                self.restart_with_current_layers();
                let ms = t0.elapsed().as_secs_f64() * 1000.0;
                godot_print!("AlgoTiming step Restart: {:.3} ms", ms);
            } else {
                // Property-only change (terrain params): send to worker
                // While the worker is busy, commands queue up; it drains and keeps only the last.
                self.structural_dirty = false;
                if let Some(tp) = self.find_terrain_params() {
                    if let Some(ref tx) = self.texture_cmd_tx {
                        let _ = tx.send(TextureCommand::ParamsChanged {
                            terrain_params: tp,
                            radius: self.radius,
                        });
                    }
                }
                self.values_updated = true;
            }
            return;
        }

        // Check if cubemap resolution changed — send lightweight regenerate command
        if self.check_cubemap_restart_needed() {
            let new_resolution = self.find_cubemap_resolution();
            if let Some(ref tx) = self.texture_cmd_tx {
                let _ = tx.send(TextureCommand::Regenerate {
                    size: new_resolution,
                    radius: self.radius,
                    terrain_params: self.find_terrain_params(),
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
                // Normal cubemap
                if result.normal_main_texture_rid.is_valid() {
                    let mut ntex = TextureCubemapRd::new_gd();
                    ntex.set_texture_rd_rid(result.normal_main_texture_rid);
                    self.normal_cubemap_texture = Some(ntex);
                }
                if result.normal_old_main_texture_rid.is_valid() {
                    let old_rid = result.normal_old_main_texture_rid;
                    crate::compute_utils::on_render_thread(move || {
                        let mut main_rd =
                            RenderingServer::singleton().get_rendering_device().unwrap();
                        main_rd.free_rid(old_rid);
                    });
                }
                self.values_updated = true;
            }
        }

        if self.show_process_timing {
            godot_print!("simulated_process_delay_ms = {}", self.simulated_process_delay_ms);
        }
        if self.simulated_process_delay_ms > 0 {
            thread::sleep(Duration::from_millis(
                self.simulated_process_delay_ms as u64,
            ));
        }
        // 1. Check for completed results (non-blocking)
        if let Some(ref result_rx) = self.result_rx {
            if let Ok(result) = result_rx.try_recv() {
                self.gen_mesh_running = false;
                self.apply_mesh_result(result);
            }
        }

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

        let cam_local = global_transform.affine_inverse() * cam_pos;

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

        let config = WorkerConfig {
            subdivisions: self.subdivisions,
            radius: self.radius,
            triangle_screen_size: self.triangle_screen_size,
            precise_normals: self.precise_normals,
            low_poly_look: self.low_poly_look,
            show_debug_messages: self.show_debug_messages,
            debug_snapshot_angle_offset: self.debug_snapshot_angle_offset,
            show_snapshot_borders: read_show_snapshot_borders(
                &self.texture_layers,
                self.show_snapshot_borders,
            ),
            scatter_params: read_enabled_scatter_layer_params(&self.scatter_layers),
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
        self.disconnect_layer_signals();
        self.destroy_scatter_mmi_children();

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
        self.snapshot_textures.clear();
        self.snapshot_normal_textures.clear();
        self.snapshot_info.clear();

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
                    for runtime in worker.scatter_runtimes.iter_mut() {
                        runtime.dispose_direct(&mut worker.rd);
                    }
                    // Flush any pending fire-and-forget render-thread callbacks
                    // before freeing the local RD they reference.
                    crate::compute_utils::on_render_thread_sync(|| {});
                    worker.texture_gen.dispose_local(&mut worker.rd);
                    worker.normal_texture_gen.dispose_local(&mut worker.rd);
                    worker.snapshot_chain.dispose_local(&mut worker.rd);
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
        let mut normal_texture_gen = CubemapTextureGen::new();
        let mut terrain_params: Option<TerrainParams> = None;
        if self.height_layers.is_empty() {
            runtime_layers.push(Box::new(CesSphereTerrain::new()));
        } else {
            for layer_gd in non_null_elements(&self.height_layers).into_iter() {
                if !layer_gd.bind().enabled {
                    continue;
                }
                let shader_path = layer_gd.bind().shader_path.to_string();
                if shader_path.is_empty() {
                    runtime_layers.push(Box::new(CesSphereTerrain::new()));
                } else {
                    runtime_layers.push(Box::new(CesHeightShaderTerrain::with_shader_path(
                        &shader_path,
                    )));
                }
            }
        }

        self.active_shader = None;
        let mut snapshot_chain = CameraSnapshotTexture::new(0, 512, None, None);
        for layer_gd in non_null_elements(&self.texture_layers).into_iter() {
            if !layer_gd.bind().enabled {
                continue;
            }
            let shader_var = layer_gd.get("shader");
            if let Ok(shader) = shader_var.try_to::<Gd<Shader>>() {
                self.active_shader = Some(shader);
            }
            let script_class = layer_script_class(&layer_gd);
            if self.show_debug_messages {
                eprintln!("Texture layer class: {}", script_class.to_string());
            }
            if script_class == StringName::from("CesTextureLayer") {
                let resolution = layer_gd.get("resolution").to::<u32>();
                let mut gen = CubemapTextureGen::new();
                gen.create_shared_cubemap(&mut rd, resolution);
                let mut tex = TextureCubemapRd::new_gd();
                tex.set_texture_rd_rid(gen.main_texture_rid());
                self.cubemap_texture = Some(tex);
                texture_gen = gen;
                // LOD chain for CesTextureLayer (no normals)
                let max_lod = layer_gd.get("max_snapshot_levels");
                let max_snapshot_levels = if max_lod.is_nil() {
                    0u32
                } else {
                    max_lod.to::<u32>()
                };
                if max_snapshot_levels > 0 {
                    let color_shader_var = layer_gd.get("snapshot_color_shader");
                    let color_shader = if color_shader_var.is_nil()
                        || color_shader_var.to::<String>().is_empty()
                    {
                        None
                    } else {
                        Some(color_shader_var.to::<String>())
                    };
                    let mut chain = CameraSnapshotTexture::new(
                        max_snapshot_levels,
                        resolution,
                        color_shader,
                        None,
                    );
                    chain.allocate_textures(&mut rd);
                    self.snapshot_textures.clear();
                    self.snapshot_normal_textures.clear();
                    for i in 0..chain.levels.len() {
                        let mut tex2d = Texture2Drd::new_gd();
                        tex2d.set_texture_rd_rid(chain.main_texture_rid(i));
                        self.snapshot_textures.push(Some(tex2d));
                    }
                    snapshot_chain = chain;
                    godot_print!(
                        "[Snapshot] Allocated {} LOD patches (resolution={})",
                        self.snapshot_textures.len(),
                        resolution
                    );
                }
            }
            if script_class == StringName::from("CesTerrainTextureLayer") {
                let resolution = layer_gd.get("resolution").to::<u32>();
                let compute_path = layer_gd.get("compute_shader_path").to::<String>();
                if self.show_debug_messages {
                    eprintln!("Terrain texture compute path: {}", compute_path);
                }
                let mut gen = CubemapTextureGen::with_shader_path(&compute_path);
                gen.create_shared_cubemap(&mut rd, resolution);
                let mut tex = TextureCubemapRd::new_gd();
                tex.set_texture_rd_rid(gen.main_texture_rid());
                self.cubemap_texture = Some(tex);
                texture_gen = gen;

                // Read terrain params from GDScript layer
                terrain_params = Some(read_terrain_params(&layer_gd));

                // Normal cubemap (conditional on generate_normal_map)
                let gen_normals = layer_gd.get("generate_normal_map").to::<bool>();
                if gen_normals {
                    let mut ngen = CubemapTextureGen::with_shader_path(TERRAIN_NORMAL_SHADER);
                    ngen.create_shared_cubemap(&mut rd, resolution);
                    let mut ntex = TextureCubemapRd::new_gd();
                    ntex.set_texture_rd_rid(ngen.main_texture_rid());
                    self.normal_cubemap_texture = Some(ntex);
                    normal_texture_gen = ngen;
                } else {
                    self.normal_cubemap_texture = None;
                }
                // LOD chain for CesTerrainTextureLayer
                let max_lod = layer_gd.get("max_snapshot_levels");
                let max_snapshot_levels = if max_lod.is_nil() {
                    0u32
                } else {
                    max_lod.to::<u32>()
                };
                if max_snapshot_levels > 0 {
                    let color_shader_var = layer_gd.get("snapshot_color_shader");
                    let color_shader = if color_shader_var.is_nil()
                        || color_shader_var.to::<String>().is_empty()
                    {
                        None
                    } else {
                        Some(color_shader_var.to::<String>())
                    };
                    let normal_shader_var = layer_gd.get("snapshot_normal_shader");
                    let normal_shader = if normal_shader_var.is_nil()
                        || normal_shader_var.to::<String>().is_empty()
                    {
                        None
                    } else if layer_gd.get("generate_normal_map").to::<bool>() {
                        Some(normal_shader_var.to::<String>())
                    } else {
                        None
                    };
                    let mut chain = CameraSnapshotTexture::new(
                        max_snapshot_levels,
                        resolution,
                        color_shader,
                        normal_shader,
                    );
                    if let Some(ref tp) = terrain_params {
                        chain.set_extra_params(bytemuck::bytes_of(tp).to_vec());
                    }
                    chain.allocate_textures(&mut rd);
                    self.snapshot_textures.clear();
                    self.snapshot_normal_textures.clear();
                    for i in 0..chain.levels.len() {
                        let mut tex2d = Texture2Drd::new_gd();
                        tex2d.set_texture_rd_rid(chain.main_texture_rid(i));
                        self.snapshot_textures.push(Some(tex2d));
                        if chain.has_normal_textures() {
                            let mut ntex2d = Texture2Drd::new_gd();
                            ntex2d.set_texture_rd_rid(chain.normal_main_texture_rid(i));
                            self.snapshot_normal_textures.push(Some(ntex2d));
                        }
                    }
                    snapshot_chain = chain;
                    godot_print!(
                        "[Snapshot] Allocated {} LOD patches (resolution={})",
                        self.snapshot_textures.len(),
                        resolution
                    );
                }
            }
            if script_class == StringName::from("CesScatterMaskTextureLayer") {
                let noise_v = layer_gd.get("noise_strength");
                let color_v = layer_gd.get("target_color");
                let tol_v = layer_gd.get("color_tolerance");
                self.scatter_mask_noise_strength =
                    if noise_v.is_nil() { 1.0 } else { noise_v.to::<f32>() };
                self.scatter_mask_target_color = if color_v.is_nil() {
                    Color::from_rgba(0.2, 0.6, 0.2, 1.0)
                } else {
                    color_v.to::<Color>()
                };
                self.scatter_mask_color_tolerance =
                    if tol_v.is_nil() { 0.3 } else { tol_v.to::<f32>() };
                if self.blue_noise_bytes.is_empty() {
                    self.blue_noise_bytes = load_blue_noise_bytes();
                }
                if self.blue_noise_texture.is_none() {
                    self.blue_noise_texture =
                        Some(build_blue_noise_texture_from_bytes(&self.blue_noise_bytes));
                    godot_print!("[CesCelestialRust] Built blue-noise Texture3D for scatter mask");
                }
            }
        }

        let scatter_runtimes = self.build_scatter_runtimes();

        self.spawn_worker(
            rd,
            runtime_layers,
            texture_gen,
            normal_texture_gen,
            terrain_params,
            snapshot_chain,
            scatter_runtimes,
        );
        self.cubemap_resolution_active = self.find_cubemap_resolution();
        self.last_structure_id = layers_structure_id(&self.height_layers, &self.texture_layers, &self.scatter_layers);
        self.connect_layer_signals();
        self.build_scatter_mmi_children();
    }

    fn check_cubemap_restart_needed(&self) -> bool {
        for layer_gd in non_null_elements(&self.texture_layers).into_iter() {
            if !layer_gd.bind().enabled {
                continue;
            }
            if layer_script_class(&layer_gd) == StringName::from("CesTextureLayer")
                || layer_script_class(&layer_gd) == StringName::from("CesTerrainTextureLayer")
            {
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
            debug_snapshot_angle_offset: self.debug_snapshot_angle_offset,
            show_snapshot_borders: read_show_snapshot_borders(
                &self.texture_layers,
                self.show_snapshot_borders,
            ),
        }
    }

    fn should_run_editor_preview(&self) -> bool {
        if !Engine::singleton().is_editor_hint() {
            return true;
        }

        let ei = godot::classes::EditorInterface::singleton();
        let Some(edited_root) = ei.get_edited_scene_root() else {
            return false;
        };

        let edited_root_id = edited_root.instance_id();
        let mut current: Option<Gd<Node>> = Some(self.to_gd().upcast::<Node>());
        while let Some(node) = current {
            if node.instance_id() == edited_root_id {
                return true;
            }
            current = node.get_parent();
        }

        false
    }

    fn get_camera(&self) -> Option<Gd<Camera3D>> {
        // In editor mode, prefer the editor viewport camera so the mesh
        // subdivides around the editor camera, not the scene's Camera3D node.
        if self.use_editor_camera && Engine::singleton().is_editor_hint() {
            let ei = godot::classes::EditorInterface::singleton();
            if let Some(vp) = ei.get_editor_viewport_3d() {
                if let Some(camera) = vp.get_camera_3d() {
                    return Some(camera);
                }
            }
        }

        if let Some(ref cam) = self.gameplay_camera {
            return Some(cam.clone());
        }

        if let Some(parent) = self.base().get_parent() {
            if let Some(camera) = parent.try_get_node_as::<Camera3D>("Camera3D") {
                return Some(camera);
            }
        }

        self.base().get_viewport().and_then(|vp| vp.get_camera_3d())
    }

    fn apply_mesh_result(&mut self, result: MeshResult) {
        let has_mesh = !result.pos.is_empty() && !result.triangles.is_empty();

        // Apply snapshot data if present
        if result.snapshot_updated {
            for (i, rid) in result.snapshot_main_rids.iter().enumerate() {
                if i < self.snapshot_textures.len() {
                    if let Some(ref mut tex) = self.snapshot_textures[i] {
                        tex.set_texture_rd_rid(*rid);
                    }
                }
            }
            for (i, rid) in result.snapshot_normal_main_rids.iter().enumerate() {
                if i < self.snapshot_normal_textures.len() {
                    if let Some(ref mut tex) = self.snapshot_normal_textures[i] {
                        tex.set_texture_rd_rid(*rid);
                    }
                }
            }
            self.snapshot_info = result.snapshot_level_info;
        }

        // Apply scatter transforms to per-layer MultiMeshInstance3D children.
        // Missing result entries are treated as zero-instance slots so stale
        // MultiMesh counts are cleared instead of lingering on screen.
        for (i, mmi_opt) in self.scatter_mmi_nodes.iter().enumerate() {
            let Some(ref mmi) = mmi_opt else {
                continue;
            };
            let Some(mut mm) = mmi.get_multimesh() else {
                continue;
            };
            let instance_count = result.scatter_instance_counts.get(i).copied().unwrap_or(0) as i32;
            if mm.get_instance_count() != instance_count {
                mm.set_instance_count(instance_count);
            }
            if instance_count == 0 {
                continue;
            }
            let Some(buf) = result.scatter_buffers.get(i) else {
                continue;
            };
            if buf.is_empty() {
                continue;
            }
            let packed = PackedFloat32Array::from(buf.as_slice());
            mm.set_buffer(&packed);
        }

        if !has_mesh {
            return;
        }

        let MeshResult {
            pos, triangles, uv, ..
        } = result;
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
            if let Some(ref normal_cubemap) = self.normal_cubemap_texture {
                material.set_shader_parameter("normal_cubemap", &normal_cubemap.to_variant());
                material.set_shader_parameter("use_normal_cubemap", &true.to_variant());
            } else {
                material.set_shader_parameter("use_normal_cubemap", &false.to_variant());
            }

            // Apply LOD textures and metadata to the new material
            self.apply_lod_shader_params(&mut material);

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

    /// Re-read `CesScatterMaskTextureLayer` params from the live GD resources
    /// without doing any GPU work. Called from the property-only path so a
    /// slider drag updates `self.scatter_mask_*` immediately.
    fn refresh_scatter_mask_params(&mut self) {
        for layer_gd in non_null_elements(&self.texture_layers) {
            if !layer_gd.bind().enabled {
                continue;
            }
            let script_class = layer_script_class(&layer_gd);
            if script_class != StringName::from("CesScatterMaskTextureLayer") {
                continue;
            }
            let noise_v = layer_gd.get("noise_strength");
            let color_v = layer_gd.get("target_color");
            let tol_v = layer_gd.get("color_tolerance");
            self.scatter_mask_noise_strength =
                if noise_v.is_nil() { 1.0 } else { noise_v.to::<f32>() };
            self.scatter_mask_target_color = if color_v.is_nil() {
                Color::from_rgba(0.2, 0.6, 0.2, 1.0)
            } else {
                color_v.to::<Color>()
            };
            self.scatter_mask_color_tolerance =
                if tol_v.is_nil() { 0.3 } else { tol_v.to::<f32>() };
        }
    }

    /// Re-bind the planet's shader uniforms from the current `self.scatter_mask_*`
    /// fields onto the LIVE material attached to `self.mesh`. Lets a property
    /// change reach the GPU without triggering a full mesh rebuild.
    fn reapply_planet_shader_params(&self) {
        let Some(ref mesh) = self.mesh else {
            return;
        };
        let Some(material_resource) = mesh.surface_get_material(0) else {
            return;
        };
        let Ok(mut material) = material_resource.try_cast::<ShaderMaterial>() else {
            return;
        };
        self.apply_lod_shader_params(&mut material);
    }

    fn apply_lod_shader_params(&self, material: &mut Gd<ShaderMaterial>) {
        let active_snapshot_count = self.snapshot_textures.len() as i32;
        material.set_shader_parameter("snapshot_count", &active_snapshot_count.to_variant());
        for (i, tex_opt) in self.snapshot_textures.iter().enumerate() {
            if let Some(ref tex) = tex_opt {
                let name = format!("snapshot{}_texture", i + 1);
                material.set_shader_parameter(&name, &tex.to_variant());
            }
        }
        for (i, tex_opt) in self.snapshot_normal_textures.iter().enumerate() {
            if let Some(ref tex) = tex_opt {
                let name = format!("snapshot{}_normal_texture", i + 1);
                material.set_shader_parameter(&name, &tex.to_variant());
            }
        }
        for (i, info) in self.snapshot_info.iter().enumerate() {
            let (center_dir, tangent_u, tangent_v, angular_extent) = info;
            let idx = i + 1;
            material.set_shader_parameter(
                &format!("snapshot{}_center_dir", idx),
                &(*center_dir).to_variant(),
            );
            material.set_shader_parameter(
                &format!("snapshot{}_tangent_u", idx),
                &(*tangent_u).to_variant(),
            );
            material.set_shader_parameter(
                &format!("snapshot{}_tangent_v", idx),
                &(*tangent_v).to_variant(),
            );
            material.set_shader_parameter(
                &format!("snapshot{}_angular_extent", idx),
                &angular_extent.to_variant(),
            );
        }
        // Scatter mask preview uniforms — silently ignored by other shaders.
        if let Some(ref tex3d) = self.blue_noise_texture {
            material.set_shader_parameter("blue_noise_texture", &tex3d.to_variant());
        }
        material.set_shader_parameter(
            "noise_strength",
            &self.scatter_mask_noise_strength.to_variant(),
        );
        material.set_shader_parameter(
            "target_color",
            &self.scatter_mask_target_color.to_variant(),
        );
        material.set_shader_parameter(
            "color_tolerance",
            &self.scatter_mask_color_tolerance.to_variant(),
        );
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
    #[allow(clippy::too_many_arguments)]
    fn spawn_worker(
        &mut self,
        rd: Gd<RenderingDevice>,
        layers: Vec<Box<dyn CesLayer>>,
        texture_gen: CubemapTextureGen,
        normal_texture_gen: CubemapTextureGen,
        terrain_params: Option<TerrainParams>,
        snapshot_chain: CameraSnapshotTexture,
        scatter_runtimes: Vec<CesScatterRuntime>,
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
            normal_texture_gen,
            terrain_params,
            snapshot_chain,
            scatter_runtimes,
        };

        let handle = std::thread::spawn(move || {
            let mut last_snapshot_show_borders = worker.snapshot_chain.show_borders;

            // Apply terrain params to height layers
            if let Some(ref tp) = worker.terrain_params {
                for layer in worker.layers.iter_mut() {
                    layer.set_terrain_params(tp);
                }
            }

            // Generate cubemap texture once at startup (only if a cubemap node was found)
            if worker.texture_gen.has_texture() {
                eprintln!("[CesCelestialRust] texture_gen: has_texture=true, shader={}, size={}, radius={}", worker.texture_gen.shader_path, cubemap_size, radius);
                let t0 = Instant::now();
                worker.texture_gen.init_pipeline(&mut worker.rd);
                eprintln!("[CesCelestialRust] texture_gen: pipeline init OK");
                worker.texture_gen.generate(
                    &mut worker.rd,
                    cubemap_size,
                    radius,
                    worker.terrain_params.as_ref(),
                );
                let tex_ms = t0.elapsed().as_secs_f64() * 1000.0;
                godot_print!("AlgoTiming step TextureColor: {:.3} ms", tex_ms);
            } else {
                eprintln!("[CesCelestialRust] texture_gen: has_texture=false, skipping");
            }

            // Generate normal cubemap
            if worker.normal_texture_gen.has_texture() {
                eprintln!(
                    "[CesCelestialRust] normal_texture_gen: init+generate, size={}, radius={}",
                    cubemap_size, radius
                );
                let t0 = Instant::now();
                worker.normal_texture_gen.init_pipeline(&mut worker.rd);
                worker.normal_texture_gen.generate(
                    &mut worker.rd,
                    cubemap_size,
                    radius,
                    worker.terrain_params.as_ref(),
                );
                let norm_ms = t0.elapsed().as_secs_f64() * 1000.0;
                godot_print!("AlgoTiming step TextureNormal: {:.3} ms", norm_ms);
            }

            // Initialize LOD chain pipeline if textures are allocated
            if worker.snapshot_chain.has_textures() {
                worker.snapshot_chain.init_pipeline(&mut worker.rd);
            }

            while let Ok((cam_local, config)) = work_rx.recv() {
                // Drain all pending commands — coalesce multiple ParamsChanged into one
                let mut last_params: Option<(TerrainParams, f32)> = None;
                while let Ok(cmd) = texture_cmd_rx.try_recv() {
                    match cmd {
                        TextureCommand::Regenerate {
                            size,
                            radius,
                            terrain_params,
                        } => {
                            let t0 = Instant::now();
                            let (new_rid, old_rid) = worker.texture_gen.resize(
                                &mut worker.rd,
                                size,
                                radius,
                                terrain_params.as_ref(),
                            );
                            let tex_ms = t0.elapsed().as_secs_f64() * 1000.0;
                            let t1 = Instant::now();
                            let (normal_new_rid, normal_old_rid) =
                                if worker.normal_texture_gen.has_texture() {
                                    worker.normal_texture_gen.resize(
                                        &mut worker.rd,
                                        size,
                                        radius,
                                        terrain_params.as_ref(),
                                    )
                                } else {
                                    (Rid::Invalid, Rid::Invalid)
                                };
                            let norm_ms = t1.elapsed().as_secs_f64() * 1000.0;
                            godot_print!("AlgoTiming step TextureColor: {:.3} ms", tex_ms);
                            godot_print!("AlgoTiming step TextureNormal: {:.3} ms", norm_ms);
                            let _ = texture_result_tx.send(TextureResult {
                                main_texture_rid: new_rid,
                                old_main_texture_rid: old_rid,
                                normal_main_texture_rid: normal_new_rid,
                                normal_old_main_texture_rid: normal_old_rid,
                            });
                        }
                        TextureCommand::ParamsChanged {
                            terrain_params,
                            radius,
                        } => {
                            // Keep only the latest ParamsChanged
                            last_params = Some((terrain_params, radius));
                        }
                    }
                }

                // Process the latest ParamsChanged (if any)
                let mut snapshot_force_regenerated = false;
                if let Some((terrain_params, radius)) = last_params {
                    for layer in worker.layers.iter_mut() {
                        layer.set_terrain_params(&terrain_params);
                    }
                    worker.terrain_params = Some(terrain_params);

                    let size = worker.texture_gen.current_size();
                    let t0 = Instant::now();
                    worker.texture_gen.generate(
                        &mut worker.rd,
                        size,
                        radius,
                        worker.terrain_params.as_ref(),
                    );
                    let tex_ms = t0.elapsed().as_secs_f64() * 1000.0;
                    let t1 = Instant::now();
                    if worker.normal_texture_gen.has_texture() {
                        worker.normal_texture_gen.generate(
                            &mut worker.rd,
                            size,
                            radius,
                            worker.terrain_params.as_ref(),
                        );
                    }
                    let norm_ms = t1.elapsed().as_secs_f64() * 1000.0;
                    godot_print!("AlgoTiming step TextureColor: {:.3} ms", tex_ms);
                    godot_print!("AlgoTiming step TextureNormal: {:.3} ms", norm_ms);

                    // Update snapshot chain params and force-regenerate
                    if worker.snapshot_chain.has_textures() {
                        worker
                            .snapshot_chain
                            .set_extra_params(bytemuck::bytes_of(&terrain_params).to_vec());
                        worker
                            .snapshot_chain
                            .force_regenerate(&mut worker.rd, config.radius);
                        snapshot_force_regenerated = true;
                    }

                    // Reset subdivision so heights are recomputed
                    worker.graph_generator = None;
                }

                if worker.snapshot_chain.has_textures()
                    && config.show_snapshot_borders != last_snapshot_show_borders
                {
                    worker.snapshot_chain.show_borders = config.show_snapshot_borders;
                    worker
                        .snapshot_chain
                        .force_regenerate(&mut worker.rd, config.radius);
                    snapshot_force_regenerated = true;
                    last_snapshot_show_borders = config.show_snapshot_borders;
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
                        snapshot_main_rids: vec![],
                        snapshot_normal_main_rids: vec![],
                        snapshot_level_info: vec![],
                        snapshot_updated: false,
                        scatter_buffers: vec![],
                        scatter_instance_counts: vec![],
                    });
                    continue;
                }

                // Generate camera snapshots after mesh algo (coupled to mesh update)
                let snapshot_updated = if worker.snapshot_chain.has_textures() {
                    // Apply show_borders setting
                    worker.snapshot_chain.show_borders = config.show_snapshot_borders;
                    // Apply debug offset: rotate the snapshot center direction
                    let snapshot_cam = if config.debug_snapshot_angle_offset != 0.0 {
                        let dir = cam_local.normalized();
                        let up = if dir.y.abs() > 0.99 {
                            Vector3::new(0.0, 0.0, 1.0)
                        } else {
                            Vector3::new(0.0, 1.0, 0.0)
                        };
                        let right = up.cross(dir).normalized();
                        let angle = config.debug_snapshot_angle_offset;
                        (dir * angle.cos() + right * angle.sin()).normalized() * cam_local.length()
                    } else {
                        cam_local
                    };
                    let updated =
                        worker
                            .snapshot_chain
                            .update(&mut worker.rd, snapshot_cam, config.radius);
                    updated || snapshot_force_regenerated
                } else {
                    false
                };
                let (snapshot_main_rids, snapshot_normal_main_rids, snapshot_level_info) =
                    if snapshot_updated {
                        if config.show_debug_messages {
                            godot_print!(
                                "[Snapshot] Updated {} patches",
                                worker.snapshot_chain.levels.len()
                            );
                        }
                        let rids = (
                            worker
                                .snapshot_chain
                                .levels
                                .iter()
                                .map(|l| l.back_main_texture_rid)
                                .collect(),
                            worker
                                .snapshot_chain
                                .levels
                                .iter()
                                .map(|l| l.back_normal_main_texture_rid)
                                .collect(),
                            worker
                                .snapshot_chain
                                .levels
                                .iter()
                                .map(|l| (l.center_dir, l.tangent_u, l.tangent_v, l.angular_extent))
                                .collect(),
                        );
                        // Swap front↔back so next generation writes to the old front
                        worker.snapshot_chain.swap_buffers();
                        rids
                    } else {
                        (vec![], vec![], vec![])
                    };

                // Dispatch scatter runtimes. Triangle-indexed: one thread per
                // triangle of the LOD mesh, filtered to `subdivision_level`.
                let scatter_slot_count = worker.scatter_runtimes.len();
                let (mut scatter_buffers, mut scatter_instance_counts) =
                    zeroed_scatter_results(scatter_slot_count);
                if scatter_slot_count > 0 {
                    let WorkerState {
                        ref mut rd,
                        ref graph_generator,
                        ref terrain_params,
                        ref mut scatter_runtimes,
                        ..
                    } = worker;

                    let tp_for_scatter = terrain_params.unwrap_or_default();
                    let state_opt = graph_generator.as_ref().and_then(|g| g.state.as_ref());

                    if let Some(state) = state_opt {
                        if state.n_tris > 0 {
                            // Debug: print distribution of LOD levels in current state.
                            if config.show_debug_messages {
                                let levels = state.get_level(rd);
                                let deactivated = state.get_t_deactivated_mask(rd);
                                let mut hist = std::collections::BTreeMap::<i32, u32>::new();
                                for (i, &lv) in levels.iter().enumerate() {
                                    if deactivated.get(i).copied().unwrap_or(0) == 0 {
                                        *hist.entry(lv).or_insert(0) += 1;
                                    }
                                }
                                godot_print!(
                                    "[CesCelestialRust] LOD level histogram (active tris): {:?}",
                                    hist
                                );
                            }
                            for (slot, runtime) in scatter_runtimes.iter_mut().enumerate() {
                                let layer_params = config
                                    .scatter_params
                                    .get(slot)
                                    .copied()
                                    .unwrap_or_else(|| *runtime.params());
                                let params = scatter_params_with_runtime_context(
                                    layer_params,
                                    config.radius,
                                );
                                runtime.set_params(params);
                                runtime.init_pipeline(rd);
                                runtime.dispatch(rd, &tp_for_scatter, state);
                                let count = runtime.readback_count(rd);
                                scatter_buffers[slot] = runtime.readback_compact(rd, count);
                                scatter_instance_counts[slot] = count;
                                if config.show_debug_messages {
                                    godot_print!(
                                        "[CesCelestialRust] Scatter layer {} (level {}): {} instances spawned (n_tris={})",
                                        slot,
                                        params.subdivision_level,
                                        count,
                                        state.n_tris
                                    );
                                }
                            }
                        }
                    }
                }

                let result = MeshResult {
                    pos: output.pos,
                    triangles: output.tris,
                    uv: output.uv,
                    snapshot_main_rids,
                    snapshot_normal_main_rids,
                    snapshot_level_info,
                    snapshot_updated,
                    scatter_buffers,
                    scatter_instance_counts,
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
            if layer_script_class(&layer_gd) == StringName::from("CesTextureLayer")
                || layer_script_class(&layer_gd) == StringName::from("CesTerrainTextureLayer")
            {
                return layer_gd.get("resolution").to::<u32>();
            }
        }
        512
    }

    fn find_terrain_params(&self) -> Option<TerrainParams> {
        for layer_gd in non_null_elements(&self.texture_layers).into_iter() {
            if !layer_gd.bind().enabled {
                continue;
            }
            if layer_script_class(&layer_gd) == StringName::from("CesTerrainTextureLayer") {
                return Some(read_terrain_params(&layer_gd));
            }
        }
        None
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

    /// Build a `Vec<CesScatterRuntime>` sized and ordered to match the
    /// currently-enabled scatter layers. Runtimes are created without any GPU
    /// pipeline/buffer work yet — that happens on the worker thread.
    fn build_scatter_runtimes(&mut self) -> Vec<CesScatterRuntime> {
        let mut runtimes: Vec<CesScatterRuntime> = Vec::new();
        // Has any enabled scatter layer? If so, ensure the noise bytes are loaded
        // so we can hand them to every runtime (the runtime uploads them to a
        // GPU storage buffer on first dispatch).
        let has_scatter = non_null_elements(&self.scatter_layers)
            .iter()
            .any(|l| l.bind().enabled);
        if has_scatter && self.blue_noise_bytes.is_empty() {
            self.blue_noise_bytes = load_blue_noise_bytes();
        }
        let noise_dim = BLUE_NOISE_SIZE[0];
        for layer_gd in non_null_elements(&self.scatter_layers).into_iter() {
            if !layer_gd.bind().enabled {
                continue;
            }
            let raw_path = layer_gd.get("placement_shader_path");
            let shader_path = if raw_path.is_nil() {
                DEFAULT_SCATTER_SHADER_PATH.to_string()
            } else {
                let s = raw_path.to::<GString>().to_string();
                if s.is_empty() {
                    DEFAULT_SCATTER_SHADER_PATH.to_string()
                } else {
                    s
                }
            };
            let mut runtime = CesScatterRuntime::new(&shader_path);
            runtime.set_params(scatter_params_with_runtime_context(
                read_scatter_layer_params(&layer_gd),
                self.radius,
            ));
            if !self.blue_noise_bytes.is_empty() {
                runtime.set_blue_noise(self.blue_noise_bytes.clone(), noise_dim);
            }
            runtimes.push(runtime);
        }
        runtimes
    }

    /// Create a `MultiMeshInstance3D` child per enabled scatter layer. Layers
    /// with no mesh assigned still get a slot (as `None`) so indexes align
    /// with `WorkerState::scatter_runtimes` and `MeshResult::scatter_buffers`.
    fn build_scatter_mmi_children(&mut self) {
        self.destroy_scatter_mmi_children();
        let enabled_layer_count = count_enabled_scatter_layers(&self.scatter_layers);
        let enabled_layers: Vec<Gd<CesScatterLayerResource>> =
            non_null_elements(&self.scatter_layers)
                .into_iter()
                .filter(|l| l.bind().enabled)
                .collect();

        let mut new_nodes: Vec<Option<Gd<MultiMeshInstance3D>>> =
            Vec::with_capacity(enabled_layer_count);
        for (i, layer_gd) in enabled_layers.iter().enumerate() {
            let mesh_var = layer_gd.get("mesh");
            let mesh_opt: Option<Gd<Mesh>> = if mesh_var.is_nil() {
                None
            } else {
                mesh_var.try_to::<Option<Gd<Mesh>>>().ok().flatten()
            };
            let Some(mesh) = mesh_opt else {
                new_nodes.push(None);
                continue;
            };

            let mut mmi = MultiMeshInstance3D::new_alloc();
            mmi.set_name(&format!("ScatterLayer{}", i));

            let mut mm = MultiMesh::new_gd();
            mm.set_transform_format(TransformFormat::TRANSFORM_3D);
            mm.set_mesh(&mesh);
            mm.set_instance_count(0);
            mmi.set_multimesh(&mm);

            let material_var = layer_gd.get("material");
            if !material_var.is_nil() {
                if let Ok(Some(material)) = material_var.try_to::<Option<Gd<Material>>>() {
                    mmi.set_material_override(&material);
                }
            }

            self.base_mut().add_child(&mmi);
            new_nodes.push(Some(mmi));
        }
        self.scatter_mmi_nodes = new_nodes;
    }

    /// Free all existing scatter MMI3D children. Safe to call multiple times.
    fn destroy_scatter_mmi_children(&mut self) {
        let nodes = std::mem::take(&mut self.scatter_mmi_nodes);
        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            for mut mmi in nodes.into_iter().flatten() {
                mmi.queue_free();
            }
        }));
    }

    /// Connect the `Resource.changed` signal on all active layers so that
    /// property edits (e.g. slider drags) set the dirty flag without polling.
    fn connect_layer_signals(&mut self) {
        let callable = Callable::from_object_method(&self.to_gd(), "on_layer_changed");
        for l in non_null_elements(&self.height_layers) {
            let id = l.instance_id().to_i64();
            if self.connected_layer_ids.contains(&id) {
                continue;
            }
            l.clone().upcast::<Object>().connect("changed", &callable);
            self.connected_layer_ids.push(id);
            godot_print!(
                "[CesCelestialRust] Connected changed signal on height layer id={}",
                id
            );
        }
        for l in non_null_elements(&self.texture_layers) {
            let id = l.instance_id().to_i64();
            if self.connected_layer_ids.contains(&id) {
                continue;
            }
            l.clone().upcast::<Object>().connect("changed", &callable);
            self.connected_layer_ids.push(id);
            godot_print!(
                "[CesCelestialRust] Connected changed signal on texture layer id={}",
                id
            );
        }
        for l in non_null_elements(&self.scatter_layers) {
            let id = l.instance_id().to_i64();
            if self.connected_layer_ids.contains(&id) {
                continue;
            }
            l.clone().upcast::<Object>().connect("changed", &callable);
            self.connected_layer_ids.push(id);
            godot_print!(
                "[CesCelestialRust] Connected changed signal on scatter layer id={}",
                id
            );
        }
    }

    /// Disconnect all previously connected `Resource.changed` signals.
    fn disconnect_layer_signals(&mut self) {
        let callable = Callable::from_object_method(&self.to_gd(), "on_layer_changed");
        for id in self.connected_layer_ids.drain(..) {
            let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                if let Ok(obj) = Gd::<Object>::try_from_instance_id(InstanceId::from_i64(id)) {
                    obj.clone().disconnect("changed", &callable);
                }
            }));
        }
    }
}

#[godot_api]
impl CesCelestialRust {
    #[func]
    fn on_layer_changed(&mut self) {
        godot_print!("[CesCelestialRust] on_layer_changed signal received");
        self.layers_dirty = true;
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

    #[test]
    fn test_snapshot_chain_result_is_send() {
        fn assert_send<T: Send>() {}
        assert_send::<super::MeshResult>();
    }

    #[test]
    fn test_scatter_structure_part_format() {
        // Changing instance_id, enabled, subdivision_level, or noise_strength
        // must produce different strings (so the structure id triggers rebuild).
        assert_ne!(
            super::scatter_structure_part(1, true, 5, 1.0),
            super::scatter_structure_part(1, false, 5, 1.0)
        );
        assert_ne!(
            super::scatter_structure_part(1, true, 5, 1.0),
            super::scatter_structure_part(2, true, 5, 1.0)
        );
        assert_ne!(
            super::scatter_structure_part(1, true, 5, 1.0),
            super::scatter_structure_part(1, true, 6, 1.0)
        );
        assert_ne!(
            super::scatter_structure_part(1, true, 5, 1.0),
            super::scatter_structure_part(1, true, 5, 0.5)
        );
    }

    #[test]
    fn test_mesh_result_scatter_fields_send() {
        fn assert_send<T: Send>() {}
        assert_send::<Vec<Vec<f32>>>();
        assert_send::<Vec<u32>>();
        assert_send::<super::MeshResult>();
    }

    #[test]
    fn test_count_enabled_flags_empty() {
        assert_eq!(super::count_enabled_flags(&[]), 0);
    }

    #[test]
    fn test_count_enabled_flags_mixed() {
        assert_eq!(
            super::count_enabled_flags(&[true, false, true, true, false]),
            3
        );
        assert_eq!(super::count_enabled_flags(&[false, false, false]), 0);
        assert_eq!(super::count_enabled_flags(&[true, true, true]), 3);
    }

    #[test]
    fn scatter_child_count_matches_enabled_layers() {
        let enabled_flags = [true, false, true, true, false];
        let enabled_count = super::count_enabled_flags(&enabled_flags);
        let (scatter_buffers, scatter_instance_counts) =
            super::zeroed_scatter_results(enabled_count);

        assert_eq!(enabled_count, 3);
        assert_eq!(scatter_buffers.len(), enabled_count);
        assert_eq!(scatter_instance_counts, vec![0, 0, 0]);
        assert!(scatter_buffers.iter().all(|buf| buf.is_empty()));
    }

    #[test]
    fn test_scatter_params_with_runtime_context_overrides_radius() {
        let params = super::ScatterParams {
            seed: 7,
            subdivision_level: 4,
            planet_radius: 1.0,
            ..super::ScatterParams::default()
        };
        let updated = super::scatter_params_with_runtime_context(params, 42.5);
        assert_eq!(updated.seed, 7);
        assert_eq!(updated.subdivision_level, 4);
        assert_eq!(updated.planet_radius, 42.5);
    }

    #[test]
    fn scatter_params_matches_layer_fields() {
        let params = super::scatter_params_from_layer_fields(super::ScatterLayerFieldValues {
            seed: 23,
            subdivision_level: 4,
            noise_strength: 0.7,
            height_min: 0.18,
            height_max: 0.74,
            albedo_target: [0.11, 0.29, 0.47],
            albedo_tolerance: 0.09,
        });

        assert_eq!(params.seed, 23);
        assert_eq!(params.subdivision_level, 4);
        assert!((params.noise_strength - 0.7).abs() < 1e-5);
        assert_eq!(params.height_min, 0.18);
        assert_eq!(params.height_max, 0.74);
        assert_eq!(params.albedo_target, [0.11, 0.29, 0.47]);
        assert_eq!(params.albedo_tolerance, 0.09);
        assert_eq!(
            params.planet_radius,
            super::ScatterParams::default().planet_radius
        );
    }
}
