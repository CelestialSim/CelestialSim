use crate::algo::run_algo::{CesRunAlgo, RunAlgoConfig};
use crate::camera_snapshot_texture::CameraSnapshotTexture;
use crate::layer_resources::{
    CesHeightLayerResource, CesScatterLayerResource, CesTextureLayerResource,
};
use crate::layers::height_shader_terrain::CesHeightShaderTerrain;
use crate::layers::scatter::{CesScatterRuntime, ScatterParams, DEFAULT_SCATTER_SHADER_PATH};
use crate::layers::sphere_terrain::CesSphereTerrain;
use crate::layers::CesLayer;
use crate::shared_texture::{PackedTexture2DExtent, SharedPositionTexture};
use crate::texture_gen::{CubemapTextureGen, TerrainParams};
use godot::builtin::Callable;
use godot::builtin::{
    Aabb, Color, PackedFloat32Array, PackedInt32Array, PackedVector2Array, PackedVector3Array,
    Transform3D, Variant, Vector2, Vector3,
};
use godot::classes::mesh::ArrayType;
use godot::classes::mesh::PrimitiveType;
use godot::classes::multi_mesh::TransformFormat;
use godot::classes::notify::Node3DNotification;
use godot::classes::{
    ArrayMesh, Camera3D, Engine, INode3D, Image, ImageTexture3D, Material, Mesh, MultiMesh,
    MultiMeshInstance3D, Node, Node3D, RenderingDevice, RenderingServer, Script, Shader,
    ShaderMaterial, Texture2Drd, TextureCubemapRd,
};
use godot::prelude::*;
use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
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
        let Ok(gd) = variant.try_to::<Gd<T>>() else {
            continue;
        };
        result.push(gd);
    }
    result
}

/// Pure helper: counts how many booleans in the slice are `true`.
/// Extracted so we can unit-test the core logic without a Godot runtime.
#[cfg(test)]
fn count_enabled_flags(flags: &[bool]) -> usize {
    flags.iter().filter(|&&x| x).count()
}

/// Reads a row-major mat3x4 from 12 floats and reconstructs the matching
/// `Transform3D`. Layout matches the MultiMesh TRANSFORM_3D buffer convention
/// emitted by `ScatterPlacement.slang`:
///   [b0.x, b1.x, b2.x, origin.x,
///    b0.y, b1.y, b2.y, origin.y,
///    b0.z, b1.z, b2.z, origin.z]
/// where (b0, b1, b2) are the basis columns. In Godot the `Basis` is stored
/// row-major, so `rows[i] = (b0.<axis_i>, b1.<axis_i>, b2.<axis_i>)`.
fn transform3d_from_floats12(chunk: &[f32]) -> Transform3D {
    debug_assert_eq!(chunk.len(), 12, "expected 12 floats");
    let row0 = Vector3::new(chunk[0], chunk[1], chunk[2]);
    let row1 = Vector3::new(chunk[4], chunk[5], chunk[6]);
    let row2 = Vector3::new(chunk[8], chunk[9], chunk[10]);
    let origin = Vector3::new(chunk[3], chunk[7], chunk[11]);
    Transform3D::new(Basis::from_rows(row0, row1, row2), origin)
}

/// Inverse of `transform3d_from_floats12`: writes a `Transform3D` into 12
/// floats using the MultiMesh TRANSFORM_3D layout described above.
fn write_transform3d_floats12(chunk: &mut [f32], t: Transform3D) {
    debug_assert_eq!(chunk.len(), 12, "expected 12 floats");
    let row0 = t.basis.rows[0];
    let row1 = t.basis.rows[1];
    let row2 = t.basis.rows[2];
    chunk[0] = row0.x;
    chunk[1] = row0.y;
    chunk[2] = row0.z;
    chunk[3] = t.origin.x;
    chunk[4] = row1.x;
    chunk[5] = row1.y;
    chunk[6] = row1.z;
    chunk[7] = t.origin.y;
    chunk[8] = row2.x;
    chunk[9] = row2.y;
    chunk[10] = row2.z;
    chunk[11] = t.origin.z;
}

#[allow(dead_code)]
fn transform3d_iter_from_floats12(buf: &[f32]) -> impl Iterator<Item = Transform3D> + '_ {
    debug_assert!(
        buf.len() % 12 == 0,
        "buffer length must be a multiple of 12"
    );
    buf.chunks_exact(12).map(transform3d_from_floats12)
}

// Order matches `apply_baked_transform_in_place`: per_instance * baked is the
// existing convention; planet wraps it as the outermost transform.
fn compose_scatter_transform(
    planet: Transform3D,
    baked: Transform3D,
    per_instance: Transform3D,
) -> Transform3D {
    planet * per_instance * baked
}

#[allow(dead_code)]
fn resize_routing_flags(target_len: usize) -> Vec<bool> {
    vec![false; target_len]
}

/// Defensively reads a `use_rendering_server`-style flag from a `Variant`.
/// Returns the inner bool if the variant holds one; otherwise `false` (covers
/// nil, missing-property, and wrong-type cases). The actual conversion is
/// delegated to `read_use_rs_from_opt` so the defensive-default behavior is
/// unit-testable without a Godot runtime (Variant construction requires one).
fn read_use_rs(variant: Variant) -> bool {
    read_use_rs_from_opt(variant.try_to::<bool>().ok())
}

/// Pure core of `read_use_rs`: maps the `try_to::<bool>()` result to the
/// final flag value, defaulting to `false` when conversion failed (nil or
/// wrong-type variant).
fn read_use_rs_from_opt(parsed: Option<bool>) -> bool {
    parsed.unwrap_or(false)
}

/// Grow or shrink a `Vec<Rid>` to `target_len` by invoking the user-supplied
/// alloc/free callbacks. Used by the RenderingServer scatter path to keep
/// the per-variant instance pool sized to the worker's instance count.
/// - Grow: `alloc()` is called repeatedly until `pool.len() == target_len`.
/// - Shrink: each excess RID is popped from the back and passed to `free()`.
/// - Equal: no callbacks fire.
#[allow(dead_code)]
fn rid_pool_sync_size(
    pool: &mut Vec<Rid>,
    target_len: usize,
    mut alloc: impl FnMut() -> Rid,
    mut free: impl FnMut(Rid),
) {
    while pool.len() < target_len {
        pool.push(alloc());
    }
    while pool.len() > target_len {
        if let Some(rid) = pool.pop() {
            free(rid);
        }
    }
}

/// Incremental scatter applier (pure helper). For each `(tri_id, transform)`
/// in `added`, allocates a new RID via `alloc`, sets its transform, and
/// inserts into `map`. For each tri_id in `removed`, frees the matching RID
/// via `free` and removes it from `map`. Mirror of `rid_pool_sync_size` but
/// keyed by stable tri_id so re-pushes are unnecessary when the camera
/// hasn't moved enough to change the LOD shape.
pub fn apply_scatter_diff_with(
    map: &mut std::collections::HashMap<u64, Rid>,
    added: &[(u64, [f32; 12])],
    removed: &[u64],
    planet: Transform3D,
    baked: Transform3D,
    mut create_rid: impl FnMut() -> Rid,
    mut set_transform: impl FnMut(Rid, Transform3D),
    mut free: impl FnMut(Rid),
) {
    for tri_id in removed.iter().copied() {
        if let Some(rid) = map.remove(&tri_id) {
            free(rid);
        }
    }
    for (tri_id, per_inst_floats) in added.iter() {
        let rid = create_rid();
        let per_inst = transform3d_from_floats12(per_inst_floats);
        let composed = compose_scatter_transform(planet, baked, per_inst);
        set_transform(rid, composed);
        map.insert(*tri_id, rid);
    }
}

/// Zips each `Rid` with the matching `Transform3D` decoded from `buffer`,
/// composing `planet * per_instance * baked` and invoking `set` once per
/// pair. Pure helper extracted from the RS scatter apply path so the
/// composition order can be unit-tested without a `RenderingServer`.
#[allow(dead_code)]
fn apply_rs_transforms(
    rids: &[Rid],
    buffer: &[f32],
    planet: Transform3D,
    baked: Transform3D,
    mut set: impl FnMut(Rid, Transform3D),
) {
    for (rid, per_instance) in rids
        .iter()
        .copied()
        .zip(transform3d_iter_from_floats12(buffer))
    {
        set(rid, compose_scatter_transform(planet, baked, per_instance));
    }
}

/// Post-multiplies each 12-float mat3x4 chunk in `packed` by `baked` so
/// rendered instances appear at `scatter_t * baked_node_t`. The buffer
/// layout is row-major mat3x4 (per Godot MultiMesh::set_buffer convention).
fn apply_baked_transform_in_place(packed: &mut [f32], baked: Transform3D) {
    debug_assert_eq!(
        packed.len() % 12,
        0,
        "MultiMesh instance buffer must be a multiple of 12 floats"
    );
    for chunk in packed.chunks_exact_mut(12) {
        let scatter_t = transform3d_from_floats12(chunk);
        let composed = scatter_t * baked;
        write_transform3d_floats12(chunk, composed);
    }
}

/// Strips X and Z translation when `bake_xz` is false. Mirrors
/// `mesh_source::bake_node_transform` but operates directly on
/// `Transform3D` to avoid a NodeTrs round-trip at the call site.
fn bake_transform3d(t: Transform3D, bake_xz: bool) -> Transform3D {
    if bake_xz {
        return t;
    }
    Transform3D::new(t.basis, Vector3::new(0.0, t.origin.y, 0.0))
}

/// Build placeholder scatter results sized to the number of enabled layers.
/// This keeps worker outputs index-aligned with runtimes and MultiMesh children
/// even when a given frame generates zero instances.
///
/// The outer index is the scatter slot (one per enabled layer). The middle
/// index is the per-slot variant; `variant_counts[slot]` controls how many
/// inner buffers are pre-allocated for that slot. A slot with zero variants
/// receives an empty inner Vec (kept so the outer slot index stays aligned
/// with `WorkerState::scatter_runtimes`).
fn zeroed_scatter_results(variant_counts: &[u32]) -> (Vec<Vec<Vec<f32>>>, Vec<Vec<u32>>) {
    let scatter_buffers: Vec<Vec<Vec<f32>>> = variant_counts
        .iter()
        .map(|&n| vec![Vec::new(); n as usize])
        .collect();
    let scatter_instance_counts: Vec<Vec<u32>> = variant_counts
        .iter()
        .map(|&n| vec![0u32; n as usize])
        .collect();
    (scatter_buffers, scatter_instance_counts)
}

/// Counts the number of enabled scatter layers in a typed array, skipping null slots.
#[cfg(test)]
fn count_enabled_scatter_layers(arr: &Array<Gd<CesScatterLayerResource>>) -> usize {
    let flags: Vec<bool> = non_null_elements(arr)
        .iter()
        .map(|l| l.bind().enabled)
        .collect();
    count_enabled_flags(&flags)
}

/// Finalises runtime scatter params with the active planet radius.
fn scatter_params_with_runtime_context(mut params: ScatterParams, radius: f32) -> ScatterParams {
    params.planet_radius = radius;
    params
}

fn should_enable_vertex_position_texture_material(
    experimental_vertex_position_texture_spike: bool,
    has_vertex_texture: bool,
) -> bool {
    experimental_vertex_position_texture_spike && has_vertex_texture
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

/// One resolved variant from a scatter layer's `mesh_source`: the picked
/// mesh and the baked per-instance pre-transform extracted from that pick's
/// glTF node TRS.
struct ResolvedVariant {
    mesh: Gd<Mesh>,
    /// Optional node name from the source's `picks` array, used to build a
    /// readable child node name (`ScatterLayer{i}_Variant{v}_{node_name}`).
    node_name: Option<String>,
    baked: Transform3D,
}

/// Resolves a scatter layer's `mesh_source` resource (if set) to one entry
/// per `enabled = true` pick: the picked mesh + a baked per-instance
/// pre-transform. Returns an empty vec if `mesh_source` is unset, points to
/// a null gltf, or all picks are disabled.
///
/// Order: matches the `picks` array order (which mirrors the glTF child
/// order). Today only `CesMeshSourceGltf` is supported; other subclasses
/// produce an empty vec.
///
/// The instantiated scene root is created exactly once and freed at the end
/// so multi-variant resolution doesn't pay an N-times instantiate cost.
fn resolve_mesh_source_variants(layer_gd: &Gd<Resource>) -> Vec<ResolvedVariant> {
    use crate::mesh_source::all_enabled_pick_indices;
    use godot::classes::{MeshInstance3D, PackedScene};

    let source_var = layer_gd.get("mesh_source");
    if source_var.is_nil() {
        return Vec::new();
    }
    let Ok(source) = source_var.try_to::<Gd<Resource>>() else {
        return Vec::new();
    };

    // Discriminate by script class — same pattern as layer_script_class.
    let script_var = source.get("script");
    let Ok(script) = script_var.try_to::<Gd<Script>>() else {
        return Vec::new();
    };
    if script.get_global_name() != StringName::from("CesMeshSourceGltf") {
        return Vec::new();
    }

    let gltf_var = source.get("gltf");
    if gltf_var.is_nil() {
        return Vec::new();
    }
    let Ok(gltf) = gltf_var.try_to::<Gd<PackedScene>>() else {
        return Vec::new();
    };

    let picks_var = source.get("picks");
    if picks_var.is_nil() {
        return Vec::new();
    }
    let Ok(picks) = picks_var.try_to::<Array<Gd<Resource>>>() else {
        return Vec::new();
    };

    let bake_xz: bool = source
        .get("bake_xz_translation")
        .try_to::<bool>()
        .unwrap_or(false);

    // Walk the picks, collecting (index, enabled, node_name) tuples while
    // we still have ergonomic access to the array. Names are matched
    // against the instantiated scene below.
    let picks_vec = non_null_elements(&picks);
    let enabled_flags: Vec<bool> = picks_vec
        .iter()
        .map(|p| p.get("enabled").try_to::<bool>().unwrap_or(false))
        .collect();
    let enabled_indices = all_enabled_pick_indices(&enabled_flags);
    if enabled_indices.is_empty() {
        return Vec::new();
    }
    let chosen_names: Vec<StringName> = enabled_indices
        .iter()
        .filter_map(|&i| picks_vec[i].get("node_name").try_to::<StringName>().ok())
        .collect();
    if chosen_names.is_empty() {
        return Vec::new();
    }

    // Instantiate the scene exactly once and walk children, picking out
    // every MeshInstance3D whose name matches an enabled pick. We preserve
    // the *pick* order (which mirrors the glTF child order in the editor),
    // not the scene-graph traversal order.
    let Some(mut root) = gltf.instantiate() else {
        return Vec::new();
    };

    let mut found_by_name: std::collections::HashMap<StringName, (Gd<Mesh>, Transform3D)> =
        std::collections::HashMap::new();
    let children = root.get_children();
    for child_var in children.iter_shared() {
        let child_node: Gd<Node> = child_var;
        let child_name: StringName = child_node.get_name().into();
        if !chosen_names.contains(&child_name) {
            continue;
        }
        let Ok(mi3d) = child_node.try_cast::<MeshInstance3D>() else {
            continue;
        };
        let Some(mesh) = mi3d.get_mesh() else {
            continue;
        };
        let node_t = mi3d.get_transform();
        let baked = bake_transform3d(node_t, bake_xz);
        found_by_name.entry(child_name).or_insert((mesh, baked));
    }
    root.queue_free();

    // Emit in pick order, skipping any pick whose node_name didn't match a
    // child in the instantiated scene (defensive — the editor UI should
    // prevent this but we don't want a panic if it happens).
    let mut variants: Vec<ResolvedVariant> = Vec::new();
    for name in chosen_names.iter() {
        if let Some((mesh, baked)) = found_by_name.remove(name) {
            variants.push(ResolvedVariant {
                mesh,
                node_name: Some(name.to_string()),
                baked,
            });
        }
    }
    variants
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

/// One MultiMesh + baked per-instance transform pair for a single variant
/// inside a scatter layer slot. A layer with N enabled picks produces N
/// `ScatterVariant`s.
pub(crate) struct ScatterVariant {
    pub mmi: Gd<MultiMeshInstance3D>,
    pub baked_transform: Transform3D,
}

/// RenderingServer-path variant: each instance is its own `RID` (no
/// `MultiMeshInstance3D`). Used for slots routed through the RS path.
/// `instance_map` is a `tri_id → Rid` table maintained by the incremental
/// diff applier on the main thread. `material_rid` is `None` when the
/// layer's `material` is unset (skip `instance_geometry_set_material_override`
/// in that case).
pub(crate) struct ScatterRsVariant {
    pub mesh_rid: Rid,
    pub instance_map: std::collections::HashMap<u64, Rid>,
    pub baked_transform: Transform3D,
    pub material_rid: Option<Rid>,
}

#[derive(Clone, Copy, PartialEq)]
struct SettingsSnapshot {
    radius: f32,
    subdivisions: u32,
    triangle_screen_size: f32,
    low_poly_look: bool,
    precise_normals: bool,
    show_debug_messages: bool,
    show_debug_lod_histogram: bool,
    seed: i32,
    debug_snapshot_angle_offset: f32,
    show_snapshot_borders: bool,
    minimum_lod_update_time_ms: u32,
    scatter_force_shrink_each_dispatch: bool,
    experimental_vertex_position_texture_spike: bool,
}

#[derive(Clone, Copy, Debug)]
struct VertexTextureBinding {
    main_rid: Rid,
    extent: PackedTexture2DExtent,
    enabled: bool,
    /// Vertex count the worker dispatched for this frame. Carried separately
    /// from any placeholder Vec so the worker can skip allocating a multi-MB
    /// zeroed buffer just to derive the count on the main thread.
    vertex_count: u32,
}

impl VertexTextureBinding {
    fn disabled() -> Self {
        Self {
            main_rid: Rid::Invalid,
            extent: PackedTexture2DExtent {
                width: 1,
                height: 1,
            },
            enabled: false,
            vertex_count: 0,
        }
    }
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
    /// Per-slot, per-variant transform buffers. Outer index is the scatter
    /// slot (same order as enabled scatter layers). Middle index is the
    /// variant within that slot (each variant becomes its own MultiMesh).
    /// Inner is the packed mat3x4 float buffer. Empty when no scatter work
    /// was done.
    scatter_buffers: Vec<Vec<Vec<f32>>>,
    /// Number of transforms written per (slot, variant). Outer/middle index
    /// alignment matches `scatter_buffers`.
    scatter_instance_counts: Vec<Vec<u32>>,
    /// Per-slot, per-variant **added** instances since the previous worker
    /// iteration. Only populated for RS-routed slots (where the worker runs
    /// the tri_id diffing path); empty for MMI-routed slots.
    scatter_added: Vec<Vec<Vec<(u64, [f32; 12])>>>,
    /// Per-slot, per-variant **removed** tri_ids since the previous worker
    /// iteration. Same outer/middle alignment as `scatter_added`.
    scatter_removed: Vec<Vec<Vec<u64>>>,
    /// Worker-side timing snapshot for this frame. None when timing disabled.
    timing: Option<crate::perf::TimingTree>,
    /// Experimental shared position texture written on the worker/local RD and
    /// sampled by the main material in `vertex()`.
    vertex_texture: VertexTextureBinding,
    placeholder: Option<Vec<Vector3>>,
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
    show_debug_lod_histogram: bool,
    debug_snapshot_angle_offset: f32,
    show_snapshot_borders: bool,
    minimum_lod_update_time_ms: u32,
    scatter_force_shrink_each_dispatch: bool,
    experimental_vertex_position_texture_spike: bool,
    /// Debug bisect toggle forwarded from the inspector. When enabled, the
    /// worker returns before running the expensive mesh-generation path.
    debug_return_early_from_process: bool,
    scatter_params: Vec<ScatterParams>,
    /// Per-slot total variant count (number of enabled glTF picks; 1 for
    /// legacy single-mesh). Parallel to `scatter_params`. Drives the
    /// N-dispatch loop in the worker thread.
    scatter_variant_counts: Vec<u32>,
    /// Per-slot routing flag. When `true`, the worker emits per-variant
    /// (added, removed) diffs against the previous iteration; when `false`,
    /// the legacy flat-buffer output is used (consumed by MultiMesh).
    /// Parallel to `scatter_params`.
    scatter_slot_uses_rs: Vec<bool>,
}

struct WorkerState {
    rd: Gd<RenderingDevice>,
    graph_generator: Option<CesRunAlgo>,
    layers: Vec<Box<dyn CesLayer>>,
    texture_gen: CubemapTextureGen,
    normal_texture_gen: CubemapTextureGen,
    terrain_params: Option<TerrainParams>,
    snapshot_chain: CameraSnapshotTexture,
    position_texture: SharedPositionTexture,
    /// Parallel to the enabled scatter layers (same order). One runtime per layer.
    scatter_runtimes: Vec<CesScatterRuntime>,
    /// Last `vertex_count` for which we shipped a placeholder Vec to the main
    /// thread. None until the first texture-branch frame.
    last_placeholder_vertex_count: Option<u32>,
    /// Test-only: cloned from `CesCelestialRust::pending_dump_path`. The worker
    /// checks it at the end of each frame and dumps state if Some.
    pending_dump_path: Arc<Mutex<Option<PathBuf>>>,
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
    show_debug_messages: bool,

    /// Path of a CSV file that receives one row per `TimingNode` per frame
    /// when `show_debug_messages` is also enabled. Empty disables logging.
    /// Accepts `res://` and `user://` paths (resolved via `ProjectSettings`)
    /// or absolute filesystem paths.
    #[export]
    perf_csv_path: GString,

    /// Tag written to the `phase` column of every CSV row. Set from GDScript
    /// to mark before/after measurements.
    #[export]
    phase_label: GString,

    /// Debug perf toggle: force scatter runtimes to free + reallocate their
    /// output transform buffer every dispatch to test oversized-buffer
    /// readback cost.
    #[export]
    scatter_force_shrink_each_dispatch: bool,

    /// When true, perform extra GPU readbacks each frame to print the LOD
    /// histogram. Skews timing measurements; keep off when profiling.
    #[export]
    show_debug_lod_histogram: bool,

    #[export]
    show_process_timing: bool,

    /// Debug toggle: when enabled, `_process` returns immediately before any
    /// timing-tree setup, worker-result application, or new work dispatch.
    #[export]
    debug_return_early_from_process: bool,

    #[export]
    simulated_process_delay_ms: u32,

    #[export]
    seed: i32,

    /// Minimum time between LOD-driven worker updates.
    #[export]
    minimum_lod_update_time_ms: u32,

    /// Debug: angular offset (radians) applied to snapshot center direction.
    /// Set non-zero to animate snapshot positions independently of camera.
    #[export]
    debug_snapshot_angle_offset: f32,

    /// Show red debug borders on LOD snapshot patches.
    #[export]
    show_snapshot_borders: bool,

    /// Experimental shared vertex-position texture path. Disabled by default
    /// and intentionally separate from the normal CPU-readback final-state path.
    #[export]
    experimental_vertex_position_texture_spike: bool,

    #[export]
    height_layers: Array<Gd<CesHeightLayerResource>>,

    #[export]
    texture_layers: Array<Gd<CesTextureLayerResource>>,

    #[export]
    scatter_layers: Array<Gd<CesScatterLayerResource>>,

    instance: Rid,
    mesh: Option<Gd<ArrayMesh>>,
    /// `ShaderMaterial` reused with the cached placeholder mesh. Building a
    /// fresh `ShaderMaterial` and calling `surface_set_material` every frame
    /// costs ~14 ms in the spike path; with the cached mesh we keep the same
    /// material bound and only update parameters that actually change per
    /// frame (snapshot LOD metadata via `apply_lod_shader_params`).
    placeholder_material_cache: Option<Gd<ShaderMaterial>>,
    /// Cache of the placeholder `ArrayMesh` used by the experimental vertex-
    /// position-texture path, keyed by `vertex_count`. When the next worker
    /// result reports the same vertex count, we reuse the cached mesh and skip
    /// `pack_vertices`/`pack_indices`/`pack_uvs`/`add_surface_from_arrays`.
    /// Cleared whenever `self.mesh = None` is set so legacy and spike paths
    /// can never alias the same `Gd<ArrayMesh>`.
    placeholder_mesh_cache: Option<(u32, Gd<ArrayMesh>)>,
    active_shader: Option<Gd<Shader>>,
    cubemap_texture: Option<Gd<TextureCubemapRd>>,
    normal_cubemap_texture: Option<Gd<TextureCubemapRd>>,
    vertex_position_texture: Option<Gd<Texture2Drd>>,
    vertex_position_texture_size: Vector2,
    snapshot_textures: Vec<Option<Gd<Texture2Drd>>>,
    snapshot_normal_textures: Vec<Option<Gd<Texture2Drd>>>,
    snapshot_info: Vec<(Vector3, Vector3, Vector3, f32)>,
    // Threading fields
    work_tx: Option<std::sync::mpsc::Sender<(Vector3, WorkerConfig)>>,
    /// Test-only: when set, the worker dumps its post-frame state to this path
    /// at the end of the next algo run, then clears the slot. Wired by
    /// `dump_test_state` #[func] and read by the worker thread.
    pending_dump_path: Arc<Mutex<Option<PathBuf>>>,
    result_rx: Option<std::sync::mpsc::Receiver<MeshResult>>,
    texture_cmd_tx: Option<std::sync::mpsc::Sender<TextureCommand>>,
    texture_result_rx: Option<std::sync::mpsc::Receiver<TextureResult>>,
    worker_handle: Option<std::thread::JoinHandle<WorkerState>>,
    gen_mesh_running: bool,
    last_cam_position: Vector3,
    last_obj_transform: Transform3D,
    last_lod_update_time: Option<Instant>,
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
    /// Per scatter layer slot: the N variants spawned for that layer.
    /// Outer index is the slot (one per enabled scatter layer) and must
    /// stay aligned with `WorkerState::scatter_runtimes` and
    /// `MeshResult::scatter_buffers`. Inner is one entry per variant.
    /// `variants.is_empty()` means the layer is disabled or has no mesh.
    /// `variants.len() == 1` for legacy single-mesh; `>= 1` for glTF with
    /// multiple enabled picks.
    scatter_slots: Vec<Vec<ScatterVariant>>,
    /// Parallel to `scatter_slots`: per-slot flag indicating whether that
    /// slot's instances are rendered via direct `RenderingServer` instances
    /// (Phase 2) instead of the `MultiMeshInstance3D` child. All `false` in
    /// Phase 1 — wiring only.
    scatter_slot_uses_rs: Vec<bool>,
    /// Parallel to `scatter_slots`: for slots routed through the
    /// RenderingServer path, holds per-variant mesh/material RIDs and the
    /// pool of live instance RIDs. For slots on the MMI path the inner Vec
    /// is empty (and the corresponding `scatter_slots` entry holds the
    /// variants instead).
    scatter_slots_rs: Vec<Vec<ScatterRsVariant>>,
    /// Cached scenario RID captured during `build_scatter_mmi_children` so
    /// the apply path can call `instance_create2` without re-querying
    /// `World3D` every frame. `Rid::Invalid` until first build.
    scenario_rid: Rid,
    /// Blue-noise Texture3D built once per active `CesScatterMaskTextureLayer`.
    blue_noise_texture: Option<Gd<ImageTexture3D>>,
    /// Raw R8 blue-noise bytes (length = noise_dim^3). Loaded once and shared
    /// between the mask Texture3D and every scatter runtime's GPU buffer.
    blue_noise_bytes: Vec<u8>,
    /// Scatter mask preview parameters forwarded to the planet shader material.
    scatter_mask_noise_strength: f32,
    scatter_mask_target_color: Color,
    scatter_mask_color_tolerance: f32,

    /// Monotonic per-instance frame counter, incremented each time a frame's
    /// timing tree is finalised and printed.
    frame_counter: u64,
    /// Lazily-opened buffered writer for `perf_csv_path`.
    perf_csv_file: Option<BufWriter<File>>,
    /// Path string used to open `perf_csv_file` (so we can detect changes and reopen).
    perf_csv_open_path: String,
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
            show_debug_messages: false,
            perf_csv_path: GString::new(),
            phase_label: GString::new(),
            scatter_force_shrink_each_dispatch: false,
            show_debug_lod_histogram: false,
            show_process_timing: false,
            debug_return_early_from_process: false,
            simulated_process_delay_ms: 0,
            seed: 0,
            minimum_lod_update_time_ms: 100,
            debug_snapshot_angle_offset: 0.0,
            show_snapshot_borders: true,
            experimental_vertex_position_texture_spike: false,
            height_layers: Array::new(),
            texture_layers: Array::new(),
            scatter_layers: Array::new(),
            instance: Rid::Invalid,
            mesh: None,
            placeholder_mesh_cache: None,
            placeholder_material_cache: None,
            active_shader: None,
            cubemap_texture: None,
            normal_cubemap_texture: None,
            vertex_position_texture: None,
            vertex_position_texture_size: Vector2::new(1.0, 1.0),
            snapshot_textures: Vec::new(),
            snapshot_normal_textures: Vec::new(),
            snapshot_info: Vec::new(),

            work_tx: None,
            pending_dump_path: Arc::new(Mutex::new(None)),
            result_rx: None,
            texture_cmd_tx: None,
            texture_result_rx: None,
            worker_handle: None,
            gen_mesh_running: false,
            last_cam_position: Vector3::ZERO,
            last_obj_transform: Transform3D::IDENTITY,
            last_lod_update_time: None,
            last_settings: SettingsSnapshot {
                radius: 1.0,
                subdivisions: 3,
                triangle_screen_size: 0.1,
                low_poly_look: true,
                precise_normals: false,
                show_debug_messages: false,
                show_debug_lod_histogram: false,
                seed: 0,
                debug_snapshot_angle_offset: 0.0,
                show_snapshot_borders: true,
                minimum_lod_update_time_ms: 100,
                scatter_force_shrink_each_dispatch: false,
                experimental_vertex_position_texture_spike: false,
            },
            values_updated: false,
            is_shutting_down: false,
            cubemap_resolution_active: 0,
            last_structure_id: String::new(),
            layers_dirty: false,
            structural_dirty: false,
            connected_layer_ids: Vec::new(),
            scatter_slots: Vec::new(),
            scatter_slot_uses_rs: Vec::new(),
            scatter_slots_rs: Vec::new(),
            scenario_rid: Rid::Invalid,
            blue_noise_texture: None,
            blue_noise_bytes: Vec::new(),
            scatter_mask_noise_strength: 1.0,
            scatter_mask_target_color: Color::from_rgba(0.2, 0.6, 0.2, 1.0),
            scatter_mask_color_tolerance: 0.3,
            frame_counter: 0,
            perf_csv_file: None,
            perf_csv_open_path: String::new(),
        }
    }

    fn enter_tree(&mut self) {
        if Engine::singleton().is_editor_hint() {
            self.is_shutting_down = false;
            self.gen_mesh_running = false;
            self.last_lod_update_time = None;
            self.values_updated = true;
            return;
        }

        self.is_shutting_down = false;
        self.gen_mesh_running = false;
        self.last_lod_update_time = None;
        self.values_updated = true;

        let mut rs = RenderingServer::singleton();
        self.instance = rs.instance_create();

        let scenario = self.base().get_world_3d().unwrap().get_scenario();
        rs.instance_set_scenario(self.instance, scenario);

        let mesh = ArrayMesh::new_gd();
        rs.instance_set_base(self.instance, mesh.get_rid());
        self.mesh = Some(mesh);
        self.vertex_position_texture = None;
        self.vertex_position_texture_size = Vector2::new(1.0, 1.0);

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
        snapshot_chain.set_minimum_update_interval_ms(self.minimum_lod_update_time_ms);
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
                    chain.set_minimum_update_interval_ms(self.minimum_lod_update_time_ms);
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
                    chain.set_minimum_update_interval_ms(self.minimum_lod_update_time_ms);
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
                self.scatter_mask_noise_strength = if noise_v.is_nil() {
                    1.0
                } else {
                    noise_v.to::<f32>()
                };
                self.scatter_mask_target_color = if color_v.is_nil() {
                    Color::from_rgba(0.2, 0.6, 0.2, 1.0)
                } else {
                    color_v.to::<Color>()
                };
                self.scatter_mask_color_tolerance = if tol_v.is_nil() {
                    0.3
                } else {
                    tol_v.to::<f32>()
                };
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

        if self.active_shader.is_none() && self.experimental_vertex_position_texture_spike {
            let shader =
                load::<Shader>("res://addons/celestial_sim/shaders/planet_texture.gdshader");
            self.active_shader = Some(shader);
            godot_print!(
                "[CesCelestialRust] Spike fallback: bound planet_texture.gdshader (no texture layer enabled)"
            );
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
        self.last_structure_id = layers_structure_id(
            &self.height_layers,
            &self.texture_layers,
            &self.scatter_layers,
        );
        self.connect_layer_signals();
        self.build_scatter_mmi_children();
    }

    fn ready(&mut self) {
        self.last_settings = self.current_settings();
        self.values_updated = true;
    }

    fn process(&mut self, _delta: f64) {
        let _process_timer =
            ScopedTimer::new("CesCelestialRust::process", self.show_process_timing);

        // Install a per-frame timing tree for the main thread so any
        // ThreadScope::enter() calls inside the body (including those nested
        // inside apply_mesh_result) accumulate into a single tree we can
        // emit at the end of the frame.
        let mut process_tree = if self.show_debug_messages {
            crate::perf::TimingTree::new()
        } else {
            crate::perf::TimingTree::disabled()
        };

        let applied_worker_timing: Option<crate::perf::TimingTree> = {
            let _tree_guard = crate::perf::install_thread_tree(&mut process_tree);
            let _root_scope = crate::perf::ThreadScope::enter("process_body");
            self.process_body()
        };

        // Emit ASCII tree + CSV row for the frame, but only when something
        // meaningful happened (apply_mesh_result was called this frame).
        if self.show_debug_messages && applied_worker_timing.is_some() {
            if let Some(worker_tree) = applied_worker_timing {
                process_tree.merge_subtree("worker", worker_tree);
            }
            let frame_id = self.frame_counter;
            self.frame_counter = self.frame_counter.wrapping_add(1);
            godot_print!("Frame timing tree:\n{}", process_tree.render_ascii());
            self.log_timing_tree_csv(&process_tree, frame_id);
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
    /// Resolve a Godot-style path (`res://` / `user://`) to an absolute
    /// filesystem path via `ProjectSettings::globalize_path`. Plain paths are
    /// returned unchanged.
    fn resolve_log_path(raw: &str) -> PathBuf {
        if raw.starts_with("res://") || raw.starts_with("user://") {
            let ps = godot::classes::ProjectSettings::singleton();
            let globalised = ps.globalize_path(raw);
            PathBuf::from(globalised.to_string())
        } else {
            PathBuf::from(raw)
        }
    }

    /// Append every node in `tree` to the configured CSV log file (if any),
    /// tagged with the supplied frame id and the current `phase_label`.
    /// Opens / re-opens the file lazily, writing the header on first use.
    fn log_timing_tree_csv(&mut self, tree: &crate::perf::TimingTree, frame_id: u64) {
        let raw = self.perf_csv_path.to_string();
        if raw.is_empty() {
            return;
        }

        // Reopen if path changed since last open.
        if self.perf_csv_open_path != raw {
            self.perf_csv_file = None;
            self.perf_csv_open_path.clear();
        }

        if self.perf_csv_file.is_none() {
            let path = Self::resolve_log_path(&raw);
            if let Some(parent) = path.parent() {
                let _ = std::fs::create_dir_all(parent);
            }
            // If the file doesn't yet exist, we'll need to write the header.
            let needs_header = !path.exists()
                || std::fs::metadata(&path)
                    .map(|m| m.len() == 0)
                    .unwrap_or(true);
            match OpenOptions::new().create(true).append(true).open(&path) {
                Ok(f) => {
                    let mut w = BufWriter::new(f);
                    if needs_header {
                        if let Err(e) = writeln!(w, "{}", crate::perf::csv_header()) {
                            godot_warn!("perf_csv: failed to write header to {raw}: {e}");
                            return;
                        }
                    }
                    self.perf_csv_file = Some(w);
                    self.perf_csv_open_path = raw.clone();
                }
                Err(e) => {
                    godot_warn!("perf_csv: failed to open {raw}: {e}");
                    return;
                }
            }
        }

        let phase = self.phase_label.to_string();
        let rows = crate::perf::render_tree_csv_rows(tree.root(), frame_id, &phase);
        if let Some(w) = self.perf_csv_file.as_mut() {
            for row in &rows {
                if let Err(e) = writeln!(w, "{}", crate::perf::format_csv_row(row)) {
                    godot_warn!("perf_csv: write failed: {e}");
                    break;
                }
            }
            if let Err(e) = w.flush() {
                godot_warn!("perf_csv: flush failed: {e}");
            }
        }
    }

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
            // The placeholder cache shares its underlying RID with `self.mesh`
            // when the spike path is active; freeing the mesh above already
            // released that RID, so just drop the cached `Gd<ArrayMesh>` here.
            self.placeholder_mesh_cache = None;
            self.placeholder_material_cache = None;
            self.vertex_position_texture = None;
            self.vertex_position_texture_size = Vector2::new(1.0, 1.0);
        }))
        .is_err()
        {
            eprintln!("CesCelestialRust shutdown: failed to free scene RIDs (engine unavailable)");
            self.instance = Rid::Invalid;
            self.mesh = None;
            self.placeholder_mesh_cache = None;
            self.placeholder_material_cache = None;
        }

        // Signal worker to stop and stop accepting results.
        drop(self.work_tx.take());
        drop(self.texture_cmd_tx.take());
        self.result_rx = None;
        self.texture_result_rx = None;
        self.snapshot_textures.clear();
        self.snapshot_normal_textures.clear();
        self.snapshot_info.clear();
        self.vertex_position_texture = None;
        self.vertex_position_texture_size = Vector2::new(1.0, 1.0);

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
                    worker.position_texture.dispose_local(&mut worker.rd);
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
        self.last_lod_update_time = None;
        self.values_updated = true;

        let mut rs = RenderingServer::singleton();
        self.instance = rs.instance_create();

        let scenario = self.base().get_world_3d().unwrap().get_scenario();
        rs.instance_set_scenario(self.instance, scenario);

        let mesh = ArrayMesh::new_gd();
        rs.instance_set_base(self.instance, mesh.get_rid());
        self.mesh = Some(mesh);
        self.vertex_position_texture = None;
        self.vertex_position_texture_size = Vector2::new(1.0, 1.0);

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
        snapshot_chain.set_minimum_update_interval_ms(self.minimum_lod_update_time_ms);
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
                    chain.set_minimum_update_interval_ms(self.minimum_lod_update_time_ms);
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
                    chain.set_minimum_update_interval_ms(self.minimum_lod_update_time_ms);
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
                self.scatter_mask_noise_strength = if noise_v.is_nil() {
                    1.0
                } else {
                    noise_v.to::<f32>()
                };
                self.scatter_mask_target_color = if color_v.is_nil() {
                    Color::from_rgba(0.2, 0.6, 0.2, 1.0)
                } else {
                    color_v.to::<Color>()
                };
                self.scatter_mask_color_tolerance = if tol_v.is_nil() {
                    0.3
                } else {
                    tol_v.to::<f32>()
                };
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

        if self.active_shader.is_none() && self.experimental_vertex_position_texture_spike {
            let shader =
                load::<Shader>("res://addons/celestial_sim/shaders/planet_texture.gdshader");
            self.active_shader = Some(shader);
            godot_print!(
                "[CesCelestialRust] Spike fallback: bound planet_texture.gdshader (no texture layer enabled)"
            );
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
        self.last_structure_id = layers_structure_id(
            &self.height_layers,
            &self.texture_layers,
            &self.scatter_layers,
        );
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
            show_debug_messages: self.show_debug_messages,
            show_debug_lod_histogram: self.show_debug_lod_histogram,
            seed: self.seed,
            debug_snapshot_angle_offset: self.debug_snapshot_angle_offset,
            show_snapshot_borders: read_show_snapshot_borders(
                &self.texture_layers,
                self.show_snapshot_borders,
            ),
            minimum_lod_update_time_ms: self.minimum_lod_update_time_ms,
            scatter_force_shrink_each_dispatch: self.scatter_force_shrink_each_dispatch,
            experimental_vertex_position_texture_spike: self
                .experimental_vertex_position_texture_spike,
        }
    }

    fn minimum_lod_update_interval(&self) -> Duration {
        Duration::from_millis(self.minimum_lod_update_time_ms as u64)
    }

    fn can_dispatch_lod_update(&self) -> bool {
        match self.last_lod_update_time {
            None => true,
            Some(t) => t.elapsed() >= self.minimum_lod_update_interval(),
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

    /// Main-thread `_process` body. Extracted from `process` so the timing
    /// tree install at the top of `process` can wrap the whole body even
    /// across early returns. Returns `Some(worker_tree)` if `apply_mesh_result`
    /// was called this tick (signalling that a frame timing tree should be
    /// emitted), and `None` otherwise.
    fn process_body(&mut self) -> Option<crate::perf::TimingTree> {
        let should_run_preview = self.should_run_editor_preview();
        if !should_run_preview {
            if self.instance.is_valid() || self.work_tx.is_some() || self.worker_handle.is_some() {
                self.shutdown_for_reload_or_exit();
            }
            return None;
        }

        if !self.instance.is_valid() || self.work_tx.is_none() || self.worker_handle.is_none() {
            self.restart_with_current_layers();
            return None;
        }

        if self.is_shutting_down {
            return None;
        }

        // Detect structural changes (layer added/removed/reordered/enabled toggle)
        {
            let _g = crate::perf::ThreadScope::enter("dirty_handling");
            let current_sid = layers_structure_id(
                &self.height_layers,
                &self.texture_layers,
                &self.scatter_layers,
            );
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
                return None;
            }
        }

        // Check if cubemap resolution changed — send lightweight regenerate command
        {
            let _g = crate::perf::ThreadScope::enter("check_cubemap_restart");
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
        }

        // Check for completed texture regeneration results
        {
            let _g = crate::perf::ThreadScope::enter("texture_result_drain");
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
        }

        if self.show_process_timing {
            godot_print!(
                "simulated_process_delay_ms = {}",
                self.simulated_process_delay_ms
            );
        }
        if self.simulated_process_delay_ms > 0 {
            thread::sleep(Duration::from_millis(
                self.simulated_process_delay_ms as u64,
            ));
        }
        // 1. Check for completed results (non-blocking)
        let mut applied_worker_timing: Option<crate::perf::TimingTree> = None;
        {
            let _g = crate::perf::ThreadScope::enter("worker_result_consume");
            if let Some(ref result_rx) = self.result_rx {
                if let Ok(result) = result_rx.try_recv() {
                    self.gen_mesh_running = false;
                    applied_worker_timing = self.apply_mesh_result(result);
                    // Ensure we always return a Some (so the caller emits the
                    // frame tree) even if the worker did not include timing.
                    if applied_worker_timing.is_none() {
                        applied_worker_timing = Some(crate::perf::TimingTree::disabled());
                    }
                }
            }
        }

        let global_transform = self.base().get_global_transform();
        let mut rs = RenderingServer::singleton();
        rs.instance_set_transform(self.instance, global_transform);

        // 2. Check if we need new work
        let cam = self.get_camera();
        if cam.is_none() {
            return applied_worker_timing;
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
            return applied_worker_timing;
        }

        if !self.can_dispatch_lod_update() {
            return applied_worker_timing;
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
            show_debug_lod_histogram: self.show_debug_lod_histogram,
            debug_snapshot_angle_offset: self.debug_snapshot_angle_offset,
            show_snapshot_borders: read_show_snapshot_borders(
                &self.texture_layers,
                self.show_snapshot_borders,
            ),
            minimum_lod_update_time_ms: self.minimum_lod_update_time_ms,
            scatter_force_shrink_each_dispatch: self.scatter_force_shrink_each_dispatch,
            experimental_vertex_position_texture_spike: self
                .experimental_vertex_position_texture_spike,
            debug_return_early_from_process: self.debug_return_early_from_process,
            scatter_params: read_enabled_scatter_layer_params(&self.scatter_layers),
            scatter_variant_counts: self.scatter_variant_counts(),
            scatter_slot_uses_rs: self.scatter_slot_uses_rs.clone(),
        };

        if let Some(ref work_tx) = self.work_tx {
            if work_tx.send((cam_local, config)).is_ok() {
                self.gen_mesh_running = true;
                self.last_lod_update_time = Some(Instant::now());
            } else {
                godot_print!("[CesCelestialRust] ERROR: Failed to send work to worker thread");
            }
        } else {
            godot_print!("[CesCelestialRust] ERROR: work_tx is None, worker not spawned");
        }

        applied_worker_timing
    }

    /// Apply a worker mesh result on the main thread. Returns the worker's
    /// timing tree (if any) so the caller can merge it under the frame's
    /// timing tree before emitting CSV.
    fn apply_mesh_result(&mut self, result: MeshResult) -> Option<crate::perf::TimingTree> {
        let _scope_apply = crate::perf::ThreadScope::enter("apply_mesh_result");

        let has_mesh = (!result.pos.is_empty() && !result.triangles.is_empty())
            || (result.vertex_texture.enabled && result.vertex_texture.vertex_count > 0);
        let worker_timing = result.timing;

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

        if result.vertex_texture.enabled && result.vertex_texture.main_rid.is_valid() {
            if let Some(ref mut tex) = self.vertex_position_texture {
                tex.set_texture_rd_rid(result.vertex_texture.main_rid);
            } else {
                let mut tex = Texture2Drd::new_gd();
                tex.set_texture_rd_rid(result.vertex_texture.main_rid);
                self.vertex_position_texture = Some(tex);
            }
            self.vertex_position_texture_size = Vector2::new(
                result.vertex_texture.extent.width as f32,
                result.vertex_texture.extent.height as f32,
            );
        } else {
            self.vertex_position_texture = None;
            self.vertex_position_texture_size = Vector2::new(1.0, 1.0);
        }

        // Apply scatter transforms to per-(slot, variant) MultiMeshInstance3D
        // children. Missing result entries are treated as zero-instance slots
        // so stale MultiMesh counts are cleared instead of lingering on screen.
        {
            let _g = crate::perf::ThreadScope::enter("apply_scatter_mm");

            // RS-routed slots: grow/shrink instance RID pools and push
            // composed transforms via RenderingServer. Planet transform is
            // captured once per apply (planet is static).
            let planet_xform = self.base().get_global_transform();
            let scenario = self.scenario_rid;
            for (slot_idx, rs_variants) in self.scatter_slots_rs.iter_mut().enumerate() {
                if rs_variants.is_empty() {
                    continue;
                }
                let added_slot = result.scatter_added.get(slot_idx);
                let removed_slot = result.scatter_removed.get(slot_idx);
                for (v_idx, variant) in rs_variants.iter_mut().enumerate() {
                    let added: &[(u64, [f32; 12])] = added_slot
                        .and_then(|s| s.get(v_idx))
                        .map(|v| v.as_slice())
                        .unwrap_or(&[]);
                    let removed: &[u64] = removed_slot
                        .and_then(|s| s.get(v_idx))
                        .map(|v| v.as_slice())
                        .unwrap_or(&[]);
                    if added.is_empty() && removed.is_empty() {
                        continue;
                    }
                    let mesh_rid = variant.mesh_rid;
                    let material_rid = variant.material_rid;
                    let baked = variant.baked_transform;
                    apply_scatter_diff_with(
                        &mut variant.instance_map,
                        added,
                        removed,
                        planet_xform,
                        baked,
                        || {
                            let mut rs = RenderingServer::singleton();
                            let rid = rs.instance_create2(mesh_rid, scenario);
                            if let Some(mat) = material_rid {
                                rs.instance_geometry_set_material_override(rid, mat);
                            }
                            rid
                        },
                        |rid, t| RenderingServer::singleton().instance_set_transform(rid, t),
                        |rid| RenderingServer::singleton().free_rid(rid),
                    );
                }
            }

            for (slot_idx, variants) in self.scatter_slots.iter().enumerate() {
                let slot_buffers = result.scatter_buffers.get(slot_idx);
                let slot_counts = result.scatter_instance_counts.get(slot_idx);
                for (v_idx, variant) in variants.iter().enumerate() {
                    let Some(mut mm) = variant.mmi.get_multimesh() else {
                        continue;
                    };
                    let instance_count = slot_counts
                        .and_then(|c| c.get(v_idx))
                        .copied()
                        .unwrap_or(0) as i32;
                    if mm.get_instance_count() != instance_count {
                        mm.set_instance_count(instance_count);
                    }
                    if instance_count == 0 {
                        continue;
                    }
                    let Some(buf) = slot_buffers.and_then(|s| s.get(v_idx)) else {
                        continue;
                    };
                    if buf.is_empty() {
                        continue;
                    }
                    let baked = variant.baked_transform;
                    let packed = if baked == Transform3D::IDENTITY {
                        PackedFloat32Array::from(buf.as_slice())
                    } else {
                        let mut owned: Vec<f32> = buf.clone();
                        apply_baked_transform_in_place(&mut owned, baked);
                        PackedFloat32Array::from(owned.as_slice())
                    };
                    let layer_label = format!("set_buffer_slot_{slot_idx}_variant_{v_idx}");
                    let _g2 = crate::perf::ThreadScope::enter(&layer_label);
                    mm.set_buffer(&packed);
                }
            }
        }

        if !has_mesh {
            return worker_timing;
        }

        // Detect the experimental vertex-texture branch. On this branch the
        // worker no longer ships placeholder Vecs through `MeshResult`; it just
        // reports `vertex_count` so the main thread can build (or reuse) the
        // placeholder `ArrayMesh` itself.
        let texture_branch =
            self.experimental_vertex_position_texture_spike && result.vertex_texture.enabled;
        let texture_vertex_count = result.vertex_texture.vertex_count;

        let MeshResult {
            pos,
            triangles,
            uv,
            vertex_texture: _,
            placeholder,
            ..
        } = result;
        let vertex_count = if texture_branch {
            texture_vertex_count
        } else {
            u32::try_from(pos.len()).unwrap_or(u32::MAX)
        };
        let triangle_count = if texture_branch {
            (vertex_count / 3) as usize
        } else {
            triangles.len() / 3
        };

        // On the texture branch, the worker decides whether to ship a
        // placeholder Vec (only when vertex_count changed). If shipped, we
        // build the `ArrayMesh` from it; otherwise reuse the cached mesh.
        let cached_count = self.placeholder_mesh_cache.as_ref().map(|(c, _)| *c);
        let cache_hit = texture_branch
            && placeholder.is_none()
            && crate::vertex_texture_spike::placeholder_mesh_cache_hits(cached_count, vertex_count);

        let mut new_mesh: Gd<ArrayMesh> = if cache_hit {
            let _g = crate::perf::ThreadScope::enter("consume_placeholder");
            // Safe to unwrap: cache_hit implies the cache entry exists.
            let (_, ref cached_mesh) = self
                .placeholder_mesh_cache
                .as_ref()
                .expect("cache_hit implies populated cache");
            cached_mesh.clone()
        } else if texture_branch {
            // Worker shipped a payload (vertex_count changed) — or the cache
            // is unexpectedly empty. Either way, build the `ArrayMesh` from
            // the worker-built Vec, falling back to a main-thread zero-fill
            // if no payload arrived.
            let positions = placeholder.unwrap_or_else(|| {
                // Race fallback: worker decided "unchanged" but main-thread cache
                // was cleared (e.g. spike toggled off and back on). Rebuild here.
                let _g = crate::perf::ThreadScope::enter("build_placeholder_mesh_main");
                vec![Vector3::ZERO; vertex_count as usize]
            });
            let packed_verts = {
                let _g = crate::perf::ThreadScope::enter("pack_vertices");
                PackedVector3Array::from(&positions[..])
            };

            let _g = crate::perf::ThreadScope::enter("add_surface_from_arrays");
            let mut surface_array = varray![];
            surface_array.resize(ArrayType::MAX.ord() as usize, &Variant::nil());
            surface_array.set(ArrayType::VERTEX.ord() as usize, &packed_verts.to_variant());

            let mut new_mesh = ArrayMesh::new_gd();
            new_mesh.add_surface_from_arrays(PrimitiveType::TRIANGLES, &surface_array);
            new_mesh
        } else {
            let packed_verts = {
                let _g = crate::perf::ThreadScope::enter("pack_vertices");
                PackedVector3Array::from(pos)
            };
            let packed_indices = {
                let _g = crate::perf::ThreadScope::enter("pack_indices");
                PackedInt32Array::from(triangles)
            };
            let packed_uvs: PackedVector2Array = {
                let _g = crate::perf::ThreadScope::enter("pack_uvs");
                PackedVector2Array::from(uv)
            };

            let _g = crate::perf::ThreadScope::enter("add_surface_from_arrays");
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
            new_mesh
        };

        if let Some(ref shader) = self.active_shader {
            let _g = crate::perf::ThreadScope::enter("material_setup");
            // Build the `ShaderMaterial` lazily and reuse it across frames.
            // On the spike path the heavy cost is `surface_set_material` (which
            // we used to call every frame even on cache hit); reusing the
            // material lets us skip that entirely once it's bound. Snapshot /
            // LOD parameters that DO change per frame are still re-applied via
            // `apply_lod_shader_params` below.
            let needs_new_material = self.placeholder_material_cache.is_none() || !cache_hit;
            if needs_new_material {
                let mut material = self
                    .placeholder_material_cache
                    .clone()
                    .unwrap_or_else(ShaderMaterial::new_gd);
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
                {
                    let _g2 = crate::perf::ThreadScope::enter("apply_lod_shader_params");
                    self.apply_lod_shader_params(&mut material);
                }
                new_mesh.surface_set_material(0, &material);
                self.placeholder_material_cache = Some(material);
            } else {
                // Cache hit and we already have a material bound to this
                // mesh — only re-apply the per-frame LOD parameters.
                if let Some(ref mut material) = self.placeholder_material_cache {
                    let _g2 = crate::perf::ThreadScope::enter("apply_lod_shader_params");
                    let mut m = material.clone();
                    self.apply_lod_shader_params(&mut m);
                }
            }
        }

        {
            let _g = crate::perf::ThreadScope::enter("instance_set_base_and_free_old");
            let mut rs = RenderingServer::singleton();

            // Only rebind the instance base when the mesh RID actually changed.
            let prev_rid = self.mesh.as_ref().map(|m| m.get_rid());
            let new_rid = new_mesh.get_rid();
            if prev_rid != Some(new_rid) {
                rs.instance_set_base(self.instance, new_rid);
            }

            if self.experimental_vertex_position_texture_spike {
                let r = self.radius * 1.5;
                let aabb = Aabb {
                    position: Vector3::new(-r, -r, -r),
                    size: Vector3::new(2.0 * r, 2.0 * r, 2.0 * r),
                };
                rs.instance_set_custom_aabb(self.instance, aabb);
            }

            // Free the previously-bound mesh only when it isn't the cached
            // mesh we're about to reuse. Without this guard the cache hit
            // path would free its own RID.
            if !cache_hit {
                if let Some(ref old_mesh) = self.mesh {
                    if old_mesh.get_rid() != new_rid {
                        rs.free_rid(old_mesh.get_rid());
                    }
                }
            }
            self.mesh = Some(new_mesh.clone());

            // Populate / refresh the placeholder cache after a miss on the
            // texture branch so the next frame can hit. On the legacy path
            // (texture_branch == false) we do not cache — that path needs a
            // fresh mesh built from real positions every frame.
            if texture_branch && !cache_hit {
                self.placeholder_mesh_cache = Some((vertex_count, new_mesh));
            } else if !texture_branch {
                // Spike flag flipped off (or worker didn't produce a vertex
                // texture this frame): drop the cache so it doesn't alias the
                // legacy mesh and so the next spike-on frame rebuilds.
                self.placeholder_mesh_cache = None;
                self.placeholder_material_cache = None;
            }
        }

        if self.show_debug_messages {
            godot_print!("Mesh Triangles: {}", triangle_count);
        }

        worker_timing
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
            self.scatter_mask_noise_strength = if noise_v.is_nil() {
                1.0
            } else {
                noise_v.to::<f32>()
            };
            self.scatter_mask_target_color = if color_v.is_nil() {
                Color::from_rgba(0.2, 0.6, 0.2, 1.0)
            } else {
                color_v.to::<Color>()
            };
            self.scatter_mask_color_tolerance = if tol_v.is_nil() {
                0.3
            } else {
                tol_v.to::<f32>()
            };
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
        let use_vertex_position_texture = should_enable_vertex_position_texture_material(
            self.experimental_vertex_position_texture_spike,
            self.vertex_position_texture.is_some(),
        );
        material.set_shader_parameter(
            "use_vertex_position_texture",
            &use_vertex_position_texture.to_variant(),
        );
        material.set_shader_parameter(
            "vertex_position_texture_size",
            &self.vertex_position_texture_size.to_variant(),
        );
        if let Some(ref vertex_texture) = self.vertex_position_texture {
            material.set_shader_parameter("vertex_position_texture", &vertex_texture.to_variant());
        }

        // Scatter mask preview uniforms — silently ignored by other shaders.
        if let Some(ref tex3d) = self.blue_noise_texture {
            material.set_shader_parameter("blue_noise_texture", &tex3d.to_variant());
        }
        material.set_shader_parameter(
            "noise_strength",
            &self.scatter_mask_noise_strength.to_variant(),
        );
        material.set_shader_parameter("target_color", &self.scatter_mask_target_color.to_variant());
        material.set_shader_parameter(
            "color_tolerance",
            &self.scatter_mask_color_tolerance.to_variant(),
        );
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

        let pending_dump_path = self.pending_dump_path.clone();
        let mut worker = WorkerState {
            rd,
            graph_generator: None,
            layers,
            texture_gen,
            normal_texture_gen,
            terrain_params,
            snapshot_chain,
            position_texture: SharedPositionTexture::new(),
            scatter_runtimes,
            last_placeholder_vertex_count: None,
            pending_dump_path,
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
                worker
                    .snapshot_chain
                    .set_minimum_update_interval_ms(config.minimum_lod_update_time_ms);

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
                    show_debug_lod_histogram: config.show_debug_lod_histogram,
                };

                if worker.graph_generator.is_none() {
                    worker.graph_generator = Some(CesRunAlgo::new());
                }

                // Per-frame timing tree on the worker thread. When debug
                // messages are off, the tree is `disabled()` so all scope
                // construction becomes a no-op.
                let mut worker_tree = if config.show_debug_messages {
                    crate::perf::TimingTree::new()
                } else {
                    crate::perf::TimingTree::disabled()
                };

                // Install the tree as this thread's current tree for the
                // remainder of this iteration. The guard restores the previous
                // pointer on Drop, so the tree stays "active" for both the algo
                // and the scatter dispatches that follow.
                let _tree_guard = crate::perf::install_thread_tree(&mut worker_tree);
                let _frame_scope = crate::perf::ThreadScope::enter("worker_frame");

                let (cpu_output, texture_output) = {
                    let WorkerState {
                        ref mut rd,
                        ref mut graph_generator,
                        ref mut layers,
                        ref mut position_texture,
                        ..
                    } = worker;
                    let gen = graph_generator.as_mut().unwrap();
                    if config.experimental_vertex_position_texture_spike {
                        (
                            None,
                            Some(gen.update_triangle_graph_to_position_texture(
                                rd,
                                cam_local,
                                &algo_config,
                                layers,
                                false,
                                position_texture,
                                config.debug_return_early_from_process,
                            )),
                        )
                    } else {
                        (
                            Some(gen.update_triangle_graph(
                                rd,
                                cam_local,
                                &algo_config,
                                layers,
                                false,
                            )),
                            None,
                        )
                    }
                };

                // Test-only: if a dump was requested, run it now (before the
                // result_tx send) so the file exists by the time the test
                // scene's poll loop next checks. Take() so each request fires
                // exactly once.
                let dump_path: Option<PathBuf> = worker
                    .pending_dump_path
                    .lock()
                    .ok()
                    .and_then(|mut g| g.take());
                if let Some(path) = dump_path {
                    let WorkerState {
                        ref mut rd,
                        ref mut graph_generator,
                        ..
                    } = worker;
                    if let Some(gen) = graph_generator.as_mut() {
                        match gen.dump_visible_state(rd, config.radius, &path) {
                            Ok(()) => godot_print!(
                                "[CesCelestialRust] Dumped LOD test state to {}",
                                path.display()
                            ),
                            Err(e) => godot_print!(
                                "[CesCelestialRust] Failed to dump LOD test state to {}: {}",
                                path.display(),
                                e
                            ),
                        }
                    }
                }

                let has_final_mesh = if let Some(ref output) = cpu_output {
                    !output.pos.is_empty() && !output.tris.is_empty()
                } else {
                    texture_output
                        .as_ref()
                        .map(|output| !output.is_empty())
                        .unwrap_or(false)
                };

                if !has_final_mesh {
                    drop(_frame_scope);
                    drop(_tree_guard);
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
                        scatter_added: vec![],
                        scatter_removed: vec![],
                        timing: if config.show_debug_messages {
                            Some(worker_tree)
                        } else {
                            None
                        },
                        vertex_texture: VertexTextureBinding::disabled(),
                        placeholder: None,
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
                // For layers with N>1 variants we dispatch N times (one per
                // variant_id), and the shader filters each dispatch to the
                // triangles hashing to that variant. This reuses the existing
                // single-counter buffer layout — N dispatches, N readbacks.
                let scatter_slot_count = worker.scatter_runtimes.len();
                // Variant counts parallel to scatter_runtimes; fall back to 1
                // for any slot the config didn't populate (shouldn't happen,
                // but keeps the indexing safe).
                let scatter_variant_counts: Vec<u32> = (0..scatter_slot_count)
                    .map(|i| {
                        config
                            .scatter_variant_counts
                            .get(i)
                            .copied()
                            .filter(|&n| n > 0)
                            .unwrap_or(1)
                    })
                    .collect();
                let (mut scatter_buffers, mut scatter_instance_counts) =
                    zeroed_scatter_results(&scatter_variant_counts);
                // Parallel structures for the incremental (RS-routed) path.
                let mut scatter_added: Vec<Vec<Vec<(u64, [f32; 12])>>> = scatter_variant_counts
                    .iter()
                    .map(|&n| vec![Vec::new(); n as usize])
                    .collect();
                let mut scatter_removed: Vec<Vec<Vec<u64>>> = scatter_variant_counts
                    .iter()
                    .map(|&n| vec![Vec::new(); n as usize])
                    .collect();
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
                            // Gated on a separate flag because the readbacks pollute timing.
                            if config.show_debug_lod_histogram {
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
                            let _scatter_root = crate::perf::ThreadScope::enter("scatter");
                            for (slot, runtime) in scatter_runtimes.iter_mut().enumerate() {
                                let variant_count = scatter_variant_counts[slot];
                                let base_params = config
                                    .scatter_params
                                    .get(slot)
                                    .copied()
                                    .unwrap_or_else(|| *runtime.params());
                                runtime.set_force_shrink_each_dispatch(
                                    config.scatter_force_shrink_each_dispatch,
                                );
                                let uses_incremental = config
                                    .scatter_slot_uses_rs
                                    .get(slot)
                                    .copied()
                                    .unwrap_or(false);
                                if runtime.uses_incremental() != uses_incremental {
                                    runtime.set_incremental(
                                        uses_incremental,
                                        variant_count as usize,
                                    );
                                }
                                runtime.init_pipeline(rd);
                                let layer_label = format!("layer_{slot}");
                                let _layer_scope = crate::perf::ThreadScope::enter(&layer_label);

                                for variant_id in 0..variant_count {
                                    let mut params = scatter_params_with_runtime_context(
                                        base_params,
                                        config.radius,
                                    );
                                    params.variant_id = variant_id;
                                    params.variant_count = variant_count;
                                    runtime.set_params(params);

                                    let variant_label = format!("variant_{variant_id}");
                                    let _variant_scope =
                                        crate::perf::ThreadScope::enter(&variant_label);
                                    {
                                        let _g = crate::perf::ThreadScope::enter("dispatch");
                                        runtime.dispatch(rd, &tp_for_scatter, state);
                                    }
                                    let count = {
                                        let _g = crate::perf::ThreadScope::enter("readback_count");
                                        runtime.readback_count(rd)
                                    };
                                    if uses_incremental {
                                        let transforms = {
                                            let _g = crate::perf::ThreadScope::enter(
                                                "readback_compact",
                                            );
                                            runtime.readback_compact(rd, count)
                                        };
                                        let tri_ids = {
                                            let _g = crate::perf::ThreadScope::enter(
                                                "readback_tri_ids",
                                            );
                                            runtime.readback_tri_ids(rd, count)
                                        };
                                        let _g = crate::perf::ThreadScope::enter("diff");
                                        let (added, removed) = runtime.swap_previous_and_diff(
                                            variant_id as usize,
                                            tri_ids,
                                            &transforms,
                                        );
                                        scatter_added[slot][variant_id as usize] = added;
                                        scatter_removed[slot][variant_id as usize] = removed;
                                    } else {
                                        let _g =
                                            crate::perf::ThreadScope::enter("readback_compact");
                                        scatter_buffers[slot][variant_id as usize] =
                                            runtime.readback_compact(rd, count);
                                    }
                                    scatter_instance_counts[slot][variant_id as usize] = count;
                                    if config.show_debug_messages {
                                        godot_print!(
                                            "[CesCelestialRust] Scatter layer {} variant {}/{} (level {}): {} instances spawned (n_tris={})",
                                            slot,
                                            variant_id,
                                            variant_count,
                                            params.subdivision_level,
                                            count,
                                            state.n_tris
                                        );
                                    }
                                }
                            }
                        }
                    }
                }

                let (mesh_pos, mesh_triangles, mesh_uv) = if let Some(output) = cpu_output {
                    (output.pos, output.tris, output.uv)
                } else {
                    (vec![], vec![], vec![])
                };
                let mut vertex_texture = VertexTextureBinding::disabled();

                if let Some(texture_output) = texture_output {
                    let _vertex_texture_scope =
                        crate::perf::ThreadScope::enter("vertex_texture_final_output");
                    let extent = worker.position_texture.extent();
                    vertex_texture = VertexTextureBinding {
                        main_rid: worker.position_texture.back_main_rid(),
                        extent,
                        enabled: true,
                        vertex_count: texture_output.vertex_count(),
                    };
                    worker.position_texture.swap_buffers();
                    if config.show_debug_messages {
                        godot_print!(
                            "[CesCelestialRust] Experimental vertex texture final-state path updated ({} verts, {}x{})",
                            texture_output.vertex_count(),
                            extent.width,
                            extent.height
                        );
                    }
                }

                // Texture branch only: build the zero-fill placeholder Vec on the
                // worker when vertex_count changed since the last frame we shipped
                // one. Main thread owns the Vec→PackedVector3Array copy.
                let placeholder = if vertex_texture.enabled {
                    let vc = vertex_texture.vertex_count;
                    if crate::vertex_texture_spike::should_build_placeholder(
                        worker.last_placeholder_vertex_count,
                        vc,
                    ) {
                        let _g = crate::perf::ThreadScope::enter("build_placeholder_worker");
                        let positions = vec![Vector3::ZERO; vc as usize];
                        worker.last_placeholder_vertex_count = Some(vc);
                        Some(positions)
                    } else {
                        None
                    }
                } else {
                    None
                };

                // Tear down the frame scope and the thread-tree pointer BEFORE
                // we move `worker_tree` into `MeshResult` (their Drop impls
                // would otherwise dereference a moved-out value).
                drop(_frame_scope);
                drop(_tree_guard);

                // Drain render-thread wall-time samples and fold them into the
                // timing tree's `gpu_ms` column. This intentionally replaces
                // the old GPU-timestamp path so the tree now reports the full
                // render-thread cost (including submit/sync/free work) that
                // actually blocks progress in practice.
                if config.show_debug_messages {
                    let render_thread_samples = crate::compute_utils::drain_render_thread_timings();
                    for (path, dur) in render_thread_samples {
                        worker_tree.add_gpu_ns(&path, dur);
                    }
                }

                let result = MeshResult {
                    pos: mesh_pos,
                    triangles: mesh_triangles,
                    uv: mesh_uv,
                    snapshot_main_rids,
                    snapshot_normal_main_rids,
                    snapshot_level_info,
                    snapshot_updated,
                    scatter_buffers,
                    scatter_instance_counts,
                    scatter_added,
                    scatter_removed,
                    timing: if config.show_debug_messages {
                        Some(worker_tree)
                    } else {
                        None
                    },
                    vertex_texture,
                    placeholder,
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
            runtime.set_force_shrink_each_dispatch(self.scatter_force_shrink_each_dispatch);
            if !self.blue_noise_bytes.is_empty() {
                runtime.set_blue_noise(self.blue_noise_bytes.clone(), noise_dim);
            }
            runtimes.push(runtime);
        }
        runtimes
    }

    /// Build the per-slot variant list. For each enabled scatter layer:
    /// - If `mesh_source` resolves to N variants (one per enabled glTF
    ///   pick), spawn N `MultiMeshInstance3D` children — one per variant.
    /// - Else if the legacy `mesh` field is set, spawn 1 MMI with an
    ///   identity baked transform.
    /// - Else push an empty inner Vec for the slot.
    ///
    /// The outer index aligns with `WorkerState::scatter_runtimes` and
    /// `MeshResult::scatter_buffers`; the inner index aligns with the
    /// `variant_id` passed to the shader.
    fn build_scatter_mmi_children(&mut self) {
        self.destroy_scatter_mmi_children();
        let enabled_layers: Vec<Gd<CesScatterLayerResource>> =
            non_null_elements(&self.scatter_layers)
                .into_iter()
                .filter(|l| l.bind().enabled)
                .collect();

        // Capture scenario once — planet is static, no need to re-query per frame.
        self.scenario_rid = self
            .base()
            .get_world_3d()
            .expect("CesCelestialRust must be in a World3D")
            .get_scenario();

        let mut new_slots: Vec<Vec<ScatterVariant>> = Vec::with_capacity(enabled_layers.len());
        let mut new_slots_rs: Vec<Vec<ScatterRsVariant>> =
            Vec::with_capacity(enabled_layers.len());
        let mut new_uses_rs: Vec<bool> = Vec::with_capacity(enabled_layers.len());
        for (i, layer_gd) in enabled_layers.iter().enumerate() {
            // Upcast to plain Resource so we can read `mesh_source` (a property
            // on the GDScript subclass that gdext doesn't see on the Rust base).
            let layer_resource: Gd<Resource> = layer_gd.clone().upcast::<Resource>();

            // mesh_source takes precedence; falls back to the legacy `mesh` field.
            let resolved = resolve_mesh_source_variants(&layer_resource);
            let variants_to_spawn: Vec<(Gd<Mesh>, Option<String>, Transform3D)> = if !resolved
                .is_empty()
            {
                resolved
                    .into_iter()
                    .map(|v| (v.mesh, v.node_name, v.baked))
                    .collect()
            } else {
                let mesh_var = layer_gd.get("mesh");
                let mesh_opt: Option<Gd<Mesh>> = if mesh_var.is_nil() {
                    None
                } else {
                    mesh_var.try_to::<Option<Gd<Mesh>>>().ok().flatten()
                };
                match mesh_opt {
                    Some(mesh) => vec![(mesh, None, Transform3D::IDENTITY)],
                    None => Vec::new(),
                }
            };

            // Per-layer routing flag — exposed as `use_rendering_server` on
            // the GDScript layer. Read once per layer so the empty-variants
            // branch records the same flag value as the populated branch.
            let uses_rs = read_use_rs(layer_gd.get("use_rendering_server"));

            if variants_to_spawn.is_empty() {
                new_slots.push(Vec::new());
                new_slots_rs.push(Vec::new());
                new_uses_rs.push(uses_rs);
                continue;
            }

            // Read the material once per layer — all variants share it.
            let material_var = layer_gd.get("material");
            let material_opt: Option<Gd<Material>> = if material_var.is_nil() {
                None
            } else {
                material_var
                    .try_to::<Option<Gd<Material>>>()
                    .ok()
                    .flatten()
            };

            // Phase 3: per-layer routing via `use_rendering_server` export.
            // RS path = direct RenderingServer instances (per-instance
            // frustum culling / LOD). MMI path = batched MultiMeshInstance3D.
            if uses_rs {
                let mut rs_variants: Vec<ScatterRsVariant> =
                    Vec::with_capacity(variants_to_spawn.len());
                for (mesh, _node_name, baked) in variants_to_spawn.into_iter() {
                    let material_rid = material_opt.as_ref().map(|m| m.get_rid());
                    rs_variants.push(ScatterRsVariant {
                        mesh_rid: mesh.get_rid(),
                        instance_map: std::collections::HashMap::new(),
                        baked_transform: baked,
                        material_rid,
                    });
                }
                new_slots.push(Vec::new());
                new_slots_rs.push(rs_variants);
                new_uses_rs.push(true);
                continue;
            }

            let mut variants: Vec<ScatterVariant> = Vec::with_capacity(variants_to_spawn.len());
            for (v_idx, (mesh, node_name, baked)) in variants_to_spawn.into_iter().enumerate() {
                let mut mmi = MultiMeshInstance3D::new_alloc();
                let label = match &node_name {
                    Some(name) => format!("ScatterLayer{}_Variant{}_{}", i, v_idx, name),
                    None => format!("ScatterLayer{}_Variant{}", i, v_idx),
                };
                mmi.set_name(&label);

                let mut mm = MultiMesh::new_gd();
                mm.set_transform_format(TransformFormat::TRANSFORM_3D);
                mm.set_mesh(&mesh);
                mm.set_instance_count(0);
                mmi.set_multimesh(&mm);

                if let Some(ref material) = material_opt {
                    mmi.set_material_override(material);
                }

                self.base_mut().add_child(&mmi);
                variants.push(ScatterVariant {
                    mmi,
                    baked_transform: baked,
                });
            }
            new_slots.push(variants);
            new_slots_rs.push(Vec::new());
            new_uses_rs.push(false);
        }
        self.scatter_slots = new_slots;
        self.scatter_slots_rs = new_slots_rs;
        self.scatter_slot_uses_rs = new_uses_rs;
    }

    /// Free all existing scatter MMI3D children. Safe to call multiple times.
    fn destroy_scatter_mmi_children(&mut self) {
        self.scatter_slot_uses_rs.clear();
        let slots = std::mem::take(&mut self.scatter_slots);
        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            for variants in slots.into_iter() {
                for variant in variants.into_iter() {
                    let ScatterVariant { mut mmi, .. } = variant;
                    mmi.queue_free();
                }
            }
        }));

        // Free all RenderingServer-path instance RIDs. mesh_rid /
        // material_rid are owned by the source Gd<Mesh> / Gd<Material>
        // resources, so we do NOT free those here.
        let rs_slots = std::mem::take(&mut self.scatter_slots_rs);
        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let mut rs = RenderingServer::singleton();
            for variants in rs_slots.into_iter() {
                for variant in variants.into_iter() {
                    for rid in variant.instance_map.into_values() {
                        rs.free_rid(rid);
                    }
                }
            }
        }));
    }

    /// Returns the per-slot variant count parallel to `scatter_slots` /
    /// `scatter_slots_rs`. Used to drive the N-dispatch loop on the worker
    /// thread and the per-variant readback. A slot with zero variants
    /// reports `0`. Each slot lives in exactly one of the two collections.
    fn scatter_variant_counts(&self) -> Vec<u32> {
        self.scatter_slots
            .iter()
            .zip(self.scatter_slots_rs.iter())
            .map(|(mm, rs)| (mm.len() + rs.len()) as u32)
            .collect()
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

    /// Test-only: schedule a binary dump of the next post-algo visible mesh
    /// state to `path`. The dump is performed by the worker at the end of the
    /// next frame. Caller (typically a GDScript test driver) should poll for
    /// `path` to exist before reading. See `rust/src/algo/dump_format.rs` for
    /// the on-disk layout.
    #[func]
    fn dump_test_state(&mut self, path: GString) {
        let path_buf = PathBuf::from(path.to_string());
        if let Ok(mut slot) = self.pending_dump_path.lock() {
            *slot = Some(path_buf);
        }
        // Force the next _process tick to dispatch a worker frame so the
        // worker actually picks up the pending dump request. Without this the
        // worker idles until something else triggers a re-run.
        self.values_updated = true;
    }
}

#[cfg(test)]
mod tests {
    use crate::algo::run_algo::RunAlgoConfig;
    use godot::prelude::Rid;

    #[test]
    fn test_default_config() {
        let config = RunAlgoConfig {
            subdivisions: 3,
            radius: 1.0,
            triangle_screen_size: 0.1,
            precise_normals: true,
            low_poly_look: false,
            show_debug_messages: false,
            show_debug_lod_histogram: false,
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
    fn test_apply_baked_identity_is_noop() {
        use godot::prelude::Transform3D;
        let original: Vec<f32> = vec![
            1.0, 0.0, 0.0, 5.0, 0.0, 1.0, 0.0, 6.0, 0.0, 0.0, 1.0, 7.0, 0.5, 0.5, 0.5, 2.5, -0.5,
            0.5, -0.5, 3.5, 0.0, 0.0, 1.0, 4.5,
        ];
        let mut buf = original.clone();
        super::apply_baked_transform_in_place(&mut buf, Transform3D::IDENTITY);
        // Allow tiny floating-point drift from the unpack/repack round trip.
        for (i, (got, want)) in buf.iter().zip(original.iter()).enumerate() {
            assert!(
                (got - want).abs() < 1e-6,
                "float {i}: got {got}, want {want}"
            );
        }
    }

    #[test]
    fn test_apply_baked_composes_correctly() {
        use godot::prelude::{Basis, Transform3D, Vector3};

        // scatter_t = identity rotation, origin (10, 20, 30)
        let scatter_t = Transform3D::new(Basis::IDENTITY, Vector3::new(10.0, 20.0, 30.0));
        // baked = uniform scale 2x, origin (1, 2, 3)
        let baked = Transform3D::new(
            Basis::from_scale(Vector3::new(2.0, 2.0, 2.0)),
            Vector3::new(1.0, 2.0, 3.0),
        );

        let mut buf: Vec<f32> = vec![0.0; 12];
        super::write_transform3d_floats12(&mut buf, scatter_t);
        super::apply_baked_transform_in_place(&mut buf, baked);

        let composed = super::transform3d_from_floats12(&buf);
        let expected = scatter_t * baked;

        for i in 0..3 {
            assert!(
                (composed.basis.rows[i] - expected.basis.rows[i]).length() < 1e-5,
                "basis row {i}: got {:?}, want {:?}",
                composed.basis.rows[i],
                expected.basis.rows[i],
            );
        }
        assert!(
            (composed.origin - expected.origin).length() < 1e-5,
            "origin: got {:?}, want {:?}",
            composed.origin,
            expected.origin,
        );
        // Sanity: a point at origin transformed by (scatter * baked) should be
        // scatter.origin + scatter.basis * baked.origin (= 10+1, 20+2, 30+3).
        assert!(
            (composed.origin - Vector3::new(11.0, 22.0, 33.0)).length() < 1e-5,
            "expected origin (11, 22, 33), got {:?}",
            composed.origin,
        );
    }

    #[test]
    fn test_apply_baked_empty_buffer_is_noop() {
        use godot::prelude::Transform3D;
        let mut buf: Vec<f32> = Vec::new();
        super::apply_baked_transform_in_place(&mut buf, Transform3D::IDENTITY);
        assert!(buf.is_empty());
    }

    #[test]
    fn test_transform3d_round_trip_through_floats12() {
        use godot::prelude::{Basis, Transform3D, Vector3};
        let original = Transform3D::new(
            Basis::from_rows(
                Vector3::new(0.1, 0.2, 0.3),
                Vector3::new(0.4, 0.5, 0.6),
                Vector3::new(0.7, 0.8, 0.9),
            ),
            Vector3::new(11.0, 22.0, 33.0),
        );
        let mut buf: Vec<f32> = vec![0.0; 12];
        super::write_transform3d_floats12(&mut buf, original);
        let restored = super::transform3d_from_floats12(&buf);

        for i in 0..3 {
            assert!(
                (restored.basis.rows[i] - original.basis.rows[i]).length() < 1e-6,
                "basis row {i}: got {:?}, want {:?}",
                restored.basis.rows[i],
                original.basis.rows[i],
            );
        }
        assert!(
            (restored.origin - original.origin).length() < 1e-6,
            "origin: got {:?}, want {:?}",
            restored.origin,
            original.origin,
        );
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
    fn vertex_texture_material_contract_requires_feature_flag_and_texture() {
        assert!(super::should_enable_vertex_position_texture_material(
            true, true
        ));
        assert!(!super::should_enable_vertex_position_texture_material(
            true, false
        ));
        assert!(!super::should_enable_vertex_position_texture_material(
            false, true
        ));
        assert!(!super::should_enable_vertex_position_texture_material(
            false, false
        ));
    }

    #[test]
    fn disabled_vertex_texture_binding_defaults_to_safe_material_state() {
        let binding = super::VertexTextureBinding::disabled();
        assert!(!binding.enabled);
        assert_eq!(binding.main_rid, Rid::Invalid);
        assert_eq!(binding.extent.width, 1);
        assert_eq!(binding.extent.height, 1);
        assert!(!super::should_enable_vertex_position_texture_material(
            true,
            binding.main_rid.is_valid(),
        ));
    }

    #[test]
    fn scatter_child_count_matches_enabled_layers() {
        let enabled_flags = [true, false, true, true, false];
        let enabled_count = super::count_enabled_flags(&enabled_flags);
        // One variant per enabled layer (legacy single-mesh case).
        let variant_counts: Vec<u32> = vec![1; enabled_count];
        let (scatter_buffers, scatter_instance_counts) =
            super::zeroed_scatter_results(&variant_counts);

        assert_eq!(enabled_count, 3);
        assert_eq!(scatter_buffers.len(), enabled_count);
        assert_eq!(scatter_instance_counts.len(), enabled_count);
        // Each slot has exactly one variant slot, all zeroed.
        assert!(scatter_buffers
            .iter()
            .all(|slot| slot.len() == 1 && slot[0].is_empty()));
        assert!(scatter_instance_counts
            .iter()
            .all(|slot| slot == &vec![0u32]));
    }

    #[test]
    fn zeroed_scatter_results_handles_multi_variant_slots() {
        // Three slots: 2 variants, 1 variant, 0 variants (disabled / no mesh).
        let counts = [2u32, 1, 0];
        let (buffers, instance_counts) = super::zeroed_scatter_results(&counts);

        assert_eq!(buffers.len(), 3);
        assert_eq!(buffers[0].len(), 2);
        assert_eq!(buffers[1].len(), 1);
        assert_eq!(buffers[2].len(), 0);
        assert!(buffers[0].iter().all(|v| v.is_empty()));
        assert!(buffers[1].iter().all(|v| v.is_empty()));

        assert_eq!(instance_counts.len(), 3);
        assert_eq!(instance_counts[0], vec![0u32, 0u32]);
        assert_eq!(instance_counts[1], vec![0u32]);
        assert!(instance_counts[2].is_empty());
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
    fn transform3d_iter_from_floats12_yields_one_per_12_floats() {
        let buf: Vec<f32> = vec![
            1.0, 0.0, 0.0, 5.0, 0.0, 1.0, 0.0, 6.0, 0.0, 0.0, 1.0, 7.0, 0.5, 0.5, 0.5, 2.5, -0.5,
            0.5, -0.5, 3.5, 0.0, 0.0, 1.0, 4.5,
        ];
        let got: Vec<_> = super::transform3d_iter_from_floats12(&buf).collect();
        assert_eq!(got.len(), 2);
        assert_eq!(got[0], super::transform3d_from_floats12(&buf[0..12]));
        assert_eq!(got[1], super::transform3d_from_floats12(&buf[12..24]));
    }

    #[test]
    fn transform3d_iter_from_floats12_handles_empty_buffer() {
        let buf: Vec<f32> = Vec::new();
        let got: Vec<_> = super::transform3d_iter_from_floats12(&buf).collect();
        assert!(got.is_empty());
    }

    #[test]
    fn compose_scatter_transform_order_matches_current_baking_convention() {
        use godot::prelude::{Basis, Transform3D, Vector3};
        let planet = Transform3D::new(Basis::IDENTITY, Vector3::new(100.0, 0.0, 0.0));
        let per_instance = Transform3D::new(Basis::IDENTITY, Vector3::new(0.0, 20.0, 0.0));
        let baked = Transform3D::new(Basis::IDENTITY, Vector3::new(0.0, 0.0, 3.0));
        let composed = super::compose_scatter_transform(planet, baked, per_instance);
        // planet * per_instance * baked applied to origin = (100, 20, 3).
        assert!((composed.origin - Vector3::new(100.0, 20.0, 3.0)).length() < 1e-5);
    }

    #[test]
    fn resize_routing_flags_defaults_to_false() {
        let flags = super::resize_routing_flags(4);
        assert_eq!(flags.len(), 4);
        assert!(flags.iter().all(|&f| !f));
        assert!(super::resize_routing_flags(0).is_empty());
    }

    #[test]
    fn rid_pool_grows_to_target_len_by_calling_alloc() {
        let mut pool: Vec<Rid> = Vec::new();
        let mut alloc_calls = 0u32;
        let mut free_calls = 0u32;
        let mut next_id = 1u64;
        super::rid_pool_sync_size(
            &mut pool,
            3,
            || {
                alloc_calls += 1;
                let r = Rid::new(next_id);
                next_id += 1;
                r
            },
            |_| free_calls += 1,
        );
        assert_eq!(alloc_calls, 3);
        assert_eq!(free_calls, 0);
        assert_eq!(pool.len(), 3);
    }

    #[test]
    fn rid_pool_shrinks_to_target_len_by_calling_free() {
        let mut pool: Vec<Rid> =
            (1..=5).map(Rid::new).collect();
        let mut alloc_calls = 0u32;
        let mut free_calls = 0u32;
        super::rid_pool_sync_size(
            &mut pool,
            2,
            || {
                alloc_calls += 1;
                Rid::Invalid
            },
            |_| free_calls += 1,
        );
        assert_eq!(alloc_calls, 0);
        assert_eq!(free_calls, 3);
        assert_eq!(pool.len(), 2);
    }

    #[test]
    fn rid_pool_target_len_zero_releases_all() {
        let mut pool: Vec<Rid> = (1..=4).map(Rid::new).collect();
        let mut free_calls = 0u32;
        super::rid_pool_sync_size(&mut pool, 0, || Rid::Invalid, |_| free_calls += 1);
        assert_eq!(free_calls, 4);
        assert!(pool.is_empty());
    }

    #[test]
    fn rid_pool_idempotent_when_already_at_target() {
        let mut pool: Vec<Rid> = (1..=3).map(Rid::new).collect();
        let mut alloc_calls = 0u32;
        let mut free_calls = 0u32;
        super::rid_pool_sync_size(
            &mut pool,
            3,
            || {
                alloc_calls += 1;
                Rid::Invalid
            },
            |_| free_calls += 1,
        );
        assert_eq!(alloc_calls, 0);
        assert_eq!(free_calls, 0);
        assert_eq!(pool.len(), 3);
    }

    #[test]
    fn apply_rs_transforms_zips_rids_and_buffer_into_set_calls() {
        use godot::prelude::{Basis, Transform3D, Vector3};
        let rids: Vec<Rid> = (1..=3).map(Rid::new).collect();
        // 3 transforms: origins (5,6,7), (2.5,3.5,4.5), (0,0,0)
        let buf: Vec<f32> = vec![
            1.0, 0.0, 0.0, 5.0, 0.0, 1.0, 0.0, 6.0, 0.0, 0.0, 1.0, 7.0, // t0
            1.0, 0.0, 0.0, 2.5, 0.0, 1.0, 0.0, 3.5, 0.0, 0.0, 1.0, 4.5, // t1
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, // t2
        ];
        let planet = Transform3D::new(Basis::IDENTITY, Vector3::new(100.0, 0.0, 0.0));
        let baked = Transform3D::new(Basis::IDENTITY, Vector3::new(0.0, 0.0, 1.0));
        let mut captured: Vec<(Rid, Transform3D)> = Vec::new();
        super::apply_rs_transforms(&rids, &buf, planet, baked, |rid, t| {
            captured.push((rid, t));
        });
        assert_eq!(captured.len(), 3);
        for (i, per_inst) in super::transform3d_iter_from_floats12(&buf).enumerate() {
            let expected = super::compose_scatter_transform(planet, baked, per_inst);
            assert_eq!(captured[i].0, rids[i]);
            assert!((captured[i].1.origin - expected.origin).length() < 1e-5);
        }
    }

    // `Variant` construction requires a live Godot runtime, so these tests
    // exercise the pure core (`read_use_rs_from_opt`) which receives the
    // `try_to::<bool>().ok()` result instead of the Variant itself. The
    // mapping is: bool→Some(bool), nil/wrong-type→None.

    fn id_floats(marker: f32) -> [f32; 12] {
        let mut t = [0.0f32; 12];
        // identity basis (X=col0, Y=col1, Z=col2 in row-major mat3x4)
        t[0] = 1.0;
        t[5] = 1.0;
        t[10] = 1.0;
        // origin x = marker so we can recognise which add went where.
        t[3] = marker;
        t
    }

    fn alloc_counter() -> impl FnMut() -> Rid {
        let mut next = 1u64;
        move || {
            let r = Rid::new(next);
            next += 1;
            r
        }
    }

    #[test]
    fn apply_diff_adds_inserts_into_map_and_calls_create_set() {
        use godot::prelude::Transform3D;
        let mut map: std::collections::HashMap<u64, Rid> = std::collections::HashMap::new();
        let added = vec![(10u64, id_floats(1.0)), (20u64, id_floats(2.0))];
        let mut sets: Vec<(Rid, Transform3D)> = Vec::new();
        super::apply_scatter_diff_with(
            &mut map,
            &added,
            &[],
            Transform3D::IDENTITY,
            Transform3D::IDENTITY,
            alloc_counter(),
            |rid, t| sets.push((rid, t)),
            |_| panic!("should not free"),
        );
        assert_eq!(map.len(), 2);
        assert_eq!(sets.len(), 2);
        assert!(map.contains_key(&10));
        assert!(map.contains_key(&20));
    }

    #[test]
    fn apply_diff_removes_drop_from_map_and_calls_free() {
        use godot::prelude::Transform3D;
        let mut map: std::collections::HashMap<u64, Rid> = std::collections::HashMap::new();
        map.insert(5, Rid::new(100));
        map.insert(6, Rid::new(101));
        let mut frees: Vec<Rid> = Vec::new();
        super::apply_scatter_diff_with(
            &mut map,
            &[],
            &[5u64, 6],
            Transform3D::IDENTITY,
            Transform3D::IDENTITY,
            || panic!("should not create"),
            |_, _| panic!("should not set"),
            |rid| frees.push(rid),
        );
        assert!(map.is_empty());
        assert_eq!(frees.len(), 2);
    }

    #[test]
    fn apply_diff_mixed_add_remove_consistent_map_state() {
        use godot::prelude::Transform3D;
        let mut map: std::collections::HashMap<u64, Rid> = std::collections::HashMap::new();
        map.insert(1, Rid::new(100));
        map.insert(2, Rid::new(101));
        let added = vec![(3u64, id_floats(3.0))];
        let removed = vec![1u64];
        let mut sets: Vec<Rid> = Vec::new();
        let mut frees: Vec<Rid> = Vec::new();
        super::apply_scatter_diff_with(
            &mut map,
            &added,
            &removed,
            Transform3D::IDENTITY,
            Transform3D::IDENTITY,
            alloc_counter(),
            |rid, _| sets.push(rid),
            |rid| frees.push(rid),
        );
        assert_eq!(map.len(), 2);
        assert!(map.contains_key(&2));
        assert!(map.contains_key(&3));
        assert!(!map.contains_key(&1));
        assert_eq!(frees, vec![Rid::new(100)]);
        assert_eq!(sets.len(), 1);
    }

    #[test]
    fn apply_diff_uses_correct_composed_transform() {
        use godot::prelude::{Basis, Transform3D, Vector3};
        let mut map: std::collections::HashMap<u64, Rid> = std::collections::HashMap::new();
        let per_inst_floats = id_floats(7.0);
        let added = vec![(42u64, per_inst_floats)];
        let planet = Transform3D::new(Basis::IDENTITY, Vector3::new(100.0, 0.0, 0.0));
        let baked = Transform3D::new(Basis::IDENTITY, Vector3::new(0.0, 0.0, 1.0));
        let mut captured: Vec<(Rid, Transform3D)> = Vec::new();
        super::apply_scatter_diff_with(
            &mut map,
            &added,
            &[],
            planet,
            baked,
            alloc_counter(),
            |rid, t| captured.push((rid, t)),
            |_| {},
        );
        let per_inst = super::transform3d_from_floats12(&per_inst_floats);
        let expected = super::compose_scatter_transform(planet, baked, per_inst);
        assert_eq!(captured.len(), 1);
        assert!((captured[0].1.origin - expected.origin).length() < 1e-5);
    }

    #[test]
    fn apply_diff_empty_diff_is_noop() {
        use godot::prelude::Transform3D;
        let mut map: std::collections::HashMap<u64, Rid> = std::collections::HashMap::new();
        map.insert(7, Rid::new(70));
        super::apply_scatter_diff_with(
            &mut map,
            &[],
            &[],
            Transform3D::IDENTITY,
            Transform3D::IDENTITY,
            || panic!("no create"),
            |_, _| panic!("no set"),
            |_| panic!("no free"),
        );
        assert_eq!(map.len(), 1);
        assert_eq!(map.get(&7), Some(&Rid::new(70)));
    }

    #[test]
    fn read_use_rs_returns_true_for_bool_true_variant() {
        assert!(super::read_use_rs_from_opt(Some(true)));
    }

    #[test]
    fn read_use_rs_returns_false_for_bool_false_variant() {
        assert!(!super::read_use_rs_from_opt(Some(false)));
    }

    #[test]
    fn read_use_rs_returns_false_for_nil_variant() {
        assert!(!super::read_use_rs_from_opt(None));
    }

    #[test]
    fn read_use_rs_returns_false_for_non_bool_variant() {
        // `Variant::from(42i64).try_to::<bool>()` would yield Err → None.
        assert!(!super::read_use_rs_from_opt(None));
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
