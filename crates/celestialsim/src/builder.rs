//! `CesBuilder` — the terrain-builder resource.
//!
//! A planet has ONE `builder` (its terrain source); when it is unset the planet
//! renders a plain white sphere. The base `CesBuilder` is the SINGLE builder
//! class: whether it is a built-in noise example or YOUR custom terrain is
//! decided by two fields — `device` (GPU/CPU) and a hidden `builtin_shader`
//! selector (`None` = custom, `Terrain` = the built-in noise). The four paths
//! ([`BuilderRoute`]) are:
//!
//! * `builtin_shader == Terrain`, `device == GPU` — built-in GPU noise (the rich
//!   inline shader, driven by the noise knobs). The auto-added default, shipped
//!   as the `CesGPUNoiseExample` GDScript (carries the knobs as its own
//!   `@export`s).
//! * `builtin_shader == Terrain`, `device == CPU` — the same noise baked on CPU
//!   worker threads ([`crate::noise_provider::NoiseProvider`]); shipped as
//!   `CesCPUNoiseExample`.
//! * `builtin_shader == None`, `device == GPU` — YOUR terrain: point
//!   `shader_file` at a `.glsl` defining `terrain_height` / `terrain_color`
//!   (compiled at runtime). May add `@export var name: float` knobs (surfaced to
//!   the GLSL as `#define NAME`).
//! * `builtin_shader == None`, `device == CPU` — YOUR terrain in GDScript:
//!   `extends CesBuilder`, define `height` / `color` / `normal`, and set the
//!   device to `CPU`. If it instead defines `_bake_requested`, the bake is
//!   ASYNCHRONOUS (CEL-86) — see [`BuilderRoute::CpuCustomAsync`].
//!
//! The planet routes on (`device`, `builtin_shader`) plus, for the CPU-custom
//! pair, whether the script defines `_bake_requested` — see [`route_of`].
//! The noise knobs live on the example GDScripts (not this base); a custom
//! builder simply doesn't declare them.

use std::sync::Arc;

use godot::classes::Resource;
use godot::prelude::*;
use godot::register::info::{PropertyInfo, PropertyUsageFlags};

use crate::async_bake::{self, RawSubmission, SubmitQueue};
use crate::descriptors::{HeightGpu, TextureGpu};
use crate::noise_provider::NoiseParams;

/// The GDScript method whose presence selects the async CPU bake route.
pub const BAKE_REQUESTED: &str = "_bake_requested";
/// Optional GDScript method gating whether the baked surface is shown yet.
pub const BASE_READY: &str = "_base_ready";

/// Which device runs a builder — an exported dropdown on [`CesBuilder`].
/// Combined with [`BuiltinShader`] it selects the [`BuilderRoute`].
#[allow(clippy::upper_case_acronyms)]
#[derive(GodotConvert, Var, Export, Default, Clone, Copy, PartialEq, Eq, Debug)]
#[godot(via = i64)]
pub enum BuilderDevice {
    /// GPU: either the built-in inline noise shader, or a custom `.glsl` (see
    /// [`CesBuilder::shader_file`]) compiled at runtime. The default.
    #[default]
    GPU = 0,
    /// CPU: either the built-in noise baked on worker threads, or custom terrain
    /// in GDScript (`extends CesBuilder`, define `height`/`color`/`normal`).
    CPU = 1,
}

/// Which BUILT-IN library shader a builder runs — a hidden (STORAGE-only)
/// selector on [`CesBuilder`]. `None` = a custom builder (user `.glsl` or
/// GDScript); `Terrain` = the built-in noise example. Combined with
/// [`BuilderDevice`] it selects the [`BuilderRoute`].
#[derive(GodotConvert, Var, Export, Default, Clone, Copy, PartialEq, Eq, Debug)]
#[godot(via = i64)]
pub enum BuiltinShader {
    /// No built-in shader — a custom builder (GPU `.glsl` or CPU GDScript).
    #[default]
    None = 0,
    /// The built-in terrain noise example (GPU inline shader / CPU provider).
    Terrain = 1,
}

/// The terrain paths the planet routes between. Derived from a builder's
/// [`CesBuilder::device`] and [`CesBuilder::builtin_shader`], plus — for the
/// CPU-custom pair — whether the GDScript defines [`BAKE_REQUESTED`]. NOT an
/// exported field.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum BuilderRoute {
    /// Built-in GPU noise (rich inline shader, driven by knobs).
    GpuNoise,
    /// Built-in noise baked on CPU worker threads.
    CpuNoise,
    /// A custom builder on [`BuilderDevice::GPU`] — a `.glsl` surface.
    GpuCustom,
    /// A custom builder on [`BuilderDevice::CPU`] — GDScript `height`/`color`/
    /// `normal`, baked synchronously on the main thread.
    CpuCustom,
    /// A custom builder on [`BuilderDevice::CPU`] defining `_bake_requested` —
    /// the planet hands it chunks and it submits surfaces back whenever they
    /// are ready (threads, network, disk). See [`crate::async_bake`].
    CpuCustomAsync,
}

/// Resolve a builder's routing path.
///
/// `(device, builtin_shader)` decides everything except which of the two CPU-
/// custom routes applies: a script that defines [`BAKE_REQUESTED`] is async.
/// Auto-detection (rather than a flag) is what lets every pre-existing
/// `height(dirs)` builder keep working with no edit.
pub fn route_of(builder: &Gd<CesBuilder>) -> BuilderRoute {
    let (builtin, device) = {
        let b = builder.bind();
        (b.builtin_shader, b.device)
    };
    match (builtin, device) {
        (BuiltinShader::Terrain, BuilderDevice::GPU) => BuilderRoute::GpuNoise,
        (BuiltinShader::Terrain, BuilderDevice::CPU) => BuilderRoute::CpuNoise,
        (BuiltinShader::None, BuilderDevice::GPU) => BuilderRoute::GpuCustom,
        (BuiltinShader::None, BuilderDevice::CPU) => {
            if builder.clone().upcast::<Object>().has_method(BAKE_REQUESTED) {
                BuilderRoute::CpuCustomAsync
            } else {
                BuilderRoute::CpuCustom
            }
        }
    }
}

/// Plain, testable noise params (the Example builders' knobs). Defaults MUST
/// reproduce [`HeightGpu::default`] / [`TextureGpu::default`] so a default
/// builder renders like the historical HQ terrain.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BuilderParams {
    pub frequency: f32,
    pub height_octaves: f32,
    pub height_amp: f32,
    pub height_gain: f32,
    pub height_lacunarity: f32,
    pub ridge_tiles: f32,
    pub ridge_octaves: f32,
    pub ridge_gain: f32,
    pub ridge_lacunarity: f32,
    pub ridge_strength: f32,
    pub height_scale: f32,
    pub fd_eps: f32,
    pub water_height: f32,
}

impl Default for BuilderParams {
    fn default() -> Self {
        let h = HeightGpu::default();
        let t = TextureGpu::default();
        Self {
            frequency: h.frequency,
            height_octaves: h.height_octaves,
            height_amp: h.height_amp,
            height_gain: h.height_gain,
            height_lacunarity: h.height_lacunarity,
            ridge_tiles: h.ridge_tiles,
            ridge_octaves: h.ridge_octaves,
            ridge_gain: h.ridge_gain,
            ridge_lacunarity: h.ridge_lacunarity,
            ridge_strength: h.ridge_strength,
            height_scale: h.height_scale,
            fd_eps: h.fd_eps,
            water_height: t.water_height,
        }
    }
}

/// Map the params onto the geometry (height) GPU struct (`enabled` always on —
/// the planet's builder SELECTION gates displacement, not this flag).
pub fn builder_height_gpu(p: &BuilderParams) -> HeightGpu {
    HeightGpu {
        frequency: p.frequency,
        height_octaves: p.height_octaves,
        height_amp: p.height_amp,
        height_gain: p.height_gain,
        height_lacunarity: p.height_lacunarity,
        ridge_tiles: p.ridge_tiles,
        ridge_octaves: p.ridge_octaves,
        ridge_gain: p.ridge_gain,
        ridge_lacunarity: p.ridge_lacunarity,
        ridge_strength: p.ridge_strength,
        height_scale: p.height_scale,
        fd_eps: p.fd_eps,
        enabled: 1.0,
    }
}

/// Map the params onto the surface (texture) GPU struct.
pub fn builder_texture_gpu(p: &BuilderParams) -> TextureGpu {
    TextureGpu { water_height: p.water_height }
}

/// A terrain builder resource. See the module docs.
#[derive(GodotClass)]
#[class(base = Resource, tool, init)]
pub struct CesBuilder {
    base: Base<Resource>,

    /// How the surface is produced. Switching this reshapes the inspector (only
    /// the relevant fields stay visible) via `on_validate_property`.
    #[var(get = get_device, set = set_device)]
    #[export]
    pub device: BuilderDevice,

    /// HIDDEN (STORAGE-only) selector for WHICH built-in library shader this
    /// builder runs. `None` = custom (user `.glsl`/GDScript); `Terrain` = the
    /// built-in noise. Set by the example GDScripts' `_init`; combined with
    /// `device` it selects the [`BuilderRoute`].
    #[var(get = get_builtin_shader, set = set_builtin_shader)]
    #[export]
    pub builtin_shader: BuiltinShader,

    /// `GPU`-device custom only: `res://` path to your terrain `.glsl`
    /// (defines `terrain_height` / `terrain_color`, optional `terrain_normal`).
    #[var(get = get_shader_file, set = set_shader_file)]
    #[export(file = "*.glsl")]
    pub shader_file: GString,

    /// **Water level** — normalized sea level (0..1). A NATIVE field (was a
    /// per-subclass GDScript `@export`) so EVERY builder — noise or custom —
    /// carries the same water level. Also drives terrain shore colouring and is
    /// surfaced to custom shaders as `CELS_WATER_HEIGHT`. `0.5` = sea at the
    /// planet radius (noise midpoint / custom `h = 0` baseline); raise to flood
    /// low land. Always visible. A `changed`-emitting setter (below) reshades the
    /// WHOLE terrain on edit — the water level controls the land/sea split, not
    /// just the sphere — so a plain field would leave the terrain stale.
    #[var(get = get_water_height, set = set_water_height)]
    #[export(range = (0.0, 1.0, 0.001))]
    #[init(val = 0.549)]
    pub water_height: f32,

    /// **Water toggle** — draw an analytic ocean at the water level. Off hides the
    /// water surface AND the appearance params below (they reappear when on).
    #[export]
    #[init(val = true)]
    pub water_enabled: bool,
    /// Deep-water body colour (far from shore).
    #[export]
    #[init(val = Color::from_rgb(0.05, 0.22, 0.42))]
    pub water_deep_color: Color,
    /// Shallow-water body colour (near shore).
    #[export]
    #[init(val = Color::from_rgb(0.20, 0.55, 0.70))]
    pub water_shallow_color: Color,
    /// Wave normal blend strength (0 = flat mirror).
    #[export(range = (0.0, 1.0, 0.01))]
    #[init(val = 0.55)]
    pub water_wave_strength: f32,
    /// Wave tiling frequency.
    #[export(range = (0.01, 1.0, 0.01))]
    #[init(val = 0.15)]
    pub water_wave_scale: f32,
    /// Wave scroll speed.
    #[export(range = (0.0, 0.5, 0.005))]
    #[init(val = 0.04)]
    pub water_wave_speed: f32,
    /// Underwater fog tint (Beer–Lambert) when the camera is below the surface.
    #[export]
    #[init(val = Color::from_rgb(0.04, 0.16, 0.28))]
    pub water_underwater_color: Color,
    /// Underwater fog density (per world unit of water column).
    #[export(range = (0.0, 0.2, 0.001))]
    #[init(val = 0.02)]
    pub water_underwater_density: f32,

    /// Async-bake hand-back channel (CEL-86). NOT a Godot property: plain shared
    /// state that `submit_chunk` pushes into from any thread and the planet
    /// drains on the main thread. Owned here, so nothing points back at the
    /// planet.
    submits: Arc<SubmitQueue>,
}

/// Water APPEARANCE exports, hidden in the inspector when `water_enabled` is off
/// (the toggle and `water_height` level stay visible). See `on_validate_property`.
const WATER_LOOK_PROPS: &[&str] = &[
    "water_deep_color",
    "water_shallow_color",
    "water_wave_strength",
    "water_wave_scale",
    "water_wave_speed",
    "water_underwater_color",
    "water_underwater_density",
];

#[godot_api]
impl IResource for CesBuilder {
    /// Reshape the inspector by (`device`, `builtin_shader`) — read as plain
    /// fields, NEVER via `self.to_gd()` (which would free a refcount-0
    /// introspection object and crash). `builtin_shader` is always hidden
    /// (STORAGE only). `shader_file` shows ONLY for the GPU-custom route. Every
    /// other property (including the example GDScripts' `@export` knobs) is left
    /// visible. Hidden fields keep `STORAGE` so values still persist.
    fn on_validate_property(&self, property: &mut PropertyInfo) {
        let name = property.property_name.to_string();
        // `device` and `builtin_shader` are set in code, never edited by hand: a
        // bare CesBuilder is always GPU (a CPU builder MUST be a subclass that
        // defines height/color and sets `device = 1` in `_init`), so exposing the
        // dropdown would only let someone pick a broken CPU-on-bare-builder combo.
        if name == "builtin_shader" || name == "device" {
            property.usage = PropertyUsageFlags::STORAGE;
            return;
        }
        if name == "shader_file" {
            let show = self.device == BuilderDevice::GPU && self.builtin_shader == BuiltinShader::None;
            if !show {
                property.usage = PropertyUsageFlags::STORAGE;
            }
            return;
        }
        // Water APPEARANCE params show only when the water toggle is on (the
        // toggle + `water_height` level stay visible so the sea can be placed /
        // enabled). Hidden fields keep STORAGE so their values persist.
        if WATER_LOOK_PROPS.contains(&name.as_str()) && !self.water_enabled {
            property.usage = PropertyUsageFlags::STORAGE;
            return;
        }
        // anything a user/example subclass adds: leave as-is.
    }
}

impl CesBuilder {
    /// Read a noise knob by GDScript-property name from the script instance,
    /// falling back to `default` when the property is absent (a bare custom
    /// builder has no knobs). Safe in an ordinary method (`to_gd()` re-acquires
    /// the live `Gd`); NEVER call from `on_validate_property`.
    fn read_knob(&self, name: &str, default: f32) -> f32 {
        self.to_gd().get(name).try_to::<f32>().unwrap_or(default)
    }

    /// Collect the noise knobs (declared as `@export`s on the example GDScripts)
    /// into the plain params struct, defaulting to the HQ terrain when absent.
    pub fn params(&self) -> BuilderParams {
        let d = BuilderParams::default();
        BuilderParams {
            frequency: self.read_knob("frequency", d.frequency),
            height_octaves: self.read_knob("octaves", d.height_octaves),
            height_amp: self.read_knob("amp", d.height_amp),
            height_gain: self.read_knob("gain", d.height_gain),
            height_lacunarity: self.read_knob("lacunarity", d.height_lacunarity),
            ridge_tiles: self.read_knob("ridge_tiles", d.ridge_tiles),
            ridge_octaves: self.read_knob("ridge_octaves", d.ridge_octaves),
            ridge_gain: self.read_knob("ridge_gain", d.ridge_gain),
            ridge_lacunarity: self.read_knob("ridge_lacunarity", d.ridge_lacunarity),
            ridge_strength: self.read_knob("ridge_strength", d.ridge_strength),
            height_scale: self.read_knob("height_scale", d.height_scale),
            fd_eps: self.read_knob("fd_eps", d.fd_eps),
            water_height: self.read_knob("water_height", d.water_height),
        }
    }

    /// Geometry (height) params for the inline example shader.
    pub fn to_height_gpu(&self) -> HeightGpu {
        builder_height_gpu(&self.params())
    }
    /// Surface (texture) params for the inline example shader.
    pub fn to_texture_gpu(&self) -> TextureGpu {
        builder_texture_gpu(&self.params())
    }
    /// CPU-noise params for [`crate::noise_provider::NoiseProvider`]
    /// (`CpuNoise`). `radius` is the planet radius (for the FD normal).
    pub fn to_noise_params(&self, radius: f32) -> NoiseParams {
        let p = self.params();
        NoiseParams {
            tiles: p.frequency,
            octaves: p.height_octaves.round().clamp(1.0, 12.0) as u32,
            gain: p.height_gain,
            lacunarity: p.height_lacunarity,
            amp: p.height_amp,
            height_scale: p.height_scale,
            water_height: p.water_height,
            ridge_tiles: p.ridge_tiles,
            ridge_octaves: p.ridge_octaves.round().clamp(1.0, 12.0) as u32,
            ridge_gain: p.ridge_gain,
            ridge_lacunarity: p.ridge_lacunarity,
            ridge_strength: p.ridge_strength,
            radius,
        }
    }

    /// The builder's normalized sea level (0..1). Read as a knob so a custom
    /// GPU builder (no `water_height` @export) falls back to the default.
    pub fn water_height(&self) -> f32 {
        self.read_knob("water_height", BuilderParams::default().water_height)
    }

    /// The builder's geometry displacement multiplier. Read as a knob so a
    /// custom GPU builder (no `height_scale` @export) falls back to the default.
    pub fn height_scale(&self) -> f32 {
        self.read_knob("height_scale", BuilderParams::default().height_scale)
    }

    /// The async-bake hand-back queue (the planet drains it each frame).
    pub fn submits(&self) -> Arc<SubmitQueue> {
        Arc::clone(&self.submits)
    }
}

// ---- CpuCustom surface functions ------------------------------------------
//
// A `CpuCustom` builder is a GDScript `extends CesBuilder` that DEFINES (not
// overrides — the base deliberately has no such methods, so there is no
// native-shadow warning) the same three functions as the GPU `.glsl`, BATCHED
// over one chunk's texel directions:
//
//   func height(dirs: PackedVector3Array) -> PackedFloat32Array
//   func color(dirs: PackedVector3Array, hs: PackedFloat32Array) -> PackedColorArray
//   func normal(dirs: PackedVector3Array, hs: PackedFloat32Array) -> PackedVector3Array  # OPTIONAL
//
// The planet calls whichever are present (`Object::has_method`); a missing
// `height` = flat, missing `color` = white, missing `normal` = finite-difference.
//
// ---- CpuCustomAsync surface functions (CEL-86) -----------------------------
//
// Instead, a builder may define `_bake_requested` — then the bake is async and
// the planet never blocks on it:
//
//   func _bake_requested(requests: Array) -> void   # main thread; MUST NOT BLOCK
//   func _base_ready() -> bool                      # OPTIONAL (default true)
//
// Each request is `{handle, corners, tile_res, depth}`. Bake however you like
// (WorkerThreadPool, HTTPRequest, a disk cache) and call `submit_chunk` when a
// surface is ready — from any thread, at any time. Submitting an already-
// resident chunk again REFINES it (a coarse tile now, a finer one when the
// download lands); submitting a chunk that has left the view is dropped, which
// is what makes cancellation a no-op rather than an API.

#[godot_api]
impl CesBuilder {
    // ---- async bake (CEL-86) ------------------------------------------------

    /// Hand a finished chunk surface back to the planet. **Callable from any
    /// thread** (it takes a mutex and returns; the planet drains next frame).
    ///
    /// * `handle` — the `handle` from the matching `_bake_requested` entry.
    /// * `heights` — `tile_res²` displacements in YOUR vertical unit, row-major
    ///   (the `height_scale` property converts them to displaced radius).
    /// * `colors` — `tile_res²` albedos, row-major.
    /// * `normals` — optional; leave empty to have the library finite-difference
    ///   the height grid for you.
    ///
    /// Idempotent: call it again for the same handle to refine that chunk.
    /// Wrong-length arrays are rejected (one error is printed, then silence);
    /// non-finite heights are sanitized to `0.0`.
    #[func]
    fn submit_chunk(
        &self,
        handle: i64,
        heights: PackedFloat32Array,
        colors: PackedColorArray,
        // Empty (the default) ⇒ no normals ⇒ finite-difference. The `&` is
        // required: packed arrays are passed to Godot by reference.
        #[opt(default = &PackedVector3Array::new())] normals: PackedVector3Array,
    ) {
        // Copy out of the PackedArrays: they are not `Send`, and the queue must
        // be readable from the main thread while a worker keeps pushing.
        self.submits.push(RawSubmission {
            handle,
            heights: heights.as_slice().to_vec(),
            colors: colors.as_slice().to_vec(),
            normals: normals.as_slice().to_vec(),
        });
    }

    /// The `tile_res²` texel-centre world directions of a chunk, row-major —
    /// the same mapping the shaders use.
    ///
    /// Not included in a bake request (786 KB per chunk at `tile_res = 256`, and
    /// a streaming builder wants a lat/lon box, not directions), so materialize
    /// them only if you need them. Pure: safe to call from a worker thread.
    #[func]
    fn chunk_dirs(&self, corners: PackedVector3Array, tile_res: i64) -> PackedVector3Array {
        let c = corners.as_slice();
        if c.len() != 3 || tile_res <= 0 {
            godot_error!("chunk_dirs: expected 3 corners and tile_res > 0");
            return PackedVector3Array::new();
        }
        let dirs = async_bake::chunk_dirs([c[0], c[1], c[2]], tile_res as u32);
        PackedVector3Array::from(&dirs[..])
    }

    // ---- property accessors (each emits `changed` so live edits reshade) ----

    #[func]
    fn get_device(&self) -> BuilderDevice {
        self.device
    }
    #[func]
    fn set_device(&mut self, v: BuilderDevice) {
        if self.device != v {
            self.device = v;
            // Reshape the inspector (show/hide fields for the new mode) and let
            // the planet re-route/rebuild.
            self.base_mut().notify_property_list_changed();
            self.base_mut().emit_changed();
        }
    }
    #[func]
    fn get_builtin_shader(&self) -> BuiltinShader {
        self.builtin_shader
    }
    #[func]
    fn set_builtin_shader(&mut self, v: BuiltinShader) {
        if self.builtin_shader != v {
            self.builtin_shader = v;
            self.base_mut().notify_property_list_changed();
            self.base_mut().emit_changed();
        }
    }
    #[func]
    fn get_shader_file(&self) -> GString {
        self.shader_file.clone()
    }
    #[func]
    fn set_shader_file(&mut self, v: GString) {
        if self.shader_file != v {
            self.shader_file = v;
            self.base_mut().emit_changed();
        }
    }
    #[func]
    fn get_water_height(&self) -> f32 {
        self.water_height
    }
    #[func]
    fn set_water_height(&mut self, v: f32) {
        if self.water_height != v {
            self.water_height = v;
            // The water level is a TERRAIN parameter (it sets the land/sea split
            // and shore colouring), not only the analytic sphere's radius. A bare
            // `#[export]` would move the sphere (read live by `update_water`) but
            // leave the baked terrain stale. Emitting `changed` runs the planet's
            // connected reshade — re-realizing + re-baking the whole planet at the
            // new level — exactly like the noise-knob poll does for script vars.
            self.base_mut().emit_changed();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::descriptors::assemble;

    #[test]
    fn default_params_map_to_hq_defaults() {
        let p = BuilderParams::default();
        assert_eq!(builder_height_gpu(&p), HeightGpu::default());
        assert_eq!(builder_texture_gpu(&p), TextureGpu::default());
    }

    #[test]
    fn default_builder_assembles_to_current_terrain() {
        let p = BuilderParams::default();
        let from_builder = assemble(&builder_height_gpu(&p), &builder_texture_gpu(&p));
        let from_defaults = assemble(&HeightGpu::default(), &TextureGpu::default());
        assert_eq!(from_builder, from_defaults);
    }

    #[test]
    fn default_device_is_gpu() {
        assert_eq!(BuilderDevice::default(), BuilderDevice::GPU);
    }

    #[test]
    fn default_builtin_shader_is_none() {
        assert_eq!(BuiltinShader::default(), BuiltinShader::None);
    }

    #[test]
    fn water_edit_only_touches_texture_gpu() {
        let mut p = BuilderParams::default();
        p.water_height = 0.9;
        assert_eq!(builder_texture_gpu(&p).water_height, 0.9);
        assert_eq!(builder_height_gpu(&p), HeightGpu::default());
    }
}
