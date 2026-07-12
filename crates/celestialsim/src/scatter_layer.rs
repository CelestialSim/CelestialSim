//! `CesScatterLayer` — the editor-facing scatter layer resource (CEL-73).
//!
//! One layer = one mesh scattered over the planet. The day-to-day sliders —
//! `density`, `min_height`/`max_height` — are LIVE: editing them re-runs only
//! the scatter-compact dispatch (no re-realize, no bake, no cache invalidation).
//!
//! `lod_level` sets how fine the **stable world lattice** the candidates live on
//! is (higher = denser + shorter reach; fine ≈ 12 for grass, coarse ≈ 6 for
//! trees), with `instances_per_cell` (K) the count per cell. Editing `lod_level`
//! / `scale` / `seed` re-places resident chunks (GPU-only, via
//! `ChunkCache::invalidate_all`); changing `max_instances` or the layer list
//! rebuilds the job. Every setter `emit_changed()`s so the planet node can diff
//! what actually changed.

use godot::classes::{Mesh, Resource};
use godot::prelude::*;

/// A scatter layer: mesh + density/height sliders + lattice (LOD) tuning.
#[derive(GodotClass)]
#[class(base = Resource, tool, init)]
pub struct CesScatterLayer {
    base: Base<Resource>,

    /// Display name (e.g. "Grass", "Oak trees"). Mirrored into
    /// `resource_name`, so the inspector's layer array shows it per element —
    /// name your layers instead of hunting through "CesScatterLayer" entries.
    #[var(get = get_layer_name, set = set_layer_name)]
    #[export]
    pub layer_name: GString,
    /// LIVE on/off toggle: a disabled layer draws nothing but keeps its cached
    /// GPU placement, so re-enabling is instant (compact-only update).
    #[var(get = get_enabled, set = set_enabled)]
    #[export]
    #[init(val = true)]
    pub enabled: bool,
    /// Fraction of lattice candidates drawn (LIVE — compact-only update).
    #[var(get = get_density, set = set_density)]
    #[export(range = (0.0, 1.0, 0.005))]
    #[init(val = 0.5)]
    pub density: f32,
    /// Instance mesh. Empty ⇒ the layer is INACTIVE (renders nothing). For
    /// grass, assign `res://addons/celestialsim/grass_blade.tres`.
    #[var(get = get_mesh, set = set_mesh)]
    #[export]
    pub mesh: Option<Gd<Mesh>>,
    /// Base scale applied to every instance (on top of a ±20% per-instance
    /// jitter), so you can size a mesh without re-authoring it. 1.0 = the
    /// mesh's native size.
    #[var(get = get_scale, set = set_scale)]
    #[export(range = (0.01, 100.0, 0.01))]
    #[init(val = 1.0)]
    pub scale: f32,
    /// LIVE lower bound on normalized terrain height (0..1) where this layer may
    /// appear. Default 0.45 = the default sea level, so nothing scatters
    /// underwater; drop to 0 to allow it. (Terrain height is normalized: ~0.45
    /// is the shoreline, 1.0 the highest peaks.)
    #[var(get = get_min_height, set = set_min_height)]
    #[export(range = (0.0, 1.0, 0.01))]
    #[init(val = 0.45)]
    pub min_height: f32,
    /// LIVE upper bound on normalized terrain height (0..1). Default 1.0 = no
    /// upper limit; lower it to keep this layer off the mountain tops.
    #[var(get = get_max_height, set = set_max_height)]
    #[export(range = (0.0, 1.0, 0.01))]
    #[init(val = 1.0)]
    pub max_height: f32,
    /// LOD level the candidates are keyed to — the fineness of the stable world
    /// lattice. Higher = denser and shorter-range; the layer only appears on
    /// terrain chunks at depth `>= lod_level - 3` (per-slot capacity bound), so
    /// this is the single knob for both density and reach. Instances never move
    /// when chunks split/merge.
    #[var(get = get_lod_level, set = set_lod_level)]
    #[export(range = (0.0, 20.0, 1.0))]
    #[init(val = 9)]
    pub lod_level: i64,
    /// Candidates per lattice cell (K). Per-slot pool capacity = `K * 64`.
    #[var(get = get_instances_per_cell, set = set_instances_per_cell)]
    #[export(range = (1.0, 64.0, 1.0))]
    #[init(val = 4)]
    pub instances_per_cell: i64,
    /// MultiMesh pool cap: the compact pass clamps the drawn count to this.
    #[var(get = get_max_instances, set = set_max_instances)]
    #[export(range = (64.0, 4000000.0, 64.0))]
    #[init(val = 100000)]
    pub max_instances: i64,
    /// Placement seed.
    #[var(get = get_seed, set = set_seed)]
    #[export]
    #[init(val = 0)]
    pub seed: i64,
}

#[godot_api]
impl CesScatterLayer {
    /// Build the procedural grass blade mesh (the source of the committed
    /// `addons/celestialsim/grass_blade.tres`). Callable from GDScript:
    /// `CesScatterLayer.make_grass_blade_mesh()`.
    #[func]
    fn make_grass_blade_mesh() -> Gd<godot::classes::ArrayMesh> {
        crate::scatter_mesh::grass_blade_mesh()
    }

    #[func]
    fn get_layer_name(&self) -> GString {
        self.layer_name.clone()
    }
    #[func]
    fn set_layer_name(&mut self, v: GString) {
        if self.layer_name != v {
            self.layer_name = v.clone();
            // resource_name is what the inspector shows on array elements.
            self.base_mut().set_name(&v);
            self.base_mut().emit_changed();
        }
    }

    #[func]
    fn get_enabled(&self) -> bool {
        self.enabled
    }
    #[func]
    fn set_enabled(&mut self, v: bool) {
        if self.enabled != v {
            self.enabled = v;
            self.base_mut().emit_changed();
        }
    }

    #[func]
    fn get_density(&self) -> f32 {
        self.density
    }
    #[func]
    fn set_density(&mut self, v: f32) {
        if self.density != v {
            self.density = v;
            self.base_mut().emit_changed();
        }
    }

    #[func]
    fn get_scale(&self) -> f32 {
        self.scale
    }
    #[func]
    fn set_scale(&mut self, v: f32) {
        if self.scale != v {
            self.scale = v;
            self.base_mut().emit_changed();
        }
    }

    #[func]
    fn get_min_height(&self) -> f32 {
        self.min_height
    }
    #[func]
    fn set_min_height(&mut self, v: f32) {
        if self.min_height != v {
            self.min_height = v;
            self.base_mut().emit_changed();
        }
    }

    #[func]
    fn get_max_height(&self) -> f32 {
        self.max_height
    }
    #[func]
    fn set_max_height(&mut self, v: f32) {
        if self.max_height != v {
            self.max_height = v;
            self.base_mut().emit_changed();
        }
    }

    #[func]
    fn get_mesh(&self) -> Option<Gd<Mesh>> {
        self.mesh.clone()
    }
    #[func]
    fn set_mesh(&mut self, mesh: Option<Gd<Mesh>>) {
        if self.mesh != mesh {
            self.mesh = mesh;
            self.base_mut().emit_changed();
        }
    }

    #[func]
    fn get_lod_level(&self) -> i64 {
        self.lod_level
    }
    #[func]
    fn set_lod_level(&mut self, v: i64) {
        if self.lod_level != v {
            self.lod_level = v;
            self.base_mut().emit_changed();
        }
    }

    #[func]
    fn get_instances_per_cell(&self) -> i64 {
        self.instances_per_cell
    }
    #[func]
    fn set_instances_per_cell(&mut self, v: i64) {
        if self.instances_per_cell != v {
            self.instances_per_cell = v;
            self.base_mut().emit_changed();
        }
    }

    #[func]
    fn get_max_instances(&self) -> i64 {
        self.max_instances
    }
    #[func]
    fn set_max_instances(&mut self, v: i64) {
        if self.max_instances != v {
            self.max_instances = v;
            self.base_mut().emit_changed();
        }
    }

    #[func]
    fn get_seed(&self) -> i64 {
        self.seed
    }
    #[func]
    fn set_seed(&mut self, v: i64) {
        if self.seed != v {
            self.seed = v;
            self.base_mut().emit_changed();
        }
    }
}
