//! Scene-side water surface: one transparent proxy sphere per planet driving the
//! analytic `water_surface.gdshader`.
//!
//! The proxy mesh is geometry-irrelevant — the shader computes the true smooth
//! sea-level sphere analytically. The proxy just needs to over-cover that
//! sphere's silhouette, so a low-segment `SphereMesh` at `water_radius * margin`
//! is plenty. Per-frame we push the planet center, the live `water_radius`
//! (recomputed from the builder's water level), the sun direction, and the look
//! uniforms (all sourced from the active `CesBuilder`).

use godot::classes::fast_noise_lite::NoiseType;
use godot::classes::{
    FastNoiseLite, MeshInstance3D, Node3D, NoiseTexture2D, Shader, ShaderMaterial, SphereMesh,
    Texture2D,
};
use godot::prelude::*;

use crate::water::proxy_radius;

const WATER_SHADER: &str = "res://addons/celestialsim/water_surface.gdshader";

/// Tunable look params pushed to the water material each frame (read from the
/// active builder's water exports; the rest use the shader's own defaults).
#[derive(Clone, Copy)]
pub struct WaterParams {
    pub deep_color: Color,
    pub shallow_color: Color,
    pub wave_strength: f32,
    pub wave_scale: f32,
    pub wave_speed: f32,
    pub underwater_color: Color,
    pub underwater_density: f32,
    /// World-space direction pointing TOWARD the sun (found from the scene's
    /// `DirectionalLight3D`). Drives the analytic diffuse + specular glint.
    pub sun_dir: Vector3,
}

pub struct WaterRuntime {
    mmi: Gd<MeshInstance3D>,
    mesh: Gd<SphereMesh>,
    material: Gd<ShaderMaterial>,
    /// Proxy radius currently baked into the SphereMesh (rebuilt only when the
    /// water radius changes meaningfully — cheap, but not worth doing per frame).
    proxy_r: f32,
}

impl WaterRuntime {
    /// A tiling normal map for the wave detail. Generated here so the water is
    /// self-sufficient (the shader's `wave_normal_*` are sampled triplanar; no
    /// scene/HUD needs to feed them). Seamless simplex bumped into a normal map.
    fn make_wave_normal_tex(seed: i32, frequency: f32) -> Gd<Texture2D> {
        let mut noise = FastNoiseLite::new_gd();
        noise.set_noise_type(NoiseType::SIMPLEX_SMOOTH);
        noise.set_frequency(frequency);
        noise.set_seed(seed);
        let mut tex = NoiseTexture2D::new_gd();
        tex.set_width(256);
        tex.set_height(256);
        tex.set_seamless(true);
        tex.set_as_normal_map(true);
        tex.set_bump_strength(4.0);
        tex.set_noise(&noise);
        tex.upcast()
    }

    /// Build the proxy sphere child under `parent` and attach the water material.
    pub fn create(parent: &mut Gd<Node3D>) -> Self {
        let shader = godot::tools::load::<Shader>(WATER_SHADER);
        let mut material = ShaderMaterial::new_gd();
        material.set_shader(&shader);
        // Two differently-seeded seamless normal maps => a livelier, non-repeating
        // swell when the shader scrolls + RNM-blends them (Lague uses two maps).
        let wave_a = Self::make_wave_normal_tex(1337, 0.015);
        let wave_b = Self::make_wave_normal_tex(9001, 0.026);
        material.set_shader_parameter("wave_normal_a", &wave_a.to_variant());
        material.set_shader_parameter("wave_normal_b", &wave_b.to_variant());

        let mut mesh = SphereMesh::new_gd();
        // Coarse: the surface is analytic, so segments only affect silhouette
        // coverage, not smoothness.
        mesh.set_radial_segments(24);
        mesh.set_rings(16);
        mesh.set_material(&material);

        let mut mmi = MeshInstance3D::new_alloc();
        mmi.set_mesh(&mesh);
        // GPU transforms aside, the proxy is a plain sphere; let it always draw
        // (its own bounds are correct) and never cast shadows.
        mmi.set_cast_shadows_setting(godot::classes::geometry_instance_3d::ShadowCastingSetting::OFF);
        parent.add_child(&mmi);

        Self { mmi, mesh, material, proxy_r: 0.0 }
    }

    /// Push per-frame state: planet world center, the live analytic water radius,
    /// the look params, and visibility. Rebuilds the proxy sphere only when the
    /// radius changed.
    pub fn update(&mut self, center: Vector3, water_radius: f32, p: WaterParams, visible: bool) {
        self.mmi.set_visible(visible);
        if !visible {
            return;
        }
        let pr = proxy_radius(water_radius);
        if (pr - self.proxy_r).abs() > self.proxy_r.max(1.0) * 1e-3 {
            self.mesh.set_radius(pr);
            self.mesh.set_height(pr * 2.0);
            self.proxy_r = pr;
        }
        self.mmi.set_global_position(center);

        let m = &mut self.material;
        m.set_shader_parameter("planet_center", &center.to_variant());
        m.set_shader_parameter("water_radius", &water_radius.to_variant());
        m.set_shader_parameter("deep_color", &p.deep_color.to_variant());
        m.set_shader_parameter("shallow_color", &p.shallow_color.to_variant());
        m.set_shader_parameter("wave_strength", &p.wave_strength.to_variant());
        m.set_shader_parameter("wave_scale", &p.wave_scale.to_variant());
        m.set_shader_parameter("wave_speed", &p.wave_speed.to_variant());
        m.set_shader_parameter("underwater_color", &p.underwater_color.to_variant());
        m.set_shader_parameter("underwater_density", &p.underwater_density.to_variant());
        m.set_shader_parameter("sun_direction", &p.sun_dir.to_variant());
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn proxy_margin_is_applied() {
        // The proxy must strictly over-cover the analytic sphere.
        assert!(crate::water::PROXY_MARGIN > 1.0);
    }
}
