shader_type spatial;

// ── Planet color sources (same uniforms as planet_texture.gdshader) ─────────
uniform samplerCube planet_texture : source_color;

uniform sampler2D snapshot1_texture : source_color;
uniform sampler2D snapshot2_texture : source_color;
uniform sampler2D snapshot3_texture : source_color;
uniform sampler2D snapshot4_texture : source_color;

uniform int snapshot_count = 0;

uniform vec3 snapshot1_center_dir = vec3(0.0, 0.0, 1.0);
uniform vec3 snapshot1_tangent_u  = vec3(1.0, 0.0, 0.0);
uniform vec3 snapshot1_tangent_v  = vec3(0.0, 1.0, 0.0);
uniform float snapshot1_angular_extent = 0.7854;

uniform vec3 snapshot2_center_dir = vec3(0.0, 0.0, 1.0);
uniform vec3 snapshot2_tangent_u  = vec3(1.0, 0.0, 0.0);
uniform vec3 snapshot2_tangent_v  = vec3(0.0, 1.0, 0.0);
uniform float snapshot2_angular_extent = 0.3927;

uniform vec3 snapshot3_center_dir = vec3(0.0, 0.0, 1.0);
uniform vec3 snapshot3_tangent_u  = vec3(1.0, 0.0, 0.0);
uniform vec3 snapshot3_tangent_v  = vec3(0.0, 1.0, 0.0);
uniform float snapshot3_angular_extent = 0.1963;

uniform vec3 snapshot4_center_dir = vec3(0.0, 0.0, 1.0);
uniform vec3 snapshot4_tangent_u  = vec3(1.0, 0.0, 0.0);
uniform vec3 snapshot4_tangent_v  = vec3(0.0, 1.0, 0.0);
uniform float snapshot4_angular_extent = 0.0982;

// ── Blue-noise dither volume ─────────────────────────────────────────────────
uniform sampler3D blue_noise_texture;

// ── Biome / spawn-filter parameters ─────────────────────────────────────────
uniform vec4 target_color = vec4(0.2, 0.6, 0.2, 1.0);
uniform float color_tolerance : hint_range(0.001, 2.0) = 0.3;
uniform float noise_strength  : hint_range(0.0,  1.0)  = 1.0;

// ── Varying ──────────────────────────────────────────────────────────────────
varying vec3 model_position;

void vertex() {
	model_position = VERTEX;
}

// Verbatim from planet_texture.gdshader — projects a fragment direction onto a
// snapshot patch and returns the local [0,1]^2 UV if it falls within the patch.
bool in_snapshot_patch_tangent(vec3 frag_dir, vec3 center, vec3 tu, vec3 tv,
		float extent, out vec2 local_uv) {
	float cos_angle = dot(frag_dir, center);
	if (cos_angle <= 0.0) return false;

	vec3 projected = frag_dir / cos_angle - center;
	float u_offset = dot(projected, tu);
	float v_offset = dot(projected, tv);

	float angle_u = atan(u_offset);
	float angle_v = atan(v_offset);

	if (abs(angle_u) > extent || abs(angle_v) > extent) return false;

	local_uv = vec2(angle_u / (2.0 * extent) + 0.5,
	                angle_v / (2.0 * extent) + 0.5);
	return true;
}

void fragment() {
	vec3 cubemap_dir = normalize(model_position);

	// ── Sample the highest-detail available color source ─────────────────
	vec4 tex_color = texture(planet_texture, cubemap_dir);
	if (snapshot_count > 0) {
		vec2 local_uv;
		if (snapshot_count >= 4 && in_snapshot_patch_tangent(cubemap_dir,
				snapshot4_center_dir, snapshot4_tangent_u, snapshot4_tangent_v,
				snapshot4_angular_extent, local_uv)) {
			tex_color = texture(snapshot4_texture, local_uv);
		} else if (snapshot_count >= 3 && in_snapshot_patch_tangent(cubemap_dir,
				snapshot3_center_dir, snapshot3_tangent_u, snapshot3_tangent_v,
				snapshot3_angular_extent, local_uv)) {
			tex_color = texture(snapshot3_texture, local_uv);
		} else if (snapshot_count >= 2 && in_snapshot_patch_tangent(cubemap_dir,
				snapshot2_center_dir, snapshot2_tangent_u, snapshot2_tangent_v,
				snapshot2_angular_extent, local_uv)) {
			tex_color = texture(snapshot2_texture, local_uv);
		} else if (in_snapshot_patch_tangent(cubemap_dir,
				snapshot1_center_dir, snapshot1_tangent_u, snapshot1_tangent_v,
				snapshot1_angular_extent, local_uv)) {
			tex_color = texture(snapshot1_texture, local_uv);
		}
	}

	// ── Biome density: closeness of fragment color to target ──────────────
	float density = clamp(
		1.0 - length(tex_color.rgb - target_color.rgb) / max(color_tolerance, 0.001),
		0.0, 1.0);

	// ── Blue-noise threshold: map unit-sphere direction to [0,1]^3 UVW ───
	vec3 uvw = fract(cubemap_dir * 0.5 + 0.5);
	float noise_sample = texture(blue_noise_texture, uvw).r;

	// White where the object would spawn, black where it wouldn't.
	float mask = step(noise_sample * noise_strength, density);
	ALBEDO = vec3(mask);
}
