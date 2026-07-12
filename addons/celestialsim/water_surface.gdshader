// Analytic planetary water — MIT-based ocean shading + Beer–Lambert underwater.
//
// The water surface is NOT tessellated. A coarse proxy sphere (see WaterRuntime)
// only spawns fragments; this shader ray-intersects the *mathematical* sea-level
// sphere (planet_center, water_radius) per pixel, so the surface is perfectly
// smooth at any distance — no faceting. Terrain occlusion and water depth come
// from the hinted depth texture (the already-rendered opaque terrain), compared
// along the camera ray.
//
// ── Licensing / provenance ───────────────────────────────────────────────────
// The ocean *shading* is a Godot port of Sebastian Lague's planetary ocean,
// which is MIT-licensed:
//   SebLague/Solar-System  (© 2020 Sebastian Lague, MIT)
//   Assets/Scripts/Celestial/Shaders/PostProcessing/OceanEffect.shader
//   Assets/Scripts/Celestial/Shaders/Includes/{Math,Triplanar}.cginc
// Ported methods, faithful to that source:
//   * raySphere               (Math.cginc)      — analytic ocean-sphere hit.
//   * exponential depth colour (OceanEffect)     — shallow→deep via 1-exp(-d/·).
//   * triplanarNormal + blend_rnm (Triplanar.cginc) — Reoriented-Normal-Mapping
//     triplanar wave normals (itself based on Ben Golus' triplanar article).
//   * acos-Gaussian specular   (OceanEffect)     — sun glint off the wave normal.
// Depth reconstruction is the standard Godot-docs method (hinted depth texture).

shader_type spatial;
// cull_front: render only the proxy's BACK faces — one fragment per covered
// pixel (no front+back overdraw), correct whether the camera is outside the
// shell, inside it, or under the surface (so no separate underwater shell flip
// and no "water vanishes a hair above the surface"). depth_test/­draw disabled:
// the back faces sit at the far side of the sphere, so we occlude MANUALLY
// against the reconstructed terrain distance. unshaded: we light the water
// ourselves (no GGX env sampling). world_vertex_coords: VERTEX is world-space.
render_mode cull_front, depth_draw_never, depth_test_disabled, blend_mix, unshaded, world_vertex_coords;

// Godot 4.4+ removed the DEPTH_TEXTURE builtin; opaque depth is a hinted uniform.
uniform sampler2D depth_tex : hint_depth_texture, filter_nearest;

uniform vec3 planet_center = vec3(0.0);
uniform float water_radius = 970.0;

// ── Body colour (Lague colA/colB, exp depth blend) ───────────────────────────
uniform vec3 shallow_color : source_color = vec3(0.20, 0.55, 0.70); // colA
uniform vec3 deep_color : source_color = vec3(0.05, 0.22, 0.42);    // colB
// World-space depths (units) at which colour saturates to deep / alpha to opaque.
uniform float color_depth = 40.0;
uniform float alpha_depth = 12.0;
uniform float shallow_alpha = 0.30;

// ── Shoreline foam ───────────────────────────────────────
uniform vec3 foam_color : source_color = vec3(0.85, 0.93, 0.98);
uniform float foam_width = 3.0;

// ── Waves: two scrolling tiling normal maps, triplanar RNM blend (Lague) ─────
uniform sampler2D wave_normal_a : hint_normal, filter_linear_mipmap, repeat_enable;
uniform sampler2D wave_normal_b : hint_normal, filter_linear_mipmap, repeat_enable;
uniform float wave_strength = 0.55; // 0 = flat mirror (skips the texture taps)
uniform float wave_scale = 0.15;    // tiling frequency
uniform float wave_speed = 0.04;    // scroll speed

// ── Lighting (Lague diffuse + acos-Gaussian specular) ────────────────────────
// sun_direction points TOWARD the sun (pushed from the scene DirectionalLight3D).
uniform vec3 sun_direction = vec3(0.4, 0.7, 0.55);
uniform vec3 specular_color : source_color = vec3(1.0, 0.96, 0.85);
uniform float smoothness = 0.92;  // specular tightness (0..1)
uniform float ambient = 0.12;     // keep the night side from going pure black

// ── Fresnel sky rim reflection ───────────────────────────
uniform float fresnel_strength = 0.7;
uniform vec3 horizon_color : source_color = vec3(0.55, 0.70, 0.85);
uniform vec3 zenith_color : source_color = vec3(0.12, 0.30, 0.62);
uniform float sky_gradient = 1.5;

// ── Underwater (Beer–Lambert fog when the camera is below the surface) ────────
uniform vec3 underwater_color : source_color = vec3(0.04, 0.16, 0.28);
uniform float underwater_density = 0.02;

varying vec3 v_wpos;

void vertex() {
	// world_vertex_coords -> VERTEX is the proxy fragment's world position, which
	// lies on the camera ray for this pixel (used to reconstruct the ray dir).
	v_wpos = VERTEX;
}

// Analytic ray vs sphere (Lague Math.cginc::raySphere). Returns
// (distance-to-near-hit, thickness through the sphere). Misses => (BIG, 0).
// rayDir is assumed normalized (a == 1).
vec2 ray_sphere(vec3 centre, float radius, vec3 ro, vec3 rd) {
	vec3 off = ro - centre;
	float b = dot(off, rd);
	float c = dot(off, off) - radius * radius;
	float d = b * b - c;
	if (d > 0.0) {
		float s = sqrt(d);
		float near = max(0.0, -b - s);
		float far = -b + s;
		if (far >= 0.0) {
			return vec2(near, far - near);
		}
	}
	return vec2(1e12, 0.0);
}

// Reoriented Normal Mapping blend (Lague Triplanar.cginc::blend_rnm; itself
// self-shadow.com/publications/blending-in-detail). n1/n2 are signed normals.
vec3 blend_rnm(vec3 n1, vec3 n2) {
	n1.z += 1.0;
	n2.xy = -n2.xy;
	return n1 * dot(n1, n2) / n1.z - n2;
}

// Triplanar tangent-space normal sampled in world space (Lague
// Triplanar.cginc::triplanarNormal). `off` scrolls the swell. 3 texture taps.
vec3 triplanar_normal(vec3 p, vec3 N, float scale, vec2 off, sampler2D nmap) {
	vec3 absN = abs(N);
	// Square-and-normalize the blend weights (pow(abs(N),4) sharpens the seams).
	vec3 w = pow(absN, vec3(4.0));
	w /= dot(w, vec3(1.0)) + 1e-5;
	vec2 uvX = p.zy * scale + off;
	vec2 uvY = p.xz * scale + off;
	vec2 uvZ = p.xy * scale + off;
	vec3 tnX = texture(nmap, uvX).xyz * 2.0 - 1.0;
	vec3 tnY = texture(nmap, uvY).xyz * 2.0 - 1.0;
	vec3 tnZ = texture(nmap, uvZ).xyz * 2.0 - 1.0;
	// RNM-blend each plane's tangent normal onto the geometric normal.
	tnX = blend_rnm(vec3(N.zy, absN.x), tnX);
	tnY = blend_rnm(vec3(N.xz, absN.y), tnY);
	tnZ = blend_rnm(vec3(N.xy, absN.z), tnZ);
	vec3 axis = sign(N);
	tnX.z *= axis.x;
	tnY.z *= axis.y;
	tnZ.z *= axis.z;
	// Swizzle each back to world space and weight.
	return normalize(tnX.zyx * w.x + tnY.xzy * w.y + tnZ.xyz * w.z);
}

void fragment() {
	// --- camera ray for this pixel ---
	vec3 ro = CAMERA_POSITION_WORLD;
	vec3 rd = normalize(v_wpos - ro);

	// --- analytic ray vs sea-level sphere ---
	vec2 hit = ray_sphere(planet_center, water_radius, ro, rd);
	float t = hit.x;             // distance to the near ocean hit
	float thickness = hit.y;     // distance the ray travels through the ocean
	if (thickness <= 0.0) {
		discard; // ray misses the ocean entirely
	}
	float cam_dist = length(ro - planet_center);
	bool inside = cam_dist < water_radius;

	// --- opaque scene point as an along-ray WORLD distance (occlusion + depth) --
	float scene_d = texture(depth_tex, SCREEN_UV).r;
	vec4 vpos = INV_PROJECTION_MATRIX * vec4(SCREEN_UV * 2.0 - 1.0, scene_d, 1.0);
	vpos.xyz /= vpos.w;
	vec3 terrain_world = (INV_VIEW_MATRIX * vec4(vpos.xyz, 1.0)).xyz;
	float terrain_t = distance(terrain_world, ro);

	// Ocean depth the view ray travels before it hits terrain or exits the ocean
	// (Lague: min(dstThroughOcean, sceneDepth - dstToOcean)).
	float ocean_view_depth = min(thickness, terrain_t - t);

	if (inside) {
		// --- underwater: Beer–Lambert fog toward underwater_color over the water
		// column (distance until the ray exits the sphere or hits terrain). ---
		float path = min(thickness, terrain_t);
		float fogf = clamp(1.0 - exp(-path * underwater_density), 0.0, 1.0);
		ALBEDO = underwater_color;
		ALPHA = fogf;
	} else if (ocean_view_depth <= 0.0) {
		// Terrain pokes above the surface here (or the ocean is behind it):
		// nothing to draw.
		discard;
	} else {
		vec3 hit_pos = ro + rd * t;
		vec3 N = normalize(hit_pos - planet_center);

		// --- wave normal: two scrolling triplanar samples, RNM-blended (Lague) --
		vec3 wn = N;
		if (wave_strength > 0.001) {
			float tt = TIME * wave_speed;
			vec2 offA = vec2(tt, tt * 0.8);
			vec2 offB = vec2(-tt * 0.8, -tt * 0.3);
			wn = triplanar_normal(hit_pos, N, wave_scale, offA, wave_normal_a);
			wn = triplanar_normal(hit_pos, wn, wave_scale, offB, wave_normal_b);
			wn = normalize(mix(N, wn, clamp(wave_strength, 0.0, 1.0)));
		}

		// --- body colour by depth (Lague exp shallow→deep) ---
		float ct = 1.0 - exp(-ocean_view_depth / max(color_depth, 1e-3));
		vec3 body = mix(shallow_color, deep_color, ct);

		// --- diffuse (Lague dot(N, sun)) with a small ambient floor ---
		float ndl = clamp(dot(N, normalize(sun_direction)), 0.0, 1.0);
		float diff = ndl + ambient * (1.0 - ndl);

		// --- specular sun glint (Lague acos-Gaussian off the wave normal) ---
		vec3 H = normalize(normalize(sun_direction) - rd);
		float spec_ang = acos(clamp(dot(H, wn), -1.0, 1.0));
		float spec_exp = spec_ang / max(1e-3, 1.0 - smoothness);
		float spec = exp(-spec_exp * spec_exp);

		// --- fresnel sky rim reflection ---
		vec3 R = reflect(rd, wn);
		float elev = clamp(dot(R, N) * 0.5 + 0.5, 0.0, 1.0);
		vec3 sky = mix(horizon_color, zenith_color, pow(elev, sky_gradient));
		float fres = clamp(pow(1.0 - clamp(dot(wn, -rd), 0.0, 1.0), 5.0) * fresnel_strength, 0.0, 1.0);

		// --- shoreline foam ---
		float foam = 1.0 - smoothstep(0.0, foam_width, ocean_view_depth);
		// break the foam ring up with the wave normal's deviation from radial
		foam *= smoothstep(0.25, 0.9, (1.0 - dot(wn, N)) * 6.0 + 0.35);

		vec3 col = body * diff;
		col = mix(col, sky, fres);
		col += spec * specular_color;
		col = mix(col, foam_color, foam);

		float alpha = 1.0 - exp(-ocean_view_depth / max(alpha_depth, 1e-3));
		alpha = max(max(max(alpha, shallow_alpha), fres), foam);

		ALBEDO = col;
		ALPHA = clamp(alpha, 0.0, 1.0);
	}
}
