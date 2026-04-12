shader_type spatial;

uniform samplerCube planet_texture : source_color;
uniform samplerCube normal_cubemap : hint_default_black;
uniform bool use_normal_cubemap = true;

// LOD patch textures (up to 4 levels)
uniform sampler2D lod1_texture : source_color;
uniform sampler2D lod2_texture : source_color;
uniform sampler2D lod3_texture : source_color;
uniform sampler2D lod4_texture : source_color;

// Number of active LOD levels (0 = cubemap only)
uniform int lod_count = 0;

// Per-LOD metadata: center direction, tangent basis, angular half-extent
uniform vec3 lod1_center_dir = vec3(0.0, 0.0, 1.0);
uniform vec3 lod1_tangent_u  = vec3(1.0, 0.0, 0.0);
uniform vec3 lod1_tangent_v  = vec3(0.0, 1.0, 0.0);
uniform float lod1_angular_extent = 0.7854;

uniform vec3 lod2_center_dir = vec3(0.0, 0.0, 1.0);
uniform vec3 lod2_tangent_u  = vec3(1.0, 0.0, 0.0);
uniform vec3 lod2_tangent_v  = vec3(0.0, 1.0, 0.0);
uniform float lod2_angular_extent = 0.3927;

uniform vec3 lod3_center_dir = vec3(0.0, 0.0, 1.0);
uniform vec3 lod3_tangent_u  = vec3(1.0, 0.0, 0.0);
uniform vec3 lod3_tangent_v  = vec3(0.0, 1.0, 0.0);
uniform float lod3_angular_extent = 0.1963;

uniform vec3 lod4_center_dir = vec3(0.0, 0.0, 1.0);
uniform vec3 lod4_tangent_u  = vec3(1.0, 0.0, 0.0);
uniform vec3 lod4_tangent_v  = vec3(0.0, 1.0, 0.0);
uniform float lod4_angular_extent = 0.0982;

varying vec3 model_position;

void vertex() {
    model_position = VERTEX;
}

// Project a fragment direction onto a LOD patch defined by tangent-plane basis.
// Returns true if the fragment is within the patch and outputs local_uv in [0,1]^2.
bool in_lod_patch_tangent(vec3 frag_dir, vec3 center, vec3 tu, vec3 tv, float extent, out vec2 local_uv) {
    // Check if fragment is on the same hemisphere as the patch center
    float cos_angle = dot(frag_dir, center);
    if (cos_angle <= 0.0) return false;

    // Gnomonic projection onto tangent plane at center
    vec3 projected = frag_dir / cos_angle - center;
    float u_offset = dot(projected, tu);
    float v_offset = dot(projected, tv);

    // Convert offsets to angular coordinates and check bounds
    float angle_u = atan(u_offset);
    float angle_v = atan(v_offset);

    if (abs(angle_u) > extent || abs(angle_v) > extent) return false;

    // Map from [-extent, +extent] to [0, 1]
    local_uv = vec2(angle_u / (2.0 * extent) + 0.5, angle_v / (2.0 * extent) + 0.5);
    return true;
}

void fragment() {
    vec3 cubemap_dir = normalize(model_position);

    // Default: sample cubemap
    vec4 tex_color = texture(planet_texture, cubemap_dir);

    ALBEDO = tex_color.rgb;

    // Apply precomputed normal from normal cubemap
    if (use_normal_cubemap) {
        vec3 stored_normal = texture(normal_cubemap, cubemap_dir).rgb * 2.0 - 1.0;
        NORMAL = normalize(mat3(VIEW_MATRIX) * mat3(MODEL_MATRIX) * stored_normal);
    }
}
