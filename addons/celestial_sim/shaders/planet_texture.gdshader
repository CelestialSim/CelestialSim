shader_type spatial;

uniform samplerCube planet_texture : source_color;
uniform samplerCube normal_cubemap : hint_default_black;
uniform bool use_normal_cubemap = true;

// snapshot patch textures (up to 4 levels) — color
uniform sampler2D snapshot1_texture : source_color;
uniform sampler2D snapshot2_texture : source_color;
uniform sampler2D snapshot3_texture : source_color;
uniform sampler2D snapshot4_texture : source_color;

// snapshot patch textures — normals
uniform sampler2D snapshot1_normal_texture : hint_default_black;
uniform sampler2D snapshot2_normal_texture : hint_default_black;
uniform sampler2D snapshot3_normal_texture : hint_default_black;
uniform sampler2D snapshot4_normal_texture : hint_default_black;

// Number of active snapshot levels (0 = cubemap only)
uniform int snapshot_count = 0;

// Per-LOD metadata: center direction, tangent basis, angular half-extent
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

varying vec3 model_position;

void vertex() {
    model_position = VERTEX;
}

// Project a fragment direction onto a snapshot patch defined by tangent-plane basis.
// Returns true if the fragment is within the patch and outputs local_uv in [0,1]^2.
bool in_snapshot_patch_tangent(vec3 frag_dir, vec3 center, vec3 tu, vec3 tv, float extent, out vec2 local_uv) {
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
    bool used_snapshot = false;

    // Try snapshot patches from highest detail to lowest, fall back to cubemap
    if (snapshot_count > 0) {
        vec2 local_uv;

        if (snapshot_count >= 4 && in_snapshot_patch_tangent(cubemap_dir, snapshot4_center_dir, snapshot4_tangent_u, snapshot4_tangent_v, snapshot4_angular_extent, local_uv)) {
            tex_color = texture(snapshot4_texture, local_uv);
            if (use_normal_cubemap) {
                vec3 stored_normal = texture(snapshot4_normal_texture, local_uv).rgb * 2.0 - 1.0;
                NORMAL = normalize(mat3(VIEW_MATRIX) * mat3(MODEL_MATRIX) * stored_normal);
            }
            used_snapshot = true;
        } else if (snapshot_count >= 3 && in_snapshot_patch_tangent(cubemap_dir, snapshot3_center_dir, snapshot3_tangent_u, snapshot3_tangent_v, snapshot3_angular_extent, local_uv)) {
            tex_color = texture(snapshot3_texture, local_uv);
            if (use_normal_cubemap) {
                vec3 stored_normal = texture(snapshot3_normal_texture, local_uv).rgb * 2.0 - 1.0;
                NORMAL = normalize(mat3(VIEW_MATRIX) * mat3(MODEL_MATRIX) * stored_normal);
            }
            used_snapshot = true;
        } else if (snapshot_count >= 2 && in_snapshot_patch_tangent(cubemap_dir, snapshot2_center_dir, snapshot2_tangent_u, snapshot2_tangent_v, snapshot2_angular_extent, local_uv)) {
            tex_color = texture(snapshot2_texture, local_uv);
            if (use_normal_cubemap) {
                vec3 stored_normal = texture(snapshot2_normal_texture, local_uv).rgb * 2.0 - 1.0;
                NORMAL = normalize(mat3(VIEW_MATRIX) * mat3(MODEL_MATRIX) * stored_normal);
            }
            used_snapshot = true;
        } else if (in_snapshot_patch_tangent(cubemap_dir, snapshot1_center_dir, snapshot1_tangent_u, snapshot1_tangent_v, snapshot1_angular_extent, local_uv)) {
            tex_color = texture(snapshot1_texture, local_uv);
            if (use_normal_cubemap) {
                vec3 stored_normal = texture(snapshot1_normal_texture, local_uv).rgb * 2.0 - 1.0;
                NORMAL = normalize(mat3(VIEW_MATRIX) * mat3(MODEL_MATRIX) * stored_normal);
            }
            used_snapshot = true;
        }
    }

    ALBEDO = tex_color.rgb;

    // Apply normal cubemap fallback when no snapshot patch was used
    if (!used_snapshot && use_normal_cubemap) {
        vec3 stored_normal = texture(normal_cubemap, cubemap_dir).rgb * 2.0 - 1.0;
        NORMAL = normalize(mat3(VIEW_MATRIX) * mat3(MODEL_MATRIX) * stored_normal);
    }
}
