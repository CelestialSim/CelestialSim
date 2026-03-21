// Shader for CesFinalStateCompositor (scene-space)

#[vertex]
#version 450

layout(set = 0, binding = 0, std430) readonly buffer Vertices {
    vec4 positions[];
};

layout(set = 0, binding = 1, std430) readonly buffer Triangles {
    uvec4 tris[];
};

layout(push_constant) uniform PushConstants {
    mat4 model;
    mat4 view;
    mat4 projection;
    vec4 light_dir;    // xyz = direction (view space), w = intensity
    vec4 light_color;  // rgb = color, a = ambient
    vec4 base_color;   // rgb = base color, a unused
} pc;

layout(location = 0) flat out vec3 frag_normal;

void main() {
    uint tri_id = gl_VertexIndex / 3u;
    uint corner = gl_VertexIndex % 3u;

    uvec4 tri = tris[tri_id];
    vec3 v0 = positions[tri.x].xyz;
    vec3 v1 = positions[tri.y].xyz;
    vec3 v2 = positions[tri.z].xyz;

    vec3 pos = (corner == 0u) ? v0 : ((corner == 1u) ? v1 : v2);
    vec3 face_normal = normalize(cross(v1 - v0, v2 - v0));

    mat3 normal_matrix = mat3(transpose(inverse(pc.view * pc.model)));
    frag_normal = normalize(normal_matrix * face_normal);

    vec4 world_pos = pc.model * vec4(pos, 1.0);
    vec4 view_pos = pc.view * world_pos;
    gl_Position = pc.projection * view_pos;
}

#[fragment]
#version 450

layout(location = 0) out vec4 frag_color;

layout(location = 0) flat in vec3 frag_normal;

layout(push_constant) uniform PushConstants {
    mat4 model;
    mat4 view;
    mat4 projection;
    vec4 light_dir;
    vec4 light_color;
    vec4 base_color;
} pc;

void main() {
    vec3 n = normalize(frag_normal);
    vec3 light_dir = normalize(pc.light_dir.xyz);
    float intensity = pc.light_dir.w;
    float ambient = pc.light_color.a;

    float diff = max(dot(n, light_dir), 0.0);
    vec3 color = pc.light_color.rgb * (ambient + diff * intensity);
    color += pc.base_color.rgb * ambient;

    frag_color = vec4(color, 1.0);
}
