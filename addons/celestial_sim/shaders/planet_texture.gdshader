shader_type spatial;

uniform samplerCube planet_texture : source_color;

varying vec3 model_position;

void vertex() {
    model_position = VERTEX;
}

void fragment() {
    vec3 cubemap_dir = normalize(model_position);
    vec4 tex_color = texture(planet_texture, cubemap_dir);
    ALBEDO = tex_color.rgb;
}
