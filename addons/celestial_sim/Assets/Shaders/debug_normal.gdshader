shader_type spatial;


void vertex() {
	// Called for every vertex the material is visible on.
  	//VERTEX.y += sin(TIME * 1.0 + VERTEX.z);
	UV = VERTEX.xy;
	UV2 = VERTEX.zz;

}

uniform float radius = 100.0;

void fragment() {
	// Called for every pixel the material is visible on.
	vec3 pos;
	pos.xy = UV.xy;
	pos.z = UV2.x;
	
	vec3 top_color = vec3(0.92, 0.72, 0.34);
	vec3 bottom_color = vec3(0.4, 0.61, 0.42);
	
	//vec3 top_color = vec3(1.0, 0.0, 0.0);
	//vec3 bottom_color = vec3(0.0, 1.0, 0.0);
	
	// Linear Distance from the center
	float dist = distance(pos,vec3(0)) / (radius * 1.1);
	
	vec3 color = mix(bottom_color,top_color,pow(dist,18));

	ALBEDO = color;
	//ALBEDO = pos.xxx / radius;

}