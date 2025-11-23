shader_type spatial;


void vertex() {
	// Called for every vertex the material is visible on.
  	//VERTEX.y += 10.0 * sin(TIME * 1.0 + VERTEX.z);
	//VERTEX.xyz += NORMAL.xyz * UV.x;
	//UV.y = VERTEX.x;
	//UV2 = VERTEX.yz;

}

uniform float radius = 1.0;

void fragment() {
    // Called for every pixel the material is visible on.

    vec3 color = vec3(0, 0, 0);
	
	float c = UV.x/2.0;
	
	color  = (1.0-c) * vec3(0, 0, 0) + c * vec3(0, 0, 1);
	//color = vec3(0.01, 0.01, 0.01);
	//color = vec3(0.4, 0.61, 0.42);
    ALBEDO = color;
}