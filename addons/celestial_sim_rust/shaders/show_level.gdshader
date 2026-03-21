shader_type spatial;


void vertex() {
	// Called for every vertex the material is visible on.
  	//VERTEX.y += sin(TIME * 1.0 + VERTEX.z);
	//UV = VERTEX.xy;
	//UV2 = VERTEX.zz;

}

uniform float radius = 1.0;

void fragment() {
    // Called for every pixel the material is visible on.

    vec3 color = vec3(0, 0, 0);
	
	float c = UV.x/2.0;
	
	//if (c > 8.9){
		//c = c - 9.0;
	//}

    // Assign a different color for each level
    if (c > 0.9) {
        color = vec3(0.6, 0.4, 0.2);
    }
	if (c > 1.9) {
        color = vec3(0.92, 0.72, 0.34);
    }
	if (c > 2.9) {
        color = vec3(0.4, 0.61, 0.42);
    }
	if (c > 3.9) {
        color = vec3(0.1, 0.2, 0.3);
    }
	if (c > 4.9) {
        color = vec3(0.1, 0.3, 0.5);
    }
	if (c > 5.9) {
        color = vec3(0.4, 0.5, 0.6);
    }
	if (c > 6.9) {
        color = vec3(0.7, 0.8, 0.9);
        color = vec3(0.6, 0.7, 0.8);
    } 
	//color = vec3(0.4, 0.61, 0.42);
	//color = vec3(0.01, 0.01, 0.01);
	//color = vec3(0.4, 0.61, 0.42);
    ALBEDO = color;
}