// Slang Compute Shader — Cubemap Noise Generator
// Generates cubemap faces by projecting pixels onto a sphere and sampling FBM noise.
//
// Noise algorithms used:
//  1. 3D Gradient Noise (Perlin-style, hash-based)
//     - Uses quintic interpolation (Ken Perlin's "Improved Noise", 2002)
//     - https://mrl.cs.nyu.edu/~perlin/paper445.pdf
//     - https://iquilezles.org/articles/gradientnoise/
//  2. Fractal Brownian Motion (FBM), 4 octaves
//     - Classic layered noise with halving amplitude & doubling frequency
//     - https://iquilezles.org/articles/fbm/
//     - https://thebookofshaders.com/13/

[[vk::binding(0, 0)]]
RWTexture2DArray<float4> out_cubemap;

[[vk::binding(1, 0)]]
ConstantBuffer<uint> face_index;

[[vk::binding(2, 0)]]
ConstantBuffer<uint> tex_size;

[[vk::binding(3, 0)]]
ConstantBuffer<float> radius;

typealias vec3 = float3;

// --- 3D gradient noise (hash-based, Perlin-style) ---

float3 hash(float3 p)
{
    p = float3(
        dot(p, float3(127.1, 311.7, 74.7)),
        dot(p, float3(269.5, 183.3, 246.1)),
        dot(p, float3(113.5, 271.9, 124.6))
    );
    return frac(sin(p) * 43758.5453123) * 2.0 - 1.0;
}

// returns 3D gradient noise
float noise3d(float3 p)
{
    // grid
    vec3 i = floor(p);
    vec3 f = fract(p);
    
    // quintic interpolant
    vec3 u = f*f*f*(f*(f*6.0-15.0)+10.0);
    
    // gradients
    vec3 ga = hash( i+vec3(0.0,0.0,0.0) );
    vec3 gb = hash( i+vec3(1.0,0.0,0.0) );
    vec3 gc = hash( i+vec3(0.0,1.0,0.0) );
    vec3 gd = hash( i+vec3(1.0,1.0,0.0) );
    vec3 ge = hash( i+vec3(0.0,0.0,1.0) );
    vec3 gf = hash( i+vec3(1.0,0.0,1.0) );
    vec3 gg = hash( i+vec3(0.0,1.0,1.0) );
    vec3 gh = hash( i+vec3(1.0,1.0,1.0) );
    
    // projections
    float va = dot( ga, f-vec3(0.0,0.0,0.0) );
    float vb = dot( gb, f-vec3(1.0,0.0,0.0) );
    float vc = dot( gc, f-vec3(0.0,1.0,0.0) );
    float vd = dot( gd, f-vec3(1.0,1.0,0.0) );
    float ve = dot( ge, f-vec3(0.0,0.0,1.0) );
    float vf = dot( gf, f-vec3(1.0,0.0,1.0) );
    float vg = dot( gg, f-vec3(0.0,1.0,1.0) );
    float vh = dot( gh, f-vec3(1.0,1.0,1.0) );
	
    // interpolation
    return va + 
           u.x*(vb-va) + 
           u.y*(vc-va) + 
           u.z*(ve-va) + 
           u.x*u.y*(va-vb-vc+vd) + 
           u.y*u.z*(va-vc-ve+vg) + 
           u.z*u.x*(va-vb-ve+vf) + 
           u.x*u.y*u.z*(-va+vb+vc-vd+ve-vf-vg+vh);
}

// --- Fractal Brownian Motion (4 octaves) ---

float fbm(float3 p)
{
    float value = 0.0;
    float amplitude = 0.5;
    float frequency = 1.0;
    for (int i = 0; i < 4; i++)
    {
        value += amplitude * noise3d(p * frequency);
        frequency *= 2.0;
        amplitude *= 0.5;
    }
    return value;
}

// --- Cube face → 3D direction ---

float3 cube_face_direction(uint face, float u, float v)
{
    switch (face)
    {
    case 0: return float3( 1.0,   v,  -u);   // +X
    case 1: return float3(-1.0,   v,   u);   // -X
    case 2: return float3(  -u, 1.0,  -v);   // +Y
    case 3: return float3(  -u,-1.0,   v);   // -Y
    case 4: return float3(   u,   v, 1.0);   // +Z
    case 5: return float3(  -u,   v,-1.0);   // -Z
    default: return float3(0, 0, 0);
    }
}

// --- Terrain color gradient ---

float4 terrain_color(float n)
{
    if (n < -0.1)
        return float4(0.1, 0.2, 0.5, 1.0);   // ocean
    else if (n < 0.1)
        return float4(0.76, 0.70, 0.50, 1.0); // beach
    else if (n < 0.5)
        return float4(0.2, 0.5, 0.2, 1.0);    // land
    else if (n < 0.8)
        return float4(0.5, 0.35, 0.2, 1.0);   // mountain
    else
        return float4(0.9, 0.9, 0.95, 1.0);   // snow
}

// --- Main entry point ---

[shader("compute")]
[numthreads(8, 8, 1)]
void computeMain(uint3 DTid : SV_DispatchThreadID)
{
    uint x = DTid.x;
    uint y = DTid.y;

    if (x >= tex_size || y >= tex_size) return;

    // Map pixel to [-1, +1]
    float u = (float(x) + 0.5) / float(tex_size) * 2.0 - 1.0;
    float v = (float(y) + 0.5) / float(tex_size) * 2.0 - 1.0;

    // Get 3D direction on cube face, project to sphere
    float3 cube_pt = cube_face_direction(face_index, u, v);
    float3 sphere_pt = normalize(cube_pt) * radius;

    // Sample FBM noise and map to terrain color
    float n = fbm(sphere_pt);
    float4 color = terrain_color(n);

    out_cubemap[uint3(x, y, face_index)] = color;
}
