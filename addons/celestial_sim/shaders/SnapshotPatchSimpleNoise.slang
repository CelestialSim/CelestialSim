// Slang Compute Shader — Simple FBM Snapshot Patch
// Matches CubemapNoise.slang visual output via tangent-plane projection.
// No extra params (binding 7) needed — self-contained FBM + terrain_color.

import snapshot_patch_utils;

[[vk::binding(0, 0)]]
RWTexture2D<float4> out_patch;

typealias vec3 = float3;

// --- 3D gradient noise (identical to CubemapNoise.slang) ---

float3 hash(float3 p)
{
    p = float3(
        dot(p, float3(127.1, 311.7, 74.7)),
        dot(p, float3(269.5, 183.3, 246.1)),
        dot(p, float3(113.5, 271.9, 124.6)));
    return frac(sin(p) * 43758.5453123) * 2.0 - 1.0;
}

float noise3d(float3 p)
{
    vec3 i = floor(p);
    vec3 f = fract(p);
    vec3 u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);

    vec3 ga = hash(i + vec3(0.0, 0.0, 0.0));
    vec3 gb = hash(i + vec3(1.0, 0.0, 0.0));
    vec3 gc = hash(i + vec3(0.0, 1.0, 0.0));
    vec3 gd = hash(i + vec3(1.0, 1.0, 0.0));
    vec3 ge = hash(i + vec3(0.0, 0.0, 1.0));
    vec3 gf = hash(i + vec3(1.0, 0.0, 1.0));
    vec3 gg = hash(i + vec3(0.0, 1.0, 1.0));
    vec3 gh = hash(i + vec3(1.0, 1.0, 1.0));

    float va = dot(ga, f - vec3(0.0, 0.0, 0.0));
    float vb = dot(gb, f - vec3(1.0, 0.0, 0.0));
    float vc = dot(gc, f - vec3(0.0, 1.0, 0.0));
    float vd = dot(gd, f - vec3(1.0, 1.0, 0.0));
    float ve = dot(ge, f - vec3(0.0, 0.0, 1.0));
    float vf = dot(gf, f - vec3(1.0, 0.0, 1.0));
    float vg = dot(gg, f - vec3(0.0, 1.0, 1.0));
    float vh = dot(gh, f - vec3(1.0, 1.0, 1.0));

    return va +
           u.x * (vb - va) +
           u.y * (vc - va) +
           u.z * (ve - va) +
           u.x * u.y * (va - vb - vc + vd) +
           u.y * u.z * (va - vc - ve + vg) +
           u.z * u.x * (va - vb - ve + vf) +
           u.x * u.y * u.z * (-va + vb + vc - vd + ve - vf - vg + vh);
}

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

float4 terrain_color(float n)
{
    if (n < -0.1)
        return float4(0.1, 0.2, 0.5, 1.0);
    else if (n < 0.1)
        return float4(0.76, 0.70, 0.50, 1.0);
    else if (n < 0.5)
        return float4(0.2, 0.5, 0.2, 1.0);
    else if (n < 0.8)
        return float4(0.5, 0.35, 0.2, 1.0);
    else
        return float4(0.9, 0.9, 0.95, 1.0);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void computeMain(uint3 DTid: SV_DispatchThreadID)
{
    uint x = DTid.x;
    uint y = DTid.y;

    if (x >= tex_size || y >= tex_size)
        return;

    float3 dir = snapshot_world_direction(uint2(x, y));
    float3 sphere_pt = dir * radius;

    float n = fbm(sphere_pt);
    float4 color = terrain_color(n);

    // Red border for debug visibility (toggled by show_border uniform)
    if (show_border != 0 && is_border_pixel(uint2(x, y), 3))
        color = float4(1.0, 0.0, 0.0, 1.0);

    out_patch[uint2(x, y)] = color;
}
