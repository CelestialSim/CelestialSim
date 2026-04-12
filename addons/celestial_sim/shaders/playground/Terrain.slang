// From
// https://www.shadertoy.com/view/7ljcRW

import playground;
import rendering;

/*
==========================================================================================

This shader is a fork of clayjohn's awesome "Eroded Terrain Noise":
https://www.shadertoy.com/view/MtGcWh

The original function seemed to have some biasing problems which I've attempted to fix.
I also wanted to improve the visualization to really show how great the result is.
I take no credit for the actual noise/math. I simply wanted to help showcase it.

==========================================================================================
*/

/*
==========================================================================================

Erosion parameters

==========================================================================================
*/

#define EROSION_TILES 4.0
#define EROSION_OCTAVES 5
#define EROSION_GAIN 0.5
#define EROSION_LACUNARITY 2.0

// Scale the input slope, leading to more erosion.
#define EROSION_SLOPE_STRENGTH 3.0
// Continuously modify the noise direction based on the previous fractal sample.
// This is what gives the slopes an interesting "branching" structure.
// A higher value will give you more branches.
#define EROSION_BRANCH_STRENGTH 3.0
// Maximum amount the erosion will modify the base height map
#define EROSION_STRENGTH 0.04

// Debug slider comparing the heightmap with and without erosion
// #define COMPARISON_SLIDER

/*
==========================================================================================

The stuff below is not strictly related to the erosion effect

==========================================================================================
*/

#define PI 3.14159265358979
#define DEG_TO_RAD (PI / 180.0)
#define saturate(x) clamp(x, 0.0, 1.0)
#define sq(x) (x * x)

// Base height noise parameters
#define HEIGHT_TILES 3.0
#define HEIGHT_OCTAVES 3
#define HEIGHT_AMP 0.25
#define HEIGHT_GAIN 0.1
#define HEIGHT_LACUNARITY 2.0

// Renderer settings

#define SHADOWS
#define WATER
#define WATER_HEIGHT 0.45
#define DETAIL_TEXTURE
// #define GREYSCALE
// #define SHOW_DIFFUSE
// #define SHOW_NORMALS
// #define SHOW_BUFFER

#define M_GROUND 0
#define M_STRATA 1
#define M_WATER 2

#define CLIFF_COLOR vec3(0.22, 0.2, 0.2)
#define DIRT_COLOR vec3(0.6, 0.5, 0.4)
#define GRASS_COLOR1 vec3(0.15, 0.3, 0.1)
#define GRASS_COLOR2 vec3(0.4, 0.5, 0.2)
#define SAND_COLOR vec3(0.8, 0.7, 0.6)

#define WATER_COLOR vec3(0.0, 0.05, 0.1)
#define WATER_SHORE_COLOR vec3(0.0, 0.25, 0.25)

// #define FIXED_SUN
#define SUN_COLOR (vec3(1.0, 0.98, 0.95) * 2.0)
#define AMBIENT_COLOR (vec3(0.3, 0.5, 0.7) * 0.1)

[playground::TIME]
uniform float timeFromStart;

[playground::MOUSE_POSITION]
uniform float4 _mousePosition;

// ╔══════════════════════════════════════════════════════════════════════════════╗
// ║ 🪐 Aliases                                                                   ║
// ╚══════════════════════════════════════════════════════════════════════════════╝
typealias vec2 = float2;
typealias vec3 = float3;
typealias vec4 = float4;
typealias mat2 = matrix<float, 2, 2>;
typealias mat3 = matrix<float, 3, 3>;

float2 GetBufferSize(float2 iResolution)
{
    float s = min(min(iResolution.x, iResolution.y), 768.0);
    return float2(s, s);
}

float TimeScrollOffset(float iTime)
{
    return cos(iTime * 0.2371) * 2.0;
}

float4 ToShadertoyMouse(float4 rawMouse)
{
    float isDown = rawMouse.z < 0.0 ? 1.0 : 0.0;
    return float4(rawMouse.xy, isDown, rawMouse.w);
}

// ╔══════════════════════════════════════════════════════════════════════════════╗
// ║ 🪐 Common                                                                    ║
// ╚══════════════════════════════════════════════════════════════════════════════╝

vec2 hash(in vec2 x)
{
    const vec2 k = vec2(0.3183099, 0.3678794);
    x = x * k + k.yx;
    return -1.0 + 2.0 * fract(16.0 * k * fract(x.x * x.y * (x.x + x.y)));
}

// from https://www.shadertoy.com/view/XdXBRH
vec3 noised(in vec2 p)
{
    vec2 i = floor(p);
    vec2 f = fract(p);

    vec2 u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
    vec2 du = 30.0 * f * f * (f * (f - 2.0) + 1.0);

    vec2 ga = hash(i + vec2(0.0, 0.0));
    vec2 gb = hash(i + vec2(1.0, 0.0));
    vec2 gc = hash(i + vec2(0.0, 1.0));
    vec2 gd = hash(i + vec2(1.0, 1.0));

    float va = dot(ga, f - vec2(0.0, 0.0));
    float vb = dot(gb, f - vec2(1.0, 0.0));
    float vc = dot(gc, f - vec2(0.0, 1.0));
    float vd = dot(gd, f - vec2(1.0, 1.0));

    return vec3(va + u.x * (vb - va) + u.y * (vc - va) + u.x * u.y * (va - vb - vc + vd),
                ga + u.x * (gb - ga) + u.y * (gc - ga) + u.x * u.y * (ga - gb - gc + gd) +
                    du * (u.yx * (va - vb - vc + vd) + vec2(vb, vc) - va));
}

vec3 RNM(vec3 n1, vec3 n2)
{
    n1 += vec3(0.0, 0.0, 1.0);
    n2 *= vec3(-1.0, -1.0, 1.0);
    return n1 * dot(n1, n2) / n1.z - n2;
}

// https://iquilezles.org/articles/intersectors
vec2 boxIntersection(in vec3 ro, in vec3 rd, vec3 boxSize, out vec3 outNormal)
{
    vec3 m = 1.0 / rd; // can precompute if traversing a set of aligned boxes
    vec3 n = m * ro;   // can precompute if traversing a set of aligned boxes
    vec3 k = abs(m) * boxSize;
    vec3 t1 = -n - k;
    vec3 t2 = -n + k;
    float tN = max(max(t1.x, t1.y), t1.z);
    float tF = min(min(t2.x, t2.y), t2.z);
    if (tN > tF || tF < 0.0)
        return vec2(-1.0); // no intersection
    outNormal = -sign(rd) * step(t1.yzx, t1.xyz) * step(t1.zxy, t1.xyz);
    return vec2(tN, tF);
}

// https://www.shadertoy.com/view/XsB3Rm
vec3 CameraRay(float fov, vec2 size, vec2 pos)
{
    vec2 xy = pos - size * 0.5;
    float cot_half_fov = tan((90.0 - fov * 0.5) * DEG_TO_RAD);
    float z = size.y * 0.5 * cot_half_fov;
    return normalize(vec3(xy, -z));
}
mat3 CameraRotation(vec2 angle)
{
    vec2 c = cos(angle);
    vec2 s = sin(angle);
    return mat3(
        c.y, 0.0, -s.y,
        s.y * s.x, c.x, c.y * s.x,
        s.y * c.x, -s.x, c.y * c.x);
}

vec3 SkyColor(vec3 rd, vec3 sun)
{
    float costh = dot(rd, sun);
    return AMBIENT_COLOR * PI * (1.0 - abs(costh) * 0.8);
}

vec3 Tonemap_ACES(vec3 x)
{
    // Narkowicz 2015, "ACES Filmic Tone Mapping Curve"
    const float a = 2.51;
    const float b = 0.03;
    const float c = 2.43;
    const float d = 0.59;
    const float e = 0.14;
    return (x * (a * x + b)) / (x * (c * x + d) + e);
}

//------------------------------------------------------------------------------
// BRDF
// https://www.shadertoy.com/view/XlKSDR
//------------------------------------------------------------------------------

float pow5(float x)
{
    float x2 = x * x;
    return x2 * x2 * x;
}

float D_GGX(float linearRoughness, float NoH, const vec3 h)
{
    // Walter et al. 2007, "Microfacet Models for Refraction through Rough Surfaces"
    float oneMinusNoHSquared = 1.0 - NoH * NoH;
    float a = NoH * linearRoughness;
    float k = linearRoughness / (oneMinusNoHSquared + a * a);
    float d = k * k * (1.0 / PI);
    return d;
}

float V_SmithGGXCorrelated(float linearRoughness, float NoV, float NoL)
{
    // Heitz 2014, "Understanding the Masking-Shadowing Function in Microfacet-Based BRDFs"
    float a2 = linearRoughness * linearRoughness;
    float GGXV = NoL * sqrt((NoV - a2 * NoV) * NoV + a2);
    float GGXL = NoV * sqrt((NoL - a2 * NoL) * NoL + a2);
    return 0.5 / (GGXV + GGXL);
}

vec3 F_Schlick(const vec3 f0, float VoH)
{
    // Schlick 1994, "An Inexpensive BRDF Model for Physically-Based Rendering"
    return f0 + (vec3(1.0) - f0) * pow5(1.0 - VoH);
}

float F_Schlick(float f0, float f90, float VoH)
{
    return f0 + (f90 - f0) * pow5(1.0 - VoH);
}

float Fd_Burley(float linearRoughness, float NoV, float NoL, float LoH)
{
    // Burley 2012, "Physically-Based Shading at Disney"
    float f90 = 0.5 + 2.0 * linearRoughness * LoH * LoH;
    float lightScatter = F_Schlick(1.0, f90, NoL);
    float viewScatter = F_Schlick(1.0, f90, NoV);
    return lightScatter * viewScatter * (1.0 / PI);
}

float Fd_Lambert()
{
    return 1.0 / PI;
}

vec3 Shade(vec3 diffuse, vec3 f0, float smoothness, vec3 n, vec3 v, vec3 l, vec3 lc)
{
    vec3 h = normalize(v + l);

    float NoV = abs(dot(n, v)) + 1e-5;
    float NoL = saturate(dot(n, l));
    float NoH = saturate(dot(n, h));
    float LoH = saturate(dot(l, h));

    float roughness = 1.0 - smoothness;
    float linearRoughness = roughness * roughness;
    float D = D_GGX(linearRoughness, NoH, h);
    float V = V_SmithGGXCorrelated(linearRoughness, NoV, NoL);
    vec3 F = F_Schlick(f0, LoH);
    vec3 Fr = (D * V) * F;

    vec3 Fd = diffuse * Fd_Burley(linearRoughness, NoV, NoL, LoH);

    return (Fd + Fr) * lc * NoL;
}

//------------------------------------------------------------------------------
// Atmosphere
//------------------------------------------------------------------------------

#define C_RAYLEIGH (vec3(5.802, 13.558, 33.100) * 1e-6)
#define C_MIE (vec3(3.996, 3.996, 3.996) * 1e-6)

float PhaseRayleigh(float costh)
{
    return 3.0 * (1.0 + costh * costh) / (16.0 * PI);
}

float PhaseMie(float costh, float g)
{
    g = min(g, 0.9381);
    float k = 1.55 * g - 0.55 * g * g * g;
    float kcosth = k * costh;
    return (1.0 - k * k) / ((4.0 * PI) * (1.0 - kcosth) * (1.0 - kcosth));
}

// ╔══════════════════════════════════════════════════════════════════════════════╗
// ║ 🪐 Buffer A                                                                  ║
// ╚══════════════════════════════════════════════════════════════════════════════╝
// Copyright 2020 Clay John

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software
// and associated documentation files (the "Software"), to deal in the Software without restriction,
// including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do
// so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or
// substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
// NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
// IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
// WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
// SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

/*
==========================================================================================

Buffer A generates the heightmap (X), normals (YZ) and erosion mask used for coloring (W)

==========================================================================================
*/

// code adapted from https://www.shadertoy.com/view/llsGWl
// name: Gavoronoise
// author: guil
// license: Creative Commons Attribution-NonCommercial-ShareAlike 3.0 Unported License
// Code has been modified to return analytic derivatives and to favour
// direction quite a bit.
vec3 Erosion(in vec2 p, vec2 dir)
{
    vec2 ip = floor(p);
    vec2 fp = fract(p);
    float f = 2.0 * PI;
    vec3 va = vec3(0.0);
    float wt = 0.0;
    for (int i = -2; i <= 1; i++)
    {
        for (int j = -2; j <= 1; j++)
        {
            vec2 o = vec2(i, j);
            vec2 h = hash(ip - o) * 0.5;
            vec2 pp = fp + o - h;
            float d = dot(pp, pp);
            float w = exp(-d * 2.0);
            wt += w;
            float mag = dot(pp, dir);
            va += vec3(cos(mag * f), -sin(mag * f) * (pp * 0.0 + dir)) * w;
        }
    }
    return va / wt;
}

vec2 Heightmap(vec2 uv, float2 iResolution, float iTime, float4 iMouse)
{
    vec2 p = uv * HEIGHT_TILES;

    // FBM terrain
    vec3 n = vec3(0.0);
    float nf = 1.0;
    float na = HEIGHT_AMP;
    for (int i = 0; i < HEIGHT_OCTAVES; i++)
    {
        n += noised(p * nf) * na * vec3(1.0, nf, nf);
        na *= HEIGHT_GAIN;
        nf *= HEIGHT_LACUNARITY;
    }

    // [-1, 1] -> [0, 1]
    n.x = n.x * 0.5 + 0.5;

    // Take the curl of the normal to get the gradient facing down the slope
    vec2 dir = n.zy * vec2(1.0, -1.0) * EROSION_SLOPE_STRENGTH;

    // Now we compute another fbm type noise
    // erosion is a type of noise with a strong directionality
    // we pass in the direction based on the slope of the terrain
    // erosion also returns the slope. we add that to a running total
    // so that the direction of successive layers are based on the
    // past layers
    vec3 h = vec3(0.0);

    float a = 0.5;
    float f = 1.0;

    // Smooth valleys
    // a *= (smoothstep(0.0, 1.0, n.x));

    a *= smoothstep(WATER_HEIGHT - 0.1, WATER_HEIGHT + 0.2, n.x);

    int octaves = EROSION_OCTAVES;



    for (int i = 0; i < octaves; i++)
    {
        h += Erosion(p * EROSION_TILES * f, dir + h.zy * vec2(1.0, -1.0) * EROSION_BRANCH_STRENGTH) * a * vec3(1.0, f, f);
        a *= EROSION_GAIN;
        ;
        f *= EROSION_LACUNARITY;
    }

    return vec2(n.x + (h.x - 0.5) * EROSION_STRENGTH, h.x);
}

vec4 SampleBufferA(vec2 uv, float2 iResolution, float iTime, float4 iMouse)
{
    vec2 uvSample = uv;
    // uvSample.x += TimeScrollOffset(iTime);

    vec2 h = Heightmap(uvSample, iResolution, iTime, iMouse);

    // Calculate an accurate normal from neighbouring points
    vec2 uv1 = uvSample + vec2(1.0, 0.0) / 512.0;
    vec2 uv2 = uvSample + vec2(0.0, 1.0) / 512.0;
    vec2 h1 = Heightmap(uv1, iResolution, iTime, iMouse);
    vec2 h2 = Heightmap(uv2, iResolution, iTime, iMouse);
    vec3 v1 = vec3(uv1 - uvSample, (h1.x - h.x));
    vec3 v2 = vec3(uv2 - uvSample, (h2.x - h.x));
    vec3 normal = normalize(cross(v1, v2)).xzy;

    return vec4(h.x, normal.xz, h.y);
}

vec4 SampleBufferB(vec2 uv, float2 iResolution, float iTime)
{
    float2 bufferSize = GetBufferSize(iResolution);
    vec2 uvSample = fract(uv);
    uvSample *= bufferSize / iResolution;
    // uvSample.x += TimeScrollOffset(iTime);

    vec3 color = vec3(0.0);

    float a = 0.5;
    float f = 2.0;
    for (int i = 0; i < 8; i++)
    {
        color += noised(uvSample * f) * a;
        a *= 0.95;
        f *= 2.0;
    }

    return vec4(color, 1.0);
}
// ╔══════════════════════════════════════════════════════════════════════════════╗
// ║ 🪐 Main                                                                      ║
// ╚══════════════════════════════════════════════════════════════════════════════╝

vec4 map(vec3 p, float2 iResolution, float iTime, float4 iMouse, out float erosion)
{
    float2 bufferSize = GetBufferSize(iResolution);
    vec2 uv = p.xz + vec2(0.5) - vec2(0.5) / bufferSize;
    vec2 uvSample = clamp(uv, vec2(0.001), vec2(0.999));
    uvSample *= bufferSize / iResolution;
    vec4 tex = SampleBufferA(uvSample, iResolution, iTime, iMouse);
    float height = tex.x;
    vec3 normal = tex.yzz;
    normal.y = sqrt(1.0 - dot(normal.xz, normal.xz)); // Recover Y

    erosion = tex.w;

    return vec4(height, normal);
}

// Ray marching
float march(vec3 ro, vec3 rd, float2 iResolution, float iTime, float4 iMouse, out vec3 normal, out int material, out float s_t)
{
    s_t = 9999.0;

    vec3 boxNormal;
    vec2 box = boxIntersection(ro, rd, vec3(0.5, 1.0, 0.5), boxNormal);

    if (box.y < 0.0)
    {
        return -1.0;
    }

    float tStart = max(0.0, box.x) + 1e-2;
    float tEnd = box.y - 1e-2;

    material = M_GROUND;

    float stepSize = 0.0;
    float stepScale = 1.0;
    float t = tStart;
    float altitude = 0.0;
    for (int i = 0; i < 32; i++)
    {
        vec3 pos = ro + rd * t;

        float foo;
        vec4 tex = map(pos, iResolution, iTime, iMouse, foo);
        float h = tex.x;
        normal = tex.yzw;

        altitude = pos.y - h;

        s_t = max(0.0, min(s_t, altitude / t));

        if (altitude < 0.0)
        {
            if (i < 1) // Sides
            {
                /*
                if (diff < -0.1) // Bottom
                {
                    return -1.0;
                }
                */
                if (pos.y < 0.35) // Flat bottom
                {
                    s_t = 9999.0;
                    return -1.0;
                }
                normal = boxNormal;
                material = M_STRATA;
                break;
            }
        }

        if (altitude < 0.0)
        {
            // Step back (contact/edge refinement)
            stepScale *= 0.5;
            t -= stepSize * stepScale;
        }
        else
        {
            // Step forward
            // Accelerate the ray by distance to terrain. This would result in horrible aliasing if we didn't do refinement above
            stepSize = abs(altitude) + 1e-2;
            // stepSize = (tEnd - tStart) / float(stepCount);
            t += stepSize * stepScale;
        }
    }

    if (t > tEnd)
    {
        s_t = 9999.0;
        return -1.0;
    }

#ifdef WATER
    vec3 waterNormal;
    vec2 water = boxIntersection(ro, rd, vec3(0.5, WATER_HEIGHT, 0.5), waterNormal);
    if ((water.y > 0.0 && (water.x < t || t < 0.0)) && material != M_STRATA)
    {
        t = max(0.0, water.x);
        normal = waterNormal;
        material = M_WATER;
    }
    else
#endif

        if (box.y < 0.0)
    {
        return -1.0;
    }

    return t;
}

vec3 GetReflection(vec3 p, vec3 r, vec3 sun, float smoothness, float2 iResolution, float iTime, float4 iMouse)
{
    vec3 refl = SkyColor(r, sun) * 4.0;

    vec3 foo;
    float r_t;
    int r_material;
    march(p, r, iResolution, iTime, iMouse, foo, r_material, r_t);
    return refl * (1.0 - exp(-r_t * 10.0 * sq(smoothness)));
}

// Main image output
void mainImage(out vec4 fragColor, in vec2 fragCoord, float2 iResolution, float iTime, float4 iMouse)
{
#ifdef SHOW_BUFFER
    float debugWidth = iResolution.y / 2.0;
#else
    float debugWidth = 0.0;
#endif

    // ==========================================================================================
    // Set up camera
    // ==========================================================================================
    vec2 cameraAngle = vec2(PI, -0.25 * PI);
    float cameraDistance = 5.0;

    vec3 ro = vec3(0.0, 0.325, 0.0);
    vec3 rd = CameraRay(11.0, iResolution.xy, fragCoord.xy - vec2(debugWidth / 2.0, 0.0));

    mat3 rot = CameraRotation(cameraAngle.yx);
    rd = mul(rot, rd);
    ro = mul(rot, ro) + mul(rot, vec3(0, 0, cameraDistance));

    // ==========================================================================================
    // Ray march
    // ==========================================================================================
    vec4 foo;
    vec3 normal;
    int material;
    float t = march(ro, rd, iResolution, iTime, iMouse, normal, material, foo.w);

    // ==========================================================================================
    // Shade
    // ==========================================================================================
#ifdef FIXED_SUN
    vec3 sun = normalize(vec3(-1.0, 0.4, 0.05));
#else
    vec3 sun = mul(rot, normalize(vec3(-1.0, 0.1, 0.25)));
#endif

    vec3 fogColor = 1.0 - exp(-SkyColor(rd, sun) * 2.0);

    vec3 color;

    if (t < 0.0)
    {
        // Sky
        color = fogColor * (1.0 + pow(fragCoord.y / iResolution.y, 3.0) * 3.0) * 0.5;
#ifdef SHOW_NORMALS
        color = vec3(0.5, 0.5, 1.0);
#endif
    }
    else
    {
        vec3 pos = ro + rd * t;

        float erosion;
        // float h = map(pos, erosion).x;
        float diff = pos.y - map(pos, iResolution, iTime, iMouse, erosion).x;

        vec4 breakupTex = vec4(0.0);

#ifdef DETAIL_TEXTURE
        breakupTex = SampleBufferB(pos.xz + 0.5, iResolution, iTime);
        vec3 breakupNormal = breakupTex.zyw;
        if (material != M_STRATA)
        {
            normal = normalize(normal + breakupNormal.xzy * 0.1);
        }
#endif
        float breakup = breakupTex.x;

        vec3 f0 = vec3(0.04);
        float smoothness = 0.0;
        float reflAmount = 0.0;
        float occlusion = 1.0;

        vec3 r = reflect(rd, normal);

        vec3 diffuseColor = vec3(0.5);
        if (material == M_GROUND)
        {
#ifndef GREYSCALE
            occlusion = sq(saturate(erosion + 0.5));

            // Cliffs / Dirt
            diffuseColor = CLIFF_COLOR * smoothstep(0.4, 0.52, pos.y);
            // diffuseColor = mix(diffuseColor, DIRT_COLOR, smoothstep(0.8, 0.9, normal.y - erosion * 0.1 + breakup * 0.05));
            diffuseColor = lerp(diffuseColor, DIRT_COLOR, smoothstep(0.3, 0.0, occlusion + breakup * 1.0));

            // Grass
            vec3 grassMix = lerp(GRASS_COLOR1, GRASS_COLOR2, smoothstep(0.4, 0.6, pos.y - erosion * 0.05 + breakup * 0.3));
            // diffuseColor = mix(diffuseColor, grassMix, smoothstep(0.8, 0.95, normal.y - erosion * 0.2 + breakup * 0.1));
            diffuseColor = lerp(diffuseColor, grassMix, smoothstep(WATER_HEIGHT + 0.05, WATER_HEIGHT + 0.02, pos.y - breakup * 0.02) * smoothstep(0.8, 1.0, normal.y + breakup * 0.1));

            // Snow
            diffuseColor = lerp(diffuseColor, vec3(1.0), smoothstep(0.53, 0.6, pos.y + breakup * 0.1));
#ifdef WATER
            // Sand (beach)
            diffuseColor = lerp(diffuseColor, SAND_COLOR, smoothstep(WATER_HEIGHT + 0.005, WATER_HEIGHT, pos.y + breakup * 0.01));
#endif
            diffuseColor *= 1.0 + breakup * 0.5;
#endif
        }
        else if (material == M_STRATA)
        {
#ifndef GREYSCALE
            vec3 strata = smoothstep(0.0, 1.0, cos(diff * vec3(130.0, 190.0, 250.0)));
            diffuseColor = vec3(0.3);
            diffuseColor = lerp(diffuseColor, vec3(0.50), strata.x);
            diffuseColor = lerp(diffuseColor, vec3(0.55), strata.y);
            diffuseColor = lerp(diffuseColor, vec3(0.60), strata.z);

            diffuseColor *= exp(diff * 10.0) * vec3(1.0, 0.9, 0.7);
#endif
        }
        else if (material == M_WATER)
        {
            float shore = normal.y > 1e-2 ? exp(-diff * 60.0) : 0.0;
            float foam = normal.y > 1e-2 ? smoothstep(0.005, 0.0, diff + breakup * 0.005) : 0.0;

            diffuseColor = lerp(WATER_COLOR, WATER_SHORE_COLOR, shore);

            diffuseColor = lerp(diffuseColor, vec3(1.0), foam);

            // f0 = vec3(0.2);
            smoothness = 0.95;
        }

        float shadow = 1.0;

#ifdef SHADOWS
        if (material != M_STRATA)
        {
            // Shadow ray
            float s_t;
            int s_material;
            march(pos + vec3(0.0, 1.0, 0.0) * 1e-4, sun, iResolution, iTime, iMouse, foo.xyz, s_material, s_t);
            shadow = 1.0 - exp(-s_t * 20.0);
            // fragColor = vec4(shadow, shadow, shadow, 1.0);
            // return;
        }
#endif

        // Ambient
        color = diffuseColor * SkyColor(normal, sun) * occlusion * Fd_Lambert();
        // Direct
        color += Shade(diffuseColor, f0, smoothness, normal, -rd, sun, SUN_COLOR * shadow);
        // Bounce
        color += diffuseColor * SUN_COLOR * (dot(normal, sun * vec3(1.0, -1.0, 1.0)) * 0.5 + 0.5) * Fd_Lambert() / PI;
        // Reflection
        color += GetReflection(pos, r, sun, smoothness, iResolution, iTime, iMouse) * F_Schlick(f0, dot(-rd, normal));
        // Fog
        float fog = exp(-t * t * smoothstep(WATER_HEIGHT, WATER_HEIGHT - 0.5, pos.y) * 0.5);
        // color = mix(fogColor, color, fog);

        // color = vec3(exp(-max(0.0, -erosion * 2.0)));
        // color = vec3(occlusion);

#ifdef SHOW_DIFFUSE
        color = pow(diffuseColor, vec3(1.0 / 2.2));
#elif defined(SHOW_NORMALS)
        color = normal.xzy * 0.5 + 0.5;
#endif
    }

    vec3 boxNormal;
    vec2 box = boxIntersection(ro, rd, vec3(0.5, 1.0, 0.5), boxNormal);

    float costh = dot(rd, sun);
    float phaseR = PhaseRayleigh(costh);
    float phaseM = PhaseMie(costh, 0.6);

    vec2 od = vec2(0.0);
    vec3 tsm;
    vec3 sct = vec3(0.0);
    float rayLength = (t > 0.0 ? t : box.y) - box.x;
    float stepSize = rayLength / 16.0;
    for (float i = 0.0; i < 16.0; i++)
    {
        vec3 p = ro + rd * (box.x + (i + 0.5) * stepSize);

        float h = max(0.0, p.y - 0.35);
        // float d = exp(-h * 5.0);
        float d = 1.0 - saturate(h / 0.2);

        if (p.y < 0.35)
        {
            d = 0.0;
        }

        float densityR = d * 1e5;
        float densityM = d * 1e5;

        od += stepSize * vec2(densityR, densityM);

        tsm = exp(-(od.x * C_RAYLEIGH + od.y * C_MIE));

        sct += tsm * C_RAYLEIGH * phaseR * densityR * stepSize;
        sct += tsm * C_MIE * phaseM * densityM * stepSize;
    }

    color = color * tsm + sct * 10.0;

#if !defined(SHOW_NORMALS) && !defined(SHOW_DIFFUSE)
    color = Tonemap_ACES(color);
    color = pow(color, vec3(1.0 / 2.2));
#endif

    // Dither
    color += hash(fragCoord.xy).x / 255.0;

#ifdef SHOW_BUFFER
    vec2 debugUV = fragCoord / debugWidth;
    if (debugUV.x > 0.0 && debugUV.x < 1.0 && debugUV.y > 0.0 && debugUV.y < 2.0)
    {
        float2 bufferSize = GetBufferSize(iResolution);
        vec2 uvSample = clamp(fract(debugUV), vec2(0.001), vec2(0.999));
        uvSample *= bufferSize / iResolution;
        vec4 tex = SampleBufferA(uvSample, iResolution, iTime, iMouse);
        vec3 normal = tex.yzz;
        normal.y = sqrt(1.0 - dot(normal.xz, normal.xz));
        color = debugUV.y > 1.0 ? normal.xzy * 0.5 + 0.5 : tex.xxx * 2.5 - 0.75;
    }
#endif

    fragColor = vec4(color, 1.0);
}

[shader("compute")]
[numthreads(16, 16, 1)]
[playground::CALL::SIZE_OF("outputTexture")]
void imageMain(uint2 dispatchThreadID: SV_DispatchThreadID)
{
    drawPixel(dispatchThreadID, (int2 screenSize) =>
                                {
                                    float2 iResolution = float2(screenSize);
                                    float iTime = timeFromStart;
                                    float4 iMouse = ToShadertoyMouse(_mousePosition);
                                    vec4 color = vec4(0.0);
                                    mainImage(color, float2(dispatchThreadID), iResolution, iTime, iMouse);
                                    return color;
                                });
}
