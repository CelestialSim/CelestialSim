module terrain_noise_3d;

// 3D gradient noise utilities and Terrain3D-style terrain helpers.
// Based on: https://iquilezles.org/articles/gradientnoise/

static const float PI = 3.14159265358979;

static const float HEIGHT_TILES = 3.0;
static const int HEIGHT_OCTAVES = 3;
static const float HEIGHT_AMP = 0.25;
static const float HEIGHT_GAIN = 0.1;
static const float HEIGHT_LACUNARITY = 2.0;

static const float EROSION_TILES = 4.0;
static const int EROSION_OCTAVES = 5;
static const float EROSION_GAIN = 0.5;
static const float EROSION_LACUNARITY = 1.8;
static const float EROSION_SLOPE_STRENGTH = 3.0;
static const float EROSION_BRANCH_STRENGTH = 3.0;
static const float EROSION_STRENGTH = 0.04;

static const float WATER_HEIGHT = 0.45;

// Runtime-configurable terrain parameters.
// Passed via ConstantBuffer<TerrainParams> from the CPU side.
public struct TerrainParams
{
    float height_tiles;
    int height_octaves;
    float height_amp;
    float height_gain;
    float height_lacunarity;
    float erosion_tiles;
    int erosion_octaves;
    float erosion_gain;
    float erosion_lacunarity;
    float erosion_slope_strength;
    float erosion_branch_strength;
    float erosion_strength;
    float water_height;
};

public TerrainParams default_terrain_params()
{
    TerrainParams p;
    p.height_tiles = HEIGHT_TILES;
    p.height_octaves = HEIGHT_OCTAVES;
    p.height_amp = HEIGHT_AMP;
    p.height_gain = HEIGHT_GAIN;
    p.height_lacunarity = HEIGHT_LACUNARITY;
    p.erosion_tiles = EROSION_TILES;
    p.erosion_octaves = EROSION_OCTAVES;
    p.erosion_gain = EROSION_GAIN;
    p.erosion_lacunarity = EROSION_LACUNARITY;
    p.erosion_slope_strength = EROSION_SLOPE_STRENGTH;
    p.erosion_branch_strength = EROSION_BRANCH_STRENGTH;
    p.erosion_strength = EROSION_STRENGTH;
    p.water_height = WATER_HEIGHT;
    return p;
}

static const float3 CLIFF_COLOR = float3(0.22, 0.2, 0.2);
static const float3 DIRT_COLOR = float3(0.6, 0.5, 0.4);
static const float3 GRASS_COLOR1 = float3(0.15, 0.3, 0.1);
static const float3 GRASS_COLOR2 = float3(0.4, 0.5, 0.2);
static const float3 SAND_COLOR = float3(0.8, 0.7, 0.6);
static const float3 WATER_COLOR = float3(0.10, 0.38, 0.65);
static const float3 WATER_SHORE_COLOR = float3(0.45, 0.72, 0.86);

public float saturate(float x)
{
    return clamp(x, 0.0, 1.0);
}

public float sq(float x)
{
    return x * x;
}

public float3 hash3(float3 p)
{
    p = float3(
        dot(p, float3(127.1, 311.7, 74.7)),
        dot(p, float3(269.5, 183.3, 246.1)),
        dot(p, float3(113.5, 271.9, 124.6)));
    return -1.0 + 2.0 * fract(sin(p) * 43758.5453123);
}

public float2 hash2(float2 x)
{
    const float2 k = float2(0.3183099, 0.3678794);
    x = x * k + k.yx;
    return -1.0 + 2.0 * fract(16.0 * k * fract(x.x * x.y * (x.x + x.y)));
}

public float4 noised3d(float3 x)
{
    float3 i = floor(x);
    float3 f = fract(x);

    float3 u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
    float3 du = 30.0 * f * f * (f * (f - 2.0) + 1.0);

    float3 ga = hash3(i + float3(0.0, 0.0, 0.0));
    float3 gb = hash3(i + float3(1.0, 0.0, 0.0));
    float3 gc = hash3(i + float3(0.0, 1.0, 0.0));
    float3 gd = hash3(i + float3(1.0, 1.0, 0.0));
    float3 ge = hash3(i + float3(0.0, 0.0, 1.0));
    float3 gf = hash3(i + float3(1.0, 0.0, 1.0));
    float3 gg = hash3(i + float3(0.0, 1.0, 1.0));
    float3 gh = hash3(i + float3(1.0, 1.0, 1.0));

    float va = dot(ga, f - float3(0.0, 0.0, 0.0));
    float vb = dot(gb, f - float3(1.0, 0.0, 0.0));
    float vc = dot(gc, f - float3(0.0, 1.0, 0.0));
    float vd = dot(gd, f - float3(1.0, 1.0, 0.0));
    float ve = dot(ge, f - float3(0.0, 0.0, 1.0));
    float vf = dot(gf, f - float3(1.0, 0.0, 1.0));
    float vg = dot(gg, f - float3(0.0, 1.0, 1.0));
    float vh = dot(gh, f - float3(1.0, 1.0, 1.0));

    float v = va +
              u.x * (vb - va) +
              u.y * (vc - va) +
              u.z * (ve - va) +
              u.x * u.y * (va - vb - vc + vd) +
              u.y * u.z * (va - vc - ve + vg) +
              u.z * u.x * (va - vb - ve + vf) +
              u.x * u.y * u.z * (-va + vb + vc - vd + ve - vf - vg + vh);

    float3 d = ga +
               u.x * (gb - ga) +
               u.y * (gc - ga) +
               u.z * (ge - ga) +
               u.x * u.y * (ga - gb - gc + gd) +
               u.y * u.z * (ga - gc - ge + gg) +
               u.z * u.x * (ga - gb - ge + gf) +
               u.x * u.y * u.z * (-ga + gb + gc - gd + ge - gf - gg + gh) +
               du * (float3(vb - va, vc - va, ve - va) +
                     u.yzx * float3(va - vb - vc + vd, va - vc - ve + vg, va - vb - ve + vf) +
                     u.zxy * float3(va - vb - ve + vf, va - vb - vc + vd, va - vc - ve + vg) +
                     u.yzx * u.zxy * (-va + vb + vc - vd + ve - vf - vg + vh));

    return float4(v, d);
}

public float fbm3d(float3 p)
{
    float value = 0.0;
    float amplitude = 0.5;
    float frequency = 1.0;

    for (int i = 0; i < 4; i++)
    {
        float4 n = noised3d(p * frequency);
        value += n.x * amplitude;
        amplitude *= 0.5;
        frequency *= 2.0;
    }

    return value;
}

public float3 noised2(float2 p)
{
    float4 n = noised3d(float3(p.x, 0.0, p.y));
    return float3(n.x, n.y, n.w);
}

public float3 erosion(float2 p, float2 dir)
{
    float2 ip = floor(p);
    float2 fp = fract(p);
    float f = 2.0 * PI;
    float3 va = float3(0.0, 0.0, 0.0);
    float wt = 0.0;

    for (int i = -2; i <= 1; i++)
    {
        for (int j = -2; j <= 1; j++)
        {
            float2 o = float2(i, j);
            float2 h = hash2(ip - o) * 0.5;
            float2 pp = fp + o - h;
            float d = dot(pp, pp);
            float w = exp(-d * 2.0);
            wt += w;
            float mag = dot(pp, dir);
            va += float3(cos(mag * f), -sin(mag * f) * (pp + dir)) * w;
        }
    }

    return va / wt;
}

// Returns float2(height, erosionMask) with same structure as Terrain3D Heightmap().
public float2 terrain_height_erosion(float2 uv)
{
    float2 p = uv * HEIGHT_TILES;

    float3 n = float3(0.0, 0.0, 0.0);
    float nf = 1.0;
    float na = HEIGHT_AMP;
    for (int i = 0; i < HEIGHT_OCTAVES; i++)
    {
        n += noised2(p * nf) * na * float3(1.0, nf, nf);
        na *= HEIGHT_GAIN;
        nf *= HEIGHT_LACUNARITY;
    }

    n.x = n.x * 0.5 + 0.5;

    float2 dir = n.zy * float2(1.0, -1.0) * EROSION_SLOPE_STRENGTH;

    float3 h = float3(0.0, 0.0, 0.0);
    float a = 0.5;
    float f = 1.0;

    a *= smoothstep(WATER_HEIGHT - 0.1, WATER_HEIGHT + 0.2, n.x);

    for (int i = 0; i < EROSION_OCTAVES; i++)
    {
        h += erosion(p * EROSION_TILES * f, dir + h.zy * float2(1.0, -1.0) * EROSION_BRANCH_STRENGTH) * a * float3(1.0, f, f);
        a *= EROSION_GAIN;
        f *= EROSION_LACUNARITY;
    }

    return float2(n.x + (h.x - 0.5) * EROSION_STRENGTH, h.x);
}

public float terrain_breakup(float2 uv)
{
    float3 color = float3(0.0, 0.0, 0.0);
    float a = 0.5;
    float f = 2.0;

    for (int i = 0; i < 8; i++)
    {
        color += noised2(uv * f) * a;
        a *= 0.95;
        f *= 2.0;
    }

    return color.x;
}

public float2 terrain_height_erosion_triplanar(float3 normal_dir)
{
    float3 w = abs(normal_dir);
    w = pow(w, 4.0);
    w /= max(w.x + w.y + w.z, 1e-5);

    float2 hx = terrain_height_erosion(normal_dir.yz * 0.5 + 0.5);
    float2 hy = terrain_height_erosion(normal_dir.xz * 0.5 + 0.5);
    float2 hz = terrain_height_erosion(normal_dir.xy * 0.5 + 0.5);

    return hx * w.x + hy * w.y + hz * w.z;
}

public float terrain_breakup_triplanar(float3 normal_dir)
{
    float3 w = abs(normal_dir);
    w = pow(w, 4.0);
    w /= max(w.x + w.y + w.z, 1e-5);

    float bx = terrain_breakup(normal_dir.yz * 0.5 + 0.5);
    float by = terrain_breakup(normal_dir.xz * 0.5 + 0.5);
    float bz = terrain_breakup(normal_dir.xy * 0.5 + 0.5);

    return bx * w.x + by * w.y + bz * w.z;
}

// Terrain3D-inspired diffuse color from height/erosion/breakup only.
public float3 terrain_albedo(float height, float erosion_mask, float breakup)
{
    return terrain_albedo_slope(height, erosion_mask, breakup, 1.0);
}

// Terrain3D-inspired diffuse color with slope-dependent grass.
// normal_y is the Y (up) component of the surface normal (1.0 = flat, 0.0 = vertical cliff).
public float3 terrain_albedo_slope(float height, float erosion_mask, float breakup, float normal_y)
{
    float water_mask = 1.0 - smoothstep(WATER_HEIGHT - 0.015, WATER_HEIGHT + 0.015, height);
    float shore_mask = smoothstep(WATER_HEIGHT - 0.03, WATER_HEIGHT + 0.01, height) * (1.0 - water_mask);

    float occlusion = sq(saturate(erosion_mask + 0.5));

    float3 diffuse = CLIFF_COLOR * smoothstep(0.4, 0.52, height);
    diffuse = lerp(diffuse, DIRT_COLOR, smoothstep(0.3, 0.0, occlusion + breakup));

    float3 grass_mix = lerp(
        GRASS_COLOR1,
        GRASS_COLOR2,
        smoothstep(0.4, 0.6, height - erosion_mask * 0.05 + breakup * 0.3));
    // Grass only on relatively flat areas (normal_y close to 1.0), matching Terrain3D
    diffuse = lerp(diffuse, grass_mix,
                   smoothstep(WATER_HEIGHT + 0.05, WATER_HEIGHT + 0.02, height - breakup * 0.02) * smoothstep(0.8, 1.0, normal_y + breakup * 0.1));

    diffuse = lerp(diffuse, float3(1.0, 1.0, 1.0), smoothstep(0.53, 0.6, height + breakup * 0.1));
    diffuse = lerp(diffuse, SAND_COLOR, smoothstep(WATER_HEIGHT + 0.005, WATER_HEIGHT, height + breakup * 0.01));

    float3 water = lerp(WATER_COLOR, WATER_SHORE_COLOR, shore_mask);
    diffuse = lerp(diffuse, water, water_mask);

    diffuse *= 1.0 + breakup * 0.5;
    return diffuse;
}

// ── Overloads accepting TerrainParams ───────────────────────────────────

public float2 terrain_height_erosion(float2 uv, TerrainParams p)
{
    float2 pos = uv * p.height_tiles;

    float3 n = float3(0.0, 0.0, 0.0);
    float nf = 1.0;
    float na = p.height_amp;
    for (int i = 0; i < p.height_octaves; i++)
    {
        n += noised2(pos * nf) * na * float3(1.0, nf, nf);
        na *= p.height_gain;
        nf *= p.height_lacunarity;
    }

    n.x = n.x * 0.5 + 0.5;

    float2 dir = n.zy * float2(1.0, -1.0) * p.erosion_slope_strength;

    float3 h = float3(0.0, 0.0, 0.0);
    float a = 0.5;
    float f = 1.0;

    a *= smoothstep(p.water_height - 0.1, p.water_height + 0.2, n.x);

    for (int i = 0; i < p.erosion_octaves; i++)
    {
        h += erosion(pos * p.erosion_tiles * f, dir + h.zy * float2(1.0, -1.0) * p.erosion_branch_strength) * a * float3(1.0, f, f);
        a *= p.erosion_gain;
        f *= p.erosion_lacunarity;
    }

    return float2(n.x + (h.x - 0.5) * p.erosion_strength, h.x);
}

public float2 terrain_height_erosion_triplanar(float3 normal_dir, TerrainParams p)
{
    float3 w = abs(normal_dir);
    w = pow(w, 4.0);
    w /= max(w.x + w.y + w.z, 1e-5);

    float2 hx = terrain_height_erosion(normal_dir.yz * 0.5 + 0.5, p);
    float2 hy = terrain_height_erosion(normal_dir.xz * 0.5 + 0.5, p);
    float2 hz = terrain_height_erosion(normal_dir.xy * 0.5 + 0.5, p);

    return hx * w.x + hy * w.y + hz * w.z;
}

public float3 terrain_albedo_slope(float height, float erosion_mask, float breakup, float normal_y, TerrainParams p)
{
    float water_mask = 1.0 - smoothstep(p.water_height - 0.015, p.water_height + 0.015, height);
    float shore_mask = smoothstep(p.water_height - 0.03, p.water_height + 0.01, height) * (1.0 - water_mask);

    float occlusion = sq(saturate(erosion_mask + 0.5));

    float3 diffuse = CLIFF_COLOR * smoothstep(0.4, 0.52, height);
    diffuse = lerp(diffuse, DIRT_COLOR, smoothstep(0.3, 0.0, occlusion + breakup));

    float3 grass_mix = lerp(
        GRASS_COLOR1,
        GRASS_COLOR2,
        smoothstep(0.4, 0.6, height - erosion_mask * 0.05 + breakup * 0.3));
    diffuse = lerp(diffuse, grass_mix,
                   smoothstep(p.water_height + 0.05, p.water_height + 0.02, height - breakup * 0.02) * smoothstep(0.8, 1.0, normal_y + breakup * 0.1));

    diffuse = lerp(diffuse, float3(1.0, 1.0, 1.0), smoothstep(0.53, 0.6, height + breakup * 0.1));
    diffuse = lerp(diffuse, SAND_COLOR, smoothstep(p.water_height + 0.005, p.water_height, height + breakup * 0.01));

    float3 water = lerp(WATER_COLOR, WATER_SHORE_COLOR, shore_mask);
    diffuse = lerp(diffuse, water, water_mask);

    diffuse *= 1.0 + breakup * 0.5;
    return diffuse;
}

public float3 terrain_albedo(float height, float erosion_mask, float breakup, TerrainParams p)
{
    return terrain_albedo_slope(height, erosion_mask, breakup, 1.0, p);
}
