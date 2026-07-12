module terrain_noise_3d;

// 3D noise utilities and terrain helpers.
//
// Third-party material in this module (see THIRD_PARTY_NOTICES.md):
//   - Gradient noise (`hash3`, `noised3d`): Inigo Quilez, MIT —
//     https://iquilezles.org/articles/gradientnoise/
//   - Simplex noise (`snoise` + helpers): Ashima Arts webgl-noise, MIT,
//     (c) 2011 Ashima Arts (via Keijiro Takahashi's HLSL translation) —
//     https://github.com/ashima/webgl-noise
//   - Elevation composition (`terrain_height` + its helpers): derived from
//     Sebastian Lague's Solar-System `EarthHeight.compute` /
//     `FractalNoise.cginc` / `Math.cginc`, MIT, (c) 2020 Sebastian Lague —
//     https://github.com/SebLague/Solar-System
//   - Land colour ramp stops (`LAND_POS` / `LAND_COLOR` structure): from the
//     terrain colormap in Daniel Andrino's terrain-erosion-3-ways (`util.py`,
//     `_TERRAIN_CMAP`), MIT, (c) 2018 Daniel Andrino; colour values modified —
//     https://github.com/dandrino/terrain-erosion-3-ways

// Default knobs, tuned for the Lague-style composition below (frequencies are
// cycles across the unit sphere, so ~1-2 gives planet-scale continents).
static const float FREQUENCY = 1.4;
static const int HEIGHT_OCTAVES = 8;
static const float HEIGHT_AMP = 0.35;
static const float HEIGHT_GAIN = 0.396;
static const float HEIGHT_LACUNARITY = 2.0;

static const float RIDGE_TILES = 2.4;
static const int RIDGE_OCTAVES = 5;
static const float RIDGE_GAIN = 0.5;
static const float RIDGE_LACUNARITY = 2.0;
// Mountain relief per unit of ridge_strength — matches the CPU provider's
// RIDGE_RELIEF so the same value gives the same relief on both routes.
static const float RIDGE_RELIEF = 3.2;
static const float RIDGE_STRENGTH = 0.0552;

static const float WATER_HEIGHT = 0.549;

// Runtime-configurable terrain parameters, unpacked from the 56-byte
// `TerrainGpu` block (`descriptors.rs`) by each shader's `to_terrain_params`.
// The `ridge_*` knobs drive the ridged-fBm mountain detail.
public struct TerrainParams
{
    public float frequency;
    public int height_octaves;
    public float height_amp;
    public float height_gain;
    public float height_lacunarity;
    public float ridge_tiles;
    public int ridge_octaves;
    public float ridge_gain;
    public float ridge_lacunarity;
    public float ridge_strength;
    public float water_height;
};

public TerrainParams default_terrain_params()
{
    TerrainParams p;
    p.frequency = FREQUENCY;
    p.height_octaves = HEIGHT_OCTAVES;
    p.height_amp = HEIGHT_AMP;
    p.height_gain = HEIGHT_GAIN;
    p.height_lacunarity = HEIGHT_LACUNARITY;
    p.ridge_tiles = RIDGE_TILES;
    p.ridge_octaves = RIDGE_OCTAVES;
    p.ridge_gain = RIDGE_GAIN;
    p.ridge_lacunarity = RIDGE_LACUNARITY;
    p.ridge_strength = RIDGE_STRENGTH;
    p.water_height = WATER_HEIGHT;
    return p;
}

// Land colour ramp: stop positions and band structure from Daniel Andrino's
// terrain colormap (terrain-erosion-3-ways `util.py`, `_TERRAIN_CMAP`, MIT,
// (c) 2018 Daniel Andrino); colour values modified. `land_ramp` evaluates it
// the way matplotlib's LinearSegmentedColormap does: piecewise-linear between
// (position, colour) stops.
// Alpine bands: bright sunlit meadow -> mid green -> dark forest green ->
// warm grey rock -> warm snow/scree.
static const int LAND_STOPS = 5;
static const float LAND_POS[LAND_STOPS] = { 0.00, 0.25, 0.50, 0.80, 1.00 };
static const float3 LAND_COLOR[LAND_STOPS] = {
    float3(0.40, 0.54, 0.19),
    float3(0.24, 0.42, 0.15),
    float3(0.13, 0.26, 0.11),
    float3(0.56, 0.53, 0.47),
    float3(0.97, 0.95, 0.90),
};

// Submerged ground. The analytic ocean shader supplies ALL the blue (it tints
// whatever is seen through it with a Beer-Lambert depth falloff), so the seabed
// albedo must be sand, not water: a pale, slightly warm lagoon sand right at the
// shore, fading to a cooler/darker grey-tan as the bed drops away. The thin dry
// beach reuses the shore colour so it reads as the same material continuing
// under water.
static const float3 SEABED_SHORE_COLOR = float3(0.88, 0.79, 0.62);
static const float3 SEABED_DEEP_COLOR = float3(0.53, 0.50, 0.43);

float3 land_ramp(float t)
{
    t = clamp(t, 0.0, 1.0);
    float3 c = LAND_COLOR[0];
    [unroll]
    for (int i = 1; i < LAND_STOPS; i++)
    {
        float s = clamp((t - LAND_POS[i - 1]) / (LAND_POS[i] - LAND_POS[i - 1]), 0.0, 1.0);
        c = lerp(c, LAND_COLOR[i], s);
    }
    return c;
}

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

// Inigo Quilez gradient noise (MIT). Returns (value, gradient.xyz).
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

public float3 noised2(float2 p)
{
    float4 n = noised3d(float3(p.x, 0.0, p.y));
    return float3(n.x, n.y, n.w);
}

// ---------------------------------------------------------------------------
// 3D simplex noise — Ashima Arts webgl-noise (MIT, (c) 2011 Ashima Arts,
// https://github.com/ashima/webgl-noise), via Keijiro Takahashi's HLSL
// translation (https://github.com/keijiro/NoiseShader) as vendored by
// Sebastian Lague's Solar-System. Used as the primitive for the terrain
// elevation: unlike lattice gradient noise, simplex has no axis-aligned cell
// grid, so finite-difference normals over it don't pick up grid creases.
// ---------------------------------------------------------------------------

float3 mod289(float3 x)
{
    return x - floor(x / 289.0) * 289.0;
}

float4 mod289(float4 x)
{
    return x - floor(x / 289.0) * 289.0;
}

float4 permute(float4 x)
{
    return mod289((x * 34.0 + 1.0) * x);
}

float4 taylor_inv_sqrt(float4 r)
{
    return 1.79284291400159 - r * 0.85373472095314;
}

public float snoise(float3 v)
{
    const float2 C = float2(1.0 / 6.0, 1.0 / 3.0);

    // First corner
    float3 i = floor(v + dot(v, C.yyy));
    float3 x0 = v - i + dot(i, C.xxx);

    // Other corners
    float3 g = step(x0.yzx, x0.xyz);
    float3 l = 1.0 - g;
    float3 i1 = min(g.xyz, l.zxy);
    float3 i2 = max(g.xyz, l.zxy);

    float3 x1 = x0 - i1 + C.xxx;
    float3 x2 = x0 - i2 + C.yyy;
    float3 x3 = x0 - 0.5;

    // Permutations
    i = mod289(i); // Avoid truncation effects in permutation
    float4 p =
        permute(permute(permute(i.z + float4(0.0, i1.z, i2.z, 1.0))
                              + i.y + float4(0.0, i1.y, i2.y, 1.0))
                              + i.x + float4(0.0, i1.x, i2.x, 1.0));

    // Gradients: 7x7 points over a square, mapped onto an octahedron.
    // The ring size 17*17 = 289 is close to a multiple of 49 (49*6 = 294)
    float4 j = p - 49.0 * floor(p / 49.0); // mod(p,7*7)

    float4 x_ = floor(j / 7.0);
    float4 y_ = floor(j - 7.0 * x_); // mod(j,N)

    float4 x = (x_ * 2.0 + 0.5) / 7.0 - 1.0;
    float4 y = (y_ * 2.0 + 0.5) / 7.0 - 1.0;

    float4 h = 1.0 - abs(x) - abs(y);

    float4 b0 = float4(x.xy, y.xy);
    float4 b1 = float4(x.zw, y.zw);

    float4 s0 = floor(b0) * 2.0 + 1.0;
    float4 s1 = floor(b1) * 2.0 + 1.0;
    float4 sh = -step(h, float4(0.0, 0.0, 0.0, 0.0));

    float4 a0 = b0.xzyw + s0.xzyw * sh.xxyy;
    float4 a1 = b1.xzyw + s1.xzyw * sh.zzww;

    float3 g0 = float3(a0.xy, h.x);
    float3 g1 = float3(a0.zw, h.y);
    float3 g2 = float3(a1.xy, h.z);
    float3 g3 = float3(a1.zw, h.w);

    // Normalise gradients
    float4 norm = taylor_inv_sqrt(float4(dot(g0, g0), dot(g1, g1), dot(g2, g2), dot(g3, g3)));
    g0 *= norm.x;
    g1 *= norm.y;
    g2 *= norm.z;
    g3 *= norm.w;

    // Mix final noise value
    float4 m = max(0.6 - float4(dot(x0, x0), dot(x1, x1), dot(x2, x2), dot(x3, x3)), 0.0);
    m = m * m;
    m = m * m;

    float4 px = float4(dot(x0, g0), dot(x1, g1), dot(x2, g2), dot(x3, g3));
    return 42.0 * dot(m, px);
}

// ---------------------------------------------------------------------------
// Terrain elevation — composition derived from Sebastian Lague's Solar-System
// `EarthHeight.compute` + `FractalNoise.cginc` (MIT, (c) 2020 Sebastian Lague,
// https://github.com/SebLague/Solar-System): continent fBm with a smooth-max
// flattened ocean floor and deepened basins, plus weight-compounded ridged
// mountains smoothed by a 5-tap tangent average and gated by a low-frequency
// mask so ranges form in belts instead of everywhere. The noise primitive is
// the Ashima simplex `snoise` above, as in Lague's upstream.
// ---------------------------------------------------------------------------

// Shaping constants (normalized-height units).
static const float OCEAN_FLOOR_DEPTH = 0.09;   // abyssal plain below LAND_CENTER
static const float OCEAN_FLOOR_SMOOTHING = 0.04;
static const float OCEAN_DEPTH_MULT = 1.5;     // extra deepening below the shelf
static const float RIDGE_POWER = 3.0;          // crest sharpening exponent
// 5-tap smoothing radius in NOISE-SPACE cycles: the world-space tap spacing is
// RIDGE_SMOOTH_OFFSET / ridge_tiles, so it shrinks with the feature size. A
// FIXED spacing comb-filters the ridged noise once its wavelength approaches
// the spacing (raising ridge_tiles produced parallel stripes along the tap
// axes); scaling it keeps the filter a constant fraction of a feature.
static const float RIDGE_SMOOTH_OFFSET = 0.05;
static const float MASK_SCALE = 1.3;           // mountain-belt mask frequency
static const int MASK_OCTAVES = 3;
static const float MOUNTAIN_BLEND = 0.6;       // mask edge softness

// Smooth maximum (IQ's smin family, https://iquilezles.org/articles/smin/,
// in the polynomial form used by Lague's Math.cginc).
float smooth_max(float a, float b, float k)
{
    k = min(0.0, -k);
    float h = clamp((b - a + k) / (2.0 * k), 0.0, 1.0);
    return a * h + b * (1.0 - h) - k * h * (1.0 - h);
}

// Plain fBm of the 3D simplex noise.
float fbm_layers(float3 pos, int octaves, float scale, float persistence, float lacunarity)
{
    float sum = 0.0;
    float amplitude = 1.0;
    float frequency = scale;
    for (int i = 0; i < octaves; i++)
    {
        sum += snoise(pos * frequency) * amplitude;
        amplitude *= persistence;
        frequency *= lacunarity;
    }
    return sum;
}

// Ridged multifractal: (1 - |n|)^power per octave, each octave weighted by the
// previous one so detail compounds along crests.
float ridged_layers(float3 pos, int octaves, float scale, float persistence, float lacunarity)
{
    float sum = 0.0;
    float amplitude = 1.0;
    float frequency = scale;
    float weight = 1.0;
    for (int i = 0; i < octaves; i++)
    {
        float n = 1.0 - abs(snoise(pos * frequency));
        n = pow(abs(n), RIDGE_POWER);
        n *= weight;
        weight = saturate(n);
        sum += n * amplitude;
        amplitude *= persistence;
        frequency *= lacunarity;
    }
    return sum;
}

// Average the ridged noise over 5 taps in the sphere's tangent plane to soften
// crest aliasing. Axes are deliberately unnormalized (they shrink toward the
// poles, where the smoothing radius gracefully collapses).
float smoothed_ridged(float3 dir, int octaves, float scale, float persistence, float lacunarity)
{
    float3 axis_a = cross(dir, float3(0.0, 1.0, 0.0));
    float3 axis_b = cross(dir, axis_a);
    float offset = RIDGE_SMOOTH_OFFSET / max(scale, 1e-3);

    float sum = ridged_layers(dir, octaves, scale, persistence, lacunarity);
    sum += ridged_layers(dir - axis_a * offset, octaves, scale, persistence, lacunarity);
    sum += ridged_layers(dir + axis_a * offset, octaves, scale, persistence, lacunarity);
    sum += ridged_layers(dir - axis_b * offset, octaves, scale, persistence, lacunarity);
    sum += ridged_layers(dir + axis_b * offset, octaves, scale, persistence, lacunarity);
    return sum / 5.0;
}

// Returns float2(height01, detail): normalized elevation in ~[0, 1] for a unit
// sphere direction, plus a signed roughness signal the albedo uses for relief.
//
// The elevation is centred on a FIXED LAND_CENTER, INDEPENDENT of
// `water_height`, so the water level is a real sea-level slider: raising it
// floods this fixed terrain instead of lifting the whole planet.
public float2 terrain_height(float3 dir, TerrainParams p)
{
    // Continents: signed fBm swing around the land centre.
    float continent = fbm_layers(dir, p.height_octaves, p.frequency,
                                 p.height_gain, p.height_lacunarity) * p.height_amp;

    // Flatten deep basins into an abyssal plain, then deepen everything below
    // the shelf so coasts drop off instead of sloping forever.
    continent = smooth_max(continent, -OCEAN_FLOOR_DEPTH, OCEAN_FLOOR_SMOOTHING);
    if (continent < 0.0)
    {
        continent *= 1.0 + OCEAN_DEPTH_MULT;
    }

    // Mountain belts: ridged noise gated by a soft low-frequency mask.
    float mask = smoothstep(-MOUNTAIN_BLEND * 0.5, MOUNTAIN_BLEND * 0.5,
                            fbm_layers(dir, MASK_OCTAVES, MASK_SCALE, 0.5, 2.0));
    float ridge = smoothed_ridged(dir, p.ridge_octaves, p.ridge_tiles,
                                  p.ridge_gain, p.ridge_lacunarity);
    float mountains = ridge * mask * p.ridge_strength * RIDGE_RELIEF;

    const float LAND_CENTER = 0.5;
    float height = LAND_CENTER + continent + mountains;

    // Relief channel in [0, 1]: 0 on plains and basins, rising toward 1 on
    // masked ridge crests. The albedo uses it to bias and shade the bands.
    float detail = saturate(ridge * mask);
    return float2(height, detail);
}

public float2 terrain_height(float3 dir)
{
    return terrain_height(dir, default_terrain_params());
}

// Signed per-pixel roughness used to break up the colour bands: a short
// gradient-noise fBm, tuned against the committed example scene's tile scale.
public float terrain_breakup(float2 uv)
{
    float v = 0.0;
    float amp = 0.4;
    float freq = 3.0;

    for (int i = 0; i < 6; i++)
    {
        v += noised2(uv * freq).x * amp;
        amp *= 0.62;
        freq *= 2.3;
    }

    return v;
}

public float terrain_breakup_triplanar(float3 normal_dir)
{
    float3 w = abs(normal_dir);
    w = pow(w, 4.0);
    w /= max(w.x + w.y + w.z, 1e-5);

    // Single inlined copy of terrain_breakup (see height triplanar note).
    float2 uvs[3] = { normal_dir.yz, normal_dir.xz, normal_dir.xy };
    float ws[3] = { w.x, w.y, w.z };
    float acc = 0.0;
    [loop]
    for (uint i = 0; i < 3; i++)
    {
        acc += terrain_breakup(uvs[i] * 0.5 + 0.5) * ws[i];
    }
    return acc;
}

// Diffuse terrain colour: the Andrino land colormap sampled by height above the
// waterline, with slope-driven bare rock, relief shading and a thin beach.
// normal_y is the Y (up) component of the surface normal (1.0 = flat, 0.0 = vertical cliff).
public float3 terrain_albedo_slope(float height, float relief, float breakup, float normal_y, TerrainParams p)
{
    float water_mask = 1.0 - smoothstep(p.water_height - 0.015, p.water_height + 0.015, height);
    // How far the bed lies below the waterline (0 = at the shore, 1 = deep).
    // Broken up a touch by the breakup signal so the bed is not a flat wash.
    float depth = saturate((p.water_height - height + breakup * 0.01) / 0.10);

    // Bands are keyed on height ABOVE the waterline so they track the sea-level
    // slider: the ramp's snow stop is reached LAND_RAMP_SPAN above the shore.
    // `breakup` jitters the ramp coordinate so band edges don't read as contour
    // rings, and ridge crests (`relief`, 0..1 from terrain_height) bias toward
    // the rocky stops.
    const float LAND_RAMP_SPAN = 0.35;
    float t = (height - p.water_height) / LAND_RAMP_SPAN + breakup * 0.18 + relief * 0.08;
    float3 diffuse = land_ramp(t);

    // Steep ground loses its vegetation tint and falls back to bare rock (the
    // colormap's own rock stop, darkened a touch), whatever the altitude. The
    // breakup signal dithers the slope threshold so the treeline is ragged.
    float rock = smoothstep(0.62 + breakup * 0.08, 0.35 + breakup * 0.08, normal_y);
    diffuse = lerp(diffuse, LAND_COLOR[3] * 0.85, rock);

    // Relief shading: valley floors sit in shadowed accumulation zones, ridge
    // crests catch the light. Multiplicative so it survives on every band.
    diffuse *= lerp(0.85, 1.06, saturate(relief + breakup * 0.4));

    // Thin dry beach hugging the waterline: the submerged shore sand continues
    // just above the water so land meets the analytic ocean on a light strip.
    float beach = 1.0 - smoothstep(0.003, 0.011, height - p.water_height + breakup * 0.004);
    diffuse = lerp(diffuse, SEABED_SHORE_COLOR, beach * (1.0 - rock));

    // Submerged ground: pale sand continuing from the beach, cooling/darkening
    // with depth. No blue here — the water shader tints it on the way out.
    float3 seabed = lerp(SEABED_SHORE_COLOR, SEABED_DEEP_COLOR, smoothstep(0.0, 0.85, depth));
    // Keep the seabed textured: same relief modulation the land uses.
    seabed *= 0.85 + 0.3 * sq(relief);
    diffuse = lerp(diffuse, seabed, water_mask);

    return diffuse;
}

public float3 terrain_albedo_slope(float height, float relief, float breakup, float normal_y)
{
    return terrain_albedo_slope(height, relief, breakup, normal_y, default_terrain_params());
}

// Diffuse terrain colour from height/erosion/breakup only (flat-ground slope).
public float3 terrain_albedo(float height, float relief, float breakup, TerrainParams p)
{
    return terrain_albedo_slope(height, relief, breakup, 1.0, p);
}

public float3 terrain_albedo(float height, float relief, float breakup)
{
    return terrain_albedo_slope(height, relief, breakup, 1.0, default_terrain_params());
}
