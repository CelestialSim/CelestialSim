// Slang Module — Shared Snapshot Patch Utilities
// Provides tangent-plane bindings (1-6) and the pixel → world-direction mapping
// used by all snapshot patch entry-point shaders.
//
// Binding convention for snapshot patch shaders:
//   0 — RWTexture2D<float4> out_patch   (declared in each entry point)
//   1 — tex_size       (uint)
//   2 — radius         (float)
//   3 — center_dir     (Vec3Param)
//   4 — tangent_u      (Vec3Param)
//   5 — tangent_v      (Vec3Param)
//   6 — angular_extent (float)
//   7+ — noise-specific params  (declared in each entry point)

module snapshot_patch_utils;

public struct Vec3Param
{
    float x;
    float y;
    float z;
};

[[vk::binding(1, 0)]]
public ConstantBuffer<uint> tex_size;

[[vk::binding(2, 0)]]
public ConstantBuffer<float> radius;

[[vk::binding(3, 0)]]
public ConstantBuffer<Vec3Param> center_dir;

[[vk::binding(4, 0)]]
public ConstantBuffer<Vec3Param> tangent_u;

[[vk::binding(5, 0)]]
public ConstantBuffer<Vec3Param> tangent_v;

[[vk::binding(6, 0)]]
public ConstantBuffer<float> angular_extent;

/// Toggle for debug red border (0 = off, non-zero = on).
[[vk::binding(8, 0)]]
public ConstantBuffer<uint> show_border;

/// Maps a pixel coordinate to a 3D direction on the unit sphere via tangent-plane projection.
public float3 snapshot_world_direction(uint2 pixel)
{
    float lu = (float(pixel.x) + 0.5) / float(tex_size) * 2.0 - 1.0;
    float lv = (float(pixel.y) + 0.5) / float(tex_size) * 2.0 - 1.0;

    float angle_u = lu * angular_extent;
    float angle_v = lv * angular_extent;

    float3 c  = float3(center_dir.x, center_dir.y, center_dir.z);
    float3 tu = float3(tangent_u.x, tangent_u.y, tangent_u.z);
    float3 tv = float3(tangent_v.x, tangent_v.y, tangent_v.z);

    float3 dir = c + tan(angle_u) * tu + tan(angle_v) * tv;
    return normalize(dir);
}

/// Returns true if the pixel is within `border` pixels of the patch edge.
public bool is_border_pixel(uint2 pixel, uint border)
{
    return pixel.x < border || pixel.y < border
        || pixel.x >= tex_size - border || pixel.y >= tex_size - border;
}
