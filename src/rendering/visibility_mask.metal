#include <metal_stdlib>
using namespace metal;

struct VisibilityMaskParameters {
  float centre_x, centre_y;
  float east_x, east_y;
  float north_x, north_y;
  uint width, height, count;
  float grid_easting, grid_northing;
  float grid_step_easting, grid_step_northing;
  uint grid_size;
};

kernel void visibility_mask_scatter(
    device const float2 *points [[buffer(0)]],
    device atomic_uint *occupancy [[buffer(1)]],
    constant VisibilityMaskParameters &map [[buffer(2)]],
    device const float2 *grid [[buffer(3)]],
    uint index [[thread_position_in_grid]]
) {
  if (index >= map.count)
    return;
  const float2 point = points[index];
  if (!all(isfinite(point)))
    return;
  float2 pixel = float2(map.centre_x, map.centre_y) + point.x * float2(map.east_x, map.east_y) +
                 point.y * float2(map.north_x, map.north_y);
  if (map.grid_size > 1) {
    const float2 cell = (point - float2(map.grid_easting, map.grid_northing)) /
                        float2(map.grid_step_easting, map.grid_step_northing);
    if (any(cell < 0.0f) || any(cell > float(map.grid_size - 1)))
      return;
    const uint2 base = min(uint2(cell), uint2(map.grid_size - 2));
    const float2 t = cell - float2(base);
    const uint index0 = base.y * map.grid_size + base.x;
    pixel =
        mix(mix(grid[index0], grid[index0 + 1], t.x),
            mix(grid[index0 + map.grid_size], grid[index0 + map.grid_size + 1], t.x),
            t.y);
  }
  // Reject before float-to-int conversion (also protects against overflow).
  if (!all(isfinite(pixel)) || any(pixel < -1.0f) ||
      any(pixel > float2(map.width, map.height) + 1.0f))
    return;
  // Two backing pixels, matching the former Metal point footprint. Pixel
  // centres inside [pixel - 1, pixel + 1) are covered, including clipped edges.
  const int2 base = int2(ceil(pixel - 1.5f));
  for (int y = 0; y < 2; ++y) {
    for (int x = 0; x < 2; ++x) {
      const int2 p = base + int2(x, y);
      if (all(p >= 0) && p.x < int(map.width) && p.y < int(map.height))
        atomic_store_explicit(
            &occupancy[uint(p.y) * map.width + uint(p.x)],
            1u,
            memory_order_relaxed
        );
    }
  }
}

kernel void visibility_mask_resolve(
    device atomic_uint *occupancy [[buffer(0)]],
    device uchar4 *pixels [[buffer(1)]],
    constant VisibilityMaskParameters &map [[buffer(2)]],
    uint2 position [[thread_position_in_grid]]
) {
  const uint index = position.y * map.width + position.x;
  if (index >= map.width * map.height)
    return;
  // Premultiplied RGBA: duplicate hits never accumulate opacity.
  pixels[index] = atomic_load_explicit(&occupancy[index], memory_order_relaxed)
                      ? uchar4(0, 42, 87, 87)
                      : uchar4(0);
}
