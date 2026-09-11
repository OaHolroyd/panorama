#include "../src/raytracing/terrain_tile_bvh.metalh"

// Exercise actual GPU rounding at distant shadow origins. The predecessor of
// the returned distance must still have the rejected ownership point, while
// the returned distance must reach a new one (without skipping an interval).
kernel void check_ownership_progress(
    device const float4 *rays [[buffer(0)]],
    device const float *distances [[buffer(1)]],
    device float4 *results [[buffer(2)]],
    uint index [[thread_position_in_grid]]
) {
  const float2 origin = rays[index].xy, direction = rays[index].zw;
  const float distance = distances[index];
  const float2 point = origin + distance * direction;
  const float next = bvh_next_distinct_point(origin, direction, distance);
  results[index] = float4(
      next,
      all(origin + nextafter(next, -INFINITY) * direction == point),
      any(origin + next * direction != point),
      any(origin + nextafter(distance, INFINITY) * direction != point)
  );
}
