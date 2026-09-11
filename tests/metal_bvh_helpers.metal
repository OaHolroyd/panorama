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

// Compare the indexed polygon lists with the same leaves scanned exhaustively.
// Both paths evaluate the actual GPU boundary tolerances and interval merging.
kernel void check_coverage_index(
    device const BvhCoveragePolygon *plain [[buffer(0)]],
    device const BvhCoveragePolygon *indexed [[buffer(1)]],
    device const BvhCoverageVertex *vertices [[buffer(2)]],
    device const float4 *queries [[buffer(3)]],
    device uint4 *results [[buffer(4)]],
    uint index [[thread_position_in_grid]]
) {
  const float4 ray = queries[2U * index];
  const float4 query = queries[2U * index + 1U];
  const uint source = uint(query.z);
  float a = query.x, b = query.y, c = a, d = b;
  const bool expected =
      bvh_source_interval(plain[source], plain, vertices, bool(query.w), ray.xy, ray.zw, a, b);
  const bool actual =
      bvh_source_interval(indexed[source], indexed, vertices, bool(query.w), ray.xy, ray.zw, c, d);
  const float2 point = ray.xy + query.x * ray.zw;
  const bool owned = bvh_owned_hit(source, plain, vertices, point);
  results[index] = uint4(
      expected == actual && (!expected || (a == c && b == d)),
      owned == bvh_owned_hit(source, indexed, vertices, point),
      expected,
      owned
  );
}
