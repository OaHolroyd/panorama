#include "../src/raytracing/terrain_tile_bvh.metalh"
#include "coverage_reference.metalh"

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
  const bool expected = reference_bvh_source_interval(
      plain[source],
      plain,
      vertices,
      bool(query.w),
      ray.xy,
      ray.zw,
      a,
      b
  );
  const bool actual =
      bvh_source_interval(indexed[source], indexed, vertices, bool(query.w), ray.xy, ray.zw, c, d);
  bool same_interval = expected == actual && (!expected || (a == c && b == d));
  if (expected && actual && !bool(query.w) && a == c && d > c) {
    // Physical coverage may now return a longer (or cache-bounded shorter)
    // connected prefix. Prove its entirety with the independent old walker;
    // ownership scheduling still requires exactly the original interval.
    float progress = c;
    while (progress < d) {
      float start = progress, end = query.y;
      if (!reference_bvh_source_interval(
              plain[source],
              plain,
              vertices,
              false,
              ray.xy,
              ray.zw,
              start,
              end
          ) ||
          start > progress + max(1e-3F, 8.0F * FLT_EPSILON * max(1.0F, progress)) ||
          !(end > progress))
        break;
      progress = end;
    }
    same_interval = progress >= d;
  }
  const float2 point = ray.xy + query.x * ray.zw;
  const bool owned = reference_bvh_owned_hit(source, plain, vertices, point);
  results[index] = uint4(
      same_interval,
      owned == bvh_owned_hit(source, indexed, vertices, point),
      expected,
      owned
  );
}

// Compare complete physical-coverage walks, including their established
// Float32 continuity tolerance. Cache capacity affects work, never gap location.
kernel void check_coverage_walk(
    device const BvhCoveragePolygon *plain [[buffer(0)]],
    device const BvhCoveragePolygon *indexed [[buffer(1)]],
    device const BvhCoverageVertex *vertices [[buffer(2)]],
    device const float4 *queries [[buffer(3)]],
    device float4 *results [[buffer(4)]]
) {
  const float4 ray = queries[0], query = queries[1];
  const uint source = uint(query.z);
  float4 result(0.0F);
  for (uint variant = 0; variant < 2; ++variant) {
    float progress = query.x;
    uint steps = 0;
    while (progress < query.y && steps < 4096U) {
      float start = progress, end = query.y;
      ++steps;
      const bool found = variant == 0 ? reference_bvh_source_interval(
                                            plain[source],
                                            plain,
                                            vertices,
                                            false,
                                            ray.xy,
                                            ray.zw,
                                            start,
                                            end
                                        )
                                      : bvh_source_interval(
                                            indexed[source],
                                            indexed,
                                            vertices,
                                            false,
                                            ray.xy,
                                            ray.zw,
                                            start,
                                            end
                                        );
      const float tolerance = max(1e-3F, 8.0F * FLT_EPSILON * max(1.0F, progress));
      if (!found || start > progress + tolerance || !(end > progress))
        break;
      progress = end;
    }
    result[variant] = progress;
    result[variant + 2U] = float(steps);
  }
  results[0] = result;
}
