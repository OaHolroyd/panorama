#include "../shared/threadgroup_sizes.h"
#include "gpu_camera_types.metalh"
#include <metal_stdlib>
using namespace metal;
using namespace panorama::camera_gpu;

// Same packed five-float ABI as panorama::RayDirection.
struct CameraRay {
  float x, y, inverse_x, inverse_y, slope;
};

static bool distorted(constant Camera &c) {
  return c.radial_1 != 0 || c.radial_2 != 0 || c.radial_3 != 0 || c.tangential_1 != 0 ||
         c.tangential_2 != 0;
}

static float2 tangential(float2 p, constant Camera &c) {
  const float r2 = dot(p, p);
  return float2(
      2 * c.tangential_1 * p.x * p.y + c.tangential_2 * (r2 + 2 * p.x * p.x),
      c.tangential_1 * (r2 + 2 * p.y * p.y) + 2 * c.tangential_2 * p.x * p.y
  );
}

static float radial(float2 p, constant Camera &c) {
  const float r2 = dot(p, p);
  return 1 + r2 * (c.radial_1 + r2 * (c.radial_2 + r2 * c.radial_3));
}

// Invert the CLI's Brown-Conrady calibration. A failed inversion becomes NaN
// and rejects the completed ray image instead of publishing invalid geometry.
static float2 camera_pixel(float2 pixel, constant Camera &c) {
  const float2 target =
      (pixel - float2(c.principal_x, c.principal_y)) / float2(c.focal_x, c.focal_y);
  if (!distorted(c))
    return target;
  float2 p = target;
  for (uint i = 0; i < 32; ++i) {
    const float r = radial(p, c);
    if (!isfinite(r) || abs(r) < 1e-12F)
      return float2(NAN);
    const float2 next = (target - tangential(p, c)) / r;
    const bool converged = length(next - p) <= 2 * FLT_EPSILON * (1 + length(next));
    p = next;
    if (converged)
      break;
  }
  if (!all(isfinite(p)) ||
      length(p * radial(p, c) + tangential(p, c) - target) > 2e-6F * (1 + length(target)))
    return float2(NAN);
  return p;
}

kernel void generate_camera_rays(
    constant Camera &camera [[buffer(0)]],
    device CameraRay *rays [[buffer(1)]],
    device atomic_uint *invalid [[buffer(2)]],
    uint2 position [[thread_position_in_grid]]
) {
  const uint index = position.y * camera.width + position.x;
  if (index >= camera.width * camera.height)
    return;
  const float2 pixel(float(index % camera.width) + 0.5F, float(index / camera.width) + 0.5F);
  if (camera.angular) {
    const float azimuth = camera.azimuth_start + pixel.x * camera.azimuth_step;
    const float elevation = camera.elevation_start + pixel.y * camera.elevation_step;
    const float2 east(camera.right[0], camera.right[1]);
    const float2 north(camera.forward[0], camera.forward[1]);
    const float2 horizontal = sin(azimuth) * east + cos(azimuth) * north;
    const float x = horizontal.x, y = horizontal.y, slope = tan(elevation);
    if (!isfinite(slope))
      atomic_fetch_or_explicit(invalid, 1U, memory_order_relaxed);
    rays[index] = {x,
                   y,
                   x == 0 ? INFINITY : 1 / x,
                   y == 0 ? INFINITY : 1 / y,
                   isfinite(slope) ? slope : 0};
    return;
  }
  const float2 xy = camera_pixel(pixel, camera);
  const float3 forward(camera.forward[0], camera.forward[1], camera.forward[2]);
  const float3 right(camera.right[0], camera.right[1], camera.right[2]);
  const float3 up(camera.up[0], camera.up[1], camera.up[2]);
  const float3 world = forward + xy.x * right - xy.y * up;
  const float horizontal = length(world.xy);
  float3 direction = world / horizontal;
  if (!(horizontal > 1e-12F) || !all(isfinite(direction))) {
    atomic_fetch_or_explicit(invalid, 1U, memory_order_relaxed);
    direction = float3(0, 1, 0); // Safe traversal; the completed frame is rejected.
  }
  rays[index] = {direction.x,
                 direction.y,
                 direction.x == 0 ? INFINITY : 1 / direction.x,
                 direction.y == 0 ? INFINITY : 1 / direction.y,
                 direction.z};
}

// Adjacent pinhole vectors a=(x,y,1), b=(x+step,y,1). Derive their
// cross product algebraically to avoid subtracting almost equal unit vectors.
// Rigid camera rotation preserves this angle, so it can be cached across pans.
static float adjacent_chord(float x, float y, float step) {
  const float cross_length = abs(step) * sqrt(1 + y * y);
  const float dot_product = 1 + x * (x + step) + y * y;
  return 2 * sin(0.5F * atan2(cross_length, dot_product));
}

kernel void camera_pixel_footprint(
    constant Camera &camera [[buffer(0)]],
    device float *partial [[buffer(1)]],
    uint2 position [[thread_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]],
    uint2 group [[threadgroup_position_in_grid]]
) {
  threadgroup float
      values[panorama::threadgroups::spatial_width * panorama::threadgroups::spatial_height];
  float value = INFINITY;
  const uint column = position.x, row = position.y;
  if (column < camera.width && row < camera.height) {
    if (camera.angular) {
      value = min(abs(camera.azimuth_step), abs(camera.elevation_step));
    } else if (distorted(camera)) {
      const float2 pixel(float(column) + 0.5F, float(row) + 0.5F);
      const float3 a = normalize(float3(camera_pixel(pixel, camera), 1));
      if (column + 1 < camera.width)
        value = distance(a, normalize(float3(camera_pixel(pixel + float2(1, 0), camera), 1)));
      if (row + 1 < camera.height)
        value =
            min(value,
                distance(a, normalize(float3(camera_pixel(pixel + float2(0, 1), camera), 1))));
    } else {
      const float2 xy = camera_pixel(float2(float(column) + 0.5F, float(row) + 0.5F), camera);
      if (column + 1 < camera.width)
        value = adjacent_chord(xy.x, xy.y, 1 / camera.focal_x);
      if (row + 1 < camera.height)
        value = min(value, adjacent_chord(xy.y, xy.x, 1 / camera.focal_y));
    }
  }
  values[lane] = isfinite(value) && value > 0 ? value : INFINITY;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint stride =
           panorama::threadgroups::spatial_width * panorama::threadgroups::spatial_height / 2;
       stride != 0;
       stride >>= 1) {
    if (lane < stride)
      values[lane] = min(values[lane], values[lane + stride]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (lane == 0)
    partial
        [group.y * ((camera.width + panorama::threadgroups::spatial_width - 1) /
                    panorama::threadgroups::spatial_width) +
         group.x] = values[0];
}

kernel void reduce_camera_footprint(
    device const float *partial [[buffer(0)]],
    constant uint &count [[buffer(1)]],
    constant Camera &camera [[buffer(2)]],
    device float *angle [[buffer(3)]],
    uint lane [[thread_index_in_threadgroup]]
) {
  threadgroup float values[panorama::threadgroups::linear_width];
  float value = INFINITY;
  for (uint i = lane; i < count; i += panorama::threadgroups::linear_width)
    value = min(value, partial[i]);
  values[lane] = value;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint stride = panorama::threadgroups::linear_width / 2; stride != 0; stride >>= 1) {
    if (lane < stride)
      values[lane] = min(values[lane], values[lane + stride]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (lane == 0)
    angle[0] =
        isfinite(values[0]) && values[0] > 0 ? values[0] : 1 / max(camera.focal_x, camera.focal_y);
}

kernel void select_camera_lods(
    device const Source *sources [[buffer(0)]],
    constant Lod &settings [[buffer(1)]],
    device const float *angle [[buffer(2)]],
    device uint *lods [[buffer(3)]],
    uint index [[thread_position_in_grid]]
) {
  if (index >= settings.count)
    return;
  const Source source = sources[index];
  const float2 observer(settings.observer_x, settings.observer_y);
  const float2 nearest = clamp(
      observer,
      float2(source.minimum_x, source.minimum_y),
      float2(source.maximum_x, source.maximum_y)
  );
  const float ratio = settings.scale * distance(nearest, observer) * angle[0] / source.cell_size;
  uint lod = 1;
  // Comparing powers of two avoids log2 rounding up just below a threshold.
  float spacing = 2;
  // At a Float32 threshold prefer the finer variant to a rounding-induced
  // upgrade. Away from a few ulps this is the original power-of-two policy.
  while (lod < source.lod_count && ratio >= spacing * (1 + 4 * FLT_EPSILON)) {
    ++lod;
    spacing *= 2;
  }
  lods[index] = lod;
}
