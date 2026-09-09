#include "gpu_camera_types.metalh"
#include <metal_stdlib>
using namespace metal;
using namespace panorama::camera_gpu;

// Same packed five-float ABI as panorama::RayDirection.
struct CameraRay {
  float x, y, inverse_x, inverse_y, slope;
};

kernel void generate_camera_rays(
    constant Camera &camera [[buffer(0)]],
    device CameraRay *rays [[buffer(1)]],
    device atomic_uint *invalid [[buffer(2)]],
    uint index [[thread_position_in_grid]]
) {
  if (index >= camera.width * camera.height)
    return;
  const float x = (float(index % camera.width) + 0.5F - camera.principal_x) / camera.focal_x;
  const float y = (float(index / camera.width) + 0.5F - camera.principal_y) / camera.focal_y;
  const float3 forward(camera.forward[0], camera.forward[1], camera.forward[2]);
  const float3 right(camera.right[0], camera.right[1], camera.right[2]);
  const float3 up(camera.up[0], camera.up[1], camera.up[2]);
  const float3 world = forward + x * right - y * up;
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
    uint index [[thread_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]],
    uint group [[threadgroup_position_in_grid]]
) {
  threadgroup float values[256];
  float value = INFINITY;
  const uint column = index % camera.width, row = index / camera.width;
  if (row < camera.height) {
    const float x = (float(column) + 0.5F - camera.principal_x) / camera.focal_x;
    const float y = (float(row) + 0.5F - camera.principal_y) / camera.focal_y;
    if (column + 1 < camera.width)
      value = adjacent_chord(x, y, 1 / camera.focal_x);
    if (row + 1 < camera.height)
      value = min(value, adjacent_chord(y, x, 1 / camera.focal_y));
  }
  values[lane] = isfinite(value) && value > 0 ? value : INFINITY;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint stride = 128; stride != 0; stride >>= 1) {
    if (lane < stride)
      values[lane] = min(values[lane], values[lane + stride]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (lane == 0)
    partial[group] = values[0];
}

kernel void reduce_camera_footprint(
    device const float *partial [[buffer(0)]],
    constant uint &count [[buffer(1)]],
    constant Camera &camera [[buffer(2)]],
    device float *angle [[buffer(3)]],
    uint lane [[thread_index_in_threadgroup]]
) {
  threadgroup float values[256];
  float value = INFINITY;
  for (uint i = lane; i < count; i += 256)
    value = min(value, partial[i]);
  values[lane] = value;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint stride = 128; stride != 0; stride >>= 1) {
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
  const float2 origin =
      float2(source.x, source.y) - float2(settings.observer_x, settings.observer_y);
  const float2 nearest = clamp(float2(0), origin, origin + settings.tile_width);
  const float ratio = settings.scale * length(nearest) * angle[0] / settings.cell_size;
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
