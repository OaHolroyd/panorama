#include <metal_stdlib>

// Metal Shading Language provides GPU-specific types and functions in this
// namespace, including `uint`, `device`, and the `kernel` entry-point keyword.
using namespace metal;

// The first terrain-tracing ABI. This scalar-only structure intentionally
// mirrors RaytraceParameters in raytrace_setup.mm; all projected tile
// coordinates have already been rebased around the observer before upload.
struct RaytraceParameters {
  float tile_x_min;
  float tile_y_min;
  float cell_size;
  float observer_elevation;
  uint num_cell;
  uint num_azimuth;
  uint num_polar;
  float max_distance;
};

// Placeholder for the first independent-ray terrain kernel. The host already
// uploads and binds all terrain, angular, parameter, and output buffers. The
// next stage will dispatch one thread per (azimuth, polar) output and replace
// this body with level-1 DDA traversal and midpoint collision handling.
kernel void trace_single_tile(
    device const float *level_1_cells [[buffer(0)]],
    device const float2 *azimuth_directions [[buffer(1)]],
    device const float *polar_slopes [[buffer(2)]],
    constant RaytraceParameters &params [[buffer(3)]],
    device float *distances [[buffer(4)]],
    device float *elevations [[buffer(5)]],
    uint2 ray_index [[thread_position_in_grid]],
    uint2 threadgroup_position [[threadgroup_position_in_grid]],
    uint2 thread_position [[thread_position_in_threadgroup]],
    uint2 threadgroups [[threadgroups_per_grid]],
    uint2 threads_per_threadgroup [[threads_per_threadgroup]]
) {
  // thread positions
  const uint threadgroup_id = threadgroup_position.y * threadgroups.x + threadgroup_position.x;
  const uint thread_id = thread_position.y * threads_per_threadgroup.x + thread_position.x;

  // Bounds check and convert to flat indexing
  if (ray_index.x >= params.num_azimuth || ray_index.y >= params.num_polar) {
    return;
  }
  const uint output_index = ray_index.y * params.num_azimuth + ray_index.x;

  // Get ray params
  const float u = azimuth_directions[ray_index.x].x;
  const float v = azimuth_directions[ray_index.x].y;
  const int stepx = (int)(u > 0.0) - (int)(u < 0.0);
  const int stepy = (int)(v > 0.0) - (int)(v < 0.0);
  const float delta0 = params.cell_size;
  const float dtx = stepx == 0 ? INFINITY : fabs(delta0 / u);
  const float dty = stepy == 0 ? INFINITY : fabs(delta0 / v);
  const float z0 = params.observer_elevation;
  const float dz = polar_slopes[ray_index.y];

  // The local observer is exactly at (0, 0), which may lie on one or both
  // shared cell boundaries. Nudge only the coordinate used for ownership so a
  // south/west ray starts in its forward cell; all DDA distances still use the
  // exact observer position and therefore retain t = 0 as their geometry.
  const int n = int(params.num_cell);
  const float boundary_nudge = 1e-3F * delta0;
  const float x_classify = stepx == 0 ? 0.0F : copysign(boundary_nudge, u);
  const float y_classify = stepy == 0 ? 0.0F : copysign(boundary_nudge, v);
  const int i0 = clamp(int(floor((y_classify - params.tile_y_min) / delta0)), 0, n - 1);
  const int j0 = clamp(int(floor((x_classify - params.tile_x_min) / delta0)), 0, n - 1);
  int i = i0;
  int j = j0;

  // initial level information (fixed at 1 for now)
  int level = 1;
  int scale = 1 << (level - 1);
  // TODO: adjust index to account for starting level if level != 1
  // i = (i0 / scale) * scale;
  // j = (j0 / scale) * scale;

  // cell-traversal distances
  float ty =
      stepy == 0 ? INFINITY : stepy * (params.tile_y_min / delta0 + i + scale * (stepy > 0)) * dty;
  float tx =
      stepx == 0 ? INFINITY : stepx * (params.tile_x_min / delta0 + j + scale * (stepx > 0)) * dtx;
  // -1 marks the first cell, which has no preceding DDA boundary.
  int dprev = -1;

  // ray tracing
  while (1) {
    // Check for tile exit
    if (i < 0 || j < 0 || i >= n || j >= n) {
      break;
    }

    // Find current ray location
    const float t = min(ty, tx);
    float z_check = z0 + t * dz;
    if (dz > 0) {
      if (dprev == -1) {
        // The observer begins inside this cell, so its near boundary is the
        // ray origin rather than an artificial previous X/Y boundary.
        z_check = z0;
      } else {
        const float previous_step = dprev == 0 ? dty : dtx;
        z_check = z0 + (t - previous_step * scale) * dz;
      }
    }

    // Collision check
    const uint ij_level = (i >> (level - 1)) * n + (j >> (level - 1));
    if (z_check <= level_1_cells[ij_level]) {
      if (level == 1) {
        // TODO: we're not supporting level-0 collisions yet

        // Level-1 only collision checks are faster but less accurate: they assume
        // that the ray collides with the terrain halfway across the cell.
        float t_previous = 0.0F;
        if (stepx != 0) {
          t_previous = max(t_previous, tx - scale * dtx);
        }
        if (stepy != 0) {
          t_previous = max(t_previous, ty - scale * dty);
        }
        t_previous = min(t_previous, t);
        const float t_intersect = (t + t_previous) * 0.5F;
        distances[output_index] = t_intersect;
        elevations[output_index] = z0 + t_intersect * dz;
        break;
      } else {
        // TODO: don't handle other levels yet
      }
    }

    // Step forwards
    if (ty < tx) {
      dprev = 0;
      ty += scale * dty;
      i += scale * stepy;
    } else if (tx < ty) {
      dprev = 1;
      tx += scale * dtx;
      j += scale * stepx;
    } else {
      break;
    }
  } // end while loop
}
