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
    constant RaytraceParameters &parameters [[buffer(3)]],
    device float *distances [[buffer(4)]],
    device float *elevations [[buffer(5)]],
    uint2 ray_index [[thread_position_in_grid]]
) {
  // Keep every currently bound argument named in the stub. This both documents
  // the host/device ABI and avoids accidentally changing it before traversal.
  (void)level_1_cells;
  (void)azimuth_directions;
  (void)polar_slopes;
  (void)parameters;
  (void)distances;
  (void)elevations;
  (void)ray_index;
}
