#pragma once

#import <CoreGraphics/CoreGraphics.h>
#import <Metal/Metal.h>

#include "ray_projection.h"
#include <array>
#include <cstdint>
#include <vector>

namespace panorama::app {

/// Encodes into the terrain producer, including its corrected fallback pass.
/// Fresh immutable snapshots allow the trace to reuse its own output buffers.
class GpuVisibilityPointProjector {
public:
  GpuVisibilityPointProjector(id<MTLDevice> device, id<MTLLibrary> library);
  [[nodiscard]] id<MTLBuffer> project(
      id<MTLBuffer> rays,
      id<MTLBuffer> distances,
      ImageSize image,
      id<MTLCommandBuffer> command
  ) const;

private:
  id<MTLDevice> device_;
  id<MTLComputePipelineState> pipeline_;
};

// Pixel coordinates have their origin at the north-west corner. Scalar ABI
// mirrored in visibility_mask.metal; ray points are observer-relative metres.
struct VisibilityMaskParameters {
  float centre_x, centre_y;
  float east_x, east_y;
  float north_x, north_y;
  uint32_t width, height, count;
  float grid_easting = 0, grid_northing = 0;
  float grid_step_easting = 0, grid_step_northing = 0;
  uint32_t grid_size = 0;
};
static_assert(sizeof(VisibilityMaskParameters) == 56);

// Main-thread snapshot: MapKit world coordinates use a 2^28-wide Mercator map.
struct VisibilityMapRegion {
  double x, y, width, height;
  double observer_easting, observer_northing, max_distance;
  uint32_t epsg;
  bool operator==(const VisibilityMapRegion &) const = default;
};

/// Small terrain-to-map lookup, refined against exact CRS samples. Exposed to
/// allow projection accuracy tests independent of rasterization.
struct VisibilityProjectionGrid {
  double easting, northing, step_easting, step_northing;
  uint32_t size;
  std::vector<std::array<float, 2>> pixels;
};
VisibilityProjectionGrid
make_visibility_projection_grid(const VisibilityMapRegion &region, uint32_t width, uint32_t height);

/// Serial-worker-only workspace. Reuses small GPU buffers; each returned image
/// owns a copy of the completed RGBA bytes and can outlive subsequent jobs.
class VisibilityMask {
public:
  VisibilityMask(id<MTLDevice> device, id<MTLCommandQueue> queue, id<MTLLibrary> library);
  [[nodiscard]] CGImageRef render(
      id<MTLBuffer> points,
      VisibilityMaskParameters parameters,
      const VisibilityMapRegion *region = nullptr
  ) CF_RETURNS_RETAINED;
  void clear();

private:
  id<MTLDevice> device_;
  id<MTLCommandQueue> queue_;
  id<MTLComputePipelineState> scatter_, resolve_;
  id<MTLBuffer> occupancy_, pixels_, grid_;
  VisibilityMapRegion grid_region_ = {};
  VisibilityProjectionGrid projection_ = {};
  uint32_t grid_width_ = 0, grid_height_ = 0;
};

} // namespace panorama::app
