#pragma once

#include "raytrace_gpu.h"
#include "terrain_catalogue.h"

#import <Metal/Metal.h>

#include <cstdint>
#include <memory>
#include <span>

namespace panorama {

class Timer;

/// Geometry shared by the shadow initializer, traversal, and successor pass.
struct ShadowTraceParameters {
  RaytraceParameters trace;
  RayDirection direction;
  float grid_x_min;
  float grid_y_max;
  float tile_width;
  uint32_t catalogue_hash_capacity;
};

/// GPU storage and kernels for one any-hit sun ray per primary collision.
class GpuTerrainShadowResources {
public:
  GpuTerrainShadowResources(
      id<MTLDevice> device,
      id<MTLCommandQueue> queue,
      id<MTLLibrary> library,
      bool trace_quantized
  );
  ~GpuTerrainShadowResources();

  void resize(uint32_t ray_count);

  /// Construct ray origins and return their source-bucketed initial frontier.
  [[nodiscard]] std::span<const DeferredRayWork> initialise(
      id<MTLBuffer> camera_rays,
      id<MTLBuffer> distances,
      id<MTLBuffer> elevations,
      id<MTLBuffer> surface_gradients,
      id<MTLBuffer> catalogue_hash,
      const ShadowTraceParameters &parameters,
      Timer &timer
  );

  [[nodiscard]] GpuFrontierPassResult trace_frontier(
      const TileManagerBindings &cache,
      id<MTLBuffer> catalogue_hash,
      const ShadowTraceParameters &parameters,
      uint32_t mipmap_value_count,
      uint32_t active_count,
      Timer &timer
  );

  [[nodiscard]] std::span<const DeferredRayWork> deferred_work(uint32_t count) const;
  [[nodiscard]] id<MTLBuffer> active_frontier() const;
  [[nodiscard]] id<MTLBuffer> visibility() const;
  void fill_visibility(uint8_t value);

private:
  struct State;
  std::unique_ptr<State> state_;
};

} // namespace panorama
