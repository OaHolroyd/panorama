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
  /// Shared terrain dimensions, curvature, range, and ray count.
  RaytraceParameters trace;
  /// Common horizontal direction and vertical slope toward the sun.
  RayDirection direction;
  /// Observer-relative western edge of catalogue column zero.
  float grid_x_min;
  /// Observer-relative northern edge of catalogue row zero.
  float grid_y_max;
  /// Width of every square catalogue tile in projected metres.
  float tile_width;
  /// Power-of-two capacity of the GPU catalogue hash table.
  uint32_t catalogue_hash_capacity;
};

/// GPU storage and kernels for one any-hit sun ray per primary collision.
class GpuTerrainShadowResources {
public:
  /// Compile shadow kernels on the primary trace's device, queue, and library.
  GpuTerrainShadowResources(
      id<MTLDevice> device,
      id<MTLCommandQueue> queue,
      id<MTLLibrary> library,
      bool trace_quantized,
      bool bilinear_collisions,
      bool c1_normals
  );
  /// Release per-ray buffers and pipelines after their queue has completed.
  ~GpuTerrainShadowResources();

  /// Reallocate per-ray state when the primary image dimensions change.
  void resize(uint32_t ray_count);

  /// Select a precompiled traversal specialization without reallocating rays
  /// or visibility storage. C1 normals do not affect any-hit shadows, but the
  /// argument keeps this interface aligned with the primary trace resource.
  void set_collision_options(bool bilinear_collisions, bool c1_normals);

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

  /// Trace one resident batch and emit source-indexed continuations.
  [[nodiscard]] GpuFrontierPassResult trace_frontier(
      const TileManagerBindings &cache,
      id<MTLBuffer> catalogue_hash,
      const ShadowTraceParameters &parameters,
      uint32_t mipmap_value_count,
      uint32_t active_count,
      Timer &timer
  );

  /// Return mapped successor work written by the latest frontier pass.
  [[nodiscard]] std::span<const DeferredRayWork> deferred_work(uint32_t count) const;
  /// Return the shared work buffer populated by HostFrontier.
  [[nodiscard]] id<MTLBuffer> active_frontier() const;
  /// Return one byte per primary pixel: nonzero is visible to the sun.
  [[nodiscard]] id<MTLBuffer> visibility() const;
  /// Fill visibility without tracing, for vertical or below-horizon sunlight.
  void fill_visibility(uint8_t value);

private:
  struct State;
  std::unique_ptr<State> state_;
};

} // namespace panorama
