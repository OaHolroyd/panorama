#pragma once

#include "ray_projection.h"
#include "resident_tile_cache.h"
#include "terrain_catalogue.h"

#import <Metal/Metal.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>

namespace panorama {

/// Scalar-only tracing ABI shared with the Metal frontier kernels.
struct RaytraceParameters {
  float cell_size;
  float observer_elevation;
  float curvature_coefficient;
  float global_maximum_elevation;
  uint32_t num_levels;
  uint32_t ray_count;
  float max_distance;
};

/// One unresolved ray segment in a resident terrain tile.
struct RayWorkItem {
  uint32_t slot;
  uint32_t ray_index;
  uint32_t start_level;
  float entry_distance;
};

/// One continuation whose successor tile is not resident yet.
struct DeferredRayWork {
  uint32_t ray_index;
  uint32_t source_index;
  float entry_distance;
};

/// Results produced by one ordered trace-and-emit command buffer.
struct GpuFrontierPassResult {
  uint32_t deferred_count;
  uint32_t locally_skipped_tiles;
  uint32_t globally_skipped_tiles;
  double device_milliseconds;
};

/// Per-collision GPU products that may be compiled out of the trace pipeline.
/// Distance is intentionally absent because it is always produced.
struct GpuTraceOutputRequirements {
  bool elevations;
  bool surface_gradients;
};

/// Long-lived Metal resources for repeated multi-tile frontier passes.
///
/// This class owns device/pipeline state and buffers whose lifetime spans the
/// complete render. The scheduler retains policy: it decides which work is
/// active, when resident terrain is installed, and how source-bucketed
/// successors are reactivated. One call encodes the ordered trace and
/// continuation-culling kernels.
class GpuRaytraceResources {
public:
  /// Create all reusable Metal resources for an initial per-pixel ray field.
  ///
  /// Optional outputs specialize the trace pipeline, removing their collision
  /// arithmetic, buffer writes, and full-size allocations when disabled.
  GpuRaytraceResources(
      std::span<const RayDirection> rays,
      std::span<const TerrainSource> sources,
      bool trace_quantized,
      GpuTraceOutputRequirements outputs
  );

  GpuRaytraceResources(const GpuRaytraceResources &) = delete;
  GpuRaytraceResources &operator=(const GpuRaytraceResources &) = delete;

  /// Stop an active capture before releasing the owned command queue.
  ~GpuRaytraceResources();

  /// Replace the fixed-size ray field and clear outputs from the preceding frame.
  void update_rays(std::span<const RayDirection> rays);

  /// Reallocate ray-dependent buffers for a differently sized output image.
  void resize_rays(std::span<const RayDirection> rays);

  /// Fill the current frontier with every ray in the observer tile's current slot.
  void initialise_frontier(uint32_t observer_slot);

  /// Return the Metal device shared with resident terrain atlas allocation.
  [[nodiscard]] id<MTLDevice> device() const;

  /// Return the queue shared with post-trace GPU presentation work.
  [[nodiscard]] id<MTLCommandQueue> command_queue() const;

  /// Return the library containing both tracing and presentation kernels.
  [[nodiscard]] id<MTLLibrary> library() const;

  /// Return the buffer holding the frontier which the next pass will trace.
  [[nodiscard]] id<MTLBuffer> active_frontier() const;

  /// Encode, submit, and synchronously complete one trace-and-emit pass.
  [[nodiscard]] GpuFrontierPassResult trace_frontier(
      const ResidentTileCacheBindings &cache,
      const RaytraceParameters &parameters,
      uint32_t mipmap_value_count,
      uint32_t active_count,
      Timer &timer
  );

  /// Return completed deferred successor entries from the most recent pass.
  [[nodiscard]] std::span<const DeferredRayWork> deferred_work(uint32_t count) const;

  /// Return the shared distance output buffer after tracing completes.
  [[nodiscard]] id<MTLBuffer> distances() const;

  /// Return the per-pixel ray directions corresponding to the current outputs.
  ///
  /// Post-trace GPU consumers can combine these with `distances()` without
  /// duplicating the potentially large ray field or reading it back through
  /// the host.
  [[nodiscard]] id<MTLBuffer> ray_directions() const;

  /// Return the shared elevation output buffer after tracing completes.
  ///
  /// Throws if elevation output was disabled when this object was created.
  [[nodiscard]] id<MTLBuffer> elevations() const;

  /// Return packed float16 east/north surface gradients.
  ///
  /// Throws if normal computation was disabled when this object was created.
  [[nodiscard]] id<MTLBuffer> surface_gradients() const;

  /// Begin an opt-in queue-scoped Metal capture, if requested by the environment.
  void start_capture_if_requested();

  /// Stop a capture previously started by this resource owner.
  void stop_capture();

private:
  struct State;
  std::unique_ptr<State> state_;
};

} // namespace panorama
