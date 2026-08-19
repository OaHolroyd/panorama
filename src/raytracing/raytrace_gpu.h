#pragma once

#include "resident_tile_cache.h"

#import <Metal/Metal.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>

namespace panorama {

/// One horizontal direction with the same two-float layout as Metal `float2`.
struct HorizontalDirection {
  float x;
  float y;
};

/// Scalar-only tracing ABI shared with the Metal frontier kernels.
struct RaytraceParameters {
  float cell_size;
  float observer_elevation;
  float curvature_coefficient;
  float global_maximum_elevation;
  uint32_t num_levels;
  uint32_t num_azimuth;
  uint32_t num_polar;
  float max_distance;
};

/// One unresolved azimuth-column segment in the GPU work frontier.
struct TileWorkItem {
  uint32_t slot;
  uint32_t azimuth;
  uint32_t first_polar;
  uint32_t start_level;
  float entry_distance;
};

/// One continuation whose successor tile is not resident yet.
struct DeferredTileWork {
  uint32_t azimuth;
  uint32_t first_polar;
  float entry_distance;
};

/// Results produced by one ordered trace-and-emit command buffer.
struct GpuFrontierPassResult {
  uint32_t next_count;
  uint32_t deferred_count;
  double device_milliseconds;
};

/// Long-lived Metal resources for repeated multi-tile frontier passes.
///
/// This class owns device/pipeline state and buffers whose lifetime spans the
/// complete render. The scheduler retains policy: it decides which work is
/// active, when resident terrain is installed, and how deferred successors
/// are reactivated. One call to `trace_frontier` encodes the two kernels that
/// trace current segments then emit their successor work items.
class GpuRaytraceResources {
public:
  /// Create all reusable Metal resources for a fixed angular output field.
  GpuRaytraceResources(
      std::span<const HorizontalDirection> directions,
      std::span<const float> slopes,
      size_t ray_count,
      uint32_t frontier_capacity,
      bool trace_quantized
  );

  GpuRaytraceResources(const GpuRaytraceResources &) = delete;
  GpuRaytraceResources &operator=(const GpuRaytraceResources &) = delete;

  /// Stop an active capture before releasing the owned command queue.
  ~GpuRaytraceResources();

  /// Fill the current frontier with the observer tile's initial columns.
  void initialise_frontier(uint32_t num_azimuth);

  /// Return the Metal device shared with resident terrain atlas allocation.
  [[nodiscard]] id<MTLDevice> device() const;

  /// Return the buffer holding the frontier which the next pass will trace.
  [[nodiscard]] id<MTLBuffer> active_frontier() const;

  /// Swap active and successor frontier buffers after a completed pass.
  void swap_frontiers();

  /// Encode, submit, and synchronously complete one trace-and-emit pass.
  [[nodiscard]] GpuFrontierPassResult trace_frontier(
      const ResidentTileCacheBindings &cache,
      const RaytraceParameters &parameters,
      uint32_t mipmap_value_count,
      uint32_t active_count,
      Timer &timer
  );

  /// Return completed deferred successor entries from the most recent pass.
  [[nodiscard]] std::span<const DeferredTileWork> deferred_work(uint32_t count) const;

  /// Return the shared distance output buffer after tracing completes.
  [[nodiscard]] id<MTLBuffer> distances() const;

  /// Return the shared elevation output buffer after tracing completes.
  [[nodiscard]] id<MTLBuffer> elevations() const;

  /// Begin an opt-in queue-scoped Metal capture, if requested by the environment.
  void start_capture_if_requested();

  /// Stop a capture previously started by this resource owner.
  void stop_capture();

private:
  struct State;
  std::unique_ptr<State> state_;
};

} // namespace panorama
