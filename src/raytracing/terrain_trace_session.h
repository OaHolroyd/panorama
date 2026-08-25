#pragma once

#include "crs.h"
#include "ray_projection.h"
#include "raytrace_config.h"
#include "raytrace_gpu.h"

#import <Metal/Metal.h>

#include <memory>

namespace panorama {

/// Persistent terrain-tracing state shared by a sequence of camera views.
///
/// Catalogue discovery, Metal pipelines, preparation workers, and the
/// resident terrain/mipmap atlas live for the complete session. Each call to
/// `trace` replaces only the ray field and transient frontier, allowing turns
/// and resolution changes to reuse cached terrain. Future movement support can
/// update observer-specific state without making atlas lifetime frame-specific.
class TerrainTraceSession {
public:
  TerrainTraceSession(
      const RaytraceConfig &config,
      const RayField &initial_field,
      GpuTraceOutputRequirements outputs
  );

  TerrainTraceSession(const TerrainTraceSession &) = delete;
  TerrainTraceSession &operator=(const TerrainTraceSession &) = delete;
  ~TerrainTraceSession();

  /// Trace a new view, resizing only ray-dependent GPU buffers when necessary.
  void trace(const RayField &field);

  [[nodiscard]] ImageSize image() const;
  /// Return the projected coordinate system shared by the resident terrain.
  [[nodiscard]] Crs crs() const;
  [[nodiscard]] id<MTLDevice> device() const;
  [[nodiscard]] id<MTLCommandQueue> command_queue() const;
  [[nodiscard]] id<MTLLibrary> library() const;
  [[nodiscard]] id<MTLBuffer> ray_directions() const;
  [[nodiscard]] id<MTLBuffer> distances() const;
  [[nodiscard]] id<MTLBuffer> elevations() const;
  [[nodiscard]] id<MTLBuffer> surface_gradients() const;

  /// Print cumulative cache, preparation, frontier, and timing statistics.
  void print_statistics() const;

private:
  struct State;
  std::unique_ptr<State> state_;
};

} // namespace panorama
