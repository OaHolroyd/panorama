#pragma once

#include "crs.h"
#include "ray_projection.h"
#include "raytrace_config.h"
#include "raytrace_gpu.h"
#include "terrain_catalogue.h"

#import <Metal/Metal.h>

#include <memory>

namespace panorama {

/// Persistent terrain-tracing state shared by a sequence of camera views.
///
/// Catalogue discovery, Metal pipelines, preparation workers, and the
/// resident terrain/mipmap atlas live for the complete session. Each call to
/// `trace` replaces only the ray field and transient frontier, allowing turns,
/// resolution changes, and movement across retained catalogue tiles to reuse
/// cached terrain.
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

  /// Move the observer without rebuilding terrain resources when the new
  /// position belongs to any source retained by this session's catalogue.
  /// Returns false when the caller must construct a catalogue and session
  /// centred on terrain outside the current finite source set.
  [[nodiscard]] bool relocate_observer(ObserverLocation observer);

  /// Trace one directional sun ray from each eligible primary collision.
  /// Angles are radians; azimuth is clockwise from grid north and elevation
  /// is above the horizontal plane.
  void trace_shadows(double sun_azimuth, double sun_elevation);

  [[nodiscard]] ImageSize image() const;
  /// Return the projected coordinate system shared by the resident terrain.
  [[nodiscard]] Crs crs() const;
  [[nodiscard]] ObserverLocation observer() const;
  [[nodiscard]] const TerrainCoverage &terrain_coverage() const;
  [[nodiscard]] id<MTLDevice> device() const;
  [[nodiscard]] id<MTLCommandQueue> command_queue() const;
  [[nodiscard]] id<MTLLibrary> library() const;
  [[nodiscard]] id<MTLBuffer> ray_directions() const;
  [[nodiscard]] id<MTLBuffer> distances() const;
  [[nodiscard]] id<MTLBuffer> elevations() const;
  [[nodiscard]] id<MTLBuffer> surface_gradients() const;
  [[nodiscard]] id<MTLBuffer> shadow_visibility() const;

  /// Print cumulative cache, preparation, frontier, and timing statistics.
  void print_statistics() const;

private:
  struct State;
  std::unique_ptr<State> state_;
};

} // namespace panorama
