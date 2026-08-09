#pragma once

#include <cstdint>
#include <filesystem>

namespace panorama {

/// A fixed projected observer position used to establish local trace axes.
struct ObserverLocation {
  double easting;
  double northing;
  double elevation;
};

/// Host-only configuration for the first independent-ray, single-tile pass.
struct RaytraceConfig {
  std::filesystem::path tile_path;
  ObserverLocation observer;
  uint32_t num_azimuth;
  uint32_t num_polar;
  double azimuth_start;
  double azimuth_end;
  double polar_start;
  double polar_end;
  float max_distance;
};

/// Load one level-1 tile and encode all future ray-tracing kernel bindings.
///
/// The function intentionally stops immediately before dispatching the Metal
/// kernel. It validates the host/device payload and is the staging point for
/// the first GPU traversal implementation.
void prepare_single_tile_raytrace(const RaytraceConfig &config);

} // namespace panorama
