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

/// Load one level-1 tile and submit the first ray-tracing kernel dispatch.
///
/// The currently bound kernel is a traversal stub, but this function validates
/// and submits the complete host/device payload used by later GPU traversal.
void perform_single_tile_raytrace(const RaytraceConfig &config);

} // namespace panorama
