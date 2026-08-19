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

/// Host-only configuration for fixed-observer, level-0 multi-tile tracing.
struct RaytraceConfig {
  /// Directory containing prepared level-0 GeoTIFF or custom terrain tiles.
  std::filesystem::path tile_dir;

  ObserverLocation observer;
  /// Output resolution.
  uint32_t num_azimuth;
  uint32_t num_polar;
  /// Azimuthal angle range.
  double azimuth_start;
  double azimuth_end;
  /// Polar angle range.
  double polar_start;
  double polar_end;
  /// Maximum horizontal trace distance and source-catalogue radius.
  float max_distance;
  /// Maximum number of sources retained in the terrain catalogue; zero means no limit.
  uint32_t max_tile_count;
  /// Total byte budget for the resident GPU terrain-tile cache.
  uint64_t tile_cache_size_bytes;
  /// Maximum background terrain-preparation workers; zero selects all hardware threads.
  uint32_t max_tile_preparation_workers;
  /// Keep uint16 custom terrain quantized through atlas residency and tracing.
  bool retain_quantized;
};

/// Trace the fixed angular field through available level-0 terrain tiles.
void raytrace_tiled_heightmap(const RaytraceConfig &config);

} // namespace panorama
