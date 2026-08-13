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
  std::filesystem::path tile_path;
  ObserverLocation observer;
  uint32_t num_azimuth;
  uint32_t num_polar;
  double azimuth_start;
  double azimuth_end;
  double polar_start;
  double polar_end;
  float max_distance;
  /// Maximum number of available tiles submitted to Metal; zero means no limit.
  uint32_t max_tile_count;
  /// Total byte budget for the resident GPU terrain-tile cache.
  uint64_t tile_cache_size_bytes;
};

/// Trace the fixed angular field through available level-0 terrain tiles.
///
/// Stage 1 keeps the tile scheduler on the CPU and synchronises after each
/// tile. It exists to validate shared-boundary hand-offs before later GPU
/// wavefront scheduling and asynchronous tile caching are introduced. Set
/// `RaytraceConfig::max_tile_count` to one to trace only the observer tile.
void perform_multi_tile_raytrace(const RaytraceConfig &config);

} // namespace panorama
