#pragma once

#include "ray_projection.h"

#include <cstdint>
#include <filesystem>

namespace panorama {

/// Effective-Earth curvature used by tracing: 0.1695 metres per mile squared.
/// Multiplying this coefficient by horizontal distance squared gives the
/// ray's elevation gain relative to the curved terrain datum.
inline constexpr double kCurvatureCoefficient = 0.1695 / (1609.344 * 1609.344);

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
  /// Compute one compact terrain-surface normal parameterization per collision.
  bool compute_normals;
};

/// Trace an explicitly supplied per-pixel ray field.
void raytrace_tiled_heightmap(const RaytraceConfig &config, const RayField &field);

} // namespace panorama
