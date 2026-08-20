#pragma once

#include <cstdint>
#include <filesystem>
#include <vector>

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

/// One projected ray direction expressed per unit horizontal distance.
///
/// `x` and `y` form a normalized horizontal direction, `slope` is the
/// corresponding vertical change, and the reciprocals avoid repeated DDA
/// divisions. This layout is mirrored by Metal's `RayDirection`.
struct RayDirection {
  float x;
  float y;
  float inverse_x;
  float inverse_y;
  float slope;
};

/// An arbitrary row-major ray field for a rectangular output.
struct RayField {
  uint32_t width;
  uint32_t height;
  std::vector<RayDirection> rays;
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

/// Trace an explicitly supplied per-pixel ray field.
void raytrace_tiled_heightmap(const RaytraceConfig &config, const RayField &field);

} // namespace panorama
