#pragma once

#include <cstdint>
#include <filesystem>

namespace panorama {

enum class Raytracer : uint32_t { Software, MetalBvh };

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

/// Host configuration for fixed-observer, level-0 multi-tile tracing.
///
/// Output selection deliberately lives outside this type. The application
/// derives optional trace fields from its selected render products, preventing
/// unused collision buffers or shader work from being requested here.
struct RaytraceConfig {
  /// Directory containing prepared `.ptile` terrain tiles.
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
  /// Prefer keeping uint16 custom terrain quantized through residency and tracing.
  /// Other terrain representations continue to use the Float32 atlas path.
  bool retain_quantized;
  /// Use a bilinear patch rather than splitting into two triangles.
  bool bilinear_collisions;
  /// Enforce C1-continuous (rather than C0-continuous) surface normals
  bool c1_normals;
  /// Move an unavailable observer to a dataset-derived default instead of failing.
  bool allow_observer_fallback = false;
  /// Scale used by the per-source terrain LOD policy. Zero disables LOD
  /// selection and retains the original, LOD-1 terrain everywhere.
  float lod_scale = 0.0F;
  Raytracer raytracer = Raytracer::Software;
  /// Maximum terrain cells per axis in a procedural BVH primitive.
  uint32_t bvh_block_cells = 4U;
  /// Detailed BVHs, owned samples, metadata, and peak construction workspace.
  /// Independent of the terrain atlas and ray-sized working buffers.
  uint64_t bvh_cache_size_bytes = 512ULL * 1024ULL * 1024ULL;
  /// Use shared catalogue acceleration for mipmap tile selection when supported.
  /// Disabling this retains grid walking as a reference and compatibility path.
  bool use_tile_bvh = true;
};

} // namespace panorama
