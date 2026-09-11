#pragma once

#include <cstdint>
#include <filesystem>
#include <utility>
#include <vector>

namespace panorama {

enum class Raytracer : uint32_t { Software, MetalBvh };

/// One prepared terrain dataset in descending ownership priority.
struct TerrainDatasetConfig {
  std::filesystem::path directory;
  double vertical_offset_metres = 0.0;
};

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
  RaytraceConfig() = default;
  RaytraceConfig(
      std::filesystem::path directory,
      ObserverLocation observer_location,
      float distance,
      uint32_t tile_count,
      uint64_t tile_cache_bytes,
      uint32_t preparation_workers,
      bool keep_quantized,
      bool use_bilinear_collisions,
      bool use_c1_normals,
      bool observer_fallback = false,
      float terrain_lod_scale = 0.0F,
      Raytracer tracer = Raytracer::Software,
      uint32_t block_cells = 4U,
      uint64_t bvh_cache_bytes = 512ULL * 1024ULL * 1024ULL,
      bool shared_tile_bvh = true,
      std::vector<TerrainDatasetConfig> datasets = {}
  )
      : tile_dir(std::move(directory)), observer(observer_location), max_distance(distance),
        max_tile_count(tile_count), tile_cache_size_bytes(tile_cache_bytes),
        max_tile_preparation_workers(preparation_workers), retain_quantized(keep_quantized),
        bilinear_collisions(use_bilinear_collisions), c1_normals(use_c1_normals),
        allow_observer_fallback(observer_fallback), lod_scale(terrain_lod_scale), raytracer(tracer),
        bvh_block_cells(block_cells), bvh_cache_size_bytes(bvh_cache_bytes),
        use_tile_bvh(shared_tile_bvh), terrain_datasets(std::move(datasets)) {}

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
  /// Ordered terrain datasets. When empty, `tile_dir` supplies the single
  /// compatibility dataset. Earlier entries own overlapping terrain.
  std::vector<TerrainDatasetConfig> terrain_datasets;
};

/// Resolve the compatibility spelling into one ordered dataset list.
[[nodiscard]] inline std::vector<TerrainDatasetConfig>
configured_terrain_datasets(const RaytraceConfig &config) {
  if (!config.terrain_datasets.empty())
    return config.terrain_datasets;
  return config.tile_dir.empty() ? std::vector<TerrainDatasetConfig>{}
                                 : std::vector<TerrainDatasetConfig>{{config.tile_dir, 0.0}};
}

} // namespace panorama
