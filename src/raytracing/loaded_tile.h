#pragma once

#include "crs.h"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <vector>

namespace panorama {

/// Host metadata and any CPU-resident terrain for one square tracer tile.
///
/// `size` is the number of level-1 cells along one edge. Host arrays, when
/// present, are Float32 and row-major; row zero is the southern edge, so Y
/// increases northward as required by tracing. Custom Metal tiles contain
/// metadata only because the cache loads their payloads directly into
/// GPU-visible buffers, either retaining or expanding fixed-point samples.
struct LoadedTile {
  // True when the terrain representation supports exact bilinear collisions.
  // GeoTIFFs then own `(size + 1)²` values in `vertices`; custom Metal tiles
  // leave the host pointer null because their payload stays on the GPU path.
  bool supports_level_0_collisions;
  Crs crs;
  float maximum_elevation;
  uint32_t size;

  // Canonical south-west origin in projected metres. For level-0 data this is
  // vertex (0, 0); for level-1-only data it is the boundary of cell (0, 0).
  // `delta` is the positive, square-grid spacing between vertices or cells.
  double lower_left_x;
  double lower_left_y;
  double delta;

  // Host-resident vertex terrain for bilinear level-0 collisions. This is
  // null for level-1-only input and for GPU-loaded custom Metal tiles.
  std::unique_ptr<std::vector<float>> vertices;

  // Number of levels in the maximum mipmap, including level 1 and its final
  // 1×1 maximum level. Always greater than or equal to 1.
  uint32_t num_levels;

  // Maximum mipmap stored as a contiguous block of memory. It is laid out
  // from finest to coarsest levels (that is, level 1, level 2, ...). A custom
  // Metal tile leaves this empty on the host because the GPU builds it from
  // the representation selected for the resident atlas.
  std::vector<float> mipmap;

  /// Load a single-band, north-up `.tif` into south-to-north tracer row order.
  ///
  /// When `supports_level_0_collisions` is true, interpret source values as
  /// vertices and build the required first maximum-mipmap level from them.
  /// Otherwise interpret source values as level-1 cells directly. Declared
  /// GeoTIFF no-data samples become the project's zero-elevation placeholder,
  /// allowing partially covered chunks to retain their valid terrain. This
  /// does not create any additional maximum-mipmap levels beyond the required
  /// first one.
  [[nodiscard]] static LoadedTile
  load_tif(const std::filesystem::path &path, bool supports_level_0_collisions);

  /// Load either GeoTIFF terrain or custom-tile metadata into the host model.
  ///
  /// Custom files retain their terrain payload on disk: the returned object
  /// contains metadata only, and the cache loads the required record range
  /// through Metal I/O.
  [[nodiscard]] static LoadedTile load(const std::filesystem::path &path);

  /// Fill the mipmap with every coarser maximum level, if not already present.
  void compute_mipmap();
};

} // namespace panorama
