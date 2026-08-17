#pragma once

#include "crs.h"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <vector>

namespace panorama {

/// One fully resident square terrain tile in tracer order.
///
/// `size` is the number of level-1 cells along one edge. All arrays are
/// float32 and row-major; row zero is the southern edge, so Y increases
/// northward as required by tracing.
struct LoadedTile {
  // True when `vertices` owns exact `(size + 1)²` vertex elevations.
  // Level-1-only tiles leave the pointer null and only support approximate
  // collisions against the first `mipmap` level.
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

  // Exact vertex terrain for bilinear level-0 collisions, or null when the
  // source tile supplies only level-1 cell values.
  std::unique_ptr<std::vector<float>> vertices;

  // Number of levels in the maximum mipmap, including level 1 and its final
  // 1×1 maximum level. Always greater than or equal to 1.
  uint32_t num_levels;

  // Maximum mipmap stored as a contiguous block of memory. It is laid out
  // from finest to coarsest levels (that is, level 1, level 2, ...).
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

  /// Fill the mipmap with every coarser maximum level, if not already present.
  void compute_mipmap();
};

} // namespace panorama
