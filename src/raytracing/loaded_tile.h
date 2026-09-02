#pragma once

#include "crs.h"

#include <cstdint>
#include <filesystem>

namespace panorama {

/// Header metadata for one square prepared terrain tile.
///
/// Ray tracing consumes only `.ptile` files. Their payload stays on disk until
/// the resident cache loads the selected LOD directly into GPU-visible memory.
struct LoadedTile {
  Crs crs;
  float maximum_elevation;
  uint32_t size;

  // Canonical south-west vertex and positive square-grid spacing in metres.
  double lower_left_x;
  double lower_left_y;
  double delta;

  // Number of levels in the maximum mipmap, including level 1 and its final
  // 1×1 maximum level. Always greater than or equal to 1.
  uint32_t num_levels;

  /// Read metadata from a prepared `.ptile` file.
  [[nodiscard]] static LoadedTile load(const std::filesystem::path &path);
};

} // namespace panorama
