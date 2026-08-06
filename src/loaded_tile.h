#pragma once

#include "crs.h"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <vector>

namespace panorama {

// One fully resident square terrain tile in tracer order. `size` is the number
// of level-1 cells along one edge. All arrays are float32 and row-major; row
// zero is the southern edge, so Y increases northward as required by tracing.
struct LoadedTile {
  // True when level_0_vertices owns exact `(size + 1)²` vertex elevations.
  // Level-1-only tiles leave the pointer null and only support approximate
  // collisions against level_1_cells.
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
  std::unique_ptr<std::vector<float>> level_0_vertices;

  // The required N×N first maximum-mipmap level. For level-0 input it is the
  // maximum of each 2×2 vertex patch; for level-1 input it is read directly.
  std::vector<float> level_1_cells;

  /// Load a single-band, north-up `.tif` into south-to-north tracer row order.
  /// When `supports_level_0_collisions` is true, interpret source values as
  /// vertices and build the required first maximum-mipmap level from them.
  /// Otherwise interpret source values as level-1 cells directly.
  [[nodiscard]] static LoadedTile load_tif(const std::filesystem::path &path,
                                           bool supports_level_0_collisions);
};

} // namespace panorama
