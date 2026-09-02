#pragma once

#include "crs.h"

#include <cstdint>

namespace panorama {

/// Geometry read from a prepared tile header and shared by compatible atlas slots.
struct TileGeometry {
  /// Projected horizontal and vertical coordinate reference system.
  Crs crs;
  /// Conservative highest elevation published by the reference tile.
  float maximum_elevation;
  /// Number of LOD-1 cells along either square edge.
  uint32_t cell_count;
  /// Projected easting of the south-west LOD-1 vertex.
  double lower_left_x;
  /// Projected northing of the south-west LOD-1 vertex.
  double lower_left_y;
  /// LOD-1 vertex spacing in projected metres.
  double cell_size;
  /// Maximum hierarchy depth, including the finest and final 1x1 levels.
  uint32_t mipmap_level_count;
};

} // namespace panorama
