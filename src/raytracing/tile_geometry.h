#pragma once

#include "crs.h"

#include <cstdint>

namespace panorama {

/// Geometry read from a prepared tile header and shared by compatible atlas slots.
struct TileGeometry {
  Crs crs;
  float maximum_elevation;
  uint32_t cell_count;
  double lower_left_x;
  double lower_left_y;
  double cell_size;
  uint32_t mipmap_level_count;
};

} // namespace panorama
