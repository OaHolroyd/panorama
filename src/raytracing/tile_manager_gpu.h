#pragma once

#import <Metal/Metal.h>

#include <cstdint>

namespace panorama {

/// Immutable spatial metadata for one resident GPU atlas slot.  The layout is
/// mirrored exactly by `ResidentTile` in panorama.metal.
struct ResidentTile {
  float tile_x_min;
  float tile_y_min;
  float maximum_elevation;
  uint32_t lod;
  int64_t row;
  int64_t column;
};

/// Fixed byte offsets for the retained uint16 tracing specialization.  This
/// layout is mirrored exactly by `QuantizedTerrainLayout` in panorama.metal.
struct QuantizedTerrainLayout {
  uint32_t record_stride;
  uint32_t vertex_offset;
  uint32_t elevation_base_offset;
};

} // namespace panorama
