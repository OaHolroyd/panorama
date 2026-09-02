#pragma once

#import <Metal/Metal.h>

#include <cstdint>

namespace panorama {

/// Immutable spatial metadata for one resident GPU atlas slot.  The layout is
/// mirrored exactly by `ResidentTile` in panorama.metal.
struct ResidentTile {
  /// South-west X coordinate relative to the current observer, in metres.
  float tile_x_min;
  /// South-west Y coordinate relative to the current observer, in metres.
  float tile_y_min;
  /// Conservative elevation bound used to reject the complete tile segment.
  float maximum_elevation;
  /// One-based terrain LOD stored at the start of this atlas slot.
  uint32_t lod;
  /// Global north-to-south catalogue row used to find successor tiles.
  int64_t row;
  /// Global west-to-east catalogue column used to find successor tiles.
  int64_t column;
};

/// Fixed byte offsets for the retained uint16 tracing specialization.  This
/// layout is mirrored exactly by `QuantizedTerrainLayout` in panorama.metal.
struct QuantizedTerrainLayout {
  /// Byte stride between fixed-size atlas records.
  uint32_t record_stride;
  /// Byte offset from a record to its uint16 vertex array.
  uint32_t vertex_offset;
  /// Byte offset from a record to its signed decimetre elevation base.
  uint32_t elevation_base_offset;
};

} // namespace panorama
