#pragma once

#include "raytrace_config.h"
#include "tile_preparer.h"

#import <Metal/Metal.h>

#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace panorama {

/// Immutable spatial metadata stored beside one resident atlas payload.
///
/// All other raytrace parameters are common to compatible atlas slots and are
/// supplied once per GPU frontier dispatch.
struct ResidentTile {
  float tile_x_min;
  float tile_y_min;
  float maximum_elevation;
  uint32_t padding;
  int64_t row;
  int64_t column;
};

/// Fixed byte offsets used by the uint16 trace specialization.
/// This must remain identical to `QuantizedTerrainLayout` in panorama.metal.
struct QuantizedTerrainLayout {
  uint32_t record_stride;
  uint32_t vertex_offset;
  uint32_t elevation_base_offset;
};

/// Host-visible Metal buffers and dimensions required by a frontier dispatch.
struct ResidentTileCacheBindings {
  id<MTLBuffer> mipmap_atlas;
  id<MTLBuffer> vertex_atlas;
  id<MTLBuffer> metadata;
  QuantizedTerrainLayout quantized_layout;
};

/// Final counters describing atlas residency, copies, and evictions.
struct ResidentTileCacheStatistics {
  uint64_t installations;
  uint64_t bytes_copied;
  uint64_t bytes_loaded_with_metal_io;
  uint64_t evictions;
  uint32_t resident_tiles;
  uint32_t slot_capacity;
};

/// A bounded fixed-stride terrain atlas with synchronous LRU replacement.
///
/// The cache owns the GPU-visible vertex/mipmap buffers and source-to-slot
/// maps. Between completed frontier commands it installs all currently
/// prepared tiles, waiting for custom I/O, optional fixed-point conversion,
/// and GPU mipmap generation before publishing resident entries.
class ResidentTileCache {
public:
  /// Allocate the atlas and install the observer tile in slot zero.
  ///
  /// A GeoTIFF observer's payload is already resident on the CPU. A custom
  /// observer supplies metadata only and is loaded directly or staged for
  /// conversion according to its representation and the trace configuration.
  ResidentTileCache(
      id<MTLDevice> device,
      std::span<const TerrainSource> sources,
      const LoadedTile &origin,
      TileKey origin_key,
      const RaytraceConfig &config,
      uint32_t slot_capacity,
      Timer &timer
  );

  ResidentTileCache(const ResidentTileCache &) = delete;
  ResidentTileCache &operator=(const ResidentTileCache &) = delete;

  /// Release atlas and command resources after synchronous installation work.
  ~ResidentTileCache();

  /// Return the resident slot for a source, or `slot_capacity()` if absent.
  [[nodiscard]] uint32_t slot_for_source(uint32_t source_index) const;

  /// Install prepared tiles and return the source indices published to safe slots.
  [[nodiscard]] std::vector<uint32_t> install_prepared(
      AsyncTilePreparer &preparer,
      std::span<const uint8_t> pinned_slots,
      Timer &timer
  );

  /// Mark the supplied resident slots as recently used by the GPU frontier.
  void record_slot_use(std::span<const uint32_t> slots);

  /// Return the atlas buffers and layout metadata for the next GPU dispatch.
  [[nodiscard]] ResidentTileCacheBindings bindings() const;

  /// Return the number of fixed-size resident atlas slots.
  [[nodiscard]] uint32_t slot_capacity() const;

  /// Return final cache counters for command-line diagnostics.
  [[nodiscard]] ResidentTileCacheStatistics statistics() const;

private:
  struct State;
  std::unique_ptr<State> state_;
};

} // namespace panorama
