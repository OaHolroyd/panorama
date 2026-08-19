#pragma once

#include "raytrace_setup.h"
#include "tile_preparer.h"

#import <Metal/Metal.h>

#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace panorama {

/// One immutable tile origin stored beside the corresponding atlas payload.
///
/// All other raytrace parameters are common to compatible atlas slots and are
/// supplied once per GPU frontier dispatch.
struct ResidentTile {
  float tile_x_min;
  float tile_y_min;
  int64_t row;
  int64_t column;
};

/// One open-addressed GPU lookup-table entry for a resident global tile key.
struct ResidentTileHashEntry {
  int64_t row;
  int64_t column;
  uint32_t slot;
  uint32_t occupied;
};

/// Host-visible Metal buffers and dimensions required by a frontier dispatch.
struct ResidentTileCacheBindings {
  id<MTLBuffer> mipmap_atlas;
  id<MTLBuffer> vertex_atlas;
  id<MTLBuffer> metadata;
  id<MTLBuffer> hash;
  uint32_t hash_slot_count;
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
/// The cache owns the GPU-visible vertex/mipmap buffers, source-to-slot maps,
/// and resident key hash. Between completed frontier commands it installs all
/// currently prepared tiles, waiting for custom I/O, fixed-point conversion,
/// and GPU mipmap generation before publishing their resident hash entries.
class ResidentTileCache {
public:
  /// Allocate the atlas and install the observer tile in slot zero.
  ///
  /// A GeoTIFF observer is already resident on the CPU. A custom observer
  /// supplies metadata only and is staged and converted from its file.
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

  /// Install prepared tiles into safe slots before the next frontier command.
  void install_prepared(
      AsyncTilePreparer &preparer,
      std::span<const uint8_t> pinned_slots,
      Timer &timer
  );

  /// Mark the supplied resident slots as recently used and return their pin mask.
  [[nodiscard]] std::vector<uint8_t> pin_slots(std::span<const uint32_t> slots, bool record_use);

  /// Return the atlas buffers and resident hash for the next GPU dispatch.
  [[nodiscard]] ResidentTileCacheBindings bindings() const;

  /// Return the number of fixed-size resident atlas slots.
  [[nodiscard]] uint32_t slot_capacity() const;

  /// Return the number of slots currently populated with a source tile.
  [[nodiscard]] uint32_t resident_tile_count() const;

  /// Return final cache counters for command-line diagnostics.
  [[nodiscard]] ResidentTileCacheStatistics statistics() const;

private:
  struct State;
  std::unique_ptr<State> state_;
};

} // namespace panorama
