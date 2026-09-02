#pragma once

#include "raytrace_config.h"
#include "tile_geometry.h"
#include "tile_manager.h"
#include "tile_preparer.h"

#import <Metal/Metal.h>

#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace panorama {

/// Private TileManager implementation: a bounded fixed-stride terrain atlas
/// with synchronous LRU replacement.
///
/// The cache owns the GPU-visible vertex/mipmap buffers and variant-to-slot
/// map. Between completed frontier commands it installs all currently
/// prepared tiles, waiting for custom I/O, optional fixed-point conversion,
/// and GPU mipmap generation before publishing resident entries.
class ResidentTileCache {
public:
  /// Allocate the atlas and install the observer tile in slot zero.
  ///
  /// The observer payload is loaded directly or staged for conversion according
  /// to the prepared tile's representation and trace configuration.
  ResidentTileCache(
      id<MTLDevice> device,
      std::span<const TerrainSource> sources,
      const TileGeometry &origin,
      TileKey origin_key,
      const RaytraceConfig &config,
      bool retain_quantized,
      uint32_t slot_capacity,
      Timer &timer
  );

  ResidentTileCache(const ResidentTileCache &) = delete;
  ResidentTileCache &operator=(const ResidentTileCache &) = delete;

  /// Release atlas and command resources after synchronous installation work.
  ~ResidentTileCache();

  /// Return the slot holding this source/LOD variant, or `slot_capacity()`.
  [[nodiscard]] uint32_t slot_for_variant(TileVariant variant) const;

  /// Install prepared tiles and return the source/LOD variants published to safe slots.
  [[nodiscard]] std::vector<TileVariant> install_prepared(
      AsyncTilePreparer &preparer,
      std::span<const uint8_t> pinned_slots,
      Timer &timer
  );

  /// Mark the supplied resident slots as recently used by the GPU frontier.
  void record_slot_use(std::span<const uint32_t> slots);

  /// Rebase every resident tile's GPU metadata around a moved observer while
  /// retaining all atlas payloads and residency state.
  void rebase_observer(ObserverLocation observer);

  /// Return the atlas buffers and layout metadata for the next GPU dispatch.
  [[nodiscard]] TileManagerBindings bindings() const;

  /// Return the number of fixed-size resident atlas slots.
  [[nodiscard]] uint32_t slot_capacity() const;

  /// Return final cache counters for command-line diagnostics.
  [[nodiscard]] TileManagerStatistics statistics() const;

private:
  struct State;
  std::unique_ptr<State> state_;
};

} // namespace panorama
