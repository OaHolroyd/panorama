#pragma once

#include "raytrace_gpu.h"
#include "terrain_catalogue.h"
#include "tile_preparer.h"

#include <cstdint>
#include <optional>
#include <span>
#include <vector>

namespace panorama {

/// Host-side state for unresolved terrain continuations between GPU passes.
///
/// The GPU stores work for resident tiles in its double-buffered frontier.
/// This component owns the complementary deferred suffixes, requests missing
/// sources from the asynchronous preparer, and enforces the current one-
/// unresolved-suffix-per-azimuth scheduling invariant.
class HostFrontier {
public:
  /// Construct the scheduler state for one fixed observer and ray direction set.
  HostFrontier(
      const RaytraceConfig &config,
      const TerrainCatalogue &catalogue,
      std::span<const HorizontalDirection> directions,
      const RaytraceParameters &parameters
  );

  /// Request the eight catalogue neighbours surrounding the observer tile.
  void prefetch_observer_neighbours(AsyncTilePreparer &preparer) const;

  /// Append GPU-emitted continuations whose successor terrain was absent.
  void append_deferred(std::span<const DeferredTileWork> deferred);

  /// Activate deferred work whose successor source now occupies an atlas slot.
  [[nodiscard]] uint32_t activate_resident(
      id<MTLBuffer> buffer,
      uint32_t count,
      ResidentTileCache &cache,
      AsyncTilePreparer &preparer
  );

  /// Return the atlas slots read by a frontier and optionally update their LRU use.
  [[nodiscard]] std::vector<uint8_t> pin_frontier(
      id<MTLBuffer> buffer,
      uint32_t count,
      ResidentTileCache &cache,
      bool record_use
  ) const;

  /// Return whether unresolved work is waiting for nonresident terrain.
  [[nodiscard]] bool has_deferred_work() const;

#if defined(PANORAMA_DEBUG_VALIDATION)
  /// Check that a GPU frontier contains at most one suffix per azimuth.
  void validate_frontier(id<MTLBuffer> buffer, uint32_t count, const char *name);

  /// Check the same invariant for host-resident deferred continuations.
  void validate_deferred_work();
#endif

private:
  const RaytraceConfig &config_;
  const TerrainCatalogue &catalogue_;
  std::span<const HorizontalDirection> directions_;
  const RaytraceParameters &parameters_;
  std::vector<DeferredTileWork> deferred_;

#if defined(PANORAMA_DEBUG_VALIDATION)
  std::vector<uint8_t> claimed_azimuth_;
#endif
};

} // namespace panorama
