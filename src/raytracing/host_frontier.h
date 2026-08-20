#pragma once

#include "raytrace_gpu.h"
#include "terrain_catalogue.h"
#include "tile_preparer.h"

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace panorama {

/// Host-side state for unresolved terrain continuations between GPU passes.
///
/// The GPU stores work for resident tiles in a reusable frontier buffer. This
/// component owns complementary deferred rays, groups them by required source,
/// and requests each source once from the asynchronous preparer.
class HostFrontier {
public:
  /// Construct the scheduler state for one fixed observer and ray direction set.
  HostFrontier(
      const TerrainCatalogue &catalogue,
      std::span<const RayDirection> rays,
      const RaytraceParameters &parameters
  );

  /// Append GPU-emitted continuations grouped by their next required source.
  void append_deferred(std::span<const DeferredRayWork> deferred);

  /// Clear pending-request state for sources just published by the atlas cache.
  void mark_installed(std::span<const uint32_t> source_indices);

  /// Activate nearby source buckets and request any required nonresident tiles.
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
  /// Check that a GPU frontier contains at most one segment per ray.
  void validate_frontier(id<MTLBuffer> buffer, uint32_t count, const char *name);

  /// Check the same invariant for host-resident deferred continuations.
  void validate_deferred_work();
#endif

private:
  struct SourceBucket {
    std::vector<DeferredRayWork> pending;
    std::vector<DeferredRayWork> waiting;
    float minimum_pending_distance;
  };

  const TerrainCatalogue &catalogue_;
  std::span<const RayDirection> rays_;
  const RaytraceParameters &parameters_;
  /// Deferred rays grouped by their next required catalogue source.
  std::vector<SourceBucket> source_buckets_;
  /// Sources with nonempty pending buckets, ordered only by insertion.
  std::vector<uint32_t> pending_sources_;
  std::vector<uint8_t> source_is_pending_;
  size_t pending_count_ = 0U;
  size_t waiting_count_ = 0U;
  /// Sources with preparation already queued, loading, or awaiting installation.
  std::vector<uint8_t> request_outstanding_;
#if defined(PANORAMA_DEBUG_VALIDATION)
  std::vector<uint8_t> claimed_ray_;
#endif
};

} // namespace panorama
