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
      const RaytraceParameters &parameters,
      std::span<const uint32_t> lod_by_source,
      uint32_t resident_slot_capacity,
      uint32_t observer_slot
  );

  /// Construct an initially empty scheduler for rays whose first source is
  /// supplied as deferred work (for example, per-collision shadow rays).
  HostFrontier(
      const TerrainCatalogue &catalogue,
      size_t ray_capacity,
      uint32_t num_levels,
      std::span<const uint32_t> lod_by_source,
      uint32_t resident_slot_capacity,
      std::span<const float> scheduling_distances = {}
  );

  /// Clear pending-request state for matching variants published by the atlas cache.
  void mark_installed(std::span<const TerrainTileVariant> variants);

  /// Consume new GPU continuations, activate nearby work, and request sources.
  [[nodiscard]] uint32_t activate_resident(
      id<MTLBuffer> buffer,
      uint32_t count,
      ResidentTileCache &cache,
      AsyncTilePreparer &preparer,
      std::span<const DeferredRayWork> incoming = {}
  );

  /// Update LRU use for the resident slots referenced by the active frontier.
  void record_active_slot_use(ResidentTileCache &cache) const;

  /// Return whether unresolved work is waiting for nonresident terrain.
  [[nodiscard]] bool has_deferred_work() const;

#if defined(PANORAMA_DEBUG_VALIDATION)
  /// Check that a GPU frontier contains at most one segment per ray.
  void validate_frontier(id<MTLBuffer> buffer, uint32_t count, const char *name);

  /// Check the same invariant across host state and new GPU continuations.
  void validate_deferred_work(std::span<const DeferredRayWork> incoming = {});
#endif

private:
  struct SourceBucket {
    std::vector<DeferredRayWork> pending;
    std::vector<DeferredRayWork> waiting;
    float minimum_pending_distance;
  };

  const TerrainCatalogue &catalogue_;
  size_t ray_capacity_;
  uint32_t num_levels_;
  /// Immutable per-source terrain LOD plan selected by the owning session.
  std::span<const uint32_t> lod_by_source_;
  /// Optional per-ray priority independent of the ray-local traversal distance.
  std::span<const float> scheduling_distances_;
  /// Deferred rays grouped by their next required catalogue source.
  std::vector<SourceBucket> source_buckets_;
  /// Sources with nonempty pending buckets, ordered only by insertion.
  std::vector<uint32_t> pending_sources_;
  std::vector<uint8_t> source_is_pending_;
  size_t pending_count_ = 0U;
  size_t waiting_count_ = 0U;
  /// Sources with preparation already queued, loading, or awaiting installation.
  std::vector<uint8_t> request_outstanding_;
  /// Scratch state reused while routing one GPU continuation buffer.
  std::vector<float> request_distances_;
  std::vector<uint32_t> request_sources_;
  std::vector<uint32_t> activation_slots_;
  /// Unique resident slots referenced by the current active GPU frontier.
  std::vector<uint32_t> active_slots_;
  std::vector<uint8_t> active_slot_seen_;
#if defined(PANORAMA_DEBUG_VALIDATION)
  std::vector<uint8_t> claimed_ray_;
#endif
};

} // namespace panorama
