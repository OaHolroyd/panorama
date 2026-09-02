#pragma once

#include "raytrace_gpu.h"
#include "tile_manager.h"

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace panorama {

/// Host-side state for unresolved terrain continuations between GPU passes.
///
/// The GPU stores work for resident tiles in a reusable frontier buffer. This
/// component owns complementary deferred rays, groups them by required source,
/// and asks TileManager to make each required variant resident.
class HostFrontier {
public:
  /// Construct the scheduler state for one fixed observer and ray direction set.
  HostFrontier(
      TileManager &tiles,
      std::span<const RayDirection> rays,
      const RaytraceParameters &parameters,
      uint32_t resident_slot_capacity,
      uint32_t observer_slot
  );

  /// Construct an initially empty scheduler for rays whose first source is
  /// supplied as deferred work (for example, per-collision shadow rays).
  HostFrontier(
      TileManager &tiles,
      size_t ray_capacity,
      uint32_t num_levels,
      uint32_t resident_slot_capacity,
      std::span<const float> scheduling_distances = {}
  );

  /// Clear pending-request state for variants published by TileManager.
  void mark_installed(std::span<const TileVariant> variants);

  /// Consume new GPU continuations, activate nearby work, and request sources.
  [[nodiscard]] uint32_t activate_resident(
      id<MTLBuffer> buffer,
      uint32_t count,
      std::span<const DeferredRayWork> incoming = {}
  );

  /// Update LRU use for the resident slots referenced by the active frontier.
  void record_active_slot_use() const;

  /// Return whether unresolved work is waiting for nonresident terrain.
  [[nodiscard]] bool has_deferred_work() const;

#if defined(PANORAMA_DEBUG_VALIDATION)
  /// Check that a GPU frontier contains at most one segment per ray.
  void validate_frontier(id<MTLBuffer> buffer, uint32_t count, const char *name);

  /// Check the same invariant across host state and new GPU continuations.
  void validate_deferred_work(std::span<const DeferredRayWork> incoming = {});
#endif

private:
  /// Deferred work for one catalogue source.
  struct SourceBucket {
    /// Work delayed only to keep activation near the closest ray segment.
    std::vector<DeferredRayWork> pending;
    /// Work close enough to run but blocked on a nonresident tile variant.
    std::vector<DeferredRayWork> waiting;
    /// Smallest scheduling distance in `pending`, or infinity when empty.
    float minimum_pending_distance;
  };

  /// Tile residency and request service shared with primary and shadow traces.
  TileManager &tiles_;
  /// Maximum number of active plus deferred ray segments.
  size_t ray_capacity_;
  /// LOD-1 hierarchy depth used to derive each variant's start level.
  uint32_t num_levels_;
  /// Optional per-ray priority independent of the ray-local traversal distance.
  std::span<const float> scheduling_distances_;
  /// Deferred rays grouped by their next required catalogue source.
  std::vector<SourceBucket> source_buckets_;
  /// Sources with nonempty pending buckets, ordered only by insertion.
  std::vector<uint32_t> pending_sources_;
  /// O(1) membership flags corresponding to `pending_sources_`.
  std::vector<uint8_t> source_is_pending_;
  /// Total work in every bucket's distance-delayed `pending` vector.
  size_t pending_count_ = 0U;
  /// Total work blocked in every bucket's nonresident `waiting` vector.
  size_t waiting_count_ = 0U;
  /// Sources with preparation already queued, loading, or awaiting installation.
  std::vector<uint8_t> request_outstanding_;
  /// Minimum priority accumulated for each source during one routing call.
  std::vector<float> request_distances_;
  /// Sources touched in `request_distances_`, avoiding a full-array scan.
  std::vector<uint32_t> request_sources_;
  /// Per-call source-to-slot lookup cache; the slot sentinel means unchecked.
  std::vector<uint32_t> activation_slots_;
  /// Unique resident slots referenced by the current active GPU frontier.
  std::vector<uint32_t> active_slots_;
  /// O(1) deduplication flags corresponding to `active_slots_`.
  std::vector<uint8_t> active_slot_seen_;
#if defined(PANORAMA_DEBUG_VALIDATION)
  /// Per-ray ownership marks reused by expensive debug invariant checks.
  std::vector<uint8_t> claimed_ray_;
#endif
};

} // namespace panorama
