#include "host_frontier.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace panorama {
namespace {

/// Convert a one-based terrain LOD into the deepest valid maximum-mipmap
/// level for that representation. LOD 1 keeps the reference tile unchanged.
[[nodiscard]] uint32_t mipmap_levels_for_lod(uint32_t base_levels, uint32_t lod) {
  if (base_levels == 0U || lod == 0U || lod > base_levels) {
    throw std::out_of_range("Terrain LOD exceeds the available mipmap hierarchy");
  }
  return base_levels - (lod - 1U);
}

} // namespace

HostFrontier::HostFrontier(
    const TerrainCatalogue &catalogue,
    std::span<const RayDirection> rays,
    const RaytraceParameters &parameters,
    std::span<const uint32_t> lod_by_source,
    uint32_t resident_slot_capacity,
    uint32_t observer_slot
)
    : catalogue_(catalogue), ray_capacity_(rays.size()), num_levels_(parameters.num_levels),
      lod_by_source_(lod_by_source), scheduling_distances_(),
      source_buckets_(catalogue.sources().size(), {{}, {}, std::numeric_limits<float>::infinity()}),
      source_is_pending_(catalogue.sources().size(), 0U),
      request_outstanding_(catalogue.sources().size(), 0U),
      request_distances_(catalogue.sources().size(), std::numeric_limits<float>::infinity()),
      activation_slots_(catalogue.sources().size(), std::numeric_limits<uint32_t>::max()),
      active_slots_{observer_slot}, active_slot_seen_(resident_slot_capacity, 0U)
#if defined(PANORAMA_DEBUG_VALIDATION)
      ,
      claimed_ray_(rays.size(), 0U)
#endif
{
  if (rays.empty() || rays.size() != parameters.ray_count ||
      lod_by_source_.size() != catalogue.sources().size() || resident_slot_capacity == 0U ||
      observer_slot >= resident_slot_capacity) {
    throw std::invalid_argument("Host frontier requires one direction per output ray");
  }
  // A persistent session may have moved the observer source away from slot
  // zero before this frame. The first frontier still references exactly that
  // one resident slot.
  active_slot_seen_[observer_slot] = 1U;
}

HostFrontier::HostFrontier(
    const TerrainCatalogue &catalogue,
    size_t ray_capacity,
    uint32_t num_levels,
    std::span<const uint32_t> lod_by_source,
    uint32_t resident_slot_capacity,
    std::span<const float> scheduling_distances
)
    : catalogue_(catalogue), ray_capacity_(ray_capacity), num_levels_(num_levels),
      lod_by_source_(lod_by_source), scheduling_distances_(scheduling_distances),
      source_buckets_(catalogue.sources().size(), {{}, {}, std::numeric_limits<float>::infinity()}),
      source_is_pending_(catalogue.sources().size(), 0U),
      request_outstanding_(catalogue.sources().size(), 0U),
      request_distances_(catalogue.sources().size(), std::numeric_limits<float>::infinity()),
      activation_slots_(catalogue.sources().size(), std::numeric_limits<uint32_t>::max()),
      active_slot_seen_(resident_slot_capacity, 0U)
#if defined(PANORAMA_DEBUG_VALIDATION)
      ,
      claimed_ray_(ray_capacity, 0U)
#endif
{
  if (ray_capacity == 0U || num_levels == 0U ||
      lod_by_source_.size() != catalogue.sources().size() || resident_slot_capacity == 0U ||
      (!scheduling_distances.empty() && scheduling_distances.size() != ray_capacity)) {
    throw std::invalid_argument("Host frontier requires a valid ray and terrain capacity");
  }
}

void HostFrontier::mark_installed(std::span<const TerrainTileVariant> variants) {
  for (TerrainTileVariant variant : variants) {
    const uint32_t source_index = variant.source_index;
    if (source_index >= lod_by_source_.size() || variant.lod != lod_by_source_[source_index]) {
      // A persistent preparer can complete an obsolete variant after the
      // observer has moved. It is valid cache content, but not this
      // frontier's pending request.
      continue;
    }
    request_outstanding_.at(source_index) = 0U;
    SourceBucket &bucket = source_buckets_.at(source_index);
    if (bucket.waiting.empty()) {
      continue;
    }
    if (source_is_pending_[source_index] == 0U) {
      pending_sources_.push_back(source_index);
      source_is_pending_[source_index] = 1U;
    }
    for (const DeferredRayWork &work : bucket.waiting) {
      const float priority = scheduling_distances_.empty() ? work.entry_distance
                                                           : scheduling_distances_[work.ray_index];
      bucket.minimum_pending_distance = std::min(bucket.minimum_pending_distance, priority);
    }
    pending_count_ += bucket.waiting.size();
    waiting_count_ -= bucket.waiting.size();
    bucket.pending.insert(bucket.pending.end(), bucket.waiting.begin(), bucket.waiting.end());
    bucket.waiting.clear();
  }
}

uint32_t HostFrontier::activate_resident(
    id<MTLBuffer> buffer,
    uint32_t count,
    ResidentTileCache &cache,
    AsyncTilePreparer &preparer,
    std::span<const DeferredRayWork> incoming
) {
  auto *items = static_cast<RayWorkItem *>(buffer.contents);
  if (items == nullptr) {
    throw std::runtime_error("Could not map active frontier buffer");
  }
  float nearest_entry = std::numeric_limits<float>::infinity();
  for (uint32_t source_index : pending_sources_) {
    nearest_entry = std::min(nearest_entry, source_buckets_[source_index].minimum_pending_distance);
  }
  for (const DeferredRayWork &work : incoming) {
    if (work.ray_index >= ray_capacity_ || work.source_index >= source_buckets_.size()) {
      throw std::runtime_error("Deferred frontier contains an invalid ray or source index");
    }
    const float priority =
        scheduling_distances_.empty() ? work.entry_distance : scheduling_distances_[work.ray_index];
    nearest_entry = std::min(nearest_entry, priority);
  }
  // Keep independently advancing rays within one tile width of the nearest
  // work. This limits cache churn without restoring any projection-specific
  // column grouping.
  const float activation_limit = nearest_entry + static_cast<float>(catalogue_.grid().width);
  if (count == 0U) {
    for (uint32_t slot : active_slots_) {
      active_slot_seen_[slot] = 0U;
    }
    active_slots_.clear();
  }
  std::fill(
      activation_slots_.begin(),
      activation_slots_.end(),
      std::numeric_limits<uint32_t>::max()
  );
  request_sources_.clear();
  const auto slot_for_source = [&](uint32_t source_index) {
    uint32_t &slot = activation_slots_[source_index];
    if (slot == std::numeric_limits<uint32_t>::max()) {
      slot = cache.slot_for_variant({source_index, lod_by_source_[source_index]});
      if (slot != cache.slot_capacity()) {
        if (slot >= active_slot_seen_.size()) {
          throw std::logic_error("Active frontier references an invalid resident slot");
        }
        active_slot_seen_[slot] = 1U;
        active_slots_.push_back(slot);
      }
    }
    return slot;
  };
  const auto queue_request = [&](uint32_t source_index, float entry_distance) {
    if (request_outstanding_[source_index] != 0U) {
      return;
    }
    float &request_distance = request_distances_[source_index];
    if (!std::isfinite(request_distance)) {
      request_sources_.push_back(source_index);
    }
    request_distance = std::min(request_distance, entry_distance);
  };
  const auto append_pending = [&](const DeferredRayWork &work) {
    SourceBucket &bucket = source_buckets_[work.source_index];
    if (source_is_pending_[work.source_index] == 0U) {
      pending_sources_.push_back(work.source_index);
      source_is_pending_[work.source_index] = 1U;
    }
    bucket.pending.push_back(work);
    const float priority =
        scheduling_distances_.empty() ? work.entry_distance : scheduling_distances_[work.ray_index];
    bucket.minimum_pending_distance = std::min(bucket.minimum_pending_distance, priority);
    pending_count_++;
  };
  size_t retained_source_count = 0U;
  const size_t pending_source_count = pending_sources_.size();
  for (size_t source_position = 0U; source_position < pending_source_count; source_position++) {
    const uint32_t source_index = pending_sources_[source_position];
    SourceBucket &bucket = source_buckets_[source_index];
    if (bucket.minimum_pending_distance > activation_limit) {
      pending_sources_[retained_source_count++] = source_index;
      continue;
    }

    const uint32_t slot = slot_for_source(source_index);
    const bool resident = slot != cache.slot_capacity();
    const float request_distance = bucket.minimum_pending_distance;
    float remaining_minimum = std::numeric_limits<float>::infinity();
    size_t retained_work_count = 0U;
    const size_t work_count = bucket.pending.size();
    for (size_t work_index = 0U; work_index < work_count; work_index++) {
      const DeferredRayWork work = bucket.pending[work_index];
      const float priority = scheduling_distances_.empty() ? work.entry_distance
                                                           : scheduling_distances_[work.ray_index];
      if (priority > activation_limit) {
        bucket.pending[retained_work_count++] = work;
        remaining_minimum = std::min(remaining_minimum, priority);
        continue;
      }
      pending_count_--;
      if (!resident) {
        bucket.waiting.push_back(work);
        waiting_count_++;
      } else {
        if (count >= ray_capacity_) {
          throw std::runtime_error("GPU frontier exceeds the ray frontier capacity");
        }
        items[count] = {
            slot,
            work.ray_index,
            mipmap_levels_for_lod(num_levels_, lod_by_source_[source_index]),
            work.entry_distance,
        };
        count++;
      }
    }
    bucket.pending.resize(retained_work_count);
    bucket.minimum_pending_distance = remaining_minimum;
    if (retained_work_count != 0U) {
      pending_sources_[retained_source_count++] = source_index;
    } else {
      source_is_pending_[source_index] = 0U;
    }
    if (resident) {
      request_outstanding_[source_index] = 0U;
    } else {
      queue_request(source_index, request_distance);
    }
  }
  pending_sources_.resize(retained_source_count);

  // Newly emitted continuations already reside in a mapped shared buffer.
  // Route work inside the current distance window directly instead of first
  // copying every ray through a persistent pending bucket.
  for (const DeferredRayWork &work : incoming) {
    const float priority =
        scheduling_distances_.empty() ? work.entry_distance : scheduling_distances_[work.ray_index];
    if (priority > activation_limit) {
      append_pending(work);
      continue;
    }
    const uint32_t slot = slot_for_source(work.source_index);
    if (slot == cache.slot_capacity()) {
      source_buckets_[work.source_index].waiting.push_back(work);
      waiting_count_++;
      queue_request(work.source_index, priority);
      continue;
    }
    request_outstanding_[work.source_index] = 0U;
    if (count >= ray_capacity_) {
      throw std::runtime_error("GPU frontier exceeds the ray frontier capacity");
    }
    items[count] = {
        slot,
        work.ray_index,
        mipmap_levels_for_lod(num_levels_, lod_by_source_[work.source_index]),
        work.entry_distance,
    };
    count++;
  }
  for (uint32_t source_index : request_sources_) {
    preparer.request(source_index, lod_by_source_[source_index], request_distances_[source_index]);
    request_outstanding_[source_index] = 1U;
    request_distances_[source_index] = std::numeric_limits<float>::infinity();
  }
  return count;
}

void HostFrontier::record_active_slot_use(ResidentTileCache &cache) const {
  cache.record_slot_use(active_slots_);
}

bool HostFrontier::has_deferred_work() const {
  return pending_count_ != 0U || waiting_count_ != 0U;
}

#if defined(PANORAMA_DEBUG_VALIDATION)
void HostFrontier::validate_frontier(id<MTLBuffer> buffer, uint32_t count, const char *name) {
  if (count > ray_capacity_) {
    throw std::runtime_error(std::string(name) + " exceeds the ray frontier capacity");
  }
  const auto *items = static_cast<const RayWorkItem *>(buffer.contents);
  if (items == nullptr) {
    throw std::runtime_error(std::string("Could not map ") + name);
  }
  std::fill(claimed_ray_.begin(), claimed_ray_.end(), 0U);
  std::vector<uint8_t> observed_slots(active_slot_seen_.size(), 0U);
  for (uint32_t index = 0U; index < count; index++) {
    const uint32_t ray = items[index].ray_index;
    const uint32_t slot = items[index].slot;
    if (ray >= ray_capacity_ || claimed_ray_[ray] != 0U) {
      throw std::runtime_error(std::string(name) + " violates the one-segment-per-ray invariant");
    }
    if (slot >= active_slot_seen_.size() || active_slot_seen_[slot] == 0U) {
      throw std::runtime_error(std::string(name) + " has inconsistent active-slot tracking");
    }
    claimed_ray_[ray] = 1U;
    observed_slots[slot] = 1U;
  }
  for (uint32_t slot : active_slots_) {
    if (slot >= observed_slots.size() || observed_slots[slot] == 0U) {
      throw std::runtime_error(std::string(name) + " has inconsistent active-slot tracking");
    }
  }
}

void HostFrontier::validate_deferred_work(std::span<const DeferredRayWork> incoming) {
  if (pending_count_ + waiting_count_ + incoming.size() > ray_capacity_) {
    throw std::runtime_error("Deferred frontier exceeds the ray frontier capacity");
  }
  std::fill(claimed_ray_.begin(), claimed_ray_.end(), 0U);
  std::vector<uint8_t> listed_source(source_buckets_.size(), 0U);
  for (uint32_t source_index : pending_sources_) {
    if (source_index >= source_buckets_.size() || listed_source[source_index] != 0U ||
        source_is_pending_[source_index] == 0U) {
      throw std::runtime_error("Pending source list contains an invalid or duplicate bucket");
    }
    listed_source[source_index] = 1U;
  }
  const auto claim = [this](const DeferredRayWork &deferred) {
    if (deferred.ray_index >= ray_capacity_ || deferred.source_index >= source_buckets_.size() ||
        claimed_ray_[deferred.ray_index] != 0U) {
      throw std::runtime_error("Deferred frontier violates the one-segment-per-ray invariant");
    }
    claimed_ray_[deferred.ray_index] = 1U;
  };
  for (const DeferredRayWork &work : incoming) {
    claim(work);
  }
  size_t observed_pending = 0U;
  size_t observed_waiting = 0U;
  for (uint32_t source_index = 0U; source_index < source_buckets_.size(); source_index++) {
    const SourceBucket &bucket = source_buckets_[source_index];
    if ((bucket.pending.empty() ? 0U : 1U) != source_is_pending_[source_index] ||
        source_is_pending_[source_index] != listed_source[source_index]) {
      throw std::runtime_error("Deferred source bucket has inconsistent pending state");
    }
    observed_pending += bucket.pending.size();
    observed_waiting += bucket.waiting.size();
    for (const DeferredRayWork &work : bucket.pending) {
      if (work.source_index != source_index) {
        throw std::runtime_error("Deferred ray is stored in the wrong source bucket");
      }
      claim(work);
    }
    for (const DeferredRayWork &work : bucket.waiting) {
      if (work.source_index != source_index) {
        throw std::runtime_error("Waiting ray is stored in the wrong source bucket");
      }
      claim(work);
    }
  }
  if (observed_pending != pending_count_ || observed_waiting != waiting_count_) {
    throw std::runtime_error("Deferred source-bucket counts are inconsistent");
  }
}
#endif

} // namespace panorama
