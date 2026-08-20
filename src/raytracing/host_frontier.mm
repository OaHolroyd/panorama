#include "host_frontier.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace panorama {
HostFrontier::HostFrontier(
    const TerrainCatalogue &catalogue,
    std::span<const RayDirection> rays,
    const RaytraceParameters &parameters
)
    : catalogue_(catalogue), rays_(rays), parameters_(parameters),
      source_buckets_(
          catalogue.sources().size(),
          {{}, {}, std::numeric_limits<float>::infinity()}
      ),
      source_is_pending_(catalogue.sources().size(), 0U),
      request_outstanding_(catalogue.sources().size(), 0U),
      request_distances_(catalogue.sources().size(), std::numeric_limits<float>::infinity()),
      activation_slots_(catalogue.sources().size(), std::numeric_limits<uint32_t>::max())
#if defined(PANORAMA_DEBUG_VALIDATION)
      ,
      claimed_ray_(rays.size(), 0U)
#endif
{
  if (rays_.empty() || rays_.size() != parameters_.ray_count) {
    throw std::invalid_argument("Host frontier requires one direction per output ray");
  }
}

void HostFrontier::mark_installed(std::span<const uint32_t> source_indices) {
  for (uint32_t source_index : source_indices) {
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
      bucket.minimum_pending_distance =
          std::min(bucket.minimum_pending_distance, work.entry_distance);
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
    nearest_entry = std::min(
        nearest_entry, source_buckets_[source_index].minimum_pending_distance
    );
  }
  for (const DeferredRayWork &work : incoming) {
    if (work.ray_index >= rays_.size() || work.source_index >= source_buckets_.size()) {
      throw std::runtime_error("Deferred frontier contains an invalid ray or source index");
    }
    nearest_entry = std::min(nearest_entry, work.entry_distance);
  }
  // Keep independently advancing rays within one tile width of the nearest
  // work. This limits cache churn without restoring any projection-specific
  // column grouping.
  const float activation_limit = nearest_entry + static_cast<float>(catalogue_.grid().width);
  std::fill(
      activation_slots_.begin(),
      activation_slots_.end(),
      std::numeric_limits<uint32_t>::max()
  );
  request_sources_.clear();
  const auto slot_for_source = [&](uint32_t source_index) {
    uint32_t &slot = activation_slots_[source_index];
    if (slot == std::numeric_limits<uint32_t>::max()) {
      slot = cache.slot_for_source(source_index);
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
    bucket.minimum_pending_distance =
        std::min(bucket.minimum_pending_distance, work.entry_distance);
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
      if (work.entry_distance > activation_limit) {
        bucket.pending[retained_work_count++] = work;
        remaining_minimum = std::min(remaining_minimum, work.entry_distance);
        continue;
      }
      pending_count_--;
      if (!resident) {
        bucket.waiting.push_back(work);
        waiting_count_++;
      } else {
        if (count >= rays_.size()) {
          throw std::runtime_error("GPU frontier exceeds the ray frontier capacity");
        }
        items[count] = {slot, work.ray_index, parameters_.num_levels, work.entry_distance};
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
    if (work.entry_distance > activation_limit) {
      append_pending(work);
      continue;
    }
    const uint32_t slot = slot_for_source(work.source_index);
    if (slot == cache.slot_capacity()) {
      source_buckets_[work.source_index].waiting.push_back(work);
      waiting_count_++;
      queue_request(work.source_index, work.entry_distance);
      continue;
    }
    request_outstanding_[work.source_index] = 0U;
    if (count >= rays_.size()) {
      throw std::runtime_error("GPU frontier exceeds the ray frontier capacity");
    }
    items[count] = {slot, work.ray_index, parameters_.num_levels, work.entry_distance};
    count++;
  }
  for (uint32_t source_index : request_sources_) {
    preparer.request(source_index, request_distances_[source_index]);
    request_outstanding_[source_index] = 1U;
    request_distances_[source_index] = std::numeric_limits<float>::infinity();
  }
  return count;
}

std::vector<uint8_t> HostFrontier::pin_frontier(
    id<MTLBuffer> buffer,
    uint32_t count,
    ResidentTileCache &cache,
    bool record_use
) const {
  const auto *items = static_cast<const RayWorkItem *>(buffer.contents);
  if (items == nullptr) {
    throw std::runtime_error("Could not map active frontier buffer");
  }
  std::vector<uint8_t> seen(cache.slot_capacity(), 0U);
  std::vector<uint32_t> slots;
  slots.reserve(std::min(count, cache.slot_capacity()));
  for (uint32_t index = 0U; index < count; index++) {
    const uint32_t slot = items[index].slot;
    if (slot >= seen.size()) {
      throw std::runtime_error("Active frontier references an invalid resident slot");
    }
    if (seen[slot] == 0U) {
      seen[slot] = 1U;
      slots.push_back(slot);
    }
  }
  return cache.pin_slots(slots, record_use);
}

bool HostFrontier::has_deferred_work() const { return pending_count_ != 0U || waiting_count_ != 0U; }

#if defined(PANORAMA_DEBUG_VALIDATION)
void HostFrontier::validate_frontier(id<MTLBuffer> buffer, uint32_t count, const char *name) {
  if (count > rays_.size()) {
    throw std::runtime_error(std::string(name) + " exceeds the ray frontier capacity");
  }
  const auto *items = static_cast<const RayWorkItem *>(buffer.contents);
  if (items == nullptr) {
    throw std::runtime_error(std::string("Could not map ") + name);
  }
  std::fill(claimed_ray_.begin(), claimed_ray_.end(), 0U);
  for (uint32_t index = 0U; index < count; index++) {
    const uint32_t ray = items[index].ray_index;
    if (ray >= rays_.size() || claimed_ray_[ray] != 0U) {
      throw std::runtime_error(std::string(name) + " violates the one-segment-per-ray invariant");
    }
    claimed_ray_[ray] = 1U;
  }
}

void HostFrontier::validate_deferred_work(std::span<const DeferredRayWork> incoming) {
  if (pending_count_ + waiting_count_ + incoming.size() > rays_.size()) {
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
    if (deferred.ray_index >= rays_.size() || deferred.source_index >= source_buckets_.size() ||
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
