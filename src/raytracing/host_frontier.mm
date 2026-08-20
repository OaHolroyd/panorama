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
      waiting_by_source_(catalogue.sources().size()),
      request_outstanding_(catalogue.sources().size(), 0U),
      request_distances_(catalogue.sources().size(), std::numeric_limits<float>::infinity())
#if defined(PANORAMA_DEBUG_VALIDATION)
      ,
      claimed_ray_(rays.size(), 0U)
#endif
{
  if (rays_.empty() || rays_.size() != parameters_.ray_count) {
    throw std::invalid_argument("Host frontier requires one direction per output ray");
  }
}

void HostFrontier::append_deferred(std::span<const DeferredRayWork> deferred) {
  pending_.insert(pending_.end(), deferred.begin(), deferred.end());
}

void HostFrontier::mark_installed(std::span<const uint32_t> source_indices) {
  for (uint32_t source_index : source_indices) {
    request_outstanding_.at(source_index) = 0U;
    std::vector<DeferredRayWork> &waiting = waiting_by_source_.at(source_index);
    waiting_count_ -= waiting.size();
    pending_.insert(pending_.end(), waiting.begin(), waiting.end());
    waiting.clear();
  }
}

uint32_t HostFrontier::activate_resident(
    id<MTLBuffer> buffer,
    uint32_t count,
    ResidentTileCache &cache,
    AsyncTilePreparer &preparer
) {
  auto *items = static_cast<RayWorkItem *>(buffer.contents);
  if (items == nullptr) {
    throw std::runtime_error("Could not map active frontier buffer");
  }
  const size_t source_count = catalogue_.sources().size();
  float nearest_entry = std::numeric_limits<float>::infinity();
  for (const DeferredRayWork &deferred : pending_) {
    nearest_entry = std::min(nearest_entry, deferred.entry_distance);
  }
  // Keep independently advancing rays within one tile width of the nearest
  // work. This limits cache churn without restoring any projection-specific
  // column grouping.
  const float activation_limit = nearest_entry + static_cast<float>(catalogue_.grid().width);
  request_sources_.clear();
  size_t retained_count = 0U;
  const size_t pending_count = pending_.size();
  for (size_t index = 0U; index < pending_count; index++) {
    const DeferredRayWork deferred = pending_[index];
    if (deferred.ray_index >= rays_.size() || deferred.source_index >= source_count) {
      throw std::runtime_error("Deferred frontier contains an invalid ray or source index");
    }
    if (deferred.entry_distance > activation_limit) {
      pending_[retained_count++] = deferred;
      continue;
    }
    const uint32_t slot = cache.slot_for_source(deferred.source_index);
    if (slot == cache.slot_capacity()) {
      waiting_count_++;
      waiting_by_source_[deferred.source_index].push_back(deferred);
      if (request_outstanding_[deferred.source_index] == 0U) {
        float &request_distance = request_distances_[deferred.source_index];
        if (!std::isfinite(request_distance)) {
          request_sources_.push_back(deferred.source_index);
        }
        request_distance = std::min(request_distance, deferred.entry_distance);
      }
      continue;
    }
    request_outstanding_[deferred.source_index] = 0U;
    if (count >= rays_.size()) {
      throw std::runtime_error("GPU frontier exceeds the ray frontier capacity");
    }
    items[count] = {slot, deferred.ray_index, parameters_.num_levels, deferred.entry_distance};
    count++;
  }
  pending_.resize(retained_count);
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

bool HostFrontier::has_deferred_work() const { return !pending_.empty() || waiting_count_ != 0U; }

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

void HostFrontier::validate_deferred_work() {
  if (pending_.size() + waiting_count_ > rays_.size()) {
    throw std::runtime_error("Deferred frontier exceeds the ray frontier capacity");
  }
  std::fill(claimed_ray_.begin(), claimed_ray_.end(), 0U);
  const auto claim = [this](const DeferredRayWork &deferred) {
    if (deferred.ray_index >= rays_.size() || claimed_ray_[deferred.ray_index] != 0U) {
      throw std::runtime_error("Deferred frontier violates the one-segment-per-ray invariant");
    }
    claimed_ray_[deferred.ray_index] = 1U;
  };
  for (const DeferredRayWork &deferred : pending_) {
    claim(deferred);
  }
  for (const std::vector<DeferredRayWork> &waiting : waiting_by_source_) {
    for (const DeferredRayWork &deferred : waiting) {
      claim(deferred);
    }
  }
}
#endif

} // namespace panorama
