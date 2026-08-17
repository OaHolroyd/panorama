#include "host_frontier.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace panorama {

HostFrontier::HostFrontier(
    const RaytraceConfig &config,
    const TerrainCatalogue &catalogue,
    std::span<const HorizontalDirection> directions,
    const RaytraceParameters &parameters
) : config_(config), catalogue_(catalogue), directions_(directions), parameters_(parameters)
#if defined(PANORAMA_DEBUG_VALIDATION)
  , claimed_azimuth_(config.num_azimuth, 0U)
#endif
{}

void HostFrontier::prefetch_observer_neighbours(AsyncTilePreparer &preparer) const {
  const TileKey origin = catalogue_.origin().key;
  for (int64_t row_offset = -1; row_offset <= 1; row_offset++) {
    for (int64_t column_offset = -1; column_offset <= 1; column_offset++) {
      if (row_offset == 0 && column_offset == 0) {
        continue;
      }
      const TileKey neighbour = {origin.row + row_offset, origin.column + column_offset};
      const std::optional<uint32_t> source = catalogue_.find_source(neighbour);
      if (source.has_value()) {
        preparer.request(
            *source,
            static_cast<float>(tile_minimum_distance(catalogue_.grid(), neighbour, config_.observer))
        );
      }
    }
  }
}

void HostFrontier::append_deferred(std::span<const DeferredTileWork> deferred) {
  deferred_.insert(deferred_.end(), deferred.begin(), deferred.end());
}

uint32_t HostFrontier::activate_resident(
    id<MTLBuffer> buffer,
    uint32_t count,
    ResidentTileCache &cache,
    AsyncTilePreparer &preparer
) {
  auto *items = static_cast<TileWorkItem *>(buffer.contents);
  if (items == nullptr) {
    throw std::runtime_error("Could not map active frontier buffer");
  }
  std::vector<DeferredTileWork> still_waiting;
  still_waiting.reserve(deferred_.size());
  for (const DeferredTileWork &deferred : deferred_) {
    float x = deferred.entry_distance * directions_[deferred.azimuth].x;
    float y = deferred.entry_distance * directions_[deferred.azimuth].y;
    const float nudge = std::max(
        1e-3F * parameters_.cell_size,
        8.0F * std::numeric_limits<float>::epsilon() *
            std::max(1.0F, std::max(std::abs(x), std::abs(y)))
    );
    if (directions_[deferred.azimuth].x != 0.0F) {
      x += std::copysign(nudge, directions_[deferred.azimuth].x);
    }
    if (directions_[deferred.azimuth].y != 0.0F) {
      y += std::copysign(nudge, directions_[deferred.azimuth].y);
    }
    const TileKey key = tile_key_at(
        catalogue_.grid(),
        config_.observer.easting + static_cast<double>(x),
        config_.observer.northing + static_cast<double>(y)
    );
    const std::optional<uint32_t> source = catalogue_.find_source(key);
    if (!source.has_value()) {
      continue;
    }
    const uint32_t slot = cache.slot_for_source(*source);
    if (slot == cache.slot_capacity()) {
      preparer.request(*source, deferred.entry_distance);
      still_waiting.push_back(deferred);
      continue;
    }
    if (count >= config_.num_azimuth) {
      throw std::runtime_error("GPU frontier exceeds the azimuth frontier capacity");
    }
    items[count] = {slot, deferred.azimuth, deferred.first_polar, parameters_.num_levels,
                    deferred.entry_distance};
    count++;
  }
  deferred_ = std::move(still_waiting);
  return count;
}

std::vector<uint8_t> HostFrontier::pin_frontier(
    id<MTLBuffer> buffer,
    uint32_t count,
    ResidentTileCache &cache,
    bool record_use
) const {
  const auto *items = static_cast<const TileWorkItem *>(buffer.contents);
  if (items == nullptr) {
    throw std::runtime_error("Could not map active frontier buffer");
  }
  std::vector<uint32_t> slots;
  slots.reserve(count);
  for (uint32_t index = 0U; index < count; index++) {
    slots.push_back(items[index].slot);
  }
  return cache.pin_slots(slots, record_use);
}

bool HostFrontier::has_deferred_work() const { return !deferred_.empty(); }

#if defined(PANORAMA_DEBUG_VALIDATION)
void HostFrontier::validate_frontier(id<MTLBuffer> buffer, uint32_t count, const char *name) {
  if (count > config_.num_azimuth) {
    throw std::runtime_error(std::string(name) + " exceeds the azimuth frontier capacity");
  }
  const auto *items = static_cast<const TileWorkItem *>(buffer.contents);
  if (items == nullptr) {
    throw std::runtime_error(std::string("Could not map ") + name);
  }
  std::fill(claimed_azimuth_.begin(), claimed_azimuth_.end(), 0U);
  for (uint32_t index = 0U; index < count; index++) {
    const uint32_t azimuth = items[index].azimuth;
    if (azimuth >= config_.num_azimuth || claimed_azimuth_[azimuth] != 0U) {
      throw std::runtime_error(std::string(name) + " violates the one-suffix-per-azimuth invariant");
    }
    claimed_azimuth_[azimuth] = 1U;
  }
}

void HostFrontier::validate_deferred_work() {
  if (deferred_.size() > config_.num_azimuth) {
    throw std::runtime_error("Deferred frontier exceeds the azimuth frontier capacity");
  }
  std::fill(claimed_azimuth_.begin(), claimed_azimuth_.end(), 0U);
  for (const DeferredTileWork &deferred : deferred_) {
    if (deferred.azimuth >= config_.num_azimuth || claimed_azimuth_[deferred.azimuth] != 0U) {
      throw std::runtime_error("Deferred frontier violates the one-suffix-per-azimuth invariant");
    }
    claimed_azimuth_[deferred.azimuth] = 1U;
  }
}
#endif

} // namespace panorama
