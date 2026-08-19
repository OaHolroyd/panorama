#include "host_frontier.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace panorama {
namespace {

constexpr float kElevationCullingMargin = 1.0F;

/// Evaluate the curved ray using the same nested form as the Metal kernel.
[[nodiscard]] float curved_ray_elevation(
    float origin,
    float slope,
    float curvature,
    float distance
) {
  return std::fma(curvature, distance * distance, std::fma(slope, distance, origin));
}

/// Return the lowest elevation reached by one curved ray over a tile interval.
[[nodiscard]] float minimum_curved_ray_elevation(
    float origin,
    float slope,
    float curvature,
    float entry_distance,
    float exit_distance
) {
  const float minimum_distance =
      curvature > 0.0F
          ? std::clamp(-slope / (2.0F * curvature), entry_distance, exit_distance)
          : (slope > 0.0F ? entry_distance : exit_distance);
  return curved_ray_elevation(origin, slope, curvature, minimum_distance);
}

/// Return the first boundary of a catalogue tile strictly beyond its entry.
/// Float32 origins and arithmetic intentionally mirror the resident GPU tile.
[[nodiscard]] float tile_exit_distance(
    const TileGrid &grid,
    TileKey key,
    const ObserverLocation &observer,
    HorizontalDirection direction,
    const RaytraceParameters &parameters,
    float entry_distance
) {
  const float tile_x_min = static_cast<float>(
      grid.origin_x + static_cast<double>(key.column) * grid.width - observer.easting
  );
  const float tile_y_min = static_cast<float>(
      grid.origin_y - (static_cast<double>(key.row) + 1.0) * grid.width - observer.northing
  );
  const uint32_t cell_count = 1U << (parameters.num_levels - 1U);
  const float tile_width = static_cast<float>(cell_count) * parameters.cell_size;
  const float tile_x_max = tile_x_min + tile_width;
  const float tile_y_max = tile_y_min + tile_width;
  float tile_exit = std::numeric_limits<float>::infinity();
  if (direction.x > 0.0F) {
    const float candidate = tile_x_max / direction.x;
    if (candidate > entry_distance) {
      tile_exit = std::min(tile_exit, candidate);
    }
  } else if (direction.x < 0.0F) {
    const float candidate = tile_x_min / direction.x;
    if (candidate > entry_distance) {
      tile_exit = std::min(tile_exit, candidate);
    }
  }
  if (direction.y > 0.0F) {
    const float candidate = tile_y_max / direction.y;
    if (candidate > entry_distance) {
      tile_exit = std::min(tile_exit, candidate);
    }
  } else if (direction.y < 0.0F) {
    const float candidate = tile_y_min / direction.y;
    if (candidate > entry_distance) {
      tile_exit = std::min(tile_exit, candidate);
    }
  }
  return tile_exit;
}

} // namespace

HostFrontier::HostFrontier(
    const RaytraceConfig &config,
    const TerrainCatalogue &catalogue,
    std::span<const HorizontalDirection> directions,
    std::span<const float> slopes,
    const RaytraceParameters &parameters
) : config_(config), catalogue_(catalogue), directions_(directions), slopes_(slopes),
    parameters_(parameters), request_outstanding_(catalogue.sources().size(), 0U)
#if defined(PANORAMA_DEBUG_VALIDATION)
  , claimed_azimuth_(config.num_azimuth, 0U)
#endif
{
  if (directions_.size() != config_.num_azimuth || slopes_.size() != config_.num_polar ||
      !std::is_sorted(slopes_.begin(), slopes_.end())) {
    throw std::invalid_argument("Host frontier requires ordered ray directions and slopes");
  }
}

void HostFrontier::append_deferred(std::span<const DeferredTileWork> deferred) {
  deferred_.insert(deferred_.end(), deferred.begin(), deferred.end());
}

void HostFrontier::mark_installed(std::span<const uint32_t> source_indices) {
  for (uint32_t source_index : source_indices) {
    request_outstanding_.at(source_index) = 0U;
  }
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
  for (DeferredTileWork deferred : deferred_) {
    if (deferred.azimuth >= directions_.size() || deferred.first_polar >= slopes_.size()) {
      throw std::runtime_error("Deferred frontier contains an invalid ray index");
    }
    const HorizontalDirection direction = directions_[deferred.azimuth];
    const float slope = slopes_[deferred.first_polar];

    // A deferred suffix covers this ray and every larger polar slope. Advance
    // across any sequence of tiles whose maxima lie below the lowest member.
    for (;;) {
      if (deferred.entry_distance >= parameters_.max_distance) {
        break;
      }
      const float elevation_at_entry = curved_ray_elevation(
          parameters_.observer_elevation,
          slope,
          parameters_.curvature_coefficient,
          deferred.entry_distance
      );
      const float elevation_derivative =
          slope + 2.0F * parameters_.curvature_coefficient * deferred.entry_distance;
      if (std::isfinite(parameters_.global_maximum_elevation) &&
          elevation_derivative >= 0.0F &&
          elevation_at_entry >
              parameters_.global_maximum_elevation + kElevationCullingMargin) {
        globally_skipped_tiles_++;
        break;
      }

      float x = deferred.entry_distance * direction.x;
      float y = deferred.entry_distance * direction.y;
      const float nudge = std::max(
          1e-3F * parameters_.cell_size,
          8.0F * std::numeric_limits<float>::epsilon() *
              std::max(1.0F, std::max(std::abs(x), std::abs(y)))
      );
      if (direction.x != 0.0F) {
        x += std::copysign(nudge, direction.x);
      }
      if (direction.y != 0.0F) {
        y += std::copysign(nudge, direction.y);
      }
      const TileKey key = tile_key_at(
          catalogue_.grid(),
          config_.observer.easting + static_cast<double>(x),
          config_.observer.northing + static_cast<double>(y)
      );
      const std::optional<uint32_t> source_index = catalogue_.find_source(key);
      if (!source_index.has_value()) {
        break;
      }
      const TerrainSource &source = catalogue_.sources()[*source_index];
      const float exit_distance = tile_exit_distance(
          catalogue_.grid(),
          key,
          config_.observer,
          direction,
          parameters_,
          deferred.entry_distance
      );
      if (source.maximum_elevation.has_value() && std::isfinite(exit_distance) &&
          minimum_curved_ray_elevation(
              parameters_.observer_elevation,
              slope,
              parameters_.curvature_coefficient,
              deferred.entry_distance,
              exit_distance
          ) > *source.maximum_elevation + kElevationCullingMargin) {
        locally_skipped_tiles_++;
        deferred.entry_distance = exit_distance;
        continue;
      }

      const uint32_t slot = cache.slot_for_source(*source_index);
      if (slot == cache.slot_capacity()) {
        if (request_outstanding_[*source_index] == 0U) {
          preparer.request(*source_index, deferred.entry_distance);
          request_outstanding_[*source_index] = 1U;
        }
        still_waiting.push_back(deferred);
        break;
      }
      request_outstanding_[*source_index] = 0U;
      if (count >= config_.num_azimuth) {
        throw std::runtime_error("GPU frontier exceeds the azimuth frontier capacity");
      }
      items[count] = {slot, deferred.azimuth, deferred.first_polar, parameters_.num_levels,
                      deferred.entry_distance};
      count++;
      break;
    }
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

HostFrontierStatistics HostFrontier::statistics() const {
  return {locally_skipped_tiles_, globally_skipped_tiles_};
}

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
