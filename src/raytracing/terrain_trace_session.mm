#include "terrain_trace_session.h"

#include "host_frontier.h"
#include "loaded_tile.h"
#include "metal_tile.h"
#include "resident_tile_cache.h"
#include "terrain_catalogue.h"
#include "tile_preparer.h"
#include "timer.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace panorama {
namespace {

void validate_configuration(const RaytraceConfig &config) {
  if (config.tile_cache_size_bytes == 0U || config.tile_dir.empty()) {
    throw std::invalid_argument("Terrain trace session requires a tile directory and cache");
  }
  if (!std::isfinite(config.observer.easting) || !std::isfinite(config.observer.northing) ||
      !std::isfinite(config.observer.elevation) || !std::isfinite(config.max_distance) ||
      config.max_distance <= 0.0F) {
    throw std::invalid_argument("Raytrace configuration must be finite");
  }
}

[[nodiscard]] uint32_t validate_ray_field(const RayField &field) {
  const uint64_t ray_count = static_cast<uint64_t>(field.image.width) * field.image.height;
  if (ray_count == 0U || ray_count > std::numeric_limits<uint32_t>::max() ||
      field.rays.size() != ray_count) {
    throw std::invalid_argument("Ray field has invalid dimensions or storage");
  }
  for (const RayDirection &ray : field.rays) {
    const float horizontal_length = std::hypot(ray.x, ray.y);
    if (!std::isfinite(ray.x) || !std::isfinite(ray.y) || !std::isfinite(ray.slope) ||
        !std::isfinite(horizontal_length) || std::abs(horizontal_length - 1.0F) > 1e-4F ||
        (ray.x != 0.0F && !std::isfinite(ray.inverse_x)) ||
        (ray.y != 0.0F && !std::isfinite(ray.inverse_y))) {
      throw std::invalid_argument("Ray field contains an invalid projected direction");
    }
  }
  return static_cast<uint32_t>(ray_count);
}

[[nodiscard]] size_t checked_byte_count(size_t count, size_t size, const char *name) {
  if (count > std::numeric_limits<size_t>::max() / size) {
    throw std::overflow_error(std::string(name) + " payload is too large");
  }
  return count * size;
}

[[nodiscard]] RaytraceParameters make_parameters(
    const LoadedTile &tile,
    const RaytraceConfig &config,
    const TerrainCatalogue &catalogue,
    uint32_t ray_count
) {
  const double curvature_lift =
      kCurvatureCoefficient * static_cast<double>(config.max_distance) * config.max_distance;
  if (tile.delta > static_cast<double>(std::numeric_limits<float>::max()) ||
      config.observer.elevation < static_cast<double>(std::numeric_limits<float>::lowest()) ||
      config.observer.elevation > static_cast<double>(std::numeric_limits<float>::max()) ||
      curvature_lift > static_cast<double>(std::numeric_limits<float>::max()) ||
      tile.num_levels == 0U || tile.num_levels >= 32U ||
      tile.size != (1U << (tile.num_levels - 1U))) {
    throw std::overflow_error("Raytrace geometry has an invalid float32 mipmap layout");
  }
  return {
      static_cast<float>(tile.delta),
      static_cast<float>(config.observer.elevation),
      static_cast<float>(kCurvatureCoefficient),
      catalogue.maximum_elevation().value_or(std::numeric_limits<float>::infinity()),
      tile.num_levels,
      ray_count,
      config.max_distance,
  };
}

} // namespace

struct TerrainTraceSession::State {
  RaytraceConfig config;
  ImageSize image;
  uint32_t ray_count;
  Timer timer{"Total elapsed"};
  std::unique_ptr<TerrainCatalogue> catalogue;
  std::unique_ptr<LoadedTile> origin;
  uint32_t mipmap_value_count = 0U;
  RaytraceParameters parameters = {};
  std::unique_ptr<GpuRaytraceResources> gpu;
  std::unique_ptr<ResidentTileCache> cache;
  std::unique_ptr<AsyncTilePreparer> preparer;
  ResidentTileCacheBindings cache_bindings = {};
  uint64_t deferred_successor_work = 0U;
  uint64_t locally_skipped_tiles = 0U;
  uint64_t globally_skipped_tiles = 0U;
  uint64_t frames = 0U;

  State(
      const RaytraceConfig &config_value,
      const RayField &initial_field,
      GpuTraceOutputRequirements outputs
  )
      : config(config_value), image(initial_field.image),
        ray_count(validate_ray_field(initial_field)) {
    validate_configuration(config);
    timer.start_wall("Initial setup");

    catalogue = std::make_unique<TerrainCatalogue>(TerrainCatalogue::discover(
        config.tile_dir,
        config.observer,
        config.max_distance,
        config.max_tile_count
    ));
    const TileGrid &grid = catalogue->grid();
    const TileKey origin_key = catalogue->origin().key;

    // The observer tile establishes the dimensions and CRS shared by every
    // fixed-stride atlas slot for the lifetime of this session.
    timer.start_work("Tile load");
    origin = std::make_unique<LoadedTile>(LoadedTile::load(catalogue->origin().path));
    timer.stop("Tile load");

    const bool custom_origin = is_metal_tile_path(catalogue->origin().path);
    if (!custom_origin) {
      timer.start_work("Mipmap generation");
      origin->compute_mipmap();
      timer.stop("Mipmap generation");
    }
    validate_terrain_tile_position(*origin, origin_key, grid);

    bool trace_quantized = false;
    QuantizedMetalTileRecordLayout quantized_layout = {};
    if (config.retain_quantized) {
      if (!custom_origin) {
        throw std::invalid_argument("--retain-quantized requires uint16 custom terrain tiles");
      }
      const MetalTileHeader header = read_metal_tile_header(catalogue->origin().path);
      if (header.sample_type != MetalTileSampleType::Uint16Decimeters) {
        throw std::invalid_argument("--retain-quantized requires uint16 custom terrain tiles");
      }
      quantized_layout = quantized_metal_tile_record_layout(header);
      trace_quantized = true;
    }

    const std::vector<TerrainSource> &sources = catalogue->sources();
    if (trace_quantized &&
        std::any_of(sources.begin(), sources.end(), [](const TerrainSource &source) {
          return !is_metal_tile_path(source.path);
        })) {
      throw std::invalid_argument("--retain-quantized requires a custom-only terrain directory");
    }

    const size_t mip_count = static_cast<size_t>(metal_tile_mipmap_value_count(origin->size));
    const size_t vertex_side = static_cast<size_t>(origin->size) + 1U;
    const size_t vertex_count = vertex_side * vertex_side;
    if (mip_count > std::numeric_limits<uint32_t>::max() ||
        vertex_count > std::numeric_limits<uint32_t>::max()) {
      throw std::overflow_error("Terrain tile arrays exceed Metal uint indexing");
    }
    const size_t tile_bytes =
        trace_quantized
            ? checked_byte_count(mip_count, sizeof(uint16_t), "terrain mipmap") +
                  quantized_layout.stride
            : checked_byte_count(mip_count + vertex_count, sizeof(float), "terrain tile");
    const uint64_t slot_capacity = config.tile_cache_size_bytes / tile_bytes;
    if (slot_capacity == 0U) {
      throw std::runtime_error("Tile-cache byte budget cannot hold one terrain tile");
    }
    const uint64_t bounded_slots = std::min(slot_capacity, static_cast<uint64_t>(sources.size()));
    if (bounded_slots > std::numeric_limits<uint32_t>::max()) {
      throw std::overflow_error("Tile-cache slot count exceeds Metal uint range");
    }
    const uint32_t atlas_slots = static_cast<uint32_t>(bounded_slots);
    mipmap_value_count = static_cast<uint32_t>(mip_count);
    parameters = make_parameters(*origin, config, *catalogue, ray_count);

    gpu = std::make_unique<GpuRaytraceResources>(
        initial_field.rays,
        sources,
        trace_quantized,
        outputs
    );
    cache = std::make_unique<
        ResidentTileCache>(gpu->device(), sources, *origin, origin_key, config, atlas_slots, timer);
    preparer = std::make_unique<AsyncTilePreparer>(
        gpu->device(),
        sources,
        *origin,
        grid,
        atlas_slots,
        config.max_tile_preparation_workers,
        timer
    );
    cache_bindings = cache->bindings();
    preparer->start();
    timer.stop("Initial setup");
  }

  [[nodiscard]] uint32_t ensure_observer_resident() {
    uint32_t slot = cache->slot_for_source(0U);
    const std::vector<uint8_t> no_pinned_slots(cache->slot_capacity(), 0U);
    // The origin can be evicted after an earlier view. No GPU pass is active
    // here, so publishing it again may use any cache slot.
    while (slot == cache->slot_capacity()) {
      preparer->request(0U, 0.0F);
      (void)cache->install_prepared(*preparer, no_pinned_slots, timer);
      slot = cache->slot_for_source(0U);
      if (slot == cache->slot_capacity()) {
        timer.start_wall("Tile availability wait");
        preparer->wait_for_prepared();
        timer.stop("Tile availability wait");
      }
    }
    return slot;
  }
};

TerrainTraceSession::TerrainTraceSession(
    const RaytraceConfig &config,
    const RayField &initial_field,
    GpuTraceOutputRequirements outputs
)
    : state_(std::make_unique<State>(config, initial_field, outputs)) {}

TerrainTraceSession::~TerrainTraceSession() = default;

void TerrainTraceSession::trace(const RayField &field) {
  State &state = *state_;
  const uint32_t ray_count = validate_ray_field(field);
  const bool dimensions_changed = field.image.width != state.image.width ||
                                  field.image.height != state.image.height ||
                                  ray_count != state.ray_count;

  // Ray-dependent storage changes with the viewport; catalogue discovery,
  // terrain preparation, pipelines, and the resident atlas remain intact.
  const uint32_t observer_slot = state.ensure_observer_resident();
  if (dimensions_changed) {
    state.gpu->resize_rays(field.rays);
    state.image = field.image;
    state.ray_count = ray_count;
    state.parameters.ray_count = ray_count;
  } else {
    state.gpu->update_rays(field.rays);
  }
  state.gpu->initialise_frontier(observer_slot);

  HostFrontier frontier(
      *state.catalogue,
      field.rays,
      state.parameters,
      state.cache->slot_capacity(),
      observer_slot
  );
  const std::vector<uint8_t> no_pinned_slots(state.cache->slot_capacity(), 0U);
  uint32_t active_count = state.ray_count;

  state.gpu->start_capture_if_requested();
  state.timer.start_wall("GPU raytrace");
  try {
    while (active_count != 0U) {
      state.timer.start_wall("Frontier bookkeeping");
#if defined(PANORAMA_DEBUG_VALIDATION)
      frontier.validate_frontier(state.gpu->active_frontier(), active_count, "active frontier");
#endif
      frontier.record_active_slot_use(*state.cache);
      state.timer.stop("Frontier bookkeeping");

      const GpuFrontierPassResult pass = state.gpu->trace_frontier(
          state.cache_bindings,
          state.parameters,
          state.mipmap_value_count,
          active_count,
          state.timer
      );
      state.timer.add_work("GPU raytrace", pass.device_milliseconds);
      state.locally_skipped_tiles += pass.locally_skipped_tiles;
      state.globally_skipped_tiles += pass.globally_skipped_tiles;
      if (pass.deferred_count > state.ray_count) {
        throw std::runtime_error("GPU frontier exceeds the ray frontier capacity");
      }
      state.deferred_successor_work += pass.deferred_count;

      const std::span<const DeferredRayWork> deferred =
          state.gpu->deferred_work(pass.deferred_count);
#if defined(PANORAMA_DEBUG_VALIDATION)
      frontier.validate_deferred_work(deferred);
#endif
      state.timer.start_wall("Frontier bookkeeping");
      // The completed pass no longer reads the atlas. Publish available tiles,
      // then reactivate every continuation whose source is now resident.
      frontier.mark_installed(
          state.cache->install_prepared(*state.preparer, no_pinned_slots, state.timer)
      );
      active_count = frontier.activate_resident(
          state.gpu->active_frontier(),
          0U,
          *state.cache,
          *state.preparer,
          deferred
      );
#if defined(PANORAMA_DEBUG_VALIDATION)
      frontier.validate_deferred_work();
      frontier.validate_frontier(state.gpu->active_frontier(), active_count, "activated frontier");
#endif
      state.timer.stop("Frontier bookkeeping");

      while (active_count == 0U && frontier.has_deferred_work()) {
        // Deferred work remains but none of it is resident. Wait for one of
        // the already-requested sources instead of submitting an empty pass.
        state.timer.start_wall("Tile availability wait");
        state.preparer->wait_for_prepared();
        state.timer.stop("Tile availability wait");

        state.timer.start_wall("Frontier bookkeeping");
        frontier.mark_installed(
            state.cache->install_prepared(*state.preparer, no_pinned_slots, state.timer)
        );
        active_count = frontier.activate_resident(
            state.gpu->active_frontier(),
            active_count,
            *state.cache,
            *state.preparer
        );
#if defined(PANORAMA_DEBUG_VALIDATION)
        frontier.validate_deferred_work();
        frontier
            .validate_frontier(state.gpu->active_frontier(), active_count, "activated frontier");
#endif
        state.timer.stop("Frontier bookkeeping");
      }
    }
  } catch (...) {
    state.preparer->stop_and_join();
    state.gpu->stop_capture();
    throw;
  }
  state.gpu->stop_capture();
  state.timer.stop("GPU raytrace");
  state.frames++;
}

ImageSize TerrainTraceSession::image() const { return state_->image; }

Crs TerrainTraceSession::crs() const { return state_->origin->crs; }

id<MTLDevice> TerrainTraceSession::device() const { return state_->gpu->device(); }

id<MTLCommandQueue> TerrainTraceSession::command_queue() const {
  return state_->gpu->command_queue();
}

id<MTLLibrary> TerrainTraceSession::library() const { return state_->gpu->library(); }

id<MTLBuffer> TerrainTraceSession::ray_directions() const { return state_->gpu->ray_directions(); }

id<MTLBuffer> TerrainTraceSession::distances() const { return state_->gpu->distances(); }

id<MTLBuffer> TerrainTraceSession::elevations() const { return state_->gpu->elevations(); }

id<MTLBuffer> TerrainTraceSession::surface_gradients() const {
  return state_->gpu->surface_gradients();
}

void TerrainTraceSession::print_statistics() const {
  const State &state = *state_;
  const TilePreparationStatistics preparation = state.preparer->statistics();
  const ResidentTileCacheStatistics cache = state.cache->statistics();
  std::printf(
      "Terrain sources: %zu (resident slots %u / cache capacity %u, preparation workers %u).\n",
      state.catalogue->sources().size(),
      cache.resident_tiles,
      cache.slot_capacity,
      preparation.worker_count
  );
  if (state.frames == 1U) {
    std::printf(
        "  GPU frontier continuations: %llu rays deferred to source buckets.\n",
        static_cast<unsigned long long>(state.deferred_successor_work)
    );
  } else {
    std::printf(
        "  GPU frontier continuations: %llu rays deferred across %llu frames.\n",
        static_cast<unsigned long long>(state.deferred_successor_work),
        static_cast<unsigned long long>(state.frames)
    );
  }
  const uint64_t skipped = state.locally_skipped_tiles + state.globally_skipped_tiles;
  std::printf(
      "  Tile I/O: %llu requests (%llu unique, %llu duplicate); %llu skips "
      "(%llu local, %llu global; %s).\n",
      static_cast<unsigned long long>(preparation.requests),
      static_cast<unsigned long long>(preparation.unique_requests),
      static_cast<unsigned long long>(preparation.duplicate_requests),
      static_cast<unsigned long long>(skipped),
      static_cast<unsigned long long>(state.locally_skipped_tiles),
      static_cast<unsigned long long>(state.globally_skipped_tiles),
      state.catalogue->maximum_elevation().has_value() ? "GPU cutoff enabled" : "no complete maxima"
  );
  std::printf(
      "  Atlas installations: %llu, copied: %.3f GiB, Metal I/O: %.3f GiB, "
      "evictions: %llu.\n",
      static_cast<unsigned long long>(cache.installations),
      static_cast<double>(cache.bytes_copied) / (1024.0 * 1024.0 * 1024.0),
      static_cast<double>(cache.bytes_loaded_with_metal_io) / (1024.0 * 1024.0 * 1024.0),
      static_cast<unsigned long long>(cache.evictions)
  );
  state.timer.print();
}

} // namespace panorama
