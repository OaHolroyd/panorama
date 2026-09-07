#include "terrain_trace_session.h"

#include "host_frontier.h"
#include "metal_bvh_trace.h"
#include "terrain_shadow_gpu.h"
#include "tile_manager.h"
#include "timer.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace panorama {
namespace {

void validate_configuration(const RaytraceConfig &config) {
  if (config.bvh_block_cells == 0U) {
    throw std::invalid_argument("BVH block cell count must be positive");
  }
  if (config.bvh_cache_size_bytes == 0U)
    throw std::invalid_argument("BVH cache size must be positive");
  if (config.raytracer != Raytracer::Software && config.raytracer != Raytracer::MetalBvh) {
    throw std::invalid_argument("Unknown terrain raytracer");
  }
  if (config.tile_cache_size_bytes == 0U || config.tile_dir.empty()) {
    throw std::invalid_argument("Terrain trace session requires a tile directory and cache");
  }
  if (!std::isfinite(config.observer.easting) || !std::isfinite(config.observer.northing) ||
      !std::isfinite(config.observer.elevation) || !std::isfinite(config.max_distance) ||
      config.max_distance <= 0.0F || !std::isfinite(config.lod_scale) || config.lod_scale < 0.0F) {
    throw std::invalid_argument("Raytrace configuration must be finite");
  }
}

[[nodiscard]] uint32_t validate_ray_field(const RayField &field) {
  const uint64_t ray_count = static_cast<uint64_t>(field.image.width) * field.image.height;
  if (ray_count == 0U || ray_count > std::numeric_limits<uint32_t>::max() ||
      field.rays.size() != ray_count) {
    throw std::invalid_argument("Ray field has invalid dimensions or storage");
  }
  if (!std::isfinite(field.minimum_pixel_angle) || field.minimum_pixel_angle <= 0.0F) {
    throw std::invalid_argument("Ray field has no finite positive pixel angle");
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

[[nodiscard]] RaytraceParameters make_parameters(
    const TileGeometry &tile,
    const RaytraceConfig &config,
    const TerrainCatalogue &catalogue,
    uint32_t ray_count
) {
  const double curvature_lift =
      kCurvatureCoefficient * static_cast<double>(config.max_distance) * config.max_distance;
  if (tile.cell_size > static_cast<double>(std::numeric_limits<float>::max()) ||
      config.observer.elevation < static_cast<double>(std::numeric_limits<float>::lowest()) ||
      config.observer.elevation > static_cast<double>(std::numeric_limits<float>::max()) ||
      curvature_lift > static_cast<double>(std::numeric_limits<float>::max()) ||
      tile.mipmap_level_count == 0U || tile.mipmap_level_count >= 32U ||
      tile.cell_count != (1U << (tile.mipmap_level_count - 1U))) {
    throw std::overflow_error("Raytrace geometry has an invalid float32 mipmap layout");
  }
  return {
      static_cast<float>(tile.cell_size),
      static_cast<float>(config.observer.elevation),
      static_cast<float>(kCurvatureCoefficient),
      catalogue.maximum_elevation().value_or(std::numeric_limits<float>::infinity()),
      tile.mipmap_level_count,
      ray_count,
      config.max_distance,
  };
}

} // namespace

struct TerrainTraceSession::State {
  // Observer, image, and scalar kernel parameters updated across frames.
  RaytraceConfig config;
  ImageSize image;
  uint32_t ray_count;
  Timer timer{"Total elapsed"};
  MetalBvhStatistics frame_bvh_before;
  double frame_wall_ms = 0.0;
  double frame_gpu_ms = 0.0;
  uint64_t frame_passes = 0U;

  // Long-lived owners. TileManager serves both primary and shadow frontiers;
  // ray-sized GPU buffers remain valid for presentation after tracing.
  std::unique_ptr<TileManager> tiles;
  RaytraceParameters parameters = {};
  std::unique_ptr<GpuRaytraceResources> gpu;
  std::unique_ptr<MetalBvhTrace> bvh;
  GpuTraceOutputRequirements outputs;
  std::unique_ptr<GpuTerrainShadowResources> shadows;

  // Cumulative scheduling diagnostics printed by the command-line frontend.
  uint64_t deferred_successor_work = 0U;
  uint64_t locally_skipped_tiles = 0U;
  uint64_t globally_skipped_tiles = 0U;
  uint64_t frames = 0U;

  // Shadow results are reusable only for the same primary trace and sun.
  uint64_t trace_revision = 0U;
  uint64_t shadow_revision = std::numeric_limits<uint64_t>::max();
  double shadow_azimuth = 0.0;
  double shadow_elevation = 0.0;

  State(
      const RaytraceConfig &config_value,
      const RayField &initial_field,
      GpuTraceOutputRequirements outputs
  )
      : config(config_value), image(initial_field.image),
        ray_count(validate_ray_field(initial_field)), outputs(outputs) {
    validate_configuration(config);
    timer.start_wall("Initial setup");

    // Tile discovery must precede ray-resource construction because the GPU
    // catalogue hash uses stable source indices. Atlas attachment follows so
    // it can reuse the device selected by the primary tracing resources.
    tiles = std::make_unique<TileManager>(config, initial_field.minimum_pixel_angle);
    config.observer = tiles->catalogue().observer();
    parameters = make_parameters(tiles->origin_geometry(), config, tiles->catalogue(), ray_count);

    gpu = std::make_unique<GpuRaytraceResources>(
        initial_field.rays,
        tiles->sources(),
        tiles->traces_quantized(),
        config.bilinear_collisions,
        config.c1_normals,
        outputs
    );
    tiles->attach_gpu(gpu->device(), timer);
    if (config.raytracer == Raytracer::MetalBvh) {
      bvh = std::make_unique<MetalBvhTrace>(
          *gpu,
          config.bvh_block_cells,
          outputs,
          config.bvh_cache_size_bytes
      );
    }
    timer.stop("Initial setup");
  }
};

TerrainTraceSession::TerrainTraceSession(
    const RaytraceConfig &config,
    const RayField &initial_field,
    GpuTraceOutputRequirements outputs
)
    : state_(std::make_unique<State>(config, initial_field, outputs)) {}

TerrainTraceSession::~TerrainTraceSession() = default;

MetalBvhStatistics TerrainTraceSession::bvh_statistics() const {
  return state_->bvh ? state_->bvh->statistics() : MetalBvhStatistics{};
}

void TerrainTraceSession::set_raytracer(Raytracer raytracer) {
  State &state = *state_;
  if (raytracer != Raytracer::Software && raytracer != Raytracer::MetalBvh) {
    throw std::invalid_argument("Unknown terrain raytracer");
  }
  if (state.config.raytracer == raytracer)
    return;
  if (raytracer == Raytracer::MetalBvh && !state.bvh) {
    state.bvh = std::make_unique<MetalBvhTrace>(
        *state.gpu,
        state.config.bvh_block_cells,
        state.outputs,
        state.config.bvh_cache_size_bytes
    );
  }
  state.config.raytracer = raytracer;
  state.shadow_revision = std::numeric_limits<uint64_t>::max();
}

bool TerrainTraceSession::relocate_observer(ObserverLocation observer) {
  State &state = *state_;
  if (!std::isfinite(observer.easting) || !std::isfinite(observer.northing) ||
      !std::isfinite(observer.elevation)) {
    throw std::invalid_argument("Terrain relocation requires a finite observer");
  }
  if (!state.tiles->relocate_observer(observer)) {
    return false;
  }
  state.config.observer = observer;
  state.parameters.observer_elevation = static_cast<float>(observer.elevation);
  state.shadow_revision = std::numeric_limits<uint64_t>::max();
  return true;
}

void TerrainTraceSession::set_lod_scale(float lod_scale) {
  if (!std::isfinite(lod_scale) || lod_scale < 0.0F) {
    throw std::invalid_argument("Terrain LOD scale must be finite and nonnegative");
  }
  State &state = *state_;
  if (state.config.lod_scale == lod_scale) {
    return;
  }
  state.config.lod_scale = lod_scale;
  state.tiles->set_lod_scale(lod_scale);
  state.shadow_revision = std::numeric_limits<uint64_t>::max();
}

void TerrainTraceSession::set_collision_options(bool bilinear_collisions, bool c1_normals) {
  State &state = *state_;
  if (state.config.bilinear_collisions == bilinear_collisions &&
      state.config.c1_normals == c1_normals) {
    return;
  }

  // Collision and normal choices are represented by already-specialized
  // pipeline objects. TileManager residency and every ray-sized buffer remain
  // valid, so changing an inspector switch must not reconstruct the session.
  state.gpu->set_collision_options(bilinear_collisions, c1_normals);
  if (state.shadows != nullptr) {
    state.shadows->set_collision_options(bilinear_collisions, c1_normals);
  }
  state.config.bilinear_collisions = bilinear_collisions;
  state.config.c1_normals = c1_normals;
  state.shadow_revision = std::numeric_limits<uint64_t>::max();
}

void TerrainTraceSession::trace(const RayField &field) {
  State &state = *state_;
  const auto started = std::chrono::steady_clock::now();
  state.frame_bvh_before = bvh_statistics();
  state.frame_gpu_ms = 0.0;
  state.frame_passes = 0U;
  const uint32_t ray_count = validate_ray_field(field);
  state.tiles->set_pixel_angle(field.minimum_pixel_angle);
  const bool dimensions_changed = field.image.width != state.image.width ||
                                  field.image.height != state.image.height ||
                                  ray_count != state.ray_count;

  // Ray-dependent storage changes with the viewport; catalogue discovery,
  // terrain preparation, pipelines, and the resident atlas remain intact.
  if (dimensions_changed) {
    state.gpu->resize_rays(field.rays);
    state.image = field.image;
    state.ray_count = ray_count;
    state.parameters.ray_count = ray_count;
  } else {
    state.gpu->update_rays(field.rays);
  }
  if (state.config.raytracer == Raytracer::MetalBvh) {
    state.bvh->prepare(*state.tiles, state.config.observer, state.parameters, state.timer);
    state.gpu->start_capture_if_requested();
    state.timer.start_wall("BVH streaming trace");
    try {
      state.bvh->trace(state.parameters, state.timer);
    } catch (...) {
      state.timer.stop("BVH streaming trace");
      state.gpu->stop_capture();
      throw;
    }
    state.timer.stop("BVH streaming trace");
    state.gpu->stop_capture();
    const auto after = bvh_statistics();
    state.frame_gpu_ms = after.selection_gpu_ms - state.frame_bvh_before.selection_gpu_ms +
                         after.trace_gpu_ms - state.frame_bvh_before.trace_gpu_ms;
    state.frame_passes = after.selection_passes - state.frame_bvh_before.selection_passes +
                         after.trace_passes - state.frame_bvh_before.trace_passes;
    state.frame_wall_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started)
            .count();
    state.frames++;
    state.trace_revision++;
    return;
  }
  const uint32_t observer_slot = state.tiles->ensure_observer_resident(state.timer);
  state.gpu->initialise_frontier(observer_slot);

  HostFrontier frontier(
      *state.tiles,
      field.rays,
      state.parameters,
      state.tiles->slot_capacity(),
      observer_slot
  );
  const std::vector<uint8_t> no_pinned_slots(state.tiles->slot_capacity(), 0U);
  uint32_t active_count = state.ray_count;

  state.gpu->start_capture_if_requested();
  state.timer.start_wall("GPU raytrace");
  try {
    // Each iteration traces only resident segments. GPU-emitted successors
    // return to HostFrontier, while TileManager progresses missing sources in
    // parallel and installs them only after the command has completed.
    while (active_count != 0U) {
      state.timer.start_wall("Frontier bookkeeping");
#if defined(PANORAMA_DEBUG_VALIDATION)
      frontier.validate_frontier(state.gpu->active_frontier(), active_count, "active frontier");
#endif
      frontier.record_active_slot_use();
      state.timer.stop("Frontier bookkeeping");

      const GpuFrontierPassResult pass = state.gpu->trace_frontier(
          state.tiles->bindings(),
          state.parameters,
          state.tiles->mipmap_value_count(),
          active_count,
          state.timer
      );
      state.timer.add_work("GPU raytrace", pass.device_milliseconds);
      state.frame_gpu_ms += pass.device_milliseconds;
      ++state.frame_passes;
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
      frontier.mark_installed(state.tiles->install_available(no_pinned_slots, state.timer));
      active_count = frontier.activate_resident(state.gpu->active_frontier(), 0U, deferred);
#if defined(PANORAMA_DEBUG_VALIDATION)
      frontier.validate_deferred_work();
      frontier.validate_frontier(state.gpu->active_frontier(), active_count, "activated frontier");
#endif
      state.timer.stop("Frontier bookkeeping");

      while (active_count == 0U && frontier.has_deferred_work()) {
        // Deferred work remains but none of it is resident. Wait for one of
        // the already-requested sources instead of submitting an empty pass.
        state.timer.start_wall("Tile availability wait");
        state.tiles->wait_for_available();
        state.timer.stop("Tile availability wait");

        state.timer.start_wall("Frontier bookkeeping");
        frontier.mark_installed(state.tiles->install_available(no_pinned_slots, state.timer));
        active_count = frontier.activate_resident(state.gpu->active_frontier(), active_count);
#if defined(PANORAMA_DEBUG_VALIDATION)
        frontier.validate_deferred_work();
        frontier
            .validate_frontier(state.gpu->active_frontier(), active_count, "activated frontier");
#endif
        state.timer.stop("Frontier bookkeeping");
      }
    }
  } catch (...) {
    state.tiles->stop();
    state.gpu->stop_capture();
    throw;
  }
  state.gpu->stop_capture();
  state.timer.stop("GPU raytrace");
  state.frame_wall_ms =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
  state.frames++;
  state.trace_revision++;
}

void TerrainTraceSession::trace_shadows(double sun_azimuth, double sun_elevation) {
  State &state = *state_;
  if (!std::isfinite(sun_azimuth) || !std::isfinite(sun_elevation)) {
    throw std::invalid_argument("Sun direction must be finite");
  }
  if (state.shadows == nullptr) {
    // Construction is deliberately lazy: disabled shadows allocate no
    // per-pixel storage and compile no secondary pipelines.
    state.shadows = std::make_unique<GpuTerrainShadowResources>(
        state.gpu->device(),
        state.gpu->command_queue(),
        state.gpu->library(),
        state.gpu->traces_quantized(),
        state.gpu->bilinear_collisions(),
        state.gpu->c1_normals()
    );
  }
  state.shadows->resize(state.ray_count);
  if (state.shadow_revision == state.trace_revision && state.shadow_azimuth == sun_azimuth &&
      state.shadow_elevation == sun_elevation) {
    return;
  }
  // No direct sunlight reaches terrain when the sun is below the horizon.
  // A vertical ray cannot cross another heightfield location, so it is clear.
  if (sun_elevation <= 0.0 || std::cos(sun_elevation) < 1e-6) {
    state.shadows->fill_visibility(sun_elevation > 0.0 ? 1U : 0U);
    state.shadow_revision = state.trace_revision;
    state.shadow_azimuth = sun_azimuth;
    state.shadow_elevation = sun_elevation;
    return;
  }

  const float direction_x = static_cast<float>(std::sin(sun_azimuth));
  const float direction_y = static_cast<float>(std::cos(sun_azimuth));
  const float slope = static_cast<float>(std::tan(sun_elevation));
  const TileGrid &grid = state.tiles->catalogue().grid();
  ShadowTraceParameters parameters = {
      state.parameters,
      {
          direction_x,
          direction_y,
          direction_x == 0.0F ? std::numeric_limits<float>::infinity() : 1.0F / direction_x,
          direction_y == 0.0F ? std::numeric_limits<float>::infinity() : 1.0F / direction_y,
          slope,
      },
      static_cast<float>(grid.origin_x - state.config.observer.easting),
      static_cast<float>(grid.origin_y - state.config.observer.northing),
      static_cast<float>(grid.width),
      state.gpu->catalogue_hash_capacity(),
  };

  state.timer.start_wall("GPU shadow trace");
  const std::span<const DeferredRayWork> initial = state.shadows->initialise(
      state.gpu->ray_directions(),
      state.gpu->distances(),
      state.gpu->elevations(),
      state.gpu->surface_gradients(),
      state.gpu->catalogue_hash(),
      parameters,
      state.timer
  );
  const auto *primary_distances = static_cast<const float *>(state.gpu->distances().contents);
  if (primary_distances == nullptr) {
    throw std::runtime_error("Could not map shadow scheduling distances");
  }
  HostFrontier frontier(
      *state.tiles,
      state.ray_count,
      state.parameters.num_levels,
      state.tiles->slot_capacity(),
      std::span<const float>(primary_distances, state.ray_count)
  );
  const std::vector<uint8_t> no_pinned_slots(state.tiles->slot_capacity(), 0U);
  uint32_t active_count = frontier.activate_resident(state.shadows->active_frontier(), 0U, initial);
#if defined(PANORAMA_DEBUG_VALIDATION)
  frontier.validate_deferred_work();
  frontier.validate_frontier(state.shadows->active_frontier(), active_count, "shadow frontier");
#endif
  while (active_count != 0U || frontier.has_deferred_work()) {
    if (active_count == 0U) {
      state.timer.start_wall("Tile availability wait");
      state.tiles->wait_for_available();
      state.timer.stop("Tile availability wait");
      frontier.mark_installed(state.tiles->install_available(no_pinned_slots, state.timer));
      active_count = frontier.activate_resident(state.shadows->active_frontier(), 0U);
#if defined(PANORAMA_DEBUG_VALIDATION)
      frontier.validate_deferred_work();
      frontier.validate_frontier(state.shadows->active_frontier(), active_count, "shadow frontier");
#endif
      continue;
    }
#if defined(PANORAMA_DEBUG_VALIDATION)
    frontier.validate_frontier(state.shadows->active_frontier(), active_count, "shadow frontier");
#endif
    frontier.record_active_slot_use();
    const GpuFrontierPassResult pass = state.shadows->trace_frontier(
        state.tiles->bindings(),
        state.gpu->catalogue_hash(),
        parameters,
        state.tiles->mipmap_value_count(),
        active_count,
        state.timer
    );
    state.timer.add_work("GPU shadow trace", pass.device_milliseconds);
    const std::span<const DeferredRayWork> deferred =
        state.shadows->deferred_work(pass.deferred_count);
#if defined(PANORAMA_DEBUG_VALIDATION)
    frontier.validate_deferred_work(deferred);
#endif
    frontier.mark_installed(state.tiles->install_available(no_pinned_slots, state.timer));
    active_count = frontier.activate_resident(state.shadows->active_frontier(), 0U, deferred);
#if defined(PANORAMA_DEBUG_VALIDATION)
    frontier.validate_deferred_work();
    frontier.validate_frontier(state.shadows->active_frontier(), active_count, "shadow frontier");
#endif
  }
  state.timer.stop("GPU shadow trace");
  state.shadow_revision = state.trace_revision;
  state.shadow_azimuth = sun_azimuth;
  state.shadow_elevation = sun_elevation;
}

ImageSize TerrainTraceSession::image() const { return state_->image; }

Crs TerrainTraceSession::crs() const { return state_->tiles->origin_geometry().crs; }

ObserverLocation TerrainTraceSession::observer() const { return state_->config.observer; }

const TerrainCoverage &TerrainTraceSession::terrain_coverage() const {
  return state_->tiles->catalogue().coverage();
}

std::optional<float> TerrainTraceSession::sample_terrain(double easting, double northing) {
  return state_->tiles->sample_terrain(easting, northing);
}

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

id<MTLBuffer> TerrainTraceSession::num_steps() const { return state_->gpu->num_steps(); }

id<MTLBuffer> TerrainTraceSession::num_evaluations() const {
  return state_->gpu->num_evaluations();
}

id<MTLBuffer> TerrainTraceSession::shadow_visibility() const {
  if (state_->shadows == nullptr || state_->shadow_revision != state_->trace_revision) {
    throw std::logic_error("Shadows have not been traced for the current terrain view");
  }
  return state_->shadows->visibility();
}

void TerrainTraceSession::print_trace_statistics() const {
  const State &state = *state_;
  std::printf(
      "Trace %llu %s %ux%u: wall %.3f ms, GPU traversal sum %.3f ms, %llu passes",
      static_cast<unsigned long long>(state.frames),
      state.config.raytracer == Raytracer::MetalBvh ? "BVH" : "Mipmap",
      state.image.width,
      state.image.height,
      state.frame_wall_ms,
      state.frame_gpu_ms,
      static_cast<unsigned long long>(state.frame_passes)
  );
  if (state.config.raytracer == Raytracer::MetalBvh) {
    const auto b = bvh_statistics();
    const auto &a = state.frame_bvh_before;
    std::printf(
        "; tiles built/hit/evicted %llu/%llu/%llu, catalogue/instance builds %llu/%llu, "
        "BVH submissions %llu, GPU selection/detail/build %.3f/%.3f/%.3f ms, "
        "CPU grouping %.3f ms, resident %.1f/%.1f MiB",
        static_cast<unsigned long long>(b.builds - a.builds),
        static_cast<unsigned long long>(b.cache_hits - a.cache_hits),
        static_cast<unsigned long long>(b.evictions - a.evictions),
        static_cast<unsigned long long>(b.catalogue_builds - a.catalogue_builds),
        static_cast<unsigned long long>(b.instance_builds - a.instance_builds),
        static_cast<unsigned long long>(b.submissions - a.submissions),
        b.selection_gpu_ms - a.selection_gpu_ms,
        b.trace_gpu_ms - a.trace_gpu_ms,
        b.build_gpu_ms - a.build_gpu_ms,
        b.grouping_cpu_ms - a.grouping_cpu_ms,
        double(b.resident_bytes) / 1048576.0,
        double(b.budget_bytes) / 1048576.0
    );
  }
  std::putchar('\n');
  std::fflush(stdout);
}

void TerrainTraceSession::print_statistics() const {
  const State &state = *state_;
  const TileManagerStatistics tiles = state.tiles->statistics();
  if (state.bvh) {
    const auto bvh = state.bvh->statistics();
    std::printf(
        "Metal BVH cache: %.3f MiB resident, %.3f MiB peak including build workspace / %.3f MiB "
        "budget; "
        "%llu builds, %llu cache hits, %llu evictions; catalogue %.3f MiB.\n",
        double(bvh.resident_bytes) / 1048576.0,
        double(bvh.peak_bytes) / 1048576.0,
        double(bvh.budget_bytes) / 1048576.0,
        static_cast<unsigned long long>(bvh.builds),
        static_cast<unsigned long long>(bvh.cache_hits),
        static_cast<unsigned long long>(bvh.evictions),
        double(bvh.catalogue_bytes) / 1048576.0
    );
  }
  std::printf(
      "Terrain sources: %zu (resident slots %u / cache capacity %u, preparation workers %u).\n",
      state.tiles->sources().size(),
      tiles.resident_tiles,
      tiles.slot_capacity,
      tiles.worker_count
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
      static_cast<unsigned long long>(tiles.requests),
      static_cast<unsigned long long>(tiles.unique_requests),
      static_cast<unsigned long long>(tiles.duplicate_requests),
      static_cast<unsigned long long>(skipped),
      static_cast<unsigned long long>(state.locally_skipped_tiles),
      static_cast<unsigned long long>(state.globally_skipped_tiles),
      state.tiles->catalogue().maximum_elevation().has_value() ? "GPU cutoff enabled"
                                                               : "no complete maxima"
  );
  std::printf(
      "  Atlas installations: %llu, Metal I/O: %.3f GiB, "
      "evictions: %llu.\n",
      static_cast<unsigned long long>(tiles.installations),
      static_cast<double>(tiles.bytes_loaded_with_metal_io) / (1024.0 * 1024.0 * 1024.0),
      static_cast<unsigned long long>(tiles.evictions)
  );
  state.timer.print();
}

} // namespace panorama
