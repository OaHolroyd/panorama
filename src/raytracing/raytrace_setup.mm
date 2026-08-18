#include "raytrace_setup.h"

#include "host_frontier.h"
#include "loaded_tile.h"
#include "png_writer.h"
#include "raytrace_gpu.h"
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

/// Reject a count that cannot be represented by the Metal `uint` interface.
void validate_configuration(const RaytraceConfig &config) {
  if (config.num_azimuth == 0U || config.num_polar == 0U) {
    throw std::invalid_argument("Ray counts must both be positive");
  }
  if (config.tile_cache_size_bytes == 0U) {
    throw std::invalid_argument("Tile-cache byte budget must be positive");
  }
  if (config.tile_dir.empty()) {
    throw std::invalid_argument("Terrain tile directory must not be empty");
  }
  if (!std::isfinite(config.observer.easting) || !std::isfinite(config.observer.northing) ||
      !std::isfinite(config.observer.elevation) || !std::isfinite(config.azimuth_start) ||
      !std::isfinite(config.azimuth_end) || !std::isfinite(config.polar_start) ||
      !std::isfinite(config.polar_end) || !std::isfinite(config.max_distance) ||
      config.max_distance <= 0.0F) {
    throw std::invalid_argument("Raytrace configuration must be finite");
  }
}

/// Return a checked byte count for one host-side terrain payload allocation.
[[nodiscard]] size_t checked_byte_count(size_t count, size_t size, const char *name) {
  if (count > std::numeric_limits<size_t>::max() / size) {
    throw std::overflow_error(std::string(name) + " payload is too large");
  }
  return count * size;
}

/// Build the scalar tracing ABI shared by compatible resident terrain tiles.
[[nodiscard]] RaytraceParameters
make_raytrace_parameters(const LoadedTile &tile, const RaytraceConfig &config) {
  if (tile.delta > static_cast<double>(std::numeric_limits<float>::max()) ||
      config.observer.elevation < static_cast<double>(std::numeric_limits<float>::lowest()) ||
      config.observer.elevation > static_cast<double>(std::numeric_limits<float>::max()) ||
      tile.num_levels == 0U || tile.num_levels >= 32U ||
      tile.size != (1U << (tile.num_levels - 1U))) {
    throw std::overflow_error("Raytrace geometry has an invalid float32 mipmap layout");
  }
  return {
      static_cast<float>(tile.delta),
      static_cast<float>(config.observer.elevation),
      tile.num_levels,
      config.num_azimuth,
      config.num_polar,
      config.max_distance,
  };
}

/// Construct float32 compass directions from evenly spaced azimuth centres.
[[nodiscard]] std::vector<HorizontalDirection>
make_azimuth_directions(const RaytraceConfig &config) {
  std::vector<HorizontalDirection> directions(config.num_azimuth);
  const double step = (config.azimuth_end - config.azimuth_start) / config.num_azimuth;
  for (uint32_t index = 0U; index < config.num_azimuth; index++) {
    const double azimuth = config.azimuth_start + (static_cast<double>(index) + 0.5) * step;
    directions[index] = {static_cast<float>(std::sin(azimuth)),
                         static_cast<float>(std::cos(azimuth))};
  }
  return directions;
}

/// Construct float32 vertical slopes from evenly spaced polar-angle centres.
[[nodiscard]] std::vector<float> make_polar_slopes(const RaytraceConfig &config) {
  std::vector<float> slopes(config.num_polar);
  const double step = (config.polar_end - config.polar_start) / config.num_polar;
  for (uint32_t index = 0U; index < config.num_polar; index++) {
    const double slope = std::tan(config.polar_start + (static_cast<double>(index) + 0.5) * step);
    if (!std::isfinite(slope) || slope < std::numeric_limits<float>::lowest() ||
        slope > std::numeric_limits<float>::max()) {
      throw std::invalid_argument("Polar range produces an invalid float32 slope");
    }
    slopes[index] = static_cast<float>(slope);
  }
  return slopes;
}

/// Return a non-owning float32 view of a completed shared Metal buffer.
[[nodiscard]] std::span<const float>
view_float_buffer(id<MTLBuffer> buffer, size_t count, const char *name) {
  const auto *contents = static_cast<const float *>(buffer.contents);
  if (contents == nullptr) {
    throw std::runtime_error(std::string("Could not map ") + name + " Metal buffer");
  }
  return {contents, count};
}

} // namespace

/// Trace a fixed observer's angular ray field through a set of terrain tiles
/// using a GPU-owned frontier and a CPU-owned resident-tile cache.
///
/// Initially, one work item represents the unresolved polar-ray column for
/// each azimuth entering the observer tile. Each frontier iteration traces
/// every resident work item through its tile, writes terrain hits, and emits
/// at most one unresolved suffix per azimuth. A suffix whose successor tile
/// is resident immediately joins the next frontier; otherwise the host keeps
/// its exact entry distance, requests that tile's preparation, and activates
/// the suffix once an atlas slot becomes available.
///
/// Background workers load, validate, and mipmap requested source tiles. The
/// main thread installs completed tiles into fixed-stride vertex and maximum-
/// mipmap atlases between GPU command buffers, rebuilding the GPU tile-key
/// lookup table after changes. When the atlas is full, it evicts the least-
/// recently-used slot not referenced by the imminent frontier. The loop ends
/// when every azimuth column has intersected terrain, reached the range limit,
/// or left available terrain coverage.
void raytrace_tiled_heightmap(const RaytraceConfig &config) {
  validate_configuration(config);
  // Start a composite timer.
  Timer timer("Total elapsed");
  timer.start_wall("Initial setup");

  // Discover the finite source set once. The catalogue provides the observer
  // source at index zero and an immutable key-to-source lookup thereafter.
  const TerrainCatalogue catalogue = TerrainCatalogue::discover(
      config.tile_dir,
      config.observer,
      config.max_distance,
      config.max_tile_count
  );
  const TileGrid &grid = catalogue.grid();
  const TileKey origin_key = catalogue.origin().key;

  // The observer tile establishes common data dimensions and the projected
  // coordinate system required by every fixed-stride atlas slot.
  timer.start_work("Tile load");
  LoadedTile origin = LoadedTile::load_tif(catalogue.origin().path, true);
  timer.stop("Tile load");

  timer.start_work("Mipmap generation");
  origin.compute_mipmap();
  timer.stop("Mipmap generation");

  validate_terrain_tile_position(origin, origin_key, grid);

  const std::vector<TerrainSource> &paths = catalogue.sources();

  // Every rechunked tile must have the origin tile's dimensions, allowing
  // constant per-slot atlas strides. Derive the number of permitted slots
  // from the byte budget rather than hard-coding a tile count.
  const size_t mip_count = origin.mipmap.size();
  const size_t vertex_count = origin.vertices->size();
  const size_t tile_bytes =
      checked_byte_count(mip_count + vertex_count, sizeof(float), "terrain tile");
  const uint64_t slot_capacity = config.tile_cache_size_bytes / tile_bytes;
  if (slot_capacity == 0U) {
    throw std::runtime_error("Tile-cache byte budget cannot hold one terrain tile");
  }
  const uint32_t tile_count = static_cast<uint32_t>(paths.size());
  const uint64_t bounded_slot_count = std::min(slot_capacity, static_cast<uint64_t>(tile_count));
  if (bounded_slot_count > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
    throw std::overflow_error("Tile-cache slot count exceeds Metal uint range");
  }
  const uint32_t atlas_slot_count = static_cast<uint32_t>(bounded_slot_count);

  // A column has exactly one unresolved suffix globally: it either hits,
  // terminates at the range limit, or exits one tile and becomes one successor
  // segment. Therefore every active, deferred, and waiting frontier is bounded
  // by the azimuth count rather than by the number of terrain sources.
  const size_t frontier_capacity = config.num_azimuth;

  // Angular data and output images are shared by the complete GPU frontier,
  // rather than recreated for every individual terrain-tile dispatch.
  const std::vector<HorizontalDirection> directions = make_azimuth_directions(config);
  const std::vector<float> slopes = make_polar_slopes(config);
  const size_t ray_count = static_cast<size_t>(config.num_azimuth) * config.num_polar;
  const RaytraceParameters shared_parameters = make_raytrace_parameters(origin, config);

  @autoreleasepool {
    // GPU resources own the device, reusable command/pipeline state, static
    // angular inputs, output images, and double-buffered frontier storage.
    GpuRaytraceResources
        gpu(directions, slopes, ray_count, static_cast<uint32_t>(frontier_capacity));
    gpu.initialise_frontier(config.num_azimuth);

    // The cache shares the GPU resource owner's device, while the preparer
    // remains host-only and returns completed source tiles to that cache.
    ResidentTileCache cache(gpu.device(), paths, origin, origin_key, config, atlas_slot_count);
    AsyncTilePreparer preparer(
        paths,
        origin,
        grid,
        static_cast<uint32_t>(mip_count),
        static_cast<uint32_t>(vertex_count),
        atlas_slot_count,
        config.max_tile_preparation_workers,
        timer
    );
    const ResidentTileCacheBindings cache_bindings = cache.bindings();

    // The observer tile is resident, so start GPU tracing while background
    // workers continue to prepare later source tiles.
    timer.stop("Initial setup");

    // Command buffers are still synchronised one frontier iteration at a time
    // because the CPU needs the next append count to size the following grid.
    // Removing this wait is a later asynchronous-cache optimisation.
    gpu.start_capture_if_requested();
    uint32_t active_count = config.num_azimuth;

    // Host-frontier state owns deferred continuations, source lookup, and the
    // one-suffix-per-azimuth invariant used by the current column scheduler.
    HostFrontier frontier(config, catalogue, directions, shared_parameters);

    // Keep GPU lookup outcomes distinct from host load requests. A deferred
    // continuation can later terminate as open sky when no source covers it.
    uint64_t resident_successor_work = 0U;
    uint64_t deferred_successor_work = 0U;

    // Wall time includes the CPU's per-pass buffer setup, command encoding,
    // submission, and waits. The device timestamps recorded below separately
    // report the GPU's execution-only work within this same region.
    timer.start_wall("GPU raytrace");
    try {
      // Loading starts only after every permanent buffer and the command queue
      // exists. Any later failure is caught below, which stops and joins these
      // threads before unwinding their owning vector.
      preparer.start();
      frontier.prefetch_observer_neighbours(preparer);

      // Keep going until all rays have completed
      while (active_count != 0U) {
        // Account separately for CPU frontier bookkeeping before the Metal
        // command is created. This includes LRU use stamps and counter reset.
        timer.start_wall("Frontier bookkeeping");

        // Each azimuth can have only one unresolved tile segment. Validate
        // that the preceding emission/install pass preserved that contract
        // before this buffer is handed to Metal again.
#if defined(PANORAMA_DEBUG_VALIDATION)
        frontier.validate_frontier(gpu.active_frontier(), active_count, "active frontier");
#endif

        // This pass is about to read every referenced slot. Updating the LRU
        // stamp now makes recently traced terrain the last cache victim.
        (void)frontier.pin_frontier(gpu.active_frontier(), active_count, cache, true);

        // The GPU resource resets its append counters, encodes the ordered
        // trace/emit kernels, waits for their shared results, and returns the
        // host-visible successor counts.
        timer.stop("Frontier bookkeeping");
        const GpuFrontierPassResult pass = gpu.trace_frontier(
            cache_bindings,
            shared_parameters,
            static_cast<uint32_t>(mip_count),
            active_count,
            timer
        );
        timer.add_work("GPU raytrace", pass.device_milliseconds);

        // One active column emits at most one successor. Since the incoming
        // frontier was checked above, both GPU append buffers must fit the
        // same per-azimuth capacity. The kernels guard their writes; these
        // checks turn a counter overflow into a useful host-side failure.
        if (pass.next_count > frontier_capacity || pass.deferred_count > frontier_capacity) {
          throw std::runtime_error("GPU frontier exceeds the azimuth frontier capacity");
        }

        gpu.swap_frontiers();
#if defined(PANORAMA_DEBUG_VALIDATION)
        frontier.validate_frontier(gpu.active_frontier(), pass.next_count, "next frontier");
#endif
        resident_successor_work += static_cast<uint64_t>(pass.next_count);
        deferred_successor_work += static_cast<uint64_t>(pass.deferred_count);

        // Preserve every continuation that did not find a resident successor.
        // The host maps it to a source tile and retries it after installation.
        const std::span<const DeferredTileWork> deferred = gpu.deferred_work(pass.deferred_count);
        frontier.append_deferred(deferred);
#if defined(PANORAMA_DEBUG_VALIDATION)
        frontier.validate_deferred_work();
#endif

        timer.start_wall("Frontier bookkeeping");
        // The next pass may already reference some resident slots. Pin those
        // slots before installing prepared terrain, so LRU eviction cannot
        // overwrite a tile which the imminent GPU dispatch will read.
        const std::vector<uint8_t> pinned_slots =
            frontier.pin_frontier(gpu.active_frontier(), pass.next_count, cache, false);
        cache.install_prepared(preparer, pinned_slots, timer);

        // Append deferred continuations whose terrain became resident while
        // the preceding command buffer was executing.
        active_count =
            frontier.activate_resident(gpu.active_frontier(), pass.next_count, cache, preparer);
#if defined(PANORAMA_DEBUG_VALIDATION)
        frontier.validate_frontier(gpu.active_frontier(), active_count, "activated frontier");
#endif
        timer.stop("Frontier bookkeeping");

        // If no resident continuation is ready, wait for a requested tile to
        // complete preparation. There is no GPU work to submit in this case,
        // so an empty pin set makes every currently resident slot evictable.
        while (active_count == 0U && frontier.has_deferred_work()) {
          timer.start_wall("Tile availability wait");
          preparer.wait_for_prepared();
          timer.stop("Tile availability wait");

          const std::vector<uint8_t> no_pinned_slots(cache.slot_capacity(), 0U);
          cache.install_prepared(preparer, no_pinned_slots, timer);
          active_count =
              frontier.activate_resident(gpu.active_frontier(), active_count, cache, preparer);
#if defined(PANORAMA_DEBUG_VALIDATION)
          frontier.validate_frontier(gpu.active_frontier(), active_count, "activated frontier");
#endif
        }
      }
    } catch (...) {
      preparer.stop_and_join();
      gpu.stop_capture();
      throw;
    }
    preparer.stop_and_join();
    gpu.stop_capture();
    timer.stop("GPU raytrace");

    // The completed shared output buffers can now be handed directly to the
    // PNG writer. Its elapsed time is a named wall-clock region, separate
    // from terrain preparation and GPU device work.
    timer.start_wall("PNG generation");
    write_colormapped_png(
        "distances.png",
        view_float_buffer(gpu.distances(), ray_count, "distance output"),
        config.num_azimuth,
        config.num_polar,
        colormaps::viridis
    );
    write_colormapped_png(
        "elevations.png",
        view_float_buffer(gpu.elevations(), ray_count, "elevation output"),
        config.num_azimuth,
        config.num_polar,
        colormaps::viridis
    );
    timer.stop("PNG generation");

    // Report the finite source catalogue separately from the bounded resident
    // cache, so a memory-budget change is visible when rechunk levels vary.
    const TilePreparationStatistics preparation_statistics = preparer.statistics();
    const ResidentTileCacheStatistics cache_statistics = cache.statistics();
    std::printf(
        "Terrain sources: %u (resident slots %u / cache capacity %llu, preparation workers %u).\n",
        tile_count,
        cache_statistics.resident_tiles,
        static_cast<unsigned long long>(slot_capacity),
        preparation_statistics.worker_count
    );
    std::printf(
        "  GPU resident successors: %llu, deferred successors: %llu.\n",
        static_cast<unsigned long long>(resident_successor_work),
        static_cast<unsigned long long>(deferred_successor_work)
    );
    std::printf(
        "  Tile requests: %llu, load operations: %llu, unique loads: %llu, reloads: %llu.\n",
        static_cast<unsigned long long>(preparation_statistics.requests),
        static_cast<unsigned long long>(preparation_statistics.load_operations),
        static_cast<unsigned long long>(preparation_statistics.unique_loads),
        static_cast<unsigned long long>(preparation_statistics.reloads)
    );
    std::printf(
        "  Atlas installations: %llu, copied: %.3f GiB, evictions: %llu.\n",
        static_cast<unsigned long long>(cache_statistics.installations),
        static_cast<double>(cache_statistics.bytes_copied) / (1024.0 * 1024.0 * 1024.0),
        static_cast<unsigned long long>(cache_statistics.evictions)
    );
    timer.print();
  }
}

} // namespace panorama
